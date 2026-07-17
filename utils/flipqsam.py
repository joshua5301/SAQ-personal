"""
FlipQSAM: weighted-L0 adversarial rounding flips for quantized weights
+ QSAM-identical continuous-parameter SAM.

Norm-ball view (theory section):
    SAM      = L2 ball          (dense, gradient-proportional allocation)
    top-k    = L0 ball          (uniform sparse selection)
    FlipQSAM = weighted-L0 ball (cost = distance-to-boundary) for weights,
               plus the standard L2 SAM ball for continuous params with
               the SAME global normalization as QSAM (weight grads included
               in the denominator), so the continuous treatment is
               bit-identical to the QSAM baseline.

Knobs:
    rho_flip : latent-movement budget per weight (bin units) for flips.
               rho_flip = 0 -> nearest-rounding baseline (no flips).
    rho      : continuous SAM radius, inherited from / comparable to QSAM.
               rho = 0 -> no continuous perturbation.
"""

import torch
import torch.nn as nn
from models.LIQ_wn_qsam import QConv2d, QLinear


class FlipQSAM:

    _CONT_MODES = ("none", "clip", "clip_bias", "all", "qsam_default")

    def __init__(self, optimizer, model, rho_flip=0.0025, rho=0.05,
                 perturb_continuous="clip", mask_grid_exact=True):
        assert perturb_continuous in self._CONT_MODES, perturb_continuous
        self.optimizer = optimizer
        self.model = model
        self.rho_flip = rho_flip
        self.rho = rho
        self.perturb_continuous = perturb_continuous
        self.mask_grid_exact = mask_grid_exact
        self.flip_stats = {}
        self._backups = {}   # {param: saved data}; restore iterates only this

    def _is_quantized(self, m):
        return isinstance(m, (QConv2d, QLinear)) and m.bits_weights != 32

    # ------------------------------------------------------------------ #
    # rounding geometry
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _layer_geometry(self, m):
        """Returns (gain, delta_flip, valid, r) or None."""
        g = m.x.grad
        if g is None:
            return None
        cache = getattr(m, "rounding_cache", None)
        if cache is None:
            raise RuntimeError(
                "[FlipQSAM] rounding_cache missing. "
                "Did you patch quantize_weight() in LIQ_wn_qsam.py?"
            )
        r, nearest_is_floor, floor_lvl, n, step_out = cache

        dir_sign = torch.where(
            nearest_is_floor, torch.ones_like(r), -torch.ones_like(r)
        )
        delta_flip = dir_sign * step_out

        valid = ~(nearest_is_floor & (floor_lvl >= n))
        if self.mask_grid_exact:
            valid &= r > 0.0

        gain = g * delta_flip
        # distance from latent position to the rounding boundary (bin units)
        dist_to_boundary = 0.5 - torch.minimum(r, 1.0 - r)
        return gain, delta_flip, valid, dist_to_boundary

    # ------------------------------------------------------------------ #
    # weights: weighted-L0 knapsack flips (Option B, per-layer budget)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _knapsack_eps(self, gain, delta_flip, valid, cost, budget):
        eps = torch.zeros_like(delta_flip)
        cand = valid & (gain > 0)
        if budget <= 0 or not cand.any():
            return eps, 0, 0.0
        g = gain[cand].flatten()
        c = cost[cand].flatten()
        ratio = g / c.clamp_min(1e-8)          # cost->0: flips for free first
        order = torch.argsort(ratio, descending=True)
        csum = torch.cumsum(c[order], dim=0)
        take = csum <= budget                   # LP-relaxation greedy
        idx = cand.flatten().nonzero(as_tuple=True)[0][order[take]]
        n_flipped = int(idx.numel())
        spent = float(csum[take][-1].item()) if n_flipped > 0 else 0.0
        if n_flipped > 0:
            mask = torch.zeros(gain.numel(), dtype=torch.bool,
                               device=gain.device)
            mask[idx] = True
            eps = mask.view_as(delta_flip).to(delta_flip.dtype) * delta_flip
        return eps, n_flipped, spent

    # ------------------------------------------------------------------ #
    # SAM interface
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def ascent_step(self):
        self._backups.clear()

        # ---- pass 1: global grad norm, QSAM-identical denominator ----
        # (quantized weight grads INCLUDED, exactly as in QSAM)
        cont_params = []
        grad_norms = []
        for _, m in self.model.named_modules():
            if isinstance(m, (QConv2d, QLinear)):
                if self._is_quantized(m) and m.x.grad is not None:
                    grad_norms.append(m.x.grad.norm(p=2))
                mode = self.perturb_continuous
                if mode != "none":
                    cands = []
                    if mode in ("clip", "clip_bias", "all"):
                        cands += [m.weight_clip_value,
                                  getattr(m, "activation_clip_value", None)]
                    if mode in ("clip_bias", "all", "qsam_default"):
                        cands.append(getattr(m, "bias", None))
                    for p in cands:
                        if p is not None and p.grad is not None:
                            cont_params.append(p)
                            grad_norms.append(p.grad.norm(p=2))
            if self.perturb_continuous in ("all", "qsam_default") and \
                    isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                for p in (m.weight, m.bias):
                    if p.grad is not None:
                        cont_params.append(p)
                        grad_norms.append(p.grad.norm(p=2))

        # ---- weights: knapsack flips, per-layer budget rho_flip * d ----
        for name, m in self.model.named_modules():
            if not self._is_quantized(m):
                continue
            res = self._layer_geometry(m)
            if res is None:
                continue
            gain, delta_flip, valid, cost = res
            d = gain.numel()
            budget = self.rho_flip * d
            eps, n_flipped, spent = self._knapsack_eps(
                gain, delta_flip, valid, cost, budget
            )
            m.epsilon = eps
            self.flip_stats[name] = {
                "flips": n_flipped,
                "flip_frac": n_flipped / max(d, 1),
                "budget": budget,
                "spent": spent,
                "pos_gain_frac": (gain[valid] > 0).float().mean().item()
                if valid.any() else 0.0,
            }

        # ---- continuous params: standard SAM step, QSAM scale ----
        if cont_params and grad_norms and self.rho > 0:
            grad_norm = torch.norm(torch.stack(grad_norms), p=2)
            scale = self.rho / (grad_norm + 1e-12)
            for p in cont_params:
                self._backups[p] = p.data.clone()
                p.add_(p.grad * scale.to(p))

        self.optimizer.zero_grad()

    @torch.no_grad()
    def _restore(self):
        for p, data in self._backups.items():
            p.data = data
        self._backups.clear()

    @torch.no_grad()
    def descent_step(self):
        self._restore()
        self.optimizer.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def restore_step(self):
        self._restore()
        self.optimizer.zero_grad()
