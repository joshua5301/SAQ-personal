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
from models.LIQ_wn_qsam import QConv2d, QLinear
from utils.cont_perturb import CONT_MODES, gather_cont_params, apply_qsam_radius


class FlipQSAM:

    _CONT_MODES = CONT_MODES

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

        # ---- continuous params: QSAM-identical scope + scale ----
        # (shared helper; quantized weight grads included in denominator)
        cont_params = gather_cont_params(self.model, self.perturb_continuous)
        apply_qsam_radius(self.model, cont_params, self.rho, self._backups)

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
