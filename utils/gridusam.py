"""
GridUSAM: penalized (unnormalized) SAM with grid-realizable weight
perturbations, ported onto the SAQ-style ascent/descent stack.

Inner problem (penalized USAM form, coordinate-separable):

    max_eps  g^T eps - ||eps||^2 / (2 rho)

For continuous parameters (clip / bias / BN) the solution is the standard
USAM step  e_p = rho * grad_p.

For quantized weights the only realizable moves are rounding flips, so each
coordinate is flip-or-nothing with the closed-form threshold

    flip_i  <=>  gain_i > cost_i^2 / (2 rho),     gain_i = g_i * delta_i,

where the cost depends on the threat model (`space`):

    space="output":  cost_i = |delta_i| = step_i
        (perturbation length of the deployed quantized weight; the
         Euclidean grid move itself)

    space="latent":  cost_i = dist_i = (1/2 - min(r_i, 1-r_i)) * step_i
        (latent movement needed to cross the rounding boundary;
         boundary-adjacent weights flip for free)

No dense continuous perturbation is applied to quantized weights: moves
that are not grid-realizable are excluded by construction (this is the
flip-or-nothing correction of the reference design).

Single knob: rho, shared verbatim with the USAM step on continuous
parameters. rho -> 0 recovers the nearest-rounding baseline with no
perturbation anywhere. The flip count is emergent (threshold), not
budgeted: penalty form trades explicit budget control (KLTilt's tau) for
a closed-form O(n) rule with no bisection.
"""

import torch
from models.LIQ_wn_qsam import QConv2d, QLinear
from utils.cont_perturb import CONT_MODES, gather_cont_params


class GridUSAM:

    _SPACES = ("output", "latent")

    def __init__(self, optimizer, model, rho=0.05, space="output",
                 perturb_continuous="none", mask_grid_exact=True):
        assert space in self._SPACES, space
        assert perturb_continuous in CONT_MODES, perturb_continuous
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.space = space
        self.perturb_continuous = perturb_continuous
        self.mask_grid_exact = mask_grid_exact
        self._backups = {}   # {param: saved data}; restore iterates this
        self._stats = {}     # tensors only; materialized via stats()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _is_quantized(self, m):
        return isinstance(m, (QConv2d, QLinear)) and m.bits_weights != 32

    @torch.no_grad()
    def _layer_flip(self, m):
        """
        Threshold flip decision for one quantized layer.
        Returns (eps, n_flip, n_valid) or None if grad/cache missing.
        """
        g = m.x.grad
        if g is None:
            return None
        cache = getattr(m, "rounding_cache", None)
        if cache is None:
            raise RuntimeError(
                "[GridUSAM] rounding_cache missing. "
                "Did you patch quantize_weight() in LIQ_wn_qsam.py?"
            )
        r, nearest_is_floor, floor_lvl, n, step_out = cache

        # flip direction: toward the non-nearest rounding candidate
        dir_sign = torch.where(
            nearest_is_floor, torch.ones_like(r), -torch.ones_like(r)
        )
        delta_flip = dir_sign * step_out

        valid = ~(nearest_is_floor & (floor_lvl >= n))
        if self.mask_grid_exact:
            valid &= r > 0.0

        gain = g * delta_flip                      # 1st-order flip gain

        if self.space == "output":
            cost = step_out.expand_as(gain) if step_out.dim() == 0 \
                else step_out
        else:  # latent: distance to the rounding boundary, output units
            cost = (0.5 - torch.minimum(r, 1.0 - r)) * step_out

        # penalized-USAM threshold: flip iff gain - cost^2/(2 rho) > 0.
        # gain > 0 is implied for space="output" (cost > 0), and made
        # explicit so that latent free flips (cost ~ 0) still require an
        # adversarial direction.
        flip = valid & (gain > 0) & (gain > cost * cost / (2.0 * self.rho))

        eps = flip.to(delta_flip.dtype) * delta_flip
        return eps, flip.sum(), valid.sum()

    # ------------------------------------------------------------------ #
    # SAM interface (engine.py calls these)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def ascent_step(self):
        self._backups.clear()

        n_flip = None
        n_valid = None
        for name, m in self.model.named_modules():
            if not self._is_quantized(m):
                continue
            res = self._layer_flip(m)
            if res is None:
                continue
            eps, layer_flip, layer_valid = res
            m.epsilon = eps
            n_flip = layer_flip if n_flip is None else n_flip + layer_flip
            n_valid = layer_valid if n_valid is None else n_valid + layer_valid

        # ---- continuous params: plain USAM step with the SAME rho ----
        # penalized-form solution: e_p = rho * grad_p  (no normalization)
        cont_params = gather_cont_params(self.model, self.perturb_continuous)
        cont_norm_sq = None
        if cont_params and self.rho > 0:
            for p in cont_params:
                if p.grad is None:
                    continue
                self._backups[p] = p.data.clone()
                p.add_(p.grad, alpha=self.rho)
                sq = p.grad.pow(2).sum()
                cont_norm_sq = sq if cont_norm_sq is None else cont_norm_sq + sq

        if n_flip is not None:
            self._stats = {
                "flip_frac": n_flip.float() / n_valid.clamp_min(1).float(),
                "n_flip": n_flip,
                "n_valid": n_valid,
            }
            if cont_norm_sq is not None:
                # effective continuous radius ||e|| = rho * ||grad_cont||
                self._stats["rho_eff_cont"] = self.rho * cont_norm_sq.sqrt()

        self.optimizer.zero_grad()

    @torch.no_grad()
    def stats(self):
        """Materialize diagnostics (one sync). Call every N steps."""
        return {k: (float(v) if torch.is_tensor(v) else v)
                for k, v in self._stats.items()}

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