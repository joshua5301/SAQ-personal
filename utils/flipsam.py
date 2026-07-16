from collections import defaultdict

import torch
from models.LIQ_wn_qsam import QConv2d, QLinear


class FlipSAM:
    """
    FlipSAM: per-layer top-k% adversarial rounding-flip minimizer for QAT.

    The only perturbation FlipSAM applies is a discrete flip of the rounding
    decision of quantized weights. Continuous parameters (clip values, bias,
    BN) are NOT adversarially perturbed; they are trained normally by the
    optimizer using the gradient measured at the flipped point (the second,
    "descent" pass). The single hyperparameter is kappa.

    Two-pass step (driven by engine.py):
        1. clean forward/backward  -> g_i = dL/dQ(w)_i, rounding_cache written
        2. ascent_step()           -> choose top-k% adversarial flips per layer,
                                       store as m.epsilon
        3. flipped forward/backward -> gradient at the adversarially-rounded point
        4. descent_step()          -> optimizer.step()

    Per weight i:
        gain_i        = g_i * delta_flip_i                 (signed 1st-order gain)
        delta_flip_i  = (non-nearest candidate) - (nearest candidate)
                        = +step if nearest rounded down, -step if rounded up
                        (direction from rounding geometry, NOT gradient sign)
    Only positive-gain flips among the per-layer top-k are applied, so the
    number of flips is min(k, #{gain > 0}).

    Limits:
        kappa = 0  -> exact nearest-rounding QAT baseline (eps == 0).
        kappa = 1  -> hard adversarial flip (beta -> +inf of the Gibbs form).

    Rounding geometry is read from m.rounding_cache, written by
    quantize_weight() during the first forward pass (see LIQ_wn_qsam patch).
    """

    def __init__(self, optimizer, model, kappa=0.01, mask_grid_exact=True):
        self.optimizer = optimizer
        self.model = model
        self.kappa = kappa
        self.mask_grid_exact = mask_grid_exact
        # per-layer diagnostics, refreshed every ascent step; log to wandb
        self.flip_stats = {}

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _is_quantized(self, m):
        return isinstance(m, (QConv2d, QLinear)) and m.bits_weights != 32

    @torch.no_grad()
    def _flip_epsilon(self, m, name):
        """
        Per-layer top-k% adversarial rounding flip for one quantized layer.
        Returns epsilon (same shape as m.x), or None if gradient/cache absent.
        """
        g = m.x.grad
        if g is None:
            return None
        cache = getattr(m, "rounding_cache", None)
        if cache is None:
            raise RuntimeError(
                f"[FlipSAM] rounding_cache missing on layer '{name}'. "
                "Did you patch quantize_weight() in LIQ_wn_qsam.py?"
            )
        r, nearest_is_floor, floor_lvl, n, step_out = cache

        # flip direction: toward the non-nearest rounding candidate
        dir_sign = torch.where(
            nearest_is_floor, torch.ones_like(r), -torch.ones_like(r)
        )
        delta_flip = dir_sign * step_out  # +-step_out, geometry-determined

        # validity: top-level weights cannot flip up beyond the grid
        valid = ~(nearest_is_floor & (floor_lvl >= n))
        if self.mask_grid_exact:
            # exclude weights sitting exactly on a grid point (degenerate
            # rounding decision; mostly clip-saturated weights)
            valid &= r > 0.0

        # first-order flip gain (signed): > 0 means the flip raises the loss
        gain = g * delta_flip
        neg_inf = torch.finfo(gain.dtype).min
        gain_ranked = torch.where(valid, gain, torch.full_like(gain, neg_inf))

        d = gain.numel()
        k = int(self.kappa * d)

        eps = torch.zeros_like(g)
        n_flipped = 0
        if k > 0:
            flat = gain_ranked.flatten()
            topk_vals, topk_idx = torch.topk(flat, min(k, d), sorted=False)
            keep = topk_vals > 0  # only genuinely adversarial flips
            idx = topk_idx[keep]
            n_flipped = int(idx.numel())
            if n_flipped > 0:
                flip_mask = torch.zeros(d, dtype=torch.bool, device=g.device)
                flip_mask[idx] = True
                eps = flip_mask.view_as(g).to(g.dtype) * delta_flip

        self.flip_stats[name] = {
            "flips": n_flipped,
            "budget": k,
            "flip_frac": n_flipped / max(d, 1),
            "pos_gain_frac": (gain[valid] > 0).float().mean().item()
            if valid.any() else 0.0,
        }
        return eps

    # ------------------------------------------------------------------ #
    # SAM interface (engine.py calls these)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def ascent_step(self):
        for name, m in self.model.named_modules():
            if self._is_quantized(m):
                eps = self._flip_epsilon(m, name)
                if eps is not None:
                    m.epsilon = eps
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        # clip / bias / BN are updated here by the optimizer using the
        # gradient from the flipped (second) pass. No restoration needed,
        # since FlipSAM never perturbs them.
        self.optimizer.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def restore_step(self):
        # No continuous perturbation to undo; just clear grads.
        self.optimizer.zero_grad()