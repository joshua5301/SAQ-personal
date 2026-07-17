"""
TiltedSR: dense adversarial rounding perturbation anchored at stochastic
rounding (SR). Second arm of the rounding-robustness study; the sparse
worst-case arm is FlipSAM (utils/flipsam.py). Both arms share identical
rounding geometry via flipsam.layer_rounding_geometry.
"""

import torch

from models.LIQ_wn_qsam import QConv2d, QLinear

@torch.no_grad()
def layer_rounding_geometry(m, mask_grid_exact=True):
    """
    Returns (gain, delta_flip, valid, prior_logit) for one quantized layer,
    or None if the gradient is absent.

      gain        = g * delta_flip          (signed 1st-order flip gain)
      delta_flip  = +-step toward the non-nearest rounding candidate
                    (direction from rounding geometry, NOT gradient sign)
      valid       = flippable mask (grid boundary / grid-exact excluded)
      prior_logit = logit of the SR probability of the flip target
                    (the minor side, always <= 1/2)
    """
    g = m.x.grad
    if g is None:
        return None
    cache = getattr(m, "rounding_cache", None)
    if cache is None:
        raise RuntimeError(
            "[FlipSAM] rounding_cache missing. "
            "Did you patch quantize_weight() in LIQ_wn_qsam.py?"
        )
    r, nearest_is_floor, floor_lvl, n, step_out = cache

    dir_sign = torch.where(
        nearest_is_floor, torch.ones_like(r), -torch.ones_like(r)
    )
    delta_flip = dir_sign * step_out

    valid = ~(nearest_is_floor & (floor_lvl >= n))
    if mask_grid_exact:
        valid &= r > 0.0

    gain = g * delta_flip

    p_sr = torch.where(nearest_is_floor, r, 1.0 - r)
    p_sr = p_sr.clamp(1e-6, 0.5 - 1e-6)
    prior_logit = torch.logit(p_sr)

    return gain, delta_flip, valid, prior_logit


class TiltedSR:
    """
    Tilted stochastic rounding minimizer for QAT.

    Every valid weight flips independently with probability

        p_i = sigmoid( logit(p_SR_i) + beta * gain_hat_i ),

    where gain_hat_i = gain_i / mean_j |gain_j| (per-layer, over valid j)
    makes the tilt dimensionless, so beta is transferable across layers
    and training time.

    Single hyperparameter: beta.

    Anchors / limits:
        beta = 0     -> exact stochastic rounding; training is SR-QAT.
                        (This run doubles as the SR-QAT baseline.)
        beta -> +inf -> hard flip of all positive-gain weights
                        (dense adversarial; ~pos_gain_frac, NOT top-k).
        beta < 0     -> friendly-tilted SR (AdaRound-flavored inner min).

    Small-beta: F_beta ~ E_SR[L] + (beta/2) Var_SR[L] — beta penalizes the
    variance of the loss under rounding noise on top of unbiased SR.

    The flip fraction is emergent (not controlled): ~E[min(r,1-r)] ~ 25%
    at beta=0, moving toward pos_gain_frac as beta grows. This method
    lives in the dense-perturbation regime; the sparse regime (~1%) is
    FlipSAM's.
    """

    def __init__(self, optimizer, model, beta=1.0, mask_grid_exact=True):
        self.optimizer = optimizer
        self.model = model
        self.beta = beta
        self.mask_grid_exact = mask_grid_exact
        # per-layer diagnostics, refreshed every ascent step; log to wandb
        self.flip_stats = {}

    def _is_quantized(self, m):
        return isinstance(m, (QConv2d, QLinear)) and m.bits_weights != 32

    # ------------------------------------------------------------------ #
    # SAM interface (engine.py calls these)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def ascent_step(self):
        for name, m in self.model.named_modules():
            if not self._is_quantized(m):
                continue
            res = layer_rounding_geometry(m, self.mask_grid_exact)
            if res is None:
                continue
            gain, delta_flip, valid, prior_logit = res

            vg = gain[valid]
            if vg.numel() == 0:
                m.epsilon = torch.zeros_like(delta_flip)
                continue

            # dimensionless tilt
            scale = vg.abs().mean().clamp_min(1e-12)
            tilt = self.beta * (gain / scale)

            p = torch.sigmoid(prior_logit + tilt)
            p = p * valid.to(p.dtype)  # masked entries never flip

            flips = torch.bernoulli(p)
            m.epsilon = flips * delta_flip

            d = gain.numel()
            self.flip_stats[name] = {
                "flips": int(flips.sum().item()),
                "expected_flips": float(p.sum().item()),
                "flip_frac": float(flips.sum().item()) / max(d, 1),
                "sr_flip_frac": float(
                    torch.sigmoid(prior_logit)[valid].mean().item()
                ),
                "pos_gain_frac": (gain[valid] > 0).float().mean().item(),
            }
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        # No continuous perturbation to undo; clip/bias/BN are updated by
        # the optimizer using the gradient from the tilted (second) pass.
        self.optimizer.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def restore_step(self):
        self.optimizer.zero_grad()