"""
TiltedSR: dense adversarial rounding perturbation anchored at stochastic
rounding (SR). Second arm of the rounding-robustness study; the sparse
worst-case arm is FlipSAM (utils/flipsam.py). Both arms share identical
rounding geometry via flipsam.layer_rounding_geometry.
"""

import torch

from models.LIQ_wn_qsam import QConv2d, QLinear
from utils.cont_perturb import CONT_MODES, gather_cont_params, apply_qsam_radius

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
    def __init__(self, optimizer, model, beta=1.0,
                 scale_mode="local", perturb_continuous="none",
                 rho=0.05, mask_grid_exact=True):
        assert scale_mode in ("local", "global"), scale_mode
        assert perturb_continuous in CONT_MODES, perturb_continuous
        self.optimizer = optimizer
        self.model = model
        self.beta = beta
        self.perturb_continuous = perturb_continuous
        self.rho = rho          # continuous SAM radius (QSAM-identical scale)
        self.mask_grid_exact = mask_grid_exact
        # per-layer diagnostics, refreshed every ascent step; log to wandb
        self.flip_stats = {}
        # {param_tensor: backup_data}; single source of truth for restore
        self._backups = {}

        self.scale_mode = scale_mode

    def _is_quantized(self, m):
        return isinstance(m, (QConv2d, QLinear)) and m.bits_weights != 32
    
    @torch.no_grad()
    def ascent_step(self):
        self._backups.clear()
        # pass 0 (global only): streaming global scale
        global_scale = None
        if self.scale_mode == "global":
            tot_abs, tot_cnt = 0.0, 0
            for _, m in self.model.named_modules():
                if not self._is_quantized(m):
                    continue
                res = layer_rounding_geometry(m, self.mask_grid_exact)
                if res is None:
                    continue
                gain, _, valid, _ = res
                vg = gain[valid]
                tot_abs += float(vg.abs().sum().item())
                tot_cnt += int(vg.numel())
                m._geom_cache = res          # 재계산 방지용 1스텝 캐시
            global_scale = max(tot_abs / max(tot_cnt, 1), 1e-12)

        for name, m in self.model.named_modules():
            if not self._is_quantized(m):
                continue
            res = getattr(m, "_geom_cache", None) or \
                  layer_rounding_geometry(m, self.mask_grid_exact)
            m._geom_cache = None
            if res is None:
                continue
            gain, delta_flip, valid, prior_logit = res
            vg = gain[valid]
            if vg.numel() == 0:
                m.epsilon = torch.zeros_like(delta_flip)
                continue

            if self.scale_mode == "global":
                scale = global_scale
            else:
                scale = vg.abs().mean().clamp_min(1e-12)

            p = torch.sigmoid(prior_logit + self.beta * (gain / scale))
            p = p * valid.to(p.dtype)
            flips = torch.bernoulli(p)
            m.epsilon = flips * delta_flip
            # flip_stats 동일 (+ "scale": float(scale) 로깅 권장)

        # continuous params: same scope + QSAM-identical scale as the other
        # minimizers, so comparison runs differ only in the weight perturbation
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