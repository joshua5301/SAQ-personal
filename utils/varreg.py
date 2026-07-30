"""
VarReg: 1-pass rounding-variance regularizer.

Adds lam(epoch) * sqrt(R) to the training loss, where

    R = sum_i V_i * gain_i^2                     (one global scalar)
    gain_i = g_i * step_l,  g_i = dL/d(quantized weight_i) = module.x.grad

Two measure arms select V:
    sr    : V = r(1-r)                    (SR posterior variance)
    gauss : two-boundary truncation of an isotropic Gaussian pushforward at
            sigma = alpha * step, alpha = 0.12 (fixed constant, see spec).

Three lambda schedules end at the same lambda at the final epoch:
    constant / linear / cosine.

Coupling:
    coupled   (SGD default) : force added to param.grad before optimizer.step().
    decoupled (Adam default): p.data -= lr * force AFTER optimizer.step().

Single-pass: no adversarial forward, no second backward, no weight perturbation
during forward. lambda_final = 0 makes the regularizer completely inert and the
baseline reproduces exactly.

Discovery answers for this repo (see spec section 3):
  * g_i = module.x.grad, module.x = quantized OUTPUT tensor of quantize_weight.
  * f(w) = ((w - w.detach().mean()) / w.detach().std() + 1)/2 * n. mean/std are
    detached, so f is affine with diag(n/(2*std)) Jacobian per layer. We still
    route through autograd (spec 4b) for safety against future changes to f.
  * valid = ~(nearest_is_floor & (floor_lvl >= n)) & (r > 0), matching KLTilt
    / GridSAM.
"""

import math

import torch

from models.LIQ_wn_qsam import QConv2d, QLinear


_APPLY_MODES = ("coupled", "decoupled", "auto")
_MEASURES = ("sr", "gauss")
_SCHEDULES = ("constant", "linear", "cosine")


class VarReg:

    def __init__(self, model, optimizer, lambda_final=0.0,
                 measure="sr", schedule="cosine", apply_mode="auto",
                 total_epochs=1, alpha=0.12, mask_grid_exact=True):
        assert measure in _MEASURES, measure
        assert schedule in _SCHEDULES, schedule
        assert apply_mode in _APPLY_MODES, apply_mode
        assert 0.0 < alpha < 0.5, alpha
        self.model = model
        self.optimizer = optimizer
        self.lambda_final = float(lambda_final)
        self.measure = measure
        self.schedule = schedule
        self.total_epochs = max(1, int(total_epochs))
        self.alpha = float(alpha)
        self.mask_grid_exact = bool(mask_grid_exact)
        if apply_mode == "auto":
            apply_mode = ("decoupled"
                          if "Adam" in type(optimizer).__name__
                          else "coupled")
        self.apply_mode = apply_mode
        self.epoch = 0
        self._forces = []
        self._stats = {}

    # ------------------------------------------------------------------ #
    # schedule
    # ------------------------------------------------------------------ #

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def lam(self):
        if self.lambda_final == 0.0:
            return 0.0
        t = self.epoch / max(1, self.total_epochs - 1)
        if self.schedule == "constant":
            return self.lambda_final
        if self.schedule == "linear":
            return self.lambda_final * t
        return self.lambda_final * 0.5 * (1.0 - math.cos(math.pi * t))

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #

    def _is_quantized(self, m):
        return isinstance(m, (QConv2d, QLinear)) and m.bits_weights != 32

    def _entries(self):
        for _, m in self.model.named_modules():
            if not self._is_quantized(m):
                continue
            cache = getattr(m, "rounding_cache", None)
            if cache is None:
                continue
            x = getattr(m, "x", None)
            if x is None or x.grad is None:
                continue
            yield m, cache, x.grad

    def _V_and_dVdu(self, r, nearest_is_floor):
        """(V, dV/du), both same shape as r, computed in fp32."""
        r = r.float()
        if self.measure == "sr":
            V = r * (1.0 - r)
            dVdu = 1.0 - 2.0 * r
            return V, dVdu
        # gauss: two-boundary truncation, sigma = alpha (in bin units)
        u = torch.where(nearest_is_floor, r, r - 1.0)
        a = self.alpha
        z_p = (0.5 - u) / a
        z_m = (0.5 + u) / a
        Phi = torch.special.ndtr
        p_plus = Phi(-z_p)
        p_minus = Phi(-z_m)
        s = p_plus - p_minus
        V = p_plus + p_minus - s * s
        inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
        phi_p = inv_sqrt_2pi * torch.exp(-0.5 * z_p * z_p)
        phi_m = inv_sqrt_2pi * torch.exp(-0.5 * z_m * z_m)
        dVdu = (phi_p * (1.0 - 2.0 * s) - phi_m * (1.0 + 2.0 * s)) / a
        return V, dVdu

    @staticmethod
    def _jacobian_push(module, force_scaled, n):
        """
        force_scaled lives in the space of the 'scaled' tensor whose
        fractional part is r (= x_in [0,1] times n). Reconstruct the EXACT
        forward map w -> scaled per module type and push force through it
        via one autograd pass.

        QConv2d forward does weight-norm (self.weight - mean)/std BEFORE
        quantize_weight; QLinear passes raw self.weight. Both then go
        through normalization_on_weights = x/clip -> clamp[-1,1], then
        (x+1)/2 * n. Clamp saturation zeros the force at clipped weights
        (correct: they cannot move onto the grid without unclipping first).
        """
        with torch.enable_grad():
            w_ = module.weight.detach().requires_grad_(True)
            clip = module.weight_clip_value.detach().abs()
            if isinstance(module, QConv2d):
                mean = w_.detach().mean()
                std = w_.detach().std()
                x = (w_ - mean) / std
            else:                           # QLinear: no weight-norm
                x = w_
            x = x / clip
            x = torch.clamp(x, -1.0, 1.0)
            scaled = (x + 1.0) / 2.0 * n
        (force_w,) = torch.autograd.grad(scaled, w_,
                                         grad_outputs=force_scaled)
        return force_w

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def arm(self):
        self._forces = []
        self._stats = {}

    @torch.no_grad()
    def stage(self):
        """Phase A: global R. Phase B: per-layer force in scaled-space,
        pushed to raw-weight space via the normalization Jacobian.
        Skips silently on non-finite grads (AMP)."""
        lam_val = self.lam()
        if lam_val == 0.0:
            return

        cached = []
        R = None
        d_valid_total = 0
        for m, cache, grad in self._entries():
            r, nearest_is_floor, floor_lvl, n, step_out = cache
            if not torch.isfinite(grad).all():
                return                     # AMP: skip patch this step
            valid = ~(nearest_is_floor & (floor_lvl >= n))
            if self.mask_grid_exact:
                valid = valid & (r > 0.0)
            V, dVdu = self._V_and_dVdu(r, nearest_is_floor)
            gain = grad.float() * step_out
            gain_sq = gain * gain
            V_masked = torch.where(valid, V, torch.zeros_like(V))
            contrib = (V_masked * gain_sq).sum()
            R = contrib if R is None else R + contrib
            d_valid_total += int(valid.sum().item())
            cached.append((m, r, nearest_is_floor, step_out, valid,
                           gain_sq, dVdu, n))

        if R is None:
            return
        sqrtR = torch.sqrt(R.clamp_min(1e-24))
        scale = lam_val / (2.0 * sqrtR.clamp_min(1e-12))

        forces = []
        force_reg_norm_sq = torch.zeros((), device=sqrtR.device,
                                        dtype=torch.float32)
        force_task_norm_sq = torch.zeros((), device=sqrtR.device,
                                         dtype=torch.float32)
        for (m, r, nearest_is_floor, step_out, valid,
             gain_sq, dVdu, n) in cached:
            dVdu_masked = torch.where(valid, dVdu, torch.zeros_like(dVdu))
            force_scaled = (gain_sq * dVdu_masked * scale).to(m.weight.dtype)
            force_w = self._jacobian_push(m, force_scaled, n)
            forces.append((m.weight, force_w))
            force_reg_norm_sq = (force_reg_norm_sq
                                 + (force_w.float() ** 2).sum())
            if m.weight.grad is not None:
                force_task_norm_sq = (force_task_norm_sq
                                      + (m.weight.grad.float() ** 2).sum())

        self._forces = forces
        self._stats = dict(
            lam=torch.as_tensor(lam_val),
            std=sqrtR,
            force_task_norm=force_task_norm_sq.sqrt(),
            force_reg_norm=force_reg_norm_sq.sqrt(),
            d_valid=torch.as_tensor(float(d_valid_total)),
            lambda_final=torch.as_tensor(self.lambda_final),
        )

    @torch.no_grad()
    def commit(self):
        if self.apply_mode != "coupled":
            return
        for p, f in self._forces:
            if p.grad is None:
                p.grad = f.clone()
            else:
                p.grad.add_(f)

    @torch.no_grad()
    def apply(self, lr):
        if self.apply_mode != "decoupled":
            return
        for p, f in self._forces:
            p.data.add_(f, alpha=-float(lr))

    def clear(self):
        self._forces = []

    # ------------------------------------------------------------------ #

    def summary(self):
        s = {k: (float(v) if torch.is_tensor(v) else v)
             for k, v in self._stats.items()}
        d = max(s.get("d_valid", 1.0), 1.0)
        lam_final = s.get("lambda_final", 0.0)
        s["tau"] = (lam_final * lam_final) / (2.0 * d)
        t = s.get("force_task_norm", 0.0)
        s["force_ratio"] = (s.get("force_reg_norm", 0.0) / t) if t > 0 else 0.0
        s["measure"] = self.measure
        s["schedule"] = self.schedule
        s["apply_mode"] = self.apply_mode
        return s
