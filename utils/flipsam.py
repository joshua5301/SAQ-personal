from collections import defaultdict

import torch
from models.LIQ_wn_qsam import QConv2d, QLinear
from utils.cont_perturb import CONT_MODES, gather_cont_params, apply_qsam_radius


class FlipSAM:
    """
    FlipSAM: top-k% adversarial rounding-flip minimizer for QAT, with
    kappa-induced continuous perturbation on non-quantized parameters.

    Quantized weights are perturbed by discrete rounding flips (top-k% by
    first-order flip gain). Continuous parameters (clip values, bias, BN)
    are perturbed in the gradient direction with a radius controlled by
    cont_radius:

        cont_radius == "induced" (default) -> radius INHERITED from the
        flip perturbation, no separate rho:
            r_l      = ||eps_flip^(l)||_2 / ||Q(w^(l))||_2      (per layer)
            r_global = sqrt(sum_l ||eps_flip^(l)||^2) / sqrt(sum_l ||Q(w^(l))||^2)
            e_p = r * ||p|| * grad_p / ||grad_p||

        cont_radius == "qsam" -> QSAM-identical global norm (quantized
        weight grads INCLUDED in the denominator) with an independent rho:
            e_p = rho * grad_p / ||[grad_w; grad_cont]||

    Single hyperparameter for the flip side: kappa. kappa = 0 recovers the
    exact nearest-rounding QAT baseline (both perturbations vanish).

    perturb_continuous:
        "none"         -> flip-only (previous behavior)
        "clip"         -> + weight/activation clip values (grid robustness)
        "clip_bias"    -> + bias
        "all"          -> + BN affine params
        "qsam_default" -> activation clip + bias + BN affine — the QSAM
                           baseline's default scope (include_aclip=True,
                           include_bn=True, include_wclip=False)
    """

    _CONT_MODES = CONT_MODES
    _RADIUS_MODES = ("induced", "qsam")

    def __init__(self, optimizer, model, kappa=0.01,
                 kappa_mode="local", perturb_continuous="none",
                 cont_radius="induced", rho=0.05,
                 mask_grid_exact=True):
        assert kappa_mode in ("local", "global"), kappa_mode
        assert perturb_continuous in self._CONT_MODES, perturb_continuous
        assert cont_radius in self._RADIUS_MODES, cont_radius
        self.optimizer = optimizer
        self.model = model
        self.kappa = kappa
        self.kappa_mode = kappa_mode
        self.perturb_continuous = perturb_continuous
        self.cont_radius = cont_radius
        self.rho = rho          # used only when cont_radius == "qsam"
        self.mask_grid_exact = mask_grid_exact
        self.flip_stats = {}
        # {param_tensor: backup_data}; single source of truth for restore
        self._backups = {}

    def _is_quantized(self, m):
        return isinstance(m, (QConv2d, QLinear)) and m.bits_weights != 32

    # ------------------------------------------------------------------ #
    # geometry + topk (unchanged)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _layer_gain(self, m):
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
        if self.mask_grid_exact:
            valid &= r > 0.0

        gain = g * delta_flip
        return gain, delta_flip, valid

    @torch.no_grad()
    def _apply_topk(self, gain, delta_flip, valid, k):
        eps = torch.zeros_like(delta_flip)
        if k <= 0:
            return eps, 0
        neg_inf = torch.finfo(gain.dtype).min
        gain_ranked = torch.where(valid, gain, torch.full_like(gain, neg_inf))
        flat = gain_ranked.flatten()
        d = flat.numel()
        topk_vals, topk_idx = torch.topk(flat, min(k, d), sorted=False)
        keep = topk_vals > 0
        idx = topk_idx[keep]
        n_flipped = int(idx.numel())
        if n_flipped > 0:
            mask = torch.zeros(d, dtype=torch.bool, device=gain.device)
            mask[idx] = True
            eps = mask.view_as(delta_flip).to(delta_flip.dtype) * delta_flip
        return eps, n_flipped

    # ------------------------------------------------------------------ #
    # continuous perturbation with kappa-induced radius
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _apply_continuous(self, layer_r):
        """
        SAM ascent on continuous params. Two radius modes:

            cont_radius == "induced" (default): radius inherited from the
            flip perturbation, no separate rho:
                r_global = sqrt(sum_l ||eps_flip||^2) / sqrt(sum_l ||Q(w)||^2)
                rho      = r_global * ||p_cont||
                e_p      = rho * grad_p / ||grad_cont||

            cont_radius == "qsam": QSAM-identical global norm (quantized
            weight grads INCLUDED in the denominator) with an independent
            rho, so the continuous treatment is bit-identical to QSAM:
                e_p = rho * grad_p / ||[grad_w; grad_cont]||
        """
        mode = self.perturb_continuous
        if mode == "none":
            return

        # shared gather so every minimizer perturbs the identical param set
        params = gather_cont_params(self.model, mode)
        if not params:
            return

        if self.cont_radius == "qsam":
            # QSAM-identical: global norm INCLUDES quantized weight grads,
            # independent rho (inherit SAQ's tuned value)
            apply_qsam_radius(self.model, params, self.rho, self._backups)
            return

        # induced: radius inherited from flip perturbation (no rho knob)
        tot_eps_sq = sum(v["eps_sq"] for v in layer_r.values())
        tot_w_sq = sum(v["w_sq"] for v in layer_r.values())
        r_global = (tot_eps_sq ** 0.5) / max(tot_w_sq ** 0.5, 1e-12)
        grad_norm = torch.norm(
            torch.stack([p.grad.norm(p=2) for p in params]), p=2
        )
        if grad_norm <= 1e-12:
            return
        p_cont_norm = torch.norm(
            torch.stack([p.data.norm(p=2) for p in params]), p=2
        )
        scale = float(r_global * p_cont_norm) / float(grad_norm)

        for p in params:
            self._backups[p] = p.data.clone()
            p.add_(p.grad * scale)

    # ------------------------------------------------------------------ #
    # SAM interface
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def ascent_step(self):
        self._backups.clear()
        layer_r = {}
        if self.kappa_mode == "local":
            self._ascent_local(layer_r)
        else:
            self._ascent_global(layer_r)
        self._apply_continuous(layer_r)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def _record_r(self, layer_r, m, eps):
        layer_r[m] = {
            "eps_sq": float(eps.pow(2).sum().item()),
            "w_sq": float(m.x.data.pow(2).sum().item()),
        }

    @torch.no_grad()
    def _ascent_local(self, layer_r):
        for name, m in self.model.named_modules():
            if not self._is_quantized(m):
                continue
            res = self._layer_gain(m)
            if res is None:
                continue
            gain, delta_flip, valid = res
            d = gain.numel()
            k = int(self.kappa * d)
            eps, n_flipped = self._apply_topk(gain, delta_flip, valid, k)
            m.epsilon = eps
            self._record_r(layer_r, m, eps)
            self.flip_stats[name] = {
                "flips": n_flipped,
                "budget": k,
                "flip_frac": n_flipped / max(d, 1),
                "pos_gain_frac": (gain[valid] > 0).float().mean().item()
                if valid.any() else 0.0,
            }

    @torch.no_grad()
    def _ascent_global(self, layer_r):
        layers = []
        pooled_valid_gains = []
        total_valid = 0
        for name, m in self.model.named_modules():
            if not self._is_quantized(m):
                continue
            res = self._layer_gain(m)
            if res is None:
                continue
            gain, delta_flip, valid = res
            layers.append((name, m, gain, delta_flip, valid))
            vg = gain[valid]
            if vg.numel() > 0:
                pooled_valid_gains.append(vg.flatten())
            total_valid += int(valid.sum().item())

        k_global = int(self.kappa * total_valid)
        if k_global > 0 and pooled_valid_gains:
            pooled = torch.cat(pooled_valid_gains)
            kth = min(k_global, pooled.numel())
            thresh = torch.topk(pooled, kth, sorted=True).values[-1]
        else:
            thresh = None

        for name, m, gain, delta_flip, valid in layers:
            if thresh is None:
                eps = torch.zeros_like(delta_flip)
                n_flipped = 0
            else:
                flip_mask = valid & (gain >= thresh) & (gain > 0)
                eps = flip_mask.to(delta_flip.dtype) * delta_flip
                n_flipped = int(flip_mask.sum().item())
            m.epsilon = eps
            self._record_r(layer_r, m, eps)
            d = gain.numel()
            self.flip_stats[name] = {
                "flips": n_flipped,
                "budget_global": k_global,
                "flip_frac": n_flipped / max(d, 1),
                "pos_gain_frac": (gain[valid] > 0).float().mean().item()
                if valid.any() else 0.0,
            }

    @torch.no_grad()
    def _restore(self):
        # restore exactly what was backed up — no per-flag guards, no
        # asymmetry possible between perturb and restore paths
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