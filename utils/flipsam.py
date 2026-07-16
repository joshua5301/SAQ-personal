from collections import defaultdict

import torch
from models.LIQ_wn_qsam import QConv2d, QLinear


class FlipSAM:
    """
    FlipSAM: top-k% adversarial rounding-flip minimizer for QAT.

    Budget mode (kappa_mode):
        "local"  -> per-layer top-k%: each quantized layer flips the top
                    kappa-fraction of ITS OWN weights by flip gain.
        "global" -> network-wide top-k%: a single threshold is applied
                    across all quantized weights, so the flip budget is
                    allocated preferentially to the most vulnerable layers.
                    Per-layer flip share becomes a sensitivity readout.

    In both modes only positive-gain flips are applied, so the realized
    flip count is min(budget, #{gain > 0}).

    Limits (both modes):
        kappa = 0  -> exact nearest-rounding QAT baseline (eps == 0).
        kappa = 1  -> hard adversarial flip (beta -> +inf of the Gibbs form).
    """

    def __init__(self, optimizer, model, kappa=0.01,
                 kappa_mode="local", mask_grid_exact=True):
        assert kappa_mode in ("local", "global"), kappa_mode
        self.optimizer = optimizer
        self.model = model
        self.kappa = kappa
        self.kappa_mode = kappa_mode
        self.mask_grid_exact = mask_grid_exact
        self.flip_stats = {}

    def _is_quantized(self, m):
        return isinstance(m, (QConv2d, QLinear)) and m.bits_weights != 32

    # ------------------------------------------------------------------ #
    # per-layer geometry: gain, delta_flip, validity (shared by both modes)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _layer_gain(self, m):
        """
        Returns (gain, delta_flip, valid) for one quantized layer, or None.
        gain/delta_flip/valid all have the same shape as m.x.
        """
        g = m.x.grad
        if g is None:
            return None
        cache = getattr(m, "rounding_cache", None)
        if cache is None:
            raise RuntimeError(
                f"[FlipSAM] rounding_cache missing. "
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
        """
        Given a flat-or-nd gain tensor, a matching delta_flip, a validity
        mask, and an integer budget k, return (eps, n_flipped) where eps is
        the flip perturbation (same shape) keeping only the top-k positive
        flips among valid entries.
        """
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
    # SAM interface
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def ascent_step(self):
        if self.kappa_mode == "local":
            self._ascent_local()
        else:
            self._ascent_global()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def _ascent_local(self):
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
            self.flip_stats[name] = {
                "flips": n_flipped,
                "budget": k,
                "flip_frac": n_flipped / max(d, 1),
                "pos_gain_frac": (gain[valid] > 0).float().mean().item()
                if valid.any() else 0.0,
            }

    @torch.no_grad()
    def _ascent_global(self):
        # pass 1: collect per-layer gain/geometry and the global valid-gain pool
        layers = []          # (name, m, gain, delta_flip, valid)
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

        # global budget over the total number of valid weights
        k_global = int(self.kappa * total_valid)

        # single global threshold from the pooled positive gains
        if k_global > 0 and pooled_valid_gains:
            pooled = torch.cat(pooled_valid_gains)
            kth = min(k_global, pooled.numel())
            # threshold = k_global-th largest valid gain
            thresh = torch.topk(pooled, kth, sorted=True).values[-1]
        else:
            thresh = None

        # pass 2: apply the shared threshold layer by layer
        for name, m, gain, delta_flip, valid in layers:
            if thresh is None:
                eps = torch.zeros_like(delta_flip)
                n_flipped = 0
            else:
                # flip valid entries whose gain >= global threshold AND > 0
                flip_mask = valid & (gain >= thresh) & (gain > 0)
                eps = flip_mask.to(delta_flip.dtype) * delta_flip
                n_flipped = int(flip_mask.sum().item())
            m.epsilon = eps
            d = gain.numel()
            self.flip_stats[name] = {
                "flips": n_flipped,
                "budget_global": k_global,
                "flip_frac": n_flipped / max(d, 1),
                # share of the global flip budget landing in this layer
                "flip_share": n_flipped / max(
                    sum(s["flips"] for s in self.flip_stats.values())
                    + n_flipped, 1
                ),
                "pos_gain_frac": (gain[valid] > 0).float().mean().item()
                if valid.any() else 0.0,
            }

    @torch.no_grad()
    def descent_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def restore_step(self):
        self.optimizer.zero_grad()