"""
EfficientGridSAM: GridSAM restricted to the LAST N quantized layers.

Same inner problem as GridSAM (constraint form, shared radius rho for flips +
continuous), but the flip candidate pool is limited to the last N QConv2d /
QLinear modules in forward order. All earlier quantized layers get
epsilon = 0, so the second pass evaluates the loss at (nearest weights,
perturbed continuous params, no flips) for them.

Motivation: measure the accuracy impact of concentrating the rounding
adversary in the network tail (where per-sample gradient magnitude tends
to be largest under a cross-entropy head), before committing to the
backward-truncation speedup that would follow.

Note on realized savings: this file does NOT yet truncate the first
backward -- it consumes the full-network gradient. It only restricts the
selection pool, so the accuracy delta is isolated from any compute change.
Backward truncation is a separate follow-up (e.g., toggling requires_grad
on non-tail params, or a tail-only proxy loss).

Knob:
    last_n_layers (int, default 3): number of trailing quantized layers
        eligible for flips. 1 = last conv/linear only; ~3 covers a typical
        MobileNetV2 inverted-residual tail; increasing recovers GridSAM.

Continuous-parameter perturbation is UNCHANGED from GridSAM (full network,
shared radius). Restrict there too if you want a pure "tail-only adversary".
"""

import torch
from models.LIQ_wn_qsam import QConv2d, QLinear
from utils.cont_perturb import CONT_MODES, gather_cont_params


class EfficientGridSAM:

    def __init__(self, optimizer, model, rho=0.05,
                 perturb_continuous="qsam_default", mask_grid_exact=True,
                 m_floor_frac=0.01, last_n_layers=3):
        assert perturb_continuous in CONT_MODES, perturb_continuous
        assert 0.0 <= m_floor_frac < 0.5, m_floor_frac
        assert last_n_layers >= 1, last_n_layers
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.perturb_continuous = perturb_continuous
        self.mask_grid_exact = mask_grid_exact
        self.m_floor_frac = float(m_floor_frac)
        self.last_n_layers = int(last_n_layers)
        self._backups = {}
        self._stats = {}

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _is_quantized(self, m):
        return isinstance(m, (QConv2d, QLinear)) and m.bits_weights != 32

    @torch.no_grad()
    def _layer_candidates(self, m):
        g = m.x.grad
        if g is None:
            return None
        cache = getattr(m, "rounding_cache", None)
        if cache is None:
            raise RuntimeError(
                "[EfficientGridSAM] rounding_cache missing. "
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

        m_dist = (0.5 - torch.minimum(r, 1.0 - r)) * step_out
        if self.m_floor_frac > 0.0:
            m_dist = torch.maximum(m_dist, self.m_floor_frac * step_out)
        m_sq = m_dist.square()

        cand = valid & (gain > 0)
        return gain, delta_flip, m_sq, cand

    # ------------------------------------------------------------------ #
    # SAM interface (engine.py calls these)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def ascent_step(self):
        self._backups.clear()
        rho_sq = self.rho * self.rho

        # ---- gather ALL quantized layers in forward order ----
        # named_modules() traversal order matches the forward pass for the
        # networks in this repo, so the trailing slice is the tail.
        all_layers = []              # (m, delta_flip, cand, gain, m_sq) or None
        q_modules = [m for _, m in self.model.named_modules()
                     if self._is_quantized(m)]
        tail_start = max(0, len(q_modules) - self.last_n_layers)

        layers = []                  # tail only: (m, delta_flip, cand)
        pool_gain, pool_cost = [], []
        pool_layer, pool_pos = [], []

        for lid, m in enumerate(q_modules):
            res = self._layer_candidates(m)
            in_tail = lid >= tail_start

            if not in_tail:
                # earlier layers: zero epsilon so second pass sees no flip
                if res is not None:
                    _, delta_flip, _, _ = res
                    m.epsilon = torch.zeros_like(delta_flip)
                continue

            if res is None:
                continue
            gain, delta_flip, m_sq, cand = res
            layers.append((m, delta_flip, cand))
            idx = cand.flatten().nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                pool_gain.append(gain.flatten()[idx])
                pool_cost.append(m_sq.flatten()[idx])
                pool_layer.append(torch.full_like(idx, len(layers) - 1))
                pool_pos.append(idx)

        # ---- continuous gradient norm (unchanged: full network) ----
        cont_params = [p for p in
                       gather_cont_params(self.model, self.perturb_continuous)
                       if p.grad is not None]
        if cont_params:
            gc_norm = torch.norm(torch.stack(
                [p.grad.norm(p=2) for p in cont_params]), p=2)
        else:
            gc_norm = None

        # ---- solver: global argmax-prefix by ratio = gain / m^2 ----
        n_flip = 0
        spent = None
        if pool_gain:
            G = torch.cat(pool_gain)
            C = torch.cat(pool_cost)
            L = torch.cat(pool_layer)
            P = torch.cat(pool_pos)

            order = torch.argsort(G / C, descending=True)
            cum_g = G[order].cumsum(0)
            cum_c = C[order].cumsum(0)
            feasible = cum_c <= rho_sq

            zero = torch.zeros((), device=G.device, dtype=G.dtype)
            resid = (rho_sq - cum_c).clamp_min(0.0).sqrt()
            if gc_norm is not None:
                phi = cum_g + gc_norm * resid
                phi0 = gc_norm * self.rho + zero
            else:
                phi = cum_g
                phi0 = zero
            phi = torch.where(feasible, phi,
                              torch.full_like(phi, torch.finfo(phi.dtype).min))
            best = int(torch.argmax(torch.cat([phi0.reshape(1), phi])))

            if best > 0:
                chosen = order[:best]
                n_flip = best
                spent = cum_c[best - 1]
                ch_layer = L[chosen]
                ch_pos = P[chosen]
                for lid, (m, delta_flip, cand) in enumerate(layers):
                    sel = ch_pos[ch_layer == lid]
                    mask = torch.zeros(delta_flip.numel(), dtype=torch.bool,
                                       device=delta_flip.device)
                    if sel.numel() > 0:
                        mask[sel] = True
                    m.epsilon = (mask.view_as(delta_flip)
                                 .to(delta_flip.dtype) * delta_flip)
            else:
                for m, delta_flip, _ in layers:
                    m.epsilon = torch.zeros_like(delta_flip)
        else:
            for m, delta_flip, _ in layers:
                m.epsilon = torch.zeros_like(delta_flip)

        dev = (cont_params[0].device if cont_params
               else (layers[0][1].device if layers else "cpu"))
        if spent is None:
            spent = torch.zeros((), device=dev)

        # ---- continuous params: standard SAM with the REMAINING radius ----
        rho_c = (rho_sq - spent).clamp_min(0.0).sqrt()
        if cont_params and gc_norm is not None and gc_norm > 1e-12:
            scale = rho_c / gc_norm
            for p in cont_params:
                self._backups[p] = p.data.clone()
                p.add_(p.grad * scale)

        self._stats = {
            "n_flip": torch.as_tensor(n_flip),
            "budget_spent_frac": spent / max(rho_sq, 1e-24),
            "rho_c": rho_c,
            "n_tail_layers": torch.as_tensor(len(layers)),
            "n_total_layers": torch.as_tensor(len(q_modules)),
        }

    @torch.no_grad()
    def stats(self):
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
