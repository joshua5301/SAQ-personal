"""
Latent-Boundary GridSAM: constrained SAM with a single fixed radius rho
shared between rounding flips and continuous-parameter perturbation.

Inner problem (constraint form; exact SAM reduction when no flips):

    max_{S, e}  sum_{i in S} gain_i  +  g_c^T e
    s.t.        sum_{i in S} m_i^2  +  ||e||_2^2  <=  rho^2

where, per quantized coordinate i,
    gain_i = g_i * d_i          (1st-order flip gain, d_i = +-step_i)
    m_i    = latent distance to the rounding boundary, in output units
             = (1/2 - min(r_i, 1 - r_i)) * step_i
and e is the perturbation of the continuous parameters (clip / bias / BN).

Solution structure (concave in the spent budget):
    - candidates (valid, gain > 0) are sorted globally by ratio gain / m^2
    - for every prefix k:  Phi(k) = G_k + ||g_c|| * sqrt(rho^2 - C_k)
    - S* = argmax-prefix;  remaining radius rho_c = sqrt(rho^2 - C*)
      is spent on a standard (normalized) SAM step for continuous params:
          e = rho_c * g_c / ||g_c||

Properties:
    - no quantized flips selected  ->  exactly standard SAM with radius rho
    - rho -> 0                     ->  nearest-rounding baseline
    - rho is a genuine fixed L2 radius, inherited verbatim from SAQ

Design constants (not knobs):
    - m^2 floor: m_i^2 >= (M_FLOOR_FRAC * step_i)^2, preventing
      boundary-sitting (oscillating) coordinates from consuming the
      budget for free with vanishing cost.

Implementation note: flips are realized through the epsilon hook
(m.epsilon = +-step on selected coords), i.e. the second pass evaluates
Q(w) + eps via quantize_weight_add_epsilon. Under LIQ_wn this coincides
with actually moving the latent weight across its boundary up to the
tiny mean/std re-normalization side effect; the epsilon path is used
because it is exact in the cached geometry's domain and reuses the
battle-tested set_second_forward machinery.
"""

import torch
from models.LIQ_wn_qsam import QConv2d, QLinear
from utils.cont_perturb import CONT_MODES, gather_cont_params


class GridSAM:

    M_FLOOR_FRAC = 0.01   # design constant: minimum flip cost as a step fraction

    def __init__(self, optimizer, model, rho=0.05,
                 perturb_continuous="qsam_default", mask_grid_exact=True):
        assert perturb_continuous in CONT_MODES, perturb_continuous
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
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
    def _layer_candidates(self, m):
        """
        Returns (gain, delta_flip, m_sq, cand) for one quantized layer,
        or None if grad / cache is unavailable.
          cand: mask of flip candidates (valid & gain > 0)
          m_sq: floored squared latent boundary distance (output units)
        """
        g = m.x.grad
        if g is None:
            return None
        cache = getattr(m, "rounding_cache", None)
        if cache is None:
            raise RuntimeError(
                "[GridSAM] rounding_cache missing. "
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
        m_floor = self.M_FLOOR_FRAC * step_out
        m_sq = torch.maximum(m_dist, m_floor).square()

        cand = valid & (gain > 0)
        return gain, delta_flip, m_sq, cand

    # ------------------------------------------------------------------ #
    # SAM interface (engine.py calls these)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def ascent_step(self):
        self._backups.clear()
        rho_sq = self.rho * self.rho

        # ---- pass 1: gather candidates across all quantized layers ----
        layers = []                # (m, delta_flip, cand, d)
        pool_gain, pool_cost = [], []
        pool_layer, pool_pos = [], []
        for lid, (name, m) in enumerate(
                (n_, m_) for n_, m_ in self.model.named_modules()
                if self._is_quantized(m_)):
            res = self._layer_candidates(m)
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

        # ---- continuous gradient norm (for the sqrt term) ----
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

            # Phi(k) for prefixes k = 0..K (k = 0: pure SAM)
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
                # scatter chosen candidates back to per-layer epsilon
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
        }

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