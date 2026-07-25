"""
GridSAM (count-budgeted): adversarial rounding flips, with the FLIP FRACTION
as the knob. No continuous-parameter perturbation -- flips only.

Inner problem (count budget):

    max_S  sum_{i in S} gain_i     s.t.  |S| <= kappa * d_valid

    gain_i = g_i * d_i                  (1st-order flip gain, d_i = +-step_i)
    m_i    = (1/2 - min(r_i, 1-r_i)) * step_i   (latent distance to boundary)

Solved exactly for this budget: rank candidates (gain > 0) by efficiency
gain_i / m_i^2 and take the top kappa * d_valid.

Why kappa instead of a distance budget rho^2: with an absolute rho^2 budget
the realized flip COUNT is fixed but the flip FRACTION scales as 1/d, and the
cost m^2 carries weight-space units (step = 2*clip/n), so the same rho means
wildly different interventions across network size, layer size, bit-width,
and even across training as clip is learned. kappa is dimensionless and
invariant to all four: "flip this fraction of the weights".

The boundary distance m still sets the ORDER (position-aware: weights close
to their rounding boundary are cheap and get flipped first) -- it just no
longer sets the budget. Ordering uses m in weight-space units, so a layer
with a coarser grid (larger step) is correctly deprioritized in global mode:
its flips genuinely require more latent movement.

scope="global": one network-wide count, kappa * d_valid_total, allocated
                across layers by efficiency (concentration allowed).
scope="local" : kappa * d_valid_layer per layer (uniform fraction, no
                cross-layer transfer).

Properties: kappa -> 0  ->  nearest-rounding QAT baseline (no flips).
Design constant: m^2 floor (m >= M_FLOOR_FRAC * step) so boundary-sitting
(oscillating) coordinates do not dominate the ranking with vanishing cost.

Spent distance sum_{i in S} m_i^2 is logged for post-hoc reporting (it is the
budget the constraint-form formulation would have used).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from models.LIQ_wn_qsam import QConv2d, QLinear


class GridSAM:

    M_FLOOR_FRAC = 0.00   # design constant: minimum flip cost, step fraction

    def __init__(self, optimizer, model, kappa: float = 0.01,
                 scope: str = "global", mask_grid_exact: bool = True,
                 min_flips_per_layer: int = 0):
        assert 0.0 <= kappa < 1.0, kappa
        assert scope in ("global", "local"), scope
        self.optimizer = optimizer
        self.model = model
        self.kappa = float(kappa)          # target flip fraction
        self.scope = scope
        self.mask_grid_exact = mask_grid_exact
        # local scope only: floor so tiny layers still get trained adversarially
        self.min_flips_per_layer = int(min_flips_per_layer)
        self.flip_stats: Dict[str, dict] = {}

    def _is_quantized(self, m) -> bool:
        return isinstance(m, (QConv2d, QLinear)) and m.bits_weights != 32

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _geometry(self, module) -> Optional[dict]:
        g = module.x.grad
        if g is None:
            return None
        cache = getattr(module, "rounding_cache", None)
        if cache is None:
            raise RuntimeError(
                "[GridSAM] rounding_cache missing. "
                "Did you patch quantize_weight() in LIQ_wn_qsam.py?"
            )
        r, nearest_is_floor, floor_lvl, n, step_out = cache

        dir_sign = torch.where(
            nearest_is_floor, torch.ones_like(r), -torch.ones_like(r))
        delta_flip = dir_sign * step_out

        valid = ~(nearest_is_floor & (floor_lvl >= n))
        if self.mask_grid_exact:
            valid &= r > 0.0

        gain = g * delta_flip
        m_dist = torch.clamp((0.5 - torch.minimum(r, 1.0 - r)) * step_out,
                             min=0.0)
        m_dist = torch.maximum(m_dist, self.M_FLOOR_FRAC * step_out)
        m_sq = m_dist.square()

        cand = valid & (gain > 0)
        return {"module": module, "delta": delta_flip,
                "gain": gain, "cost": m_sq, "cand": cand,
                "n_valid": int(valid.sum().item())}

    @torch.no_grad()
    def _write_epsilon(self, e: dict, selected_flat: torch.Tensor) -> None:
        mask = selected_flat.view_as(e["delta"])
        e["module"].epsilon = mask.to(e["delta"].dtype) * e["delta"]

    # ------------------------------------------------------------------ #
    # SAM interface (engine.py calls these)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def ascent_step(self) -> None:
        self.flip_stats = {}

        entries: List[dict] = []
        names: List[str] = []
        for name, m in self.model.named_modules():
            if not self._is_quantized(m):
                continue
            e = self._geometry(m)
            if e is None:
                continue
            entries.append(e)
            names.append(name)

        if not entries:
            self.optimizer.zero_grad()
            return

        if self.scope == "local":
            self._select_local(names, entries)
        else:
            self._select_global(names, entries)

        self.optimizer.zero_grad()

    # ---- kappa * d_valid_layer flips per layer ---- #
    @torch.no_grad()
    def _select_local(self, names, entries) -> None:
        total_flips, total_valid, total_spent = 0, 0, 0.0
        for name, e in zip(names, entries):
            idx = e["cand"].flatten().nonzero(as_tuple=True)[0]
            selected = torch.zeros(e["delta"].numel(), dtype=torch.bool,
                                   device=e["delta"].device)
            k_target = int(self.kappa * e["n_valid"])
            if self.min_flips_per_layer > 0 and e["n_valid"] > 0:
                k_target = max(k_target, self.min_flips_per_layer)
            spent = 0.0
            if idx.numel() > 0 and k_target > 0:
                g = e["gain"].flatten()[idx]
                c = e["cost"].flatten()[idx]
                k = min(k_target, idx.numel())
                top = torch.topk(g / c, k).indices
                selected[idx[top]] = True
                spent = float(c[top].sum().item())
            self._write_epsilon(e, selected)
            nf = int(selected.sum().item())
            total_flips += nf
            total_valid += e["n_valid"]
            total_spent += spent
            self.flip_stats[name] = {
                "flips": nf, "n_valid": e["n_valid"],
                "flip_frac_valid": nf / max(e["n_valid"], 1),
                "k_target": k_target, "spent_m_sq": spent,
            }
        self.flip_stats["__global__"] = {
            "flips": total_flips, "n_valid": total_valid,
            "flip_frac_valid": total_flips / max(total_valid, 1),
            "kappa_target": self.kappa, "spent_m_sq": total_spent,
        }

    # ---- kappa * d_valid_total flips network-wide ---- #
    @torch.no_grad()
    def _select_global(self, names, entries) -> None:
        pool_g, pool_c, pool_layer, pool_pos = [], [], [], []
        for lid, e in enumerate(entries):
            idx = e["cand"].flatten().nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                pool_g.append(e["gain"].flatten()[idx])
                pool_c.append(e["cost"].flatten()[idx])
                pool_layer.append(torch.full_like(idx, lid))
                pool_pos.append(idx)

        n_valid_total = sum(e["n_valid"] for e in entries)
        k_target = int(self.kappa * n_valid_total)

        chosen_layer = chosen_pos = None
        spent = 0.0
        if pool_g and k_target > 0:
            G = torch.cat(pool_g)
            C = torch.cat(pool_c)
            Lyr = torch.cat(pool_layer)
            Pos = torch.cat(pool_pos)
            k = min(k_target, G.numel())
            top = torch.topk(G / C, k).indices
            chosen_layer = Lyr[top]
            chosen_pos = Pos[top]
            spent = float(C[top].sum().item())

        total_flips = 0
        for lid, (name, e) in enumerate(zip(names, entries)):
            selected = torch.zeros(e["delta"].numel(), dtype=torch.bool,
                                   device=e["delta"].device)
            if chosen_pos is not None:
                here = chosen_pos[chosen_layer == lid]
                if here.numel() > 0:
                    selected[here] = True
            self._write_epsilon(e, selected)
            nf = int(selected.sum().item())
            total_flips += nf
            self.flip_stats[name] = {
                "flips": nf, "n_valid": e["n_valid"],
                "flip_frac_valid": nf / max(e["n_valid"], 1),
            }
        self.flip_stats["__global__"] = {
            "flips": total_flips, "n_valid": n_valid_total,
            "flip_frac_valid": total_flips / max(n_valid_total, 1),
            "kappa_target": self.kappa, "k_target": k_target,
            "spent_m_sq": spent,
        }

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _clear_rounding_eps(self) -> None:
        for m in self.model.modules():
            if not self._is_quantized(m):
                continue
            eps = getattr(m, "epsilon", None)
            if torch.is_tensor(eps):
                m.epsilon = torch.zeros_like(eps)
            elif hasattr(m, "epsilon"):
                m.epsilon = None

    @torch.no_grad()
    def descent_step(self) -> None:
        self._clear_rounding_eps()
        self.optimizer.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def restore_step(self) -> None:
        self._clear_rounding_eps()
        self.optimizer.zero_grad()