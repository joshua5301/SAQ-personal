"""
GridSAM: adversarial rounding flips under a per-parameter distance budget.
No continuous-parameter perturbation -- flips only.

Inner problem:

    max_S  sum_{i in S} gain_i     s.t.  sum_{i in S} m_i^2 <= tau * d_valid

    gain_i = g_i * d_i                  (1st-order flip gain, d_i = +-step_i)
    m_i    = (1/2 - min(r_i, 1-r_i)) * step_i   (latent distance to boundary)

tau is the AVERAGE SQUARED LATENT DISTANCE PER PARAMETER the adversary may
spend. The budget is extensive (proportional to the number of flippable
weights), so the same tau means the same intervention density regardless of
network or layer size:

    scope="global": budget = tau * d_valid_total  (one shared pool; the
                    adversary may concentrate flips in whichever layers are
                    most efficient)
    scope="local" : budget = tau * d_valid_layer  (each layer gets its own
                    budget, proportional to its own size)

Cost is measured in weight-space units (step = 2*clip/n), deliberately:
a coarser grid means the weight sits farther from its rounding boundary in
absolute terms, so that flip genuinely requires more latent movement and is
less likely to occur at deployment. Consequently the realized flip fraction
adapts on its own -- fewer flips at low bit-width or large clip, more at high
bit-width -- and it also anneals during training as weights settle away from
their boundaries. That adaptation is the intended behaviour of a fixed tau,
not a scale artifact: tau fixes the adversary's *effort*, and the number of
flips that effort buys is a property of the current geometry.

Solver: rank candidates (gain > 0) by efficiency gain_i / m_i^2 and fill
until the budget is exhausted (greedy on the fractional-knapsack relaxation;
each item's cost is tiny relative to the budget, so the gap is negligible).

Properties: tau -> 0  ->  nearest-rounding QAT baseline (no flips).

m_floor_frac (default 0): optional lower bound m >= m_floor_frac * step. With
a distance budget, coordinates sitting exactly on their boundary cost ~0 and
are effectively free, so they can be selected in large numbers without
consuming budget. Set a small value (e.g. 0.01) to cap that; leave at 0 to
let oscillating weights be flipped freely (they are, after all, the ones that
really do flip). free_frac is logged either way.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from models.LIQ_wn_qsam import QConv2d, QLinear


class GridSAM:

    def __init__(self, optimizer, model, tau: float = 1e-4,
                 scope: str = "global", mask_grid_exact: bool = True,
                 m_floor_frac: float = 0.0):
        assert tau >= 0.0, tau
        assert scope in ("global", "local"), scope
        assert 0.0 <= m_floor_frac < 0.5, m_floor_frac
        self.optimizer = optimizer
        self.model = model
        self.tau = float(tau)              # avg squared latent distance / param
        self.scope = scope
        self.mask_grid_exact = mask_grid_exact
        self.m_floor_frac = float(m_floor_frac)
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

        # latent distance to the rounding boundary, in weight-space units
        m_dist = (0.5 - torch.minimum(r, 1.0 - r)) * step_out
        if self.m_floor_frac > 0.0:
            m_dist = torch.maximum(m_dist, self.m_floor_frac * step_out)
        m_sq = m_dist.square()

        cand = valid & (gain > 0)
        return {"module": module, "delta": delta_flip,
                "gain": gain, "cost": m_sq, "cand": cand,
                "n_valid": int(valid.sum().item())}

    # ------------------------------------------------------------------ #
    # solver
    # ------------------------------------------------------------------ #

    @staticmethod
    @torch.no_grad()
    def _greedy_fill(gain: torch.Tensor, cost: torch.Tensor,
                     budget: float) -> Tuple[torch.Tensor, float]:
        """Efficiency-ordered fill. Returns (bool mask, spent)."""
        sel = torch.zeros_like(gain, dtype=torch.bool)
        if budget <= 0.0 or gain.numel() == 0:
            return sel, 0.0
        # zero-cost items rank first (inf) and consume no budget -- intended:
        # a weight sitting on its boundary is free to flip.
        ratio = gain / cost.clamp_min(torch.finfo(cost.dtype).tiny)
        order = torch.argsort(ratio, descending=True)
        keep = torch.cumsum(cost[order], dim=0) <= budget
        chosen = order[keep]
        sel[chosen] = True
        return sel, float(cost[chosen].sum().item())

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

    # ---- budget = tau * d_valid_layer, independently per layer ---- #
    @torch.no_grad()
    def _select_local(self, names, entries) -> None:
        total_flips, total_valid = 0, 0
        total_spent, total_budget = 0.0, 0.0
        for name, e in zip(names, entries):
            budget = self.tau * e["n_valid"]
            idx = e["cand"].flatten().nonzero(as_tuple=True)[0]
            selected = torch.zeros(e["delta"].numel(), dtype=torch.bool,
                                   device=e["delta"].device)
            spent = 0.0
            if idx.numel() > 0:
                g = e["gain"].flatten()[idx]
                c = e["cost"].flatten()[idx]
                chosen, spent = self._greedy_fill(g, c, budget)
                selected[idx[chosen]] = True
            self._write_epsilon(e, selected)
            nf = int(selected.sum().item())
            total_flips += nf
            total_valid += e["n_valid"]
            total_spent += spent
            total_budget += budget
            self.flip_stats[name] = {
                "flips": nf, "n_valid": e["n_valid"],
                "flip_frac_valid": nf / max(e["n_valid"], 1),
                "budget": budget, "spent": spent,
                "budget_utilization": spent / max(budget, 1e-30),
            }
        self.flip_stats["__global__"] = {
            "flips": total_flips, "n_valid": total_valid,
            "flip_frac_valid": total_flips / max(total_valid, 1),
            "tau": self.tau, "budget": total_budget, "spent": total_spent,
            "budget_utilization": total_spent / max(total_budget, 1e-30),
        }

    # ---- budget = tau * d_valid_total, one shared pool ---- #
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
        budget = self.tau * n_valid_total

        chosen_layer = chosen_pos = None
        spent = 0.0
        if pool_g:
            G = torch.cat(pool_g)
            C = torch.cat(pool_c)
            Lyr = torch.cat(pool_layer)
            Pos = torch.cat(pool_pos)
            chosen, spent = self._greedy_fill(G, C, budget)
            chosen_layer = Lyr[chosen]
            chosen_pos = Pos[chosen]

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
            "tau": self.tau, "budget": budget, "spent": spent,
            "budget_utilization": spent / max(budget, 1e-30),
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