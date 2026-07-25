"""
AdvQuant: SR-anchored adversarial rounding for quantized weights.

The adversary pins a subset S of coordinates to their non-nearest rounding
level while leaving the rest stochastic (SR). Relative to the SR posterior
P_SR (independent Bernoullis, P(ceil_i) = r_i), both the value and the cost
of a flip are measured against that SAME anchor:

    u_i    = |logit(r_i)|                          (how deep in the bin)
    value_i = gain_i * sigmoid(u_i)                (marginal loss gain over SR)
            = (g_i * delta_flip_i) * (1 - p_i),  p_i = min(r_i, 1-r_i)
    cost_i  = softplus(u_i) = KL(delta_flip_i || Bern(r_i))   (>= log 2)

Rationale for the SR anchor (both terms use it, so accounting is consistent):
  - value: SR already flips coord i with prob p_i, so deterministically
    pinning it adds only the (1 - p_i) fraction of the flip gain.
  - cost : softplus(u) is the exact KL of pinning one coordinate; its log-2
    floor means boundary weights (r ~ 1/2) are NOT free -- no cost hacks.
  - sigmoid and softplus are the mean and potential of the same
    exponential family, so value/cost is well-behaved everywhere:
        deep in bin (u large):  value -> gain,      cost -> u  (=> ~ old LogitFlip)
        at boundary (u -> 0)  :  value -> gain/2,   cost -> log 2

Inner problem (global, one network-wide budget):

    max_x  sum_i value_i x_i   s.t.  sum_i cost_i x_i <= tau * d_valid,  x in {0,1}

solved by the feasible gain/cost greedy (fractional-knapsack relaxation; not
an exact 0-1 solver, but each item's cost << budget so the gap is negligible).

Knob: tau (nats/valid weight). tau = 0 -> nearest-rounding QAT (no flips).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from models.LIQ_wn_qsam import QConv2d, QLinear
from utils.cont_perturb import CONT_MODES, gather_cont_params, apply_qsam_radius


class AdvQuant:

    def __init__(self, optimizer, model, tau: float = 0.01,
                 perturb_continuous: str = "none", rho: float = 0.05,
                 mask_grid_exact: bool = True, logit_eps: float = 1e-6):
        assert tau >= 0.0, tau
        assert perturb_continuous in CONT_MODES, perturb_continuous
        assert 0.0 < logit_eps < 0.5, logit_eps
        self.optimizer = optimizer
        self.model = model
        self.tau = float(tau)
        self.perturb_continuous = perturb_continuous
        self.rho = rho
        self.mask_grid_exact = mask_grid_exact
        self.logit_eps = logit_eps
        self.flip_stats: Dict[str, dict] = {}
        self._backups = {}

    def _is_quantized(self, m) -> bool:
        return isinstance(m, (QConv2d, QLinear)) and m.bits_weights != 32

    @torch.no_grad()
    def _geometry(self, module) -> Optional[dict]:
        """value = gain*sigmoid(u), cost = softplus(u), candidate mask."""
        g = module.x.grad
        if g is None:
            return None
        cache = getattr(module, "rounding_cache", None)
        if cache is None:
            raise RuntimeError(
                "[AdvQuant] rounding_cache missing. "
                "Did you patch quantize_weight() in LIQ_wn_qsam.py?"
            )
        r, nearest_is_floor, floor_lvl, n, step_out = cache

        direction = torch.where(
            nearest_is_floor, torch.ones_like(r), -torch.ones_like(r))
        delta_flip = direction * step_out

        valid = ~(nearest_is_floor & (floor_lvl >= n))
        if self.mask_grid_exact:
            valid &= r > 0.0

        r_safe = r.clamp(self.logit_eps, 1.0 - self.logit_eps)
        u = torch.logit(r_safe).abs()
        value = (g * delta_flip) * torch.sigmoid(u)   # SR-marginal gain
        cost = F.softplus(u)                          # exact flip KL, >= log 2

        cand = valid & (value > 0)
        return {"module": module, "delta": delta_flip,
                "value": value, "cost": cost, "cand": cand,
                "n_valid": int(valid.sum().item())}

    @torch.no_grad()
    def _write_epsilon(self, entry: dict, selected_flat: torch.Tensor) -> None:
        mask = selected_flat.view_as(entry["delta"])
        entry["module"].epsilon = mask.to(entry["delta"].dtype) * entry["delta"]

    @torch.no_grad()
    def ascent_step(self) -> None:
        self._backups.clear()
        self.flip_stats = {}

        # ---- pass 1: gather geometry, build the global candidate pool ----
        entries, names = [], []
        pool_val, pool_cost, pool_layer, pool_pos = [], [], [], []
        for name, m in self.model.named_modules():
            if not self._is_quantized(m):
                continue
            e = self._geometry(m)
            if e is None:
                continue
            lid = len(entries)
            entries.append(e)
            names.append(name)
            idx = e["cand"].flatten().nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                pool_val.append(e["value"].flatten()[idx])
                pool_cost.append(e["cost"].flatten()[idx])
                pool_layer.append(torch.full_like(idx, lid))
                pool_pos.append(idx)

        n_valid_total = sum(e["n_valid"] for e in entries)
        budget = self.tau * n_valid_total

        # ---- global gain/cost greedy over the pooled candidates ----
        chosen_layer = chosen_pos = None
        spent = 0.0
        if pool_val and budget > 0.0:
            V = torch.cat(pool_val)
            C = torch.cat(pool_cost)
            Lyr = torch.cat(pool_layer)
            Pos = torch.cat(pool_pos)
            order = torch.argsort(V / C, descending=True)
            keep = torch.cumsum(C[order], dim=0) <= budget
            sel = order[keep]
            chosen_layer = Lyr[sel]
            chosen_pos = Pos[sel]
            spent = float(C[sel].sum().item())

        # ---- scatter selection back to per-layer epsilon ----
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
            "budget_nats": budget, "spent_nats": spent,
            "budget_utilization": spent / max(budget, 1e-12),
        }

        # ---- continuous params: shared QSAM-identical treatment ----
        cont_params = gather_cont_params(self.model, self.perturb_continuous)
        apply_qsam_radius(self.model, cont_params, self.rho, self._backups)
        self.optimizer.zero_grad()

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
    def _restore(self) -> None:
        for p, data in self._backups.items():
            p.data = data
        self._backups.clear()
        self._clear_rounding_eps()

    @torch.no_grad()
    def descent_step(self) -> None:
        self._restore()
        self.optimizer.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def restore_step(self) -> None:
        self._restore()
        self.optimizer.zero_grad()