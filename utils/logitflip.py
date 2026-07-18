"""
LogitFlip: likelihood-ball adversarial rounding for quantized weights.

For stochastic-rounding probability P(ceil_i)=r_i, flipping coordinate i
away from nearest rounding has the exact excess-surprisal cost

    cost_i = |logit(r_i)|.

With first-order flip gain gain_i = g_i * delta_flip_i, LogitFlip uses the
feasible gain/cost greedy approximation to

    max_x  sum_i gain_i * x_i
    s.t.   sum_i cost_i * x_i <= tau * d,  x_i in {0, 1}.

scope="global": one network-wide budget tau * d_valid_total.
scope="local" : one independent budget tau * d_valid_layer per layer.

The greedy solver is motivated by the fractional-knapsack relaxation; it is
not an exact solver for binary 0-1 knapsack.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from models.LIQ_wn_qsam import QConv2d, QLinear
from utils.cont_perturb import CONT_MODES, gather_cont_params, apply_qsam_radius


class LogitFlip:
    def __init__(
        self,
        optimizer,
        model,
        tau: float = 0.01,
        scope: str = "global",
        perturb_continuous: str = "none",
        rho: float = 0.05,
        mask_grid_exact: bool = True,
        logit_eps: float = 1e-6,
        free_cost_eps: float = 1e-12,
    ):
        assert tau >= 0.0, tau
        assert scope in ("global", "local"), scope
        assert perturb_continuous in CONT_MODES, perturb_continuous
        assert 0.0 < logit_eps < 0.5, logit_eps

        self.optimizer = optimizer
        self.model = model
        self.tau = float(tau)  # excess-surprisal budget, nats/valid weight
        self.scope = scope
        self.perturb_continuous = perturb_continuous
        self.rho = rho
        self.mask_grid_exact = mask_grid_exact
        self.logit_eps = logit_eps
        self.free_cost_eps = free_cost_eps

        self.flip_stats: Dict[str, dict] = {}
        self._backups = {}

    def _is_quantized(self, module) -> bool:
        return (
            isinstance(module, (QConv2d, QLinear))
            and module.bits_weights != 32
        )

    @torch.no_grad()
    def _clear_rounding_eps(self) -> None:
        """Make the next first pass clean nearest-rounding QAT."""
        for module in self.model.modules():
            if not self._is_quantized(module):
                continue
            eps = getattr(module, "epsilon", None)
            if torch.is_tensor(eps):
                module.epsilon = torch.zeros_like(eps)
            elif hasattr(module, "epsilon"):
                module.epsilon = None

    @torch.no_grad()
    def _geometry(self, name: str, module) -> Optional[dict]:
        grad = module.x.grad
        if grad is None:
            return None

        cache = getattr(module, "rounding_cache", None)
        if cache is None:
            raise RuntimeError(
                "[LogitFlip] rounding_cache missing. "
                "Did you patch quantize_weight() in LIQ_wn_qsam.py?"
            )

        r, nearest_is_floor, floor_lvl, n, step_out = cache

        direction = torch.where(
            nearest_is_floor, torch.ones_like(r), -torch.ones_like(r)
        )
        delta_flip = direction * step_out
        gain = grad * delta_flip

        # r is assumed to lie in [0, 1).  A top-level floor has no valid
        # upper neighbor; r == 0 is the grid-exact/degenerate SR case.
        valid = ~(nearest_is_floor & (floor_lvl >= n))
        if self.mask_grid_exact:
            valid &= r > 0.0

        r_safe = r.clamp(self.logit_eps, 1.0 - self.logit_eps)
        cost = torch.logit(r_safe).abs()

        return {
            "name": name,
            "module": module,
            "gain": gain,
            "delta": delta_flip,
            "cost": cost,
            "valid": valid,
        }

    @torch.no_grad()
    def _gather(self) -> List[dict]:
        entries = []
        for name, module in self.model.named_modules():
            if not self._is_quantized(module):
                continue
            entry = self._geometry(name, module)
            if entry is not None:
                entries.append(entry)
        return entries

    @torch.no_grad()
    def _select(
        self,
        gains: torch.Tensor,
        costs: torch.Tensor,
        budget: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gain/cost greedy; returns a feasible binary mask and spent cost."""
        selected = torch.zeros_like(gains, dtype=torch.bool)
        spent = torch.zeros((), device=gains.device, dtype=costs.dtype)

        # tau=0 is explicitly the nearest-rounding baseline.
        if budget <= 0.0 or gains.numel() == 0:
            return selected, spent

        free = costs <= self.free_cost_eps
        selected[free] = True

        paid_idx = (~free).nonzero(as_tuple=True)[0]
        if paid_idx.numel() == 0:
            return selected, spent

        paid_gain = gains[paid_idx]
        paid_cost = costs[paid_idx]
        order = torch.argsort(paid_gain / paid_cost, descending=True)
        cumulative = torch.cumsum(paid_cost[order], dim=0)
        take = cumulative <= budget

        if take.any():
            selected[paid_idx[order[take]]] = True
            spent = costs[selected].sum()

        return selected, spent

    @torch.no_grad()
    def _write_epsilon(self, entry: dict, selected_flat: torch.Tensor) -> None:
        mask = selected_flat.view_as(entry["delta"])
        entry["module"].epsilon = mask.to(entry["delta"].dtype) * entry["delta"]

    @torch.no_grad()
    def _local_rounding_ascent(self, entries: List[dict]) -> None:
        for entry in entries:
            gain = entry["gain"].flatten()
            cost = entry["cost"].flatten()
            valid = entry["valid"].flatten()
            candidate_idx = (valid & (gain > 0)).nonzero(as_tuple=True)[0]

            n_valid = int(valid.sum().item())
            budget = self.tau * n_valid
            selected = torch.zeros_like(valid)

            if candidate_idx.numel() > 0:
                chosen, spent = self._select(
                    gain[candidate_idx], cost[candidate_idx], budget
                )
                selected[candidate_idx[chosen]] = True
            else:
                spent = torch.zeros((), device=gain.device, dtype=cost.dtype)

            self._write_epsilon(entry, selected)
            n_flipped = int(selected.sum().item())
            self.flip_stats[entry["name"]] = {
                "flips": n_flipped,
                "flip_frac_valid": n_flipped / max(n_valid, 1),
                "budget_nats": budget,
                "spent_nats": float(spent.item()),
                "budget_utilization": float(spent.item()) / max(budget, 1e-12),
                "n_valid": n_valid,
            }

    @torch.no_grad()
    def _global_rounding_ascent(self, entries: List[dict]) -> None:
        candidate_gains: List[torch.Tensor] = []
        candidate_costs: List[torch.Tensor] = []
        candidate_indices: List[torch.Tensor] = []
        candidate_counts: List[int] = []
        valid_counts: List[int] = []

        for entry in entries:
            gain = entry["gain"].flatten()
            valid = entry["valid"].flatten()
            idx = (valid & (gain > 0)).nonzero(as_tuple=True)[0]

            candidate_indices.append(idx)
            candidate_counts.append(int(idx.numel()))
            valid_counts.append(int(valid.sum().item()))
            if idx.numel() > 0:
                candidate_gains.append(gain[idx])
                candidate_costs.append(entry["cost"].flatten()[idx])

        n_valid_total = sum(valid_counts)
        budget = self.tau * n_valid_total

        if candidate_gains:
            chosen_global, spent = self._select(
                torch.cat(candidate_gains), torch.cat(candidate_costs), budget
            )
        else:
            ref = entries[0]["gain"]
            chosen_global = torch.zeros(0, device=ref.device, dtype=torch.bool)
            spent = torch.zeros((), device=ref.device, dtype=ref.dtype)

        offset = 0
        total_flips = 0
        for entry, idx, n_candidate, n_valid in zip(
            entries, candidate_indices, candidate_counts, valid_counts
        ):
            selected = torch.zeros(
                entry["gain"].numel(),
                device=entry["gain"].device,
                dtype=torch.bool,
            )
            if n_candidate > 0:
                local_chosen = chosen_global[offset:offset + n_candidate]
                selected[idx[local_chosen]] = True
            offset += n_candidate

            self._write_epsilon(entry, selected)
            n_flipped = int(selected.sum().item())
            total_flips += n_flipped
            self.flip_stats[entry["name"]] = {
                "flips": n_flipped,
                "flip_frac_valid": n_flipped / max(n_valid, 1),
                "n_valid": n_valid,
            }

        self.flip_stats["__global__"] = {
            "flips": total_flips,
            "flip_frac_valid": total_flips / max(n_valid_total, 1),
            "budget_nats": budget,
            "spent_nats": float(spent.item()),
            "budget_utilization": float(spent.item()) / max(budget, 1e-12),
            "n_valid": n_valid_total,
        }

    @torch.no_grad()
    def ascent_step(self) -> None:
        self._backups.clear()
        self.flip_stats = {}
        entries = self._gather()

        if entries:
            if self.scope == "global":
                self._global_rounding_ascent(entries)
            else:
                self._local_rounding_ascent(entries)

        cont_params = gather_cont_params(self.model, self.perturb_continuous)
        apply_qsam_radius(self.model, cont_params, self.rho, self._backups)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def _restore(self) -> None:
        for param, data in self._backups.items():
            param.data = data
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

