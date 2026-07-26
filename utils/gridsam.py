"""
GridSAM (gain-budgeted): adversarial rounding flips that raise the loss by a
fixed predicted amount, as cheaply as possible. No continuous-parameter
perturbation -- flips only.

This is the DUAL of the distance-budgeted formulation:

    distance budget:  max_S sum gain_i   s.t.  sum m_i^2 <= tau * d
    gain budget:      min_S sum m_i^2    s.t.  sum gain_i  >= c     <-- here

i.e. "the most plausible rounding perturbation that raises the loss by c",
instead of "the most damaging perturbation within a given effort".

    gain_i = g_i * d_i                  (1st-order flip gain, d_i = +-step_i)
    m_i    = (1/2 - min(r_i, 1-r_i)) * step_i   (latent distance to boundary)

Both problems have the SAME greedy solution ordering -- rank candidates by
efficiency gain_i / m_i^2 -- only the stopping rule differs: fill until the
accumulated GAIN reaches c (rather than until the accumulated cost reaches
the distance budget).

Why a gain budget: gain has units of LOSS, so c ("raise the training loss by
c nats") means the same thing across weight scales, bit-widths, clip values
and architectures -- the deepest scale-freeness available here, since
step and gradient magnitude cancel inside gain = g * step. Flip count then
scales roughly as sqrt(d) (per-coordinate gradients shrink as the model
grows), which matches the implicit scaling of standard SAM (dL = rho * ||g||).

    scope="global": one shared budget c over the whole network (the adversary
                    may concentrate wherever it is cheapest).
    scope="local" : c is split across layers in proportion to layer size, so
                    the totals of the two scopes are directly comparable.

CAVEATS (read before tuning):

  * gain is estimated from a minibatch, and we select the TOP of a noisy
    ranking, so sum gain over the selected set is biased upward (winner's
    curse). The realized loss increase can fall well short of c when the
    gradient noise is comparable to the signal; the bias also drifts with
    batch size and training stage. The distance budget does not have this
    problem (m is deterministic). Consider ranking on one half of the batch
    and accumulating gain on the other if this bites.

  * unlike the distance budget, there is no crystallization-driven annealing:
    as training converges gradients shrink, so MORE flips are needed to reach
    the same c. Flip count tends to grow late in training. Watch flip_frac.

  * near-boundary coordinates have cost ~ 0 and therefore huge efficiency,
    but contribute almost nothing to the accumulated gain, so they are all
    taken before the budget moves. Set m_floor_frac > 0 (e.g. 0.01) if flip
    counts explode.

Properties: c -> 0  ->  nearest-rounding QAT baseline (no flips).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from models.LIQ_wn_qsam import QConv2d, QLinear


class GridSAM:

    def __init__(self, optimizer, model, c: float = 1e-2,
                 scope: str = "global", mask_grid_exact: bool = True,
                 m_floor_frac: float = 0.0):
        assert c >= 0.0, c
        assert scope in ("global", "local"), scope
        assert 0.0 <= m_floor_frac < 0.5, m_floor_frac
        self.optimizer = optimizer
        self.model = model
        self.c = float(c)                  # target 1st-order loss increase
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

        m_dist = (0.5 - torch.minimum(r, 1.0 - r)) * step_out
        if self.m_floor_frac > 0.0:
            m_dist = torch.maximum(m_dist, self.m_floor_frac * step_out)
        m_sq = m_dist.square()

        cand = valid & (gain > 0)
        return {"module": module, "delta": delta_flip,
                "gain": gain, "cost": m_sq, "cand": cand,
                "n_valid": int(valid.sum().item())}

    # ------------------------------------------------------------------ #
    # solver: same efficiency ordering, stop on accumulated GAIN
    # ------------------------------------------------------------------ #

    @staticmethod
    @torch.no_grad()
    def _greedy_fill(gain: torch.Tensor, cost: torch.Tensor,
                     c: float) -> Tuple[torch.Tensor, float, float, bool]:
        """Returns (mask, accumulated gain, accumulated cost, saturated)."""
        sel = torch.zeros_like(gain, dtype=torch.bool)
        if c <= 0.0 or gain.numel() == 0:
            return sel, 0.0, 0.0, False
        ratio = gain / cost.clamp_min(torch.finfo(cost.dtype).tiny)
        order = torch.argsort(ratio, descending=True)
        cum_gain = torch.cumsum(gain[order], dim=0)
        keep = cum_gain <= c
        chosen = order[keep]
        sel[chosen] = True
        got = float(gain[chosen].sum().item())
        return (sel, got, float(cost[chosen].sum().item()),
                bool(keep.all()))          # saturated: never reached c

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

    # ---- c split across layers in proportion to layer size ---- #
    @torch.no_grad()
    def _select_local(self, names, entries) -> None:
        n_valid_total = max(sum(e["n_valid"] for e in entries), 1)
        total_flips, total_valid = 0, 0
        total_gain, total_cost = 0.0, 0.0
        n_sat = 0
        for name, e in zip(names, entries):
            c_layer = self.c * e["n_valid"] / n_valid_total
            idx = e["cand"].flatten().nonzero(as_tuple=True)[0]
            selected = torch.zeros(e["delta"].numel(), dtype=torch.bool,
                                   device=e["delta"].device)
            got_gain = got_cost = 0.0
            sat = False
            if idx.numel() > 0:
                g = e["gain"].flatten()[idx]
                cst = e["cost"].flatten()[idx]
                chosen, got_gain, got_cost, sat = self._greedy_fill(
                    g, cst, c_layer)
                selected[idx[chosen]] = True
            self._write_epsilon(e, selected)
            nf = int(selected.sum().item())
            total_flips += nf
            total_valid += e["n_valid"]
            total_gain += got_gain
            total_cost += got_cost
            n_sat += int(sat)
            self.flip_stats[name] = {
                "flips": nf, "n_valid": e["n_valid"],
                "flip_frac_valid": nf / max(e["n_valid"], 1),
                "c_layer": c_layer, "gain_spent": got_gain,
                "cost_spent": got_cost, "saturated": sat,
            }
        self.flip_stats["__global__"] = {
            "flips": total_flips, "n_valid": total_valid,
            "flip_frac_valid": total_flips / max(total_valid, 1),
            "c": self.c, "gain_spent": total_gain, "cost_spent": total_cost,
            "gain_utilization": total_gain / max(self.c, 1e-30),
            "n_saturated_layers": n_sat,
        }

    # ---- one shared budget c over the whole network ---- #
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

        chosen_layer = chosen_pos = None
        got_gain = got_cost = 0.0
        sat = False
        if pool_g:
            G = torch.cat(pool_g)
            C = torch.cat(pool_c)
            Lyr = torch.cat(pool_layer)
            Pos = torch.cat(pool_pos)
            chosen, got_gain, got_cost, sat = self._greedy_fill(G, C, self.c)
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
            "c": self.c, "gain_spent": got_gain, "cost_spent": got_cost,
            "gain_utilization": got_gain / max(self.c, 1e-30),
            "saturated": sat,
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