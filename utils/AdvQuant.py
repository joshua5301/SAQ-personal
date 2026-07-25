"""
AdvQuant (SR-anchored, SR first pass): adversarial rounding by PINNING.

Objective (what the accounting is derived from): the adversary picks a set S
and pins each i in S to its MINOR level (the non-nearest side of the SR
distribution), leaving all other coordinates stochastic:

    Q_A = prod_{i in S} delta_minor_i  x  prod_{i not in S} Bern(r_i)

Against the SR anchor P_SR this gives, per coordinate (u_i = |logit r_i|):

    value_i = E_{Q_A}[L] - E_{P_SR}[L] = gain_i * sigmoid(u_i)
    cost_i  = KL(delta_minor_i || Bern(r_i)) = softplus(u_i)   (>= log 2)

Pinning is an act on the DISTRIBUTION: the flip target is the minor level,
fixed by the geometry (nearest_is_floor), NOT by the sample the first pass
happened to draw. The SR sample plays exactly two roles:

  1. measurement point: the first pass samples P_SR, so m.x.grad is a
     1-sample unbiased estimate of the SR-expected gradient (gain estimator);
  2. base of the second pass: with common random numbers (the same sr_u),
     the second pass reproduces the first pass's rounding on unselected
     coordinates, so the loss difference is attributable to the pins alone.

Realization: for a selected coordinate the second pass must sit at the minor
level, so

    epsilon_i = (minor_level_i - applied_level_i) * step_i
              = (nearest_is_floor_i - applied_is_ceil_i) * step_i
              in {-step, 0, +step}

epsilon = 0 when the SR draw already landed on the minor side -- the pin is
already satisfied, a genuine no-op. In expectation the fraction of selected
coordinates that actually move is sigma(u): the same factor that discounts
the value. Derivation and realization are two faces of one object.

tau = 0 with CRN reproduces SR-QAT exactly (epsilon == 0 everywhere and the
second pass equals the first).

REQUIRES the quantizer patched so that
  - quantize_weight supports rounding_mode="sr" while training and stores
    self.applied_is_ceil and self.sr_u (already done), and
  - quantize_weight_add_epsilon REUSES self.sr_u (CRN) to rebuild the same
    SR base before adding epsilon (patch below, mirror of quantize_weight):

        if getattr(self, "rounding_mode", "nearest") == "sr" \
                and self.training and self.sr_u is not None:
            scaled = x.detach() * n
            floor_lvl = torch.floor(scaled)
            applied_is_ceil = self.sr_u < (scaled - floor_lvl)
            q01 = (floor_lvl + applied_is_ceil.to(x.dtype)) / n
            x_q = x + (q01 - x).detach()
        else:
            x_q = quantization(x, k)

Knob: tau (nats per valid weight); realized flip statistics are logged so a
count-parameterized (kappa) variant can be swapped in if tau drifts.
"""

from __future__ import annotations

from typing import Dict, Optional

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
        # first pass samples the SR distribution (eval stays nearest via the
        # quantizer's self.training guard)
        for m in self.model.modules():
            if self._is_quantized(m):
                m.rounding_mode = "sr"

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
                "[AdvQuant] rounding_cache missing. "
                "Did you patch quantize_weight() in LIQ_wn_qsam.py?"
            )
        applied_is_ceil = getattr(module, "applied_is_ceil", None)
        if applied_is_ceil is None:
            raise RuntimeError(
                "[AdvQuant] applied_is_ceil missing. "
                "Is rounding_mode='sr' active on the first pass?"
            )
        r, nearest_is_floor, floor_lvl, n, step_out = cache

        # ---- flip target is the MINOR level (distribution-defined) ----
        # minor = ceil  if nearest is floor  -> displacement +step
        # minor = floor if nearest is ceil   -> displacement -step
        to_minor = torch.where(nearest_is_floor,
                               torch.ones_like(r), -torch.ones_like(r))
        d_minor = to_minor * step_out          # major -> minor displacement

        # the minor level must exist on the grid
        valid = ~(nearest_is_floor & (floor_lvl >= n))
        if self.mask_grid_exact:
            valid &= r > 0.0

        # ---- SR-anchored accounting ----
        r_safe = r.clamp(self.logit_eps, 1.0 - self.logit_eps)
        u = torch.logit(r_safe).abs()
        gain = g * d_minor                     # SR-sample-point gradient est.
        value = gain * torch.sigmoid(u)        # marginal over the SR anchor
        cost = F.softplus(u)                   # exact pin KL, >= log 2

        # ---- realization: epsilon relative to the APPLIED SR sample ----
        # {-step, 0, +step}; 0 when the draw already sits on the minor side
        eps_if_selected = (nearest_is_floor.to(step_out.dtype)
                           - applied_is_ceil.to(step_out.dtype)) * step_out

        cand = valid & (value > 0)
        return {"module": module, "eps_full": eps_if_selected,
                "value": value, "cost": cost, "cand": cand,
                "n_valid": int(valid.sum().item())}

    # ------------------------------------------------------------------ #
    # SAM interface (engine.py calls these)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def ascent_step(self) -> None:
        self._backups.clear()
        self.flip_stats = {}

        # ---- gather the global candidate pool ----
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

        # ---- global value/cost greedy ----
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

        # ---- write per-layer epsilon (pin selected coords to minor) ----
        total_sel = 0
        total_moved = 0
        for lid, (name, e) in enumerate(zip(names, entries)):
            selected = torch.zeros(e["eps_full"].numel(), dtype=torch.bool,
                                   device=e["eps_full"].device)
            if chosen_pos is not None:
                here = chosen_pos[chosen_layer == lid]
                if here.numel() > 0:
                    selected[here] = True
            sel_mask = selected.view_as(e["eps_full"])
            e["module"].epsilon = (sel_mask.to(e["eps_full"].dtype)
                                   * e["eps_full"])
            nf = int(sel_mask.sum().item())
            moved = int((e["module"].epsilon != 0).sum().item())
            total_sel += nf
            total_moved += moved
            self.flip_stats[name] = {
                "selected": nf,
                "moved": moved,                 # draw was on major -> pin moves it
                "already_minor": nf - moved,    # draw was on minor -> pin is no-op
                "n_valid": e["n_valid"],
            }

        self.flip_stats["__global__"] = {
            "selected": total_sel,
            "moved": total_moved,
            "already_minor": total_sel - total_moved,
            "moved_frac_of_selected": total_moved / max(total_sel, 1),
            "selected_frac_valid": total_sel / max(n_valid_total, 1),
            "n_valid": n_valid_total,
            "budget_nats": budget,
            "spent_nats": spent,
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