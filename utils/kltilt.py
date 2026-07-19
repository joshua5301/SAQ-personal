"""
KLTilt: KL-budgeted adversarial rounding (global version).

Inner problem (exact, distributionally robust):

    max_{R : KL(R || P_SR) <= tau * d}  E_{b~R} [ L(Q_b(w)) ]

where P_SR is the stochastic-rounding posterior (independent Bernoullis,
P(ceil_i) = r_i) over the WHOLE network. The KL constraint is global, so
its dual solution is a single temperature: with first-order gains
gain_i = g_i * step_i (= L(ceil) - L(floor)), the tilted distribution is

    pi_i = sigmoid( logit(r_i) + s * gain_i ),

with ONE global s chosen so that  sum_i KL(pi_i || r_i) = tau * d_valid.

Knob: tau (nats per weight).
    tau = 0     ->  exact SR-QAT when stochastic + crn=True (pi = r and
                    the second pass reuses the first-pass SR sample, so
                    epsilon = 0 coordinate-wise). Deterministic mode gives
                    exact nearest-QAT at tau = 0 by the same construction.
    tau -> inf  ->  worst-case rounding (pi -> 1[gain > 0]).

First-pass measure is coupled to `deterministic` (diagonal 2x2):
    True  -> nearest first pass, MAP second pass
    False -> SR first pass, SR-sample second pass (with CRN by default)

Position dependence (weights deep in their bin are hard to flip) follows
from the KL geometry; no separate cost heuristic.

Performance notes:
    - single global bisection per step (not per layer)
    - log-space tensor bisection with warm start from the previous step
    - no .item() / host-device sync in the hot path; stats are kept as
      tensors and materialized only when stats() is called
"""

import math

import torch
from models.LIQ_wn_qsam import QConv2d, QLinear
from utils.cont_perturb import CONT_MODES, gather_cont_params, apply_qsam_radius


def _bern_kl(p, q):
    """KL(Bern(p) || Bern(q)), elementwise, numerically safe."""
    p = p.clamp(1e-7, 1.0 - 1e-7)
    q = q.clamp(1e-7, 1.0 - 1e-7)
    return p * (p / q).log() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).log()


class KLTilt:

    _LOG_S_MIN = math.log(1e-8)
    _LOG_S_MAX = math.log(1e12)

    def __init__(self, optimizer, model, tau=0.01,
                 deterministic=False, crn=True, mask_grid_exact=True,
                 bisect_iters=20, warm_start_halfwidth=3.0,
                 perturb_continuous="none", rho=0.05):
        assert perturb_continuous in CONT_MODES, perturb_continuous
        self.optimizer = optimizer
        self.model = model
        self.tau = tau
        self.deterministic = deterministic
        self.crn = crn
        self.mask_grid_exact = mask_grid_exact
        self.bisect_iters = bisect_iters
        # bisection bracket half-width (in log s) around the previous s
        self.warm_start_halfwidth = warm_start_halfwidth
        self.perturb_continuous = perturb_continuous
        self.rho = rho
        self._backups = {}       # {param: saved data}; restore iterates this
        self._log_s_prev = None  # warm start (python float, synced lazily)
        self._stats = {}         # tensors only; see stats()

        # couple first-pass measure to the flag: diagonal cells of the 2x2
        mode = "nearest" if deterministic else "sr"
        for m_ in model.modules():
            if isinstance(m_, (QConv2d, QLinear)):
                m_.rounding_mode = mode

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _is_quantized(self, m):
        return isinstance(m, (QConv2d, QLinear)) and m.bits_weights != 32

    @torch.no_grad()
    def _layer_entry(self, m):
        """
        Read rounding geometry from m.rounding_cache (written by
        quantize_weight in the first forward pass) and convert it to
        upper-rounding coordinates.

        Returns (logit_r, gain_up, valid, nearest_is_floor, step_out)
        or None if the gradient / cache is unavailable.
        """
        g = m.x.grad
        if g is None:
            return None
        cache = getattr(m, "rounding_cache", None)
        if cache is None:
            raise RuntimeError(
                "[KLTilt] rounding_cache missing. "
                "Did you patch quantize_weight() in LIQ_wn_qsam.py?"
            )
        r, nearest_is_floor, floor_lvl, n, step_out = cache

        # flippable coords: top-level weights cannot round further up,
        # grid-exact weights have a degenerate rounding decision
        valid = ~(nearest_is_floor & (floor_lvl >= n))
        if self.mask_grid_exact:
            valid &= r > 0.0

        logit_r = torch.logit(r.clamp(1e-6, 1.0 - 1e-6))
        gain_up = g * step_out               # ~ L(ceil) - L(floor), signed
        return logit_r, gain_up, valid, nearest_is_floor, step_out

    @torch.no_grad()
    def _bisect_global_s(self, logit_r, gain, target):
        """
        Single global temperature: find s with
            sum KL(sigmoid(logit_r + s * gain) || sigmoid(logit_r)) = target.

        KL is monotone in s (every pi_i moves monotonically away from r_i
        along the exponential-family path), so bisection in log s is valid.
        Bounds live on the GPU and are updated with torch.where; the loop
        performs no host-device synchronization. Warm start narrows the
        bracket around the previous step's s.
        """
        r = torch.sigmoid(logit_r)
        dev, dt = logit_r.device, logit_r.dtype

        if self._log_s_prev is None:
            lo, hi = self._LOG_S_MIN, self._LOG_S_MAX
        else:
            lo = max(self._log_s_prev - self.warm_start_halfwidth,
                     self._LOG_S_MIN)
            hi = min(self._log_s_prev + self.warm_start_halfwidth,
                     self._LOG_S_MAX)

        log_lo = torch.full((), lo, device=dev, dtype=dt)
        log_hi = torch.full((), hi, device=dev, dtype=dt)
        target_t = torch.as_tensor(target, device=dev, dtype=dt)

        # if the warm bracket misses the root, fall back to the full range
        kl_lo = _bern_kl(torch.sigmoid(logit_r + log_lo.exp() * gain), r).sum()
        kl_hi = _bern_kl(torch.sigmoid(logit_r + log_hi.exp() * gain), r).sum()
        bad = (kl_lo > target_t) | (kl_hi < target_t)
        log_lo = torch.where(bad, torch.full_like(log_lo, self._LOG_S_MIN),
                             log_lo)
        log_hi = torch.where(bad, torch.full_like(log_hi, self._LOG_S_MAX),
                             log_hi)

        for _ in range(self.bisect_iters):
            log_mid = 0.5 * (log_lo + log_hi)
            kl = _bern_kl(
                torch.sigmoid(logit_r + log_mid.exp() * gain), r
            ).sum()
            under = kl < target_t
            log_lo = torch.where(under, log_mid, log_lo)
            log_hi = torch.where(under, log_hi, log_mid)

        return 0.5 * (log_lo + log_hi)       # log s, 0-dim tensor on device

    # ------------------------------------------------------------------ #
    # SAM interface (engine.py calls these)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def ascent_step(self):
        self._backups.clear()

        # ---- pass 1: gather geometry, build the global pool ----
        entries = []
        pool_logit, pool_gain = [], []
        for name, m in self.model.named_modules():
            if not self._is_quantized(m):
                continue
            e = self._layer_entry(m)
            if e is None:
                continue
            logit_r, gain_up, valid, nearest_is_floor, step_out = e
            entries.append((name, m) + e)
            pool_logit.append(logit_r[valid].flatten())
            pool_gain.append(gain_up[valid].flatten())

        if not entries:
            self.optimizer.zero_grad()
            return

        pl = torch.cat(pool_logit)
        pg = torch.cat(pool_gain)
        n_valid_total = pl.numel()

        # ---- single global temperature ----
        if self.tau > 0 and n_valid_total > 0:
            log_s = self._bisect_global_s(pl, pg, self.tau * n_valid_total)
            s = log_s.exp()
            # lazy sync: one scalar readback per step, outside the loop
            self._log_s_prev = float(log_s)
        else:
            s = torch.zeros((), device=pl.device, dtype=pl.dtype)

        # ---- pass 2: tilt, sample, write epsilon ----
        kl_spent = torch.zeros((), device=pl.device, dtype=pl.dtype)
        flips = torch.zeros((), device=pl.device, dtype=pl.dtype)
        for name, m, logit_r, gain_up, valid, nearest_is_floor, step_out \
                in entries:
            pi = torch.sigmoid(logit_r + s * gain_up)
            # invalid coords are pinned to their only feasible level (nearest)
            pi = torch.where(valid, pi, (~nearest_is_floor).to(pi.dtype))

            if self.deterministic:
                b_ceil = pi > 0.5
            else:
                u = m.sr_u if (self.crn and getattr(m, "sr_u", None)
                               is not None) else torch.rand_like(pi)
                b_ceil = u < pi

            # epsilon relative to what the FIRST pass actually applied
            applied_is_ceil = m.applied_is_ceil
            m.epsilon = (b_ceil.to(step_out.dtype)
                         - applied_is_ceil.to(step_out.dtype)) * step_out

            kl_spent += _bern_kl(pi[valid],
                                 torch.sigmoid(logit_r[valid])).sum()
            flips += (b_ceil != applied_is_ceil)[valid].sum().to(kl_spent.dtype)

        # stats stay on device; materialize via stats() when logging
        self._stats = {
            "s": s,
            "kl_per_weight": kl_spent / max(n_valid_total, 1),
            "flip_from_applied_frac": flips / max(n_valid_total, 1),
            "n_valid": n_valid_total,
        }

        # ---- continuous params: shared QSAM-identical treatment ----
        cont_params = gather_cont_params(self.model, self.perturb_continuous)
        apply_qsam_radius(self.model, cont_params, self.rho, self._backups)

        self.optimizer.zero_grad()

    @torch.no_grad()
    def stats(self):
        """Materialize diagnostics (one sync). Call every N steps, not every step."""
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