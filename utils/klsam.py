"""
KLSAM: flip-count-budgeted adversarial rounding.

Same tilted-rounding geometry as KLTilt, but the budget knob is the
*fraction of weights flipped* (kappa), not KL-nats (tau). This removes
the scale ambiguity that pushed KLTilt's optimal tau to ~1e-5: kappa is a
directly interpretable count ("flip 0.5% of weights"), stable across
gradient scale, bit-width, and training stage.

Tilted distribution (unchanged from KLTilt):

    pi_i = sigmoid( logit(r_i) + s * gain_i ),   gain_i = g_i * step_i

but the single global temperature s is chosen so that the *expected number
of flips away from nearest* equals kappa * d_valid:

    deterministic=True :   sum_i 1[pi_i > 1/2]                  = kappa * d
    deterministic=False:   sum_i flip_prob_i(s)  =  anchor + kappa * d
        where flip_prob_i = pi_i     if nearest_is_floor (flip = go ceil)
                          = 1 - pi_i if nearest_is_ceil  (flip = go floor),
        and anchor = sum_i flip_prob_i(0) = sum_i min(r_i, 1-r_i) is the
        flip mass SR already produces at s=0. Targeting anchor + kappa*d
        makes kappa the TILT-INDUCED EXCESS flip fraction over SR, isolating
        the adversarial tilt from SR's own rounding noise. (Deterministic
        mode has anchor 0 since the flip side's SR prob is always <= 1/2.)

Both flip-count functions are monotone non-decreasing in s (every pi_i
moves monotonically along the exponential-family path toward the gain
direction), so a log-space bisection on s is valid.

Knob: kappa (flip fraction of valid weights, in [0, 1)).
    kappa -> 0    ->  s -> 0.
        - deterministic : no flips (pi_i <= 1/2 for the flip side) -> nearest QAT
        - stochastic    : pi = r  -> plain SR-QAT (only SR-noise flips)
    kappa large   ->  worst-case rounding (flip every gain>0 coordinate).

The KL actually spent, sum_i KL(pi_i || r_i), is measured post-hoc and
reported in stats() for the PAC-Bayes bound; it is a diagnostic, not the
control variable.

Performance notes:
    - single global bisection per step (not per layer)
    - log-space tensor bisection with warm start from the previous step
    - no .item() / host-device sync in the hot path; stats are tensors,
      materialized only when stats() is called
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


class KLSAM:

    _LOG_S_MIN = math.log(1e-8)
    _LOG_S_MAX = math.log(1e12)

    def __init__(self, optimizer, model, kappa=0.01,
                 deterministic=False, mask_grid_exact=True,
                 bisect_iters=25, warm_start_halfwidth=3.0,
                 perturb_continuous="none", rho=0.05):
        assert 0.0 <= kappa < 1.0, kappa
        assert perturb_continuous in CONT_MODES, perturb_continuous
        self.optimizer = optimizer
        self.model = model
        self.kappa = kappa                 # target flip fraction (nats-free)
        self.deterministic = deterministic
        self.mask_grid_exact = mask_grid_exact
        self.bisect_iters = bisect_iters
        self.warm_start_halfwidth = warm_start_halfwidth
        self.perturb_continuous = perturb_continuous
        self.rho = rho
        self._backups = {}
        self._log_s_prev = None            # warm start (python float)
        self._stats = {}

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _is_quantized(self, m):
        return isinstance(m, (QConv2d, QLinear)) and m.bits_weights != 32

    @torch.no_grad()
    def _layer_entry(self, m):
        """
        Read rounding geometry. Returns
            (logit_r, gain_up, valid, nearest_is_floor, step_out)
        or None if grad / cache is unavailable.

        Convention (matches KLTilt): pi_i is the probability of rounding UP
        (ceil). logit_r = logit of that up-probability r_i.
        """
        g = m.x.grad
        if g is None:
            return None
        cache = getattr(m, "rounding_cache", None)
        if cache is None:
            raise RuntimeError(
                "[KLSAM] rounding_cache missing. "
                "Did you patch quantize_weight() in LIQ_wn_qsam.py?"
            )
        r, nearest_is_floor, floor_lvl, n, step_out = cache

        valid = ~(nearest_is_floor & (floor_lvl >= n))
        if self.mask_grid_exact:
            valid &= r > 0.0

        logit_r = torch.logit(r.clamp(1e-6, 1.0 - 1e-6))
        gain_up = g * step_out             # ~ L(ceil) - L(floor), signed
        return logit_r, gain_up, valid, nearest_is_floor, step_out

    @staticmethod
    def _flip_prob(pi, nearest_is_floor):
        """
        Probability of flipping AWAY from nearest.
          nearest is floor  -> flip = round up  -> prob = pi
          nearest is ceil   -> flip = round down-> prob = 1 - pi
        At s=0 (pi=r): equals r if nearest_is_floor else 1-r, i.e. the SR
        probability of the minor (flip) side, always <= 1/2.
        """
        return torch.where(nearest_is_floor, pi, 1.0 - pi)

    @torch.no_grad()
    def _expected_flips(self, logit_r, gain, nearest_is_floor, s):
        """Expected flip count at temperature s (stochastic objective)."""
        pi = torch.sigmoid(logit_r + s * gain)
        return self._flip_prob(pi, nearest_is_floor).sum()

    @torch.no_grad()
    def _hard_flips(self, logit_r, gain, nearest_is_floor, s):
        """Deterministic flip count at s: coords whose MAP crosses to flip."""
        pi = torch.sigmoid(logit_r + s * gain)
        # flip happens when the flip-side prob exceeds 1/2
        return (self._flip_prob(pi, nearest_is_floor) > 0.5).sum().to(gain.dtype)

    @torch.no_grad()
    def _bisect_global_s(self, logit_r, gain, nearest_is_floor, target):
        """
        Find s such that flip-count(s) = target, by bisection in log s.

        flip-count is monotone non-decreasing in s: increasing s pushes every
        pi_i toward the gain direction, so each flip-probability (and each hard
        indicator) is non-decreasing. Bounds live on device; the loop has no
        host-device sync. Warm start narrows the bracket around the previous s.

        For deterministic mode the count is a step function, so the bisection
        converges to the s-interval reproducing the target count as closely as
        the discrete ladder allows.
        """
        dev, dt = logit_r.device, logit_r.dtype
        count_fn = self._hard_flips if self.deterministic else self._expected_flips

        if self._log_s_prev is None:
            lo, hi = self._LOG_S_MIN, self._LOG_S_MAX
        else:
            lo = max(self._log_s_prev - self.warm_start_halfwidth, self._LOG_S_MIN)
            hi = min(self._log_s_prev + self.warm_start_halfwidth, self._LOG_S_MAX)

        log_lo = torch.full((), lo, device=dev, dtype=dt)
        log_hi = torch.full((), hi, device=dev, dtype=dt)
        target_t = torch.as_tensor(target, device=dev, dtype=dt)

        # widen to full range if the warm bracket misses the root
        c_lo = count_fn(logit_r, gain, nearest_is_floor, log_lo.exp())
        c_hi = count_fn(logit_r, gain, nearest_is_floor, log_hi.exp())
        bad = (c_lo > target_t) | (c_hi < target_t)
        log_lo = torch.where(bad, torch.full_like(log_lo, self._LOG_S_MIN), log_lo)
        log_hi = torch.where(bad, torch.full_like(log_hi, self._LOG_S_MAX), log_hi)

        for _ in range(self.bisect_iters):
            log_mid = 0.5 * (log_lo + log_hi)
            c = count_fn(logit_r, gain, nearest_is_floor, log_mid.exp())
            under = c < target_t
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
        pool_logit, pool_gain, pool_nif = [], [], []
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
            pool_nif.append(nearest_is_floor[valid].flatten())

        if not entries:
            self.optimizer.zero_grad()
            return

        pl = torch.cat(pool_logit)
        pg = torch.cat(pool_gain)
        pn = torch.cat(pool_nif)
        n_valid_total = pl.numel()

        # ---- single global temperature to hit the flip-count target ----
        # deterministic: s=0 flips nothing (SR prior of flip side <= 1/2), so
        #   target = kappa * d directly.
        # stochastic: s=0 already flips ~sum min(r,1-r) by SR noise alone, so
        #   we target that anchor PLUS kappa*d, making kappa the tilt-induced
        #   *excess* flip fraction over SR (isolates the tilt from SR noise).
        if self.kappa > 0.0 and n_valid_total > 0:
            if self.deterministic:
                target = self.kappa * n_valid_total
            else:
                anchor = self._expected_flips(
                    pl, pg, pn, torch.zeros((), device=pl.device, dtype=pl.dtype))
                target = anchor + self.kappa * n_valid_total
            log_s = self._bisect_global_s(pl, pg, pn, target)
            s = log_s.exp()
            self._log_s_prev = float(log_s)   # one scalar readback per step
        else:
            s = torch.zeros((), device=pl.device, dtype=pl.dtype)

        # ---- pass 2: tilt, realize flips, write epsilon ----
        kl_spent = torch.zeros((), device=pl.device, dtype=pl.dtype)
        flips = torch.zeros((), device=pl.device, dtype=pl.dtype)
        for name, m, logit_r, gain_up, valid, nearest_is_floor, step_out \
                in entries:
            pi = torch.sigmoid(logit_r + s * gain_up)
            # invalid coords pinned to their only feasible level (nearest)
            pi = torch.where(valid, pi, (~nearest_is_floor).to(pi.dtype))

            if self.deterministic:
                b_ceil = pi > 0.5
            else:
                b_ceil = torch.rand_like(pi) < pi

            nearest_is_ceil = ~nearest_is_floor
            # epsilon relative to nearest: values in {-step, 0, +step}
            m.epsilon = (b_ceil.to(step_out.dtype)
                         - nearest_is_ceil.to(step_out.dtype)) * step_out

            kl_spent += _bern_kl(pi[valid],
                                 torch.sigmoid(logit_r[valid])).sum()
            flips += (b_ceil != nearest_is_ceil)[valid].sum().to(kl_spent.dtype)

        # realized flip fraction away from nearest (includes SR noise in
        # stochastic mode); for stochastic, the tilt-induced excess is the
        # quantity that tracks kappa, so report both.
        flip_frac = flips / max(n_valid_total, 1)
        if self.deterministic:
            excess_frac = flip_frac
        else:
            anchor = self._expected_flips(
                pl, pg, pn, torch.zeros((), device=pl.device, dtype=pl.dtype))
            excess_frac = flip_frac - anchor / max(n_valid_total, 1)
        self._stats = {
            "s": s,
            "flip_frac": flip_frac,                 # total flips vs nearest
            "excess_flip_frac": excess_frac,        # tilt-induced, ~= kappa
            "kappa_target": torch.as_tensor(float(self.kappa)),
            "kl_per_weight": kl_spent / max(n_valid_total, 1),  # post-hoc bound
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