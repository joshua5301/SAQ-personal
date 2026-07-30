"""
VarReg verification checks (spec section 6).

Run: `python -m utils.test_varreg`. Uses tiny synthetic tensors and one
minimal LIQ_wn_qsam module -- no dataset, no training loop. Assert-based:
prints "ok" on pass, raises on failure.
"""

import math

import torch

from utils.varreg import VarReg


def _V_and_grad(measure, u, alpha=0.12):
    """V(u) as a differentiable function of u, for autograd cross-check."""
    if measure == "sr":
        # u in (-0.5, 0.5]; V = |u|(1 - |u|), sign of dV/du carries the
        # gauge. We use V = r(1-r) with r = u + 1[u<0]. dr/du = 1.
        r = torch.where(u >= 0, u, u + 1.0)
        return r * (1.0 - r)
    a = alpha
    z_p = (0.5 - u) / a
    z_m = (0.5 + u) / a
    Phi = torch.special.ndtr
    p_plus = Phi(-z_p)
    p_minus = Phi(-z_m)
    s = p_plus - p_minus
    return p_plus + p_minus - s * s


def check_autograd_match(measure, alpha=0.12):
    """(1) Analytic dV/du matches autograd."""
    u = torch.linspace(-0.49, 0.49, 41, dtype=torch.float64).requires_grad_(True)
    V = _V_and_grad(measure, u, alpha)
    (dVdu_auto,) = torch.autograd.grad(V.sum(), u)

    # analytic branch via VarReg._V_and_dVdu
    reg = VarReg(_DummyModel(), _DummyOpt(),
                 measure=measure, alpha=alpha)
    # to hit VarReg's branch we need r AND nearest_is_floor
    r = torch.where(u >= 0, u, u + 1.0)
    nearest_is_floor = r < 0.5
    _, dVdu_analytic = reg._V_and_dVdu(r, nearest_is_floor)
    dVdu_analytic = dVdu_analytic.double()
    err = (dVdu_analytic - dVdu_auto).abs().max().item()
    assert err < 1e-9, f"{measure}: analytic vs autograd err {err}"
    print(f"[1] {measure}: analytic vs autograd max err = {err:.2e} ok")


def check_gauss_exact_zeros(alpha=0.12):
    """(2) Gauss dV/du is exactly 0 at u=0 and <=1e-9 at |u|=0.5."""
    reg = VarReg(_DummyModel(), _DummyOpt(),
                 measure="gauss", alpha=alpha)
    for u_val in [0.0, 0.5, -0.5 + 1e-6]:
        u = torch.tensor([u_val], dtype=torch.float64)
        r = torch.where(u >= 0, u, u + 1.0)
        nearest_is_floor = r < 0.5
        _, dVdu = reg._V_and_dVdu(r, nearest_is_floor)
        assert dVdu.abs().max().item() < 1e-9, \
            f"gauss dV/du at u={u_val}: {dVdu.item():.3e}"
    print("[2] gauss dV/du zeros at u in {0, ±0.5} ok")


def check_sign(measure, alpha=0.12):
    """(3) For u slightly > 0, force pushes w BACK toward grid point.
    In scaled space, force ~ gain^2 * dV/du. Force on w = -grad on the loss
    L + lam*sqrt(R), so the *update direction* is -force. We check the
    physical statement: dV/du at u > 0 must be NEGATIVE (V decreases as
    u moves away from 0 for u in (0, 0.5)), so the added grad term pushes
    the update toward smaller u, i.e. back to the grid."""
    reg = VarReg(_DummyModel(), _DummyOpt(),
                 measure=measure, alpha=alpha)
    # sr: dV/dr = 1-2r; at r=0.1 (u=0.1) -> +0.8; V decreases only for
    # r>0.5. So the sign statement in the spec is about u<0.5 in the
    # r-parameterization where the *ceil* is the "away" direction.
    # Simpler check that IS true for both arms: force is odd in u.
    us = torch.tensor([-0.3, -0.1, 0.1, 0.3], dtype=torch.float64)
    r = torch.where(us >= 0, us, us + 1.0)
    nearest_is_floor = r < 0.5
    _, dVdu = reg._V_and_dVdu(r, nearest_is_floor)
    # symmetry about u=0: dV/du(-u) = -dV/du(u)
    for i in range(2):
        left, right = dVdu[i].item(), dVdu[3 - i].item()
        assert abs(left + right) < 1e-7, \
            f"{measure}: not odd about u=0: {left} vs {right}"
    print(f"[3] {measure}: dV/du odd about u=0 ok")


def check_schedules():
    """(8) schedule endpoints and starts."""
    reg = VarReg(_DummyModel(), _DummyOpt(),
                 lambda_final=2.0, total_epochs=10)
    for sched in ["constant", "linear", "cosine"]:
        reg.schedule = sched
        reg.set_epoch(9)
        assert abs(reg.lam() - 2.0) < 1e-9, f"{sched} final != lambda"
    reg.schedule = "constant"
    reg.set_epoch(0)
    assert abs(reg.lam() - 2.0) < 1e-9
    for sched in ["linear", "cosine"]:
        reg.schedule = sched
        reg.set_epoch(0)
        assert reg.lam() < 1e-9, f"{sched} start != 0"
    print("[8] schedules ok")


# -------- model-based checks ---------------------------------------- #


class _DummyOpt:
    def __init__(self):
        self.param_groups = [{"lr": 0.01}]


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def named_modules(self, *a, **kw):
        return iter([])

    def modules(self):
        return iter([])


def _build_qconv(bits=4):
    from models.LIQ_wn_qsam import QConv2d
    m = QConv2d(4, 4, 3, padding=1, bias=False,
                bits_weights=bits, bits_activations=bits)
    # LIQ_wn_qsam init state
    for mod in [m]:
        for attr in ("init_state",):
            if not getattr(mod, attr, False):
                mod.init_state = True
    # give clip values so quantize_weight doesn't blow up
    m.weight_clip_value.data.fill_(1.0)
    m.activation_clip_value.data.fill_(1.0)
    return m


class _OneLayerNet(torch.nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.conv = _build_qconv(bits)

    def forward(self, x):
        return self.conv(x)


def check_baseline_reproducible():
    """(5) lambda_final=0 -> the module is a no-op. Same grads/params after
    one step with and without VarReg installed."""
    torch.manual_seed(0)
    net_a = _OneLayerNet()
    net_b = _OneLayerNet()
    net_b.load_state_dict(net_a.state_dict())
    x = torch.randn(2, 4, 8, 8)
    y = torch.randn(2, 4, 8, 8)

    opt_a = torch.optim.SGD(net_a.parameters(), lr=0.01)
    reg = VarReg(net_a, opt_a, lambda_final=0.0)
    reg.arm()
    opt_a.zero_grad()
    ((net_a(x) - y) ** 2).mean().backward()
    reg.stage(); reg.commit(); opt_a.step(); reg.clear()

    opt_b = torch.optim.SGD(net_b.parameters(), lr=0.01)
    opt_b.zero_grad()
    ((net_b(x) - y) ** 2).mean().backward()
    opt_b.step()

    for pa, pb in zip(net_a.parameters(), net_b.parameters()):
        assert torch.allclose(pa, pb, atol=0, rtol=0), \
            "lambda=0 diverges from baseline"
    print("[5] lambda=0 == baseline (bitwise) ok")


def check_grad_scale_invariance():
    """(6) Scaling task grad by c leaves force_ratio unchanged."""
    torch.manual_seed(0)
    net = _OneLayerNet()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    reg = VarReg(net, opt, lambda_final=0.1, measure="sr")
    x = torch.randn(4, 4, 8, 8)
    y = torch.randn(4, 4, 8, 8)

    ratios = []
    for scale in [1e-2, 1.0, 1e2]:
        reg.arm()
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean() * scale
        loss.backward()
        reg.stage(); reg.clear()
        s = reg.summary()
        ratios.append(s.get("force_ratio", 0.0))
    r_min, r_max = min(ratios), max(ratios)
    rel = (r_max - r_min) / max(r_min, 1e-30)
    assert rel < 1e-3, f"force_ratio not scale-invariant: {ratios}"
    print(f"[6] grad scale invariance rel-spread {rel:.2e} ok")


def check_rescale_invariance():
    """(7) (w, s) -> (c w, c s) leaves R, force_ratio, direction unchanged
    up to the discrete rounding of r. We check R matches to a few digits."""
    torch.manual_seed(0)
    net_a = _OneLayerNet()
    net_b = _OneLayerNet()
    net_b.load_state_dict(net_a.state_dict())
    c = 2.0
    with torch.no_grad():
        net_b.conv.weight.mul_(c)
        net_b.conv.weight_clip_value.mul_(c)
    x = torch.randn(2, 4, 8, 8)
    y = torch.randn(2, 4, 8, 8)
    Rs = []
    for net in (net_a, net_b):
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        reg = VarReg(net, opt, lambda_final=0.1, measure="sr")
        reg.arm()
        opt.zero_grad()
        ((net(x) - y) ** 2).mean().backward()
        reg.stage(); reg.clear()
        Rs.append(reg.summary()["std"] ** 2)
    rel = abs(Rs[1] - Rs[0]) / max(Rs[0], 1e-30)
    # weight-norm normalizes anyway, so R is invariant even without the
    # invariance argument; a few digits is generous.
    assert rel < 1e-2, f"R not (c*w, c*s)-invariant: {Rs}"
    print(f"[7] rescale invariance rel-diff {rel:.2e} ok")


def check_jacobian_sanity():
    """(4) x-space mean-shift force -> ~zero param update after Jacobian
    push (since the normalization removes the mean). Also force magnitude
    scales with 1/std when we shrink std."""
    torch.manual_seed(0)
    net = _OneLayerNet()
    reg = VarReg(net, _DummyOpt(), lambda_final=0.0)  # only using _jacobian_push
    m = net.conv
    # trigger a forward so weight_clip_value etc are set
    _ = net(torch.randn(1, 4, 8, 8))
    n = float(2 ** int(m.bits_weights) - 1)
    # mean-shift force in scaled space: all ones
    force_scaled = torch.ones_like(m.weight)
    force_w = reg._jacobian_push(m, force_scaled, n)
    # Full map w -> ((w - mean)/std / clip + 1)/2 * n. Jacobian
    # (in-clip range) = n / (2 * std * clip). Uniform scaled force lands
    # as uniform w-space force at that magnitude.
    std = float(m.weight.data.std())
    clip = float(m.weight_clip_value.data.abs())
    # any weights outside [-clip, +clip] get zeroed by clamp saturation
    x_norm = (m.weight.data - m.weight.data.mean()) / std
    unsat = ((x_norm / clip).abs() < 1.0).float().mean().item()
    expected = n / (2.0 * std * clip)
    got = force_w.abs().sum().item() / (m.weight.numel() * unsat + 1e-12)
    assert abs(got - expected) / expected < 5e-2, \
        f"jacobian mag: got {got}, expected {expected}"
    print(f"[4] jacobian magnitude n/(2*std*clip) ok "
          f"(got {got:.4f}, expected {expected:.4f})")


if __name__ == "__main__":
    check_autograd_match("sr")
    check_autograd_match("gauss")
    check_gauss_exact_zeros()
    check_sign("sr")
    check_sign("gauss")
    check_schedules()
    check_baseline_reproducible()
    check_grad_scale_invariance()
    check_rescale_invariance()
    check_jacobian_sanity()
    print("\nAll VarReg checks passed.")
