"""
Shared continuous-parameter SAM machinery for the rounding-robustness
minimizers (FlipSAM / FlipQSAM / TiltedSR).

The point of sharing: every minimizer must perturb the SAME parameter set
with the SAME normalization, so comparison runs differ only in how the
quantized weights are perturbed. "qsam_default" reproduces the QSAM
baseline's default flags (include_aclip=True, include_bn=True,
include_wclip=False, bias always on): activation clip + bias + BN affine.
"""

import torch
import torch.nn as nn

from models.LIQ_wn_qsam import QConv2d, QLinear

CONT_MODES = ("none", "clip", "clip_bias", "all", "qsam_default")


def gather_cont_params(model, mode):
    """Continuous parameters to perturb under the given scope, in module
    order. Only params that currently hold a gradient are returned."""
    assert mode in CONT_MODES, mode
    params = []
    for _, m in model.named_modules():
        if isinstance(m, (QConv2d, QLinear)):
            cands = []
            if mode in ("clip", "clip_bias", "all"):
                cands += [getattr(m, "weight_clip_value", None),
                          getattr(m, "activation_clip_value", None)]
            elif mode == "qsam_default":
                cands.append(getattr(m, "activation_clip_value", None))
            if mode in ("clip_bias", "all", "qsam_default"):
                cands.append(getattr(m, "bias", None))
            for p in cands:
                if p is not None and p.grad is not None:
                    params.append(p)
        if mode in ("all", "qsam_default") and \
                isinstance(m, nn.BatchNorm2d) and m.weight is not None:
            for p in (m.weight, m.bias):
                if p.grad is not None:
                    params.append(p)
    return params


@torch.no_grad()
def apply_qsam_radius(model, params, rho, backups):
    """
    QSAM-identical continuous ascent: one global grad norm with the
    quantized weight grads INCLUDED in the denominator, independent rho:

        e_p = rho * grad_p / ||[grad_w; grad_cont]||

    Perturbed originals are recorded in `backups` (param -> saved data)
    so the caller's restore path stays a single dict iteration.
    """
    if not params or rho <= 0:
        return
    grad_norms = [p.grad.norm(p=2) for p in params]
    for _, m in model.named_modules():
        if isinstance(m, (QConv2d, QLinear)) and m.bits_weights != 32:
            x = getattr(m, "x", None)
            if x is not None and x.grad is not None:
                grad_norms.append(x.grad.norm(p=2))
    grad_norm = torch.norm(torch.stack(grad_norms), p=2)
    scale = rho / (grad_norm + 1e-12)
    for p in params:
        backups[p] = p.data.clone()
        p.add_(p.grad * scale.to(p))
