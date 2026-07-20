from models.LIQ_wn_qsam import QConv2d, QLinear
from utils.qsam import QSAM


class RoundQSAM(QSAM):
    """QSAM whose adversarial point is snapped back onto the quantization grid.

    Plain QSAM perturbs the *quantized* weight by epsilon and evaluates the
    descent gradient at ``quantized_weight + epsilon`` -- a point that lies OFF
    the grid. RoundQSAM rounds that perturbed weight to its nearest grid
    neighbour (nearest rounding, via a straight-through estimator inside
    ``LIQ_wn_qsam.quantize_weight_add_epsilon``) so the gradient driving the
    update is evaluated at a valid quantized point.

    Everything else -- the ascent perturbation, the clip / bias / BN handling,
    and the descent step -- is identical to QSAM. The only change is flipping
    ``round_epsilon`` on every quantized module; the rounding itself happens in
    the module's second forward.
    """

    def __init__(
        self,
        optimizer,
        model,
        rho=0.5,
        include_wclip=False,
        include_aclip=False,
        include_bn=True,
        include_bias=True,
    ):
        super().__init__(
            optimizer,
            model,
            rho=rho,
            include_wclip=include_wclip,
            include_aclip=include_aclip,
            include_bn=include_bn,
            include_bias=include_bias,
        )
        for m in self.model.modules():
            if isinstance(m, (QConv2d, QLinear)):
                m.round_epsilon = True
