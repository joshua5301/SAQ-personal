"""Post-training-quantization (PTQ) evaluation.

Experiment A: take a full-precision checkpoint (e.g. one finetuned with plain
SAM) and measure its accuracy *after* quantization, WITHOUT any further
training. This tells us whether the FP minimum transfers well to a quantized
model (e.g. whether SAM's flat minima are quantization-robust).

The pipeline:
  1. Build a quantized network (qpreresnet20, ...).
  2. Load the FP checkpoint into it. CheckPoint.load_state is tolerant of key
     mismatches, so conv/bn/fc weights are copied and the extra clip-value
     parameters are simply left for calibration.
  3. Calibrate the per-layer clip values from a few batches of data (weight
     clips from the weight stats, activation clips from observed activations).
  4. Evaluate on the validation set. No optimizer, no backward pass.

IMPORTANT: use --quan_type LIQ (the default here), NOT LIQ_wn. LIQ_wn
standardizes weights (subtract mean / divide std) inside its forward, which
does not match a checkpoint trained without that normalization and breaks PTQ.
LIQ quantizes the raw weights, which is what an FP checkpoint expects.
"""

import os

import torch
import torch.nn as nn

import core.models as c_models
import models
import quan_models as customized_models
from core.checkpoint import CheckPoint
from core.dataloader import get_dataloader
from core.engine import val
from core.logger import get_logger
from core.utils import (
    init_distributed_mode,
    set_gpu,
    set_reproducible,
    setup_logger_for_distributed,
)
from qconfig import get_args

# Register both the full-precision (core.models) and quantized (quan_models)
# network factories under `models`, exactly like train_sam.py does.
for name in c_models.__dict__:
    if (
        name.islower()
        and not name.startswith("__")
        and callable(c_models.__dict__[name])
    ):
        models.__dict__[name] = c_models.__dict__[name]

for name in customized_models.__dict__:
    if (
        name.islower()
        and not name.startswith("__")
        and callable(customized_models.__dict__[name])
    ):
        models.__dict__[name] = customized_models.__dict__[name]


def is_quant_module(m):
    """A quantized conv/linear exposes learnable weight/activation clip values."""
    return hasattr(m, "weight_clip_value") and hasattr(m, "activation_clip_value")


@torch.no_grad()
def calibrate(model, calib_loader, device, num_batches, logger):
    """Set per-layer clip values from data (post-training calibration).

    - weight clip: 0.8 * max|w| via each module's own init_weight_clip_val().
    - activation clip: 0.8 * running max|input| observed over `num_batches`.

    We set init_state=True on every quantized module so the built-in forward
    auto-init (LIQ) does not overwrite these values from a single stray batch.
    """
    model.eval()
    quant_modules = [m for m in model.modules() if is_quant_module(m)]
    if len(quant_modules) == 0:
        logger.warning("No quantized modules found -- is --network a quantized net?")
        return

    # Weight clips: computed directly from the (already loaded) weights.
    for m in quant_modules:
        m.init_weight_clip_val()
        m.init_state = True  # freeze the built-in auto-init

    # Activation clips: observe the raw input to each quantized layer.
    act_max = {m: 0.0 for m in quant_modules}

    def make_hook(module):
        def hook(mod, inp):
            cur = inp[0].detach().abs().max().item()
            if cur > act_max[module]:
                act_max[module] = cur

        return hook

    handles = [m.register_forward_pre_hook(make_hook(m)) for m in quant_modules]

    seen = 0
    for image, _ in calib_loader:
        model(image.to(device))
        seen += 1
        if seen >= num_batches:
            break

    for h in handles:
        h.remove()

    for m in quant_modules:
        m.activation_clip_value.data.fill_(0.8 * act_max[m])

    logger.info(
        "Calibrated {} quantized modules over {} batch(es)".format(
            len(quant_modules), seen
        )
    )


if __name__ == "__main__":
    args = get_args()

    set_gpu(args)
    device = torch.device("cuda")

    args.world_size = 1
    init_distributed_mode(args)

    # Lightweight output dir just for the eval log (skip the training save-path
    # machinery, which encodes a lot of train-only hyperparameters).
    args.save_path = os.path.join(args.save_path, "ptq_eval")
    os.makedirs(args.save_path, exist_ok=True)
    logger = get_logger(args.save_path, "eval")
    setup_logger_for_distributed(args.rank == 0, logger)
    logger.info(args)

    set_reproducible(args.seed)
    train_loader, val_loader, _, _ = get_dataloader(args, logger)

    # Build the quantized network at the requested bit-widths.
    model = models.__dict__[args.network](
        num_classes=args.n_classes,
        quantize_first_last=args.quantize_first_last,
        quan_type=args.quan_type,
        bits_weights=args.qw,
        bits_activations=args.qa,
    )

    # Load the full-precision checkpoint (tolerant: clip params are skipped).
    checkpoint = CheckPoint(args.save_path, logger)
    assert args.pretrained is not None, "--pretrained (FP checkpoint) is required"
    check_point_params = torch.load(args.pretrained, map_location="cpu")
    model_state = check_point_params
    if isinstance(check_point_params, dict) and "model" in check_point_params:
        model_state = check_point_params["model"]
    new_model_state = {}
    for key, value in model_state.items():
        new_key = key.replace("module.", "") if "module." in key else key
        new_model_state[new_key] = value
    model = checkpoint.load_state(model, new_model_state)
    logger.info("|===>Loaded FP checkpoint: {}".format(args.pretrained))

    model = model.to(device)

    # Calibrate clip values, then evaluate. No training happens here.
    calibrate(model, train_loader, device, args.num_calib_batches, logger)

    criterion = nn.CrossEntropyLoss()
    logger.info(
        "|===>Evaluating W{}A{} PTQ accuracy".format(int(args.qw), int(args.qa))
    )
    val_error, val_loss, val5_error = val(
        model, val_loader, criterion, device, logger, None, 0, args
    )
    logger.info(
        "|===>PTQ W{}A{}  Top1 Acc: {:.4f}  Top5 Acc: {:.4f}".format(
            int(args.qw), int(args.qa), 100 - val_error, 100 - val5_error
        )
    )
