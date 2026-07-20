# Finetune a quantized PreResNet-20 with RoundQSAM.
# Same setup as the QSAM finetune, but the adversarial (second-forward) weight
# is snapped back onto the quantization grid via nearest rounding, so the
# gradient driving the update is evaluated at a valid quantized point.
#
#   --opt_type RoundQSAM_SGD   -> RoundQSAM minimizer on top of SGD
#   --network qsampreresnet20  -> quantized net using LIQ_wn_qsam modules
#   --quan_type LIQ_wn_qsam    -> the qsam quantizer (has quantize_weight_add_epsilon)
# NOTE: because the perturbation only changes the forward once epsilon crosses a
# rounding boundary, RoundQSAM typically needs a larger rho than plain QSAM.
#
# Exclude ALL continuous parameters from the SAM perturbation, so only the
# quantized weight is perturbed:
#   --include_wclip False  -> do not perturb weight clip value
#   --include_aclip False  -> do not perturb activation clip value
#   --include_bias  False  -> do not perturb conv/linear bias
#   --include_bn    False  -> do not perturb BatchNorm gamma/beta
# Remember to replace the dataset path and the pretrained model path.
python train_sam.py \
  --save_path ./output/cifar100/finetune/r20_roundqsam/ \
  --data_path ./dataset/ \
  --dataset cifar100 \
  --lr 0.01 --clip_lr 0.01 \
  --opt_type RoundQSAM_SGD \
  --network qsampreresnet20 \
  --rho 0.9 \
  --pretrained ./pretrained/cifar100_resnet.pth \
  --qw 4.0 --qa 4.0 \
  --quan_type LIQ_wn_qsam \
  --include_wclip False \
  --include_aclip False \
  --include_bias False \
  --include_bn False \
  --experiment_id 01 --seed 1 --gpu 0
