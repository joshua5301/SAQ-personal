# Experiment A, step 1: finetune a FULL-PRECISION PreResNet-20 with plain SAM.
# This reports the FP (non-quantized) accuracy during validation, and saves an
# FP checkpoint that eval_ptq.py later quantizes without any further training.
#
# Notes:
#   --network preresnet20      -> full-precision model (NOT qsampreresnet20)
#   --opt_type SAM_SGD         -> plain SAM minimizer on top of SGD
#   --rho 0.05                 -> standard SAM radius (the SAQ scripts' large
#                                 rho values are tuned for quantization; don't reuse them)
#   quant args (qw/qa/quan_type/clip_lr) are ignored on an FP network.
# Remember to replace the dataset path and (optionally) the pretrained model path.
python train_sam.py \
  --save_path ./output/cifar100/sam_fp/r20/ \
  --data_path ./dataset/ \
  --dataset cifar100 \
  --network preresnet20 \
  --opt_type SAM_SGD \
  --rho 0.5 \
  --lr 0.01 \
  --pretrained ./pretrained/cifar100_resnet.pth \
  --n_epochs 200 \
  --experiment_id 01 --seed 1 --gpu 0
