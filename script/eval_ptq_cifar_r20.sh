# Experiment A, step 2: PTQ evaluation of the SAM-finetuned FP checkpoint.
# Loads the FP weights into a quantized PreResNet-20, calibrates clip values
# from a few batches, and evaluates. NO training happens.
#
#   --network qpreresnet20     -> quantized architecture
#   --quan_type LIQ            -> raw-weight quantizer (do NOT use LIQ_wn here;
#                                 its weight standardization breaks PTQ of an
#                                 FP checkpoint)
#   --pretrained <path>        -> the best FP model saved by step 1, e.g.
#                                 ./output/cifar100/sam_fp/r20/<run>/best_model.pth
#   --qw/--qa                  -> target bit-widths; rerun with different values
#                                 (e.g. 8/8, 4/4, 2/2) to sweep.
python eval_ptq.py \
  --save_path ./output/cifar100/sam_fp/r20/ \
  --data_path ./dataset/ \
  --dataset cifar100 \
  --network qpreresnet20 \
  --quan_type LIQ \
  --pretrained ./output/cifar100/sam_fp/r20/REPLACE_WITH_BEST_MODEL.pth \
  --qw 4.0 --qa 4.0 \
  --num_calib_batches 0 \
  --seed 1 --gpu 0
