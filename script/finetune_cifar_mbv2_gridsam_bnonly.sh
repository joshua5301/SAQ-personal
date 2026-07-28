# GridSAM (adversarial rounding flips) + BN-only continuous SAM.
# Shared rho^2 budget: flip cost m^2 sums + ||BN-only e||^2 <= rho^2.
python train_sam.py --save_path ./output/cifar100/finetune/mbv2/ --data_path ./dataset/ --dataset cifar100 --lr 0.01 --clip_lr 0.01 --opt_type GridSAM_SGD --rho 0.5 --perturb_continuous bn_only --m_floor_frac 0.00 --network qsammobilenetv2_cifar --pretrained ./pretrained/cifar100_mobilenetv2.pth --qw 3.0 --qa 3.0 --quan_type LIQ_wn_qsam --experiment_id 01 --seed 1 --gpu 0
