# Remember to replace the path of dataset, the path pretrained model, and the arch_bits
# One-pass: no --rho/--tau/--kappa, no --perturb_continuous. Only --varreg_*.
# Probe once at any lambda, read force_ratio, rescale to land near 0.05.
python train_sam.py --save_path ./output/cifar100/finetune/r20/ --data_path ./dataset/ --dataset cifar100 --lr 0.01 --clip_lr 0.01 --opt_type VarReg_SGD --varreg_lambda 1.0 --varreg_measure sr --varreg_schedule cosine --varreg_apply auto --network qsampreresnet20 --pretrained ./pretrained/cifar100_resnet.pth --qw 2.0 --qa 2.0 --quan_type LIQ_wn_qsam --experiment_id 01 --seed 1 --gpu 0
