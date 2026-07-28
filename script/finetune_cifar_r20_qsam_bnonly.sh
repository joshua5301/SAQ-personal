# BN-only continuous SAM: perturbs BN affine params only.
# No quantized-weight perturbation, no clip/bias perturbation.
python train_sam.py --save_path ./output/cifar100/finetune/r20/ --data_path ./dataset/ --dataset cifar100 --lr 0.01 --clip_lr 0.01 --opt_type QSAM_SGD --rho 0.5 --include_wclip False --include_aclip False --include_bias False --include_bn True --include_qweight False --network qsampreresnet20 --pretrained ./pretrained/cifar100_resnet.pth --qw 2.0 --qa 2.0 --quan_type LIQ_wn_qsam --experiment_id 01 --seed 1 --gpu 0
