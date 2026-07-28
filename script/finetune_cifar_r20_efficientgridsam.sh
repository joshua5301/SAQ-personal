# Remember to replace the path of dataset, the path pretrained model, and the arch_bits
# --arch_bits "5.0, 4.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0"
python train_sam.py --save_path ./output/cifar100/finetune/r20/ --data_path ./dataset/ --dataset cifar100 --lr 0.01 --clip_lr 0.01 --opt_type EfficientGridSAM_SGD --perturb_continuous none --m_floor_frac 0.00 --last_n_layers 1 --network qsampreresnet20 --pretrained ./pretrained/cifar100_resnet.pth --qw 2.0 --qa 2.0 --quan_type LIQ_wn_qsam --experiment_id 01 --seed 1 --gpu 0
