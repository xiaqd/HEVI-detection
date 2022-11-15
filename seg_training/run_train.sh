python3 train.py --num_classes 2 \
                 --dataset_root dataset_root_path \
                 --dataset dataset_dir_name \
                 --lr 0.001 \
                 --epochs 1000 \
                 --name hev_exp_1 \
                 --optimizer RAdam \
                 --input_w 512 \
                 --input_h 512 \
                 -b 16 \
                 --num_workers 36 \
                 --class_name hev,tumor \
                 --show_arch False \
                 --mix_precision True \
                 --encoder timm-resnest14d \
                 --encoder_weight histology \
                 --arch unetplusplus \
                 --mode multilabel \
                 --tf_log_path tblog_dir/hevi_seg/hev_exp_1_log \
                 --loss tversky \
                 --loss_weight 1.0 \
                 --scheduler CosineAnnealingWarmupRestarts \
                 --act sigmoid \
                 --cosine_cycle_steps 50 \
                 --cosine_cycle_warmup 5 \
                 --cosine_cycle_gamma 0.95 \
                 --early_stopping 50