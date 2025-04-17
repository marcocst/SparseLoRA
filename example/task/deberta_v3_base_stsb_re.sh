#!/bin/bash

export num_gpus=2
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./output/de_stsb1"

# 定义种子和学习率数组
seeds=(0 21)
learning_rates=(2e-4 3e-4)
lora_r_a=16
lora_alpha_a=32

# 循环调用
for seed in "${seeds[@]}"; do
    for lr in "${learning_rates[@]}"; do
        echo "Running with seed: $seed and learning rate: $lr"
        output_file="$output_dir/r_${lora_r_a}_a${lora_alpha_a}/seed_${seed}_lr_${lr}/model"
        log_file="$output_dir/r_${lora_r_a}_a${lora_alpha_a}/seed_${seed}_lr_${lr}/log"
        mkdir -p "$(dirname "$output_file")" # 确保目录存在
        mkdir -p "$(dirname "$log_file")" # 确保目录存在
        python -m torch.distributed.launch --nproc_per_node=$num_gpus \
        examples/text-classification/run_glue.py \
        --model_name_or_path microsoft/deberta-v3-base \
        --lora_path de_mnli_reg/model/checkpoint-114500/pytorch_model.bin \
        --task_name stsb \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_train_batch_size 16 \
        --learning_rate $lr \
        --num_train_epochs 12 \
        --output_dir $output_file \
        --overwrite_output_dir \
        --logging_steps 20 \
        --logging_dir $log_file \
        --evaluation_strategy steps \
        --eval_steps 50 \
        --save_total_limit=1 \
        --metric_for_best_model "eval_pearson" \
        --save_strategy steps \
        --save_steps 50 \
        --warmup_ratio 0.1 \
        --cls_dropout 0 \
        --apply_lora \
        --lora_r $lora_r_a \
        --lora_alpha $lora_alpha_a \
        --seed $seed \
        --weight_decay 0.01
    done
done
