export num_gpus=2
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./output/qqp1"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path output/qqp1/model \
--task_name qqp \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 8 \
--learning_rate 1e-4 \
--num_train_epochs 18 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 50 \
--logging_dir $output_dir/log \
--evaluation_strategy steps \
--eval_steps 1000 \
--save_strategy steps \
--save_steps 1000 \
--warmup_steps 10000 \
--cls_dropout 0.2 \
--apply_lora \
--lora_r 16 \
--lora_alpha 32 \
--seed 0 \
--weight_decay 0.01 \
--save_total_limit=1 \
--metric_for_best_model "eval_accuracy" \
--reg_loss_wgt 0.5 \
--masking_prob 0.1
