export num_gpus=2
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./de_mrpc1"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--lora_path de_mnli_reg/model/checkpoint-114500/pytorch_model.bin \
--task_name mrpc \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 8 \
--learning_rate 2e-4 \
--num_train_epochs 20 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 20 \
--logging_dir $output_dir/log \
--evaluation_strategy steps \
--eval_steps 50 \
--save_strategy steps \
--save_steps 50 \
--warmup_ratio 0.1 \
--cls_dropout 0 \
--apply_lora \
--lora_r 16 \
--lora_alpha 32 \
--seed 0 \
--weight_decay 0.01 \
--reg_loss_wgt 0.5 \
--metric_for_best_model "eval_accuracy" \
--save_total_limit=2
