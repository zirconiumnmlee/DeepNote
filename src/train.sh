#!/bin/bash
source ~/miniconda3/bin/activate deepnote
PYTHON_EXECUTABLE=$(which python)

CUDA_VISIBLE_DEVICES=0,1 torchrun  --nnodes=1 --nproc_per_node=2 --master_addr localhost --master_port 7428 --node_rank 0 train.py \
    --model_name_or_path Qwen/Qwen2.5-3b-instruct \
    --dataset_name /home/lihaoyv/Projects/DeepNote/data/dpo/processed/train.jsonl \
    --trust_remote_code \
    --max_length 1024 \
    --max_prompt_length 2000 \
    --output_dir ../save_model/qwen \
    --save_steps 500 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-7 \
    --logging_strategy steps \
    --logging_steps 50 \
    --logging_dir ../save_model/qwen \
    --bf16 True \
    --num_train_epochs 1 \
    --report_to "tensorboard" \
    --save_only_model \
    --gradient_checkpointing \
    --deepspeed ../config/ds_config_zero3.json
