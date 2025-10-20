#!/bin/bash
# Colab用訓練スクリプト
# A100 80GB GPU対応の最適化設定

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="./logs/OSCP"

accelerate launch train_lora.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --lora_rank=64 \
    --lmd=0.001 \
    --N=2 \
    --learning_rate=1e-4 \
    --loss_type=l2 \
    --adam_weight_decay=0.0 \
    --max_train_steps=625 \
    --max_train_samples=40000 \
    --dataloader_num_workers=0 \
    --checkpointing_steps=625 \
    --checkpoints_total_limit=10 \
    --train_batch_size=64 \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --lr_scheduler=constant_with_warmup \
    --report_to=tensorboard \
    --seed=3407 \
    --use_deeplake \
    --deeplake_subset=40000 \
    --attack_iter=10 \
    --eps_iter=0.003
