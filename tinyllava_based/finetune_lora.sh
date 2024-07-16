#!/bin/bash

# Assign the arguments to variables
DATA_PATH="/dev_repo/vlms/TinyLLaVABench/experiments/pokemon/pokemon_blip_captions.json"
IMAGE_PATH="/dev_repo/vlms/TinyLLaVABench/experiments/"
OUTPUT_DIR="/dev_repo/vlms/TinyLLaVABench/experiments/TinyLLaVA-3.1B-lora"

deepspeed ../tinyllava/train/train.py \
    --deepspeed ../scripts/tiny_llava/zero3.json \
    --lora_enable True --lora_r 32 --lora_alpha 64 \
    --model_name_or_path bczhou/TinyLLaVA-3.1B \
    --version phi \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_PATH --vision_tower bczhou/TinyLLaVA-3.1B-SigLIP \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --fp16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 15 \
    --lazy_preprocess True \
    --report_to wandb
