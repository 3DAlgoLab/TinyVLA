# regular 
python vsft.py \
    --model_name_or_path="llava-hf/llava-1.5-7b-hf" \
    --report_to="wandb" \
    --learning_rate=1.4e-5 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --output_dir="data/vsft-llava-1.5-7b-hf" \
    --logging_steps=5 \
    --num_train_epochs=1 \
    --gradient_checkpointing \
    --remove_unused_columns=False \
    --torch_dtype=float16 \
    --fp16=True \
    --dataset_name=HuggingFaceH4/llava-instruct-mix-vsft \
