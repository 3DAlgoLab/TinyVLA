# %% Review Dataset
import torch
from datasets import Image as dsImage
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)

import common

# %%
parent_folder = "/data/language_table_mini_train/"
train_file = parent_folder + "_annotations.jsonl"


def add_abs_path(example):
    example["image"] = parent_folder + example["image"]
    return example


ds_raw = load_dataset("json", data_files={"train": train_file})

ds_raw = ds_raw.map(add_abs_path)
ds_raw = ds_raw.cast_column("image", dsImage(mode="RGB"))
train_ds = ds_raw["train"]

# %%

model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)
image_token = processor.tokenizer.convert_tokens_to_ids("<image>")


def collate_fn(examples):
    texts = [example["prefix"] for example in examples]
    labels = [example["suffix"] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]

    tokens = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="longest",
        tokenize_newline_separately=False,
    )
    return tokens


# %%
# Lora Configuration
lora_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_type=torch.bfloat16
)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %%
run_name = common.generate_timestamp()

args = TrainingArguments(
    num_train_epochs=5,
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=50,  # 100 -> 25
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=500,  # 1_000 -> 500
    push_to_hub=True,
    save_total_limit=1,
    output_dir="outputs/language_table_mini",
    bf16=True,
    report_to=["wandb"],  # tensorboard -> wandb
    dataloader_pin_memory=False,
    run_name=run_name,  # added newly for wandb
)


trainer = Trainer(
    model=model, train_dataset=train_ds, data_collator=collate_fn, args=args
)

trainer.train()

# %%
# register model card
trainer.push_to_hub()
# %%
