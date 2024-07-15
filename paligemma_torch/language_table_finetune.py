# %% Review Dataset
import torch
from datasets import Image as dsImage
from datasets import load_metric
from peft import LoraConfig, get_peft_model
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)
import numpy as np
import common

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# %%
train_data_folder = "/data/language_table_mini_train2/"
val_data_folder = "/data/language_table_mini_valid2/"

train_ds = common.load_vla_data_from_folder(train_data_folder, split='train')
val_ds = common.load_vla_data_from_folder(val_data_folder, split='valid')


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
    r=32,
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
    num_train_epochs=10,
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-4,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=50,  # 100 -> 25
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=500,  # 1_000 -> 500
    push_to_hub=True,
    save_total_limit=1,
    output_dir="outputs/language_table_mini2",
    bf16=True,
    report_to=["wandb"],  # tensorboard -> wandb
    dataloader_pin_memory=False,
    metric_for_best_model="eval_loss",
    run_name=run_name,  # added newly for wandb 
    load_best_model_at_end=True
)


trainer = Trainer(
    model=model, train_dataset=train_ds, 
    eval_dataset=val_ds, 
    data_collator=collate_fn, args=args, 
    compute_metrics=compute_metrics
)

trainer.train()

# %%
# register model card
trainer.push_to_hub()
# %%
