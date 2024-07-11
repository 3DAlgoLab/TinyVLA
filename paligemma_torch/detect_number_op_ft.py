# %% Review Dataset
import torch
from datasets import Image, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)

# %%
parent_folder = "/data/number-ops-1/dataset/"
test_file = parent_folder + "_annotations.test.jsonl"
train_file = parent_folder + "_annotations.train.jsonl"
valid_file = parent_folder + "_annotations.valid.jsonl"


def add_abs_path(example):
    example["image"] = parent_folder + example["image"]
    return example


ds_numbers = load_dataset(
    "json", data_files={"train": train_file, "test": test_file, "valid": valid_file}
)

ds_numbers = ds_numbers.map(add_abs_path)
ds_numbers = ds_numbers.cast_column("image", Image(mode="RGB"))
train_ds = ds_numbers["train"]

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

# NO LORA Model
# model = PaliGemmaForConditionalGeneration.from_pretrained(
#     model_id, torch_dtype=torch.bfloat16, device_map="auto"
# )

# for param in model.vision_tower.parameters():
#     param.requires_grad = False

# for param in model.multi_modal_projector.parameters():
#     param.requires_grad = False

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

args = TrainingArguments(
    num_train_epochs=2,
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=25,  # 100 -> 25
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=500,  # 1_000 -> 500
    push_to_hub=True,
    save_total_limit=1,
    output_dir="outputs/number_op_detection",
    bf16=True,
    report_to=["wandb"],  # tensorboard -> wandb
    dataloader_pin_memory=False,
    run_name="240711_1",  # added newly for wandb
)


trainer = Trainer(
    model=model, train_dataset=train_ds, data_collator=collate_fn, args=args
)

trainer.train()

# %%
