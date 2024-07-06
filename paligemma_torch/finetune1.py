# %%
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
)

# %%
device = "cuda"

ds = load_dataset("HuggingFaceM4/VQAv2", split="train")
cols_remove = ["question_type", "answers", "answer_type", "image_id", "question_id"]
ds = ds.remove_columns(cols_remove)
ds = ds.train_test_split(test_size=0.1)  # type: ignore
train_ds = ds["train"]
val_ds = ds["test"]

# %%
from transformers import PaliGemmaProcessor

model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)
image_token = processor.tokenizer.convert_tokens_to_ids("<image>")  # type: ignore


def collate_fn(examples):
    texts = ["answer " + example["question"] for example in examples]
    labels = [example["multiple_choice_answer"] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(
        text=texts, images=images, suffix=labels, return_tensors="pt", padding="longest"
    )  # type: ignore

    tokens = tokens.to(torch.bfloat16)
    return tokens


# %%
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
).to(device)


for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = True

# %%
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    num_train_epochs=2,
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    push_to_hub=False,
    save_total_limit=1,
    bf16=True,
    report_to=["wandb"],
    dataloader_pin_memory=False,
    output_dir="./out",
    run_name="paligemma-finetune-testing",
)

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    args=args,
)

trainer.train()

# %%
