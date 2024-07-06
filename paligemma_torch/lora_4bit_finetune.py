# %%
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)

# %%

ds = load_dataset("HuggingFaceM4/VQAv2", split="train")
cols_remove = ["question_type", "answers", "answer_type", "image_id", "question_id"]
ds = ds.remove_columns(cols_remove)
ds = ds.train_test_split(test_size=0.1)
train_ds = ds["train"]
val_ds = ds["test"]

model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)

# device = "cuda"
image_token = processor.tokenizer.convert_tokens_to_ids("<image>")


def collate_fn(examples):
    texts = ["answer " + example["question"] for example in examples]
    labels = [example["multiple_choice_answer"] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(
        text=texts, images=images, suffix=labels, return_tensors="pt", padding="longest"
    )

    tokens = tokens.to(torch.bfloat16)
    return tokens


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

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


model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# %%

args = TrainingArguments(
    num_train_epochs=2,
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=32,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    push_to_hub=True,
    save_total_limit=1,
    bf16=True,
    report_to=["wandb"],
    dataloader_pin_memory=False,
    output_dir="outputs",
    run_name="qlora-finetune-test",
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
