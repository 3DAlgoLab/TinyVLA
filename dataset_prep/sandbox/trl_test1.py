import logging
import os
from contextlib import nullcontext

import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    TrainingArguments,
)
from trl import SFTTrainer
from trl.commands.cli_utils import SftScriptArguments, TrlParser

LLAVA_CHAT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""


class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["images"][0])

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch


if __name__ == "__main__":
    parser = TrlParser((SftScriptArguments, TrainingArguments))
    args, training_args = parser.parse_args_and_config()

    model_id = "llava-hf/llava-1.5-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer = tokenizer

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16
    )

    raw_datasets = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft")
    train_dataset = raw_datasets["train"]  # type: ignore
    eval_dataset = raw_datasets["test"]  # type: ignore

    data_collector = LLavaDataCollator(processor)
    trainer = SFTTrainer(
        model=model,  # type: ignore
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        dataset_text_field="text",
        tokenizer=tokenizer,
        data_collator=data_collector,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer.train()

    print(training_args.output_dir)
    trainer.save_model(training_args.output_dir)
