#
# https://huggingface.co/docs/transformers/tasks/language_modeling

# %%
from datasets import load_dataset
import tensorboard
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# %%
eli5 = load_dataset("eli5_category", split="train[:5000]")
eli5 = eli5.train_test_split(test_size=0.2)

eli5["train"][0]
# %%
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
eli5 = eli5.flatten()
eli5["train"][0]

# %%
eli5["train"][0]["answers.text"]


# %%
def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])


tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=10,
    remove_columns=eli5["train"].column_names,
)

# %%
tokenized_eli5["train"][0].keys()

# %%
block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=10)

# %%
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

# %%
training_args = TrainingArguments(
    output_dir="./outputs/test_awesome_eli5_clm-model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
    logging_steps=10,
    report_to="tensorboard",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()
# %%
import math

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# %%
trainer.push_to_hub()
# %%
