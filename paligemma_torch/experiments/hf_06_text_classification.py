# %%
# https://huggingface.co/docs/transformers/tasks/sequence_classification
import token

from datasets import load_dataset

imdb = load_dataset("imdb")
imdb["test"][0]
# %%
import referencing
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def preprocess_function(example):
    return tokenizer(example["text"], truncation=True)


tokenized_imdb = imdb.map(preprocess_function, batched=True)

# %%
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# %%
# Train
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {v: k for k, v in id2label.items()}
label2id
# %%
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir="tal_text_classification_model",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %%
trainer.train()
# %%
trainer.push_to_hub()

# %%
# inference # 1
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

from transformers import pipeline

trained_model_id = "3dalgo/tal_text_classification_model"
classifier = pipeline("sentiment-analysis", model=trained_model_id, device=0)

classifier(text)

import torch

# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(trained_model_id)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

model = AutoModelForSequenceClassification.from_pretrained(trained_model_id)
model = model.to("cuda")

with torch.no_grad():
    result = model(**inputs)

print(result)
logits = result.logits
predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

# %%
