#%%
import evaluate 
from datasets import load_dataset
from evaluate import evaluator

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
from transformers import pipeline
from icecream import ic

#%%
accuracy = evaluate.load('accuracy')
accuracy.description

ic(accuracy.features)
# %%
# Full Evaluation with pipeline or model

def test1():
    pipe = pipeline("text-classification", model="lvwerra/distilbert-imdb", device=0)
    data = load_dataset("imdb", split="test").shuffle().select(range(1000))
    metric = evaluate.load('accuracy')
    task_evaluator = evaluator("text-classification")
    results = task_evaluator.compute(model_or_pipeline=pipe, 
                                    data=data, 
                                    metric=metric, 
                                    label_mapping={"NEGATIVE":0, "POSITIVE":1})
    ic(results)
# %%

dataset = load_dataset('yelp_review_full')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=12)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(200))

#%%
metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(model=model, args=training_args, 
                  train_dataset=small_train_dataset, 
                  eval_dataset=small_eval_dataset, 
                  compute_metrics=compute_metrics)

trainer.train()

# %%
