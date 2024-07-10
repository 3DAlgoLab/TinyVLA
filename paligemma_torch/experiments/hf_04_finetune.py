#%%
from datasets import load_dataset 


dataset = load_dataset("yelp_review_full")
dataset['train'][100]

# %%
from transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
# %%

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np 
import evaluate

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)

# training_args = TrainingArguments(output_dir='test_trainer')

metric = evaluate.load('accuracy')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

trainer = Trainer(model=model, args=training_args, 
                  train_dataset=small_train_dataset, 
                  eval_dataset=small_eval_dataset, 
                  compute_metrics=compute_metrics)


trainer.train() 

# %%
