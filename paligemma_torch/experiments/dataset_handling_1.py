# %%
from datasets import load_dataset

dataset = load_dataset("lhoestq/demo1")

# %%
eli5 = load_dataset("/data/eli5", trust_remote_code=True)
len(eli5)
# %%
from datasets import load_dataset

dataset = load_dataset("glue", "mrpc", split="train")

# %%
dataset["label"][:10]

# %%
from datasets import ClassLabel, Value


new_features = dataset.features.copy()
new_features["label"] = ClassLabel(names=["negative", "positive"])
