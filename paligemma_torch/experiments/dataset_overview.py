# %%
from datasets import Dataset, load_dataset
from icecream import ic
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# dataset = load_dataset("rotten_tomatoes", split="train")

# ic(dataset[0])

# result = tokenizer(dataset[0]["text"])
# ic(result)


# ds1 = load_dataset("json", data_files="sample_data.jsonl")
# ic(ds1)


# ds2 = load_dataset(
#     "json", data_files="/data/number-ops-1/dataset/_annotations.train.jsonl"
# )
# ic(ds2)

# ic(ds2["train"][0])


# ds3 = Dataset.from_json("/data/number-ops-1/dataset/_annotations.train.jsonl")
# ic(ds3, len(ds3))
# ic(ds3[0])


# ds4 = load_dataset("imagefolder", data_dir="/data/PokemonBLIPCaptions/")
# ic(ds4)
# # ic(ds4["train"][0])
# # ic(ds4["train"][1])

# # %%
# ds4["train"][0]
# # %%
# ds4["train"].column_names

# %%
ds5 = load_dataset("imagefolder", data_dir="/data/number-ops-1/dataset/")

# %%

from datasets import load_dataset

ds = load_dataset("HuggingFaceM4/VQAv2", split="train[:10%]")

# %%
from datasets import Image

ds_numbers = load_dataset(
    "json",
    data_files=["/data/number-ops-1/dataset/_annotations.train.jsonl"],
    split="train",
)


# def add_prefix(example):
#     example["text"] = "Review: " + example["text"]
#     return example

#   >>> ds = ds.map(add_prefix)

# change path relatives to abs.
parent_folder = "/data/number-ops-1/dataset/"
ds_numbers[0]


# %%
def add_abs_path(example):
    example["image"]


ds_numbers = ds_numbers.map(lambda sample: sample)


# %%
