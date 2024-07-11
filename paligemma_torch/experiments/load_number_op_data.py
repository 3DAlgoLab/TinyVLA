# %%
from datasets import Dataset, Image, load_dataset
from icecream import ic

ds_numbers = load_dataset(
    "json",
    data_files=["/data/number-ops-1/dataset/_annotations.train.jsonl"],
    split="train",
)

# change path relatives to abs.
parent_folder = "/data/number-ops-1/dataset/"

ds_numbers[0]

# %%


# def add_prefix(example):
#     example["text"] = "Review: " + example["text"]
#     return example

#   >>> ds = ds.map(add_prefix)


def add_abs_path(example):
    example["image"] = parent_folder + example["image"]
    return example


ds1 = ds_numbers.map(add_abs_path)
ds1 = ds1.cast_column("image", Image(mode="RGB"))
ds1[0]
# %%
