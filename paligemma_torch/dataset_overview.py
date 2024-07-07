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


ds2 = load_dataset(
    "json", data_files="/data/number-ops-1/dataset/_annotations.train.jsonl"
)
ic(ds2)

ic(ds2["train"][0])


ds3 = Dataset.from_json("/data/number-ops-1/dataset/_annotations.train.jsonl")
ic(ds3, len(ds3))
ic(ds3[0])
