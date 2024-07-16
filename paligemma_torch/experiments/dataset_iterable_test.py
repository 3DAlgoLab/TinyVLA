#%%
from datasets import load_dataset, Dataset, IterableDataset
from icecream import ic

#%%

imagenet = load_dataset('imagenet-1k', split='train', streaming=True)
for example in imagenet:
    print(example)
    break 
# %%
my_dataset = Dataset.from_dict({"col_1":[0, 1, 2, 3, 4, 5]})
print(my_dataset[0])
print(my_dataset[3])
# %%
def my_generator(n):
    for i in range(n):
        yield {"col_1":i}

my_iterable_dataset = IterableDataset.from_generator(my_generator, gen_kwargs={"n":10})
for example in my_iterable_dataset:
    print(example)
    break 


# ic(len(my_iterable_dataset)) Error IterableDataset has no len() 

# %%
TEST_FILE_1 = "/data/language_table_mini_train2/_annotations.jsonl"
data_files = {"train": [f"{TEST_FILE_1}"]}

my_dataset = load_dataset("json", data_files=data_files, split="train")
print(my_dataset[0])
# %%
my_iterable_dataset = load_dataset("json", data_files=data_files, split="train", streaming=True)
my_iterable_dataset = my_iterable_dataset.shuffle(seed=42, buffer_size=100)
for example in my_iterable_dataset:  # this reads the CSV file progressively as you iterate over the dataset
    print(example)
    break


# %%
my_iterable_dataset = load_dataset("json", data_files=data_files, split="train", streaming=True)
ic(my_iterable_dataset.n_shards)
# %%
def my_generator(n, sources):
    for source in sources:
        for example_id_for_current_source in range(n):
            yield {"example_id": f"{source}_{example_id_for_current_source}"}

gen_kwargs = {"n": 10, "sources": [f"path/to/data_{i}" for i in range(1024)]}
my_iterable_dataset = IterableDataset.from_generator(my_generator, gen_kwargs=gen_kwargs)
my_iterable_dataset.n_shards  # 1024
# %%
