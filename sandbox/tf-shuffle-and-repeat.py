# %%
import tensorflow as tf
from icecream import ic

# %%
arr = [1, 2, 3, 4, 5, 6]
dataset = tf.data.Dataset.from_tensor_slices(arr)


def print_ds(ds, max=20):
    for idx, elem in enumerate(dataset):
        print(f"step #{idx}: {elem.numpy()}")
        if idx > max:
            break


print_ds(dataset)
# %%
ic(dataset)
ic(len(dataset))
# %%
dataset = tf.data.Dataset.from_tensor_slices(arr)
dataset = dataset.shuffle(3)

print_ds(dataset)

# %%
dataset = tf.data.Dataset.from_tensor_slices(arr)
dataset = dataset.shuffle(3).repeat()

# %%
print_ds(dataset)

# %%
dataset = tf.data.Dataset.from_tensor_slices(arr)
dataset = dataset.shuffle(3).repeat().batch(3)

print_ds(dataset)


# %%
def test_last_element_not_in_first_batch():
    dataset = tf.data.Dataset.from_tensor_slices(arr)
    dataset = dataset.shuffle(3).repeat().batch(3)
    for idx, elem in enumerate(dataset):
        assert arr[-1] not in elem.numpy()
        return


for _ in range(100):
    test_last_element_not_in_first_batch()

test_last_element_not_in_first_batch()

# %%
