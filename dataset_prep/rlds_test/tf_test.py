# %%
import tensorflow as tf

# print("Cheking TF GPUs")
# print(tf.config.list_physical_devices("GPU"))

dataset = tf.data.Dataset.from_tensor_slices([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
list(dataset.as_numpy_iterator())

# %%
import tensorflow_datasets as tfds

# mnist_data = tfds.load("mnist", split="train", with_info=True)
mnist_data[0]
# %%
mnist_data[1]

# %%
mnist_train = mnist_data["train"]  # type: ignore
mnist_test = mnist_data["test"]  # type: ignore
# %%
mnist_data.info
# %%
