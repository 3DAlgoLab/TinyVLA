import tensorflow as tf


arr = [1, 2, 3, 4, 5, 6]

dataset = tf.data.Dataset.from_tensor_slices(arr)

for idx, elem in enumerate(dataset):
    print(f"Element {idx}: {elem.numpy()}")

for elm in dataset:
    print(elm.numpy())
