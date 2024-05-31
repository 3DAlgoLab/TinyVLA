import tensorflow as tf
import numpy as np
from icecream import ic

# dataset
np.set_printoptions(precision=4)

dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
ic(dataset)
ic(list(dataset.as_numpy_iterator()))


for elem in dataset:
    ic(elem.numpy())


d2 = dataset.map(lambda x: x + 1)
ic(list(d2.as_numpy_iterator()))
ic(d2.reduce(0, lambda state, value: state + value).numpy())


for z in d2:
    ic(z.numpy())
