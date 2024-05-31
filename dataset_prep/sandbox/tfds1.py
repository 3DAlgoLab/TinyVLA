# %%
import tensorflow as tf
import pathlib
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)
# %%
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
dataset
# %%
for elem in dataset:
    print(elem.numpy())
# %%
it = iter(dataset)
print(next(it).numpy())
# %%
dataset.reduce(0, lambda state, value: state + value).numpy()
# %%
ds1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
ds1.element_spec
# %%
