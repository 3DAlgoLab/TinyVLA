import tensorflow as tf
from icecream import ic

ds1 = tf.data.Dataset.random(seed=4).take(10)
ic(list(ds1.as_numpy_iterator()))

# Shuffle
dataset = tf.data.Dataset.range(10)
dataset = dataset.shuffle(3).repeat(2)
ic(list(dataset.as_numpy_iterator()))
