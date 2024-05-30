from typing import Any, Dict, Union, NamedTuple

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import rlds
from icecream import ic

dataset_name = "d4rl_mujoco_walker2d"
num_episodes_to_load = 10


dataset = tfds.load(dataset_name, split=f"train[:{num_episodes_to_load}]")
ic(dataset.element_spec)


shortened_dataset = dataset.skip(1).take(5)
