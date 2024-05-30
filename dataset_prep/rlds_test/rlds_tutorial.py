# %%
from typing import Any, Dict, Union, NamedTuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import rlds
from icecream import ic

dataset_name = "d4rl_mujoco_walker2d"
num_episodes_to_load = 10

dataset = tfds.load(dataset_name, split=f"train[:{num_episodes_to_load}]")
ic(dataset.element_spec)

# %%
ds_shortened = dataset.skip(1).take(5)
ds_steps = dataset.flat_map(lambda x: x["steps"])
print(ds_steps.element_spec)
# %%
rlds.STEPS
# %%
dataset = tfds.load("d4rl_mujoco_halfcheetah/v0-medium")["train"].take(50)
# %%
import time


def benchmark(f, dataset):
    start_time = time.monotonic()
    start_cpu = time.process_time()
    result = f(dataset)
    wall_time = time.monotonic() - start_time
    cpu_time = time.process_time() - start_cpu
    print(f"Result: {result}, Execution time: {wall_time}, CPU: {cpu_time}")


# %%
episodes = 0
steps = 0
for episode in dataset:
    episodes += 1
    steps += int(episode[rlds.STEPS].cardinality())

ic(episodes, steps)


# %%
def compute_return2(episode_dataset):
    result = 0
    for episode in episode_dataset:
        for step in episode[rlds.STEPS]:
            result += step[rlds.REWARD]
    return result


benchmark(compute_return2, dataset)


# %%
def compute_return(episode_dataset):
    result = 0
    for episode in episode_dataset.prefetch(2):
        for step in episode[rlds.STEPS].prefetch(2):
            result += step[rlds.REWARD]
    return result


benchmark(compute_return, dataset)


# %%
def episode_return_sum(episode):
    return episode[rlds.STEPS].reduce(
        np.float32(0), lambda x, step: step[rlds.REWARD] + x
    )


def compute_return(episode_dataset):
    return episode_dataset.reduce(
        np.float32(0), lambda x, episode: episode_return_sum(episode) + x
    )


benchmark(compute_return, dataset)


# %%
def double_reward(step):
    step[rlds.REWARD] *= 2
    return step


double_reward_dataset = dataset.flat_map(lambda x: x[rlds.STEPS]).map(
    lambda step: double_reward(step)
)


def compute_return(step_dataset):
    return step_dataset.batch(100).reduce(
        np.float32(0), lambda x, step: tf.math.reduce_sum(step[rlds.REWARD]) + x
    )


benchmark(compute_return, double_reward_dataset)


# %%
def vectorized_double_reward(steps):
    return tf.vectorized_map(double_reward, steps)


double_reward_dataset = (
    dataset.flat_map(lambda x: x[rlds.STEPS])
    .batch(100)
    .map(vectorized_double_reward)
    .unbatch()
)

benchmark(compute_return, double_reward_dataset)

# %%
