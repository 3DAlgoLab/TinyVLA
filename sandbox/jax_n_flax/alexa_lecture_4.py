import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax.training import train_state
import haiku as hk
import optax
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import functools
from typing import Any, Callable, Sequence
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic


def test_nn1():
    model = nn.Dense(features=5)
    ic(nn.Dense.__bases__)

    seed = 23
    key1, key2 = random.split(random.PRNGKey(seed))
    x = random.normal(key1, (10,))

    y, params = model.init_with_output(key2, x)
    ic(y)

    try:
        y = model(x)
    except Exception as e:
        ic(e)


def test_haiku_vs_flax():
    model = hk.transform(lambda x: hk.Linear(5)(x))
    seed = 23
    key1, key2 = random.split(random.PRNGKey(seed))
    x = random.normal(key1, (10,))

    params = model.init(key2, x)
    out = model.apply(params, None, x)
    ic(out)

    ic(hk.Linear.__bases__)


if __name__ == "__main__":
    test_nn1()
    test_haiku_vs_flax()

    print("Done.", flush=True)
