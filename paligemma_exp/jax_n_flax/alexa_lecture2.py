import jax
from icecream import ic
import jax.numpy as jnp
import numpy as np

from jax import grad, jit, vmap, pmap
from jax import random, make_jaxpr
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Tuple, NamedTuple
import functools


class Counter:
    def __init__(self):
        self.n = 0

    def count(self):
        self.n += 1
        return self.n

    def reset(self):
        self.n = 0


def test_count():
    counter = Counter()
    for _ in range(3):
        ic(counter.count())

    counter.reset()
    fast_count = jax.jit(counter.count)
    for _ in range(3):
        ic(fast_count())

    print(make_jaxpr(counter.count)())


CounterState = int


class Counter2:
    def count(self, n) -> Tuple[int, CounterState]:
        return n + 1, n + 1

    def reset(self):
        return 0


def test_count2():
    counter = Counter2()
    state = counter.reset()
    for _ in range(3):
        value, state = counter.count(state)
        print(value)

    print("Jitted version")

    state = counter.reset()
    fast_count = jax.jit(counter.count)
    for _ in range(3):
        value, state = fast_count(state)
        print(value)


def test_pytree1():
    # A contrived example for pedagogical purposes
    # (if your mind needs to attach some semantics to parse this - treat it as model params)
    pytree_example = [
        [1, "a", object()],
        (1, (2, 3), ()),
        [1, {"k1": 2, "k2": (3, 4)}, 5],
        {"a": 2, "b": (2, 3)},
        jnp.array([1, 2, 3]),
    ]

    # Let's see how many leaves they have:
    for pytree in pytree_example:
        leaves = jax.tree_leaves(pytree)  # handy little function
        print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")


if __name__ == "__main__":
    ic(jax.__version__)
    ic(jax.devices())

    # test_count2()
    test_pytree1()
