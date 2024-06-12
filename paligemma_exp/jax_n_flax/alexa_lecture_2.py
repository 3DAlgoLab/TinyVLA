import jax
from icecream import ic
import jax.numpy as jnp
import numpy as np

from jax import grad, jit, vmap, pmap
from jax import random, make_jaxpr
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Tuple, NamedTuple
import os

# TPU emulation
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


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


# A contrived example for pedagogical purposes
# (if your mind needs to attach some semantics to parse this - treat it as model params)
pytree_example = [
    [1, "a", object()],
    (1, (2, 3), ()),
    [1, {"k1": 2, "k2": (3, 4)}, 5],
    {"a": 2, "b": (2, 3)},
    jnp.array([1, 2, 3]),
]


def test_pytree1():
    # Let's see how many leaves they have:
    for pytree in pytree_example:
        leaves = jax.tree_leaves(pytree)  # handy little function
        print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")


def test_pytree2():
    leaves = jax.tree.leaves(pytree_example)
    print(f"{repr(pytree_example):<45} has {len(leaves)} leaves: {leaves}")


list_of_lists = [{"a": 3}, [1, 2, 3], [1, 2], [1, 2, 3, 4]]


def test_pytree3():
    l2 = jax.tree.map(lambda x: x * 2, list_of_lists)
    print(l2)


def test_pytree_multimap():
    l3 = jax.tree.map(lambda x, y: x * y, list_of_lists, list_of_lists)
    print(l3)


def init_mlp_params(layer_widths):
    params = []
    key = random.PRNGKey(123)

    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        key, subkey = random.split(key)
        w = random.normal(subkey, (n_in, n_out)) * jnp.sqrt(2 / n_in)
        b = jnp.ones((n_out,))
        params.append(dict(weights=w, biases=b))

    return params


def forward(params, x):
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu(jnp.dot(x, layer["weights"]) + layer["biases"])

    return jnp.dot(x, last["weights"]) + last["biases"]


def loss_fn(params, x, y):
    y_pred = forward(params, x)
    return jnp.mean((y - y_pred) ** 2)


@jit
def update(params, x, y, lr=0.01):
    grads = jax.grad(loss_fn)(params, x, y)
    # SGD update
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)


def test_init_mlp_params():
    params = init_mlp_params([1, 128, 128, 1])
    params_shape = jax.tree.map(lambda x: x.shape, params)
    print(params_shape)


def test_toy_mlp_training():
    params = init_mlp_params([1, 128, 128, 1])
    key = random.PRNGKey(124)
    xs = random.normal(key, (128, 1))
    ys = jnp.sin(xs)

    num_epochs = 2_000
    for i in range(num_epochs):
        params = update(params, xs, ys)
        if i % 100 == 0:
            print(f"Epoch: {i}, Loss: {loss_fn(params, xs, ys)}")

    plt.scatter(xs, ys)
    plt.scatter(xs, forward(params, xs), label="Model Prediction")
    plt.legend()
    plt.savefig("toy_mlp_training.png")


class MyContainer:
    def __init__(self, name, a, b, c):
        self.name = name
        self.a = a
        self.b = b
        self.c = c


def test_custom_tree():
    example_pytree = [MyContainer("alice", 1, 2, 3), MyContainer("bob", 4, 5, 6)]
    leaves = jax.tree.leaves(example_pytree)
    print(f"{repr(example_pytree):<45}\n has {len(leaves)} leaves:\n {leaves}")

    # Wrong code.
    # print(
    #     jax.tree.map(lambda x: x + 1, example_pytree)
    # )  # this will not work :/ it'd be nice if it did

    flattened = jax.tree.flatten(example_pytree)
    print(flattened)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"PointObject(x={self.x}, y={self.y})"


def point_flatten(v):
    children = (v.x, v.y)
    aux_data = None
    return (children, aux_data)


def point_unflatten(aux_data, children):
    return Point(*children)


def test_custom_tree2():
    jax.tree_util.register_pytree_node(Point, point_flatten, point_unflatten)
    test_tree = jax.tree.map(lambda x: x + 1, [Point(1, 2), Point(3, 4)])
    ic(test_tree)


def test_another_tree():
    zeros_tree = [jnp.zeros((2, 3)), jnp.zeros((3, 4))]
    ic(zeros_tree)

    shapes = jax.tree.map(lambda x: jnp.array(x.shape), zeros_tree)
    ic(shapes)

    ones_tree = jax.tree.map(jnp.ones, shapes)
    ic(ones_tree)


x = np.arange(5.0)
w = np.array(
    [
        2,
        3,
        4,
    ],
    dtype=np.float32,
)


def test_pmap1():
    # convolutional operation
    def convolve(w, x):
        output = []
        for i in range(1, len(x) - 1):
            output.append(jnp.dot(x[i - 1 : i + 2], w))

        return jnp.array(output)

    result = convolve(w, x)
    ic(result)


def test_pmap2():
    n_devices = jax.local_device_count()
    ic(n_devices)
    xs = np.arange(5 * n_devices).reshape(-1, 5)
    ic(xs)
    ws = np.stack([w] * n_devices)

    ic(xs.shape, ws.shape)


if __name__ == "__main__":
    ic(jax.__version__)
    ic(jax.devices())

    test_pmap2()

    # test_count2()
    # test_pytree_multimap()
    # test_init_mlp_params()
    # test_toy_mlp_training()
    # test_custom_tree()
    # test_custom_tree2()
    # test_another_tree()
