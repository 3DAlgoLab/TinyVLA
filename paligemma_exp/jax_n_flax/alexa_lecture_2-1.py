# simple model in parallel

import jax
import jax.numpy as jnp
from collections import namedtuple
from typing import Tuple, NamedTuple
import functools
import numpy as np
from matplotlib import pyplot as plt
from icecream import ic
import os

# TPU emulation
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


class Params(NamedTuple):
    weight: jnp.ndarray
    bias: jnp.ndarray


lr = 0.005


def init_model(rng):
    weights_key, bias_key = jax.random.split(rng)
    weight = jax.random.normal(weights_key, ())
    bias = jax.random.normal(bias_key, ())
    return Params(weight, bias)


def forward(params, xs):
    return params.weight * xs + params.bias


def loss_fn(params, xs, ys):
    return jnp.mean((forward(params, xs) - ys) ** 2)


@functools.partial(jax.pmap, axis_name="batch")
def update(params, xs, ys):
    loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)

    # combine the gradient across all devices (by taking their mean)
    grads = jax.lax.pmean(grads, axis_name="batch")

    # Also combine the loss.not necessary for the optimization, but it's useful for logging.
    loss = jax.lax.pmean(loss, axis_name="batch")

    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)

    return new_params, loss


def generate_data():
    true_w, true_b = 2, -1
    xs = np.random.normal(size=(128, 1))
    noise = 0.5 * np.random.normal(size=(128, 1))
    ys = xs * true_w + true_b + noise

    plt.scatter(xs, ys)
    plt.savefig("data.png")
    return xs, ys


def reshape_for_pmap(data, n_devices):
    return data.reshape(n_devices, data.shape[0] // n_devices, *data.shape[1:])


if __name__ == "__main__":
    n_devices = jax.local_device_count()
    xs, ys = generate_data()
    ic(xs.shape, ys.shape)

    params = init_model(jax.random.PRNGKey(0))
    replicated_params = jax.tree.map(lambda x: jnp.array([x] * n_devices), params)
    ic(replicated_params)

    x_parallel = reshape_for_pmap(xs, n_devices)
    y_parallel = reshape_for_pmap(ys, n_devices)

    ic(x_parallel.shape, y_parallel.shape)

    num_epochs = 2_000

    for epoch in range(num_epochs):
        replicated_params, loss = update(replicated_params, x_parallel, y_parallel)
        if epoch % 100 == 0:
            ic(epoch, loss)

    ic(replicated_params)
    # Like the loss, the leaves of params have an extra leading dimension,
    # so we take the params from the first device.
    params = jax.device_get(jax.tree.map(lambda x: x[0], replicated_params))
    ic(params)

    plt.scatter(xs, ys)
    plt.plot(xs, forward(params, xs), label="Model Prediction")
    plt.legend()
    plt.savefig("pmap_training.png")
