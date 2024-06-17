# %%
import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax.training import train_state
import optax
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import functools
from typing import Any, Callable, Sequence
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

n_samples = 50
x_dim = 2
y_dim = 1
noise_amplitude = 0.1

seed = 123
# %%

key, w_key, b_key = random.split(random.PRNGKey(seed), 3)
W = random.normal(w_key, (x_dim, y_dim))  # weights
b = random.normal(b_key, (y_dim,))  # biases

true_params = freeze({"params": {"bias": b, "kernel": W}})

# Generate sample with additional noise
key, x_key, noise_key = random.split(key, 3)
xs = random.normal(x_key, (n_samples, x_dim))
ys = jnp.dot(xs, W) + b
ys += random.normal(noise_key, (n_samples, y_dim)) * noise_amplitude

ic(xs.shape, ys.shape)


# %% Visualizing data

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
assert (
    xs.shape[-1] == 2 and ys.shape[-1] == 1
)  # low dimensional data so that we can plot it
ax.scatter(xs[:, 0], xs[:, 1], zs=ys)


# %%
def make_mse_loss(xs, ys):

    def mse_loss(params):
        """Gives the value of the loss on the (xs, ys) dataset for the given model (params)."""

        # Define the squared loss for a single pair (x,y)
        def squared_error(x, y):
            pred = model.apply(params, x)
            # Inner because 'y' could have in general more than 1 dims
            return jnp.inner(y - pred, y - pred) / 2.0

        # Batched version via vmap
        return jnp.mean(jax.vmap(squared_error)(xs, ys), axis=0)

    return jax.jit(
        mse_loss
    )  # and finally we jit the result (mse_loss is a pure function)


mse_loss = make_mse_loss(xs, ys)
value_and_grad_fn = jax.value_and_grad(mse_loss)

# %%
model = nn.Dense(features=y_dim)
params = model.init(key, xs)

lr = 0.3
epochs = 20
log_period_epoch = 5

for epoch in range(epochs):
    loss, grads = value_and_grad_fn(params)
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
