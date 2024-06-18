# %%
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable


class Model(nn.Module):
    dim: int
    activation_fn: Callable = nn.relu

    def setup(self):
        self.layer = nn.Dense(self.dim)

    def __call__(self, x):
        x = self.layer(x)
        return self.activation_fn(x)


class ModelCompact(nn.Module):
    dim: int
    activation_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dim)(x)
        return self.activation_fn(x)


key = jax.random.PRNGKey(123)
key, model_key = jax.random.split(key)

model = Model(dim=4)
params = model.init(model_key, jnp.zeros((1, 8)))
print(params)

# %%
import optax

optimizer = optax.sgd(learning_rate=1e-2)
optimizer
# %%
optimizer_state = optimizer.init(params)
optimizer_state
# %%
optimizer = optax.adam(learning_rate=1e-3)
optimizer_state = optimizer.init(params)
optimizer_state

# %%
