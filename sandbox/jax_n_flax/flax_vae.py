import flax
import jax
import optax
import orbax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from typing import Tuple, Callable
from math import sqrt

import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from icecream import ic

batch_size = 16
latent_dim = 32
kl_weight = 0.5
num_classes = 10
seed = 0xFFFF


key = jax.random.PRNGKey(seed)


train_dataset = MNIST("data", train=True, transform=T.ToTensor(), download=True)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)


class FeedForward(nn.Module):
    dimensions: Tuple[int, ...] = (256, 128, 64)
    activation_fn: Callable = nn.relu
    drop_last_activation: bool = False

    @nn.compact
    def __call__(self, x: ArrayLike) -> ArrayLike:
        for i, d in enumerate(self.dimensions):
            x = nn.Dense(d)(x)
            if i < len(self.dimensions) - 1 or self.drop_last_activation:
                x = self.activation_fn(x)

        return x


key, model_key = jax.random.split(key)
model = FeedForward(dimensions=(4, 2, 1), drop_last_activation=True)
# ic(model)

params = model.init(model_key, jnp.zeros((1, 8)))
# ic(params)

key, x_key = jax.random.split(key)
x = jax.random.normal(x_key, (1, 8))
y = model.apply(params, x)
# ic(y)


ic(jax.tree.map(lambda x: x.shape, params))


class VAE(nn.Module):
    encoder_dimensions: Tuple[int, ...] = (256, 128, 64)
    decoder_dimensions: Tuple[int, ...] = (128, 256, 784)
    latent_dim: int = 4
    activation_fn: Callable = nn.relu

    def setup(self):
        self.encoder = FeedForward(self.encoder_dimensions, self.activation_fn)
        self.pre_latent_proj = nn.Dense(self.latent_dim * 2)
        self.post_latent_proj = nn.Dense(self.encoder_dimensions[-1])
        self.class_proj = nn.Dense(self.encoder_dimensions[-1])
        self.decoder = FeedForward(
            self.decoder_dimensions, self.activation_fn, drop_last_activation=False
        )

    def reparam(
        self, mean: ArrayLike, logvar: ArrayLike, key: jax.random.PRNGKey
    ) -> ArrayLike:
        std = jnp.exp(logvar * 0.5)
        eps = jax.random.normal(key, mean.shape)
        return eps * std + mean

    def encode(self, x: ArrayLike):
        x = self.encoder(x)
        mean, logvar = jnp.split(self.pre_latent_proj(x), 2, axis=-1)
        return mean, logvar

    def decode(self, x: ArrayLike, c: ArrayLike):
        x = self.post_latent_proj(x)
        x = x + self.class_proj(c)
        x = self.decoder(x)
        return x

    def __call__(
        self, x: ArrayLike, c: ArrayLike, key: jax.random.PRNGKey
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        mean, logvar = self.encode(x)
        z = self.reparam(mean, logvar, key)
        y = self.decode(z, c)
        return y, mean, logvar


key = jax.random.PRNGKey(0x1234)
key, model_key = jax.random.split(key)
model = VAE(latent_dim=4)
ic(model)

key, call_key = jax.random.split(key)
params = model.init(
    model_key,
    jnp.zeros((batch_size, 784)),
    jnp.zeros((batch_size, num_classes)),
    call_key,
)

recon, mean, logvar = model.apply(
    params, jnp.zeros((batch_size, 784)), jnp.zeros((batch_size, num_classes)), call_key
)
recon.shape, mean.shape, logvar.shape
