# %%
import jax
import jax.numpy as jnp
from icecream import ic
from jax import random


def dot(v1, v2):
    return jnp.dot(v1, v2)


ic(dot(jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))


rng_key = random.PRNGKey(42)
vs = random.normal(rng_key, shape=(20, 3))

v1s = vs[:10, :]
v2s = vs[10:, :]

ic(v1s)
ic(v2s)


# error, the result is not what we want
# ic(dot(v1s, v2s))
ic([dot(v1s[i], v2s[i]) for i in range(v1s.shape[0])])


def dot_vectorized(v1s, v2s):
    return jnp.einsum("ij, ij->i", v1s, v2s)


ic(dot_vectorized(v1s, v2s))

dot_vmapped = jax.vmap(dot)
ic(dot_vmapped(v1s, v2s))

ic("Scaled dot product")


def scaled_dot(v1, v2, koeff):
    return koeff * jnp.dot(v1, v2)


v1s_ = v1s
v2s_ = v2s.T
ic(v1s_.shape)
ic(v2s_.shape)

scaled_dot_batched = jax.vmap(scaled_dot, in_axes=(0, 1, None))
ic(scaled_dot_batched(v1s_, v2s_, 1.0))


def scale(v, koeff):
    return v * koeff


# %%
scale_batched = jax.vmap(scale, in_axes=(0, None), out_axes=(1))
result_scale_batched = scale_batched(v1s, 2.0)
ic(result_scale_batched.shape)

# %%
# Named axes arguments
scale_batched = jax.vmap(scale, in_axes=(0), out_axes=(1))
result_scale_batched = scale_batched(v1s, koeff=2.0)
ic(result_scale_batched.shape)

# %%
from functools import partial

scale2 = partial(scale, koeff=2.0)
scale_batched = jax.vmap(scale2, in_axes=(0), out_axes=(1))
ic(scale_batched(v1s).shape)

# %%
