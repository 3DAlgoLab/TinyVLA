from icecream import ic
import jax
import jax.numpy as jnp

x = jnp.arange(5)
w = jnp.array([2.0, 3.0, 4.0])


def convolve(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i - 1 : i + 2], w))
    return jnp.array(output)


ic(convolve(x, w))

xs = jnp.stack([x, x])
ws = jnp.stack([w, w])


def manually_vectorized_convolve(xs, ws):
    output = []
    for i in range(1, xs.shape[-1] - 1):
        output.append(jnp.sum(xs[:, i - 1 : i + 2] * ws, axis=1))
    return jnp.stack(output, axis=1)


ic(manually_vectorized_convolve(xs, ws))
auto_batch_convolve = jax.vmap(convolve)
ic(auto_batch_convolve(xs, ws))
jitted_batch_convolve = jax.jit(auto_batch_convolve)
ic(jitted_batch_convolve(xs, ws))
