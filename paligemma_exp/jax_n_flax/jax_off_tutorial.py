# %%
import jax
import jax.numpy as jnp


x = jnp.arange(5)
isinstance(x, jax.Array)
# %%
x.devices()
# %%
x.sharding


# %%
def selu(x, alpha=1.67, lambda_=1.05):
    return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


selu_jit = jax.jit(selu)
print(selu_jit(1.0))


# %%
@jax.jit
def selu(x, alpha=1.67, lambda_=1.05):
    return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


print(selu(1.0))
# %%
params = [1, 2, (jnp.arange(3), jnp.ones(2))]
jax.tree.structure(params)
# %%
jax.tree.leaves(params)
# %%

x = jnp.arange(5)
w = jnp.array([2.0, 3.0, 4.0])


# def convolve(x, w):
#     return jnp.convolve(x, w)


def convolve(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i - 1 : i + 2], w))
    return jnp.array(output)


convolve(x, w)
# %%
xs = jnp.stack([x, x])
ws = jnp.stack([w, w])


auto_batch_convolve = jax.vmap(convolve)
auto_batch_convolve(xs, ws)


# %%
