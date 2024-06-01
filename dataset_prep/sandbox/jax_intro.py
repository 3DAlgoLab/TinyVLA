import jax
import jax.numpy as jnp
import numpy as np
import timeit
from jax import grad, jit, pmap, vmap, random
from icecream import ic

x = np.zeros(10)
y = jnp.zeros(10)


x = np.random.rand(1000, 1000)
y = jnp.array(x)

# why jax?

loop = 100

result_np = timeit.timeit(lambda: np.dot(x, x.T), number=loop)
result_jnp = timeit.timeit(lambda: jnp.dot(y, y.T).block_until_ready(), number=loop)


ic(f"NumPy time: {result_np/loop*1000} msec")
ic(f"JAX time: {result_jnp/loop*1000} msec")

# intro to grad


def f(x):
    return 3 * x**2 + 2 * x + 5


def f_prime(x):
    return 6 * x + 2


ic(grad(f)(1.0))
ic(f_prime(1.0))


def f2(x):
    for _ in range(10):
        x = 0.5 * x + 0.1 * jnp.sin(x)


# intro. to jit
g = jit(f2)

non_jit_result = timeit.timeit(lambda: f2(1.0), number=loop)
jit_result = timeit.timeit(lambda: g(1.0), number=loop)


ic(f"Non-jit time: {non_jit_result/loop*1000} msec")
ic(f"jit time: {jit_result/loop*1000} msec")


def f3(x):
    return jnp.sin(x) + x**2


ic(jax.devices())
ic(f3(np.arange(2)))
ic(pmap(f3)(np.arange(2)))


def f4(x):
    return jnp.square(x)


ic(f(jnp.arange(10)))
ic(vmap(f)(jnp.arange(10)))


# random number
key = random.PRNGKey(5)
ic(random.uniform(key))
