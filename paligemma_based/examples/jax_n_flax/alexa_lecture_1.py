import jax
from jax import jit
from jax import numpy as jnp
from icecream import ic
import numpy as np


def init():
    pass


def test_nan_debug():
    # jax.config.update("jax_debug_nans", True)
    try:
        jnp.divide(0.0, 0.0)
    except Exception as e:
        ic(e)


g = 0.0


def impure_uses_global(x):
    return x + g


def test_impure_case():
    global g
    ic(jit(impure_uses_global)(4.0))
    g = 10
    ic(jit(impure_uses_global)(5.0))
    ic(jit(impure_uses_global)(jnp.array([4.0])))


def test_np_random():
    # NumPy - PRNG is stateful!
    seed = 123

    # Let's sample calling the same function twice
    ic(np.random.random())
    ic(np.random.random())

    np.random.seed(seed)

    rng_state = np.random.get_state()
    ic(rng_state)
    ic(rng_state[2:])

    _ = np.random.uniform()
    rng_state = np.random.get_state()
    ic(rng_state[2:])

    _ = np.random.uniform()
    rng_state = np.random.get_state()
    ic(rng_state[2:])


if __name__ == "__main__":
    test_np_random()
