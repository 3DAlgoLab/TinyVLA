import jax.numpy as jnp
from jax import random
import jax

LAYER_SIZES = [200 * 200 * 3, 2048, 1024, 2]
PARAM_SCALE = 0.01


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key=random.PRNGKey(0), scale=0.01):
    "Randomly initialize all layer parameters."
    keys = random.split(key, len(sizes) - 1)
    return [
        random_layer_params(m, n, k, scale)
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


key = random.PRNGKey(42)
params = init_network_params(LAYER_SIZES, key, scale=PARAM_SCALE)


for i, layer in enumerate(params):
    w, b = layer
    print(i, w.shape, b.shape)

print("With jax.tree_util.tree_map")
shapes = jax.tree_util.tree_map(jnp.shape, params)
for i, shape in enumerate(shapes):
    print(i, shape)
