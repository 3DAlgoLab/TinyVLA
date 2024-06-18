#%%
import os
from icecream import ic
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import PositionalSharding
from jax.debug import visualize_array_sharding
from jax.experimental import mesh_utils

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
ic(jax.devices())


def dot(v1, v2):
    return jnp.vdot(v1, v2)


rng_key = random.PRNGKey(0)


vs = random.normal(rng_key, shape=(2_000_000, 100))

v1s = vs[:1_000_000, :]
v2s = vs[1_000_000:, :]

ic(v1s.shape, v2s.shape)
#%%

visualize_array_sharding(v1s)
ic()
visualize_array_sharding(v2s)

shading = PositionalSharding(mesh_utils.create_device_mesh((2, 1)))
ic(shading)

v1sp = jax.device_put(v1s, shading)
v2sp = jax.device_put(v2s, shading)

ic(type(v1sp))
ic(v1sp.shape)
visualize_array_sharding(v1sp)
d = jax.vmap(dot)(v1sp, v2sp)
ic(d.shape)
visualize_array_sharding(d)
#%%

%timeit jax.vmap(dot)(v1sp, v2sp).block_until_ready()
%timeit jax.vmap(dot)(v1s, v2s).block_until_ready()
 