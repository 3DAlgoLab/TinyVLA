import os
import jax
from icecream import ic
import jax.numpy as jnp
from jax import random

from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
ic(jax.devices())


def dot(v1, v2):
    return jnp.vdot(v1, v2)


r = dot(jnp.array([1, 1, 1.0]), jnp.array([1.0, 2.0, -1.0]))
ic(r)


rng_key = random.PRNGKey(42)

vs = random.normal(rng_key, shape=(20_000_000, 3))

v1s = vs[:10_000_000, :]
v2s = vs[10_000_000:, :]
ic(v1s.shape, v2s.shape)

arr = jnp.arange(32.0).reshape(4, 8)
ic(arr.devices())
ic(arr.sharding)
ic(jax.debug.visualize_array_sharding(arr))

P = jax.sharding.PartitionSpec
devices = mesh_utils.create_device_mesh((2, 4))
mesh = jax.sharding.Mesh(devices, ("x", "y"))
sharding = jax.sharding.NamedSharding(mesh, P("x", "y"))
ic(sharding)

arr_sharded = jax.device_put(arr, sharding)
ic(arr_sharded)
jax.debug.visualize_array_sharding(arr_sharded)


@jax.jit
def f_elementwise(x):
    return 2 * jnp.sin(x) + 1


result = f_elementwise(arr_sharded)
print("shardings match:", result.sharding == arr_sharded.sharding)
