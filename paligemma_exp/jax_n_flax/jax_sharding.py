import jax
from icecream import ic
import jax.numpy as jnp
from jax.experimental import mesh_utils
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

arr = jnp.arange(32.0).reshape(4, 8)
ic(arr.devices())

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
ic(result.sharding == arr_sharded.sharding)


@jax.jit
def f_contract_2(x):
    out = x.sum(axis=0)
    devices = mesh_utils.create_device_mesh(8)
    mesh = jax.sharding.Mesh(devices, "x")
    sharding = jax.sharding.NamedSharding(mesh, P("x"))
    return jax.lax.with_sharding_constraint(out, sharding)


result = f_contract_2(arr_sharded)
jax.debug.visualize_array_sharding(result)
ic(result)
