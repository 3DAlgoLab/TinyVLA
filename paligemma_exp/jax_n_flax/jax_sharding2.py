import jax
from icecream import ic
import jax.numpy as jnp
from jax.experimental import mesh_utils
import os
from jax.experimental.shard_map import shard_map

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
P = jax.sharding.PartitionSpec
mesh = jax.sharding.Mesh(jax.devices(), "x")


@jax.jit
def f_elementwise(x):
    return 2 * jnp.sin(x) + 1


f_elementwise_sharded = shard_map(
    f_elementwise, mesh=mesh, in_specs=P("x"), out_specs=P("x")
)

arr = jnp.arange(32)
result = f_elementwise_sharded(arr)
ic(result)
jax.debug.visualize_array_sharding(result)

x = jnp.arange(32)
ic(x.shape)


def f1(x):
    ic(x.shape)
    return x * 2


y = shard_map(f1, mesh=mesh, in_specs=P("x"), out_specs=P("x"))(x)
ic(y)


def f2(x):
    return jnp.sum(x, keepdims=True)


ic(result := shard_map(f2, mesh=mesh, in_specs=P("x"), out_specs=P("x"))(x))
jax.debug.visualize_array_sharding(result)


def f3(x):
    sum_in_shard = x.sum()
    return jax.lax.psum(sum_in_shard, "x")


# ic()
result = shard_map(f3, mesh=mesh, in_specs=P("x"), out_specs=P())(x)
ic(result)
ic(result.sharding)

arr2 = jnp.array([1, 2, 3.0])
ic(arr2)
ic(arr2.sharding)
