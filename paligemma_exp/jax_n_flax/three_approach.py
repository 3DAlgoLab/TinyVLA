import jax
import jax.numpy as jnp
import numpy as np
from icecream import ic
import os
from functools import partial
from jax.experimental.shard_map import shard_map

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
ic(jax.devices())


@jax.jit
def layer(x, weights, bias):
    return jax.nn.sigmoid(x @ weights + bias)


rng = np.random.default_rng(0)
x = rng.normal(size=(32,))
weights = rng.normal(size=(32, 4))
bias = rng.normal(size=(4,))

l1 = layer(x, weights, bias)
ic(l1)
ic(l1.sharding)

P = jax.sharding.PartitionSpec
mesh = jax.sharding.Mesh(jax.devices(), "x")
sharding = jax.sharding.NamedSharding(mesh, P("x"))

x_sharded = jax.device_put(x, sharding)
weights_sharded = jax.device_put(weights, sharding)

l2 = layer(x_sharded, weights_sharded, bias)
ic(l2)
ic(l2.sharding)


@jax.jit
def layer_auto(x, weights, bias):
    x = jax.lax.with_sharding_constraint(x, sharding)
    weights = jax.lax.with_sharding_constraint(weights, sharding)
    return layer(x, weights, bias)


ic(layer_auto(x, weights, bias))


@jax.jit
@partial(
    shard_map, mesh=mesh, in_specs=(P("x"), P("x", None), P(None)), out_specs=P(None)
)
def layer_sharded(x, weights, bias):
    return jax.nn.sigmoid(jax.lax.psum(x @ weights, "x") + bias)


ic(x @ weights)
ic(layer_sharded(x, weights, bias))
