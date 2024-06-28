import os
from jax import random
from icecream import ic
import jax

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

ic(jax.devices())
rng_key = random.PRNGKey(123)

vs = random.normal(rng_key, shape=(20_000_000, 3))
v1s = vs[:10_000_000, :].T
v2s = vs[10_000_000:, :].T

ic(v1s.shape)
ic(v2s.shape)
