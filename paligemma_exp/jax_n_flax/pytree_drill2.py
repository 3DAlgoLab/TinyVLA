import jax
import numpy as np
import jax.numpy as jnp
import collections
from icecream import ic

Point = collections.namedtuple("Point", ["x", "y"])

example_pytree = [
    {"a": [1, 2, 3], "b": jnp.array([1, 2, 3]), "c": np.array([1, 2, 3])},
    [42, [44, 46], None],
    31337,
    (50, (60, 70)),
    Point(640, 480),
    collections.OrderedDict([("a", 100), ("b", 200)]),
    "some string",
]

ic(jax.tree_util.tree_leaves(example_pytree))

""" output: 
    ic| jax.tree_util.tree_leaves(example_pytree): [1,
                                                2,
                                                3,
                                                Array([1, 2, 3], dtype=int32),
                                                array([1, 2, 3]),
                                                42,
                                                44,
                                                46,
                                                31337,
                                                50,
                                                60,
                                                70,
                                                640,
                                                480,
                                                100,
                                                200,
                                                'some string']
"""
