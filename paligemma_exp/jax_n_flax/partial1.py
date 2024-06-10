from functools import partial
from icecream import ic
from numpy import tri


def multiply(x: int, y: int):
    return x * y


# double = partial(multiply, y=2)
# ic(double(4))


@partial(multiply, y=3)
def triple(fn):
    return fn(x)


print(triple(4))
