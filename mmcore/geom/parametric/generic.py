import numpy as np

from mmcore.func import vectorize


def parametrize(fun):
    def wrap(attrs):
        @vectorize(doc=fun.doc, signature='()->(i)')
        def inner(t):
            return fun(attrs, t)

        return inner

    return wrap


@parametrize
def line(x, t: float):
    start, direction = x
    return direction * t + start


@parametrize
def circle(r, t):
    return np.array([r * np.cos(t * 2 * np.pi), r * np.sin(t * 2 * np.pi), 0.0], dtype=float)
