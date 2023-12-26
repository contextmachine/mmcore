import numpy as np
from multipledispatch import dispatch

from mmcore.func import vectorize


@vectorize(signature='(i),(i),(i)->()')
def ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:

    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


@dispatch(np.ndarray, np.ndarray)
@vectorize(signature='(j, i),(j, i)->()')
def intersects_segments(ab: np.ndarray, cd: np.ndarray) -> bool:
    a, b = ab
    c, d = cd
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


@dispatch(object, object)
@vectorize(excluded=[0], signature='(j, i)->()')
def intersects_segments(ab, cd) -> bool:
    a, b = ab.start, ab.end
    c, d = cd.start, cd.end
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
