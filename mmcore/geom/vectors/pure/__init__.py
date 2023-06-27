import functools
import math
import typing
from operator import add, sub


def dot(u, v):
    """
    3D dot product
    @param u:
    @param v:
    @return: float
    """
    if hasattr(u, "z") and hasattr(v, "z"):
        return u.x * v.x + u.y * v.y + u.z * v.z
    elif (not hasattr(u, "z")) and (not hasattr(v, "z")):
        return u.x * v.x + u.y * v.y
    else:
        raise TypeError(f"You can apply the dot product to vectors of the same dimension. \n\t{u}, {v}")


def norm(v):
    """
    norm is a length of  vector
    @param v:
    @return:
    """
    return math.sqrt(dot(v, v))


def dist(P, Q):
    """
    distance is a norm of difference
    @param P: vector
    @param Q: vector
    @return: float
    >>> from scipy.spatial.distance import euclidean
    >>> import numpy as np
    >>> p1, p2 = np.random.random((2,3))
    >>> np.allclose(d(p1,p2), euclidean(p1,p2) )
    True
    """
    return norm(P - Q)


class V2(typing.Iterable):
    """Pure implementation a vector 2D """

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def normalize(self):
        d = norm(self)
        return V2(self.x / d, self.y / d)

    def __iter__(self):
        return [self.x, self.y]

    def __getitem__(self, item):
        return [self.x, self.y][item]

    def __add__(self, other):
        return list(map(add, zip(list(self), list(other))))

    def __sub__(self, other):
        return list(map(sub, zip(list(self), list(other))))

    def __mul__(self, other: float):
        return list(map(lambda x: x * other, list(self)))

    def __truediv__(self, other: float):
        return list(map(lambda x: x / other, list(self)))

    def __matmul__(self, other: float):
        return list(map(lambda x: x / other, list(self)))

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x}, y={self.y})'


class V3(V2):
    """Pure implementation a vector 3D """

    def __init__(self, x, y, z):
        super().__init__(x, y)

        self.z = z

    def normalize(self):
        d = norm(self)
        return V3(self.x / d, self.y / d, self.z / d)

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z})'


def punit(vec: typing.Union[V2, V3, list, tuple]) -> typing.Union[V2, V3]:
    """pure unit implementation"""
    if isinstance(vec, (list, tuple)):
        if len(vec) == 2:
            vec = V2(*vec)
            nv = norm(vec)
            return V2(vec.x / nv, vec.y / nv)
        else:
            vec = V3(*vec)
            nv = norm(vec)
            return V3(vec.x / nv, vec.y / nv, vec.z / nv)


def cross(a: typing.Union[V2, V3, list, tuple], b: typing.Union[V2, V3, list, tuple]) -> V3:
    """pure cross product implementation"""

    if not isinstance(a, (list, tuple)):
        a = list(a)
        if len(a) == 2:
            a = a + [0]
    if not isinstance(b, (list, tuple)):
        b = list(b)
        if len(b) == 2:
            b = b + [0]
    return V3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def pangle(a: typing.Union[V2, V3, list, tuple], b: typing.Union[V2, V3, list, tuple]) -> float:
    """pure angle implementation"""
    try:
        return math.acos(dot(punit(a), punit(b)))
    except RuntimeWarning:
        print('bad value', punit(a), punit(b), math.acos(dot(punit(a), punit(b))))


@functools.lru_cache(512)
def pcentroid(*points):
    vl = [0, 0, 0]
    for point in points:
        vl[0] += point[0]
        vl[1] += point[1]
        vl[2] += point[2]
    return [vl[0] / len(points), vl[1] / len(points), vl[2] / len(points)]
