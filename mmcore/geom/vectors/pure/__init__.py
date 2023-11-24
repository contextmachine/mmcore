import functools
import typing
from operator import add, sub

import math

try:
    import numpy as np
    from numpy import ndarray as _ndarray
except Exception as err:
    _ndarray=list


def isvector(obj):
    return isinstance(obj, (tuple, list, _ndarray))
def dot(u, v):
    """
    3D dot product
    @param u:
    @param v:
    @return: float
    """
    res = 0
    for x, y in zip(u, v):
        res += (x * y)
    return res


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


def padd(self, other):
    return list(map(lambda x: add(*x), zip(list(self), list(other))))


def psub(self, other):
    return list(map(lambda x: sub(*x), zip(list(self), list(other))))


def pmul(self, other: float):
    return list(map(lambda x: x * other, list(self)))


def ptruediv(self, other: float):
    return list(map(lambda x: x / other, list(self)))


def pmatmul(a, b):
    return [[dot(col, row) for col in b] for row in a]




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


def punit(vec: typing.Union[V2, V3, list, tuple]) -> typing.Union[V2, V3, tuple]:
    """pure unit implementation"""
    if isvector(vec):
        if len(vec) == 2:

            nv = norm(vec)
            return vec[0] / nv, vec[1] / nv
        else:

            nv = norm(vec)
            return vec[0] / nv, vec[1] / nv, vec[2] / nv


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
