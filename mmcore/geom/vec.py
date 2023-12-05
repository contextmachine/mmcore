import numpy as np

from mmcore.func import vectorize


@vectorize(signature='(i)->()')
def norm(v):
    return np.sqrt(np.sum(v ** 2))


@vectorize(signature='(i),(i)->()')
def dist(a, b):
    return norm(a - b)


@vectorize(signature='(i)->(i)')
def unit(v):
    return v / norm(v)


@vectorize(signature='(i),(i)->(i)')
def cross(a, b):
    return np.cross(a, b)


@vectorize(signature='(i),(i)->()')
def dot(a, b):
    return np.dot(a, b)


@vectorize(signature='(i),(i)->()')
def angle(a, b):
    return np.arccos(np.dot(a, b))
@vectorize(signature='(i)->(i)')
def perp2d(vec):
    v2 = np.copy(vec)
    v2[0] = -vec[1]
    v2[1] = vec[0]
    return v2


@vectorize(signature='(i),()->(i)')
def scale_vector(vec, length):
    return vec * length


@vectorize(excluded=[0], signature='(i)->(i)')
def add_multiply_vectors(a, b):
    return a + b
