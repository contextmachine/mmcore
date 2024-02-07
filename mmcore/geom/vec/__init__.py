"""
Using vec.speedups module
-------------------------

Benchmarks:

- func: dot
    speedups (0.0, 0.0033109188079833984)
    python (0.0, 1.171842098236084)
    speedups 353.9325988334414 x faster on (1_000_000,3) ndarray

- func: norm
    speedups (0.0, 0.0027971267700195312)
    python (0.0, 2.9127328395843506)
    speedups 1041.330293215138 x faster on (1_000_000,3) ndarray

- func: unit
    speedups (0.0, 0.00701904296875)
    python (0.0, 30.24796986579895)
    speedups 4309.415115489131 x faster on (1_000_000,3) ndarray

- func: cross
    speedups (0.0, 0.005728960037231445)
    python (0.0, 11.41274905204773)
    speedups 1992.1153189895542 x faster on (1_000_000,3) ndarray

"""
import os

import numpy as np

from mmcore.func import vectorize

DEBUG_MODE = os.getenv('DEBUG_MODE')


@vectorize(signature='(i),(i)->()')
def angle(a, b):
    """
    Calculate the angle between two vectors.

    :param a: First vector
    :type a: numpy array
    :param b: Second vector
    :type b: numpy array
    :return: Angle between the two vectors
    :rtype: float | numpy.ndarray[Any, numpy.dtype[float]]
    """
    if DEBUG_MODE:
        if (not 0.9999 <= np.linalg.norm(a) <= 1.0001) or (not 0.9999 <= np.linalg.norm(b) <= 1.0001):
            raise ValueError("Both input vectors must be normalized.")

    cos_angle = np.dot(a, b)
    if cos_angle < -1 or cos_angle > 1:
        raise ValueError("Dot product must be in the range [-1, 1]. Please check your input vectors.")

    return np.arccos(np.dot(a, b))


@vectorize(signature='(i),(i),(i)->()')
def angle3pt(a, b, c):
    ba = unit(a - b)
    bc = unit(c - b)
    return angle(ba, bc)


@vectorize(signature='(i),(i),(i)->()')
def dot3pt(a, b, c):
    ba = unit(a - b)
    bc = unit(c - b)
    return dot(ba, bc)


@vectorize(signature='(i)->()')
def norm(v):
    """
    .. function:: norm(v)

        This method calculates the norm of a vector.

        :param v: The input vector.
        :type v: numpy.ndarray[Any, numpy.dtype[float]]

        :return: The norm of the input vector.
        :rtype: float | numpy.ndarray[Any, numpy.dtype[float]]

    """
    return np.sqrt(np.sum(v ** 2))


@vectorize(signature='(i),(i)->()')
def dist(a, b):
    """
    :param a: a vector representing the first point
    :type a: numpy.ndarray[Any, numpy.dtype[float]]
    :param b: a vector representing the second point
    :type b: numpy.ndarray[Any, numpy.dtype[float]]
    :return: the distance between the two points
    :rtype: float | numpy.ndarray[Any, numpy.dtype[float]]

    """
    if DEBUG_MODE:
        if len(a) != len(b):
            raise ValueError("Input vectors must have the same dimensions.")

    return norm(a - b)


@vectorize(signature='(i)->(i)')
def unit(v: 'list|tuple|np.ndarray'):
    """
    :param v: The input vector
    :type v: numpy.ndarray[Any, numpy.dtype[float]]
    :return: The normalized vector
    :rtype: numpy.ndarray[Any, numpy.dtype[float]]
    """
    if DEBUG_MODE:
        if np.isclose(np.linalg.norm(v), 0):
            raise ValueError("Input vector must not be a zero vector.")

    return v / norm(v)


@vectorize(signature='(i),(i)->(i)')
def cross(a, b):
    """
    :param a: The first vector a
    :type a: numpy.ndarray[Any, numpy.dtype[float]] or array-like
    :param b: The second vector b
    :type b: numpy.ndarray[Any, numpy.dtype[float]] or array-like
    :return: The cross product of vectors a and b
    :rtype: numpy.ndarray[Any, numpy.dtype[float]] or array-like
    """
    return np.cross(a, b)


@vectorize(signature='(i),(i)->()')
def dot(a, b):
    """
    :param a: First input array.
    :type a: numpy.ndarray[Any, numpy.dtype[float]]
    :param b: Second input array.
    :type b: numpy.ndarray[Any, numpy.dtype[float]]
    :return: The dot product of `a` and `b`.
    :rtype: numpy.ndarray[Any, numpy.dtype[float]]
    """
    return np.dot(a, b)



@vectorize(signature='(i)->(i)')
def perp2d(v):
    """
    Calculate the perpendicular vector to the given 2D vector.

    :param vec: The input 2D vector.
    :type vec: numpy.ndarray[Any, numpy.dtype[float]]
    :return: The perpendicular vector to the input vector.
    :rtype: numpy.ndarray[Any, numpy.dtype[float]]
    """

    v2 = np.copy(v)
    v2[0] = -v[1]
    v2[1] = v[0]
    return v2


@vectorize(signature='(i),()->(i)')
def scale_vector(vec, length):
    """
    :param vec: The input vector to be scaled.
    :type vec: numpy.array
    :param length: The scaling factor.
    :type length: float or int
    :return: The scaled vector.
    :rtype: numpy.ndarray
    """
    return vec * length


@vectorize(excluded=[0], signature='(i)->(i)')
def add_multiply_vectors(a, b):
    """
        This method adds two vectors element-wise and returns the result.

        :param a: The first vector.
        :type a: numpy.ndarray[Any, numpy.dtype[float]]

        :param b: The second vector.
        :type b: numpy.ndarray[Any, numpy.dtype[float]]

        :return: The resulting vector after adding the two input vectors.
        :rtype: numpy.ndarray[Any, numpy.dtype[float]]
    """
    if DEBUG_MODE:
        if len(a) != len(b):
            raise ValueError("Both input vectors must have the same dimensions.")

    return a + b


@vectorize(signature='()->(i,i)')
def rotate_matrix(a):
    """
        This method adds two vectors element-wise and returns the result.

        :param a: The first vector.
        :type a: numpy.ndarray[Any, numpy.dtype[float]]

        :param b: The second vector.
        :type b: numpy.ndarray[Any, numpy.dtype[float]]

        :return: The resulting vector after adding the two input vectors.
        :rtype: numpy.ndarray[Any, numpy.dtype[float]]
    """

    r = np.eye(3)
    r[:2, :2] = [np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]
    return r


# clamp( min, max ) {
#
# 		// assumes min < max, componentwise
#
# 		this.x = Math.max( min.x, Math.min( max.x, this.x ) );
# 		this.y = Math.max( min.y, Math.min( max.y, this.y ) );
#
# 		return this;
#
# 	}


@vectorize(signature='(i),(i)->(i)')
def gram_schmidt(v1, v2):
    """
    Applies the Gram-Schmidt process to the given vectors v1 and v2.

    :param v1: The first vector.
    :type v1: numpy.ndarray
    :param v2: The second vector.
    :type v2: numpy.ndarray
    :return: The orthogonalized vector obtained by applying the Gram-Schmidt process.
    :rtype: numpy.ndarray
    """
    v1, v2 = unit(v1), unit(v2)
    return v2 - v1 * dot(v2, v1)


def clamp(self, min, max):
    # assumes min < max, componentwise

    return np.array([np.max([min[0], np.min([max[0], self[0]])]), np.max([min[1], np.min([max[1], self[1]])])])


def expand(self, pt):
    # assumes min < max, componentwise

    return np.array([np.min([self[0], pt[0]]), np.max([self[1], pt[1]])])
