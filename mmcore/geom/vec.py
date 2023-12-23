
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
def perp2d(vec):
    """
    Calculate the perpendicular vector to the given 2D vector.

    :param vec: The input 2D vector.
    :type vec: numpy.ndarray[Any, numpy.dtype[float]]
    :return: The perpendicular vector to the input vector.
    :rtype: numpy.ndarray[Any, numpy.dtype[float]]
    """
    if DEBUG_MODE:
        if len(vec) != 2:
            raise ValueError("Input vector should have 2 dimensions.")

    v2 = np.copy(vec)
    v2[0] = -vec[1]
    v2[1] = vec[0]
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


@vectorize(excluded=[0], signature='()->(i,i)')
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
