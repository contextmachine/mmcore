from enum import IntEnum

import numpy as np


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def split_by_shapes(arr, target_shapes):
    s = np.sum(target_shapes[:-1])
    np.vsplit(np.arange(s * 3).reshape((s, 3)), np.cumsum(np.array([np.array(a).shape[0] for a in arr], int)[:-1])
              )


def insert_in_tuple(tpl, i, val):
    return *tpl[:i], val, *tpl[i:]

def add_dim(arr, val):
    first, *other = arr.shape
    if isinstance(val, int):
        val = (val,)
    arrnew = arr.reshape((first // val[-1], val[-1], *other))
    if len(val) == 1:

        return arrnew
    else:
        return add_dim(arrnew, val[:-1])


def insert_after_in_shape(tpl, i, val):
    if i < 0:
        i = len(tpl) + i

    return *tpl[:i], tpl[i] // val, val, *tpl[i + 1:]


def insert_before_in_shape(tpl, i, val):
    if i < 0:
        i = len(tpl) + i
    return *tpl[:i], val, tpl[i] // val, *tpl[i + 1:]


_shape_insertion_method = dict(before=insert_before_in_shape, after=insert_after_in_shape)


def split_dim(arr, index: int, val: int, insert_at='before'):
    '''

    :param arr:
    :type arr:
    :param index:
    :type index:
    :param val:
    :type val: int
    :param insert_at: 'before' or 'after'
    :type insert_at: str
    :return:
    :rtype:
    '''

    method = _shape_insertion_method[insert_at]

    return arr.reshape(method(arr.shape, index, val))
def remove_dim(arr, count=1):

    return arr.reshape((np.prod(arr.shape[:count + 1]), *arr.shape[count + 1:]))


def split_by_parts(arr, parts):
    ixs = np.cumsum(parts)

    if parts[-1] > len(arr):
        parts = parts[:-1]
    return np.vsplit(arr, np.cumsum(parts))


# Дана 2d кривая в параметрическом виде, представленная в виде следующей функции python:
import numpy


def cubic_spline(control_points: np.ndarray[(4, 2), np.dtype[float]]
                 ):
    """
    >>> spline=cubic_spline(np.array([(-313.8, 56.1), (-469.2, 870.1), (549.4, 855.2), (527.9, -230.7)]))
    >>> spline(0.5)
    array([ 56.8375, 625.1625])
    >>> spline(np.linspace(0,1,10))
    array([[-313.8       ,   56.1       ],
           [-325.15569273,  296.40123457],
           [-267.77146776,  473.30987654],
           [-159.87037037,  584.83333333],
           [ -19.67544582,  628.97901235],
           [ 134.59026063,  603.75432099],
           [ 284.7037037 ,  507.16666667],
           [ 412.44183813,  337.22345679],
           [ 499.58161866,   91.93209877],
           [ 527.9       , -230.7       ]])


    :param control_points:
    :type control_points:
    :return:
    :rtype:
    """
    p0, c0, c1, p1 = np.array(control_points)

    def inner(t):
        return ((p0 * ((1 - t) ** 3)
                 + 3 * c0 * t * ((1 - t) ** 2)
                 + 3 * c1 * (t ** 2) * (1 - t))
                + p1 * (t ** 3))

    return np.vectorize(inner, signature='()->(i)')
