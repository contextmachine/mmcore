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
