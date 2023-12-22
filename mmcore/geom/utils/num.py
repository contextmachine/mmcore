import numpy as np

from mmcore.func import vectorize


@vectorize(signature='(i)->()')
def min_index(vals):
    return np.where(vals == np.min(vals))[0]


@vectorize(signature='(i)->()')
def max_index(vals):
    return np.where(vals == np.max(vals))[0]


@vectorize(excluded=[2], signature='(i),()->(n)')
def similar_index(values, value, return_first=False):

    r = np.abs(np.array(values) - np.array(value))
    index = np.where(r == np.min(r))[0]
    if return_first:
        return np.atleast_1d(index[0])
    else:
        return index


def roll_to_index(value: np.ndarray, i: int):
    return value[np.roll(np.arange(len(value)), -i, axis=0)]
