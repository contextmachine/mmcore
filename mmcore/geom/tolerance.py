from enum import Enum

from mmcore.func import vectorize
import numpy as np

TOLERANCE = 1e-6
HASHING_DECIMAL = 4


def _full_hash_ndarray_float(arr: np.ndarray):
    return hash(np.round(arr, decimals=HASHING_DECIMAL).tobytes())


def _fast_very_dirty_hash_ndarray_float(arr: np.ndarray):
    return hash(repr(arr))


def _fast_dirty_hash_ndarray_float(arr: np.ndarray):
    return hash(repr(np.round(arr, decimals=HASHING_DECIMAL)))


class HashNdArrayMethod(str, Enum):
    """
    full: full hash ndarray . `hash(np.round(arr, decimals=HASHING_DECIMAL).tobytes())`
    dirty: hash ndarray fast with rounding . `hash(repr(np.round(arr, decimals=HASHING_DECIMAL)))`
    really_dirty: hash ndarray really fast without rounding by repr. `hash(repr(arr))`
        !!! Use this when the collision cost is not high and you need to hash arrays with 1M floats or more times per
        ms.
    """
    full = 'F'
    dirty = 'D'
    really_dirty = 'RD'


_NDARR_HASH_FUNCS = dict(F=_full_hash_ndarray_float, D=_fast_dirty_hash_ndarray_float,
        RD=_fast_very_dirty_hash_ndarray_float
        )


def hash_ndarray_float(arr: np.ndarray, method: HashNdArrayMethod = HashNdArrayMethod.full):
    return _NDARR_HASH_FUNCS[method.value](arr)

@vectorize(excluded=['tol'], signature='(),()->()')
def tol_lt(a, b=0, tol=TOLERANCE):
    """Tolerant less-than: return b-a>tol"""
    return b - a > tol


@vectorize(excluded=['tol'], signature='(),()->()')
def tol_gt(a, b=0, tol=TOLERANCE):
    """Tolerant greater-than: return a-b>tol"""
    return a - b > tol


@vectorize(excluded=['tol'], signature='(),()->()')
def tol_eq(a, b=0, tol=TOLERANCE):
    """Tolerant equal: return abs(a-b)<=tol"""
    return np.absolute(a - b) <= tol


@vectorize(excluded=['tol'], signature='(),()->()')
def tol_lte(a, b=0, tol=TOLERANCE):
    """Tolerant less-than-or-equal: return not tol_gt(a,b=0,tol)"""
    return not tol_gt(a, b, tol)


@vectorize(excluded=['tol'], signature='(),()->()')
def tol_gte(a, b=0, tol=TOLERANCE):
    """Tolerant greater-than-or-equal: return not tol_lt(a,b,tol)"""
    return not tol_lt(a, b, tol)


@vectorize(excluded=['tol'], signature='()->()')
def tol_round(val, tol=TOLERANCE):
    return val - (val % tol)
