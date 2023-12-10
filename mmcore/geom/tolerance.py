from mmcore.func import vectorize
import numpy as np

TOLERANCE = 1e-6


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
