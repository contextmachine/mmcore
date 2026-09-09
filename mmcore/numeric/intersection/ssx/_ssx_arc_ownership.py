"""Containment of exact parameter paths already proved to be source zero arcs.

These paths are certificates supplied by exact source tiers, not samples of
an approximate curve. Their exact affine span maps survive NURBS assembly.
"""
from fractions import Fraction

import numpy as np

from mmcore.numeric.intersection.ssx._ssx_polyline import _covered_fraction


def source_path_covered(path, keepers, charge=None):
    """Whether exact source path is covered by exact source paths, or unknown."""
    if path is None or not keepers:
        return False
    if charge is not None and not charge(max(1,len(path)+sum(len(keeper) for keeper in keepers if keeper is not None))):
        return None
    if any(len(row) != 4 or any(not isinstance(x,Fraction) for x in row) for row in path):
        return False
    segments = []
    for keeper in keepers:
        if keeper is None or any(len(row) != 4 or any(not isinstance(x,Fraction) for x in row)
                                 for row in keeper):
            continue
        segments.extend(zip(keeper,keeper[1:]))
    if len(path) < 2 or not segments:
        return False
    for a,b in zip(path,path[1:]):
        intervals = []
        for c,d in segments:
            if charge is not None and not charge(1):
                return None
            if any(max(x,y) < min(z,w) or max(z,w) < min(x,y)
                   for x,y,z,w in zip(a,b,c,d)):
                continue
            if charge is not None and not charge(128):
                return None
            p,q,r,s = (np.asarray(row,dtype=object) for row in (a,b,c,d))
            interval = _covered_fraction(p,q-p,r,s-r,(Fraction(0),)*4,exact=True)
            if interval is not None:
                intervals.append(interval)
        reach = Fraction(0)
        for lo,hi in sorted(intervals):
            if lo > reach:
                break
            reach = max(reach,hi)
        if reach < 1:
            return False
    return True


def map_source_path(path, bounds):
    if path is None:
        return None
    axes = [(Fraction.from_float(float(lo)),Fraction.from_float(float(hi))) for lo,hi in bounds]
    return tuple(tuple(lo+x*(hi-lo) for x,(lo,hi) in zip(row,axes)) for row in path)
