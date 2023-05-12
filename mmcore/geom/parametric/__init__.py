#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
import copy
from abc import ABC, ABCMeta, abstractmethod
from functools import wraps
from mmcore.geom.parametric.sketch import *
import compas.geometry as cg


def to_cmp_point(func):
    @wraps(func)
    def wrp(*a, **kw):
        return cg.Point(*func(*a, **kw))

    return wrp


