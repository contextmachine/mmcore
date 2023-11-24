#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"

from functools import wraps

import compas.geometry as cg

from mmcore.geom.parametric.algorithms import *
from mmcore.geom.parametric.sketch import *


def to_cmp_point(func):
    @wraps(func)
    def wrp(*a, **kw):
        return cg.Point(*func(*a, **kw))

    return wrp


class ParametricSupport:
    def __init_subclass__(cls, signature='()->(i)'):
        cls.__np_vec_signature__ = signature

    def __new__(cls):
        self = super().__new__(cls)
        self.__evaluate__ = np.vectorize(self.evaluate, signature=cls.__np_vec_signature__)
        return self

    def evaluate(self, *args) -> np.ndarray:
        return ...
