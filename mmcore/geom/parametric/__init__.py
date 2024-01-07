#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"

from functools import wraps

import numpy as np

from mmcore.geom.parametric.algorithms import *
from mmcore.geom.parametric.sketch import *
from mmcore.geom.plane import local_to_world, rotate_plane_inplace, translate_plane_inplace, world_to_local, \
    create_plane, WXY
from mmcore.ranges import Range, Range2D




class ParametricSupport:
    def __init_subclass__(cls, match_args=('a',), signature='()->(i)', param_range=(0.0, 1.0)):
        cls.__np_vec_signature__ = signature
        cls.__match_args__ = match_args
        shp = np.array(param_range).shape
        if len(shp) == 1:
            cls.__range__ = Range(*param_range)
        else:
            cls.__range__ = Range2D(*param_range)

    def __new__(cls, origin=None, plane=None):
        self = super().__new__(cls)
        if origin is None:
            origin = np.array([0, 0, 0])
        if plane is None:
            self.plane = create_plane(origin=origin)
        else:
            self.plane = plane

        self.__evaluate__ = np.vectorize(self.evaluate, signature=cls.__np_vec_signature__)
        return self

    @property
    def origin(self):
        return self.plane.origin

    @origin.setter
    def origin(self, v):
        self.plane.origin = np.array(v)

    def evaluate(self, *args) -> np.ndarray:
        return ...

    @property
    def range(self) -> Range:
        return self.__range__

    def __call__(self, *args):
        ...

    def to_world(self, value):
        return local_to_world(value, self.plane) if self.plane != WXY else value

    def rotate(self, angle, axis=None):
        rotate_plane_inplace(self.plane, angle, axis=axis)

    def translate(self, vector):
        translate_plane_inplace(self.plane, np.array(vector))

    def to_local(self, value, plane=WXY):
        return world_to_local(value, self.plane) if self.plane != plane else value

    def copy_to_new_plane(self, plane):
        other = copy.copy(self)
        other.plane = plane
        return other

    def deepcopy_to_new_plane(self, plane):
        other = copy.deepcopy(self)
        other.plane = plane
        return other

    def __iter__(self):
        return (self.__dict__[arg] for arg in self.__match_args__)

    def __hash__(self):
        return hash(tuple(self.__iter__()))

    def __eq__(self, other):
        if self.__match_args__ != other.__match_args__:
            return False
        else:
            return self.__hash__() == other.__hash__()
