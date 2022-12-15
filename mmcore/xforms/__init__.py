#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
from functools import wraps
from typing import Any, Union

import numpy as np
from compas.data import Data
from compas.geometry import Transformation
from numpy import ndarray

from ..baseitems import Item


def mirror(right):
    mirror = np.eye(3, 3)
    mirror[1, 1] = -1
    left = np.zeros((len(right), 3))

    for i, pt in enumerate(right):
        # print(i,pt)
        left[i, ...] = mirror @ np.asarray(pt).T
    return left


class Xf(Item):
    def __init__(self, *args, **kwargs):
        self._xform = None
        super().__init__(pass_call=True, **kwargs)

    @property
    def xform(self):
        return np.asarray(self._xform).reshape(4, 4)

    @xform.setter
    def xform(self, val):
        self._xform = np.asarray(val).reshape(4, 4)

    def __getitem__(self, item: Union[slice, int, ndarray, tuple, Any]) -> ndarray:
        return self.xform[item]

    def __setitem__(self, key: Union[slice, int, ndarray, tuple, Any], value):
        self.xform[key] = value

    def __matmul__(self, other):
        if hasattr(other, "transform"):
            if issubclass(other.__class__, Data):
                T = Transformation(self.xform[:3, :3])
                other.transform(T)
                return other
            else:
                pass
        else:
            pass

    def __rmatmul__(self, other):
        return self.__matmul__(other)

    def frame_to_frame(self, frame_a, frame_b):
        T = Transformation.from_frame_to_frame(frame_a, frame_b)
        self.xform = np.asarray(T).resize((4, 4))


class XformDecorator(Xf):

    def __init__(self, target, *args, **kwargs):
        self.target = target
        super().__init__(**kwargs)

    def __call__(self, *args, pass_call=True, **kwargs):
        @wraps(self.target)
        def wrp(*arg, **kw):
            kw |= kwargs
            return self.__matmul__(self.target(*(args + arg), **kw))

        if pass_call:
            super().__call__(*args, **kwargs)
        else:
            return wrp


class XformParametricDecorator(XformDecorator):
    def __init__(self, target, **kwargs):
        super().__init__(target, **kwargs)
        self.target = target

    def __call__(self, *args, **kwargs):
        self.frame_to_frame(self.target.old_plane, self.target.new_plane)

        return super().__call__(*args, **kwargs)
