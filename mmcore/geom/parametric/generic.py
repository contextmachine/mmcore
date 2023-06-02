from itertools import zip_longest

from typing import Iterable, Sequence

import dataclasses

import numpy as np

from mmcore.geom.parametric.base import ParametricObject
from types import LambdaType, FunctionType, MethodType
from scipy.linalg import solve
from scipy import optimize

from mmcore.geom.vectors import unit


class ParametricGeneric(ParametricObject):
    __equations__ = ()

    def evaluate(self, t):
        return list(map(lambda tt: [eq(self, tt) for eq in self.__equations__], t)) \
            if isinstance(t[0], (list, tuple, np.ndarray)) else [eq(self, t) for eq in self.__equations__]


class Parametrise:
    def __init__(self, cls):
        self.cls = cls
        self.cls.__equations__ = []

    def parametrise(self, pfun):
        self.cls.__equations__.append(pfun)

    def __call__(self, *args, **kwargs):
        def evaluate(slf, t):
            arr=np.array(list(map(lambda fp: fp[0](slf, fp[1]), zip_longest(self.cls.__equations__, [t], fillvalue=t))),
                     float)
            if arr.shape==(len(self.cls.__equations__),):
                return arr
            else:
                return arr.T
        self.cls.__call__ = evaluate

        return self.cls(*args, **kwargs)


@Parametrise
@dataclasses.dataclass
class LinAlgGeneric:
    a: Sequence[float]
    o: Sequence[float]
    def __post_init__(self):
        self.a = unit(self.a)

@LinAlgGeneric.parametrise
def x(self, t):
    return self.o[0] + self.a[0] * t


@LinAlgGeneric.parametrise
def y(self, t):
    return self.o[1] + self.a[1] * t


@LinAlgGeneric.parametrise
def z(self, t):
    return self.o[2] + self.a[2] * t


x = lambda self, t: self.a1 * t[0] + self.b1 * t[1] + self.c1
y = lambda self, t: self.a2 * t[0] + self.b2 * t[1] + self.c2
z = lambda self, t: self.a3 * t[0] + self.b3 * t[1] + self.c3


@dataclasses.dataclass
class Plane(ParametricGeneric):
    __equations__ = (lambda self, t: self.a1 * t[0] + self.b1 * t[1] + self.c1,
                     lambda self, t: self.a2 * t[0] + self.b2 * t[1] + self.c2,
                     lambda self, t: self.a3 * t[0] + self.b3 * t[1] + self.c3)

    a: float = 0.0
    b: float = 0.0
    c: float = 0.0

    a1: float = 0.0
    b1: float = 0.0
    c1: float = 0.0
    a2: float = 0.0
    b2: float = 0.0
    c2: float = 0.0
    a3: float = 0.0
    b3: float = 0.0
    c3: float = 0.0

    def implicit(self, x, y, z):
        return self.a * x + self.b * y + self.c * z + self.d


@dataclasses.dataclass
class Cylinder(ParametricGeneric):
    __equations__ = (lambda self, t: self.a1 * np.cos(t[0]) + self.b1 * np.sin(t[0]) + self.c1 * t[1] + self.d1,
                     lambda self, t: self.a2 * np.cos(t[0]) + self.b2 * np.sin(t[0]) + self.c2 * t[1] + self.d2,
                     lambda self, t: self.a3 * np.cos(t[0]) + self.b3 * np.sin(t[0]) + self.c3 * t[1] + self.d3)

    a1: float = 0.0
    b1: float = 0.0
    c1: float = 0.0
    a2: float = 0.0
    b2: float = 0.0
    c2: float = 0.0
    a3: float = 0.0
    b3: float = 0.0
    c3: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0


@dataclasses.dataclass
class CylinderI(ParametricGeneric):
    """
    Implict form:
        ax^2 + by^2 + cy^2 + 2(dxy + exz + fyz) + 2(gx + hy + iz) + l = 0
    ---
    The coefficients for a cylinder are stored as follows:
        Cone(a b c d e f g h i l)

    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    f: float = 0.0
    g: float = 0.0
    h: float = 0.0
    i: float = 0.0
    l: float = 0.0

    __equations__ = (lambda self, t: self.a1 * np.cos(t[0]) + self.b1 * np.sin(t[0]) + self.c1 * t[1] + self.d1,
                     lambda self, t: self.a2 * np.cos(t[0]) + self.b2 * np.sin(t[0]) + self.c2 * t[1] + self.d2,
                     lambda self, t: self.a3 * np.cos(t[0]) + self.b3 * np.sin(t[0]) + self.c3 * t[1] + self.d3)

    def implicit(self, x, y, z):
        return self.a * x ** 2 \
            + self.b * y ** 2 \
            + self.c * y ** 2 \
            + 2 * (self.d * x * y
                   + self.e * x * z
                   + self.f * y * z) \
            + 2 * (self.g * x
                   + self.h * y
                   + self.i * z) \
            + self.l


@dataclasses.dataclass
class Cone(ParametricGeneric):
    __equations__ = (
        lambda self, t: self.a1 * np.cos(t[0]) + self.b1 * np.sin(t[0]) + self.c1 * t[1] * np.cos(t[0]) + self.d1 * t[
            1] * np.sin(t[0]) + self.e1 * t[1] + self.f1,
        lambda self, t: self.a2 * np.cos(t[0]) + self.b2 * np.sin(t[0]) + self.c2 * t[1] * np.cos(t[0]) + self.d2 * t[
            1] * np.sin(t[0]) + self.e2 * t[1] + self.f2,
        lambda self, t: self.a3 * np.cos(t[0]) + self.b3 * np.sin(t[0]) + self.c3 * t[1] * np.cos(t[0]) + self.d3 * t[
            1] * np.sin(t[0]) + self.e3 * t[1] + self.f3)

    a1: float
    b1: float
    c1: float
    a2: float
    b2: float
    c2: float
    a3: float
    b3: float
    c3: float
    d1: float
    d2: float
    d3: float
    e1: float
    e2: float
    e3: float
    f1: float
    f2: float
    f3: float
