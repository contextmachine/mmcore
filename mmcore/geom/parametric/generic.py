import dataclasses

import numpy as np

from mmcore.geom.parametric.base import ParametricObject
from types import LambdaType, FunctionType, MethodType
from scipy.linalg import solve
from scipy import optimize

class ParametricGeneric(ParametricObject):
    __equations__ = ()

    def evaluate(self, t):

        return list(map(lambda tt: [eq(self, tt) for eq in self.__equations__], t)) \
            if isinstance(t[0], (list, tuple, np.ndarray)) else [eq(self, t) for eq in self.__equations__]

@dataclasses.dataclass
class LinAlgGeneric(ParametricObject):
    __equations__ = ()
    a: type(np.ndarray(shape=(3, 3), dtype=float))
    def evaluate(self, t):

        if len(t.shape)==1:
            t.resize((1,3) )
            t[:,2]+=1.0
        else:
            t.resize((t.shape[0], 3))
            t[:,2]+=1.0

        return solve(self.a, t.T)


x = lambda self, t: self.a1 * t[0] + self.b1 * t[1] + self.c1
y = lambda self, t: self.a2 * t[0] + self.b2 * t[1] + self.c2
z = lambda self, t: self.a3 * t[0] + self.b3 * t[1] + self.c3


@dataclasses.dataclass
class Plane(ParametricGeneric):
    __equations__ = (lambda self, t: self.a1 * t[0] + self.b1 * t[1] + self.c1,
                     lambda self, t: self.a2 * t[0] + self.b2 * t[1] + self.c2,
                     lambda self, t: self.a3 * t[0] + self.b3 * t[1] + self.c3)

    a1: float
    b1: float
    c1: float
    a2: float
    b2: float
    c2: float
    a3: float
    b3: float
    c3: float



@dataclasses.dataclass
class Cylinder(ParametricGeneric):
    __equations__ = (lambda self, t: self.a1*np.cos(t[0]) + self.b1*np.sin(t[0]) + self.c1*t[1] + self.d1  ,
                     lambda self, t: self.a2*np.cos(t[0]) + self.b2*np.sin(t[0]) + self.c2*t[1] + self.d2  ,
                     lambda self, t: self.a3*np.cos(t[0]) + self.b3*np.sin(t[0]) + self.c3*t[1] + self.d3  )

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
@dataclasses.dataclass
class Cylinder(ParametricGeneric):
    __equations__ = (lambda self, t: self.a1*np.cos(t[0]) + self.b1*np.sin(t[0]) + self.c1*t[1] + self.d1  ,
                     lambda self, t: self.a2*np.cos(t[0]) + self.b2*np.sin(t[0]) + self.c2*t[1] + self.d2  ,
                     lambda self, t: self.a3*np.cos(t[0]) + self.b3*np.sin(t[0]) + self.c3*t[1] + self.d3  )

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












@dataclasses.dataclass
class Cone(ParametricGeneric):
    __equations__ = (lambda self, t: self.a1*np.cos(t[0]) + self.b1*np.sin(t[0]) + self.c1*t[1]*np.cos(t[0]) + self.d1*t[1] *np.sin(t[0]) + self.e1*t[1] + self.f1 ,
                     lambda self, t: self.a2*np.cos(t[0]) + self.b2*np.sin(t[0]) + self.c2*t[1]*np.cos(t[0]) + self.d2*t[1] *np.sin(t[0]) + self.e2*t[1] + self.f2 ,
                     lambda self, t: self.a3*np.cos(t[0]) + self.b3*np.sin(t[0]) + self.c3*t[1]*np.cos(t[0]) + self.d3*t[1] *np.sin(t[0]) + self.e3*t[1] + self.f3 )

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
