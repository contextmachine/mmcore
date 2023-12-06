"""
Partial Differential Equation (PDE)
"""
import functools
import types
from enum import Enum
from types import LambdaType

import numpy as np

from mmcore.func import vectorize
from mmcore.geom.plane import create_plane
from mmcore.geom.vec import cross, perp2d

__all__ = ['PDE', 'Offset', 'forward', 'central', 'backward', 'PDEMethodEnum']


class PDEMethodEnum(Enum):
    """
    PDEMethodEnum

    Enum class representing different methods for solving partial differential equations (PDEs).

    Methods:
    - central: A method that calculates the central difference approximation for the derivative of a function.
    - backward: A method that calculates the backward difference approximation for the derivative of a function.
    - forward: A method that calculates the forward difference approximation for the derivative of a function.
    """
    central: LambdaType = np.vectorize(lambda f, h, t: (f(t + h) - f(t - h)) / (2 * h), excluded=[0, 1],
                                       signature='()->(i)')
    backward: LambdaType = np.vectorize(lambda f, h, t: (f(t) - f(t - h)) / h, excluded=[0, 1], signature='()->(i)')
    forward: LambdaType = np.vectorize(lambda f, h, t: (f(t + h) - f(t)) / h, excluded=[0, 1], signature='()->(i)')


__instances__ = dict()

central = PDEMethodEnum.central.value
backward = PDEMethodEnum.backward.value
forward = PDEMethodEnum.forward.value


class PDE:
    """

    The `PDE` class represents a partial differential equation. It is used to define and solve PDEs numerically.

    Attributes:
        - `func` (function): The function representing the PDE.
        - `method` (PDEMethodEnum): The numerical method used to solve the PDE. Default is `central`.
        - `h` (float): The step size used in the numerical method. Default is 0.001.
        - `kwargs` (dict): Additional keyword arguments to be passed to the PDE function.

    Methods:
        - `__new__(func, method, h, **kwargs)`: Constructor method. It creates a new instance of the `PDE` class. If an instance with the same parameters already exists, it returns that
    * instance instead.
        - `__call__(t)`: Method that evaluates the PDE at a given value of `t`.
        - `tan(t)`: Method that returns the tangent vector to the PDE at a given value of `t`.
        - `normal(t)`: Method that returns the normal vector to the PDE at a given value of `t`.
        - `plane(t)`: Method that returns the plane defined by the PDE at a given value of `t`.
    """
    def __new__(cls, func, method: PDEMethodEnum = central, h=0.001, **kwargs):
        hs = hash((id(func), method, h, frozenset(kwargs.keys()), tuple(kwargs.values())))
        dfunc = __instances__.get(hs, None)
        if dfunc is None:
            self = super().__new__(cls)

            self.func = func
            self.method = method
            self.h = h
            self._hs = hs

            self._pde = functools.lru_cache(maxsize=None)(lambda t: self.method(self.func, self.h, t))
            dfunc = self

            __instances__[hs] = self

        return dfunc

    def __call__(self, t):
        return self._pde(t)

    def tan(self, t):
        return perp2d(self._pde(t))

    def normal(self, t):
        return cross(self.func(t), self.tan(t))

    def plane(self, t):
        origin, xaxis, yaxis = self.func(t), self.tan(t), self.normal(t)
        return create_plane(x=xaxis, y=yaxis, origin=origin)

@vectorize(excluded=[0], signature='()->(i)')
def _offset_curve(curve, t):
    return curve.func(t) + curve._pde(t) * curve.distance(t)


def _create_distance_function(distance, param_range=(0, 1)):
    if isinstance(distance, (types.LambdaType, types.FunctionType, types.MethodType)):
        distance_func = np.vectorize(distance, signature='()->()')
    elif np.isscalar(distance):
        distance_func = np.vectorize(lambda x: distance, signature='()->()')
    else:
        distance_func = np.vectorize(lambda x: np.interp(x, distance, np.linspace(*param_range, len(distance))),
                                     signature='()->()')
    return distance_func


class Offset(PDE):
    def __new__(cls, func, distance, evaluate_range=(0, 1), *args, **kwargs):
        self = super().__new__(cls, func, distance=distance if isinstance(distance, float) else tuple(distance),
                               evaluate_range=tuple(evaluate_range), *args,
                               **kwargs)

        self.distance = _create_distance_function(distance, param_range=evaluate_range)

        return self

    def __call__(self, t):
        return _offset_curve(self, t)
