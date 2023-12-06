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

    The `PDEMethodEnum` class is an enumeration that represents different methods for solving partial differential equations.

    Methods:

    * `central`: Method that computes the derivative of a function using central differencing. It uses a lambda function that takes the function, step size, and time as inputs and returns
    * the derivative. The lambda function uses `np.vectorize` to ensure element-wise computation and supports a custom signature `(i)` to indicate the return type.
    * `backward`: Method that computes the derivative of a function using backward differencing. It uses a lambda function that takes the function, step size, and time as inputs and returns
    * the derivative. The lambda function uses `np.vectorize` to ensure element-wise computation and supports a custom signature `(i)` to indicate the return type.
    * `forward`: Method that computes the derivative of a function using forward differencing. It uses a lambda function that takes the function, step size, and time as inputs and returns
    * the derivative. The lambda function uses `np.vectorize` to ensure element-wise computation and supports a custom signature `(i)` to indicate the return type.

    Note: The methods in this class are implemented using `np.vectorize` from the NumPy library for efficient element-wise array computations.

    """
    central: LambdaType = np.vectorize(lambda f, h, t: (f(t + h) - f(t - h)) / (2 * h), excluded=[0, 1],
                                       signature='()->(i)')
    backward: LambdaType = np.vectorize(lambda f, h, t: (f(t) - f(t - h)) / h, excluded=[0, 1], signature='()->(i)')
    forward: LambdaType = np.vectorize(lambda f, h, t: (f(t + h) - f(t)) / h, excluded=[0, 1], signature='()->(i)')


central = PDEMethodEnum.central.value
backward = PDEMethodEnum.backward.value
forward = PDEMethodEnum.forward.value


class PDE:
    """
    PDE

    Class representing a Partial Differential Equation (PDE) object.

    Attributes:
        __instances__ (dict): Dictionary to store instances of PDE objects for memoization.

    Methods:
        __new__(func, method=PDEMethodEnum.central, h=0.001, **kwargs)
            Creates a new PDE object or returns an existing one if the parameters match.

        __call__(t)
            Evaluates the PDE at the given time t.

        tan(t)
            Calculates the tangent vector to the PDE at the given time t.

        normal(t)
            Calculates the normal vector to the PDE at the given time t.

        plane(t)
            Constructs a plane defined by the PDE at the given time t.
    """
    __instances__ = {}

    def __new__(cls, func, method: PDEMethodEnum = PDEMethodEnum.central, h=0.001, **kwargs):
        hs = hash((id(func), method, h, frozenset(kwargs.keys()), tuple(kwargs.values())))
        dfunc = cls.__instances__.get(hs, None)
        if dfunc is None:
            self = super().__new__(cls)
            self.func = func
            self.method = method
            self.h = h
            self._hs = hs
            self._pde = functools.lru_cache(maxsize=None)(lambda t: self.method(self.func, self.h, t))
            cls.__instances__[hs] = self
            dfunc = self
        return dfunc

    def __call__(self, t):
        return self._pde(t)

    def tan(self, t):
        return perp2d(self.__call__(t))

    def normal(self, t):
        return cross(self.__call__(t), self.tan(t))

    def plane(self, t):
        origin, xaxis, yaxis = self.__call__(t), self.tan(t), self.normal(t)
        return create_plane(x=xaxis, y=yaxis, origin=origin)


@vectorize(excluded=[0], signature='()->(i)')
def _offset_curve(curve, t):
    """
    :param curve: The input curve.
    :type curve: Curve object

    :param t: The parameter value along the curve where the offset is calculated.
    :type t: float

    :return: The offset point on the curve at the given parameter value.
    :rtype: float

    """
    return curve.func(t) + curve._pde(t) * curve.distance(t)


def _create_distance_function(distance, param_range=(0, 1)):
    """
    :param distance: The distance values used to create the distance function.
    :type distance: scalar value, list, or array-like object

    :param param_range: The range used to interpolate the distance values. Defaults to (0,1).
    :type param_range: tuple of scalar values, optional

    :return: A numpy vectorized distance function based on the input distance values.
    :rtype: numpy.ufunc
    """
    if isinstance(distance, (types.LambdaType, types.FunctionType, types.MethodType)):
        distance_func = np.vectorize(distance, signature='()->()')
    elif np.isscalar(distance):
        distance_func = np.vectorize(lambda x: distance, signature='()->()')
    else:
        distance_func = np.vectorize(lambda x: np.interp(x, distance, np.linspace(*param_range, len(distance))),
                                     signature='()->()')
    return distance_func


class Offset(PDE):
    """
    Module for creating offset curves.

    :class:`Offset` is a class that extends :class:`PDE` and represents an offset curve.

    Example:
        >>> from mmcore.geom.circle import Circle
        >>> c = Circle(1)
        >>> offset_c = Offset(c, 0.5)
        >>> offset_c(0.2)
        (0.28569405369638814, 0.95838210737277628)

    Attributes:
        distance (function): The distance between the original curve and the offset curve.
    """

    def __new__(cls, func, distance, evaluate_range=(0, 1), **kwargs):
        self = super().__new__(cls,
                               func,
                               distance=distance if isinstance(distance, float) else tuple(distance),
                               evaluate_range=tuple(evaluate_range),
                               **kwargs)

        self.distance = _create_distance_function(distance, param_range=evaluate_range)

        return self

    def __call__(self, t):
        return _offset_curve(self, t)
