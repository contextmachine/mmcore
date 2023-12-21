"""
Partial Differential Equation (PDE)
"""
import functools
import types
from enum import Enum
from types import LambdaType

import numpy as np

from mmcore.func import vectorize

from mmcore.geom.vec import cross, perp2d, unit

__all__ = ['PDE', 'Offset', 'forward', 'central', 'backward', 'PDEMethodEnum']

from mmcore.geom.transform.rotations import ypr_matrix, Z90

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
    centralNd: LambdaType = np.vectorize(lambda f, h, t: (f(t + h) - f(t - h)) / (2 * h), excluded=[0, 1],
                                         signature='(j)->(i)'
                                         )


class PDE2DMethodEnum(Enum):
    """

    The `PDEMethodEnum` class is an enumeration that represents different methods for solving partial differential
    equations.

    Methods:

    * `central`: Method that computes the derivative of a function using central differencing. It uses a lambda
    function that takes the function, step size, and time as inputs and returns
    * the derivative. The lambda function uses `np.vectorize` to ensure element-wise computation and supports a
    custom signature `(i)` to indicate the return type.
    * `backward`: Method that computes the derivative of a function using backward differencing. It uses a lambda
    function that takes the function, step size, and time as inputs and returns
    * the derivative. The lambda function uses `np.vectorize` to ensure element-wise computation and supports a
    custom signature `(i)` to indicate the return type.
    * `forward`: Method that computes the derivative of a function using forward differencing. It uses a lambda
    function that takes the function, step size, and time as inputs and returns
    * the derivative. The lambda function uses `np.vectorize` to ensure element-wise computation and supports a
    custom signature `(i)` to indicate the return type.

    Note: The methods in this class are implemented using `np.vectorize` from the NumPy library for efficient
    element-wise array computations.

    """
    central: LambdaType = np.vectorize(lambda f, h, t: (f(t + h) - f(t - h)) / (2 * h), excluded=[0, 1],
                                       signature='()->(i)'
                                       )
    backward: LambdaType = np.vectorize(lambda f, h, t: (f(t) - f(t - h)) / h, excluded=[0, 1], signature='()->(i)')
    forward: LambdaType = np.vectorize(lambda f, h, t: (f(t + h) - f(t)) / h, excluded=[0, 1], signature='()->(i)')
    centralNd: LambdaType = np.vectorize(lambda f, h, t: (f(t + h) - f(t - h)) / (2 * h), excluded=[0, 1],
                                         signature='(j)->(i)'
                                         )
central = PDEMethodEnum.central.value
backward = PDEMethodEnum.backward.value
forward = PDEMethodEnum.forward.value


def uv(f, h, u, v):
    return np.array([(f(u + h, v) - f(u - h, v)) / (2 * h), (f(u, v + h) - f(u, v - h)) / (2 * h)])


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

    Examples:
    * Evaluate Tangent Planes on a NURBS Curve using PDE.
    >>> from mmcore.geom.parametric import NurbsCurve
    >>> from mmcore.func import vectorize
    >>> nc1=NurbsCurve([[0.0,1.0,0.0],[1.,0.5,0.0],[2.0,0.0,0.0],[3.0,-1.5,0.0],[4.0,-3.0,0.0],[5.0,-6,0.0]])
    >>> nc2=NurbsCurve([[0.0,0.0,0.0],[1.0,1.0,0.0],[2.0,0.5,0.0],[3.0,0.0,0.0],[4.0,0.5,0.0],[5.0,1,0.0]])

    >>> @vectorize(excluded=[0],signature='()->(i)')
    >>> def evaluate_nurbs(n, t):
    ...     return n.evaluate(t)

    >>> from mmcore.geom.pde import PDE
    >>> p=PDE(lambda t:evaluate_nurbs(nc1, t))
    >>> p(np.linspace(0,1,10))
    array([[  4.49325563,  -2.24663006,   0.        ],
           [  6.41667792,  -3.37501012,   0.        ],
           [  4.66667792,  -3.00001012,   0.        ],
           [  3.75000787,  -3.37500506,   0.        ],
           [  3.41667117,  -4.125     ,   0.        ],
           [  3.41667117,  -4.875     ,   0.        ],
           [  3.75000787,  -5.62502869,   0.        ],
           [  4.66667792,  -8.50005738,   0.        ],
           [  6.41667792, -15.62505738,   0.        ],
           [  4.49325562, -13.46965369,   0.        ]])
    >>> tangent_planes=p.plane(np.linspace(0,1,10))
    >>> tangent_planes.shape
    Out[4]: (4, 10, 3)
    >>> tangent_planes=np.swapaxes(tangent_planes, 0,1)
    >>> tangent_planes[0]
    array([[ 0.        ,  1.        ,  0.        ],
           [ 0.89442701, -0.44721395,  0.        ],
           [ 0.44721395,  0.89442701,  0.        ],
           [-0.        ,  0.        ,  1.        ]])
    >>> from mmcore.geom.vec import dot
    >>> np.allclose(dot(tangent_planes[0][[1,2]]), 0.)
    True
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
            self._pde = lambda t: self.method.value(self.func, self.h, t)
            cls.__instances__[hs] = self
            dfunc = self
        return dfunc

    def __call__(self, t):
        return self._pde(np.atleast_1d(t))

    def tan(self, t):

        return self.__call__(t)

    def normal(self, t):

        return unit(Z90.dot(unit(self._pde(t)).T).T)

    def plane(self, t):
        a, b = unit(self._pde(t)), self.normal(t)
        z = cross(a, b)

        orig = self.func(t)
        return np.array([orig, a, b, z])



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
    return curve.func(t) + curve._pde(t) * curve.distance


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

        self.distance = distance

        return self

    def __call__(self, t):
        return self.func(t) + unit(self.normal(t)) * self.distance

    def plane(self, t):
        pln = super().plane(t)
        pln[0] = self(t)
        return pln


class PDE2D:
    __instances__ = dict()

    def __new__(cls, func, method: PDEMethodEnum = PDEMethodEnum.centralNd, h=0.001, **kwargs):
        hs = hash((id(func), method, h, frozenset(kwargs.keys()), tuple(kwargs.values())))
        dfunc = cls.__instances__.get(hs, None)
        if dfunc is None:
            self = object.__new__(cls)
            self.func = func
            self.method = method
            self.h = h
            self._hs = hs
            self._pde = np.vectorize(lambda u, v: uv(func, self.h, u, v), signature='(),()->(j,i)')
            cls.__instances__[hs] = self
            dfunc = self
        return dfunc

    def __call__(self, u, v):

        return self._pde(u, v)

    def normal(self, u, v):
        xy = unit(self(u, v))
        return cross(xy[..., 0, :], xy[..., 1, :])

    def plane(self, u, v):
        xy = unit(self(u, v))
        return np.stack([self.func(u, v), xy[..., 0, :], xy[..., 1, :], cross(xy[..., 0, :], xy[..., 1, :])], axis=-1)
