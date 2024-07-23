from __future__ import annotations

import abc
from typing import Union
import numpy as np

from mmcore.numeric import plane_on_curve
from mmcore.numeric.marching import marching_implicit_curve_points
from mmcore.geom.implicit.tree import ImplicitTree3D, ImplicitTree2D
from mmcore.numeric.vectors import scalar_norm

from mmcore.numeric.fdm import fdm
from mmcore.geom.curves.knot import interpolate_curve


def op_union(d1, d2):
    return min(d1, d2)


def op_subtraction(d1, d2):
    return max(-d1, d2)


def op_intersection(d1, d2):
    return max(d1, d2)


def op_xor(d1, d2):
    return max(min(d1, d2), -max(d1, d2))


def op_inv(a):
    return -a


def tangent2d(normal):
    tangent = np.zeros(2, dtype=float)
    tangent[0] = -normal[1]
    tangent[1] = normal[0]
    return tangent


def normal_from_function2d(f, d=0.0001):
    """Given a sufficiently smooth 3d function, f, returns a function approximating of the gradient of f.
    d controls the scale, smaller values are a more accurate approximation."""

    def norm(xy):
        D = np.eye(2) * d
        res = np.zeros(2, dtype=float)
        res[0] = (f(xy + D[0]) - f(xy - D[0])) / 2 / d
        res[1] = (f(xy + D[1]) - f(xy - D[1])) / 2 / d

        res / np.linalg.norm(res)

        return res

    return norm


def normal_from_function3d(f, d=0.0001):
    """Given a sufficiently smooth 3d function, f, returns a function approximating of the gradient of f.
    d controls the scale, smaller values are a more accurate approximation."""

    def norm(xyz):
        D = np.eye(3) * d

        res = np.zeros(3, dtype=float)
        res[0] = (f(xyz + D[0]) - f(xyz - D[0])) / 2 / d
        res[1] = (f(xyz + D[1]) - f(xyz - D[1])) / 2 / d
        res[2] = (f(xyz + D[2]) - f(xyz - D[2])) / 2 / d
        res / scalar_norm(res)

        return res

    return norm


class NpCache:
    def __init__(self, func):
        self._cache = dict()
        self.func = func

    def clear(self):
        self._cache.clear()

    def __call__(self, arg):
        key = repr(arg)
        if key in self._cache:
            return self._cache[key]

        res = self.func(arg)
        self._cache[key] = res
        return res


class Implicit:

    def __init__(self):
        super().__init__()

        self.dxdy = fdm(self.implicit)
        self._tree = None
        #self.implicit = NpCache(self.implicit)
        self._vimplicit = np.vectorize(self.implicit, signature="()->(i)")

    def invalidate_cache(self):
        self.implicit.clear()

    def implicit(self, v) -> float:
        ...

    def gradient(self, v):
        ...

    def point_on_curve(self, v):
        d = self.implicit(v)
        N = self.gradient(v) * d
        if d < 0:

            return v - N
        else:
            return v - N

    def v0(self):
        vmin, vmax = self.bounds()
        return vmax

    @staticmethod
    def normal_from_function(fun):
        ...

    def vimplicit(self, pt):
        return self._vimplicit(pt)

    def __call__(self, v):
        v = np.array(v) if not isinstance(v, np.ndarray) else v
        if v.ndim == 1:

            return self.implicit(v)
        else:
            return self.vimplicit(v)

    def bounds(self) -> Union[
        tuple[tuple[float, float], tuple[float, float]], tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """
        BBox
        """
        ...

    def build_tree(self, depth=3):
        ...

    @property
    def tree(self) -> Union[ImplicitTree2D, ImplicitTree3D]:
        if self._tree is None:
            self.build_tree()
        return self._tree


def is_implicit(obj):
    return getattr(obj, 'is_implicit', False)


class Implicit2D(Implicit):
    is_implicit = True

    def __init__(self, *args, **kwargs):
        super().__init__()

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        BBox
        """
        ...

    def point_inside(self, pt):
        """

        :param pt:
        :return:
        """
        res = self.implicit(pt)
        if res < 0.:
            return True
        elif np.isclose(res, 0):
            return True
        return False

    @staticmethod
    def normal_from_function(fun, d=0.0001):
        return normal_from_function2d(fun, d)

    def tangent(self, v):
        return tangent2d(self.gradient(v))

    def union(self, other: Implicit2D):
        return Union2D(self, other)

    def intersection_with_curve(self, curve):
        """

        :param curve: mmcore.geom.curves.Curve |

        Returns:

        """

    def gradient(self, xy):
        d = 0.001
        D = np.eye(2) * d
        res = np.zeros(2, dtype=float)
        res[0] = (self.implicit(xy + D[0]) - self.implicit(xy - D[0])) / 2 / d
        res[1] = (self.implicit(xy + D[1]) - self.implicit(xy - D[1])) / 2 / d

        res / np.linalg.norm(res)

        return res

    def intersection(self, other: Union[Implicit2D, Implicit3D]):
        return Intersection2D(self, other)

    def substraction(self, other: Implicit2D):
        return Sub2D(self, other)

    def __and__(self, other: Implicit2D):
        return self.intersection(other)

    def __sub__(self, other: Implicit2D):
        return self.substraction(other)

    def __add__(self, other: Implicit2D):
        return self.union(other)

    def __or__(self, other: Implicit2D):
        return self.union(other)

    def points(self, max_points: int = None, step: float = 0.2, delta: float = 0.001):
        """
        :param max_points: Specifies the maximum number of points to be generated on the curve. If not provided, all points will be generated.
        :type max_points: int
        :param step: Specifies the step size between each generated point. The default value is 0.2.
        :type step: float
        :param delta: Specifies the precision of the generated points. Smaller values result in more accurate points. The default value is 0.001.
        :type delta: float
        :return: An array of points representing the curve.
        :rtype: numpy.ndarray
        """
        v0 = np.array(self.v0())

        res = marching_implicit_curve_points(
            self.implicit, v0=v0, v1=v0, max_points=max_points, step=step, delta=delta
        )

        return np.array(res, dtype=float)

    def to_bspline(self, degree=3, step: float = 0.5):
        cpts, knots, deg = interpolate_curve(self.points(step=step), degree=degree)
        z = np.zeros((*cpts.shape[:-1], 3), dtype=float)
        z[..., :2] = cpts
        from mmcore.geom.curves.bspline import NURBSpline
        return NURBSpline(control_points=z, knots=np.array(knots, float), degree=degree)

    def build_tree(self, depth=3):
        self._tree = ImplicitTree2D(self.implicit, depth, bounds=self.bounds())


from mmcore.numeric.fdm import DEFAULT_H


class Implicit3D(Implicit):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def normal_from_function(fun, d=0.0001):
        return normal_from_function3d(fun, d)

    def union(self, other: Implicit3D):
        return Union3D(self, other)

    def intersection(self, other: Implicit3D):
        return Intersection3D(self, other)

    def substraction(self, other: Implicit3D):
        return Sub3D(self, other)

    def __and__(self, other: Implicit3D):
        return self.intersection(other)

    def __sub__(self, other: Implicit3D):
        return self.substraction(other)

    def __add__(self, other: Implicit3D):
        return self.union(other)

    def __or__(self, other: Implicit3D):
        return self.union(other)

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """
        BBox
        """
        ...

    def build_tree(self, depth=3):
        self._tree = ImplicitTree3D(self.implicit, depth, bounds=self.bounds())
        self._tree.build(depth=depth)

    def gradient(self, xyz):
        D = np.eye(3) * DEFAULT_H
        res = np.zeros(3, dtype=float)
        res[0] = (self.implicit(xyz + D[0]) - self.implicit(xyz - D[0])) / 2 / DEFAULT_H
        res[1] = (self.implicit(xyz + D[1]) - self.implicit(xyz - D[1])) / 2 / DEFAULT_H
        res[2] = (self.implicit(xyz + D[2]) - self.implicit(xyz - D[2])) / 2 / DEFAULT_H
        return res / scalar_norm(res)


class ImplicitOperator2D(Implicit2D):
    def __init__(self, a: Implicit2D, b: Implicit2D, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a
        self.b = b
        self._bounds = None
        self.solve_bounds()

    @abc.abstractmethod
    def implicit(self, v) -> float: ...

    def solve_bounds(self):
        ...


class Union2D(Implicit2D):
    def __init__(self, a: Implicit2D, b: Implicit2D):
        super().__init__()
        self.a = a
        self.b = b
        self._bounds = None
        self.solve_bounds()

    def implicit(self, v):
        return np.array(op_union(self.a.implicit(v), self.b.implicit(v)))

    def solve_bounds(self):
        (axmin, aymin), (axmax, aymax) = self.a.bounds()
        (bxmin, bymin), (bxmax, bymax) = self.b.bounds()
        self._bounds = (min(axmin, bxmin), min(aymin, bymin)), (
            max(axmax, bxmax),
            max(aymax, bymax),
        )

    def bounds(self):
        return self._bounds


class Union3D(Implicit3D):
    def __init__(self, a: Implicit3D, b: Implicit3D):
        super().__init__()
        self.a = a
        self.b = b
        self._bounds = None
        self.solve_bounds()

    def implicit(self, v):
        return np.array(op_union(self.a.implicit(v), self.b.implicit(v)))

    def solve_bounds(self):
        (axmin, aymin, azmin), (axmax, aymax, azmax) = self.a.bounds()
        (bxmin, bymin, bzmin), (bxmax, bymax, bzmax) = self.b.bounds()
        self._bounds = (min(axmin, bxmin),
                        min(aymin, bymin),
                        min(azmin, bzmin)), (
            max(axmax, bxmax),
            max(aymax, bymax),
            max(azmax, bzmax),
        )

    def bounds(self):
        return self._bounds


class Intersection2D(Implicit2D):
    def __init__(self, a: Implicit2D, b: Implicit2D):
        super().__init__()
        self.a = a
        self.b = b
        self._bounds = None
        self.solve_bounds()

    def implicit(self, v):
        def op_intersection(d1, d2):
            return max(d1, d2)

        return np.array(op_intersection(self.a.implicit(v), self.b.implicit(v)))

    def solve_bounds(self):
        (axmin, aymin), (axmax, aymax) = self.a.bounds()
        (bxmin, bymin), (bxmax, bymax) = self.b.bounds()
        self._bounds(max(axmin, bxmin), max(aymin, bymin)), (
            min(axmax, bxmax),
            min(aymax, bymax),
        )

    def bounds(self):
        return self._bounds


class Intersection3D(Implicit3D):
    def __init__(self, a: Implicit3D, b: Implicit3D):
        super().__init__()
        self.a = a
        self.b = b
        self._bounds = None
        self.solve_bounds()

    def solve_bounds(self):
        (axmin, aymin, azmin), (axmax, aymax, azmax) = self.a.bounds()
        (bxmin, bymin, bzmin), (bxmax, bymax, bzmax) = self.b.bounds()
        self._bounds = (max(axmin, bxmin),
                        max(aymin, bymin),
                        max(azmin, bzmin)), (
            min(axmax, bxmax),
            min(aymax, bymax),
            min(azmax, bzmax),
        )

    def implicit(self, v):
        return np.array(op_intersection(self.a.implicit(v), self.b.implicit(v)))

    def bounds(self):
        return self._bounds


class Sub2D(Implicit2D):
    def __init__(self, a: Implicit2D, b: Implicit2D):
        super().__init__()
        self.a = a
        self.b = b

    def implicit(self, v):
        return np.array(op_subtraction(self.a.implicit(v), self.b.implicit(v)))

    def bounds(self):
        return self.a.bounds()


class Sub3D(Implicit3D):
    def __init__(self, a: Implicit3D, b: Implicit3D):
        super().__init__()
        self.a = a
        self.b = b

    def implicit(self, v):
        return np.array(op_subtraction(self.a.implicit(v), self.b.implicit(v)))

    def bounds(self):
        return self.a.bounds()


from mmcore.geom.curves.bspline import interpolate_nurbs_curve, NURBSpline

from mmcore.geom.curves.curve import Curve
from mmcore.numeric.algorithms.implicit_point import curve_point


class ParametrizedImplicit2D(Implicit2D, Curve):
    def __init__(self, a: Implicit2D, b: NURBSpline, periodic=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._implicit_form = a
        self._parametric_form = b
        self.periodic = periodic
    def interval(self):
        return tuple(self._parametric_form.interval())
    def bounds(self):
        return self._implicit_form.bounds()

    def implicit(self, v):
        return self._implicit_form.implicit(v)

    def gradient(self, v):
        return self._implicit_form.gradient(v)

    def evaluate(self, t: float):
        return curve_point(self._implicit_form.implicit, self._parametric_form.evaluate(t)[:2],
                           grad=self._implicit_form.gradient)

    def evaluate_multi(self, t):
        pts = self._parametric_form.evaluate_multi(t)
        return np.array([
            curve_point(self._implicit_form.implicit, pt[:2],
                        grad=self._implicit_form.gradient) for pt in pts])

    def tangent(self, t):
        return self._parametric_form.tangent(t)[:2]

    def normal(self, t):
        return self._implicit_form.gradient(self._parametric_form.evaluate(t)[:2])

    def derivative(self, t):
        return self._parametric_form.derivative(t)[:2]
    def plane_at(self, t):
        return plane_on_curve(
            (*self.evaluate(t),0), (*self.tangent(t),0), (*self.second_derivative(t),0)
        )
    def second_derivative(self, t):
        return self._parametric_form.second_derivative(t)[:2]

    def is_periodic(self):
        return self.periodic

    def point_inside(self, pt):
        return self._implicit_form.point_inside(pt)

    @classmethod
    def from_implicit2d(cls, obj: Implicit2D, start=None, end=None, max_points=None, degree=3, step=0.2,
                        delta=0.001):
        return parametrize_implicit2d(obj, start, end, max_points, degree, step, delta)
    @property
    def control_points(self):
        return self._parametric_form.control_points
    @property
    def degree(self):
        return self._parametric_form.degree

    @property
    def implicit_form(self):
        return self._implicit_form
    @property
    def parametric_form(self):
        return self._parametric_form

def parametrize_implicit2d(obj: Implicit2D, start=None, end=None, max_points=None, degree=3, step=0.2,
                           delta=0.001) -> ParametrizedImplicit2D:
    periodic = True
    if start is None and end is None:
        v0 = np.array(obj.v0())
        v1 = v0
    elif end is None:
        v0 = np.array(start)
        v1 = v0
    else:
        if not np.allclose(end, start):
            periodic = False
            v0 = np.array(start)
            v1 = np.array(end)
        else:
            v0 = np.array(start)
            v1 = v0

    res = marching_implicit_curve_points(
        obj.implicit, v0=v0, v1=v1, max_points=max_points, step=step, delta=delta
    )
    pts = np.zeros((len(res), 3))
    pts[:, :2] = res
    crv = interpolate_nurbs_curve(pts, degree=degree)
    if periodic:
        crv.make_periodic()

    return ParametrizedImplicit2D(obj, crv, periodic)
