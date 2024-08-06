from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.interpolate import interp1d

from mmcore.geom.curves.curve import Curve
from mmcore.geom.parametric import ParametricCurve
from mmcore.geom.parametric import BiLinear as CBiLinear

from mmcore.geom.polygon import polygon_build_bvh
from mmcore.numeric.algorithms.point_in_curve import point_in_parametric_curve
from mmcore.numeric.fdm import Grad, DEFAULT_H, newtons_method
from mmcore.numeric.numeric import evaluate_normal2
from mmcore.numeric.vectors import scalar_dot, scalar_cross, scalar_unit, scalar_norm


class TwoPointForm:
    def __init__(self, pt1, pt2):
        self.x1, self.x2 = pt1[0], pt2[0]
        self.y1, self.y2 = pt1[1], pt2[1]
        self._xd = self.x1 - self.x2

    def y(self, x):
        return (x * self.y1 - x * self.y2 + self.x1 * self.y2 - self.x2 * self.y1) / (
                self.x1 - self.x2
        )

    def x(self, y):
        return (self.x1 * y - self.x1 * self.y2 - self.x2 * y + self.x2 * self.y1) / (
                self.y1 - self.y2
        )

    def direction(self):
        return np.array([self.x2 - self.x1, self.y2 - self.y1])

    def det_form(self, x, y):
        return np.array([[x - self.x1, y - self.y1], self.direction()])

    def __call__(self, x, y):
        return np.linalg.det(self.det_form(x, y))


def compute_intersection_curvature(Su1, Sv1, Suu1, Suv1, Svv1, Su2, Sv2, Suu2, Suv2, Svv2):
    """
    Compute the curvature of the intersection curve between two parametric surfaces given their partial derivatives.

    Parameters:
    -  Su1, Sv1, Suu1, Suv1, Svv1: partial derivatives of the first surface at the intersection point
    -  Su2, Sv2, Suu2, Suv2, Svv2: partial derivatives of the second surface at the intersection point

    Returns:
    - curvature: curvature of the intersection curve at the given point
    """

    # Compute normal vectors
    N1 = np.array(np.cross(Su1, Sv1))
    N1 /= np.linalg.norm(N1)
    N2 = np.array(np.cross(Su2, Sv2))
    N2 /= np.linalg.norm(N2)

    # Check if the surfaces intersect tangentially
    cos_theta = np.dot(N1, N2)
    if np.isclose(np.abs(cos_theta), 1):
        raise ValueError("The surfaces intersect tangentially at the given point")

    # Compute tangent vector of the intersection curve
    T = np.array(np.cross(N1, N2))
    T /= np.linalg.norm(T)

    # Compute the curvature vector
    L1 = np.dot(Suu1, N1)
    M1 = np.dot(Suv1, N1)
    N1_2 = np.dot(Svv1, N1)
    L2 = np.dot(Suu2, N2)
    M2 = np.dot(Suv2, N2)
    N2_2 = np.dot(Svv2, N2)
    k1 = (L1 * (np.dot(Su1, T)) ** 2 + 2 * M1 * np.dot(Su1, T) * np.dot(Sv1, T) + N1_2 * (np.dot(Sv1, T)) ** 2)
    k2 = (L2 * (np.dot(Su2, T)) ** 2 + 2 * M2 * np.dot(Su2, T) * np.dot(Sv2, T) + N2_2 * (np.dot(Sv2, T)) ** 2)
    curvature_vector = (k1 * N1 + k2 * N2) / (1 - cos_theta ** 2)

    # Compute the curvature magnitude
    #curvature = np.linalg.norm(curvature_vector)

    return curvature_vector, T


from mmcore.geom.implicit import ParametrizedImplicit2D


class CurveOnSurface(Curve):
    def __init__(self, surf: "Surface", curve: "Curve|Callable", interval=(0., 1.)):
        super().__init__()
        self.surf = surf
        self.curve = curve
        self._interval = interval
        self._eval_crv_func = getattr(self.curve, "evaluate", self.curve)
        self._polygon = None
        self._edges = None
        self._bvh_uv_root = None

    def build_uv_bvh(self):
        if hasattr(self.curve, 'degree'):
            if self.curve.degree == 1:
                self._polygon = self.curve.evaluate_multi(np.arange(*self.curve.interval()))[:, :2]

            elif hasattr(self.curve, 'points'):
                self._polygon = self.curve.points()[:, :2]

            else:
                self._polygon = self.curve(np.linspace(*self.curve.interval(), 50))[:, :2]
        else:
            self._polygon = self.curve(np.linspace(*self.curve.interval(), 50))[:, :2]
        self._edges = [(self._polygon[i], self._polygon[(i + 1) % len(self._polygon)]) for i in
                       range(len(self._polygon))]
        self._bvh_uv_root = polygon_build_bvh(self._edges)

    def plane_at(self, t):
        O = self.evaluate(t)
        T = scalar_unit(self.tangent(t))
        uv = self.evaluate_curve(t)
        B = scalar_unit(self.surf.normal(uv))
        N = scalar_cross(B, T)
        return np.array([O, T, N, B])

    def normal(self, t):
        uv = self.evaluate_curve(t)
        return np.array(scalar_cross(*scalar_unit([self.surf.normal(uv), self.tangent(t)])))

    def evaluate(self, t: float):
        return self.surf.evaluate(self.evaluate_curve(t))

    def evaluate_curve(self, t):
        return self._eval_crv_func(t)[..., :2]

    def interval(self):
        return self._interval

    def point_inside(self, uv):

        return point_in_parametric_curve(self.curve, uv)


from mmcore.geom.bvh import BVHNode, contains_point
from mmcore.numeric.divide_and_conquer import divide_and_conquer_min_2d
from mmcore.numeric.fdm import gradient as fgrdient


class Surface:
    _tree: Optional[BVHNode] = None

    def __init__(self):
        super().__init__()
        self._tree = None
        self.evaluate_multi = np.vectorize(self.evaluate, signature="(i)->(j)")
        self._grad = Grad(self)
        self._interval = np.array(self.interval())

        self._uh = np.array([DEFAULT_H, 0.0])
        self._vh = np.array([0.0, DEFAULT_H])
        self._plane_at_multi = np.vectorize(self.plane_at, signature="(i)->(j)")
        self._boundary = None

    def build_boundary(self):

        u0, u1 = self._interval[0]
        v0, v1 = self._interval[1]
        boundary_curve = NURBSpline(np.array([[u0, v0, 0.], [u1, v0, 0.], [u1, v1, 0.], [u0, v1, 0.], [u0, v0, 0.]]),
                                    degree=1)

        self._boundary = CurveOnSurface(self, boundary_curve, interval=tuple(boundary_curve.interval()))
        self._boundary.build_uv_bvh()

    def inversion(self, pt):
        def wrp1(uv):
            d = self.evaluate(uv) - pt
            return scalar_dot(d, d)

        def wrp(u, v):
            d = self.evaluate(np.array([u, v])) - pt
            return scalar_dot(d, d)

        cpt = contains_point(self.tree, pt)

        if len(cpt) == 0:
            (umin, umax), (vmin, vmax) = self.interval()
            return divide_and_conquer_min_2d(wrp, (umin, umax), (vmin, vmax), 1e-3)

        else:

            initial = np.average(min(cpt, key=lambda x: x.bounding_box.volume()).uvs, axis=0)
            return newtons_method(wrp1, initial)

    def implicit(self, pt):
        uv = self.inversion(pt)
        direction = pt - self.evaluate(uv)
        #val = scalar_dot(direction, scalar_cross(self.derivative_u(uv), self.derivative_v(uv)))

        return scalar_dot(direction, direction)

    def bounds(self):
        return np.array([self.tree.bounding_box.min_point, self.tree.bounding_box.max_point])

    def gradient(self, pt):

        gg = self.inversion(pt)

        return self.plane_at(gg)[-1]

    @property
    def boundary(self):
        return self._boundary

    def derivative_u(self, uv):
        if (1 - DEFAULT_H) >= uv[0] >= DEFAULT_H:
            return (
                    (self.evaluate(uv + self._uh) - self.evaluate(uv - self._uh))
                    / 2
                    / DEFAULT_H
            )
        elif uv[0] < DEFAULT_H:
            return (self.evaluate(uv + self._uh) - self.evaluate(uv)) / DEFAULT_H
        else:
            return (self.evaluate(uv) - self.evaluate(uv - self._uh)) / DEFAULT_H

    def derivative_v(self, uv):
        if (1 - DEFAULT_H) >= uv[1] >= DEFAULT_H:
            return (
                    (self.evaluate(uv + self._vh) - self.evaluate(uv - self._vh))
                    / 2
                    / DEFAULT_H
            )
        elif uv[1] < DEFAULT_H:
            return (self.evaluate(uv + self._vh) - self.evaluate(uv)) / DEFAULT_H
        else:
            return (self.evaluate(uv) - self.evaluate(uv - self._vh)) / DEFAULT_H

    def second_derivative_uu(self, uv):
        if (1 - DEFAULT_H) >= uv[0] >= DEFAULT_H:
            return (
                    (self.derivative_u(uv + self._uh) - self.derivative_u(uv - self._uh))
                    / 2
                    / DEFAULT_H
            )
        elif uv[0] < DEFAULT_H:
            return (
                    self.derivative_u(uv + self._uh) - self.derivative_u(uv)
            ) / DEFAULT_H
        else:
            return (
                    self.derivative_u(uv) - self.derivative_u(uv - self._uh)
            ) / DEFAULT_H

    def second_derivative_vv(self, uv):
        if (1 - DEFAULT_H) >= uv[1] >= DEFAULT_H:
            return (
                    (self.derivative_v(uv + self._vh) - self.derivative_v(uv - self._vh))
                    / 2
                    / DEFAULT_H
            )
        elif uv[1] < DEFAULT_H:
            return (
                    self.derivative_v(uv + self._vh) - self.derivative_v(uv)
            ) / DEFAULT_H
        else:
            return (
                    self.derivative_v(uv) - self.derivative_v(uv - self._vh)
            ) / DEFAULT_H

    def second_derivative_uv(self, uv):
        if (1 - DEFAULT_H) >= uv[1] >= DEFAULT_H:
            return (
                    (self.derivative_u(uv + self._vh) - self.derivative_u(uv - self._vh))
                    / 2
                    / DEFAULT_H
            )
        elif uv[1] < DEFAULT_H:
            return (
                    self.derivative_u(uv + self._vh) - self.derivative_u(uv)
            ) / DEFAULT_H
        else:
            return (
                    self.derivative_u(uv) - self.derivative_u(uv - self._vh)
            ) / DEFAULT_H

    def normal(self, uv):
        du = self.derivative_u(uv)
        dv = self.derivative_v(uv)
        n = scalar_cross(du, dv)
        return np.array(n) / scalar_norm(n)

    def plane_at(self, uv):
        orig = self.evaluate(uv)
        du = scalar_unit(self.derivative_u(uv))
        dn = scalar_unit(scalar_cross(du, self.derivative_v(uv)))
        dv = scalar_cross(dn, du)

        # n = evaluate_normal2(
        #    du,
        #    dv,
        #    self.second_derivative_uu(uv),
        #    self.second_derivative_uv(uv),
        #    self.second_derivative_vv(uv),
        # )
        return np.array((orig, du, dv, dn))

    def isocurve_u(self, u0: float, u1: float):
        return self.isocurve(u0, 0.0, u1, 1.0)

    def isocurve_v(self, v0: float, v1: float):
        return self.isocurve(0.0, v0, 1.0, v1)

    def isoline_u(self, u: float):
        orig = np.array([u, 0.])
        direction = np.array([0., 1.])
        crv = lambda t: orig + direction * t
        return CurveOnSurface(self, crv)

    def isoline_v(self, v: float):
        orig = np.array([0., v])
        direction = np.array([1., 0.])
        crv = lambda t: orig + direction * t
        return CurveOnSurface(self, crv)

    def interval(self):
        return (0.0, 1.0), (0.0, 1.0)

    def isocurve(self, u0=-1.0, v0=-1.0, u1=-1.0, v1=-1.0):
        start, end = self.interval()
        kfc = list((*start, *end))
        for i, val in enumerate((u0, v0, u1, v1)):
            if val < min(start):
                continue
            else:
                kfc[i] = val

        origin = np.array(kfc[:2])
        direction = np.array(kfc[2:]) - origin

        crv = lambda t: origin + direction * t
        return CurveOnSurface(self, crv)

    def evaluate(self, uv) -> NDArray[float]:
        ...

    def geodesic_length(self, start_uv, end_uv):

        d_uv = end_uv - start_uv
        #n_d_uv=d_uv/scalar_norm(d_uv)
        params = np.linspace(-1., 1., 100)

        def lnt(t):
            td = (self.evaluate(start_uv + d_uv * (t + 1e-5)) - self.evaluate(start_uv + d_uv * (t - 1e-5))) / 2 / 1e-5
            td /= scalar_norm(td)
            return td

        interpl = interp1d(np.array([quad(lnt, -1., p)[0] for p in params]), params)
        return lambda t: start_uv + d_uv * interpl(t)

    def derivatives(self, uv):

        du_prev = uv - self._uh
        du_next = uv + self._uh
        dv_prev = uv - self._vh
        dv_next = uv + self._vh
        ders = np.empty((2, 3))

        #ders[0]=self.evaluate(uv)
        pt_prev_du = self.evaluate(du_prev)
        pt_prev_dv = self.evaluate(dv_prev)
        pt_next_du = self.evaluate(du_next)
        pt_next_dv = self.evaluate(dv_next)
        ders[0] = (pt_next_du - pt_prev_du) / 2 / DEFAULT_H
        ders[1] = (pt_next_dv - pt_prev_dv) / 2 / DEFAULT_H

        return ders

    def __call__(self, uv) -> NDArray[float]:
        if uv.ndim == 1:
            return self.evaluate(uv)
        else:
            return self.evaluate_multi(uv)

    def build_tree(self, u_count=5, v_count=5):
        self._tree = surface_bvh(self, u_count, v_count)

    @property
    def tree(self):
        if self._tree is None:
            self.build_tree()
        return self._tree


from mmcore.geom.bvh import PQuad, build_bvh


def surface_bvh(surf: Surface, u_count, v_count):
    u_interval, v_interval = surf.interval()
    u = np.linspace(*u_interval, u_count)
    v = np.linspace(*v_interval, v_count)

    quads = []

    for i in range(u_count):
        for j in range(v_count):
            if i == (u_count - 1) or j == (v_count - 1):
                pass
            else:
                q = np.array(([u[i], v[j]], [u[i + 1], v[j]], [u[i + 1], v[j + 1]], [u[i], v[j + 1]]))

                quads.append(PQuad(surf(q), q))

    return build_bvh(quads)


def blossom(b, s, t):
    bs0 = (1 - s) * b[0] + s * b[1]
    # b1t = (1 - t) * b[1] + t * b[2]
    bs1 = (1 - s) * b[1] + s * b[2]
    bst = (1 - t) * bs0 + t * bs1
    # bts = (1 - s) * b0t + s * b1t
    return bst


def evaluate_bilinear(uv, b00, b01, b10, b11):
    return np.array([1 - uv[1], uv[1]]).dot(
        np.array([1 - uv[0], uv[0]]).dot(np.array([[b00, b01], [b10, b11]]))
    )


class BiLinear(Surface):
    #TODO: Оптимизировать evaluate и вывести все производные
    def __init__(self, a, b, c, d):
        super().__init__()
        self.b00, self.b01, self.b11, self.b10 = np.array([a, b, c, d], dtype=float)

        self._m = np.array([[self.b00, self.b01], [self.b10, self.b11]])

    def evaluate(self, uv):
        return np.array([1. - uv[1], uv[1]]).dot(
            np.array([1. - uv[0], uv[0]]).dot(
                self._m
            )
        )


class LinearMap:
    def __init__(self, source, target):
        self.source = np.array(source)
        self.target = np.array(target)
        self.slope = scalar_dot(self.source, self.target)

    def __call__(self, t):
        return self.slope * t

    def __invert__(self):
        return LinearMap(self.target, self.source)

    def inv(self):
        return self.__invert__()


from mmcore.geom import parametric


class Ruled(Surface):

    def __init__(self, c1: Curve, c2: Curve):
        super().__init__()
        self.c1, self.c2 = c1, c2
        self._intervals = np.array([np.array(c1.interval()), np.array(c2.interval())])
        self._remap_u = scalar_dot(np.array((0., 1.)), self._intervals[0])
        self._remap_v = scalar_dot(np.array((0., 1.)), self._intervals[1])
        self._remap_uv = np.array([self._remap_u, self._remap_v])

    def derivative_v(self, uv):
        uc1uc2 = self._remap_uv * uv[0]
        return self.c2.evaluate(uc1uc2[1]) - np.array(self.c1.evaluate(uc1uc2[0]))

    def derivative_u(self, uv):
        uc1uc2 = self._remap_uv * uv[0]
        return uv[1] * self.c2.derivative(uc1uc2[1]) + (1. - uv[1]) * self.c1.derivative(uc1uc2[0])

    def second_derivative_uu(self, uv):
        uc1uc2 = self._remap_uv * uv[0]

        return uv[1] * self.c2.second_derivative(uc1uc2[1]) + (1. - uv[1]) * self.c1.second_derivative(uc1uc2[0])

    def second_derivative_vv(self, uv):
        return np.zeros(3)

    def _remap_param(self, t):
        return self._remap_uv * t

    def evaluate(self, uv):
        uc1uc2 = self._remap_uv * uv[0]

        return (1. - uv[1]) * self.c1.evaluate(uc1uc2[0]) + uv[1] * self.c2.evaluate(uc1uc2[1])


CRuled = parametric.Ruled
PyRuled = Ruled


def _remap_param(self, t, c1, c2):
    return np.interp(t, (0.0, 1.0), self.c1.interval()), np.interp(
        t, (0.0, 1.0), self.c2.interval()
    )


def _remap_interval_ruled(uv, c1, c2):
    return np.interp(uv[0], (0.0, 1.0), c1), np.interp(uv[1], (0.0, 1.0), c2)


def Ruled(c1, c2):
    if hasattr(c1, '__pyx_table__') and hasattr(c2, '__pyx_table__'):
        return CRuled(c1, c2)
    else:
        return PyRuled(c1, c2)


from mmcore.geom.curves.bspline import NURBSpline
from mmcore.geom.curves.cubic import CubicSpline


class Coons(Surface):
    """
        import numpy as np
    def cubic_spline(p0, c0, c1, p1):
        p0, c0, c1, p1 = np.array([p0, c0, c1, p1])
        def inner(t):
            return ((p0 * ((1 - t) ** 3)
                     + 3 * c0 * t * ((1 - t) ** 2)
                     + 3 * c1 * (t ** 2) * (1 - t))
                    + p1 * (t ** 3))
        return np.vectorize(inner, signature='()->(i)')
    from mmcore.geom.surfaces import BiLinear
    a,b,c,d=np.array([
        [
            (-25.632193861977559, -25.887792238151487, -8.9649174298769161),
            (-7.6507873591044131, -28.580781837412534, -4.7727445980947056),
            (3.1180460594601840, -31.620627096247443, 11.245007095153923),
            (33.586827711309354, -30.550809492847861, 0.0)],
        [
            (33.586827711309354, -30.550809492847861, 0.0),
            (23.712213781367616, -20.477792480394431, -13.510455008728185),
            (23.624609526477588, -7.8543655761938815, -12.449036305764146),
            (27.082667168033424, 5.5380493986617410, 0.0)],
       [
            (27.082667168033424, 5.5380493986617410, 0.0),
            (8.6853191615639460, -2.1121318577726527, -10.580957050242024),
            (-3.6677924590213919, -2.9387254504549816, -13.206225703752022),
            (-20.330418684651349, 3.931006353774948, 0.0)],
    [
            (-20.330418684651349, 3.931006353774948, 0.0),
            (-22.086936165417491, -5.8423256715423690, 0.0),
            (-23.428753995169622, -15.855467779623531, -7.9942325520491337),
            (-25.632193861977559, -25.887792238151487, -8.9649174298769161)
        ]
    ])
    spls=cubic_spline(*a),cubic_spline(*b),cubic_spline(*reversed(c)),cubic_spline(*reversed(d))



    from mmcore.geom.surfaces import Coons
    cns=Coons(*spls)

    ress=[]
    for u in np.linspace(0.,1.,10):
        for v in np.linspace(0., 1., 10):
            ress.append(cns.evaluate(np.array([u,v])).tolist())

    """

    def __init__(self, c1, d1, c2, d2):
        super().__init__()
        self.c1, self.c2 = c1, c2
        self.d1, self.d2 = d1, d2
        self.pt0 = np.array(c1.evaluate(c1.interval()[0]))
        self.pt1 = np.array(c1.evaluate(c1.interval()[1]))
        self.pt2 = np.array(c2.evaluate(c2.interval()[1]))
        self.pt3 = np.array(d2.evaluate(d2.interval()[1]))

        #self.xu0, self.xu1, self.x0v, self.x1v = _bl(c1, c2, d1, d2)
        if isinstance(self.c1, ParametricCurve) and isinstance(self.c2, ParametricCurve):
            self._rc = CRuled(self.c1, self.c2)
        #elif isinstance(self.c1, NURBSpline) and isinstance(self.c2, NURBSpline):
        #    self._rc = parametric.RationalRuled(self.c1._spline, self.c2._spline)
        else:
            self._rc = PyRuled(self.c1, self.c2)

        if isinstance(self.d1, ParametricCurve) and isinstance(self.d2, ParametricCurve):
            self._rd = CRuled(self.d2, self.d1)
        #elif isinstance(self.d1, NURBSpline) and isinstance(self.d2, NURBSpline):
        #    self._rd = parametric.RationalRuled(self.d1._spline, self.d2._spline)
        else:
            self._rd = PyRuled(self.d2, self.d1)
        #self._rc, self._rd = Ruled(self.c1, self.c2), Ruled(self.d2, self.d1)

        self._rcd = CBiLinear(self.pt0, self.pt1, self.pt2, self.pt3)
        self._cached_eval = lru_cache(maxsize=None)(self._evaluate)
        #self._uv = np.array([0., 0.])

    def evaluate(self, uv):
        #self.counter.__next__()
        uv = np.array(uv)
        #print(f'{self}.evaluate({uv})')
        return self._rc.evaluate(uv) + self._rd.evaluate(uv[::-1]) - self._rcd.evaluate(uv)

        #return self._cached_eval(*uv)

    def _evaluate(self, u, v):
        uv = np.array([u, v], dtype=float)
        return self._rc.evaluate(uv) + self._rd.evaluate(uv[::-1]) - self._rcd.evaluate(uv)


"""
rc, rd, rcd = Ruled(spls[0], spls[2]), ruled(spls[1], spls[3]), BiLinear(xu0(0), xu0(1), xu1(1), x1v(1))
l = []
dd = []
rcdd = []
ress = []
xu0, xu1, x0v, x1v = bl(*spls)
for i in np.linspace(0., 1., 10):
    for j in np.linspace(0., 1., 10):
        r1, r2, r3 = rc(np.array([i, j])), rd(np.array([j, i])), rcd(np.array([i, j]))
        l.append(r1.tolist())
        dd.append(r2.tolist())
        rcdd.append(r3.tolist())
        ress.append((r1 + r2 - r3).tolist())
"""
