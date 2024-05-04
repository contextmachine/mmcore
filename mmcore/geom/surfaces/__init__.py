from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from mmcore.geom.curves.curve import Curve
from mmcore.geom.vec import unit, norm, cross
from mmcore.numeric.fdm import Grad, DEFAULT_H
from mmcore.numeric.numeric import evaluate_normal2


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


class CurveOnSurface(Curve):
    def __init__(self, surf: "Surface", curve: "Curve|Callable"):
        super().__init__()
        self.surf = surf
        self.curve = curve
        self._eval_crv_func = getattr(self.curve, "evaluate", self.curve)

    def plane_at(self, t):
        O = self.evaluate(t)
        T = unit(self.tangent(t))
        uv = self.evaluate_curve(t)
        B = unit(self.surf.normal(uv))
        N = cross(B, T)
        return np.array([O, T, N, B])

    def normal(self, t):
        uv = self.evaluate_curve(t)
        return cross(*unit([self.surf.normal(uv), self.tangent(t)]))

    def evaluate(self, t: float):
        return self.surf.evaluate(self.evaluate_curve(t))

    def evaluate_curve(self, t):
        return self._eval_crv_func(t)[..., :2]

    def interval(self):
        return self.curve.interval()


class Surface:
    def __init__(self):
        super().__init__()
        self.evaluate_multi = np.vectorize(self.evaluate, signature="(i)->(j)")
        self._grad = Grad(self)
        self._interval = np.array(self.interval())
        max_norm = norm(self._interval[:, 1])
        self._uh = np.array([DEFAULT_H, 0.0])
        self._vh = np.array([0.0, DEFAULT_H])
        self._plane_at_multi = np.vectorize(self.plane_at, signature="(i)->(j)")

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
        return evaluate_normal2(
            self.derivative_u(uv),
            self.derivative_v(uv),
            self.second_derivative_uu(uv),
            self.second_derivative_uv(uv),
            self.second_derivative_vv(uv),
        )

    def plane_at(self, uv):
        orig = self.evaluate(uv)
        du = unit(self.derivative_u(uv))
        dn = unit(cross(du, self.derivative_v(uv)))
        dv = cross(dn, du)

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

    def interval(self):
        return (0.0, 1.0), (0.0, 1.0)

    def isocurve(self, u0=-1.0, v0=-1.0, u1=-1.0, v1=-1.0):
        kfc = [0.0, 0.0, 1.0, 1.0]
        for i, val in enumerate((u0, v0, u1, v1)):
            if val < 0.0:
                continue
            else:
                kfc[i] = val

        origin = np.array(kfc[:2])
        direction = np.array(kfc[2:]) - origin

        crv = lambda t: origin + direction * t
        return CurveOnSurface(self, crv)

    def evaluate(self, uv) -> NDArray[float]:
        ...

    def __call__(self, uv) -> NDArray[float]:
        if uv.ndim == 1:
            return self.evaluate(uv)
        else:
            return self.evaluate_multi(uv)


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
    def __init__(self, a, b, c, d):
        super().__init__()
        self.b00, self.b01, self.b11, self.b10 = np.array([a, b, c, d], dtype=float)

    def evaluate(self, uv):
        return np.array([1 - uv[1], uv[1]]).dot(
            np.array([1 - uv[0], uv[0]]).dot(
                np.array([[self.b00, self.b01], [self.b10, self.b11]])
            )
        )


class Ruled(Surface):
    def __init__(self, c1, c2):
        super().__init__()
        self.c1, self.c2 = c1, c2
        self._intervals = np.array([c1.interval(), c2.interval()])

    def _remap_param(self, t):
        return np.interp(t, (0.0, 1.0), self.c1.interval()), np.interp(
            t, (0.0, 1.0), self.c2.interval()
        )

    def evaluate(self, uv):
        uc1, uc2 = self._remap_param(uv[0])

        return (1 - uv[1]) * self.c1(uc1) + uv[1] * self.c2(uc2)


def _remap_param(self, t, c1, c2):
    return np.interp(t, (0.0, 1.0), self.c1.interval()), np.interp(
        t, (0.0, 1.0), self.c2.interval()
    )


def _remap_interval_ruled(uv, c1, c2):
    return np.interp(uv[0], (0.0, 1.0), c1), np.interp(uv[1], (0.0, 1.0), c2)


def ruled(c1, c2):
    xu0, xu1 = lambda u: c1(u), lambda u: c2(u)
    c1i = c1.interval()
    c2i = c2.interval()

    def inner(uv):
        uv = _remap_interval_ruled(uv, c1i, c2i)
        return (1 - uv[1]) * xu0(uv[0]) + uv[1] * xu1(uv[0])

    return inner


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

        self.xu0, self.xu1, self.x0v, self.x1v = _bl(c1, c2, d1, d2)
        self._rc, self._rd = Ruled(self.c1, self.c2), Ruled(self.d2, self.d1)

        self._rcd = BiLinear(self.xu0(0), self.xu0(1), self.xu1(1), self.x1v(1))

    def evaluate(self, uv):
        return self._rc(uv) + self._rd(np.array([uv[1], uv[0]])) - self._rcd(uv)


def _bl(c1, c2, d1, d2):
    return (
        lambda u: c1(np.interp(u, (0.0, 1.0), c1.interval())),
        lambda u: c2(np.interp(u, (0.0, 1.0), c2.interval())),
        lambda v: d1(np.interp(v, (0.0, 1.0), d1.interval())),
        lambda v: d2(np.interp(v, (0.0, 1.0), d2.interval())),
    )


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
