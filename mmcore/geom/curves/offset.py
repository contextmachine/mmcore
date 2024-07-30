import numpy as np

from mmcore.geom.curves.curve import Curve

from mmcore.geom.surfaces import CurveOnSurface
from mmcore.numeric.vectors import det,solve2x2


def _improve_uv(du, dv, xyz_old, xyz_better):
    dxdu, dydu, dzdu = du
    dxdv, dydv, dzdv = dv

    delta = xyz_better - xyz_old

    xy = np.array([[dxdu, dxdv], [dydu, dydv]]), [delta[0], delta[1]]
    xz = np.array([[dxdu, dxdv], [dzdu, dzdv]]), [delta[0], delta[2]]
    yz = np.array([[dydu, dydv], [dzdu, dzdv]]), [delta[1], delta[2]]

    max_det = max([xy, xz, yz], key=lambda Ab: det(Ab[0]))
    res = np.zeros(2)
    solve2x2(max_det[0], np.array(max_det[1]), res)
    return res


class OffsetCurve(Curve):
    curve: Curve
    _offset: float

    def __init__(self, curve: Curve, offset=1.0):
        super().__init__()
        self.curve = curve
        self._offset = offset

    def evaluate(self, t):
        n = np.array(self.curve.plane_at(t))
        return n[0] + n[2] * self.offset

    @property
    def offset(self):
        return self.offset

    @offset.setter
    def offset(self, v):
        self._offset = v
        self.invalidate_cache()


class OffsetOnSurface(Curve):
    curve: CurveOnSurface
    offset: float

    def __init__(self, curve: CurveOnSurface, offset=1.0):
        super().__init__()
        self.curve=curve
        self.offset=offset
        if self.curve.surf.boundary is None:
            self.curve.surf.build_boundary()
        start,end=self.curve.curve.interval()
        #self.ends=curve_ppi(self.evaluate_uv,self.curve.surf.boundary,bounds1=(start=0.2,end))


    def evaluate_uv(self, t):
        pt = self.evaluate_offset(t)
        xyz = self.curve.evaluate(t)
        uv_old = self.curve.curve.evaluate(t)[:-1]
        du = self.curve.surf.derivative_u(uv_old)
        dv = self.curve.surf.derivative_v(uv_old)

        return uv_old + _improve_uv(du, dv, xyz, pt)

    def evaluate(self, t):
        return self.curve.surf.evaluate(self.evaluate_uv(t))

    def evaluate_offset(self, t):
        n = np.array(self.curve.plane_at(t))
        return n[0] + n[2] * self.offset
