import numpy as np
from scipy.interpolate import interp1d

from mmcore.geom.curves.curve import Curve, ArcLengthParameterization

from mmcore.geom.surfaces import CurveOnSurface
from mmcore.numeric.vectors import det,solve2x2

from mmcore.numeric import fdm, evaluate_length


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
def arc_length_normal_reparameterization(curve,surface:'Surface',offset):
    count=100
    start, end = curve.interval()
    params = np.linspace(start, end, count)
    n = curve.planes_at(params)
    lvals=np.zeros((count,))

    for i,v in enumerate(params):
        P=n[i,0,...]
        N=n[i,-1,...]

        iso=   ArcLengthParameterization(surface.isocurve(*P[:-2],*N[:-2]))
        iso.evaluate_length(offset)
        lvals[i]=curve.evaluate_length((start, float(v)))






    return interp1d(np.linspace(0.,1.,count),interp1d(lvals,params)(np.linspace(lvals.min(),lvals.max(),count)))


from scipy.interpolate import interp1d
from scipy.integrate import quad_vec
from mmcore.numeric.numeric import evaluate_length
from mmcore.numeric.fdm import fdm


def geodesic_length(self, start_uv, end_uv, dist):
    d_uv = end_uv - start_uv
    # n_d_uv=d_uv/scalar_norm(d_uv)
    params = np.linspace(-dist, dist, 100)

    def lnt(t):
        return self.evaluate(d_uv * t + start_uv)

    ldr = fdm(lnt)
    ddr = lambda t: ldr(t)
    d_uv1 = d_uv.reshape((1, 2))
    start_uv1 = start_uv.reshape((1, 2))
    interpl = interp1d(np.array([evaluate_length(ddr, 0, p)[0] for p in params]), params)
    return lambda t: start_uv1 + d_uv1.reshape((1, 2)) * interpl(t)


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





    def evaluate(self, t):
        n = np.array(self.curve.curve.plane_at(t))
        # n = np.array(self.curve.normal(t))
        gll = geodesic_length(self.curve.surf, n[0][:2], n[0][:2] + n[2][:2], self.offset*2)
        return self.curve.surf.evaluate( gll(self.offset)[0])


