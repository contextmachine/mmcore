from mmcore.numeric.intersection.ssx._ssi import improve_uv
from scipy.integrate import solve_ivp
import numpy as np

def pull_curve(surf, curve):
    """
    from numpy._typing import NDArray

from mmcore.geom.surfaces import Surface
#FIXME Что то на поверхности, improve_uv или еще где то рядом имеет серьезный баг из-за которого марш просто прекращается в как либо точке. Что на спиральках что здесь
class Sphere(Surface):
    def evaluate(self, uv) -> NDArray[float]:
        u,v=uv
        x = np.sin(u) * np.cos(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(u)
        return np.array([x, y, z])
    def interval(self):
        return np.array([(0.,np.pi*2),(-np.pi,0.0)])

    :param surf:
    :param curve:
    :return:
    """
    tmin,tmax=curve.interval()
    start=curve.evaluate(tmin)
    uv0=surf.inversion(start)

    def curve_derivative(t, p):
        dp_dt = curve.derivative(t)
        return dp_dt
    def orthogonal_projective_tensor(uv):
        dS_du, dS_dv = surf.derivatives(uv)
        K = np.array([[np.dot(dS_du, dS_du), np.dot(dS_du, dS_dv)],
                      [np.dot(dS_dv, dS_du), np.dot(dS_dv, dS_dv)]])
        return K

    def differential_equation(t, uv):
        K = orthogonal_projective_tensor(uv)
        dS_du, dS_dv = surf.derivatives(uv)
        dp_dt = curve_derivative(t, uv)
        uvd=improve_uv(  dS_du,  dS_dv, surf.evaluate(uv),surf.evaluate(uv)+ dp_dt )
        #du_dt = np.linalg.solve(K, dp_dt[:2])
        return uvd

    sol = solve_ivp(differential_equation, ( tmin,5.), uv0, method='RK45',max_step=0.01)
    return sol
