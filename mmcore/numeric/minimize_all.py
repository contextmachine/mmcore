import numpy as np
from scipy.optimize import minimize

from mmcore.geom.vec import norm


def minimize_all(fun, bounds: tuple, tol=1e-5, step=0.001, **kwargs):
    bounds = np.array(bounds)
    ress = []
    funs = []

    def bb(bnds):
        nonlocal ress
        ends = bnds[:, -1]
        starts = bnds[:, 0]
        if (np.abs(ends[0] - starts[0])) <= step:
            return
        else:

            res = minimize(fun, x0=starts, bounds=bnds, **kwargs)
            print(res)
            if res.fun <= tol:
                ress.append(res.x)
                funs.append(res.fun)

            bnds[0, 0] = res.x[0] + (bnds[0, 1] - res.x[0]) / 2

            bb(bnds)

    bb(np.copy(bounds))
    return np.array(ress), np.array(funs)


def intersectiont_point(crv1, crv2, tol=1e-4, step=0.001):
    def fun(t):
        return norm(crv1(t[0]) - crv2(t[1]))

    return minimize_all(fun, bounds=(crv1.interval(), crv2.interval()), tol=tol, step=step)
