from __future__ import annotations

from mmcore.geom.vec import norm_sq, cross, norm, unit, gram_schmidt
from scipy.integrate import quad
import numpy as np

from mmcore.numeric.routines import divide_interval
from mmcore.numeric.divide_and_conquer import recursive_divide_and_conquer_min, recursive_divide_and_conquer_max


def plane_on_curve(O, T, D2):
    N = unit(gram_schmidt(T, D2))
    B = cross(T, N)
    return np.array([O, T, N, B])


def normal_at(D1, D2):
    N = unit(gram_schmidt(unit(D1), D2))
    return N


def evaluate_tangent(D1, D2):
    d1 = np.linalg.norm(D1)
    if d1 == 0.0:
        d1 = np.linalg.norm(D2)
        T = D2 / d1 if d1 > 0.0 else np.zeros(D2.shape)
    else:
        T = D1 / d1
    return T, bool(d1)


evaluate_tangent_vec = np.vectorize(evaluate_tangent, signature='(i),(i)->(i),()')


def evaluate_length(first_der, t0: float, t1: float, **kwargs):
    """
  """

    def ds(t):
        return norm(first_der(t))

    return quad(ds, t0, t1, **kwargs)


evaluate_length_vec = np.vectorize(evaluate_length, excluded=[0], signature='(),()->(),()')

import math


def calculate_curvature2d( dx, dy, ddx, ddy):
    numerator = abs(dx * ddy - dy * ddx)
    denominator = math.pow((dx**2 + dy**2), 1.5)
    curvature = numerator / denominator
    return numerator, denominator, curvature

def evaluate_curvature(D1, D2) -> tuple[np.ndarray, np.ndarray, bool]:
    d1 = np.linalg.norm(D1)

    if d1 == 0.0:
        d1 = np.linalg.norm(D2)
        if d1 > 0.0:
            T = D2 / d1
        else:
            T = np.zeros_like(D2)
        K = np.zeros_like(D2)
        rc = False
    else:
        T = D1 / d1
        negD2oT = -np.dot(D2, T)
        d1 = 1.0 / (d1 * d1)
        K = d1 * (D2 + negD2oT * T)
        rc = True

    return T, K, rc


evaluate_curvature_vec = np.vectorize(evaluate_curvature, signature='(i),(i)->(i),(i),()')


def evaluate_jacobian(ds_o_ds, ds_o_dt, dt_o_dt):
    a = ds_o_ds * dt_o_dt
    b = ds_o_dt * ds_o_dt
    det = a - b;
    if ds_o_ds <= dt_o_dt * np.finfo(float).eps or dt_o_dt <= ds_o_ds * np.finfo(float).eps:
        # One of the partials is (numerically) zero w.r.t. the other partial - value of det is unreliable
        rc = False
    elif abs(det) <= max(a, b) * np.sqrt(np.finfo(float).eps):
        # Du and Dv are (numerically) (anti) parallel - value of det is unreliable.
        rc = False
    else:
        rc = True

    return det, rc


def evaluate_normal(Du, Dv, Duu, Duv, Dvv, limit_dir=None):
    DuoDu = norm_sq(Du)

    DuoDv = Du * Dv

    DvoDv = norm_sq(Dv)
    det, success = evaluate_jacobian(DuoDu, DuoDv, DvoDv)
    if success:
        return np.cross(Du, Dv)
    else:
        a, b = {
            2: [-1.0, 1.0],
            3: [-1.0, -1.0],
            4: [1.0, -1.0],
        }.get(limit_dir, [1.0, 1.0])
        V = a * Duv + b * Dvv
        Av = cross(Du, V)
        V = a * Duu + b * Duv
        Au = cross(V, Dv)
        N = Av + Au
        N = N / np.linalg.norm(N)
        return N


def nurbs_bound_points(curve, tol=1e-5):
    """
    Returns a array of parameters whose evaluation gives you a set of points at least sufficient
    for correct estimation of the AABB(Axis-Aligned Bounding Box) of the curve.
    Also the set contains parameters of all extrema of the curve,
    but does not guarantee that the curve is extreme in all parameters.
    """

    def t_x(t):
        return -curve(t)[0]

    def t_y(t):
        return -curve(t)[1]

    def t_z(t):
        return -curve(t)[2]

    t_values = []

    def solve_interval(f, bounds):
        f_min, _ = recursive_divide_and_conquer_min(f, bounds, tol)
        f_max, _ = recursive_divide_and_conquer_max(f, bounds, tol)
        return f_min, f_max

    curve_start, curve_end = curve.interval()
    for start, end in divide_interval(curve_start, curve_end, step=1.):
        t_values.extend(solve_interval(t_x, (start, end)))
        t_values.extend(solve_interval(t_y, (start, end)))
        t_values.extend(solve_interval(t_z, (start, end)))

    return np.unique(np.array([curve_start, *t_values, curve_end]))
