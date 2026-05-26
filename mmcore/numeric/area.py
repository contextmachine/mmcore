from typing import NamedTuple

import numpy as np

from mmcore.numeric.approx import adaptive_curve_sampler
from notes.offset import evaluate_nurbs_curve


def best_fit_plane_frame(P):
    """
    SVD-based best-fit plane for 3D points.
    Returns origin o, orthonormal (e1,e2), and unit normal n.
    """
    P = np.asarray(P, float)
    o = P.mean(axis=0)
    Q = P - o
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    n  = Vt[-1]          # normal
    e1 = Vt[0]; e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1); e2 /= np.linalg.norm(e2)
    # Ensure right-handed frame
    if np.dot(np.cross(e1, e2), n) < 0: e2 = -e2
    return o, e1, e2, n


def project_to_frame(P, o, e1, e2):
    Q = P - o
    return np.column_stack([Q @ e1, Q @ e2])


class SimpsonReport(NamedTuple):
    area: float
    evals: int
    max_depth_reached: int
    planarity_rms: float
    planarity_max: float


def nurbs_curve_area(crv, *,
                     area_tol: float = 1e-6,
                     max_depth: int = 40,
                     max_evals: int = 2_000_000) -> SimpsonReport:
    """
    Signed area of a flat closed NURBS curve via adaptive Simpson on g(t)=0.5(x*y' - y*x').
    Reuses evaluations via a cache; projects curve/derivatives to a best-fit plane.
    """
    # 1) Lock the plane once from a light sample (any d_order=0 evaluator will do)
    tmin, tmax = crv.interval()
    Ts,duu,evals,*_= adaptive_curve_sampler(crv, 1e-3)
    P0 = np.array([e["C"] for e in evals])
    o, e1, e2, n = best_fit_plane_frame(P0)
    dists = (P0 - o) @ n
    planarity_rms = float(np.sqrt(np.mean(dists**2)))
    planarity_max = float(np.max(np.abs(dists)))

    # 2) Cached evaluations of g(t)
    cache = {}  # t -> g(t); separate stores avoid float rounding surprises
    evals = 0

    def g_of_t(t: float) -> float:
        nonlocal evals
        # Use exact 't' objects from recursive splits; midpoints are reproducible floats
        if t in cache:
            return cache[t]
        e = evaluate_nurbs_curve(crv, float(t), d_order=1)  # C0 and C1 are enough
        C0, C1 = e["C"], e["C1"]
        xy = project_to_frame(C0[None, :], o, e1, e2)[0]
        xp = float(C1 @ e1); yp = float(C1 @ e2)
        g = 0.5 * (xy[0] * yp - xy[1] * xp)
        cache[t] = g
        evals += 1
        if evals > max_evals:
            raise RuntimeError("Exceeded max evaluations; raise max_evals or loosen area_tol.")
        return g

    def S(a, b, fa, fm, fb):
        h = b - a
        return (h / 6.0) * (fa + 4.0 * fm + fb)

    max_depth_hit = 0

    def asr(a, b, fa, fm, fb, Sab, tol, depth):
        """
        Adaptive Simpson recursion with the standard error estimator:
        err ≈ |(S_left + S_right) - S_ab| / 15.
        We return the Richardson-improved value S_left + S_right + (delta)/15
        when within tolerance.
        """
        nonlocal max_depth_hit
        max_depth_hit = max(max_depth_hit, depth)
        m  = 0.5 * (a + b)
        ml = 0.5 * (a + m)
        mr = 0.5 * (m + b)

        fml = g_of_t(ml)
        fmr = g_of_t(mr)

        S_left  = S(a, m,  fa, fml, fm)
        S_right = S(m, b,  fm, fmr, fb)
        delta = S_left + S_right - Sab

        if depth >= max_depth or abs(delta) <= 15.0 * tol:
            # Richardson improvement
            return S_left + S_right + delta / 15.0

        # Recurse with halved tolerances
        left  = asr(a, m, fa, fml, fm, S_left,  tol / 2.0, depth + 1)
        right = asr(m, b, fm, fmr, fb, S_right, tol / 2.0, depth + 1)
        return left + right

    # 3) Kick off
    a, b = float(tmin), float(tmax)
    fa = g_of_t(a)
    fb = g_of_t(b)
    m  = 0.5 * (a + b)
    fm = g_of_t(m)
    Sab = S(a, b, fa, fm, fb)
    area = asr(a, b, fa, fm, fb, Sab, area_tol, 1)

    return SimpsonReport(area=float(area),
                         evals=int(evals),
                         max_depth_reached=int(max_depth_hit),
                         planarity_rms=planarity_rms,
                         planarity_max=planarity_max)
