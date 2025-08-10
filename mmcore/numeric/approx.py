import math

import numpy as np

from mmcore.geom._nurbs_eval import NURBSCurveTuple
from mmcore.numeric import compute_parametric_curvature_tolerance_curve
from notes.offset import evaluate_nurbs_curve


def chord_length(R, h):
    return 2 * np.sqrt(2 * R * h - (h * h))


def arc_length_from_chord_height(chord_len, height):
    c = chord_len
    h = height
    a = c / 2.0
    R = (a*a + h*h) / (2.0*h)
    theta = 2.0 * math.asin(a / R)  # in radians
    return R * theta


def chord_height(radius: float, chord_length: float) -> float:
    """
    Compute the sagitta (height) of a chord in a circle.

    Parameters
    ----------
    radius : float
        Circle radius (must be > 0).
    chord_length : float
        Length of the chord (must satisfy 0 <= chord_length <= 2*radius).

    Returns
    -------
    float
        The sagitta h = r - sqrt(r^2 - (c/2)^2).
    """
    if not (np.isfinite(radius) and np.isfinite(chord_length)):
        raise ValueError("radius and chord_length must be finite numbers.")
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    if chord_length < 0:
        raise ValueError("chord_length must be >= 0.")
    if chord_length > 2 * radius:
        raise ValueError("chord_length cannot exceed the diameter (2 * radius).")

    half = chord_length / 2.0
    # clamp inside the sqrt to avoid tiny negative due to floating-point roundoff
    inside = max(0.0, radius * radius - half * half)
    return radius - np.sqrt(inside)


def adaptive_curve_sampler(crv:NURBSCurveTuple, spt:float=1e-3):
    tmin,tmax=crv.interval()
    t_current=tmin
    params=[t_current]
    evals=[]
    duu=[]
    ll=[]
    while t_current<tmax:

        c_eval=evaluate_nurbs_curve(crv,t_current, d_order=2)
        if len(evals)>0:
            l=np.linalg.norm(evals[-1]['C'] -   c_eval['C'])
            ll.append(l)
        evals.append(c_eval)
        du= compute_parametric_curvature_tolerance_curve(c_eval["C1"], c_eval["C2"], spt)
        t_current=np.clip(t_current+du,tmin,tmax)
        duu.append(du)

        params.append(t_current)

    c_eval = evaluate_nurbs_curve(crv, t_current, d_order=2)
    l = np.linalg.norm(evals[-1]["C"] - c_eval["C"])
    ll.append(l)
    evals.append(c_eval)

    return params,duu,evals,ll


def adaptive_curve_sampler_safe(crv, spt, max_param_step_fraction=1/128, max_points=500_000):
    """
    March once so each chord deviates by ~spt (sagitta) using your curvature-based
    stepper. Includes a fallback when κ≈0 so we never return inf.
    """
    tmin, tmax = crv.interval()
    t = tmin
    params = [t]
    du_list = []
    s_list = []
    evals = []

    # Parametric cap to avoid huge jumps at inflections / κ≈0
    du_cap = max_param_step_fraction * (tmax - tmin)
    tiny = np.finfo(float).eps

    n_pts = 0
    while t < tmax - 10*np.finfo(float).eps:
        ce = evaluate_nurbs_curve(crv, t, d_order=2)  # {"C","C1","C2"}
        n_pts += 1
        if n_pts > max_points:
            raise RuntimeError("Too many points; possible stagnation. Increase spt or max_points.")

        C0, C1, C2 = ce["C"], ce["C1"], ce["C2"]
        evals.append(ce)

        du = compute_parametric_curvature_tolerance_curve(C1, C2, spt)
        if not np.isfinite(du) or du <= 0:
            # Fallback: step by a small param cap using local speed
            du = du_cap

        # Don't overshoot
        du = min(du, tmax - t)
        du = max(du, tiny)

        # Arc-length estimate for this step (consistent with your derivation)
        speed = np.linalg.norm(C1)
        s_i = speed * du

        du_list.append(du)
        s_list.append(s_i)

        t += du
        params.append(t)

    # Ensure last sample is at tmax
    ce_end = evaluate_nurbs_curve(crv, tmax, d_order=2)
    evals.append(ce_end)

    return params, du_list, evals, s_list


def circular_segment_area_from_kappa_and_s(kappa_mag, s, small_theta=1e-3):
    """
    Return the (positive) area between a circular arc (constant curvature kappa_mag)
    of length s and its chord. Uses exact formula for general theta and a stable
    series for small theta.
    """
    if kappa_mag <= 0 or s <= 0:
        return 0.0
    theta = kappa_mag * s
    if theta < small_theta:
        # A = κ s^3/12 - κ^3 s^5/240 + O(s^7)
        return (kappa_mag * s**3)/12.0 - (kappa_mag**3 * s**5)/240.0
    # Exact: A = (θ - sin θ) / (2 κ^2)
    return (theta - np.sin(theta)) / (2.0 * kappa_mag**2)


def signed_curvature(C1, C2, n):
    """
    kappa_signed = ((C1 x C2) · n) / ||C1||^3   (planar curve embedded in 3D)
    """
    C1 = np.asarray(C1, float); C2 = np.asarray(C2, float); n = np.asarray(n, float)
    spd = np.linalg.norm(C1)
    if spd == 0: return 0.0
    k_mag = np.linalg.norm(np.cross(C1, C2)) / (spd**3)
    sgn   = np.sign(np.dot(np.cross(C1, C2), n))
    return float(sgn * k_mag)
