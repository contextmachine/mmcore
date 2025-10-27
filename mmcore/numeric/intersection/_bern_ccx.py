from itertools import pairwise

import numpy as np
import time

from mmcore.geom._nurbs_eval import nurbs_curve
from mmcore.geom._nurbs_knots import generate_knots,trim_curve
from mmcore.geom.bvh.lbvh import AABB
from mmcore.numeric._aabb import aabb, aabb_intersection
from mmcore.numeric.bern import *
from typing import Iterable, List, Tuple

Rect = Tuple[float, float, float, float]  # (x0, y0, x1, y1)

def _norm_rect(r: Rect) -> Rect:
    x0, y0, x1, y1 = r
    if not all(map(lambda z: isinstance(z, (int, float)), r)):
        raise ValueError("Rectangle coordinates must be numbers.")
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return x0, y0, x1, y1

def _area(r: Rect) -> float:
    x0, y0, x1, y1 = r
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)

def _clip_to_outer(r: Rect, outer: Rect) -> Rect:
    x0, y0, x1, y1 = r
    s0, t0, s1, t1 = outer
    cx0, cy0 = max(s0, x0), max(t0, y0)
    cx1, cy1 = min(s1, x1), min(t1, y1)
    return (cx0, cy0, cx1, cy1)

def _merge_intervals(intervals: List[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
    """
    Merge 1D closed intervals [a,b] with numerical tolerance.
    """
    if not intervals:
        return []
    intervals.sort()
    merged = []
    a0, b0 = intervals[0]
    for a, b in intervals[1:]:
        if a <= b0 + eps:           # overlaps or touches within eps
            b0 = max(b0, b)
        else:
            if b0 - a0 > eps:
                merged.append((a0, b0))
            a0, b0 = a, b
    if b0 - a0 > eps:
        merged.append((a0, b0))
    return merged

def _complement_intervals(intervals: List[Tuple[float, float]], base: Tuple[float, float], eps: float) -> List[Tuple[float, float]]:
    """
    Given merged cover intervals within [B0,B1], return uncovered segments inside [B0,B1].
    """
    B0, B1 = base
    if B1 - B0 <= eps:
        return []
    if not intervals:
        return [(B0, B1)]
    res = []
    y = B0
    for a, b in intervals:
        a = max(a, B0)
        b = min(b, B1)
        if a - y > eps:
            res.append((y, a))
        y = max(y, b)
    if B1 - y > eps:
        res.append((y, B1))
    return res

def rect_difference_union(outer: Rect, inners: Iterable[Rect], eps: float = 1e-12) -> List[Rect]:
    """
    Compute OUTER \ (⋃ inners) as a list of disjoint rectangles.

    - outer: (s0,t0,s1,t1), axis-aligned rectangle.
    - inners: iterable of (u0,v0,u1,v1), each possibly overlapping and all within outer (they will be clipped).
    - eps: tolerance for floating comparisons and degenerate filtering.

    Returns a list of rectangles that exactly cover the set difference, with no overlaps and no gaps.
    """
    s0, t0, s1, t1 = _norm_rect(outer)
    if _area((s0, t0, s1, t1)) <= eps:
        return []

    # Clip inners to the outer and discard empty
    clipped: List[Rect] = []
    for r in inners:
        r = _norm_rect(r)
        r = _clip_to_outer(r, (s0, t0, s1, t1))
        if _area(r) > eps:
            clipped.append(r)

    # If there are no inners after clipping, the difference is just the outer.
    if not clipped:
        return [(s0, t0, s1, t1)]

    # Build x-slab boundaries from outer edges and all inner edges
    x_edges = {s0, s1}
    for x0, _, x1, _ in clipped:
        x_edges.add(max(s0, min(s1, x0)))
        x_edges.add(max(s0, min(s1, x1)))
    xs = sorted(x_edges)

    # Helper to quantize keys to make slab-to-slab coalescing stable
    def _q(v: float) -> float:
        # scale by an exponent to reduce tiny numeric drift in keys
        return round(v / max(1.0, abs(s1 - s0), abs(t1 - t0)) / eps)  # coarse quantization for dict keys

    results: List[Rect] = []
    open_by_y = {}  # key=(qy0,qy1) -> [x_start, y0, y1]
    last_xR = None  # right edge of previous slab

    for i in range(len(xs) - 1):
        xL, xR = xs[i], xs[i + 1]
        if xR - xL <= eps:
            continue

        # Collect y-intervals covered inside this slab by any inner whose x-span covers [xL,xR]
        covered_y: List[Tuple[float, float]] = []
        for ux0, uy0, ux1, uy1 in clipped:
            if ux0 <= xL + eps and ux1 >= xR - eps:  # rect fully spans this slab in X
                covered_y.append((uy0, uy1))

        # Merge the covered intervals and compute complement within [t0,t1]
        merged_cov = _merge_intervals(covered_y, eps)
        uncovered = _complement_intervals(merged_cov, (t0, t1), eps)

        # Close any open rectangles that are not continued in this slab
        if last_xR is not None and abs(last_xR - xL) <= max(eps, 0.0):
            current_keys = set()
            for y0, y1 in uncovered:
                key = (_q(y0), _q(y1))
                current_keys.add(key)
            to_close = [k for k in list(open_by_y.keys()) if k not in current_keys]
            for k in to_close:
                x_start, y0o, y1o = open_by_y.pop(k)
                if (last_xR - x_start) > eps and (y1o - y0o) > eps:
                    results.append((x_start, y0o, last_xR, y1o))

        # Start or continue rectangles for uncovered segments in this slab
        for y0, y1 in uncovered:
            if (y1 - y0) <= eps:
                continue
            key = (_q(y0), _q(y1))
            if key not in open_by_y:
                # start a new rectangle at xL
                open_by_y[key] = [xL, y0, y1]
            # else: continuing from previous slab; nothing to change (we'll extend when closing)

        last_xR = xR

    # Close any rectangles still open at the end with x1 = s1
    for k, (x_start, y0o, y1o) in open_by_y.items():
        if (s1 - x_start) > eps and (y1o - y0o) > eps:
            results.append((x_start, y0o, s1, y1o))

    # Final pass: drop any degenerate pieces introduced by tolerance
    results = [r for r in results if _area(r) > eps]
    return results

# ---------- Bézier evaluation & derivatives ----------
def de_casteljau_split(ctrl, prms=None):  # (n+1,d)
    
    
    n = ctrl.shape[0] - 1
    if prms is None:
        prms=[0.5]*n
    pyr = [ctrl.copy()]
    for _ in range(1, n + 1):
        prev = pyr[-1]
        pyr.append(prms[_-1] * (prev[:-1] + prev[1:]))
    left  = np.stack([p[0]    for p in pyr], axis=0)
    right = np.stack([p[-1]   for p in pyr[::-1]], axis=0)
    return left, right

def eval_bezier(P, t):
    Q = P.copy()
    n = P.shape[0] - 1
    for _ in range(n):
        Q = (1.0 - t) * Q[:-1] + t * Q[1:]
    return Q[0]
from mmcore.geom._nurbs_ders import derivative_nurbs
def deriv_ctrl(P):  # first derivative control net
    n = P.shape[0] - 1
    return n * (P[1:] - P[:-1])



def eval_bezier_deriv(P, t):  # first derivative at t
    return eval_bezier(deriv_ctrl(P), t)
from mmcore.numeric.intersection.ccx._bex_ccx2 import bernstein_distance_squared_net
# ---------- Distance net envelope (pruning) ----------
def distance_squared_net(P, Q):
    #diff = P[:, None, :] - Q[None, :, :]
    return bernstein_distance_squared_net( P, Q)
def bern_no_sign_change(coeffs):
    return (np.min(coeffs) > 0.0) or (np.max(coeffs) < 0.0)
def bernstein_envelope_min(dnet):
    return dnet.min()


# ---------- Vector system G(u,v)=C1(u)-C2(v)=0 ----------
def G_and_J(C1, C2, u, v):
    p1 = eval_bezier(C1, u);
    t1 = eval_bezier_deriv(C1, u)
    p2 = eval_bezier(C2, v);
    t2 = eval_bezier_deriv(C2, v)
    G  = p1 - p2                           # d-vector
    J  = np.stack([t1, -t2], axis=1)       # d x 2
    return G, J


def newton_project_G0(C1, C2, u0, v0, tol=1e-12, it=30, lm_damp=1e-12):
    """Levenberg–Marquardt corrector to G(u,v)=0; clamps to [0,1]^2."""
    u, v = float(u0), float(v0)
    for _ in range(it):
        G, J = G_and_J(C1, C2, u, v)          # G in R^d, J in R^{d x 2}
        JT = J.T
        A  = JT @ J + lm_damp * np.eye(2)     # 2x2
        b  = -JT @ G                           # 2
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta = np.zeros(2)
        # line search with clamping
        step = 1.0
        for _ls in range(8):
            un = np.clip(u + step * delta[0], 0.0, 1.0)
            vn = np.clip(v + step * delta[1], 0.0, 1.0)
            if np.linalg.norm(G_and_J(C1, C2, un, vn)[0]) <= np.linalg.norm(G):
                u, v = un, vn
                break
            step *= 0.5
        if np.linalg.norm(G) < tol:
            break
        if step < 1e-6 and np.linalg.norm(delta) < 1e-10:
            break
    G, J = G_and_J(C1, C2, u, v)
    return u, v, G, J

# ---------- Classification via J’s rank ----------
def classify_contact(J, sv_thresh=1e-8):
    # singular values of d x 2 J; for curves in 2D/3D: rank 2 => isolated; rank 1 => overlap/slip
    s = np.linalg.svd(J, compute_uv=False)
    s_sorted = np.sort(s)[::-1]  # s_max >= s_min
    if s_sorted.shape[0] < 2:
        # degenerate curve derivative; treat as ambiguous
        return {'type': 'ambiguous', 'svals': s_sorted}
    s_max, s_min = s_sorted[0], s_sorted[-1]
    if s_min < sv_thresh and s_max > 1e-10:
        return {'type': 'overlap', 'svals': (s_max, s_min)}
    if s_min >= sv_thresh:
        return {'type': 'isolated', 'svals': (s_max, s_min)}
    return {'type': 'ambiguous', 'svals': (s_max, s_min)}

# ---------- Nullspace direction (predictor) ----------
def nullspace_dir(J):
    # Return unit vector n in R^2 spanning ker(J) (d x 2)
    # Solve min ||J n|| s.t. ||n||=1 via SVD: right singular vector for smallest s
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    n = Vt[-1, :]  # 2-vector
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        return np.array([0.0, 0.0])
    return n / n_norm



# --- Second derivative utilities (add once) ---
def second_deriv_ctrl(P):
    n = P.shape[0] - 1
    if n < 2:
        return np.zeros((1, P.shape[1]), dtype=P.dtype)
    return n * (n - 1) * (P[2:] - 2 * P[1:-1] + P[:-2])

def eval_bezier_second_deriv(P, t):
    return eval_bezier(second_deriv_ctrl(P), t)

def curvature_and_speed(C, u):
    t = eval_bezier_deriv(C, u)
    s = np.linalg.norm(t)
    if s < 1e-16:
        return 0.0, 0.0, t  # degenerate speed
    tt = eval_bezier_second_deriv(C, u)
    d = C.shape[1]
    if d == 2:
        kappa = abs(t[0]*tt[1] - t[1]*tt[0]) / (s**3 + 1e-30)
    else:  # d == 3
        kappa = np.linalg.norm(np.cross(t, tt)) / (s**3 + 1e-30)
    return kappa, s, t

def bbox_diag_len(C1, C2):
    mins = np.minimum(np.min(C1, axis=0), np.min(C2, axis=0))
    maxs = np.maximum(np.max(C1, axis=0), np.max(C2, axis=0))
    return float(np.linalg.norm(maxs - mins))

def point_line_distance(p, a, b):
    ab = b - a
    denom = np.linalg.norm(ab)
    if denom < 1e-16:
        return np.linalg.norm(p - a)
    if len(p) == 3:
        return np.linalg.norm(np.cross(ab, p - a)) / denom
    else:
        # 2D: |det(ab, ap)| / ||ab||
        return abs(ab[0]*(p[1]-a[1]) - ab[1]*(p[0]-a[0])) / denom

def append_with_decimation(uv_path, xyz_path, uv_new, x_new, angle_tol, sag_tol):
    if len(xyz_path) < 2:
        uv_path.append(uv_new); xyz_path.append(x_new); return
    a, b = xyz_path[-2], xyz_path[-1]
    c = x_new
    ab = b - a; bc = c - b
    lab = np.linalg.norm(ab); lbc = np.linalg.norm(bc)
    ang = 0.0
    if lab > 0 and lbc > 0:
        cosang = np.dot(ab, bc) / (lab * lbc)
        cosang = np.clip(cosang, -1.0, 1.0)
        ang = np.arccos(cosang)
    sag = point_line_distance(b, a, c)
    if ang < angle_tol and sag < sag_tol:
        # replace last point (merge)
        uv_path[-1] = uv_new
        xyz_path[-1] = x_new
    else:
        uv_path.append(uv_new); xyz_path.append(x_new)



# --- 1D projectors for boundary events ---------------------------------------
def project_G0_fixed_u(C1, C2, u_fixed, v0, tol=1e-12, it=40, lm_damp=1e-12):
    """
    Solve min ||C1(u_fixed) - C2(v)|| with v in [0,1].
    Returns v, G (residual vector), success(bool).
    """
    p1 = eval_bezier(C1, u_fixed)
    v = float(v0)
    for _ in range(it):
        p2 = eval_bezier(C2, v)
        t2 = eval_bezier_deriv(C2, v)
        G  = p1 - p2
        JTJ = float(np.dot(t2, t2)) + lm_damp
        JTG = -float(np.dot(t2, G))
        dv = JTG / (JTJ + 1e-30)
        step = 1.0
        g0 = np.linalg.norm(G)
        while step > 1e-6:
            vn = np.clip(v + step * dv, 0.0, 1.0)
            gn = np.linalg.norm(p1 - eval_bezier(C2, vn))
            if gn <= g0 + 1e-18:
                v = vn; break
            step *= 0.5
        if np.linalg.norm(p1 - eval_bezier(C2, v)) < tol:
            G = p1 - eval_bezier(C2, v)
            return v, G, True
    G = p1 - eval_bezier(C2, v)
    return v, G, (np.linalg.norm(G) < 5.0 * tol)

def project_G0_fixed_v(C1, C2, v_fixed, u0, tol=1e-12, it=40, lm_damp=1e-12):
    """
    Solve min ||C1(u) - C2(v_fixed)|| with u in [0,1].
    Returns u, G (residual vector), success(bool).
    """
    p2 = eval_bezier(C2, v_fixed)
    u = float(u0)
    for _ in range(it):
        p1 = eval_bezier(C1, u)
        t1 = eval_bezier_deriv(C1, u)
        G  = p1 - p2
        JTJ = float(np.dot(t1, t1)) + lm_damp
        JTG =  float(np.dot(t1, G))
        du = -JTG / (JTJ + 1e-30)
        step = 1.0
        g0 = np.linalg.norm(G)
        while step > 1e-6:
            un = np.clip(u + step * du, 0.0, 1.0)
            gn = np.linalg.norm(eval_bezier(C1, un) - p2)
            if gn <= g0 + 1e-18:
                u = un; break
            step *= 0.5
        if np.linalg.norm(eval_bezier(C1, u) - p2) < tol:
            G = eval_bezier(C1, u) - p2
            return u, G, True
    G = eval_bezier(C1, u) - p2
    return u, G, (np.linalg.norm(G) < 5.0 * tol)

# --- Event helpers ------------------------------------------------------------
def dot_sigma(C1, C2, u, v):
    t1 = eval_bezier_deriv(C1, u)
    t2 = eval_bezier_deriv(C2, v)
    return float(np.dot(t1, t2))

def earliest_boundary_alpha(u, v, du, dv):
    """
    Return (alpha, which) where which in {'u0','u1','v0','v1'},
    alpha in (0,1], or (None, None) if no boundary inside 0<alpha<=1.
    """
    candidates = []
    if du > 0 and u < 1: candidates.append(((1.0 - u) / du, 'u1'))
    if du < 0 and u > 0: candidates.append(((0.0 - u) / du, 'u0'))
    if dv > 0 and v < 1: candidates.append(((1.0 - v) / dv, 'v1'))
    if dv < 0 and v > 0: candidates.append(((0.0 - v) / dv, 'v0'))
    # keep positive alphas <= 1
    candidates = [(a, w) for (a, w) in candidates if a > 0.0 and a <= 1.0 and np.isfinite(a)]
    if not candidates:
        return None, None
    a, w = min(candidates, key=lambda t: t[0])
    return float(a), w

def sigma_flip_alpha(C1, C2, u, v, du, dv, tol_alpha=1e-12, max_it=40):
    """
    If sigma = sign(t1·t2) flips along (u,v) -> (u+du, v+dv), return alpha in (0,1) where dot = 0.
    Else return None.
    """
    s0 = dot_sigma(C1, C2, u, v)
    s1 = dot_sigma(C1, C2, u + du, v + dv)
    if s0 == 0.0:
        return 0.0
    if s0 * s1 > 0.0:
        return None
    # Bisection on alpha
    lo, hi = 0.0, 1.0
    slo, shi = s0, s1
    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        sm  = dot_sigma(C1, C2, u + mid * du, v + mid * dv)
        if sm == 0.0 or (hi - lo) < tol_alpha:
            return mid
        if slo * sm <= 0.0:
            hi, shi = mid, sm
        else:
            lo, slo = mid, sm
    return 0.5 * (lo + hi)

# --- Event-driven tracer ------------------------------------------------------
def trace_overlap_fast_events(
    C1, C2, u_seed, v_seed,
    sag_tol=None, ds_min=None, ds_max=None,
    angle_tol=1e-3, sv_thresh=1e-8,
    tol_proj=None, snap_eps=1e-12,
    enable_sigma_event=True,
    step_growth=1.5, step_shrink=0.5,
    max_points=10000
):
    """
    Curvature/sag-controlled predictor, event-driven stepping (boundary + sigma flip),
    boundary snapping with 1D projectors, and on-the-fly decimation.
    """
    # --- scales & tolerances
    def bbox_diag_len(C1, C2):
        mins = np.minimum(np.min(C1, axis=0), np.min(C2, axis=0))
        maxs = np.maximum(np.max(C1, axis=0), np.max(C2, axis=0))
        return float(np.linalg.norm(maxs - mins))

    scale = bbox_diag_len(C1, C2)
    if sag_tol is None: sag_tol = max(1e-7, 1e-5 * scale)
    if ds_min  is None: ds_min  = 1e-6 * scale
    if ds_max  is None: ds_max  = 2e-2 * scale
    if tol_proj is None: tol_proj = max(1e-12, 1e-10 * scale)

    # --- initial 2D projection
    u0, v0, G0, J0 = newton_project_G0(C1, C2, u_seed, v_seed, tol=tol_proj)
    if np.linalg.norm(G0) > 1e-7:
        return {'kind': 'none'}

    # --- classify; if not overlap, return isolated
    cls0 = classify_contact(J0, sv_thresh)
    if cls0['type'] != 'overlap':
        x0 = eval_bezier(C1, u0)
        return {'kind': cls0['type'], 'points': [(u0, v0)], 'xyz': [x0],
                'start': 'seed', 'end': 'seed'}

    # --- snap to boundary if very close (exact 0/1)
    def try_snap(u, v):
        snapped = False
        if u <= snap_eps:
            v_new, G, ok = project_G0_fixed_u(C1, C2, 0.0, v, tol=tol_proj)
            if ok: u, v, snapped = 0.0, v_new, True
        elif (1.0 - u) <= snap_eps:
            v_new, G, ok = project_G0_fixed_u(C1, C2, 1.0, v, tol=tol_proj)
            if ok: u, v, snapped = 1.0, v_new, True

        if v <= snap_eps:
            u_new, G, ok = project_G0_fixed_v(C1, C2, 0.0, u, tol=tol_proj)
            if ok: u, v, snapped = u_new, 0.0, True
        elif (1.0 - v) <= snap_eps:
            u_new, G, ok = project_G0_fixed_v(C1, C2, 1.0, u, tol=tol_proj)
            if ok: u, v, snapped = u_new, 1.0, True
        return u, v, snapped

    u0, v0, _ = try_snap(u0, v0)

    uv_path = [(u0, v0)]
    xyz_path = [eval_bezier(C1, u0)]

    # --- utilities from previous reply (reuse)
    def curvature_and_speed(C, u):
        t = eval_bezier_deriv(C, u)
        s = np.linalg.norm(t)
        if s < 1e-16:
            return 0.0, 0.0, t
        # second derivative
        tt = eval_bezier_second_deriv(C, u)
        d = C.shape[1]
        if d == 2:
            kappa = abs(t[0]*tt[1] - t[1]*tt[0]) / (s**3 + 1e-30)
        else:
            kappa = np.linalg.norm(np.cross(t, tt)) / (s**3 + 1e-30)
        return kappa, s, t

    def point_line_distance(p, a, b):
        ab = b - a; denom = np.linalg.norm(ab)
        if denom < 1e-16: return np.linalg.norm(p - a)
        if len(p) == 3:
            return np.linalg.norm(np.cross(ab, p - a)) / denom
        else:
            return abs(ab[0]*(p[1]-a[1]) - ab[1]*(p[0]-a[0])) / denom

    def append_with_decimation(uv_path, xyz_path, uv_new, x_new, angle_tol, sag_tol, force_keep=False):
        if force_keep or len(xyz_path) < 2:
            uv_path.append(uv_new); xyz_path.append(x_new); return
        a, b = xyz_path[-2], xyz_path[-1]; c = x_new
        ab = b - a; bc = c - b
        lab = np.linalg.norm(ab); lbc = np.linalg.norm(bc)
        ang = 0.0
        if lab > 0 and lbc > 0:
            cosang = np.dot(ab, bc) / (lab * lbc)
            cosang = np.clip(cosang, -1.0, 1.0)
            ang = np.arccos(cosang)
        sag = point_line_distance(b, a, c)
        if ang < angle_tol and sag < sag_tol:
            uv_path[-1] = uv_new; xyz_path[-1] = x_new
        else:
            uv_path.append(uv_new); xyz_path.append(x_new)

    # --- march in each direction with events
    def march(direction):
        nonlocal uv_path, xyz_path
        # start end according to direction
        u = uv_path[0][0] if direction < 0 else uv_path[-1][0]
        v = uv_path[0][1] if direction < 0 else uv_path[-1][1]

        # initial step from curvature
        kappa, s1, _ = curvature_and_speed(C1, u)
        if s1 <= 0: ds = ds_min
        else:
            ds = np.sqrt(8.0 * sag_tol / max(kappa, 1e-16))
            ds = float(np.clip(ds, ds_min, ds_max))

        pts = 0
        while pts < max_points:
            # tangents & speeds
            kappa, s1, t1 = curvature_and_speed(C1, u)
            t2 = eval_bezier_deriv(C2, v)
            s2 = float(np.linalg.norm(t2))
            if s1 < 1e-16 or s2 < 1e-16:
                ds = max(ds_min, ds * step_shrink)

            sigma = 1.0 if float(np.dot(t1, t2)) >= 0.0 else -1.0

            # predictor step (spatial -> parameter)
            du = direction * (ds / max(s1, 1e-16))
            dv = sigma * (s1 / max(s2, 1e-16)) * du

            # --- compute events on this step
            a_bnd, which = earliest_boundary_alpha(u, v, du, dv)
            a_sig = None
            if enable_sigma_event:
                a_sig = sigma_flip_alpha(C1, C2, u, v, du, dv)

            # target alpha
            alphas = [1.0]
            labels = ['none']
            if a_bnd is not None: alphas.append(a_bnd); labels.append(which)
            if a_sig is not None and a_sig > 0.0: alphas.append(a_sig); labels.append('sigma')
            idx = int(np.argmin(alphas))
            alpha = float(alphas[idx])
            label = labels[idx]

            # step to event/prediction
            up = float(np.clip(u + alpha * du, 0.0, 1.0))
            vp = float(np.clip(v + alpha * dv, 0.0, 1.0))

            # --- handle boundary event with exact snapping
            if label in ('u0','u1','v0','v1'):
                force_keep = True  # keep boundary points
                if label == 'u0':
                    vnew, G, ok = project_G0_fixed_u(C1, C2, 0.0, vp, tol=tol_proj)
                    if not ok: break
                    u, v = 0.0, vnew
                elif label == 'u1':
                    vnew, G, ok = project_G0_fixed_u(C1, C2, 1.0, vp, tol=tol_proj)
                    if not ok: break
                    u, v = 1.0, vnew
                elif label == 'v0':
                    unew, G, ok = project_G0_fixed_v(C1, C2, 0.0, up, tol=tol_proj)
                    if not ok: break
                    u, v = unew, 0.0
                elif label == 'v1':
                    unew, G, ok = project_G0_fixed_v(C1, C2, 1.0, up, tol=tol_proj)
                    if not ok: break
                    u, v = unew, 1.0

                x = eval_bezier(C1, u)
                if direction > 0:
                    append_with_decimation(uv_path, xyz_path, (u, v), x, angle_tol, sag_tol, force_keep=True)
                else:
                    # insert at front, force keep
                    uv_path.insert(0, (u, v)); xyz_path.insert(0, x)
                pts += 1

                # boundary means we stop in this direction
                break

            # --- otherwise: possibly a sigma flip event (or plain interior step)
            # Corrector (2D)
            uc, vc, Gc, Jc = newton_project_G0(C1, C2, up, vp, tol=tol_proj)
            if np.linalg.norm(Gc) > 5.0 * tol_proj:
                # step too large -> shrink & retry
                ds = max(ds_min, ds * step_shrink)
                if ds <= ds_min * 1.01: break
                continue

            # rank check: if not overlap anymore, try shrinking step
            cls = classify_contact(Jc, sv_thresh)
            if cls['type'] != 'overlap':
                ds = max(ds_min, ds * step_shrink)
                if ds <= ds_min * 1.01: break
                continue

            # accept
            x = eval_bezier(C1, uc)
            if direction > 0:
                append_with_decimation(uv_path, xyz_path, (uc, vc), x, angle_tol, sag_tol, force_keep=False)
            else:
                # front insert with simple decimation analog
                uv_path.insert(0, (uc, vc)); xyz_path.insert(0, x)

            # adapt ds from correction ratio
            pred_norm = np.hypot(up - u, vp - v) + 1e-18
            corr_norm = np.hypot(uc - up, vc - vp)
            ratio = corr_norm / pred_norm
            if ratio < 0.1: ds = min(ds_max, ds * step_growth)
            elif ratio > 0.5: ds = max(ds_min, ds * step_shrink)

            u, v = uc, vc
            pts += 1

        # termination reason: boundary or minimum step
        at_bnd = (u <= 0.0 or u >= 1.0 or v <= 0.0 or v >= 1.0)
        return 'boundary' if at_bnd else 'rank_change_or_min_step'

    start_reason = march(-1)
    end_reason   = march(+1)

    return {'kind': 'overlap', 'points': uv_path, 'xyz': xyz_path,
            'start': start_reason, 'end': end_reason}

# ---------- High-level routine: classify & extract contact ----------
def contact_detect_and_extract(C1, C2,
                               seed_uv=(0.5, 0.5),
                               envelope_prune=None,
                               sv_thresh=1e-2):
    """
    C1, C2: (n+1,d), (m+1,d) Bezier control nets
    seed_uv: initial guess in [0,1]^2
    envelope_prune: optional dict {'ctrl1': C1span, 'ctrl2': C2span, 'tol': eps} for early rejection
    Returns:
      - if isolated: {'type':'isolated', 'u':u, 'v':v, 'point':x}
      - if overlap:  {'type':'overlap', 'uv_path':[(u,v)...], 'xyz_path':[x...], 'start':..., 'end':...}
      - else:        {'type':'none'}
    """
    # Optional envelope prune
    if envelope_prune is not None:
        
        if bernstein_envelope_min(envelope_prune) > 0:
            return {'type': 'none'}

    # Try to land on G=0
    u0, v0 = seed_uv
    u, v, G, J = newton_project_G0(C1, C2, u0, v0, tol=1e-12)
    if np.linalg.norm(G) > 1e-8:
        return {'type': 'none'}

    cls = classify_contact(J, sv_thresh)
    if cls['type'] == 'isolated':
        x = eval_bezier(C1, u)  # equals eval_bezier(C2, v)
        return {'type': 'isolated', 'u': u, 'v': v, 'point': x}
    if cls['type'] != 'overlap':
        return {'type': 'none'}

    # Trace the fold (overlap/correspondence)
    res = trace_overlap_fast_events(C1, C2, u, v, sv_thresh=sv_thresh)
    if res['kind'] != 'overlap' or len(res['points']) < 2:
        return {'type': 'none'}

    xyz = [eval_bezier(C1, uu) for (uu, vv) in res['points']]
    return {'type': 'overlap', 'uv_path': res['points'], 'xyz_path': xyz,
            'start': res['start'], 'end': res['end']}
from typing import List, Tuple

Rect = Tuple[float, float, float, float]

def normalize_rect(r: Rect) -> Rect:
    x0, y0, x1, y1 = r
    if any(map(lambda z: z is None, r)):
        raise ValueError("Rectangle coordinates must be finite numbers.")
    if not (x0 <= x1 and y0 <= y1):
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
    return x0, y0, x1, y1

def is_strictly_inside(inner: Rect, outer: Rect) -> bool:
    u0, v0, u1, v1 = inner
    s0, t0, s1, t1 = outer
    return (s0 <= u0) and (u1 <= s1) and (t0 <= v0) and (v1 <= t1) and (
        (s0 < u0) or (u1 < s1) or (t0 < v0) or (v1 < t1)
    )

def area(r: Rect) -> float:
    x0, y0, x1, y1 = r
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)

def frame_difference(inner: Rect, outer: Rect) -> List[Rect]:
    r"""
    Partition the set difference (outer \ inner) into <= 4 rectangles:
    left, right, bottom, top (with degenerate pieces removed).
    """
    u0, v0, u1, v1 = normalize_rect(inner)
    s0, t0, s1, t1 = normalize_rect(outer)
    
    if not is_strictly_inside((u0, v0, u1, v1), (s0, t0, s1, t1)):
        raise ValueError(f"Inner rectangle must be within (or equal to) the outer rectangle. {(u0, v0, u1, v1)},{(s0, t0, s1, t1) }")

    pieces: List[Rect] = []

    # Left strip: [s0, u0] x [t0, t1]
    left = (s0, t0, u0, t1)
    if area(left) > 0:
        pieces.append(left)

    # Right strip: [u1, s1] x [t0, t1]
    right = (u1, t0, s1, t1)
    if area(right) > 0:
        pieces.append(right)

    # Bottom strip (between left/right): [u0, u1] x [t0, v0]
    bottom = (u0, t0, u1, v0)
    if area(bottom) > 0:
        pieces.append(bottom)

    # Top strip (between left/right): [u0, u1] x [v1, t1]
    top = (u0, v1, u1, t1)
    if area(top) > 0:
        pieces.append(top)

    return pieces

# Optional helper to build an outer rect by padding the inner one
def padded_outer(inner: Rect, pad_x: float, pad_y: float) -> Rect:
    u0, v0, u1, v1 = normalize_rect(inner)
    if pad_x < 0 or pad_y < 0:
        raise ValueError("Padding must be non-negative.")
    return (u0 - pad_x, v0 - pad_y, u1 + pad_x, v1 + pad_y)


import numpy as np

def spread(ctrl):
    """Geometric spread (used to choose split axis)."""
    mu = ctrl.mean(axis=0, keepdims=True)
    return float(np.max(np.linalg.norm(ctrl - mu, axis=1)))

def map_local_to_global(u_loc, v_loc, u0, u1, v0, v1):
    return (u0 + (u1 - u0) * u_loc, v0 + (v1 - v0) * v_loc)

def polyline_bbox_uv(uv_path):
    us = [uv[0] for uv in uv_path]
    vs = [uv[1] for uv in uv_path]
    return (min(us), max(us), min(vs), max(vs))

def seg_point_dist2(p, a, b):
    """Distance^2 from point p to segment ab in R^2."""
    ab = b - a
    t = 0.0
    denom = np.dot(ab, ab)
    if denom > 0:
        t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    d = p - proj
    return float(np.dot(d, d))

def center_covered_by_overlaps(overlaps_uv, u, v, du, dv):
    """
    Return True if (u,v) is already on/near an existing overlap polyline.
    Threshold scales with the cell size.
    """
    if not overlaps_uv:
        return False
    p = np.array([u, v], dtype=float)
    # generous local threshold (half of the larger side)
    thr2 = (0.55 * max(du, dv)) ** 2
    for uv_path in overlaps_uv:
        umin, umax, vmin, vmax = polyline_bbox_uv(uv_path)
        # quick bbox rejection with a small margin
        if u < umin - du or u > umax + du or v < vmin - dv or v > vmax + dv:
            continue
        # distance to polyline
        for i in range(len(uv_path) - 1):
            a = np.array(uv_path[i], dtype=float)
            b = np.array(uv_path[i+1], dtype=float)
            if seg_point_dist2(p, a, b) <= thr2:
                return True
    return False

def _sup_norm_ctrl(ctrl):
    """sup ||ctrl_i|| over control vectors"""
    if ctrl.shape[0] == 0:
        return 0.0
    return float(np.max(np.linalg.norm(ctrl, axis=1)))

def _sup_r_for_cell(Pseg, Qseg):
    """sup ||C1(u)-C2(v)|| over the subcell via Bernstein envelope"""
    return float(np.sqrt(np.max(distance_squared_net(Pseg, Qseg))))

def krawczyk_unique_G_2d(Pseg, Qseg, cond_max=1e12):
    """
    Krawczyk test on G(u,v)=C1(u)-C2(v) for a *single* isolated root inside the current [0,1]^2 subcell.
    Works for planar curves only (d=2).
    Returns 'unique', 'empty', or 'unknown'.
    """
    d = Pseg.shape[1]
    if d != 2:
        return 'unknown'  # only well-defined square system here

    # center point in local cell [0,1]^2
    uc, vc = 0.5, 0.5
    Gc, Jc = G_and_J(Pseg, Qseg, uc, vc)   # Gc in R^2, Jc in R^{2x2}
    # try to invert Jc
    try:
        cond = np.linalg.cond(Jc)
        if not np.isfinite(cond) or cond > cond_max:
            return 'unknown'
        C = np.linalg.inv(Jc)  # approximate inverse
    except np.linalg.LinAlgError:
        return 'unknown'

    # interval bounds of J over the whole cell
    # J = [[ 2||t1||^2 + 2 r·C1'' ,    -2 t1·t2 ],
    #      [    -2 t1·t2         ,  2||t2||^2 - 2 r·C2'' ]]
    t1_ctrl = deriv_ctrl(Pseg)               # (n, d)
    t2_ctrl = deriv_ctrl(Qseg)               # (m, d)
    tt1_ctrl = second_deriv_ctrl(Pseg)       # (n-1, d) or (1,d) zero if deg<2
    tt2_ctrl = second_deriv_ctrl(Qseg)

    sup_t1 = _sup_norm_ctrl(t1_ctrl)
    sup_t2 = _sup_norm_ctrl(t2_ctrl)
    sup_tt1 = _sup_norm_ctrl(tt1_ctrl)
    sup_tt2 = _sup_norm_ctrl(tt2_ctrl)
    sup_r = _sup_r_for_cell(Pseg, Qseg)

    # conservative intervals
    # note: we don't have a safe positive lower bound for ||t||; use 0 for lower.
    A_lo = 2.0 * (0.0 - sup_r * sup_tt1)
    A_hi = 2.0 * (sup_t1**2 + sup_r * sup_tt1)
    D_lo = 2.0 * (0.0 - sup_r * sup_tt2)
    D_hi = 2.0 * (sup_t2**2 + sup_r * sup_tt2)
    B_abs = 2.0 * sup_t1 * sup_t2  # off-diagonals are in [-B_abs, +B_abs]

    # interval matrix J(X)
    J_int = np.array([
        [[A_lo, A_hi], [-B_abs, +B_abs]],
        [[-B_abs, +B_abs], [D_lo, D_hi]]
    ], dtype=float)  # shape (2,2,2) -> [low, high]

    # M(X) = I - C * J(X)  (interval)
    I = np.eye(2)
    # compute C * J interval: entrywise convolution with interval arithmetic
    CJ = np.zeros_like(J_int)
    for i in range(2):
        for j in range(2):
            # (C*J)_ij = sum_k C_{ik} * J_{kj}
            low, high = 0.0, 0.0
            for k in range(2):
                c = C[i, k]
                jk_low, jk_high = J_int[k, j, 0], J_int[k, j, 1]
                prod_low = min(c * jk_low, c * jk_high)
                prod_high = max(c * jk_low, c * jk_high)
                # add intervals [prod_low, prod_high]
                low += prod_low
                high += prod_high
            CJ[i, j, 0] = low
            CJ[i, j, 1] = high
    # M_int = I - CJ
    M_int = np.zeros_like(CJ)
    for i in range(2):
        for j in range(2):
            # I_ij as exact [v,v]
            Iij = I[i, j]
            M_int[i, j, 0] = Iij - CJ[i, j, 1]  # worst-case low
            M_int[i, j, 1] = Iij - CJ[i, j, 0]  # worst-case high

    # radius vector for X-x̄ (local box): [-0.5, +0.5] in both coords
    r = np.array([0.5, 0.5], dtype=float)

    # symmetric bound R_i = sum_j sup|M_ij| * r_j
    sup_abs_M = np.maximum(np.abs(M_int[:, :, 0]), np.abs(M_int[:, :, 1]))
    R = sup_abs_M @ r  # (2,)

    # y = x̄ - C*F(x̄)
    y = np.array([uc, vc]) - C @ Gc  # point in R^2

    # K(X) = y + [-R, +R]
    K_lo = y - R
    K_hi = y + R

    # tests w.r.t X = [0,1]^2
    if (K_hi[0] < 0.0 or K_lo[0] > 1.0) or (K_hi[1] < 0.0 or K_lo[1] > 1.0):
        return 'empty'
    # interior (use small slack)
    eps = 1e-14
    if (K_lo[0] > 0.0 + eps and K_hi[0] < 1.0 - eps and
        K_lo[1] > 0.0 + eps and K_hi[1] < 1.0 - eps):
        return 'unique'
    return 'unknown'


    
def subdivide_u(dist_net,su_net,sv_net, u:float):
    left,right=zip(de_casteljau_split_nd(dist_net, axis=0,t=u),
    de_casteljau_split_nd(su_net, axis=0, t=u),
    de_casteljau_split_nd(sv_net, axis=0, t=u))
    return left,right
def subdivide_v(dist_net,su_net,sv_net, v:float):
    left,right=zip(de_casteljau_split_nd(dist_net, axis=1,t=v),
    de_casteljau_split_nd(su_net, axis=1, t=v),
    de_casteljau_split_nd(sv_net, axis=1, t=v))
    return left,right
def bern_restrict_face(ctrl, axis, which):
    """
    Restrict tensor Bernstein grid to a parameter face.
    which = 0  → face at parameter 0 ⇒ take index 0 along axis
    which = 1  → face at parameter 1 ⇒ take index -1 along axis
    """
    idx = 0 if which == 0 else -1
    sl = [slice(None)] * ctrl.ndim
    sl[axis] = idx
    return ctrl[tuple(sl)]

def has_zero_in_interval(lo, hi, eps=0.0):
    return (lo <= eps) and (hi >= -eps)

def sign_all_nonneg(A, eps=0.0):
    return np.min(A) >= -eps

def sign_all_nonpos(A, eps=0.0):
    return np.max(A) <= eps

def pm_existence_test(Du, Dv, eps=0.0):
    """
    Poincaré–Miranda (2D) using only face control nets of ∂uD and ∂vD.
    We need opposite signs of ∂uD on u=0 vs u=1 faces, and of ∂vD on v=0 vs v=1 faces.
    Sufficient (robust) patterns:
      - On u=0 face: ∂uD ≥ 0  and on u=1 face: ∂uD ≤ 0  (or vice versa)
      - On v=0 face: ∂vD ≥ 0  and on v=1 face: ∂vD ≤ 0  (or vice versa)
    """
    # u-faces for ∂uD (Du has shape (Nu, Nv+1); faces are the first/last rows along axis 0 of the *original u*,
    # which correspond to faces of D, not of Du; however for PM we still check faces of Du at u=0/u=1:
    # That is: take Du_face0 = Du[0, :], Du_face1 = Du[-1, :])
    dus=np.squeeze(Du)
    dvs=np.squeeze(Dv)
    #Du_face0 = bern_restrict_face(dus, axis=0, which=0)
    #Du_face1 = bern_restrict_face(dus, axis=0, which=1)
    #
    #cond_u = (sign_all_nonneg(Du_face0, eps) and sign_all_nonpos(Du_face1, eps)) or \
    #         (sign_all_nonpos(Du_face0, eps) and sign_all_nonneg(Du_face1, eps))
    #
    ## v-faces for ∂vD (Dv shape (Nu+1, Nv); faces are first/last cols along axis 1)
    ##Dv_face0 = bern_restrict_face(dvs, axis=1, which=0)
    ##Dv_face1 = bern_restrict_face(dvs, axis=1, which=1)
    #
    #cond_v = (sign_all_nonneg(Dv_face0, eps) and sign_all_nonpos(Dv_face1, eps)) or \
    #         (sign_all_nonpos(Dv_face0, eps) and sign_all_nonneg(Dv_face1, eps))
    #
    
    return bool((not  bern_no_sign_change(dus)) and (not  bern_no_sign_change(dvs)))

# ---------- Uniqueness via global positive-definite Hessian ----------
def pd_hessian_uniqueness(Duu, Duv, Dvv, eps=0.0):
    """
    Certify that Hessian H = [[Duu, Duv],[Duv, Dvv]] is PD everywhere on the box using
    Bernstein bounds. Sufficient global conditions:
        a_min = min(Duu) > eps
        c_min = min(Dvv) > eps
        det_min = a_min * c_min - (max|Duv|)^2 > eps
    If these hold, D is strictly convex on the box ⇒ ∇D is injective ⇒ ≤ 1 stationary point.
    """
    a_min = float(np.min(Duu))
    c_min = float(np.min(Dvv))
    b_abs_max = float(np.max(np.abs(Duv)))

    if (a_min > eps) and (c_min > eps) and (a_min * c_min - b_abs_max * b_abs_max > eps):
        return True
    return False

# ---------- Master classifier on a Dnet ----------
def classify_cell_by_grids( Du, Dv , eps_face=1e-14, eps_pd=1e-14):
    """
    Inputs:
      Dnet: 2D Bernstein control grid of the squared distance D(u,v)
    Returns:
      dict(status=..., existence=..., uniqueness=..., notes=...)
        status in {"no_stationary","unique_stationary","maybe_multiple"}
    Logic:
      1) Build Du,Dv and check PM existence on faces.
      2) If no existence → "no_stationary"
      3) Else build Duu,Duv,Dvv; if PD everywhere → "unique_stationary"
         Else "maybe_multiple" (could be tangency/overlap or multiple roots).
    """


    exists = pm_existence_test(Du, Dv, eps=eps_face)
    if not exists:
        return dict(status="no_stationary",
                    existence=False, uniqueness=False,
                    notes="∇D has no zero by Poincaré–Miranda face test")
    Duu, Duv, Dvv = bernstein_partial_derivative_coeffs(Du,axis=0), bernstein_partial_derivative_coeffs(Du,axis=1), bernstein_partial_derivative_coeffs(Dv,axis=1)

    unique = pd_hessian_uniqueness(Duu, Duv, Dvv, eps=eps_pd)
    if unique:
        return dict(status="unique_stationary",
                    existence=True, uniqueness=True,
                    notes="PM existence + globally PD Hessian from Bernstein bounds")
    else:
        return dict(status="maybe_multiple",
                    existence=True, uniqueness=False,
                    notes="PM says ≥1 root; Hessian not PD globally ⇒ maybe tangency/overlap/multiple")
def bezier_intersect_certified_full(
    C1, C2,
    tol_hit=1e-9,
    max_depth=28,
    sv_thresh=1e-8,
    use_krawczyk=True
):
    """
    Certified intersection of two Bézier curves (R^2 or R^3).
    Returns:
      {
        'isolated': [ {'u':..., 'v':..., 'point': np.array(d)}, ... ],
        'overlaps': [ {'uv_path': [(u,v)...], 'xyz_path': [x...],
                       'start': 'boundary|rank_change_or_min_step',
                       'end':   'boundary|rank_change_or_min_step'} , ... ],
        'stats': {'cells': int, 'pruned': int, 'unique_boxes': int, 'overlap_traces': int}
      }
    }
    """

    # stack holds: (Pseg, Qseg, u0,u1, v0,v1, depth)
   

    isolated = []  # list of dicts
    overlaps = []  # list of dicts with uv_path, xyz_path, start, end
    overlaps_uv_registry = []  # just the uv polylines for fast checks
    stats = {'cells': 0, 'pruned': 0, 'unique_boxes': 0, 'overlap_traces': 0,'pruned_by':[]}
    sq_dist_net=bernstein_distance_squared_net(C1,C2)[...,None]
    su_net=bernstein_partial_derivative_coeffs(sq_dist_net,0)
    sv_net = bernstein_partial_derivative_coeffs(sq_dist_net, 1)
    stack = [(C1.copy(), C2.copy(),sq_dist_net, su_net, sv_net,0.0, 1.0, 0.0, 1.0, 0)]
    def near_existing_isolated(u, v, eps=1e-6):
        for it in isolated:
            if (abs(it['u'] - u) <= eps) and (abs(it['v'] - v) <= eps):
                return True
        return False
    
    def quad_split_push(Pseg, Qseg, u0, u1, v0, v1, uvr0,uvr1=None,depth=None):
        crv1=nurbs_curve(Pseg,generate_knots(Pseg.shape[0],Pseg.shape[0]-1,interval=(u0,u1)))
        crv2 = nurbs_curve(Qseg, generate_knots(Qseg.shape[0], Qseg.shape[0] - 1,interval=(v0,v1)))
        
        #PL, PR = de_casteljau_split(Pseg)
        #QL, QR = de_casteljau_split(Qseg)
        if uvr1 is None:
            
            uvr0= (uvr0[0] - 0.001, uvr0[1] - 0.001)
            uvr1 = (uvr0[0] + 0.001, uvr0[1] + 0.001)
        
        if not is_strictly_inside((uvr0[0],uvr0[1],uvr1[0],uvr1[1]),(u0,v0, u1, v1)):
            return
        
        frames=frame_difference((uvr0[0],uvr0[1],uvr1[0],uvr1[1]),(u0,v0, u1, v1))
        for (_u0,_v0,_u1,_v1) in frames:
            sub1,sub2=trim_curve(crv1,_u0,_u1), trim_curve(crv2,_v0,_v1)
            stack.append((sub1.control_points, sub2.control_points,  *sub1.interval(), *sub2.interval(), depth + 1))
        #um = 0.5 * (u0 + u1)
        #vm = 0.5 * (v0 + v1)
        # Push all four subcells (DFS order; use any order you like)
        
        #stack.append((PR, QR, um, u1, vm, v1, depth + 1))
        #stack.append((PR, QL, um, u1, v0, vm, depth + 1))
        #stack.append((PL, QR, u0, um, vm, v1, depth + 1))
        #stack.append((PL, QL, u0, um, v0, vm, depth + 1))
    while stack:
        Pseg, Qseg, dnet,sunet, svnet ,u0, u1, v0, v1, depth = stack.pop()
        stats['cells'] += 1
        
        # 1) Envelope prune remains as-is
        if bernstein_envelope_min(dnet) >0:
            stats['pruned'] += 1
            stats['pruned_by'].append('bernstein_envelope_min')
            continue
            
            # 1) Envelope prune remains as-is
       
        res=classify_cell_by_grids(sunet,svnet)
        if res['status']=="no_stationary":
            stats['pruned'] += 1
            stats['pruned_by'].append('classify_cell_by_gri+no_stationary')
            continue
        elif res['status']=="unique_stationary":
            stats['pruned'] += 1
            uc, vc, Gc, Jc = newton_project_G0(Pseg, Qseg, 0.5, 0.5, tol=1e-12)
            if np.linalg.norm(Gc) <= tol_hit:
                ug, vg = map_local_to_global(uc, vc, u0, u1, v0, v1)
       
                if is_strictly_inside((ug, vg, ug, vg), (u0, v0, u1, v1)):
                    x = eval_bezier(C1, ug)
                    isolated.append({'u': ug, 'v': vg, 'point': x})
                    stats['overlap_traces'] += 1
                    stats['pruned_by'].append('classify_cell_by_gri+unique_stationary')
                    continue
                    
            
                    
        else:
            res=contact_detect_and_extract(C1, C2, sv_thresh=sv_thresh)
            #res = trace_overlap_fast_events(Pseg, Qseg, 0.5, 0.5, sv_thresh=sv_thresh)
            if res['type'] == 'overlap' and len(res['uv_path']) >= 2:
              
                uv_global = [map_local_to_global(uL, vL, u0, u1, v0, v1) for (uL, vL) in res['uv_path']]
                #rects=[(*low,*upp )for low,upp in pairwise(uv_global)  ]
                
                xyz_global = [eval_bezier(C1, ug) for (ug, vg) in uv_global]
                overlaps.append({'uv_path': uv_global,
                                 'xyz_path': xyz_global,
                                 'start': res['start'],
                                 'end': res['end']})
                overlaps_uv_registry.append(uv_global)
                stats['overlap_traces'] += 1
                continue
                
            # <-- DO NOT 'continue'; we still need to explore the rest of this box.
            #bb = AABB.from_points(np.array(uv_global))
            #bb.offset_inplace(1e-3)
            #bb = bb.intersection(AABB(np.array((u0, v0)), np.array((u1, v1))))
            # uvbb=np.array(aabb_intersection(bb.min, bb.max,np.array([(u0, v0), (u1, v1)])))
            
            #quad_split_push(Pseg, Qseg, u0, u1, v0, v1, bb.min, bb.max, depth)
        
    
        # split by spread if no observation to harvest
        if spread(Pseg) >= spread(Qseg):
                PL, PR = de_casteljau_split_nd(Pseg,axis=0,t=0.5)
                
                l,r=subdivide_u(dnet, sunet, svnet, u=0.5)
                um = 0.5 * (u0 + u1)
                stack.append((PR, Qseg,*r, um, u1, v0, v1, depth + 1))
                stack.append((PL, Qseg,*l, u0, um, v0, v1, depth + 1))
        else:
                QL, QR = de_casteljau_split_nd(Qseg,axis=0,t=0.5)
                l, r =subdivide_v(dnet, sunet, svnet, v=0.5)
                vm = 0.5 * (v0 + v1)
                stack.append((Pseg, QR,*r, u0, u1, vm, v1, depth + 1))
                stack.append((Pseg, QL, *l,u0, u1, v0, vm, depth + 1))
        
        


    return {'isolated': isolated, 'overlaps': overlaps, 'stats': stats}
val1 = np.array([[-19.77608536, 23.10065701, 0.],
                             [-14.86834768, 28.69713066, 0.],
                             [-5.8568525, 25.12677787, 0.],
                             [-12.62581769, 15.26478654, 0.]])


val2 = np.array([[-22.0315362, 18.75969713, 0.],
                             [-19.42270945, 28.2502867, 0.],
                             [-8.46791623, 27.56878356, 0.],
                             [-10.43007782, 19.78973126, 0.]])
s=time.perf_counter()

res1=contact_detect_and_extract(val1,val2)

print(time.perf_counter()-s)
print(res1)
assert res1['type']=='overlap'

val3 = np.array([[-28.46565557, -11.09883504, 0.],
                             [-31.79098016, 13.62423043, 0.],
                             [-12.99566723, 16.66039636, 0.],
                             [8.11291498, -6.32771715, 0.]])

val4 = np.array([[-45.36434109, -7.12015504, 0.],
                             [-25.49612403, 13.94186047, 0.],
                             [-2.13178295, -17.35271318, 0.],
                             [12.02325581, 20.42248062, 0.]])

s=time.perf_counter()
res2=contact_detect_and_extract(val3,val4)
print(time.perf_counter()-s)
print(res2)
assert res2['type']=='isolated'

pp12=bezier_intersect_certified_full(val1,val2)
print(pp12)
pp34=bezier_intersect_certified_full(val3,val4)
print(pp34)
#0.9722023329995864
#{'type': 'overlap', 'uv_path': [(np.float64(4.5640172774137406e-07), np.float64(0.19069120115770427)), (np.float64(9.128046867304141e-07), np.float64(0.1906916474749262)), (np.float64(9.381666090582477e-07), np.float64(0.19069167227639283)), (np.float64(9.854342737719924e-07), np.float64(0.1906917184995216)), (np.float64(0.008135093536834562), np.float64(0.1986460765049657)), (np.float64(0.017793389141888445), np.float64(0.2080909400346464)), (np.float64(0.027484918380936828), np.float64(0.21756830279087433)), (np.float64(0.03720971287372032), np.float64(0.22707819569465823)), (np.float64(0.046967854204110024), np.float64(0.23662069852701417)), (np.float64(0.05675893568527751), np.float64(0.24619541358821878)), (np.float64(0.06658293456694), np.float64(0.2558023186307446)), (np.float64(0.07643945172499501), np.float64(0.2654410233504719)), (np.float64(0.08632771904128043), np.float64(0.2751107766033698)), (np.float64(0.09624701383466781), np.float64(0.28481087167835817)), (np.float64(0.10619700137115856), np.float64(0.29454098123851064)), (np.float64(0.11617616783583767), np.float64(0.3042996249216851)), (np.float64(0.12618317182587038), np.float64(0.31408549096780575)), (np.float64(0.13621694994894723), np.float64(0.3238975394837564)), (np.float64(0.14627614022426907), np.float64(0.33373443858623386)), (np.float64(0.1563589011463736), np.float64(0.34359438746396115)), (np.float64(0.16646351068490095), np.float64(0.35347570214056406)), (np.float64(0.1765876436236514), np.float64(0.3633761087832201)), (np.float64(0.18672923332146202), np.float64(0.37329358642007143)), (np.float64(0.19688576637068753), np.float64(0.38322567718556444)), (np.float64(0.20705485448274463), np.float64(0.39317004556828655)), (np.float64(0.21723381577900644), np.float64(0.40312406895462843)), (np.float64(0.22741983044918), np.float64(0.4130849898473716)), (np.float64(0.23761024063815683), np.float64(0.4230502091255455)), (np.float64(0.24780159163642512), np.float64(0.4330163484229164)), (np.float64(0.2579906821644027), np.float64(0.4429802772028059)), (np.float64(0.26817440495669387), np.float64(0.45293895686516733)), (np.float64(0.27834966585272114), np.float64(0.46288936162517685)), (np.float64(0.28851304886469376), np.float64(0.4728281509826373)), (np.float64(0.2986612592076343), np.float64(0.4827521029617866)), (np.float64(0.3087910183678327), np.float64(0.4926580114985625)), (np.float64(0.31889911697549633), np.float64(0.5025427381448545)), (np.float64(0.3289824003184537), np.float64(0.5124031979023342)), (np.float64(0.33903788334957974), np.float64(0.5222364716884041)), (np.float64(0.3490626015483272), np.float64(0.5320396604934431)), (np.float64(0.35905350525928226), np.float64(0.5418097820542999)), (np.float64(0.3690084406179729), np.float64(0.5515447301032916)), (np.float64(0.3789249573799253), np.float64(0.5612421085422314)), (np.float64(0.3888006754880051), np.float64(0.5708995899092496)), (np.float64(0.39863387554384716), np.float64(0.5805154928017969)), (np.float64(0.40842250870539354), np.float64(0.5900878136537807)), (np.float64(0.4181970953022869), np.float64(0.5996463983469581)), (np.float64(0.42792482010895877), np.float64(0.6091591568169873)), (np.float64(0.4376047275563011), np.float64(0.6186251546112274)), (np.float64(0.44723591416304304), np.float64(0.6280435082138183)), (np.float64(0.4568181278425686), np.float64(0.6374139711088244)), (np.float64(0.4663508028275397), np.float64(0.6467359900314008)), (np.float64(0.47583388949803057), np.float64(0.656009516458124)), (np.float64(0.48526814062318174), np.float64(0.6652352865231839)), (np.float64(0.49465406850717164), np.float64(0.6744138012096763)), (np.float64(0.503754113566222), np.float64(0.6833127506004628)), (np.float64(0.5128194562015009), np.float64(0.6921777644328774)), (np.float64(0.5218514084898659), np.float64(0.7010101257890565)), (np.float64(0.530854104783129), np.float64(0.7098138776585676)), (np.float64(0.5398296747995539), np.float64(0.7185911026964514)), (np.float64(0.5487833774364569), np.float64(0.7273469435872113)), (np.float64(0.557716874227658), np.float64(0.7360830251475278)), (np.float64(0.5666356795586204), np.float64(0.7448047399041904)), (np.float64(0.5755409693749473), np.float64(0.7535132378163408)), (np.float64(0.58443577309379), np.float64(0.7622114813560703)), (np.float64(0.5933235960630875), np.float64(0.7709028984089887)), (np.float64(0.6022126077360103), np.float64(0.7795954778970686)), (np.float64(0.6111264702910659), np.float64(0.7883123591044874)), (np.float64(0.6200224886393191), np.float64(0.7970117904318185)), (np.float64(0.6289008259116737), np.float64(0.8056939314050094)), (np.float64(0.637766443498654), np.float64(0.814363633776181)), (np.float64(0.6466190104543044), np.float64(0.8230205739126968)), (np.float64(0.6554588185861148), np.float64(0.8316650371735694)), (np.float64(0.6643536709631248), np.float64(0.8403633282941889)), (np.float64(0.6732561666989436), np.float64(0.8490690938679003)), (np.float64(0.6821649041509662), np.float64(0.857780963226027)), (np.float64(0.691081120625172), np.float64(0.8665001463321629)), (np.float64(0.7000051030367145), np.float64(0.8752269237611165)), (np.float64(0.7089366750166634), np.float64(0.8839611230414149)), (np.float64(0.7178771621111245), np.float64(0.8927040404268325)), (np.float64(0.7268258362376248), np.float64(0.9014549639242996)), (np.float64(0.7357841671055284), np.float64(0.9102153307650904)), (np.float64(0.7447511563549927), np.float64(0.9189841646514461)), (np.float64(0.7537283946246567), np.float64(0.9277630210715859)), (np.float64(0.7627141407157623), np.float64(0.9365501973043293)), (np.float64(0.7717093856926726), np.float64(0.9453466625132034)), (np.float64(0.7807151812604854), np.float64(0.9541534451624134)), (np.float64(0.7897304299527802), np.float64(0.9629694720377182)), (np.float64(0.7987558518699761), np.float64(0.9717954473265488)), (np.float64(0.8077898281678356), np.float64(0.9806297879584196)), (np.float64(0.8167970561885987), np.float64(0.9894379714048893)), (np.float64(0.8258510869143175), np.float64(0.9982919232951586)), (0.8275977622027193, 1.0)], 'xyz_path': [array([-19.77607864,  23.10066467,   0.        ]), array([-19.77607192,  23.10067234,   0.        ]), array([-19.77607155,  23.10067276,   0.        ]), array([-19.77607085,  23.10067355,   0.        ]), array([-19.65550659,  23.23542209,   0.        ]), array([-19.51032371,  23.39070713,   0.        ]), array([-19.36253171,  23.54139814,   0.        ]), array([-19.21221752,  23.68745853,   0.        ]), array([-19.05946943,  23.82885249,   0.        ]), array([-18.90438564,  23.96553751,   0.        ]), array([-18.74706067,  24.09747711,   0.        ]), array([-18.58759737,  24.22463028,   0.        ]), array([-18.42610712,  24.34695255,   0.        ]), array([-18.26270318,  24.46440176,   0.        ]), array([-18.09749489,  24.5769417 ,   0.        ]), array([-17.93061367,  24.68452467,   0.        ]), array([-17.76219082,  24.78710804,   0.        ]), array([-17.59235544,  24.88465483,   0.        ]), array([-17.42124405,  24.97712779,   0.        ]), array([-17.24900363,  25.06448871,   0.        ]), array([-17.07578137,  25.14670436,   0.        ]), array([-16.90173682,  25.22374087,   0.        ]), array([-16.72702695,  25.2955712 ,   0.        ]), array([-16.55181801,  25.36216997,   0.        ]), array([-16.37627544,  25.42351793,   0.        ]), array([-16.20057078,  25.47959941,   0.        ]), array([-16.02487868,  25.53040381,   0.        ]), array([-15.84937135,  25.57592714,   0.        ]), array([-15.67423477,  25.61616806,   0.        ]), array([-15.49965009,  25.65113321,   0.        ]), array([-15.32579607,  25.68083607,   0.        ]), array([-15.15285008,  25.70529649,   0.        ]), array([-14.98099332,  25.72454004,   0.        ]), array([-14.81040294,  25.73859923,   0.        ]), array([-14.64125337,  25.74751312,   0.        ]), array([-14.47371508,  25.75132713,   0.        ]), array([-14.30795451,  25.75009281,   0.        ]), array([-14.14413184,  25.74386754,   0.        ]), array([-13.98240319,  25.73271417,   0.        ]), array([-13.82292196,  25.7167011 ,   0.        ]), array([-13.66582306,  25.69590014,   0.        ]), array([-13.51124194,  25.67038751,   0.        ]), array([-13.35930847,  25.64024337,   0.        ]), array([-13.21013785,  25.60554928,   0.        ]), array([-13.06384586,  25.56639079,   0.        ]), array([-12.92006841,  25.52270344,   0.        ]), array([-12.77937716,  25.47469263,   0.        ]), array([-12.64186005,  25.42244487,   0.        ]), array([-12.50760025,  25.36604731,   0.        ]), array([-12.37666812,  25.30558369,   0.        ]), array([-12.24913486,  25.24113959,   0.        ]), array([-12.12506134,  25.17279697,   0.        ]), array([-12.00449524,  25.10063074,   0.        ]), array([-11.88748523,  25.02471569,   0.        ]), array([-11.77693103,  24.94720431,   0.        ]), array([-11.66971948,  24.86617538,   0.        ]), array([-11.56589061,  24.78167285,   0.        ]), array([-11.46545336,  24.69370986,   0.        ]), array([-11.36844197,  24.60231316,   0.        ]), array([-11.27486003,  24.50747187,   0.        ]), array([-11.18475304,  24.40920656,   0.        ]), array([-11.09813116,  24.30749013,   0.        ]), array([-11.01505158,  24.20233852,   0.        ]), array([-10.9355564 ,  24.09374261,   0.        ]), array([-10.85968787,  23.98168225,   0.        ]), array([-10.7874566 ,  23.86607002,   0.        ]), array([-10.71877056,  23.74659478,   0.        ]), array([-10.65404727,  23.62383678,   0.        ]), array([-10.59334641,  23.49782695,   0.        ]), array([-10.53669739,  23.36852487,   0.        ]), array([-10.484168  ,  23.23596249,   0.        ]), array([-10.43582148,  23.10016246,   0.        ]), array([-10.3913999 ,  22.96007058,   0.        ]), array([-10.35126865,  22.81640906,   0.        ]), array([-10.31552844,  22.66920463,   0.        ]), array([-10.28426897,  22.51844109,   0.        ]), array([-10.25758551,  22.36411703,   0.        ]), array([-10.23557535,  22.20623892,   0.        ]), array([-10.2183334 ,  22.04478615,   0.        ]), array([-10.20596017,  21.8797744 ,   0.        ]), array([-10.19855389,  21.71117874,   0.        ]), array([-10.19621656,  21.53901994,   0.        ]), array([-10.19905001,  21.36326914,   0.        ]), array([-10.20715633,  21.18396231,   0.        ]), array([-10.22063898,  21.00108199,   0.        ]), array([-10.23960447,  20.81460807,   0.        ]), array([-10.26415599,  20.62456462,   0.        ]), array([-10.29440127,  20.43093812,   0.        ]), array([-10.33044035,  20.23376548,   0.        ]), array([-10.37220246,  20.03384002,   0.        ]), array([-10.42013621,  19.82953078,   0.        ]), array([-10.43007782,  19.78973126,   0.        ])], 'start': 'rank_change_or_min_step', 'end': 'boundary'}
#0.000685040999087505
#{'type': 'isolated', 'u': np.float64(0.19649579172632328), 'v': np.float64(0.2818845674995799), 'point': array([-28.01389436,   0.93016065,   0.        ])}

#0.04940679200080922
#{'type': 'overlap', 'uv_path': [(0.0, np.float64(0.19069075484142167)), (np.float64(0.000962352457093126), np.float64(0.19163184092938626)), (np.float64(0.020217914103035032), np.float64(0.21046188707526398)), (np.float64(0.03988330668389066), np.float64(0.22969270758194943)), (np.float64(0.059963740568752553), np.float64(0.2493293976782277)), (np.float64(0.08046479714162132), np.float64(0.26937741540900356)), (np.float64(0.10139364399126365), np.float64(0.2898437699715044)), (np.float64(0.1227581686984695), np.float64(0.31073617464438524)), (np.float64(0.14456760124016535), np.float64(0.3320636554377567)), (np.float64(0.16683255424654006), np.float64(0.3538365904606962)), (np.float64(0.18956528105527068), np.float64(0.37606696227291303)), (np.float64(0.21278022387241408), np.float64(0.3987688939322087)), (np.float64(0.2364924084929367), np.float64(0.4219570791889839)), (np.float64(0.2607177608422034), np.float64(0.44564709203270175)), (np.float64(0.2854719841351579), np.float64(0.46985428866396395)), (np.float64(0.31076840624504004), np.float64(0.49459170243276196)), (np.float64(0.3366153386834931), np.float64(0.5198674611806847)), (np.float64(0.36301175271108554), np.float64(0.5456805589021138)), (np.float64(0.3817286732393837), np.float64(0.5639838670021631)), (np.float64(0.3943217432241637), np.float64(0.5762986513452099)), (np.float64(0.4027680159770804), np.float64(0.5845582756929807)), (np.float64(0.40842250870539354), np.float64(0.5900878136537807)), (np.float64(0.4140629617307638), np.float64(0.5956036221655715)), (np.float64(0.42255770663049624), np.float64(0.6039106475080734)), (np.float64(0.4353692997372028), np.float64(0.616439125982313)), (np.float64(0.45473030966922995), np.float64(0.6353722902009012)), (np.float64(0.48269273789776884), np.float64(0.6627167958603071)), (np.float64(0.5107006498974062), np.float64(0.6901057801792285)), (np.float64(0.5388687302185815), np.float64(0.7176513933720557)), (np.float64(0.5670217994850147), np.float64(0.7451823272306216)), (np.float64(0.5949479386708039), np.float64(0.7724913457812147)), (np.float64(0.6224285210286062), np.float64(0.7993646535580161)), (np.float64(0.649186431223669), np.float64(0.8255312590073663)), (np.float64(0.6751200764366713), np.float64(0.8508918143395888)), (np.float64(0.7000602521708095), np.float64(0.8752808541922223)), (np.float64(0.7239309935578304), np.float64(0.8986240923929841)), (np.float64(0.746707160048258), np.float64(0.9208969439567153)), (np.float64(0.7684041769071299), np.float64(0.9421144932766314)), (np.float64(0.7890630099438372), np.float64(0.9623168008872355)), (np.float64(0.808739365598342), np.float64(0.9815583422104036)), (np.float64(0.8274493331889435), np.float64(0.9998548510179452)), (np.float64(0.8275977622022961), 1.0)], 'xyz_path': [array([-19.77608536,  23.10065701,   0.        ]), array([-19.76190506,  23.11678888,   0.        ]), array([-19.47354463,  23.42888663,   0.        ]), array([-19.17055315,  23.72671263,   0.        ]), array([-18.85324661,  24.0091513 ,   0.        ]), array([-18.52203385,  24.27505811,   0.        ]), array([-18.1774042 ,  24.52327128,   0.        ]), array([-17.8199492 ,  24.75259408,   0.        ]), array([-17.45036157,  24.96179623,   0.        ]), array([-17.06944495,  25.14960621,   0.        ]), array([-16.67812166,  25.31470456,   0.        ]), array([-16.2774376 ,  25.45571665,   0.        ]), array([-15.86860693,  25.57119193,   0.        ]), array([-15.45302625,  25.65960049,   0.        ]), array([-15.03231549,  25.71932459,   0.        ]), array([-14.60837676,  25.74865799,   0.        ]), array([-14.1834579 ,  25.74582242,   0.        ]), array([-13.76023409,  25.70900859,   0.        ]), array([-13.46789896,  25.66230934,   0.        ]), array([-13.27528699,  25.62133743,   0.        ]), array([-13.14807819,  25.5895726 ,   0.        ]), array([-13.06384586,  25.56639079,   0.        ]), array([-12.98059049,  25.54173897,   0.        ]), array([-12.85669899,  25.50174006,   0.        ]), array([-12.67339194,  25.43490611,   0.        ]), array([-12.40496611,  25.31912696,   0.        ]), array([-12.03711825,  25.12074343,   0.        ]), array([-11.69451183,  24.88545408,   0.        ]), array([-11.37867655,  24.61227445,   0.        ]), array([-11.09445765,  24.30300521,   0.        ]), array([-10.84621378,  23.9608195 ,   0.        ]), array([-10.63720958,  23.59003254,   0.        ]), array([-10.46969975,  23.19687345,   0.        ]), array([-10.34342247,  22.785895  ,   0.        ]), array([-10.2574351 ,  22.36315269,   0.        ]), array([-10.20942289,  21.93352739,   0.        ]), array([-10.19638989,  21.50101554,   0.        ]), array([-10.21505529,  21.06867213,   0.        ]), array([-10.26214558,  20.63874941,   0.        ]), array([-10.33456689,  20.21284632,   0.        ]), array([-10.42922422,  19.79311818,   0.        ]), array([-10.43007782,  19.78973126,   0.        ])], 'start': 'boundary', 'end': 'boundary'}
#0.0005249999994703103
#{'type': 'isolated', 'u': np.float64(0.19649579172632328), 'v': np.float64(0.2818845674995799), 'point': array([-28.01389436,   0.93016065,   0.        ])}
