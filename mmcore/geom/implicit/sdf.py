from __future__ import annotations

import heapq
import time
from typing import Callable, Tuple, Dict, Any, Optional

import numpy as np
NDArray = np.ndarray



import heapq
from typing import Callable, Tuple, Dict, Any, Optional

import numpy as np




import heapq
from typing import Callable, Tuple, Dict, Any, Optional


def _make_values_only_vectorized(sdf: Callable[[NDArray], Any]) -> Callable[[NDArray], NDArray]:
    probe = np.zeros((1, 3), dtype=np.float64)
    out = sdf(probe)
    if isinstance(out, tuple):
        def vals(pts: NDArray) -> NDArray:
            v = sdf(pts)[0]
            v = np.asarray(v, dtype=np.float64).ravel()
            if v.size == 1 and len(pts) != 1:
                return np.full(len(pts), float(v[0]), dtype=np.float64)
            return v
    else:
        def vals(pts: NDArray) -> NDArray:
            v = sdf(pts)
            v = np.asarray(v, dtype=np.float64).ravel()
            if v.size == 1 and len(pts) != 1:
                return np.full(len(pts), float(v[0]), dtype=np.float64)
            return v
    return vals


def _make_val_dirderiv_vectorized(
    sdf: Callable[[NDArray], Any],
    u_unit: NDArray,
    *,
    numeric_h: float = 1.0e-6,
) -> Callable[[NDArray], tuple[NDArray, NDArray]]:
    """
    Returns eval(pts) -> (vals, dphi_ds) for pts shaped (N,3),
    where dphi_ds is the directional derivative along the *unit* direction u_unit
    with respect to arc-length s (not t).

    If sdf provides gradients (vals, grads), we use grads @ u_unit.
    Otherwise we approximate directional derivative numerically in a single
    stacked vectorized call (2N points).
    """
    u = np.asarray(u_unit, dtype=np.float64).reshape(3)
    probe = np.zeros((1, 3), dtype=np.float64)
    out = sdf(probe)

    if isinstance(out, tuple) and len(out) == 2:
        def eval_vd(pts: NDArray) -> tuple[NDArray, NDArray]:
            v, g = sdf(pts)
            v = np.asarray(v, dtype=np.float64).ravel()
            g = np.asarray(g, dtype=np.float64).reshape(-1, 3)
            dphi = g @ u
            if v.size == 1 and len(pts) != 1:
                v = np.full(len(pts), float(v[0]), dtype=np.float64)
                dphi = np.full(len(pts), float(dphi.ravel()[0]), dtype=np.float64)
            return v, dphi
        return eval_vd

    # fallback: values-only + numeric directional derivative
    vals = _make_values_only_vectorized(sdf)
    h = float(numeric_h)
    du = u * h

    def eval_vd(pts: NDArray) -> tuple[NDArray, NDArray]:
        pts = np.asarray(pts, dtype=np.float64)
        v0 = vals(pts)
        pm = np.concatenate([pts + du, pts - du], axis=0)
        vm = vals(pm)
        n = len(pts)
        dphi = (vm[:n] - vm[n:]) / (2.0 * h)
        return v0, dphi

    return eval_vd


def _segment_aabb_t_range(
    p0: NDArray,
    p1: NDArray,
    box_min: NDArray,
    box_max: NDArray,
    *,
    eps_dir: float = 1.0e-15
) -> Optional[Tuple[float, float]]:
    """
    Parametric overlap of segment p(t)=p0+t*(p1-p0), t in [0,1] with AABB.
    Returns (t_lo, t_hi) or None if no intersection.
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    box_min = np.asarray(box_min, dtype=np.float64).reshape(3)
    box_max = np.asarray(box_max, dtype=np.float64).reshape(3)

    d = p1 - p0
    t_lo, t_hi = 0.0, 1.0

    for i in range(3):
        di = float(d[i])
        if abs(di) < eps_dir:
            if (p0[i] < box_min[i]) or (p0[i] > box_max[i]):
                return None
            continue

        inv = 1.0 / di
        t1 = (box_min[i] - p0[i]) * inv
        t2 = (box_max[i] - p0[i]) * inv
        t_near = t1 if t1 < t2 else t2
        t_far  = t2 if t1 < t2 else t1

        if t_near > t_lo:
            t_lo = t_near
        if t_far < t_hi:
            t_hi = t_far
        if t_lo > t_hi:
            return None

    t_lo = max(t_lo, 0.0)
    t_hi = min(t_hi, 1.0)
    if t_lo > t_hi:
        return None
    return float(t_lo), float(t_hi)


def sdf_segment_minimum_point(
    sdf: Callable[[NDArray], Any],
    aabb: Tuple[NDArray, NDArray],
    seg_a: NDArray,
    seg_b: NDArray,
    *,
    # accuracy:
    spt: float = 1.0e-3,            # spatial tolerance along the segment (arc-length)
    val_tol: float = 0.0,           # optional value tolerance: stop once global gap <= val_tol
    # bounds:
    lipschitz: float = 1.0,         # along arc-length (true SDF: 1)
    curv_bound: Optional[float] = None,
    # if you don't have curv_bound, you can enable a heuristic monotonic stop:
    monotonic_heuristic: bool = False,
    dphi_eps: float = 1.0e-8,
    curvature_safety: float = 8.0,
    # performance:
    max_intervals: int = 250_000,
    batch_size: int = 4096,
    init_samples: int = 0,
) -> Tuple[float, Dict[str, Any], NDArray]:
    """
    Global minimization of sdf along the portion of the segment inside `aabb`.

    Adds gradient-based early stopping:
      - Rigorous monotonic-interval stopping if `curv_bound` is provided.
        Interpretation: along arc-length s, |d/ds (∇f(p(s))·u)| <= curv_bound.
      - Optional heuristic monotonic stopping if `monotonic_heuristic=True`.

    Returns: (min_value, info_dict, point)
    """
    box_min = np.asarray(aabb[0], dtype=np.float64).reshape(3)
    box_max = np.asarray(aabb[1], dtype=np.float64).reshape(3)
    seg_a = np.asarray(seg_a, dtype=np.float64).reshape(3)
    seg_b = np.asarray(seg_b, dtype=np.float64).reshape(3)

    if np.any(box_min >= box_max):
        raise ValueError("Invalid AABB: min must be strictly smaller than max per-axis.")

    t_rng = _segment_aabb_t_range(seg_a, seg_b, box_min, box_max)
    if t_rng is None:
        raise ValueError("Segment does not intersect the provided AABB.")

    t_lo, t_hi = t_rng
    d = seg_b - seg_a
    seg_len = float(np.linalg.norm(d))
    if seg_len == 0.0:
        v = float(_make_values_only_vectorized(sdf)(seg_a.reshape(1, 3))[0])
        info = {
            "p": seg_a.copy(), "t": 0.0, "t_range": (t_lo, t_hi),
            "segment_length": 0.0, "processed_intervals": 0,
            "intersects_surface": (v <= 0.0), "min_distance_to_surface": float(max(v, 0.0)),
        }
        return v, info, seg_a.copy()

    u = d / seg_len  # unit direction for arc-length derivatives
    eval_vd = _make_val_dirderiv_vectorized(sdf, u)
    val_only = _make_values_only_vectorized(sdf)  # for final confirm

    def pts_from_t(ts: NDArray) -> NDArray:
        return seg_a[None, :] + ts[:, None] * d[None, :]

    L = float(lipschitz)
    spt = float(spt)
    val_tol = float(val_tol)

    # ------------------------------------------------------------------ init incumbent + root interval stats
    tc = 0.5 * (t_lo + t_hi)
    if init_samples and init_samples >= 2:
        ts = np.linspace(t_lo, t_hi, int(init_samples), dtype=np.float64)
        vS, _ = eval_vd(pts_from_t(ts))
        j = int(np.argmin(vS))
        best_val = float(vS[j])
        best_t = float(ts[j])
        # still need (t_lo, t_hi, tc) details for root item
        t_init = np.array([t_lo, t_hi, tc], dtype=np.float64)
        v_init, d_init = eval_vd(pts_from_t(t_init))
    else:
        t_init = np.array([t_lo, t_hi, tc], dtype=np.float64)
        v_init, d_init = eval_vd(pts_from_t(t_init))
        j = int(np.argmin(v_init))
        best_val = float(v_init[j])
        best_t = float(t_init[j])

    v0, v1, vm = map(float, v_init)
    d0, d1, dm = map(float, d_init)

    ht = 0.5 * (t_hi - t_lo)
    half_len = ht * seg_len  # in arc-length units
    lb = max(vm - L * half_len, 0.5 * (v0 + v1) - L * half_len)

    # Heap item:
    # (lower_bound, tc, ht, v0, v1, vm, d0, d1, dm)
    pq: list[tuple[float, float, float, float, float, float, float, float, float]] = [
        (float(lb), float(tc), float(ht), float(v0), float(v1), float(vm), float(d0), float(d1), float(dm))
    ]
    heapq.heapify(pq)

    processed = 0

    def prune_threshold(current_best: float) -> float:
        # if val_tol>0, treat intervals whose best possible value is within val_tol as “done”
        return current_best - val_tol

    # ------------------------------------------------------------------ branch and bound
    while pq and processed < max_intervals:
        if pq[0][0] >= prune_threshold(best_val):
            break

        batch: list[tuple[float, float, float, float, float, float, float, float]] = []
        # stores: (tc, ht, v0, v1, vm, d0, d1, dm) for splitting

        n_pop = min(batch_size, max_intervals - processed)
        for _ in range(n_pop):
            if not pq:
                break
            lb, tc, ht, v0, v1, vm, d0, d1, dm = heapq.heappop(pq)
            processed += 1

            if lb >= prune_threshold(best_val):
                continue

            half_len = ht * seg_len
            if half_len <= spt:
                # terminal: update incumbent from known samples (endpoints + midpoint)
                if v0 < best_val:
                    best_val = v0
                    best_t = tc - ht
                if vm < best_val:
                    best_val = vm
                    best_t = tc
                if v1 < best_val:
                    best_val = v1
                    best_t = tc + ht
                continue

            # -------------------------------------------------- gradient-based monotonic stopping
            # We consider phi(s) = sdf(p(s)), where s is arc-length.
            # phi'(s) = ∇sdf(p)·u. If we know |phi''(s)| <= K on the interval,
            # then |phi'(s) - phi'(s_mid)| <= K * half_len.
            # So if |phi'(s_mid)| > K*half_len, phi' can't cross 0 => phi monotone => min at endpoint.
            if curv_bound is not None:
                K = float(curv_bound)
                if abs(dm) > K * half_len:
                    if dm > 0.0:
                        # increasing => minimum at left endpoint
                        if v0 < best_val:
                            best_val = v0
                            best_t = tc - ht
                    else:
                        # decreasing => minimum at right endpoint
                        if v1 < best_val:
                            best_val = v1
                            best_t = tc + ht
                    continue

            elif monotonic_heuristic:
                # Quick sign-consistency heuristic (very cheap, not guaranteed)
                if (d0 > dphi_eps and dm > dphi_eps and d1 > dphi_eps):
                    if v0 < best_val:
                        best_val = v0
                        best_t = tc - ht
                    continue
                if (d0 < -dphi_eps and dm < -dphi_eps and d1 < -dphi_eps):
                    if v1 < best_val:
                        best_val = v1
                        best_t = tc + ht
                    continue

                # Slightly stronger heuristic: estimate curvature from sampled derivatives,
                # inflate it, then apply the same “no zero crossing possible” test.
                K_est = curvature_safety * max(abs(dm - d0), abs(d1 - dm)) / max(half_len, 1e-30)
                if abs(dm) > K_est * half_len:
                    if dm > 0.0:
                        if v0 < best_val:
                            best_val = v0
                            best_t = tc - ht
                    else:
                        if v1 < best_val:
                            best_val = v1
                            best_t = tc + ht
                    continue

            # -------------------------------------------------- split this interval
            batch.append((tc, ht, v0, v1, vm, d0, d1, dm))

        if not batch:
            continue

        B = len(batch)
        tc_arr = np.fromiter((b[0] for b in batch), dtype=np.float64, count=B)
        ht_arr = np.fromiter((b[1] for b in batch), dtype=np.float64, count=B)
        v0_arr = np.fromiter((b[2] for b in batch), dtype=np.float64, count=B)
        v1_arr = np.fromiter((b[3] for b in batch), dtype=np.float64, count=B)
        vm_arr = np.fromiter((b[4] for b in batch), dtype=np.float64, count=B)
        d0_arr = np.fromiter((b[5] for b in batch), dtype=np.float64, count=B)
        d1_arr = np.fromiter((b[6] for b in batch), dtype=np.float64, count=B)
        dm_arr = np.fromiter((b[7] for b in batch), dtype=np.float64, count=B)

        ht_child = 0.5 * ht_arr
        tL = tc_arr - ht_child
        tR = tc_arr + ht_child

        # Evaluate both child midpoints in one vectorized call
        t_eval = np.empty((2 * B,), dtype=np.float64)
        t_eval[0::2] = tL
        t_eval[1::2] = tR
        v_eval, d_eval = eval_vd(pts_from_t(t_eval))
        vL = v_eval[0::2]
        vR = v_eval[1::2]
        dL = d_eval[0::2]
        dR = d_eval[1::2]

        # Update incumbent from newly evaluated points
        j = int(np.argmin(v_eval))
        vj = float(v_eval[j])
        if vj < best_val:
            best_val = vj
            best_t = float(t_eval[j])

        # Compute child lower bounds (still purely Lipschitz-safe)
        half_len_child = ht_child * seg_len
        lbL = np.maximum(vL - L * half_len_child, 0.5 * (v0_arr + vm_arr) - L * half_len_child)
        lbR = np.maximum(vR - L * half_len_child, 0.5 * (vm_arr + v1_arr) - L * half_len_child)

        can_split = half_len_child > spt
        thr = prune_threshold(best_val)

        maskL = can_split & (lbL < thr)
        maskR = can_split & (lbR < thr)

        # Push competitive non-terminal children
        if np.any(maskL):
            idx = np.flatnonzero(maskL)
            for i in idx.tolist():
                heapq.heappush(
                    pq,
                    (float(lbL[i]),
                     float(tL[i]), float(ht_child[i]),
                     float(v0_arr[i]), float(vm_arr[i]), float(vL[i]),
                     float(d0_arr[i]), float(dm_arr[i]), float(dL[i]))
                )

        if np.any(maskR):
            idx = np.flatnonzero(maskR)
            for i in idx.tolist():
                heapq.heappush(
                    pq,
                    (float(lbR[i]),
                     float(tR[i]), float(ht_child[i]),
                     float(vm_arr[i]), float(v1_arr[i]), float(vR[i]),
                     float(dm_arr[i]), float(d1_arr[i]), float(dR[i]))
                )

    # ------------------------------------------------------------------ final report
    p_best = seg_a + best_t * d
    best_val = float(val_only(p_best.reshape(1, 3))[0])

    info: Dict[str, Any] = {
        "p": p_best.copy(),
        "t": float(best_t),
        "t_range": (float(t_lo), float(t_hi)),
        "segment_length": float(seg_len),
        "processed_intervals": int(processed),
        "heap_size": int(len(pq)),
        "intersects_surface": (best_val <= 0.0),
        "min_distance_to_surface": float(max(best_val, 0.0)),
    }
    return best_val, info, p_best


import heapq
from typing import Callable, Tuple, Dict, Any

import numpy as np
NDArray = np.ndarray

# -----------------------------------------------------------------------------
# Precomputed octant sign offsets (used for all subdivisions)
# -----------------------------------------------------------------------------
_SIGNS8: NDArray = np.array(
    [
        (-1.0, -1.0, -1.0),
        (+1.0, -1.0, -1.0),
        (-1.0, +1.0, -1.0),
        (+1.0, +1.0, -1.0),
        (-1.0, -1.0, +1.0),
        (+1.0, -1.0, +1.0),
        (-1.0, +1.0, +1.0),
        (+1.0, +1.0, +1.0),
    ],
    dtype=np.float64,
)

# -----------------------------------------------------------------------------
# Fast “values-only” wrapper:
#   - assumes `sdf(pts)` accepts pts shaped (N,3)
#   - supports sdf returning either values or (values, grad)
#   - probes once to avoid per-call try/except overhead
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Vectorized central-difference gradient + Hessian (3D) from values only:
#   Uses 19 points in ONE vectorized call.
# -----------------------------------------------------------------------------
def _numeric_grad_hess_value_vec(
    val_fun: Callable[[NDArray], NDArray],
    p: NDArray,
    h: float = 5.0e-4,
) -> tuple[float, NDArray, NDArray]:
    p = np.asarray(p, dtype=np.float64).reshape(3)
    h = float(h)

    pts = np.empty((19, 3), dtype=np.float64)
    pts[0] = p
    eye = np.eye(3, dtype=np.float64) * h

    # +/- axis points
    pts[1:4] = p + eye
    pts[4:7] = p - eye

    # mixed pairs: (x,y), (x,z), (y,z)
    k = 7
    for i, j in ((0, 1), (0, 2), (1, 2)):
        ei, ej = eye[i], eye[j]
        pts[k + 0] = p + ei + ej
        pts[k + 1] = p + ei - ej
        pts[k + 2] = p - ei + ej
        pts[k + 3] = p - ei - ej
        k += 4

    f = val_fun(pts)
    f = np.asarray(f, dtype=np.float64).ravel()
    if f.shape[0] != 19:
        raise ValueError("val_fun did not return 19 values for the 19-point stencil.")

    f0 = float(f[0])
    g = (f[1:4] - f[4:7]) / (2.0 * h)

    H = np.empty((3, 3), dtype=np.float64)
    inv_h2 = 1.0 / (h * h)

    # diagonal
    H[0, 0] = (f[1] - 2.0 * f[0] + f[4]) * inv_h2
    H[1, 1] = (f[2] - 2.0 * f[0] + f[5]) * inv_h2
    H[2, 2] = (f[3] - 2.0 * f[0] + f[6]) * inv_h2

    # mixed (central) terms
    # xy -> f[7:11]
    vpp, vpm, vmp, vmm = f[7], f[8], f[9], f[10]
    H[0, 1] = H[1, 0] = (vpp - vpm - vmp + vmm) / (4.0 * h * h)

    # xz -> f[11:15]
    vpp, vpm, vmp, vmm = f[11], f[12], f[13], f[14]
    H[0, 2] = H[2, 0] = (vpp - vpm - vmp + vmm) / (4.0 * h * h)

    # yz -> f[15:19]
    vpp, vpm, vmp, vmm = f[15], f[16], f[17], f[18]
    H[1, 2] = H[2, 1] = (vpp - vpm - vmp + vmm) / (4.0 * h * h)

    return f0, g, H


# -----------------------------------------------------------------------------
# Optimized global branch-and-bound
# Key speed ideas:
#   - NO scalar fallback; assumes vectorized SDF evaluation
#   - tighter bounds: keep the true AABB (no cube padding)
#   - batched expansion: pop many boxes, evaluate all their children in one call
#   - do NOT enqueue terminal boxes (child half-sizes <= spt), since they can’t be expanded
#   - in-place np.maximum for f(x) = max(sdfA, sdfB) to reduce allocations
# -----------------------------------------------------------------------------
def sdf_intersection_deepest_point(
    sdfA: Callable[[NDArray], Any],
    sdfB: Callable[[NDArray], Any],
    aabbA: Tuple[NDArray, NDArray],
    aabbB: Tuple[NDArray, NDArray],
    *,
    spt: float = 1.0e-3,
    max_boxes: int = 250_000,
    batch_size: int = 2048,
    init_grid: int = 0,
    polish: bool = True,
    newton_h: float = 5.0e-4,
    newton_eps_grad: float = 1.0e-12,
    newton_eps_det: float = 1.0e-12,
    newton_eps_step: float = 1.0e-10,
    newton_max_iter: int = 30,
) -> Tuple[float, Dict[str, Any], NDArray]:
    # ❶ AABB intersection
    box_min = np.maximum(aabbA[0], aabbB[0]).astype(np.float64)
    box_max = np.minimum(aabbA[1], aabbB[1]).astype(np.float64)
    if np.any(box_min >= box_max):
        raise ValueError("AABB intersection is empty – nothing to search.")

    spt = float(spt)
    max_boxes = int(max_boxes)
    batch_size = int(batch_size)

    # ❷ Fast, probed wrappers that return only values (vectorized)
    valA = _make_values_only_vectorized(sdfA)
    valB = _make_values_only_vectorized(sdfB)

    def f_vals(pts: NDArray) -> NDArray:
        # in-place max into A buffer to reduce allocations
        a = valA(pts)
        b = valB(pts)
        np.maximum(a, b, out=a)
        return a

    # ❸ Root box (rectangular prism), store half-lengths
    c_root = 0.5 * (box_min + box_max)
    h_root = 0.5 * (box_max - box_min)  # (hx, hy, hz)
    r_root = float(np.linalg.norm(h_root))  # half diagonal (L2)

    f_root = float(f_vals(c_root.reshape(1, 3))[0])
    best_val = f_root
    best_pt = c_root.copy()

    # Optional: coarse incumbent search (fast when SDF is highly vectorized)
    # This can dramatically reduce the number of boxes explored.
    if init_grid and init_grid >= 2:
        xs = np.linspace(box_min[0], box_max[0], init_grid, dtype=np.float64)
        ys = np.linspace(box_min[1], box_max[1], init_grid, dtype=np.float64)
        zs = np.linspace(box_min[2], box_max[2], init_grid, dtype=np.float64)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pts0 = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
        fv0 = f_vals(pts0)
        i0 = int(np.argmin(fv0))
        v0 = float(fv0[i0])
        if v0 < best_val:
            best_val = v0
            best_pt = pts0[i0].copy()

    # Heap item: (lower_bound, hx, hy, hz, cx, cy, cz)
    # lower_bound uses Lipschitz 1: lb = f(center) - r (r = half diagonal)
    lb_root = f_root - r_root
    pq: list[tuple[float, float, float, float, float, float, float]] = [
        (lb_root, float(h_root[0]), float(h_root[1]), float(h_root[2]),
         float(c_root[0]), float(c_root[1]), float(c_root[2]))
    ]
    heapq.heapify(pq)
    heappop = heapq.heappop
    heappush = heapq.heappush

    processed = 0

    # ❹ Batched best-first expansion
    while pq and processed < max_boxes:
        # global termination
        if pq[0][0] >= best_val:
            break

        centers_list: list[tuple[float, float, float]] = []
        half_list: list[tuple[float, float, float]] = []

        n_pop = min(batch_size, max_boxes - processed)
        for _ in range(n_pop):
            if not pq:
                break
            lb, hx, hy, hz, cx, cy, cz = heappop(pq)
            processed += 1

            # If the smallest lb is already >= best, all remaining are prunable.
            if lb >= best_val:
                pq.clear()
                break

            # Already terminal? (never expand; its center was evaluated when created)
            if (hx <= spt) and (hy <= spt) and (hz <= spt):
                continue

            centers_list.append((cx, cy, cz))
            half_list.append((hx, hy, hz))

        if not centers_list:
            continue

        centers = np.asarray(centers_list, dtype=np.float64)      # (B,3)
        half_sizes = np.asarray(half_list, dtype=np.float64)      # (B,3)
        child_half = half_sizes * 0.5                             # (B,3)

        # Evaluate all child centers in one big vectorized call
        child_centers = centers[:, None, :] + _SIGNS8[None, :, :] * child_half[:, None, :]
        pts = child_centers.reshape(-1, 3)                        # (8B,3)
        f_child = f_vals(pts)                                     # (8B,)

        # Update incumbent from these evaluations
        i_best = int(np.argmin(f_child))
        v_best = float(f_child[i_best])
        if v_best < best_val:
            best_val = v_best
            best_pt = pts[i_best].copy()

        # If child boxes are terminal (child_half <= spt in all dims), do not enqueue them.
        parent_need_push = np.any(child_half > spt, axis=1)       # (B,)
        if not np.any(parent_need_push):
            continue

        idx_par = np.flatnonzero(parent_need_push)                # parent indices to enqueue children for
        child_half_push = child_half[idx_par]                     # (Bp,3)
        r_child = np.linalg.norm(child_half_push, axis=1)         # (Bp,)

        # Reshape child values per parent
        f_mat = f_child.reshape(-1, 8)                            # (B,8)
        f_push = f_mat[idx_par]                                   # (Bp,8)

        # Child competitive condition:
        #   lb = f - r < best  <=>  f < best + r
        mask = f_push < (best_val + r_child[:, None])
        if not np.any(mask):
            continue

        sel_flat = np.flatnonzero(mask.ravel())
        par_sub = sel_flat >> 3                                   # which of Bp parents
        child_k = sel_flat & 7                                    # which octant (0..7)

        abs_parent = idx_par[par_sub]                             # map back to 0..B-1
        abs_child = (abs_parent << 3) + child_k                   # index into flattened children

        pts_sel = pts[abs_child]
        f_sel = f_child[abs_child]
        lb_sel = f_sel - r_child[par_sub]
        h_sel = child_half_push[par_sub]

        # Convert to Python lists (faster heap pushes than numpy scalars)
        lb_list = lb_sel.tolist()
        hx_list = h_sel[:, 0].tolist()
        hy_list = h_sel[:, 1].tolist()
        hz_list = h_sel[:, 2].tolist()
        cx_list = pts_sel[:, 0].tolist()
        cy_list = pts_sel[:, 1].tolist()
        cz_list = pts_sel[:, 2].tolist()

        for i in range(len(lb_list)):
            heappush(pq, (lb_list[i],
                          hx_list[i], hy_list[i], hz_list[i],
                          cx_list[i], cy_list[i], cz_list[i]))

    # ❺ Final evaluation dict + optional Newton polish
    def value_grad_at(
        sdf: Callable[[NDArray], Any],
        val_fun: Callable[[NDArray], NDArray],
        p: NDArray,
    ) -> tuple[float, NDArray]:
        # Always call sdf in vectorized form, even for a single point.
        p1 = np.asarray(p, dtype=np.float64).reshape(1, 3)
        out = sdf(p1)
        if isinstance(out, tuple) and len(out) == 2:
            v = float(np.asarray(out[0], dtype=np.float64).ravel()[0])
            g = np.asarray(out[1], dtype=np.float64).reshape(-1, 3)[0]
            return v, g

        v = float(np.asarray(out, dtype=np.float64).ravel()[0])
        # Numeric gradient via ONE 6-point vector call
        h = 1.0e-6
        eye = np.eye(3, dtype=np.float64) * h
        pts6 = np.vstack((p1[0] + eye, p1[0] - eye))
        vals6 = val_fun(pts6)
        g = (vals6[:3] - vals6[3:]) / (2.0 * h)
        return v, g

    p_cur = best_pt.copy()
    valA_pt, gradA = value_grad_at(sdfA, valA, p_cur)
    valB_pt, gradB = value_grad_at(sdfB, valB, p_cur)
    val_cur = max(valA_pt, valB_pt)

    eval_best: Dict[str, Any] = {
        "p": p_cur.copy(),
        "valA": valA_pt, "gradA": gradA,
        "valB": valB_pt, "gradB": gradB,
        "processed_boxes": processed,
    }

    if polish:
        for _ in range(newton_max_iter):
            # Choose which branch is active (max)
            if eval_best["valA"] >= eval_best["valB"]:
                val_fun = valA
            else:
                val_fun = valB

            _, g, H = _numeric_grad_hess_value_vec(val_fun, p_cur, h=newton_h)

            if np.linalg.norm(g, ord=1) < newton_eps_grad:
                break
            if abs(float(np.linalg.det(H))) < newton_eps_det:
                break

            step = -np.linalg.solve(H, g)
            if np.linalg.norm(step, ord=1) < newton_eps_step:
                break

            step_scale = 1.0
            improved = False
            while step_scale >= 0.125 and not improved:
                p_try = p_cur + step_scale * step
                p_try = np.minimum(np.maximum(p_try, box_min), box_max)

                vA = float(valA(p_try.reshape(1, 3))[0])
                vB = float(valB(p_try.reshape(1, 3))[0])
                v_try = max(vA, vB)

                if v_try < val_cur:
                    p_cur = p_try
                    val_cur = v_try
                    valA_pt, gradA = value_grad_at(sdfA, valA, p_cur)
                    valB_pt, gradB = value_grad_at(sdfB, valB, p_cur)
                    eval_best = {
                        "p": p_cur.copy(),
                        "valA": valA_pt, "gradA": gradA,
                        "valB": valB_pt, "gradB": gradB,
                        "processed_boxes": processed,
                    }
                    improved = True
                else:
                    step_scale *= 0.5

            if not improved:
                break

    return val_cur, eval_best, p_cur
if __name__ == "__main__":
    from mmcore.geom.primitives import Tube
    from dataclasses import dataclass, field, InitVar

    x, y, v, u, z = [
        [[12.359112840551504, -7.5948049557495425, 0.0], [2.656625109045951, 1.2155741170561933, 0.0]],
        [[7.14384241216015, -6.934735074711716, -0.1073366304415263], [7.0788761013028365, 10.016931402130641, 0.8727530304189204]],
        [
            [8.072688942425103, -2.3061831591019826, 0.2615779273274319],
            [7.173685617288537, -3.4427234423361512, 0.4324928834164773],
            [7.683972288682133, -2.74630545102506, 0.07413871667321925],
            [7.088944240699163, -4.61458155002528, -0.22460509818398067],
            [7.304629277158477, -3.9462033818505433, 0.8955725109783643],
            [7.304629277158477, -3.3362864951018985, 0.8955725109783643],
            [7.304629277158477, -2.477065729786164, 0.7989970582016114],
            [7.304629277158477, -2.0988672326949933, 0.7989970582016114],
        ],
        0.72648,
        1.0,
    ]

    aa = np.array(x)
    bb = np.array(y)

    t1 = Tube(aa[0], aa[1], z, thickness=0.2)
    t2 = Tube(bb[0], bb[1], u, thickness=0.2)
    vv = np.array(v)
    s=time.time()
    res=sdf_intersection_deepest_point(t1, t2, t1.bounds(),t2.bounds(),batch_size=256)
    print(res)

    print(time.time()-s)
