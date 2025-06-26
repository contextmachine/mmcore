from __future__ import annotations
import heapq
from typing import Callable, Tuple, Dict, Any

import numpy as np

from numpy.typing import NDArray


def _numeric_gradient(fun: Callable[[NDArray], float], x: NDArray, h: float = 1.0e-4, return_val=False, **kwargs) -> NDArray:
    """
    3‑point central finite difference gradient of a scalar field.
    """
    g = np.empty(3, dtype=float)
    for i in range(3):
        d = np.zeros(3)
        d[i] = h
        g[i] = (fun(x + d) - fun(x - d)) / (2.0 * h)
    if return_val:
        return fun(x), g
    return g


def _numeric_gradient_v(fun: Callable[[NDArray], float], x: NDArray, h: float = 1.0e-6, return_val=False) -> NDArray:
    g = np.empty(np.atleast_2d(x).shape)
    d = np.eye(3) * h
    h2 = 2 * h
    r = np.array([x, x + d[0], x - d[0], x + d[1], x - d[1], x + d[2], x - d[2]])

    result = fun(r)
    val = result[0]
    g[:, 0] = (result[1] - result[2]) / h2
    g[:, 1] = (result[3] - result[4]) / h2
    g[:, 2] = (result[5] - result[6]) / h2
    if x.ndim == 1:
        print(g.shape)
        g = g[0]

    if return_val:
        return val, g
    return g


def _numeric_hessian(grad_fun: Callable[[NDArray], NDArray], x: NDArray, h: float = 5.0e-4) -> NDArray:
    H = np.empty((3, 3), dtype=float)
    for i in range(3):
        d = np.zeros(3)
        d[i] = h
        g_plus = grad_fun(x + d)
        g_minus = grad_fun(x - d)
        H[:, i] = (g_plus - g_minus) / (2.0 * h)
    return 0.5 * (H + H.T)


##############################################################################
# SDF helpers – now vectorisation aware
##############################################################################


def _values_only(sdf: Callable[[NDArray], Any], pts: NDArray) -> NDArray:
    """
    *Best‑effort* vectorised call:

    * If `sdf(pts)` works directly, we use that.
    * Otherwise we fall back to per‑point evaluation (still transparent
      for the caller).
    * The returned array is always 1‑D with len == len(pts).
    """

    return sdf(pts)  # try vectorised


def _ensure_value_grad(sdf: Callable[[NDArray], Any], p: NDArray) -> Tuple[float, NDArray]:
    """
    Works for *scalar* query `p`.  (We never call this in bulk.)
    """

    # if isinstance(out, tuple) and len(out) == 2:
    #    return float(out[0]), np.asarray(out[1], dtype=float)
    # val_fun: Callable[[NDArray], float] = sdf        # type: ignore

    val, grad = _numeric_gradient_v(sdf, p, return_val=True)
    return val, grad


##############################################################################
# Main routine – vectorised divide‑and‑conquer
##############################################################################


def sdf_intersection_deepest_point_fast(
    sdfA: Callable[[NDArray], Any],
    sdfB: Callable[[NDArray], Any],
    aabbA: Tuple[NDArray, NDArray],
    aabbB: Tuple[NDArray, NDArray],
    *,
    spt: float = 1.0e-3,
    max_divide: int = 256,
    newton_eps_grad: float = 1.0e-12,
    newton_eps_det: float = 1.0e-12,
    newton_eps_step: float = 1.0e-10,
    newton_max_iter: int = 30,
) -> Tuple[float, Dict[str, Any], NDArray]:
    """
    Find the point that minimises  max(sdfA, sdfB)  inside AABB_A ∩ AABB_B.
    Returns  (value, evaluation_dict, point).
    """

    # ------------------------------------------------------------------ ❶  AABB set‑up
    box_min = np.maximum(aabbA[0], aabbB[0])
    box_max = np.minimum(aabbA[1], aabbB[1])
    if np.any(box_min >= box_max):
        raise ValueError("AABB intersection is empty – nothing to search.")

    # Tolerances in each direction (fixed = spt)
    xtol = ytol = ztol = float(spt)

    # ------------------------------------------------------------------ ❷  Divide‑and‑conquer (vectorised)
    xmin, ymin, zmin = box_min
    xmax, ymax, zmax = box_max
    n_div = 0

    while ((xmax - xmin) > xtol or (ymax - ymin) > ytol or (zmax - zmin) > ztol) and (n_div < max_divide):

        n_div += 1
        xmid, ymid, zmid = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))

        open_x = (xmax - xmin) > xtol
        open_y = (ymax - ymin) > ytol
        open_z = (zmax - zmin) > ztol

        xs = (xmin, xmid, xmax) if open_x else (xmid,)
        ys = (ymin, ymid, ymax) if open_y else (ymid,)
        zs = (zmin, zmid, zmax) if open_z else (zmid,)

        # Cartesian product – vectorised
        grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T  # (N,3)

        # SDF values (vectorised where possible)
        valsA = _values_only(sdfA, grid)
        valsB = _values_only(sdfB, grid)
        vals = np.maximum(valsA, valsB)  # (N,)

        idx_min = int(np.argmin(vals))
        best_p = grid[idx_min]
        best_val = float(vals[idx_min])

        # shrink the search box (¼ width in open directions)
        if open_x:
            h = 0.25 * (xmax - xmin)
            xmin = max(best_p[0] - h, box_min[0])
            xmax = min(best_p[0] + h, box_max[0])
        if open_y:
            h = 0.25 * (ymax - ymin)
            ymin = max(best_p[1] - h, box_min[1])
            ymax = min(best_p[1] + h, box_max[1])
        if open_z:
            h = 0.25 * (zmax - zmin)
            zmin = max(best_p[2] - h, box_min[2])
            zmax = min(best_p[2] + h, box_max[2])
        print(
            xmid,
            ymid,
            zmid,
        )
    # ------------------------------------------------------------------ ❸  Robust final check (vectorised)
    xmid, ymid, zmid = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))
    final_pts = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmin, ymax, zmax],
            [xmax, ymax, zmax],
            [xmid, ymid, zmid],
        ]
    )

    valsA = _values_only(sdfA, final_pts)
    valsB = _values_only(sdfB, final_pts)
    vals = np.maximum(valsA, valsB)
    idx_min = int(np.argmin(vals))
    p_cur = final_pts[idx_min].copy()

    # full evaluation dict for the best point
    valA, gradA = _ensure_value_grad(sdfA, p_cur)
    valB, gradB = _ensure_value_grad(sdfB, p_cur)
    val_cur = max(valA, valB)
    eval_cur = {
        "p": p_cur,
        "valA": valA,
        "gradA": gradA,
        "valB": valB,
        "gradB": gradB,
    }

    # ------------------------------------------------------------------ ❹  Guarded Newton refinement (single point)
    for _ in range(newton_max_iter):

        active_is_A = eval_cur["valA"] >= eval_cur["valB"]
        if active_is_A:
            g = eval_cur["gradA"]
            val_fun = lambda q: _ensure_value_grad(sdfA, q)[0]
            grad_fun = lambda q: _ensure_value_grad(sdfA, q)[1]
        else:
            g = eval_cur["gradB"]
            val_fun = lambda q: _ensure_value_grad(sdfB, q)[0]
            grad_fun = lambda q: _ensure_value_grad(sdfB, q)[1]

        if np.linalg.norm(g, ord=1) < newton_eps_grad:
            break

        H = _numeric_hessian(grad_fun, p_cur)
        if abs(np.linalg.det(H)) < newton_eps_det:
            break

        step = -np.linalg.solve(H, g)
        if np.linalg.norm(step, ord=1) < newton_eps_step:
            break

        step_scale = 1.0
        improved = False
        while step_scale >= 0.125 and not improved:

            p_try = p_cur + step_scale * step
            p_try = np.minimum(np.maximum(p_try, box_min), box_max)

            valA_t, gradA_t = _ensure_value_grad(sdfA, p_try)
            valB_t, gradB_t = _ensure_value_grad(sdfB, p_try)
            val_t = max(valA_t, valB_t)

            if val_t < val_cur:
                p_cur, val_cur = p_try, val_t
                eval_cur = {
                    "p": p_cur,
                    "valA": valA_t,
                    "gradA": gradA_t,
                    "valB": valB_t,
                    "gradB": gradB_t,
                }
                improved = True
            else:
                step_scale *= 0.5
        if not improved:
            break

    # ------------------------------------------------------------------ ❺
    return val_cur, eval_cur, p_cur


def sdf_intersection_deepest_point_exact(
    sdfA: Callable[[NDArray], Any],
    sdfB: Callable[[NDArray], Any],
    aabbA: Tuple[NDArray, NDArray],
    aabbB: Tuple[NDArray, NDArray],
    *,
    spt: float = 1.0e-3,
    max_boxes: int = 250_000,
    newton_eps_grad: float = 1.0e-12,
    newton_eps_det: float = 1.0e-12,
    newton_eps_step: float = 1.0e-10,
    newton_max_iter: int = 30,
) -> Tuple[float, Dict[str, Any], NDArray]:
    """
    ★ Guaranteed global search (within `spt`) for  min  max(sdfA, sdfB)
      inside  AABB_A ∩ AABB_B.

    Returns (min_value, evaluation_dict, point).
    """

    # ------------------------------------------------------------------ ❶  AABB intersection & sanity checks
    box_min = np.maximum(aabbA[0], aabbB[0])
    box_max = np.minimum(aabbA[1], aabbB[1])
    if np.any(box_min >= box_max):
        raise ValueError("AABB intersection is empty – nothing to search.")

    # Edge length of the root cube (we keep boxes cubic for simpler bounds)
    root_edge = np.max(box_max - box_min)
    pad = (root_edge - (box_max - box_min)) / 2.0
    box_min -= pad
    box_max += pad

    # ------------------------------------------------------------------ ❷  Helper lambdas
    def f_vals(pts: NDArray) -> NDArray:
        return np.maximum(_values_only(sdfA, pts), _values_only(sdfB, pts))

    def box_lower_bound(center: NDArray, f_center: float, edge: float) -> float:
        # Lipschitz:  f(x) ≥ f(c) − r   with  r = radius = diagonal/2
        radius = 0.8660254037844386 * edge  # √3/2 ≈ 0.8660
        return f_center - radius

    # ------------------------------------------------------------------ ❸  Priority queue initialised with root cube
    # Each item  = (lower_bound, f_center, edge, center_x, center_y, center_z)
    c_root = 0.5 * (box_min + box_max)
    f_root = float(f_vals(c_root[None, :]))
    lb_root = box_lower_bound(c_root, f_root, root_edge)
    pq: list[tuple[float, float, float, float, float, float]] = [(lb_root, f_root, root_edge, *c_root)]
    heapq.heapify(pq)

    # Current best
    best_val = f_root
    best_pt = c_root.copy()

    # ------------------------------------------------------------------ ❹  Branch‑and‑bound loop
    processed = 0
    while pq and processed < max_boxes:
        lb, f_c, edge, cx, cy, cz = heapq.heappop(pq)
        processed += 1

        # prune boxes that cannot beat the current best
        if lb >= best_val:
            if pq and pq[0][0] >= best_val:
                break  # all remaining boxes worse -> done
            continue

        if edge * 0.5 <= spt:  # cube is already within tolerance
            if f_c < best_val:
                best_val = f_c
                best_pt = np.array([cx, cy, cz])
            continue

        # -------- subdivide into 8 octants --------------------------------
        half = 0.5 * edge
        quarter = 0.5 * half
        offsets = np.array(
            [
                [-quarter, -quarter, -quarter],
                [+quarter, -quarter, -quarter],
                [-quarter, +quarter, -quarter],
                [+quarter, +quarter, -quarter],
                [-quarter, -quarter, +quarter],
                [+quarter, -quarter, +quarter],
                [-quarter, +quarter, +quarter],
                [+quarter, +quarter, +quarter],
            ]
        )
        centers = offsets + np.array([cx, cy, cz])
        f_centers = f_vals(centers)

        # update global best from the 8 samples
        idx_improve = np.argmin(f_centers)
        if f_centers[idx_improve] < best_val:
            best_val = float(f_centers[idx_improve])
            best_pt = centers[idx_improve].copy()

        # push children
        for c, fc in zip(centers, f_centers):
            lb_child = box_lower_bound(c, float(fc), half)
            if lb_child < best_val:  # only keep competitive boxes
                heapq.heappush(pq, (lb_child, float(fc), half, *c))

    # ------------------------------------------------------------------ ❺  Precise evaluation + Newton polish
    valA, gradA = _ensure_value_grad(sdfA, best_pt)
    valB, gradB = _ensure_value_grad(sdfB, best_pt)
    best_val = max(valA, valB)
    eval_best = {"p": best_pt, "valA": valA, "gradA": gradA, "valB": valB, "gradB": gradB}

    p_cur = best_pt.copy()
    val_cur = best_val

    for _ in range(newton_max_iter):
        active_is_A = eval_best["valA"] >= eval_best["valB"]
        if active_is_A:
            g = eval_best["gradA"]
            val_fun = lambda q: _ensure_value_grad(sdfA, q)[0]
            grad_fun = lambda q: _ensure_value_grad(sdfA, q)[1]
        else:
            g = eval_best["gradB"]
            val_fun = lambda q: _ensure_value_grad(sdfB, q)[0]
            grad_fun = lambda q: _ensure_value_grad(sdfB, q)[1]

        if np.linalg.norm(g, ord=1) < newton_eps_grad:
            break

        H = _numeric_hessian(grad_fun, p_cur)
        if abs(np.linalg.det(H)) < newton_eps_det:
            break

        step = -np.linalg.solve(H, g)
        if np.linalg.norm(step, ord=1) < newton_eps_step:
            break

        step_scale = 1.0
        improved = False
        while step_scale >= 0.125 and not improved:
            p_try = p_cur + step_scale * step
            # clamp to original intersection box to avoid unsafe evals
            p_try = np.minimum(np.maximum(p_try, box_min), box_max)

            vA, gA = _ensure_value_grad(sdfA, p_try)
            vB, gB = _ensure_value_grad(sdfB, p_try)
            vTry = max(vA, vB)
            if vTry < val_cur:
                p_cur, val_cur = p_try, vTry
                eval_best = {"p": p_cur, "valA": vA, "gradA": gA, "valB": vB, "gradB": gB}
                improved = True
            else:
                step_scale *= 0.5
        if not improved:
            break

    # ------------------------------------------------------------------ ❻  Done
    return val_cur, eval_best, p_cur
