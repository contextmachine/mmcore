import functools
import itertools
import os
from typing import Optional, Tuple

import numpy as np
from mmcore.geom import nurbs

from mmcore.numeric.vectors import vector_projection, scalar_dot, scalar_norm, dot

from mmcore.geom.bvh import Object3D, find_closest
from mmcore.geom.nurbs import NURBSCurve


from mmcore.geom.polygon import BoundingBox
from mmcore.geom.surfaces import Surface
from mmcore.numeric.numeric import divide_interval
from mmcore.numeric.aabb import aabb_overlap
from mmcore.numeric.fdm import PDE
from mmcore.numeric.newton.cnewton import newtons_method

from mmcore.numeric.divide_and_conquer import iterative_divide_and_conquer_min, divide_and_conquer_min_2d, \
    divide_and_conquer_min_2d_vectorized

from scipy.optimize import newton
import multiprocessing as mp

from mmcore.numeric.fdm import bounded_fdm

import math

# Utility function to calculate the Euclidean distance between two points
import math

from mmcore.geom._nurbs_eval import (
    NURBSSurfaceTuple,
    NURBSCurveTuple,
    evaluate_nurbs_surface,
    evaluate_nurbs_curve,
    _surface_interval,
    _nurbs_to_tuple,
    _curve_interval,
)

from mmcore.geom._nurbs_knots import decompose_surface,decompose_curve


# Utility function to calculate the Euclidean distance between two points
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Utility function to find the closest distance in a strip
def strip_closest(strip, d):
    min_dist = d[0]
    pair = d[1]

    # Sort the strip according to y-coordinate
    strip.sort(key=lambda p: p[1])

    # Compare each point with the next points within min_dist in the strip
    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and (strip[j][1] - strip[i][1]) < min_dist:
            current_dist = dist(strip[i], strip[j])
            if current_dist < min_dist:
                min_dist = current_dist
                pair = (strip[i][2], strip[j][2])  # Store the indices of the closest pair
            j += 1

    return (min_dist, pair)


# Recursive function to find the smallest distance and the corresponding pair of indices
def closest_util(points_sorted_x, points_sorted_y, n):
    # Base case: Use brute force for 3 or fewer points
    if n <= 3:
        min_dist = float('inf')
        pair = (-1, -1)
        for i in range(n):
            for j in range(i + 1, n):
                current_dist = dist(points_sorted_x[i], points_sorted_x[j])
                if current_dist < min_dist:
                    min_dist = current_dist
                    pair = (points_sorted_x[i][2], points_sorted_x[j][2])  # Store indices
        return (min_dist, pair)

    # Find the middle point
    mid = n // 2
    mid_point = points_sorted_x[mid]

    # Divide points_sorted_y into left and right halves
    points_sorted_y_left = [point for point in points_sorted_y if point[0] <= mid_point[0]]
    points_sorted_y_right = [point for point in points_sorted_y if point[0] > mid_point[0]]

    # Recursively find the smallest distances in the left and right halves
    dl = closest_util(points_sorted_x[:mid], points_sorted_y_left, mid)
    dr = closest_util(points_sorted_x[mid:], points_sorted_y_right, n - mid)

    # Determine the smaller distance and the corresponding pair
    if dl[0] < dr[0]:
        d = dl
    else:
        d = dr

    # Build the strip of points within distance d from the midline
    strip = [point for point in points_sorted_y if abs(point[0] - mid_point[0]) < d[0]]

    # Find the closest pair in the strip
    strip_closest_dist = strip_closest(strip, d)

    # Return the overall minimum distance and the corresponding pair
    if strip_closest_dist[0] < d[0]:
        return strip_closest_dist
    else:
        return d


# Function to find the closest pair of points and their indices
def min_distance(points):
    """
    Finds the smallest distance between any two points in the given list and returns the distance along with the indices of the points.

    # Example usage
    points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    min_dist, pair = min_distance(points)
    print(f"The smallest distance is {min_dist} between points at indices {pair}")
    # Output: The smallest distance is 1.4142135623730951 between points at indices (0, 5)

    :param points: List of tuples representing the points [(x1, y1), (x2, y2), ...]
    :return: A tuple containing the minimum distance and a tuple of the two point indices
    """
    # Enumerate points to keep track of their original indices
    enumerated_points = list(enumerate(points))

    # Represent each point as (x, y, index)
    points_with_index = [(x, y, idx) for idx, (x, y) in enumerated_points]

    # Sort the points based on x and y coordinates
    points_sorted_x = sorted(points_with_index, key=lambda p: p[0])
    points_sorted_y = sorted(points_with_index, key=lambda p: p[1])

    # Use the recursive utility to find the closest pair
    return closest_util(points_sorted_x, points_sorted_y, len(points_sorted_x))

from mmcore.numeric.newton import cnewton

from mmcore.geom.bvh import NURBSCurveObject3D, build_bvh,_find_closest_vicinity,BVHNode
from numpy.typing import NDArray
class _BVHN(BVHNode):
    object: Optional[NURBSCurveObject3D]
def _find_cls(bvh:_BVHN, point):
    if bvh.object is not None:
        sd=sdBox(point - bvh.bounding_box.min_point, bvh.bounding_box.dims)



    if bvh.left is not None and bvh.right is not None:

        left_sd = sdBox(point - bvh.left.bounding_box.min_point, bvh.left.bounding_box.dims)
        right_sd = sdBox(point - bvh.right.bounding_box.min_point, bvh.right.bounding_box.dims)
        if left_sd < right_sd:
            return _find_closest_vicinity(bvh.left, point)
        elif left_sd > right_sd:
            return _find_closest_vicinity(bvh.right, point)
        else:
            left, left_sd = _find_closest_vicinity(bvh.left, point)
            right, right_sd = _find_closest_vicinity(bvh.right, point)
            if left_sd <= right_sd:
                return left_sd
            else:
                return right_sd


from mmcore.numeric.newton.bounded import bounded_newtons_method


def closest_point_on_nurbs_curve(curve: NURBSCurve, point: NDArray[float], tol=1e-6, on_curve=False,max_iter=100)->tuple[bool, tuple[float,float]]:


    bvh = build_bvh([NURBSCurveObject3D(c) for c in nurbs.decompose_curve(curve)])
    rr = find_closest(bvh, point, breadth=not on_curve)


    if rr is None or len(rr[0])==0:
        return False,closest_point_on_nurbs_curve(curve,point,tol,on_curve=False)[1]
    def inner(crv):
        nonlocal tol
        a,b=crv.interval()


        def objective(t):
            d = (curve.evaluate(t[0]) - point)
            return scalar_dot(d, d)
        if on_curve:
            res=bounded_newtons_method(objective, [sum([a,b]) / 2], [(a,b)], tol=tol,min_value=0.)
        else:
            res=newtons_method(objective, np.array([(a+ b)/2]), tol=tol,max_iter=max_iter)
            #print(res)


        return res, objective(res)
    return True,sorted((inner(_curve.curve) for _curve in rr[0]),key=lambda x: x[1])[0]


def foot_point(S, P, s0, t0, partial_derivatives=None, epsilon=1e-6, alpha_max=20):
    """
    Find the foot point on the parametric surface S(s, t) closest to the given point P.
    """
    if partial_derivatives is None:
        _pde = PDE(S, dim=2)
        partial_derivatives = lambda uv: _pde(uv).T
    s, t = st = np.array([s0, t0])

    while True:
        p_i = S(st)
        e_s, e_t = partial_derivatives(st)
        # Solve the linear system for Δs and Δt
        A = np.array([
            [scalar_dot(e_s, e_s), scalar_dot(e_s, e_t)],
            [scalar_dot(e_s, e_t), scalar_dot(e_t, e_t)]
        ])
        b = np.array([
            scalar_dot(P - p_i, e_s),
            scalar_dot(P - p_i, e_t)
        ])
        delta = np.linalg.solve(A, b)
        delta_s, delta_t = delta
        q_i = p_i + delta_s * e_s + delta_t * e_t
        s_new = s + delta_s
        t_new = t + delta_t
        p_new = S(s_new, t_new)
        f1 = q_i - p_i
        f2 = p_new - q_i
        # Check convergence
        if np.linalg.norm(q_i - p_i) < epsilon:
            break
        # Newton step for the foot point on the tangent parabola
        a0 = scalar_dot(P - p_i, f1)
        a1 = 2 * scalar_dot(f2, P - p_i) - scalar_dot(f1, f1)
        a2 = -3 * scalar_dot(f1, f2)
        a3 = -2 * scalar_dot(f2, f2)
        alpha = 1 - (a0 + a1 + a2 + a3) / (a1 + 2 * a2 + 3 * a3)
        alpha = np.clip(alpha, 0, alpha_max)
        s = s + alpha * delta_s
        t = t + alpha * delta_t
        st[0] = s
        st[1] = t
    return S(s, t), s, t


def closest_point_on_curve_single(curve, point, tol=1e-3):
    """

    :param curve: The curve on which to find the closest point.
    :param point: The point for which to find the closest point on the curve.
    :param tol: The tolerance for the minimum finding algorithm. Defaults to 1e-5.
    :return: The closest point on the curve to the given point, distance.

    """
    _fn = getattr(curve, "evaluate", curve)

    def distance_func(t):
        return scalar_norm(point - _fn(t))

    t0, t1 = curve.interval()

    t_best, d_best = t0, distance_func(t0)
    t, d = t1, distance_func(t1)
    if d < d_best:
        t_best = t
        d_best = d

    for bnds in divide_interval(*curve.interval(), step=0.5):
        # t,d=find_best(distance_func, bnds, tol=tol)
        t, d = iterative_divide_and_conquer_min(distance_func, bnds, tol=tol)
        if d < d_best:
            t_best = t
            d_best = d

    return t_best, d_best


class _ClosestPointSolution:
    def __init__(self, curve, tol=1e-5):
        self.curve = curve
        self.tol = tol

    def __call__(self, point):
        return closest_point_on_curve_single(self.curve, point, tol=self.tol)


def closest_points_on_curve_mp(curve, points, tol=1e-3, workers=1):
    if workers == -1:
        workers = os.cpu_count()
    with mp.Pool(workers) as pool:
        solution = _ClosestPointSolution(curve, tol=tol)
        return list(pool.map(solution, points
                             ))


def closest_point_on_curve(curve, pts, tol=1e-3, workers=1):
    pts = pts if isinstance(pts, np.ndarray) else np.array(pts)

    if pts.ndim == 1:
        return closest_point_on_curve_single(curve, pts, tol=tol)

    if workers == 1:
        return [closest_point_on_curve_single(curve, pt, tol=tol) for pt in pts]
    else:
        return closest_points_on_curve_mp(curve, pts, tol=tol, workers=workers)


def local_closest_point_on_curve(curve, t0, point, tol=1e-3, **kwargs):
    def fun(t):
        # C' (u) •(C(u) - P)
        return scalar_dot(curve.derivative(t), curve.evaluate(t) - point)

    dfun = bounded_fdm(fun, curve.interval())
    res = newton(fun, t0, fprime=dfun, tol=tol, **kwargs)
    return res, np.linalg.norm(curve.evaluate(res) - point)


def closest_point_on_ray(ray, point):
    start, direction = ray

    return start + vector_projection(point - start, direction)


def closest_point_on_line(line, point):
    start, end = line
    direction = end - start
    return start + vector_projection(point - start, direction)

from mmcore.numeric.numeric import compute_parametric_tolerance_surface,compute_parametric_tolerance_curve
from mmcore.geom.bvh import BoundingBox,sdBox,contains_point,build_bvh
class NURBSSurfaceBvhObject(Object3D):
    def __init__(self,surf):
        self.surf=surf
        super().__init__(BoundingBox(*self.surf.bbox()))


_float64_eps=np.finfo(float).eps
import functools
import numpy as np
from typing import Optional, Tuple

NDArray = np.ndarray
NURBSCurveTuple = Tuple  # adjust to your real alias


def _nurbs_curve_closest_point_divide_and_conquer(
    curve: NURBSCurveTuple,
    point: NDArray,
    t_range: Optional[tuple[float, float]] = None,
    *,
    spt: float = 0.001,
    angle_tol: Optional[float] = None,
):
    """
    Return the closest point on *curve* to *point*.

    A coarse divide‑and‑conquer pass produces a tight parameter window;
    a damped Newton iteration then polishes the result while honouring
    that window.
    """
    # ---------------------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def fun(t: float):
        """distance, full evaluation, local parametric tolerance"""
        t_eval = evaluate_nurbs_curve(curve, t, d_order=2)
        dc = t_eval["C"] - point
        cur_tol = compute_parametric_tolerance_curve(
            t_eval["C1"],
            t_eval["C2"],
            spt=spt,
            angle_tol=angle_tol,
        )
        return np.linalg.norm(dc), t_eval, cur_tol      # (dist, dict, tTol)

    # ---------------------------------------------------------------------
    # ❶  Divide‑and‑conquer -------------------------------------------------
    t_lo_full, t_hi_full = _curve_interval(curve)

    if t_range is None:
        t_min, t_max = t_lo_full, t_hi_full
    else:
        t_min = max(t_range[0], t_lo_full)
        t_max = min(t_range[1], t_hi_full)

    # first tolerance probe
    _, _, t_tol = fun(t_min)

    while (t_max - t_min) > t_tol:
        width = t_max - t_min
        t_mid = 0.5 * (t_min + t_max)

        cand = [
            (fun(t_min), t_min),
            (fun(t_mid), t_mid),
            (fun(t_max), t_max),
        ]
        min_val, t_best = min(cand, key=lambda item: item[0][0])

        h = 0.25 * width
        t_min = max(t_best - h, t_min if t_range is None else t_range[0])
        t_max = min(t_best + h, t_max if t_range is None else t_range[1])

        t_tol = min_val[2]

    # 2  Robust final pick --------------------------------------------------
    t_mid = 0.5 * (t_min + t_max)
    final_cand = [
        (fun(t_min), t_min),
        (fun(t_mid), t_mid),
        (fun(t_max), t_max),
    ]
    min_val, t_best = min(final_cand, key=lambda item: item[0][0])

    # ---------------------------------------------------------------------
    # 3  Guarded Newton refinement -----------------------------------------
    MAX_NITER   = 15
    EPS_FPRIME  = 1.0e-14       # “small derivative”
    EPS_SECOND  = 1.0e-14       # “small 2nd derivative”
    EPS_DSTEP   = 1.0e-12       # min step considered meaningful

    # Keep a private, shrinking bracket; start with the final window
    t_low, t_high = t_min, t_max
    dist_best, eval_best, tol_best = min_val

    t_cur = t_best
    
    for _ in range(MAX_NITER):

        C  = eval_best["C"]
        C1 = eval_best["C1"]
        C2 = eval_best["C2"]

        dvec = C - point
        f_prime  = float(np.dot(dvec, C1))
        if abs(f_prime) < EPS_FPRIME:
            break                                # already at a stationary point

        f_second = float(np.dot(C1, C1) + np.dot(dvec, C2))
        if abs(f_second) < EPS_SECOND:
            break                                # Jacobian degenerate → stop

        # raw Newton step
        dt = -f_prime / f_second
        if abs(dt) < EPS_DSTEP:
            break                                # step too tiny → done

        # (a) keep step inside the bracket, (b) ensure descent
        success = False
        step_scale = 1.0
        while not success and step_scale > 0.125:  # 1 / 8th is our floor

            t_trial = t_cur + step_scale * dt

            # hard‑clip to window, avoiding extrapolation noise
            if t_trial < t_low:
                t_trial = t_low
            elif t_trial > t_high:
                t_trial = t_high

            dist_trial, eval_trial, tol_trial = fun(t_trial)

            if dist_trial < dist_best:           # accepted
                # tighten the bracket around the new best point
                if t_trial < t_cur:
                    t_high = t_cur
                elif t_trial > t_cur:
                    t_low = t_cur

                t_cur      = t_trial
                dist_best  = dist_trial
                eval_best  = eval_trial
                tol_best   = tol_trial
                success    = True
            else:
                step_scale *= 0.5               # back‑tracking

        if not success:
            break                               # Newton no longer makes progress

        if (t_high - t_low) <= tol_best:
            break                               # bracket smaller than local tol

    # ---------------------------------------------------------------------
    return (dist_best, eval_best, tol_best), t_cur
def _nurbs_surface_closest_point_divide_and_conquer(
    surf: NURBSSurfaceTuple,
    point: NDArray[float],
    x_range: Optional[tuple[float, float]] = None,
    y_range: Optional[tuple[float, float]] = None,
    *,
    spt: float = 0.001,
    angle_tol: Optional[float] = None,
):
    """
    Locate the (u,v) on *surf* that minimises |S(u,v) - point| using a
    divide‑and‑conquer search.  The search space is iteratively restricted
    until both parametric extents are within their *adaptive* tolerances.

    Compared with the original version, once one parametric direction
    is inside its tolerance we keep it fixed at its midpoint and refine
    only the other direction.  This avoids unnecessary surface evaluations.
    """
    # -------------------------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def fun(u: float, v: float):
        """
        Cached evaluation of
            ‑ distance  ‑ complete t‑evaluation dict  ‑ local (uTol,vTol)
        """
        t_eval = evaluate_nurbs_surface(surf, u, v, d_order=2)
        dc = t_eval["S"] - point
        cur_tol = compute_parametric_tolerance_surface(
            t_eval["Su"],
            t_eval["Sv"],
            t_eval["Suu"],
            t_eval["Suv"],
            t_eval["Svv"],
            spt=spt,
            angle_tol=angle_tol,
        )
        return np.linalg.norm(dc), t_eval, cur_tol  # -> (dist, dict, (uTol,vTol))

    # -------------------------------------------------------------------------
    interval_u, interval_v = _surface_interval(surf)
    if x_range is None:
        x_range = interval_u
    if y_range is None:
        y_range = interval_v

    x_min, x_max = x_range
    y_min, y_max = y_range

    # Initial tolerances at one corner
    _, _, (x_tol, y_tol) = fun(x_min, y_min)

    # ---------------------- main loop ----------------------------------------
    while (x_max - x_min) > x_tol or (y_max - y_min) > y_tol:
        x_width = x_max - x_min
        y_width = y_max - y_min
        x_mid = (x_min + x_max) / 2.0
        y_mid = (y_min + y_max) / 2.0

        # Build candidate list depending on which direction still needs work
        candidates = []

        both_open = (x_width > x_tol) and (y_width > y_tol)
        if both_open:
            # 9‑point stencil (unchanged from original)
            candidates.extend(
                [
                    (fun(x_min, y_min), (x_min, y_min)),
                    (fun(x_max, y_min), (x_max, y_min)),
                    (fun(x_min, y_max), (x_min, y_max)),
                    (fun(x_max, y_max), (x_max, y_max)),
                    (fun(x_mid, y_min), (x_mid, y_min)),
                    (fun(x_mid, y_max), (x_mid, y_max)),
                    (fun(x_min, y_mid), (x_min, y_mid)),
                    (fun(x_max, y_mid), (x_max, y_mid)),
                    (fun(x_mid, y_mid), (x_mid, y_mid)),
                ]
            )
        elif x_width > x_tol:  # Only u needs refinement
            candidates.extend(
                [
                    (fun(x_min, y_mid), (x_min, y_mid)),
                    (fun(x_mid, y_mid), (x_mid, y_mid)),
                    (fun(x_max, y_mid), (x_max, y_mid)),
                ]
            )
        else:  # Only v needs refinement
            candidates.extend(
                [
                    (fun(x_mid, y_min), (x_mid, y_min)),
                    (fun(x_mid, y_mid), (x_mid, y_mid)),
                    (fun(x_mid, y_max), (x_mid, y_max)),
                ]
            )

        # Select the best candidate
        min_val, (u_best, v_best) = min(candidates, key=lambda item: item[0][0])

        # Update intervals only in the directions still “open”
        if x_width > x_tol:
            h = 0.25 * x_width
            x_min = max(u_best - h, x_range[0])
            x_max = min(u_best + h, x_range[1])
        if y_width > y_tol:
            h = 0.25 * y_width
            y_min = max(v_best - h, y_range[0])
            y_max = min(v_best + h, y_range[1])

        # Refresh adaptive tolerances at the current best point
        x_tol, y_tol = min_val[2]

    # -------------------- robust final evaluation ----------------------------
    # Examine the mid‑point plus the current corners
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0

    final_candidates = [
        (fun(x_min, y_min), (x_min, y_min)),
        (fun(x_max, y_min), (x_max, y_min)),
        (fun(x_min, y_max), (x_min, y_max)),
        (fun(x_max, y_max), (x_max, y_max)),
        (fun(x_mid, y_mid), (x_mid, y_mid)),
    ]
    min_val, min_coords = min(final_candidates, key=lambda pair: pair[0][0])
    return min_val, min_coords
import itertools


def nurbs_curve_closest_point(self: NURBSCurveTuple, point: NDArray[float], spt: float = 0.001, angle_tol: float = None):
    candidates = decompose_curve(self)

    best_f = [float("inf"), {}, (None, None)]
    best_x = None
    for candidate in candidates:

        min_val, min_t = _nurbs_curve_closest_point_divide_and_conquer(candidate, point, spt=spt, angle_tol=angle_tol)
      
        if best_f[0] > min_val[0]:
            best_f = min_val
            best_x = min_t
    
    
    return best_x, (best_f[0], *best_f[1:])


def nurbs_surface_closest_point(self:NURBSSurfaceTuple, point:NDArray[float],spt:float=0.001, angle_tol:float=None):
    candidates=decompose_surface(self)

    best_f=[float('inf'),{},(None,None)]
    best_x=None
    for candidate in candidates:
      
        min_val,min_coords = _nurbs_surface_closest_point_divide_and_conquer(candidate,point,spt=spt, angle_tol=angle_tol)
       
        if best_f[0]>min_val[0]:
            best_f=min_val
            best_x=min_coords
        
    return best_x, (best_f[0],*best_f[1:])


def closest_point_on_surface(self: Surface, pt, tol=1e-3, bounds=None):
    if bounds is None:
        bounds = tuple(self.interval())
    (umin, umax), (vmin, vmax) = bounds

    def wrp1(uv):
        d = self.evaluate(uv) - pt
        return scalar_dot(d, d)

    def wrp(u, v):
        d = self.evaluate(np.array([u, v])) - pt
        return scalar_dot(d, d)

    cpt = contains_point(self.tree, pt)

    if len(cpt) == 0:
        #(umin, umax), (vmin, vmax) = self.interval()
        return np.array(divide_and_conquer_min_2d(wrp, (umin, umax), (vmin, vmax), tol))

    else:

        initial = np.average(min(cpt, key=lambda x: x.bounding_box.volume()).uvs, axis=0)
        uv = newtons_method(wrp1, initial, tol=tol)
        if uv is None:
            raise ValueError('Newtons method failed to converge')
        return uv


def closest_points_on_surface(surface, pts, tol=1e-6):
    """
    Compute the closest points on a surface to a given set of points using a classic approach.

    :param surface: The surface object.
    :param pts: The set of points as a numpy array.
    :param tol: The tolerance value for the division and conquest algorithm. Default is 1e-6.
    :return: The closest points on the surface corresponding to the given set of points as a numpy array of (u, v) pairs.
    """

    surface.build_tree(10, 10)

    def objective(u, v):
        d = surface.evaluate(np.array((u, v))) - pt
        return scalar_dot(d, d)

    uvs = np.zeros((len(pts), 2))

    for i, pt in enumerate(pts):
        objects = contains_point(surface.tree, pt)
        if len(objects) == 0:
            uvs[i] = np.array(
                divide_and_conquer_min_2d(objective, *surface.interval(), tol=tol)
            )
        else:
            uvs_ranges = np.array(
                list(itertools.chain.from_iterable(o.uvs for o in objects))
            )
            uvs[i] = np.array(
                divide_and_conquer_min_2d(
                    objective,
                    (np.min(uvs_ranges[..., 0]), np.max(uvs_ranges[..., 0])),
                    (np.min(uvs_ranges[..., 1]), np.max(uvs_ranges[..., 1])),
                    tol=tol,
                )
            )
    return uvs


def closest_point_on_surface_batched(surface, pts, tol=1e-6):
    """
    Compute the closest points on a surface to a given set of points using a vectorized approach.

    :param surface: The surface object.
    :param pts: The set of points as a numpy array.
    :param tol: The tolerance value for the division and conquest algorithm. Default is 1e-6.
    :return: The closest points on the surface corresponding to the given set of points as a numpy array of (u, v) pairs.
    """

    def objective(u, v):
        d = surface(np.array((u, v)).T) - pts
        return np.array(dot(d, d))

    (u_min, u_max), (v_min, v_max) = surface.interval()
    x_range = np.empty((2, len(pts)))
    x_range[0] = u_min
    x_range[1] = u_max
    y_range = np.empty((2, len(pts)))
    y_range[0] = v_min
    y_range[1] = v_max

    uvs = np.array(
        divide_and_conquer_min_2d_vectorized(
            objective, x_range=x_range, y_range=y_range, tol=tol
        )
    )
    return uvs.T

# Example usage
__all__ = ["closest_point_on_curve",

           "closest_point_on_line",
           "foot_point",
           "closest_point_on_curve_single",
           "closest_points_on_curve_mp",
           "closest_points_on_curve_mp",
           "local_closest_point_on_curve"
           ]
if __name__ == "__main__":
    points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    min_dist, pair = min_distance(points)
    print(f"The smallest distance is {min_dist} between points at indices {pair}")
    # Expected Output: The smallest distance is 1.4142135623730951 between points at indices (0, 5)
