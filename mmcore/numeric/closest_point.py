import itertools
from dataclasses import dataclass, field

import numpy as np

from . import sbern
from .intersection._bern_zero_1d import _newton_bernstein_root_1d
from mmcore.numeric.vectors import vector_projection, scalar_dot


from mmcore.geom.nurbs import NURBSCurve, NURBSSurface



from mmcore.numeric.fdm import PDE



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




def closest_point_on_ray(ray, point):
    start, direction = ray

    return start + vector_projection(point - start, direction)


def closest_point_on_line(line, point):
    start, end = line
    direction = end - start
    return start + vector_projection(point - start, direction)

from mmcore.numeric.numeric import compute_parametric_tolerance_surface,compute_parametric_tolerance_curve


_float64_eps=np.finfo(float).eps
import functools
import numpy as np
from typing import Optional, Tuple, Any
from numpy.typing import NDArray




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
            break                               # bracket smaller than local spt

    # ---------------------------------------------------------------------
    return (dist_best, eval_best, tol_best), t_cur

def _nurbs_surface_closest_point_divide_and_conquer(
    surf: NURBSSurfaceTuple,
    point: NDArray,
    x_range: Optional[tuple[float, float]] = None,
    y_range: Optional[tuple[float, float]] = None,
    *,
    spt: float = 0.001,
    angle_tol: Optional[float] = None,
):
    """
    Locate the (u,v) on *surf* that minimises ‖S(u,v) – point‖.

    1.  A divide‑and‑conquer search shrinks a rectangular window until both
        parametric extents are within their adaptive tolerances.
    2.  A damped, bracket‑constrained Newton iteration refines the result.
    """
    # ------------------------------------------------------------------ ❶
    @functools.lru_cache(maxsize=None)
    def fun(u: float, v: float):
        """
        Cached evaluation returning
            – distance
            – full evaluation dict
            – local (uTol, vTol) given by curvature
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
        return np.linalg.norm(dc), t_eval, cur_tol    # (dist, dict, (uTol,vTol))

    # ------------------------------------------------------------------ ❷
    # Initial parametric window
    interval_u, interval_v = _surface_interval(surf)
    if x_range is None:
        x_range = interval_u
    if y_range is None:
        y_range = interval_v

    x_min, x_max = x_range
    y_min, y_max = y_range

    # first tolerance probe
    _, _, (x_tol, y_tol) = fun(x_min, y_min)

    # --------------------- divide‑and‑conquer loop ---------------------
    while (x_max - x_min) > x_tol or (y_max - y_min) > y_tol:

        x_width = x_max - x_min
        y_width = y_max - y_min
        x_mid   = 0.5 * (x_min + x_max)
        y_mid   = 0.5 * (y_min + y_max)

        # Assemble stencil depending on which directions are still “open”
        candidates = []
        both_open = (x_width > x_tol) and (y_width > y_tol)
        if both_open:
            # 9‑point stencil, unchanged
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
        elif x_width > x_tol:         # refine u only
            candidates.extend(
                [
                    (fun(x_min, y_mid), (x_min, y_mid)),
                    (fun(x_mid, y_mid), (x_mid, y_mid)),
                    (fun(x_max, y_mid), (x_max, y_mid)),
                ]
            )
        else:                         # refine v only
            candidates.extend(
                [
                    (fun(x_mid, y_min), (x_mid, y_min)),
                    (fun(x_mid, y_mid), (x_mid, y_mid)),
                    (fun(x_mid, y_max), (x_mid, y_max)),
                ]
            )

        # best of stencil
        min_val, (u_best, v_best) = min(candidates, key=lambda item: item[0][0])

        # shrink the window (¼ of current width in open directions)
        if x_width > x_tol:
            h = 0.25 * x_width
            x_min = max(u_best - h, x_range[0])
            x_max = min(u_best + h, x_range[1])
        if y_width > y_tol:
            h = 0.25 * y_width
            y_min = max(v_best - h, y_range[0])
            y_max = min(v_best + h, y_range[1])

        x_tol, y_tol = min_val[2]      # refresh adaptive tolerances

    # --------------- robust final evaluation of window corners ----------
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)

    final_candidates = [
        (fun(x_min, y_min), (x_min, y_min)),
        (fun(x_max, y_min), (x_max, y_min)),
        (fun(x_min, y_max), (x_min, y_max)),
        (fun(x_max, y_max), (x_max, y_max)),
        (fun(x_mid, y_mid), (x_mid, y_mid)),
    ]
    min_val, (u_best, v_best) = min(final_candidates, key=lambda pair: pair[0][0])

    # ------------------------------------------------------------------ ❸
    #                 Guarded Newton refinement (2‑D)                    #
    # ------------------------------------------------------------------
    MAX_NITER  = 20
    EPS_GRAD   = 1.0e-14            # “tiny gradient”
    EPS_DET    = 1.0e-14            # “tiny determinant” (Jacobian singular)
    EPS_DSTEP  = 1.0e-12            # min accepted parameter step
    MIN_SCALE  = 0.125              # smallest damping (1/8)

    # private shrinking box; start with current window
    u_lo, u_hi = x_min, x_max
    v_lo, v_hi = y_min, y_max

    dist_best, eval_best, tol_best = min_val
    u_cur, v_cur = u_best, v_best

    for _ in range(MAX_NITER):

        S   = eval_best["S"]
        Su  = eval_best["Su"]
        Sv  = eval_best["Sv"]
        Suu = eval_best["Suu"]
        Suv = eval_best["Suv"]
        Svv = eval_best["Svv"]

        dvec = S - point

        # Gradient of squared distance ‖S-point‖² / 2  (the ½ is irrelevant)
        g_u = float(np.dot(Su, dvec))
        g_v = float(np.dot(Sv, dvec))
        grad_norm = abs(g_u) + abs(g_v)

        if grad_norm < EPS_GRAD:
            break                         # already stationary -> done

        # Gauss–Newton / true Hessian of ½‖d‖²
        J11 = float(np.dot(Su, Su) + np.dot(dvec, Suu))
        J22 = float(np.dot(Sv, Sv) + np.dot(dvec, Svv))
        J12 = float(np.dot(Su, Sv) + np.dot(dvec, Suv))

        detJ = J11 * J22 - J12 * J12
        if abs(detJ) < EPS_DET:
            break                         # near‑singular Jacobian

        # Solve J * [du, dv]^T = -grad
        du = (J12 * g_v - J22 * g_u) / detJ
        dv = (J12 * g_u - J11 * g_v) / detJ

        if abs(du) < EPS_DSTEP and abs(dv) < EPS_DSTEP:
            break                         # step too small

        # ----------------- step acceptance with back‑tracking ----------
        step_scale = 1.0
        success = False
        while step_scale >= MIN_SCALE and not success:

            u_try = u_cur + step_scale * du
            v_try = v_cur + step_scale * dv

            # hard‑clip to bracket to avoid extrapolation surprises
            u_try = min(max(u_try, u_lo), u_hi)
            v_try = min(max(v_try, v_lo), v_hi)

            dist_try, eval_try, tol_try = fun(u_try, v_try)

            if dist_try < dist_best:          # accept the step
                # tighten the box in accepted movement directions
                if u_try < u_cur:
                    u_hi = u_cur
                elif u_try > u_cur:
                    u_lo = u_cur
                if v_try < v_cur:
                    v_hi = v_cur
                elif v_try > v_cur:
                    v_lo = v_cur

                u_cur, v_cur = u_try, v_try
                dist_best, eval_best, tol_best = dist_try, eval_try, tol_try
                success = True
            else:
                step_scale *= 0.5             # back‑track / damp

        if not success:
            break                             # cannot improve further

        # stop if bracket already within local tolerances
        if (u_hi - u_lo) <= tol_best[0] and (v_hi - v_lo) <= tol_best[1]:
            break

    # ------------------------------------------------------------------ ❹
    return (dist_best, eval_best, tol_best), (u_cur, v_cur)
from mmcore.numeric.newton.cnewton import newton
from mmcore.numeric import bern_sq_dist
from mmcore.numeric.intersection._bezier_common import newton_ccx, eval_curve, _clamp01, eval_curve_d1
from mmcore.geom._nurbs_param_tol import bez_curve_param_tolerance,bez_surface_param_tolerance
from mmcore.numeric.ndinterval import interval

from more_itertools import pairwise


def _split_intervs(ints: list[interval]):
    """"""

    vls = np.unique(list(itertools.chain.from_iterable((i.l, i.u) for i in ints)))

    includes = []
    intervs = []
    for i, j in pairwise(vls):
        interv = interval(i, j)
        intervs.append(interv)
        for inter in ints:

            includes.append((inter.subseteq(interv), inter.subseteq(interv)))

    return intervs, includes

def _interv_is_nan(interv:interval):
    return np.isnan(interv.l) or np.isnan(interv.u)
class IntervTree:

    interv:interval
    value:Any
    left:'IntervTree|None'
    right:'IntervTree|None'

    def __init__(self, interv:interval, value=None):

        self.interv=interv
        self.value=value

        self.left=None
        self.right=None


    def trim_value(self,interv:interval):
        return self.value

    def split_value(self, t: float):

        first = interval(self.interv.l, t)
        sec = interval(t, self.interv.u)



        return self.__class__(first,  value=self.value),self.__class__(sec, value=self.value)
    def split(self, t:float):
        if not interval(t,t).subset(self.interv):
            return False,self
        if self.left is None:



            self.left,self.right=self.split_value(t)

            return True,(self.left,self.right)
        else:
            success1,leafs1=self.left.split(t)
            success2,leafs2=self.right.split(t)

            if success1 and success2:
                #print(success1,leafs1,success2,leafs2)
                raise ValueError("impossible")
            elif success1:
                return True,leafs1
            elif success2:
                return True,leafs2
            else:
                return True,(self.left,self.right)






    def find_leafs(self, interv:interval, _vls=None):
        if _vls is None:
            _vls=[]

        inter=self.interv.intersection(interv)
        if _interv_is_nan(inter) or inter.norm()==0:
            return _vls
        elif self.left is None and self.right is None:

            _vls.append(self)
        else:
            if self.left is not None:
                self.left.find_leafs(interv, _vls)
            if self.right is not None:
                self.right.find_leafs(interv, _vls)

        return _vls

    def __repr__(self):
        return f"{self.__class__.__name__}({self.interv},{self.value})"
    def add_child(self,child:interval, val:interval, leafs=None):
        if leafs is None:
            leafs=[]
        res=self.interv.intersection(child)
        if _interv_is_nan(res) or res.norm()==0:
            return False

        for i in [res.l,res.u]:
            succ,_leafs=self.split(i)

        for leaf in self.find_leafs(child):


            if leaf.interv.equal(child):
                leaf.value=val
                leafs.append(leaf)



            else:
                leaf.add_child( leaf.interv.intersection(child),val, leafs)




        return leafs





    def find_val(self,t:float, _vls=None):
        if _vls is None:
            _vls=[]
        t_interv=interval(t)

        if self.interv.supseteq(t_interv):
            if self.left is None and self.right is None:
                _vls.append(self)
            else:
                self.left.find_val(t, _vls)
                self.right.find_val(t, _vls)



        return _vls

from .bern import bernstein_partial_derivative_coeffs
@dataclass
class BezPointCurveTreeData:
    curve:NDArray
    F:NDArray
    Qw:NDArray
    rational:bool


    def split(self,t):
        FQwl,FQwr=_subdivide_curve(np.array(list(zip(self.F, self.Qw))), t)
        crvl, crvr = _subdivide_curve(self.curve, t)

        return BezPointCurveTreeData(crvl, FQwl[...,0],FQwl[...,1],self.rational),BezPointCurveTreeData(crvr, FQwr[...,0],FQwr[...,1],self.rational)
def _subdivide_curve(ctrl, t=0.5):
    n = ctrl.shape[0] - 1
    tmp = ctrl.copy()
    left = [tmp[0].copy()]
    right_rev = [tmp[n].copy()]
    for r in range(1, n + 1):
        tmp[: n + 1 - r] = (1.0 - t) * tmp[: n + 1 - r] + t * tmp[1 : n + 2 - r]
        left.append(tmp[0].copy())
        right_rev.append(tmp[n - r].copy())
    return np.array(left), np.array(right_rev[::-1])


class BezPointCurveTree(IntervTree):
    def __init__(self, interv, value: BezPointCurveTreeData, atol=1e-3, flat_tol=1e-6):

        super().__init__(interv, value)

        self.bounds = interval(*bern_sq_dist.bounds_point_curve(self.value.F, self.value.Qw))
        self.flat_tol = flat_tol
        self.atol = atol
        self.ptol=bez_curve_param_tolerance(value.curve, flat_tol, rational=value.rational,interval=(interv.l,interv.u))





        self.dF=np.squeeze(bernstein_partial_derivative_coeffs(value.F[:,None], 0))
        #self.dQw = bernstein_partial_derivative_coeffs(value.Qw[:,None], 0)

        self.dbounds=interval(*bern_sq_dist.bounds_point_curve(self.dF,self.value.Qw))




    @property
    def is_monotone(self):
        return not interval(0.,0.).subset(self.dbounds)

    @property
    def is_flat(self):


        return self.dbounds.norm()<self.flat_tol

    @property
    def is_small(self):
        return self.interv.norm()<self.ptol
    def split_value(self, t: float):
        first = interval(self.interv.l, t)
        sec = interval(t, self.interv.u)

        vl,vr=self.value.split(     (t-self.interv.l)/(self.interv.u-self.interv.l))
        self.left,self.right=BezPointCurveTree(first,vl,atol=self.atol,flat_tol=self.flat_tol),BezPointCurveTree(sec,vr,atol=self.atol,flat_tol=self.flat_tol)
        return self.left,self.right


def newton_closest_point(
    C,
    point,
    u0: float,
    *,
    rational: bool = False,
    tol: float = 1e-14,
    step_tol: float = 1e-14,
    max_it: int = 30,
    lm_damp: float = 1e-12,
):
    """LM-damped local closest-point solve for a Bézier curve and a point.

    Minimizes

        ||C(u) - point||^2,    u in [0, 1]

    starting from the initial guess u0.

    This is the 1D analogue of the LM-damped Newton/Gauss-Newton
    curve-curve solver. It solves the local least-squares problem obtained
    by linearizing

        C(u + du) - point ~= C(u) - point + C'(u) du.

    Runs until stationarity < tol, step < step_tol, line search fails,
    or max_it.

    Parameters
    ----------
    C : curve object / control net
        Bézier curve representation accepted by eval_curve/eval_curve_d1.
    point : array_like
        Target point.
    u0 : float
        Initial parameter guess.
    rational : bool, optional
        Passed through to eval_curve/eval_curve_d1.
    tol : float, optional
        Tolerance for the first-order optimality condition

            dot(C(u) - point, C'(u)) = 0.

    step_tol : float, optional
        Tolerance for the parameter step.
    max_it : int, optional
        Maximum number of Newton iterations.
    lm_damp : float, optional
        Levenberg-Marquardt damping added to the 1D normal equation.

    Returns
    -------
    u : float
        Best parameter found, clamped to [0, 1].
    R : ndarray
        Final residual vector C(u) - point.
    sqdist : float
        Final squared distance ||C(u) - point||^2.
    last_du : float
        Last accepted parameter step. The caller can compare abs(last_du)
        against a parametric tolerance.
    """
    point = np.asarray(point, dtype=float)

    u = _clamp01(float(u0))
    last_du = 1.0  # initial large step

    for _ in range(max_it):
        p, d = eval_curve_d1(C, u, rational=rational)

        R = p - point
        sqdist = float(np.dot(R, R))

        # First derivative of 1/2 * squared distance.
        g = float(np.dot(R, d))

        # KKT-style stationarity for the constrained interval [0, 1].
        #
        # Interior: g == 0.
        # At u = 0: valid minimum if g >= 0.
        # At u = 1: valid minimum if g <= 0.
        if (
            abs(g) < tol
            or (u <= 0.0 and g >= -tol)
            or (u >= 1.0 and g <= tol)
        ):
            last_du = 0.0
            break

        # 1D LM/Gauss-Newton normal equation:
        #
        #   (dot(d, d) + lambda) du = -dot(R, d)
        #
        # This is equivalent to minimizing the squared distance to the
        # tangent-line approximation of the curve.
        A = float(np.dot(d, d)) + lm_damp
        b = -g

        if A <= 0.0 or not np.isfinite(A):
            last_du = 0.0
            break

        du = b / A

        if not np.isfinite(du):
            last_du = 0.0
            break

        if du * du < step_tol * step_tol:
            last_du = float(du)
            break

        step = 1.0
        accepted = False

        for _ls in range(8):
            un = _clamp01(u + step * du)
            actual_du = un - u

            if actual_du * actual_du < step_tol * step_tol:
                last_du = float(actual_du)
                u = un
                accepted = True
                break

            Rn = eval_curve(C, un, rational=rational) - point
            sqdist_n = float(np.dot(Rn, Rn))

            # Monotone backtracking: accept only if squared distance
            # does not increase.
            if sqdist_n <= sqdist:
                last_du = float(actual_du)
                u = un
                accepted = True
                break

            step *= 0.5

        if not accepted:
            last_du = 0.0
            break

        if last_du * last_du < step_tol * step_tol:
            break

    R = eval_curve(C, u, rational=rational) - point
    sqdist = float(np.dot(R, R))

    return u, R, sqdist, last_du
def bez_curve_closest_point(curve:NDArray, point:NDArray,atol=1e-3,rational=False):
    F=bern_sq_dist.point_curve_distance_squared_net_homog(point, curve, rational=rational)
    Qw=curve[..., -1] if rational else np.ones_like(curve[...,0])

    root=BezPointCurveTree(interval(0., 1.), value=BezPointCurveTreeData(curve, F, Qw, rational))
    stack = [root]
    candidates=[]
    while stack:

        leaf:BezPointCurveTree=stack.pop()



        if leaf.is_monotone:

            continue




        elif leaf.is_flat:

            t = _newton_bernstein_root_1d(leaf.dF, 0.5)
            tglob = (leaf.interv.u - leaf.interv.l) * t + leaf.interv.l
            dd=eval_curve(leaf.value.curve,t,rational=rational)-point

            sqdist = np.dot(dd,dd)


            leaf.best_t,leaf.best_d=tglob,sqdist
            candidates.append(leaf)

            continue

        elif leaf.is_small:
            t = _newton_bernstein_root_1d(leaf.dF, 0.5)
            tglob = (leaf.interv.u - leaf.interv.l) * t + leaf.interv.l
            dd=eval_curve(leaf.value.curve,t,rational=rational)-point

            sqdist = np.dot(dd,dd)

            leaf.best_t, leaf.best_d = tglob, sqdist
            candidates.append(leaf)
            continue
        else:

            #t, R ,sqdist,_=newton_closest_point(leaf.value.curve,point,0.5,rational=rational,max_it=15)
            t=_newton_bernstein_root_1d(leaf.dF,0.5)
            tglob = (leaf.interv.u - leaf.interv.l) * t + leaf.interv.l
            dd = eval_curve(leaf.value.curve, t, rational=rational) - point

            sqdist = np.dot(dd, dd)




            candidates.append(leaf)



            leaf.best_t,leaf.best_d=tglob,sqdist

            if (1-t)<1e-8 or (t<1e-8):
                continue

            success,lr=leaf.split(tglob)

            if success:
                l,r=lr
                if l.is_monotone and r.is_monotone:


                    leaf.left=None
                    leaf.right=None
                elif l.is_monotone:
                    leaf.left=None
                    stack.append(r)
                elif r.is_monotone:
                    leaf.right=None
                    stack.append(l)

                else:
                    stack.append(l)
                    stack.append(r)





    best_cand=min(candidates,key=lambda x:x.best_d)
    return best_cand.best_t,best_cand.best_d















def nurbs_curve_closest_point(self: NURBSCurveTuple, point: NDArray[float], atol: float = 0.001, spt=None,angle_tol: float = None):
    if isinstance(self, NURBSCurve):
        self=_nurbs_to_tuple(self)
    candidates = decompose_curve(self)
    if spt is not None:
        atol=spt

    best_f = float("inf")
    best_x = None
    for candidate in candidates:
        rational = not np.allclose(candidate.weights, 1)
        bez=sbern.nurbs_bezier_to_bern(candidate,rational=rational)

        min_t ,min_val= bez_curve_closest_point(bez, point, atol=atol,rational=rational)

        if best_f > min_val:
            best_f = min_val
            best_x = min_t
    
    
    return best_x, best_f



def nurbs_surface_closest_point(self:NURBSSurfaceTuple, point:NDArray[float],spt:float=0.001, angle_tol:float=None):
    """
    return: (u,v), (dist, eval, parametric_tol)
    
    
    """
    if isinstance(self, NURBSSurface):
        self=_nurbs_to_tuple(self)
    candidates=decompose_surface(self)

    best_f=[float('inf'),{},(None,None)]
    best_x=None
    for candidate in candidates:
      
        (dist_best, eval_best, tol_best),min_coords = _nurbs_surface_closest_point_divide_and_conquer(candidate,point,spt=spt, angle_tol=angle_tol)
       
        if best_f[0]>dist_best:
            best_f=(dist_best,eval_best,tol_best)
            best_x=min_coords
    (dist_best,eval_best,tol_best)=best_f
    return best_x, (dist_best,eval_best,tol_best)



# Example usage
__all__ = [
    "closest_point_on_ray",
           "closest_point_on_line",
           "foot_point",
           "nurbs_surface_closest_point",
           "nurbs_curve_closest_point"
           ]
if __name__ == "__main__":
    points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    min_dist, pair = min_distance(points)
    print(f"The smallest distance is {min_dist} between points at indices {pair}")
    # Expected Output: The smallest distance is 1.4142135623730951 between points at indices (0, 5)
