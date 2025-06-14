"""
Robust NURBS curve–surface intersection – v3
===========================================

Changes compared with v2 (2025‑06‑12)
-------------------------------------
✓  Re‑designed overlap test:  only accepted if *every* sample point on the
   curve segment converges to the surface **with the same t–parameter**  
   (prevents the earlier false positives on transversal patches).

✓  Depth‑limit fallback now *confirms* overlap with the same test; otherwise
   it silently abandons the patch instead of recording a wrong overlap.

✓  Overlap records now carry a representative 3‑D point (mid‑point of the
   curve segment).  Pipelines that expect a `.point` attribute will therefore
   receive a valid NumPy array instead of `None`.


"""

import math
from typing import List, Tuple, TypedDict

import numpy as np

from mmcore.numeric._aabb import aabb_intersect_fast_3d
from mmcore.numeric.newton.cnewton import newtons_method
from mmcore.geom.nurbs import (
    NURBSCurve,
    NURBSSurface,
    split_curve,
    subdivide_surface,
    CurveSurfaceEq,
)
from numpy._typing import NDArray

from mmcore.geom._nurbs_eval import _nurbs_to_tuple,NURBSCurveTuple,EvaluateCurveData
from mmcore.numeric import compute_parametric_tolerance_curve
from mmcore.numeric.intersection.separability.spatial import spatial_separability
from mmcore.numeric.intersection.separability.spherical import spherical_separability

__all__ = ["nurbs_csx", "NURBSCurveSurfaceIntersector"]

from notes.offset import evaluate_nurbs_curve


# ---------------------------------------------------------------------------
# helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _surface_ivl(surf: NURBSSurface) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return surf.interval()


def _inflate(bb: np.ndarray, eps: float) -> np.ndarray:
    out = bb.copy()
    out[0] -= eps
    out[1] += eps
    return out


# ---------------------------------------------------------------------------
# refined generic overlap test ---------------------------------------------
# ---------------------------------------------------------------------------

def _segment_lies_on_surface(
    curve: NURBSCurve,
    surface: NURBSSurface,
    ptol: float,
    dist_tol: float,
    samples: int = 7,
) -> bool:
    """
    Decide whether an *entire* curve segment is on a surface patch.

    For each uniformly‑spaced parameter `ti` we launch Newton **with ti held
    fixed**.  Convergence is accepted only if

        ‖C(ti) – S(ui,vi)‖ <= dist_tol     and
        |t* – ti|           <= ptol

    where (t*,u*,v*) is the solver result.
    """
    eq = CurveSurfaceEq(curve, surface)
    (t0, t1) = curve.interval()
    (u0, u1), (v0, v1) = _surface_ivl(surface)
    uv_seed = np.array([(u0 + u1) * 0.5, (v0 + v1) * 0.5])

    for ti in np.linspace(t0, t1, samples):
        x0 = np.concatenate(([ti], uv_seed))
        sol = newtons_method(eq, x0, max_iter=6)

        if (
            sol is None
            or np.any(np.isnan(sol))
            or abs(sol[0] - ti) > ptol              # t drift indicates transversal
            or sol[1] < u0 - 1e-12 or sol[1] > u1 + 1e-12
            or sol[2] < v0 - 1e-12 or sol[2] > v1 + 1e-12
            or math.sqrt(eq(sol)) > dist_tol
        ):
            return False

        uv_seed[:] = sol[1:]     # good starting point for the next round
    return True


# ---------------------------------------------------------------------------
# main class ----------------------------------------------------------------
# ---------------------------------------------------------------------------
class EvaluateCurveDataFull(TypedDict):
    """
    :ivar C: Point at.
    :type C: numpy.ndarray[float]
    :ivar C1: First derivative.
    :type C1: numpy.ndarray[float]
    :ivar C2: Second derivative
    :type C2: numpy.ndarray[float]
    """
    C:NDArray[float]
    C1: NDArray[float]
    C2: NDArray[float]
    dt:float


class NURBSCurveSurfaceIntersector:
    """
    Sub‑division / separability intersector with *robust* overlap handling.

    v3 fixes:  no false overlaps, no None coordinates in returned tuples.
    """

    __slots__ = (
        "initial_curve",
        "initial_surface",
        "tolerance",
        "ptol",
        "angle_tol",
        "intersections",
        "_depth_limit",
        "_initial_curve_t",
        "_initial_curve_t_evals"
    )

    def __init__(
        self,
        curve: NURBSCurve,
        surface: NURBSSurface,
        tolerance: float = 1e-3,
        ptol: float = 1e-7,
        angle_tol: float = 0.052,
        depth_limit: int = 30,
    ):
        self.initial_curve = curve
        self.initial_surface = surface
        self.tolerance = tolerance
        self.ptol = ptol
        self.angle_tol = angle_tol
        self._depth_limit = depth_limit
        self.intersections: List[Tuple[str, object, Tuple[float, ...]]] = []
        self._initial_curve_t:NURBSCurveTuple=_nurbs_to_tuple(self.initial_curve)
        self._initial_curve_t_evals:dict[float,EvaluateCurveDataFull]=dict()
    # ---------------------------------------------------------------------

    def intersect(self) -> List[Tuple[str, object, Tuple[float, ...]]]:
        self._recurse(self.initial_curve, self.initial_surface, depth=0)
        return self.intersections
    def _eval_initial_curve(self,t)->EvaluateCurveData:
        curve_eval=self._initial_curve_t_evals.get(t, None)
        if curve_eval is None:
            curve_eval=evaluate_nurbs_curve(self._initial_curve_t, t,d_order=2)
            dt=compute_parametric_tolerance_curve(**curve_eval,spt= self.tolerance,angle_tol=self.angle_tol)
            curve_eval['dt']=dt
            self._initial_curve_t_evals[t]=curve_eval

        return curve_eval

    # ---------------------------------------------------------------------

    def _recurse(
        self,
        curve: NURBSCurve,
        surface: NURBSSurface,
        depth: int,
    ):
        # 0. separability
        if spatial_separability(
            curve.control_points,
            np.array(surface.control_points_flat),
            tol=self.tolerance,
        ):
            return

        # 1. overlap test --------------------------------------------------
        if _segment_lies_on_surface(
            curve, surface, self.ptol, self.tolerance
        ):
            self._store_overlap(curve, surface)
            return

        # 2. depth governor (confirm!) -------------------------------------
        if depth >= self._depth_limit:
            if _segment_lies_on_surface(curve, surface, self.ptol, self.tolerance):
                self._store_overlap(curve, surface)
            return

        # 3. try local Newton root -----------------------------------------
        loc_eq = CurveSurfaceEq(curve, surface)
        hit = self._newton_hit(curve, surface, loc_eq)
        (u0, u1), (v0, v1) = _surface_ivl(surface)
        t0, t1 = curve.interval()
        curve_eval = self._eval_initial_curve(t0)
        dt = 0.5 * (t1 - t0)
        if hit is None:
            # subdivision on miss

            if dt < curve_eval['dt']:
                #print("dt exit (no hit)", (t0 ,t1))
                return
            #print("dt not exit (no hit)", (t0 , t1), (t1-t0),curve_eval['dt'])
            c1, c2 = split_curve(curve, (t0 + t1) * 0.5, tol=1e-12, normalize_knots=False)
            um, vm = (u0 + u1) * 0.5, (v0 + v1) * 0.5
            
            if (abs(um - u0) < self.ptol or abs(um - u1) < self.ptol
                    or abs(vm - v0) < self.ptol or abs(vm - v1) < self.ptol):
                return
            s1, s2, s3, s4 = subdivide_surface(surface, um, vm, tol=1e-12, normalize_knots=False)
        else:
            p, (t_hit, u_hit, v_hit) = hit

            tag = "degenerate" if self._is_degenerate(curve, surface, hit[1]) else "transversal"
            self._store_point(tag, p, (t_hit, u_hit, v_hit))

            # exit if hit is on patch boundary or separability sphere ok

            if dt < curve_eval["dt"]:
                #print("dt exit", p, (t_hit, u_hit, v_hit),dt,curve_eval["dt"])
                return
            if (abs(u_hit - u0) < self.ptol or abs(u_hit - u1) < self.ptol
                    or abs(v_hit - v0) < self.ptol or abs(v_hit - v1) < self.ptol
                    or spherical_separability(
                        np.array(surface.control_points_flat),
                        curve.control_points,
                        p,
                    )):
                return
            c1, c2 = split_curve(curve, t_hit, tol=1e-12, normalize_knots=False)
            s1, s2, s3, s4 = subdivide_surface(surface, u_hit, v_hit, tol=1e-12, normalize_knots=False)

        # 4. recurse on the eight children ---------------------------------
        self._recurse(c1, s1, depth + 1)
        self._recurse(c1, s2, depth + 1)
        self._recurse(c1, s3, depth + 1)
        self._recurse(c1, s4, depth + 1)
        self._recurse(c2, s1, depth + 1)
        self._recurse(c2, s2, depth + 1)
        self._recurse(c2, s3, depth + 1)
        self._recurse(c2, s4, depth + 1)

    # ---------------------------------------------------------------------
    # point hit ------------------------------------------------------------

    def _newton_hit(
        self,
        curve: NURBSCurve,
        surface: NURBSSurface,
        eq: CurveSurfaceEq,
    ):
        bb1 = _inflate(np.array(curve.bbox()), self.tolerance)
        bb2 = _inflate(np.array(surface.bbox()), self.tolerance)
        if not aabb_intersect_fast_3d(bb1, bb2):
            return None

        t0, t1 = curve.interval()
        (u0, u1), (v0, v1) = _surface_ivl(surface)
        seed = np.array([(t0 + t1) * 0.5, (u0 + u1) * 0.5, (v0 + v1) * 0.5])

        sol = newtons_method(eq, seed, max_iter=5)
        if (sol is None or np.any(np.isnan(sol))
                or not self._inside(sol, (t0, t1), (u0, u1), (v0, v1))):
            return None

        sol = newtons_method(eq, sol, max_iter=4)
        if (sol is None or np.any(np.isnan(sol))
                or not self._inside(sol, (t0, t1), (u0, u1), (v0, v1))
                or math.sqrt(eq(sol)) > self.tolerance):
            return None

        curve.evaluate(sol[0])

        return curve.evaluate(sol[0]), tuple(sol)

    @staticmethod
    def _inside(p, t_rng, u_rng, v_rng):
        t, u, v = p
        return (t_rng[0] - 1e-12 <= t <= t_rng[1] + 1e-12
                and u_rng[0] - 1e-12 <= u <= u_rng[1] + 1e-12
                and v_rng[0] - 1e-12 <= v <= v_rng[1] + 1e-12)

    # ---------------------------------------------------------------------
    # degeneracy -----------------------------------------------------------

    def _is_degenerate(
        self,
        curve: NURBSCurve,
        surface: NURBSSurface,
        tuv: Tuple[float, float, float],
    ) -> bool:
        t, u, v = tuv
        tan = curve.tangent(t)
        n = np.cross(surface.derivative_u(np.array([u, v])),
                     surface.derivative_v(np.array([u, v])))
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            return True
        n /= ln
        return abs(np.dot(tan, n)) < 1e-3

    # ---------------------------------------------------------------------
    # storing of results ---------------------------------------------------

    def _store_point(
        self,
        tag: str,
        point: np.ndarray,
        tuv: Tuple[float, float, float],
    ):
        tv = np.asarray(tuv)
        for _, _, old in self.intersections:
            if len(old) == 3 and np.all(np.abs(tv - old) < self.ptol):
                return
        self.intersections.append((tag, point, tuple(tuv)))
        #print(tag, point, tuv)
    def _store_overlap(self, curve: NURBSCurve, surface: NURBSSurface):
        t0, t1 = curve.interval()
        (u0, u1), (v0, v1) = _surface_ivl(surface)
        point_mid = curve.evaluate(0.5 * (t0 + t1))
        box = (t0, t1, u0, u1, v0, v1)

        # merge / de‑dupe overlaps (simple containment test)
        for i, (tg, _, data) in enumerate(self.intersections):
            if tg != "overlap":
                continue

            if (t0 >= data[0] and t1 <= data[1]
                    and u0 >= data[2] and u1 <= data[3]
                    and v0 >= data[4] and v1 <= data[5]):
                return
            if (max(t0, data[0]) <= min(t1, data[1]) + self.ptol
                    and max(u0, data[2]) <= min(u1, data[3]) + self.ptol
                    and max(v0, data[4]) <= min(v1, data[5]) + self.ptol):
                merged = (
                    min(t0, data[0]), max(t1, data[1]),
                    min(u0, data[2]), max(u1, data[3]),
                    min(v0, data[4]), max(v1, data[5]),
                )
                print("overlap", point_mid, merged)
                self.intersections[i] = ("overlap", point_mid, merged)
                return
        print("overlap", point_mid, box)
        self.intersections.append(("overlap", point_mid, box))


# ---------------------------------------------------------------------------
# functional façade ---------------------------------------------------------
# ---------------------------------------------------------------------------

def nurbs_csx(
    curve: NURBSCurve,
    surface: NURBSSurface,
    tol: float = 1e-3,
    ptol: float = 1e-6,
) -> List[Tuple[str, object, Tuple[float, ...]]]:
    """
    Curve–surface intersection returning

        ('transversal',  xyz, (t,u,v))
        ('degenerate',   xyz, (t,u,v))
        ('overlap',      xyz, (t0,t1,u0,u1,v0,v1))

    *xyz is *never None*.
    """
    return NURBSCurveSurfaceIntersector(
        curve, surface, tolerance=tol, ptol=ptol
    ).intersect()
