from mmcore.numeric import (
    compute_parametric_tolerance_curve,compute_parametric_sectional_curvature_tolerance_surface,
    compute_parametric_tolerance_surface, compute_parametric_curvature_tolerance_curve
)
from mmcore.numeric.closest_point import nurbs_surface_closest_point
import numpy as np
from mmcore.numeric.intersection.csx._ncsx2 import refine_curve_surface
from mmcore.numeric.intersection.csx._ncsx_new_intersections_test import (
    new_intersection_candidates,
)
from mmcore.numeric.newton.cnewton import newtons_method
from mmcore.geom._nurbs_eval import *
from mmcore.geom._nurbs_eval import (
    _tuple_to_nurbs,
    _nurbs_to_tuple,
    _curve_interval,
    _surface_interval,
)
from mmcore.numeric.divide_and_conquer import divide_and_conquer_min_2d


def _evaluate_crv_data(crv,t,spt):
    crv_eval = evaluate_nurbs_curve(crv, t, d_order=2)
    kws = {**crv_eval}
    del kws["C"]
    crv_eval["dt"] = compute_parametric_curvature_tolerance_curve(**kws,spt=spt)
    crv_eval["T"] = crv_eval["C1"] / np.linalg.norm(crv_eval["C1"])
    return crv_eval

def _evaluate_surf_data(surf, u, v, tangent, spt):
        surf_eval = evaluate_nurbs_surface(surf, u, v, d_order=3)
        kws = {**surf_eval}
        del kws["S"]
        Su, Sv, Suu,Suv, Svv,=kws["Su"],kws["Sv"],kws["Suu"],kws["Suv"],kws["Svv"]
        du, dv= compute_parametric_sectional_curvature_tolerance_surface(Su, Sv, Suu,Suv, Svv,tangent=tangent,spt=spt)
        surf_eval["du"] = du
        surf_eval["dv"] = dv
        N = np.cross(surf_eval["Su"], surf_eval["Sv"])
        N /= np.linalg.norm(N)
        surf_eval["N"] = N
        return surf_eval

def trace_overlap(
    tuv_inter: np.ndarray,
    crv: NURBSCurveTuple,
    surf: NURBSSurfaceTuple,
    spt=1e-3,
    angle_tol=0.052,
):
    crv_n = crv
    srf_n = surf
    srf_nn=_tuple_to_nurbs(surf)
    crv_nn=_tuple_to_nurbs(crv)
    t0, t1 = _curve_interval(crv)
    (u0, u1), (v0, v1) = _surface_interval(surf)
    def evaluate_crv_data(t):
        return _evaluate_crv_data(crv_n,t, spt=spt)

    def evaluate_surf_data(u, v):
        return _evaluate_surf_data(srf_n, u, v, tangent=False, spt=spt)

    def improve_uv(old_uv, du, dv, pt):
        nonlocal crv_n, srf_n

        def equation(u, v):
            d = pt - srf_nn.evaluate_v2(u, v)
            return np.dot(d, d)

        bounds = (max(old_uv[0] - du, u1), min(old_uv[0] + du, u1)), (max(old_uv[1] - dv, v0), min(old_uv[1] + dv, v1))

        u_new, v_new = divide_and_conquer_min_2d(equation, *bounds, tol=1e-6)

        return np.array([u_new, v_new])

    tuv_inter = np.copy(tuv_inter)

    curve_eval = evaluate_crv_data(tuv_inter[0])

    surf_eval = evaluate_surf_data(tuv_inter[1], tuv_inter[2])
    if np.abs(np.dot(curve_eval["T"], surf_eval["N"])) > angle_tol:
        return False, tuv_inter
    tuv_prev = tuv_inter
    while True:

        if not (t0 <= tuv_prev[0] <= t1) or not (u0 <= tuv_prev[1] <= u1) or not (v0 <= tuv_prev[2] <= v1):
            tuv_prev = np.clip(tuv_prev, np.array([t0, u0, v0]), np.array([t1, u1, v1]))
            print("param bounds")
            return tuv_prev

        t_new = tuv_prev[0] + curve_eval["dt"]
        if not (t0 <= t_new <= t1):
            t_new = np.clip(t_new, t0, t1)

        curve_eval = evaluate_crv_data(t_new)

        uv_new = improve_uv(tuv_prev[1:], surf_eval["du"], surf_eval["dv"], curve_eval["C"])
        print(tuv_prev[1:], uv_new)
        surf_eval = evaluate_surf_data(*uv_new)

        dst = np.linalg.norm(curve_eval["C"] - surf_eval["S"])

        if dst > spt:
            print("dst>spt", t_new, uv_new, dst)
            return tuv_prev

        if abs(np.dot(curve_eval["T"], surf_eval["N"])) > angle_tol:
            print("angle_tol")
            return tuv_prev
        tuv_new = np.array([t_new, *uv_new])
        if np.allclose(tuv_new, tuv_prev):
            print("not march")
            return tuv_new

        tuv_prev = np.array([t_new, *uv_new])


def trace_overlap2(
    tuv_inter: np.ndarray,
    crv: NURBSCurveTuple,
    surf: NURBSSurfaceTuple,
    spt=1e-3,
    angle_tol=0.052,
):
    crv_n = crv
    srf_n = surf
    srf_nn = _tuple_to_nurbs(surf)
    crv_nn = _tuple_to_nurbs(crv)
    t0, t1 = _curve_interval(crv)
    (u0, u1), (v0, v1) = _surface_interval(surf)

    def evaluate_crv_data(t):
        return _evaluate_crv_data(crv_n, t, spt=spt)

    def evaluate_surf_data(u, v):
        return _evaluate_surf_data(srf_n, u, v, tangent=False, spt=spt)

    def improve_uv(old_uv, du, dv, pt):
        nonlocal crv_n, srf_n

        def equation(u, v):
            d = pt - srf_nn.evaluate_v2(u, v)
            return np.dot(d, d)

        bounds = (max(old_uv[0] - du, u1), min(old_uv[0] + du, u1)), (max(old_uv[1] - dv, v0), min(old_uv[1] + dv, v1))

        u_new, v_new = divide_and_conquer_min_2d(equation, *bounds, tol=1e-6)

        return np.array([u_new, v_new])

    tuv_inter = np.copy(tuv_inter)

    curve_eval = evaluate_crv_data(tuv_inter[0])

    surf_eval = evaluate_surf_data(tuv_inter[1], tuv_inter[2])
    t_new = tuv_inter[0] + curve_eval["dt"]
    tuv_prev = tuv_inter
    events=[tuv_inter]
    on_overlap=True

    while t_new<t1:

        if not (t0 <= tuv_prev[0] <= t1) or not (u0 <= tuv_prev[1] <= u1) or not (v0 <= tuv_prev[2] <= v1):
            tuv_prev = np.clip(tuv_prev, np.array([t0, u0, v0]), np.array([t1, u1, v1]))
            events.append(tuv_prev)

            # print("param bounds")
            return events

        if not (t0 <= t_new <= t1):
            t_new = np.clip(t_new, t0, t1)
            uv_new = improve_uv(tuv_prev[1:], surf_eval["du"], surf_eval["dv"], curve_eval["C"])
            events.append(np.array([t_new, *uv_new]))
            return events
        t_new = t_new + curve_eval["dt"]
        print(t_new)
        curve_eval = evaluate_crv_data(t_new)

        uv_new = improve_uv(tuv_prev[1:], surf_eval["du"], surf_eval["dv"], curve_eval["C"])

        surf_eval = evaluate_surf_data(*uv_new)

        dst = np.linalg.norm(curve_eval["C"] - surf_eval["S"])
        a_m=abs(np.dot(curve_eval["T"], surf_eval["N"])) <= angle_tol
        dst_m=dst <= spt
        if a_m and dst_m :
            if not on_overlap:
                on_overlap=True
                events.append(np.array([t_new, *uv_new]))

        else:
            if on_overlap:
                on_overlap = False
                events.append(np.array([t_new, *uv_new]))

        tuv_new = np.array([t_new, *uv_new])
        if np.allclose(tuv_new, tuv_prev):
            print("not march")
            return events

    return events
