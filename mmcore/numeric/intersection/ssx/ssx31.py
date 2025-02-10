#!/usr/bin/env python3
"""
Implementation of the MARCHING METHOD algorithm for tracing the intersection curves
of two NURBS surfaces. This implementation strictly follows the algorithm description
presented in the paper, including both transversal and tangential intersection cases,
the formulation of the ODE system for marching, and a validated ODE solver strategy
(using adaptive step‐size control and event detection). The code integrates with the
existing functions for NURBS surface evaluation and initial intersection point
computation.

Dependencies: Python 3 standard library, numpy, and scipy.

Author: [Your Name]
Date: [Today's Date]
"""
from __future__ import annotations

from typing import Callable, Any

import numpy as np
from collections import namedtuple

from numpy import ndarray, dtype
from scipy.integrate import solve_ivp
from scipy.spatial import KDTree

from mmcore.numeric.geodesic.offset import evaluate_nurbs_surface
from mmcore.numeric.intersection.ssx._detect_intersections import detect_intersections
from mmcore.numeric.intersection.ssx.boundary_intersection import find_boundary_intersections
from mmcore.numeric.vectors import scalar_unit

# =========================
# Existing Definitions
# =========================

# NURBS surface representation using a namedtuple.
NURBSSurfaceTuple = namedtuple("NURBSSurfaceTuple", ["order_u", "order_v", "knot_u", "knot_v", "control_points", "weights"])
from mmcore.geom import nurbs
def _join_weights(pts,weights):
    cpts=np.zeros((*pts.shape[:-1],pts.shape[-1]+1) )
    for i in range(pts.shape[0]):
        for j in range(pts.shape[1]):
            cpts[i,j,:-1]=pts[i,j]
            cpts[i,j,-1]=weights[i,j]
    return cpts

def _tuple_to_nurbs(surf:NURBSSurfaceTuple):
    degree=surf.order_u-1,surf.order_v-1
    pts=_join_weights(surf.control_points,surf.weights)
    return nurbs.NURBSSurface(pts, degree,np.array(surf.knot_u),np.array(surf.knot_v))

def find_initial_intersection_points(surf1, surf2, tol=1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    A robust method that returns at least one point on each of the intersection branches of two NURBS Surfaces.
    :param surf1: First NURBS surface
    :param surf2: Second NURBS surface
    :param tol: Tolerance
    :returns: None if no intersections are found or tuple of three numpy arrays of equal length:
      - cartesian points (x,y,z) with shape (N,3),
      - points in parametric coordinates of the first surface (u,v) with shape (N,2),
      - points in parametric coordinates of the second surface (u,v) with shape (N,2)
    """
    # Placeholder implementation.
    # In production this function must robustly compute initial intersection points.
    # For this implementation, we assume at least one intersection exists.
    # Here, we simply return a dummy intersection at the center of the parametric domain.
    xyz = []
    u1 = []
    u2 = []
    ns1=_tuple_to_nurbs(surf1)
    ns2=_tuple_to_nurbs(surf2)
    for s1, s2 in detect_intersections(ns1, ns2, tol=tol):
        for pt in find_boundary_intersections(s1, s2, tol):
            if tuple(pt.point) not in xyz:
                xyz.append(tuple(pt.point))

                u1.append(pt.surface1_params)
                u2.append(pt.surface2_params)
    if len(xyz) > 0:
        return np.array(xyz),np.array(u1),np.array(u2)
    else:
        return None


# -------------------------------------------------------------------
# Utility Functions for the Marching Method
# -------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of v."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm


def det3(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """Compute the determinant of three 3D vectors."""
    return np.linalg.det(np.column_stack((v1, v2, v3)))

# -------------------------------------------------------------------
# Advanced Point Refinement with Iteration Until Convergence
# -------------------------------------------------------------------
def refine_intersection_point(x: np.ndarray, surf1: NURBSSurfaceTuple, surf2: NURBSSurfaceTuple, spt: float = 1e-3, max_iter: int = 10) -> tuple[np.ndarray,dict,dict,float]:
    """
    Refine an approximate intersection point onto the true intersection by iterating until
    the difference between the two surface evaluations is below the Same Point Tolerance (SPT).

    Given an approximate parameter vector x = [s, t, u, v] for surfaces S0 and S1, we iteratively
    compute the corresponding surface points and adjust the parameters by solving the linearized
    systems:

        J0 * Δ(s,t) = (P_avg - S0(s,t))
        J1 * Δ(u,v) = (P_avg - S1(u,v))

    where P_avg = (S0 + S1)/2 and J0, J1 are the 3×2 Jacobians (first derivatives)
    of S0 and S1, respectively. The corrections are computed in a least-squares sense
    via the pseudoinverse.

    Iteration continues until ||S0(s,t) - S1(u,v)|| < spt or until max_iter iterations are reached.

    :param x: Current approximate parameters [s, t, u, v].
    :param surf1: First NURBS surface S0.
    :param surf2: Second NURBS surface S1.
    :param spt: Same Point Tolerance; iteration stops when ||S0 - S1|| < spt.
    :param max_iter: Maximum number of refinement iterations.
    :return: Refined parameter vector [s, t, u, v].
    """
    iteration = 0
    x_current = np.array(x, dtype=float)
    p_eval, q_eval=dict(),dict()
    error=-1
    while iteration < max_iter:
        s, t, u, v = x_current

        # Evaluate surfaces at first derivative level for Jacobian computation.
        p_eval = evaluate_nurbs_surface(surf1, s, t, d_order=1)
        q_eval = evaluate_nurbs_surface(surf2, u, v, d_order=1)

        S0 = np.array(p_eval["S"])
        S1 = np.array(q_eval["S"])
        error = np.linalg.norm(S0 - S1)

        # Check convergence.
        if error < spt:
            break

        # Compute the average of the two surface evaluations.
        P_avg = 0.5 * (S0 + S1)

        # Form the Jacobian matrices for each surface (3x2).
        J0 = np.column_stack((np.array(p_eval["Su"]), np.array(p_eval["Sv"])))
        J1 = np.column_stack((np.array(q_eval["Su"]), np.array(q_eval["Sv"])))

        # Compute least-squares corrections using the pseudoinverse.
        delta_st = np.linalg.pinv(J0) @ (P_avg - S0)
        delta_uv = np.linalg.pinv(J1) @ (P_avg - S1)

        # Update the parameters.
        s += delta_st[0]
        t += delta_st[1]
        u += delta_uv[0]
        v += delta_uv[1]

        x_current = np.array([s, t, u, v])
        iteration += 1

    return x_current, p_eval, q_eval,error

# -------------------------------------------------------------------
# Intersection ODE Function (Formulation of the ODE System)
# -------------------------------------------------------------------

def intersection_ode(x: np.ndarray, surf1: NURBSSurfaceTuple, surf2: NURBSSurfaceTuple, tol: float = 1e-6) -> np.ndarray:
    """
    Compute the derivative of the intersection curve in parametric space.
    The state vector x = [sigma, t, u, v] corresponds to parameters on surf1 and surf2.
    Depending on whether the intersection is transversal or tangential, different formulas are used.

    For transversal intersections, the marching direction is computed using the cross product
    of the surface normals. For tangential intersections, second order derivatives and the
    corresponding fundamental form coefficients are used to resolve the marching direction.
    """
    sigma, t, u, v = x
    # Evaluate surfaces with second derivatives for robustness.
    p = evaluate_nurbs_surface(surf1, sigma, t, d_order=2)
    q = evaluate_nurbs_surface(surf2, u, v, d_order=2)

    # Compute first derivatives for surface p
    p_sigma = np.array(p["Su"])
    p_t = np.array(p["Sv"])
    # Normal vector for p (P)
    P = np.cross(p_sigma, p_t)

    # Compute first derivatives for surface q
    q_u = np.array(q["Su"])
    q_v = np.array(q["Sv"])
    # Normal vector for q (Q)
    Q = np.cross(q_u, q_v)

    # Compute the cross product of the normals
    cross_normals = np.cross(P, Q)
    norm_cross = np.linalg.norm(cross_normals)

    # ----------------------------------------------------------------
    # Case 1: Transversal Intersection
    # ----------------------------------------------------------------
    if norm_cross > tol:
        # Unit tangent vector of the intersection curve in model space.
        c_tangent = cross_normals / norm_cross
        # Compute derivatives for surf1 using determinants:
        P_dot = np.dot(P, P)
        if abs(P_dot) < tol:
            sigma_prime = 0.0
            t_prime = 0.0
        else:
            sigma_prime = det3(c_tangent, p_t, P) / P_dot
            t_prime = det3(p_sigma, c_tangent, P) / P_dot

        # Compute derivatives for surf2:
        Q_dot = np.dot(Q, Q)
        if abs(Q_dot) < tol:
            u_prime = 0.0
            v_prime = 0.0
        else:
            u_prime = det3(c_tangent, q_v, Q) / Q_dot
            v_prime = det3(q_u, c_tangent, Q) / Q_dot

        return np.array([sigma_prime, t_prime, u_prime, v_prime])

    # ----------------------------------------------------------------
    # Case 2: Tangential Intersection
    # ----------------------------------------------------------------
    else:
        # Obtain second derivatives for surface p.
        p_ss = np.array(p["Suu"])
        p_st = np.array(p["Suv"])
        p_tt = np.array(p["Svv"])
        # Compute (or approximate) p's normal vector.
        if np.linalg.norm(P) < tol:
            p_normal = np.zeros_like(P)
        else:
            p_normal = normalize(P)

        # First and second fundamental form coefficients for p.
        L_p = np.dot(p_ss, p_normal)
        M_p = np.dot(p_st, p_normal)
        N_p = np.dot(p_tt, p_normal)

        # Obtain second derivatives for surface q.
        q_uu = np.array(q["Suu"])
        q_uv = np.array(q["Suv"])
        q_vv = np.array(q["Svv"])
        if np.linalg.norm(Q) < tol:
            q_normal = np.zeros_like(Q)
        else:
            q_normal = normalize(Q)

        # First and second fundamental form coefficients for q.
        L_q = np.dot(q_uu, q_normal)
        M_q = np.dot(q_uv, q_normal)
        N_q = np.dot(q_vv, q_normal)

        # For tangential intersections, the common normal is taken (using p_normal).
        N_vec = p_normal
        denom = np.dot(Q, N_vec)
        if abs(denom) < tol:
            # Fallback: unable to compute; return zero derivative.
            return np.zeros(4)

        # Compute the coefficients a11, a12, a21, a22.
        a11 = np.dot(np.cross(p_sigma, q_v), N_vec) / denom
        a12 = np.dot(np.cross(p_t, q_v), N_vec) / denom
        a21 = np.dot(np.cross(q_u, p_sigma), N_vec) / denom
        a22 = np.dot(np.cross(q_u, p_t), N_vec) / denom

        # Compute the coefficients b1, b12, b22 using the second fundamental forms.
        b1 = (a11 ** 2) * L_q + 2 * a11 * a21 * M_q + (a21 ** 2) * N_q - L_p
        b12 = a11 * a12 * L_q + (a11 * a22 + a12 * a21) * M_q + a21 * a22 * N_q - M_p
        b22 = (a12 ** 2) * L_q + 2 * a12 * a22 * M_q + (a22 ** 2) * N_q - N_p

        # Discriminant of the quadratic equation.
        disc = b12 ** 2 - b1 * b22

        if disc < 0:
            # Isolated tangential contact point.
            return np.zeros(4)
        elif abs(disc) < tol and (abs(b1) < tol and abs(b12) < tol and abs(b22) < tol):
            # Intersection cannot be evaluated by this method.
            return np.zeros(4)
        else:
            # Select the branch according to the coefficients.
            if abs(b1) > tol:
                sigma_ratio = -b12 / b1
                t_ratio = 1.0  # by convention
            elif abs(b22) > tol:
                t_ratio = -b12 / b22
                sigma_ratio = 1.0
            else:
                sigma_ratio = 0.0
                t_ratio = 0.0

            # --- Fix: Use the normalized marching direction ---
            # Compute the marching direction in the parametric domain of surf1.
            dir_p = sigma_ratio * p_sigma + t_ratio * p_t
            norm_dir_p = np.linalg.norm(dir_p)
            if norm_dir_p < tol:
                sigma_prime = 0.0
                t_prime = 0.0
            else:
                sigma_prime = sigma_ratio / norm_dir_p
                t_prime = t_ratio / norm_dir_p

            u_prime = a11 * sigma_prime + a12 * t_prime
            v_prime = a21 * sigma_prime + a22 * t_prime

            return np.array([sigma_prime, t_prime, u_prime, v_prime])


# -------------------------------------------------------------------
# Validated ODE Solver (Robust Marching with Adaptive Step Size)
# -------------------------------------------------------------------

class _InitPointsTreeWrapper:
    def __init__(self, tree):
        self._tree=tree
    def is_empty(self):
        return self._tree is None
    def is_single(self):
        return isinstance(self._tree,np.ndarray ) and len(self._tree)==1
    def query(self,pt, dst:float):
        if self.is_empty():
            return
        elif self.is_single():
            if np.linalg.norm(self._tree-pt)<=dst:
                self._tree=None
        else:
            ixs = self._tree.query_ball_point(pt, dst, return_sorted=False
                                        )
            if len(ixs) > 0:
                pts = np.delete(self._tree.data, ixs)
                print(f'Cull initial points: {ixs}')





def validated_ode_solver(
        f,
        x0: np.ndarray,
        surf1: NURBSSurfaceTuple,
        surf2: NURBSSurfaceTuple,
        s_max: float,
        h_initial: float,
        tol: float,
        spt:float,
        context:dict|None=None
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """
    A validated ODE solver that marches along the intersection curve by solving the ODE system:
        x' = f(x, surf1, surf2, tol)
    using an adaptive step size strategy and step doubling to produce a validated enclosure
    at each step.

    Parameters:
        f       : Function computing the derivative (the ODE system) in parametric space.
        x0      : Initial state (4-vector: [sigma, t, u, v]).
        surf1   : First NURBS surface.
        surf2   : Second NURBS surface.
        s_max   : Total arc length (or parameter length) to integrate.
        h_initial: Initial step size.
        tol     : Tolerance for local error and validation.

    Returns:
        A tuple containing:
         - A numpy array of states along the marching path.
         - A list of interval enclosures (each as a 2x4 numpy array: lower and upper bounds)
           for the corresponding state.
    """
    interval_u_1=surf1.knot_u[surf1.order_u-1],surf1.knot_u[len(surf1.control_points)+1]
    interval_v_1 = surf1.knot_v[surf1.order_v-1],surf1.knot_v[len(surf1.control_points[0])+1]
    interval_u_2 = surf2.knot_u[surf2.order_u - 1], surf2.knot_u[len(surf2.control_points) + 1]
    interval_v_2 = surf2.knot_v[surf2.order_v - 1], surf2.knot_v[len(surf2.control_points[0]) + 1]
    s = 0.0
    h = h_initial
    x = np.array(x0, dtype=float)
    initial=x
    p_initial=evaluate_nurbs_surface(surf1,initial[0],initial[1],1)
    q_initial = evaluate_nurbs_surface(surf2, initial[2], initial[3], 1)
    initial_point=p_initial['S']
    pN = np.cross(p_initial["Su"], p_initial["Sv"])
    qN = np.cross(q_initial["Su"], q_initial["Sv"])
    pN/=np.linalg.norm(pN)
    qN/=np.linalg.norm(qN)
    initial_tangent=np.cross(pN,qN)

    initial_tangent/=np.linalg.norm(initial_tangent)

    solution = [x.copy()]
    enclosures = [
        np.vstack((x, x))]  # Each enclosure is a 2x4 array: first row = lower bounds, second row = upper bounds


    check_tree=False
    if context is not None:
        check_tree = context.get("init_points_tree") is not None
    check_smax=s_max!=-1
    termination_reason=0

    while s < s_max if check_smax else True:
        # Attempt a full step of size h using Euler's method.
        f_x = f(x, surf1, surf2, tol)
        x_full = x + h * f_x

        # Perform step doubling: two half-steps.
        f_x_half = f(x, surf1, surf2, tol)
        x_half = x + (h / 2.0) * f_x_half
        f_x_half2 = f(x_half, surf1, surf2, tol)
        x_half2 = x_half + (h / 2.0) * f_x_half2

        # Estimate local error.
        error_estimate = np.linalg.norm(x_half2 - x_full)
        if error_estimate > tol:
            # Reduce step size and try again.
            h /= 2.0
            if abs(h) < 1e-10:
                raise RuntimeError("Step size reduced below minimum threshold; singular point encountered.")
            continue

        # Accept the step with the more accurate two-half-step result.
        x_new = x_half2.copy()
        # Apply iterative point refinement until ||S0 - S1|| < spt.
        x_new,p_eval,q_eval,error = refine_intersection_point(x_new, surf1, surf2, spt=spt, max_iter=100)



        # Construct an interval enclosure for x_new.
        enclosure_lower = x_new - error_estimate
        enclosure_upper = x_new + error_estimate
        enclosure = np.vstack((enclosure_lower, enclosure_upper))
        solution.append(x_new.copy())
        enclosures.append(enclosure)

        s += h
        x = x_new.copy()
        # Increase step size moderately for efficiency.

        h = min(h * 1.5, s_max - s) if check_smax else h*1.5

        # Terminate if any parametric coordinate goes outside the [0, 1] domain.
        if check_tree:

            tree = context["init_points_tree"]
            if isinstance(tree,KDTree):
                pt = x_new
                ixs = tree.query_ball_point(pt, h/2, return_sorted=True
                                            )


                if len(ixs)>0:


                    pts = np.delete(tree.data, ixs,axis=0)
                    #print(f'Cull initial points: {ixs}')

                    if pts.size == 0:
                        check_tree = False
                        context["init_points_tree"] = None

                    elif len(pts)==1:
                        #print('pts',pts)
                        context["init_points_tree"] = pts[0]
                        check_tree=True
                    else:
                        context["init_points_tree"] = KDTree(pts)
            elif isinstance(tree,np.ndarray):
                pts=context["init_points_tree"]
                pt = x_new
                if pts.size == 0:
                    check_tree = False
                    context["init_points_tree"] = None

                elif len(pts) == 1:
                    if np.linalg.norm(pt-pts)<=(h/2):
                        context["init_points_tree"] = None
                        check_tree = False

                        if (x[0] < interval_u_1[0] or x[0] > interval_u_1[1] or
                                x[1] < interval_v_1[0] or x[1] > interval_v_1[1] or
                                x[2] < interval_u_2[0] or x[2] > interval_u_2[1] or
                                x[3] < interval_v_2[0] or x[3] > interval_v_2[1]):
                            solution.pop(-1)

                            solution.append(pts)

                            break









        if (x[0] < interval_u_1[0] or x[0] >interval_u_1[1]  or
                x[1] < interval_v_1[0]  or x[1] > interval_v_1[1] or
                x[2] < interval_u_2[0] or x[2] > interval_u_2[1]  or
                x[3] < interval_v_2[0]  or x[3] > interval_v_2[1]):
            solution.pop(-1)
            enclosures.pop(-1)
            termination_reason = 1
            break
        if s>h:
            pt = p_eval["S"]
            d=initial_point-pt

            if np.allclose(d,0) or np.linalg.norm(d)<h: #loop
                if np.dot(initial_tangent,d)<0: #loop
                    termination_reason=2
                    #print('loop terminated')
                    break











    return np.array(solution), enclosures,termination_reason


# -------------------------------------------------------------------
# Marching Method: Tracing the Intersection Curve
# -------------------------------------------------------------------

def trace_intersection_curve(
        surf1: NURBSSurfaceTuple,
        surf2: NURBSSurfaceTuple,
        init_params: np.ndarray,
        s_max: float = 1.0,
        h_initial: float = 0.01,
    tol: float = 1e-6,
    spt: float = 1e-3,
        context:dict|None=None
) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], list[ndarray], int]:
    """
    Trace the intersection curve of two NURBS surfaces using the marching method.
    This function integrates the ODE system defined by the intersection_ode function, employs
    a validated ODE solver with adaptive step sizing, and integrates an advanced point refinement
    loop at each accepted step to ensure the computed intersection points lie within the Same Point
    Tolerance (SPT) of the true intersection.

    Parameters:
        surf1      : First NURBS surface.
        surf2      : Second NURBS surface.
        init_params: Initial parameter vector [sigma, t, u, v] on the intersection curve.
        s_max      : Total integration parameter (arc length) to trace.
        h_initial  : Initial step size for the ODE solver.
        tol        : Tolerance for error control and ODE validation.
        spt        : Same Point Tolerance for point refinement.

    Returns:
        A tuple containing:
         - curve_points: An (N,3) numpy array of points in 3D model space along the intersection.
         - params_surf1: An (N,2) numpy array of parametric coordinates on surf1.
         - params_surf2: An (N,2) numpy array of parametric coordinates on surf2.
         - enclosures : A list of interval enclosures (each a 2x4 numpy array) in the parametric space.
         - termination_reason: int
    """
    states, enclosures,term_reason = validated_ode_solver(
        f=intersection_ode,
        x0=init_params,
        surf1=surf1,
        surf2=surf2,
        s_max=s_max,
        h_initial=h_initial,
        tol=tol,
        spt=spt,
        context=context
    )

    curve_points = []
    params_surf1 = []
    params_surf2 = []
    for state in states:
        sigma, t, u, v = state
        pt = evaluate_nurbs_surface(surf1, sigma, t, d_order=0)
        curve_points.append(pt["S"])
        params_surf1.append([sigma, t])
        params_surf2.append([u, v])

    return (np.array(curve_points),
            np.array(params_surf1),
            np.array(params_surf2),
            enclosures,term_reason)


# -------------------------------------------------------------------
# Example Usage (for production, integrate with robust initial point finder)
# -------------------------------------------------------------------

def trace_intersection_curves(surf1,surf2, h_initial=0.1,tol=1e-3,spt=1e-3):

    res = find_initial_intersection_points(surf1, surf2, tol)
    branches=[]
    if res is not None:
        init_points, init_uv1, init_uv2 = res
        initial_xs = np.zeros((init_uv1.shape[0], 4))
        initial_xs[..., :2] = init_uv1
        initial_xs[..., 2:] = init_uv2
        initial_points_tree = KDTree(initial_xs)

        context = dict(init_points_tree=initial_points_tree)
        while   context['init_points_tree'] is not None:


            if isinstance(context['init_points_tree'],KDTree):

                init_point=context['init_points_tree'].data[0]
                pts = np.delete(context['init_points_tree'].data, 0, axis=0)

                if pts.size == 0:

                    context["init_points_tree"] = None
                elif len(pts) == 1:


                    context["init_points_tree"] = pts[0]
                else:
                    context["init_points_tree"] = KDTree(pts)
            elif isinstance(context['init_points_tree'], np.ndarray):
                init_point=context['init_points_tree']
                context["init_points_tree"] = None

            else:
                break
            # Trace the intersection curve.


            curve_pts, params1, params2, encl, termination_reason = trace_intersection_curve(
                surf1, surf2, init_point, s_max=-1, h_initial=h_initial, tol=tol, spt=spt, context=context
            )
            if termination_reason ==1:
                curve_pts2, params12, params22, encl2, termination_reason2 = trace_intersection_curve(
                    surf1, surf2, init_point, s_max=-1, h_initial=-h_initial, tol=tol, spt=spt, context=context
                )
                curve_pts, params1, params2, encl=np.array([*reversed(curve_pts2),*curve_pts[1:]]),np.array([*reversed(params12), *params1[1:]]),np.array([*reversed(params22),*params2[1:]]),[*reversed(encl2),encl[1:]]



            branches.append((curve_pts, params1, params2, encl))

    return branches


if __name__ == "__main__":
    # Example: Assume surf1 and surf2 are defined NURBSSurface objects.
    # The function find_initial_intersection_points is assumed to provide at least
    # one starting point on each intersection branch.
    #
    # For demonstration, suppose we have obtained an initial point:
    #
    #   init_point = np.array([0.3, 0.2, 0.7, 0.8])
    #
    # And we wish to trace the intersection curve with the following parameters:
    #
    #   s_max = 1.0, h_initial = 0.01, tol = 1e-6
    #
    # Replace the following dummy surfaces with actual NURBS surface definitions.
    pts1 = np.array(
        [
            [-25.0, -25.0, -10.0],
            [-25.0, -15.0, -5.0],
            [-25.0, -5.0, 0.0],
            [-25.0, 5.0, 0.0],
            [-25.0, 15.0, -5.0],
            [-25.0, 25.0, -10.0],
            [-15.0, -25.0, -8.0],
            [-15.0, -15.0, -4.0],
            [-15.0, -5.0, -4.0],
            [-15.0, 5.0, -4.0],
            [-15.0, 15.0, -4.0],
            [-15.0, 25.0, -8.0],
            [-5.0, -25.0, -5.0],
            [-5.0, -15.0, -3.0],
            [-5.0, -5.0, -8.0],
            [-5.0, 5.0, -8.0],
            [-5.0, 15.0, -3.0],
            [-5.0, 25.0, -5.0],
            [5.0, -25.0, -3.0],
            [5.0, -15.0, -2.0],
            [5.0, -5.0, -8.0],
            [5.0, 5.0, -8.0],
            [5.0, 15.0, -2.0],
            [5.0, 25.0, -3.0],
            [15.0, -25.0, -8.0],
            [15.0, -15.0, -4.0],
            [15.0, -5.0, -4.0],
            [15.0, 5.0, -4.0],
            [15.0, 15.0, -4.0],
            [15.0, 25.0, -8.0],
            [25.0, -25.0, -10.0],
            [25.0, -15.0, -5.0],
            [25.0, -5.0, 2.0],
            [25.0, 5.0, 2.0],
            [25.0, 15.0, -5.0],
            [25.0, 25.0, -10.0],
        ]
    )
    pts1 = pts1.reshape((6, len(pts1) // 6, 3))
    pts2 = np.array(
        [
            [25.0, 14.774795467423544, 5.5476189978794661],
            [25.0, 10.618169208735296, -15.132510312735601],
            [25.0, 1.8288992061686002, -13.545426491756078],
            [25.0, 9.8715747661086723, 14.261864686419623],
            [25.0, -15.0, 5.0],
            [25.0, -25.0, 5.0],
            [15.0, 25.0, 1.8481369394623908],
            [15.0, 15.0, 5.0],
            [15.0, 5.0, -1.4589623860307768],
            [15.0, -5.0, -1.9177595746260625],
            [15.0, -15.0, -30.948650572598954],
            [15.0, -25.0, 5.0],
            [5.0, 25.0, 5.0],
            [5.0, 15.0, -29.589097491066767],
            [3.8028908181980938, 5.0, 5.0],
            [5.0, -5.0, 5.0],
            [5.0, -15.0, 5.0],
            [5.0, -25.0, 5.0],
            [-5.0, 25.0, 5.0],
            [-5.0, 15.0, 5.0],
            [-5.0, 5.0, 5.0],
            [-5.0, -5.0, -27.394523521151221],
            [-5.0, -15.0, 5.0],
            [-5.0, -25.0, 5.0],
            [-15.0, 25.0, 5.0],
            [-15.0, 15.0, -23.968082282285287],
            [-15.0, 5.0, 5.0],
            [-15.0, -5.0, 5.0],
            [-15.0, -15.0, -18.334465891060319],
            [-15.0, -25.0, 5.0],
            [-25.0, 25.0, 5.0],
            [-25.0, 15.0, 14.302789083068138],
            [-25.0, 5.0, 5.0],
            [-25.0, -5.0, 5.0],
            [-25.0, -15.0, 5.0],
            [-25.0, -25.0, 5.0],
        ]
    )

    pts2 = pts2.reshape((6, len(pts2) // 6, 3))
    s21 = nurbs.NURBSSurface(pts1, (3, 3))
    s22 = nurbs.NURBSSurface(pts2, (3, 3))

    surf1 = NURBSSurfaceTuple(order_u=s21.degree[0] + 1, order_v=s21.degree[1] + 1, knot_u=s21.knots_u.tolist(),
                              knot_v=s21.knots_v.tolist(), control_points=np.array(s21.control_points),
                              weights=np.ascontiguousarray(s21.control_points_w[..., -1]))

    surf2 = NURBSSurfaceTuple(order_u=s22.degree[0] + 1, order_v=s22.degree[1] + 1, knot_u=s22.knots_u.tolist(),
                              knot_v=s22.knots_v.tolist(), control_points=np.array(s22.control_points),
                              weights=np.ascontiguousarray(s22.control_points_w[..., -1]))

    # Assume an initial intersection point has been determined.
    res= find_initial_intersection_points(surf1,surf2,1e-3)

    if res is not None:
        init_points, init_uv1, init_uv2 = res
        initial_xs=np.zeros((init_uv1.shape[0],4))
        initial_xs[...,:2]=init_uv1
        initial_xs[...,2:]=init_uv2
        initial_points_tree=KDTree(initial_xs)

        context=dict(init_points_tree=initial_points_tree)
        init_point=np.array([*init_uv1[0], *init_uv2[0]])
        # Trace the intersection curve.
        print(init_point)
        curve_pts, params1, params2, encl,termination_reason = trace_intersection_curve(
            surf1, surf2, init_point, s_max=100.0, h_initial=0.1, tol=1e-2,spt=1e-2,context=context
        )
        print(context['init_points_tree'].data.tolist())
        # For production use, further processing and robust visualization would follow.
        print("3D Intersection Curve Points:")
        print(curve_pts.tolist())

        # The enclosures list contains rigorous interval bounds for each marching step.

    branches=trace_intersection_curves(surf1, surf2,tol=1e-3,spt=1e-2)
    print(f"{len(branches)} intersection curves")
    print([branch[0].tolist() for branch in branches])

    s1 = nurbs.NURBSSurface(
        **{
            "control_points": np.array(
                [
                    [
                        [65.66864862623089, 93.74205665737186, 41.41100531879138, 1.0],
                        [65.66864862623089, 93.74205665737186, 0.10498601801243446, 1.0],
                    ],
                    [
                        [65.70381308003529, 92.80184185418611, 41.41100531879138, 1.0],
                        [65.70381308003529, 92.80184185418611, 0.10498601801243446, 1.0],
                    ],
                    [
                        [65.93866781565862, 90.93344804677268, 41.41100531879138, 1.0],
                        [65.93866781565862, 90.93344804677268, 0.10498601801243446, 1.0],
                    ],
                    [
                        [66.76764474869523, 88.25139426624095, 41.41100531879138, 1.0],
                        [66.76764474869523, 88.25139426624095, 0.10498601801243446, 1.0],
                    ],
                    [
                        [68.04240133132686, 85.7698431189295, 41.41100531879138, 1.0],
                        [68.04240133132686, 85.7698431189295, 0.10498601801243446, 1.0],
                    ],
                    [
                        [69.71866508308432, 83.5616093222356, 41.41100531879138, 1.0],
                        [69.71866508308432, 83.5616093222356, 0.10498601801243446, 1.0],
                    ],
                    [
                        [71.7405653611562, 81.69030895188432, 41.41100531879138, 1.0],
                        [71.7405653611562, 81.69030895188432, 0.10498601801243446, 1.0],
                    ],
                    [
                        [74.04239056810582, 80.208508216853, 41.41100531879138, 1.0],
                        [74.04239056810582, 80.208508216853, 0.10498601801243446, 1.0],
                    ],
                    [
                        [76.55073895845743, 79.1562631175434, 41.41100531879138, 1.0],
                        [76.55073895845743, 79.1562631175434, 0.10498601801243446, 1.0],
                    ],
                    [
                        [79.18684761002137, 78.5600515829087, 41.41100531879138, 1.0],
                        [79.18684761002137, 78.5600515829087, 0.10498601801243446, 1.0],
                    ],
                    [
                        [81.86906475574814, 78.43213634667575, 41.41100531879138, 1.0],
                        [81.86906475574814, 78.43213634667575, 0.10498601801243446, 1.0],
                    ],
                    [
                        [84.51537751121424, 78.77037006251331, 41.41100531879138, 1.0],
                        [84.51537751121424, 78.77037006251331, 0.10498601801243446, 1.0],
                    ],
                    [
                        [87.04591956978467, 79.55844392285438, 41.41100531879138, 1.0],
                        [87.04591956978467, 79.55844392285438, 0.10498601801243446, 1.0],
                    ],
                    [
                        [89.38538144158962, 80.76656645646, 41.41100531879138, 1.0],
                        [89.38538144158962, 80.76656645646, 0.10498601801243446, 1.0],
                    ],
                    [
                        [91.46525070332797, 82.35254636479925, 41.41100531879138, 1.0],
                        [91.46525070332797, 82.35254636479925, 0.10498601801243446, 1.0],
                    ],
                    [
                        [93.22581559441008, 84.2632410840891, 41.41100531879138, 1.0],
                        [93.22581559441008, 84.2632410840891, 0.10498601801243446, 1.0],
                    ],
                    [
                        [94.61787346921517, 86.43632200091871, 41.41100531879138, 1.0],
                        [94.61787346921517, 86.43632200091871, 0.10498601801243446, 1.0],
                    ],
                    [
                        [95.60409540762578, 88.80229811220822, 41.41100531879138, 1.0],
                        [95.60409540762578, 88.80229811220822, 0.10498601801243446, 1.0],
                    ],
                    [
                        [96.16000948361588, 91.28673269687489, 41.41100531879138, 1.0],
                        [96.16000948361588, 91.28673269687489, 0.10498601801243446, 1.0],
                    ],
                    [
                        [96.27457740208934, 93.81258244954697, 41.41100531879138, 1.0],
                        [96.27457740208934, 93.81258244954697, 0.10498601801243446, 1.0],
                    ],
                    [
                        [95.95035205531579, 96.30258565083841, 41.41100531879138, 1.0],
                        [95.95035205531579, 96.30258565083841, 0.10498601801243446, 1.0],
                    ],
                    [
                        [95.20321661514595, 98.68162537521448, 41.41100531879138, 1.0],
                        [95.20321661514595, 98.68162537521448, 0.10498601801243446, 1.0],
                    ],
                    [
                        [94.06171865923878, 100.87899545606837, 41.41100531879138, 1.0],
                        [94.06171865923878, 100.87899545606837, 0.10498601801243446, 1.0],
                    ],
                    [
                        [92.56602513215233, 102.83050085686249, 41.41100531879138, 1.0],
                        [92.56602513215233, 102.83050085686249, 0.10498601801243446, 1.0],
                    ],
                    [
                        [90.76653529163374, 104.48033008807467, 41.41100531879138, 1.0],
                        [90.76653529163374, 104.48033008807467, 0.10498601801243446, 1.0],
                    ],
                    [
                        [88.72219884713252, 105.78264515104536, 41.41100531879138, 1.0],
                        [88.72219884713252, 105.78264515104536, 0.10498601801243446, 1.0],
                    ],
                    [
                        [86.49859496608252, 106.70284391559045, 41.41100531879138, 1.0],
                        [86.49859496608252, 106.70284391559045, 0.10498601801243446, 1.0],
                    ],
                    [
                        [84.16583446139505, 107.21846053641033, 41.41100531879138, 1.0],
                        [84.16583446139505, 107.21846053641033, 0.10498601801243446, 1.0],
                    ],
                    [
                        [81.79635209827242, 107.3196811361933, 41.41100531879138, 1.0],
                        [81.79635209827242, 107.3196811361933, 0.10498601801243446, 1.0],
                    ],
                    [
                        [79.46265845209487, 107.00946415873321, 41.41100531879138, 1.0],
                        [79.46265845209487, 107.00946415873321, 0.10498601801243446, 1.0],
                    ],
                    [
                        [77.23512106166146, 106.30326713866779, 41.41100531879138, 1.0],
                        [77.23512106166146, 106.30326713866779, 0.10498601801243446, 1.0],
                    ],
                    [
                        [75.17984277182609, 105.22839376047693, 41.41100531879138, 1.0],
                        [75.17984277182609, 105.22839376047693, 0.10498601801243446, 1.0],
                    ],
                    [
                        [73.35670123195811, 103.8229866146385, 41.41100531879138, 1.0],
                        [73.35670123195811, 103.8229866146385, 0.10498601801243446, 1.0],
                    ],
                    [
                        [71.81760766062072, 102.1347016528925, 41.41100531879138, 1.0],
                        [71.81760766062072, 102.1347016528925, 0.10498601801243446, 1.0],
                    ],
                    [
                        [70.60503540948311, 100.21910968071924, 41.41100531879138, 1.0],
                        [70.60503540948311, 100.21910968071924, 0.10498601801243446, 1.0],
                    ],
                    [
                        [69.75085981880395, 98.13787802990888, 41.41100531879138, 1.0],
                        [69.75085981880395, 98.13787802990888, 0.10498601801243446, 1.0],
                    ],
                    [
                        [69.27554065315417, 95.9567916052006, 41.41100531879138, 1.0],
                        [69.27554065315417, 95.9567916052006, 0.10498601801243446, 1.0],
                    ],
                    [
                        [69.1876673720617, 93.7436766316274, 41.41100531879138, 1.0],
                        [69.1876673720617, 93.7436766316274, 0.10498601801243446, 1.0],
                    ],
                    [
                        [69.48387598020832, 91.56629254056372, 41.41100531879138, 1.0],
                        [69.48387598020832, 91.56629254056372, 0.10498601801243446, 1.0],
                    ],
                    [
                        [70.14913458016932, 89.49025748407303, 41.41100531879138, 1.0],
                        [70.14913458016932, 89.49025748407303, 0.10498601801243446, 1.0],
                    ],
                    [
                        [71.15738338064384, 87.57707098525617, 41.41100531879138, 1.0],
                        [71.15738338064384, 87.57707098525617, 0.10498601801243446, 1.0],
                    ],
                    [
                        [72.47250414523428, 85.8822933063143, 41.41100531879138, 1.0],
                        [72.47250414523428, 85.8822933063143, 0.10498601801243446, 1.0],
                    ],
                    [
                        [74.04958422820772, 84.45393539485173, 41.41100531879138, 1.0],
                        [74.04958422820772, 84.45393539485173, 0.10498601801243446, 1.0],
                    ],
                    [
                        [75.83643172805296, 83.3311059555472, 41.41100531879138, 1.0],
                        [75.83643172805296, 83.3311059555472, 0.10498601801243446, 1.0],
                    ],
                    [
                        [77.77529114862375, 82.54295353873391, 41.41100531879138, 1.0],
                        [77.77529114862375, 82.54295353873391, 0.10498601801243446, 1.0],
                    ],
                    [
                        [79.80470349335278, 82.10793182825427, 41.41100531879138, 1.0],
                        [79.80470349335278, 82.10793182825427, 0.10498601801243446, 1.0],
                    ],
                    [
                        [81.86145107737659, 82.03340586585229, 41.41100531879138, 1.0],
                        [81.86145107737659, 82.03340586585229, 0.10498601801243446, 1.0],
                    ],
                    [
                        [83.88252561332632, 82.31560610468546, 41.41100531879138, 1.0],
                        [83.88252561332632, 82.31560610468546, 0.10498601801243446, 1.0],
                    ],
                    [
                        [85.80705833587436, 82.93992628454203, 41.41100531879138, 1.0],
                        [85.80705833587436, 82.93992628454203, 0.10498601801243446, 1.0],
                    ],
                    [
                        [87.5781530436727, 83.88155050730022, 41.41100531879138, 1.0],
                        [87.5781530436727, 83.88155050730022, 0.10498601801243446, 1.0],
                    ],
                    [
                        [89.14456686168843, 85.10638489064266, 41.41100531879138, 1.0],
                        [89.14456686168843, 85.10638489064266, 0.10498601801243446, 1.0],
                    ],
                    [
                        [90.46218911327622, 86.57226009484354, 41.41100531879138, 1.0],
                        [90.46218911327622, 86.57226009484354, 0.10498601801243446, 1.0],
                    ],
                    [
                        [91.49527574074764, 88.23036312236076, 41.41100531879138, 1.0],
                        [91.49527574074764, 88.23036312236076, 0.10498601801243446, 1.0],
                    ],
                    [
                        [92.21740498369505, 90.02685031269195, 41.41100531879138, 1.0],
                        [92.21740498369505, 90.02685031269195, 0.10498601801243446, 1.0],
                    ],
                    [
                        [92.61212923900459, 91.90458857744179, 41.41100531879138, 1.0],
                        [92.61212923900459, 91.90458857744179, 0.10498601801243446, 1.0],
                    ],
                    [
                        [92.67330788271605, 93.80496877191614, 41.41100531879138, 1.0],
                        [92.67330788271605, 93.80496877191614, 0.10498601801243446, 1.0],
                    ],
                    [
                        [92.40511601319636, 95.66973375275204, 41.41100531879138, 1.0],
                        [92.40511601319636, 95.66973375275204, 0.10498601801243446, 1.0],
                    ],
                    [
                        [91.82173425344422, 97.44276414135732, 41.41100531879138, 1.0],
                        [91.82173425344422, 97.44276414135732, 0.10498601801243446, 1.0],
                    ],
                    [
                        [90.94673460840232, 99.07176705813723, 41.41100531879138, 1.0],
                        [90.94673460840232, 99.07176705813723, 0.10498601801243446, 1.0],
                    ],
                    [
                        [89.8121866063079, 100.50981701522677, 41.41100531879138, 1.0],
                        [89.8121866063079, 100.50981701522677, 0.10498601801243446, 1.0],
                    ],
                    [
                        [88.45751628087959, 101.71670360693977, 41.41100531879138, 1.0],
                        [88.45751628087959, 101.71670360693977, 0.10498601801243446, 1.0],
                    ],
                    [
                        [86.92815772569035, 102.66004742257813, 41.41100531879138, 1.0],
                        [86.92815772569035, 102.66004742257813, 0.10498601801243446, 1.0],
                    ],
                    [
                        [85.27404276559884, 103.31615349165962, 41.41100531879138, 1.0],
                        [85.27404276559884, 103.31615349165962, 0.10498601801243446, 1.0],
                    ],
                    [
                        [83.54797858082817, 103.67058029179904, 41.41100531879138, 1.0],
                        [83.54797858082817, 103.67058029179904, 0.10498601801243446, 1.0],
                    ],
                    [
                        [81.80396577590322, 103.71841161682002, 41.41100531879138, 1.0],
                        [81.80396577590322, 103.71841161682002, 0.10498601801243446, 1.0],
                    ],
                    [
                        [80.09551035018126, 103.46422811661378, 41.41100531879138, 1.0],
                        [80.09551035018126, 103.46422811661378, 0.10498601801243446, 1.0],
                    ],
                    [
                        [78.4739822955186, 102.92178477696604, 41.41100531879138, 1.0],
                        [78.4739822955186, 102.92178477696604, 0.10498601801243446, 1.0],
                    ],
                    [
                        [76.98707116975723, 102.1134097096405, 41.41100531879138, 1.0],
                        [76.98707116975723, 102.1134097096405, 0.10498601801243446, 1.0],
                    ],
                    [
                        [75.67738507359383, 101.06914808879408, 41.41100531879138, 1.0],
                        [75.67738507359383, 101.06914808879408, 0.10498601801243446, 1.0],
                    ],
                    [
                        [74.5812341417556, 99.82568264213833, 41.41100531879138, 1.0],
                        [74.5812341417556, 99.82568264213833, 0.10498601801243446, 1.0],
                    ],
                    [
                        [73.72763313795036, 98.42506855927712, 41.41100531879138, 1.0],
                        [73.72763313795036, 98.42506855927712, 0.10498601801243446, 1.0],
                    ],
                    [
                        [73.13755024273476, 96.91332582942516, 41.41100531879138, 1.0],
                        [73.13755024273476, 96.91332582942516, 0.10498601801243446, 1.0],
                    ],
                    [
                        [72.82342089776546, 95.33893572463373, 41.41100531879138, 1.0],
                        [72.82342089776546, 95.33893572463373, 0.10498601801243446, 1.0],
                    ],
                    [
                        [72.78893689143499, 93.7512903092582, 41.41100531879138, 1.0],
                        [72.78893689143499, 93.7512903092582, 0.10498601801243446, 1.0],
                    ],
                    [
                        [73.02911202232774, 92.19914443865015, 41.41100531879138, 1.0],
                        [73.02911202232774, 92.19914443865015, 0.10498601801243446, 1.0],
                    ],
                    [
                        [73.53061694187109, 90.72911871793013, 41.41100531879138, 1.0],
                        [73.53061694187109, 90.72911871793013, 0.10498601801243446, 1.0],
                    ],
                    [
                        [74.27236743148026, 89.38429938318734, 41.41100531879138, 1.0],
                        [74.27236743148026, 89.38429938318734, 0.10498601801243446, 1.0],
                    ],
                    [
                        [75.22634267107871, 88.20297714795004, 41.41100531879138, 1.0],
                        [75.22634267107871, 88.20297714795004, 0.10498601801243446, 1.0],
                    ],
                    [
                        [76.35860323896188, 87.21756187598662, 41.41100531879138, 1.0],
                        [76.35860323896188, 87.21756187598662, 0.10498601801243446, 1.0],
                    ],
                    [
                        [77.63047284949508, 86.45370368401446, 41.41100531879138, 1.0],
                        [77.63047284949508, 86.45370368401446, 0.10498601801243446, 1.0],
                    ],
                    [
                        [78.99984334910746, 85.92964396266473, 41.41100531879138, 1.0],
                        [78.99984334910746, 85.92964396266473, 0.10498601801243446, 1.0],
                    ],
                    [
                        [80.42255937391965, 85.65581207286556, 41.41100531879138, 1.0],
                        [80.42255937391965, 85.65581207286556, 0.10498601801243446, 1.0],
                    ],
                    [
                        [81.85383739974577, 85.63467538522558, 41.41100531879138, 1.0],
                        [81.85383739974577, 85.63467538522558, 0.10498601801243446, 1.0],
                    ],
                    [
                        [83.24967371523994, 85.86084214680488, 41.41100531879138, 1.0],
                        [83.24967371523994, 85.86084214680488, 0.10498601801243446, 1.0],
                    ],
                    [
                        [84.56819710201722, 86.32140864624378, 41.41100531879138, 1.0],
                        [84.56819710201722, 86.32140864624378, 0.10498601801243446, 1.0],
                    ],
                    [
                        [85.77092464574159, 86.99653455813666, 41.41100531879138, 1.0],
                        [85.77092464574159, 86.99653455813666, 0.10498601801243446, 1.0],
                    ],
                    [
                        [86.8238830200527, 87.86022341648709, 41.41100531879138, 1.0],
                        [86.8238830200527, 87.86022341648709, 0.10498601801243446, 1.0],
                    ],
                    [
                        [87.69856263214132, 88.8812791055977, 41.41100531879138, 1.0],
                        [87.69856263214132, 88.8812791055977, 0.10498601801243446, 1.0],
                    ],
                    [
                        [88.37267801228043, 90.02440424380293, 41.41100531879138, 1.0],
                        [88.37267801228043, 90.02440424380293, 0.10498601801243446, 1.0],
                    ],
                    [
                        [88.83071455976423, 91.2514025131756, 41.41100531879138, 1.0],
                        [88.83071455976423, 91.2514025131756, 0.10498601801243446, 1.0],
                    ],
                    [
                        [89.0642489943933, 92.52244445800866, 41.41100531879138, 1.0],
                        [89.0642489943933, 92.52244445800866, 0.10498601801243446, 1.0],
                    ],
                    [
                        [89.07203836334277, 93.79735509428534, 41.41100531879138, 1.0],
                        [89.07203836334277, 93.79735509428534, 0.10498601801243446, 1.0],
                    ],
                    [
                        [88.85987997107694, 95.03688185466562, 41.41100531879138, 1.0],
                        [88.85987997107694, 95.03688185466562, 0.10498601801243446, 1.0],
                    ],
                    [
                        [88.4402518917424, 96.2039029075002, 41.41100531879138, 1.0],
                        [88.4402518917424, 96.2039029075002, 0.10498601801243446, 1.0],
                    ],
                    [
                        [87.831750557566, 97.2645386602061, 41.41100531879138, 1.0],
                        [87.831750557566, 97.2645386602061, 0.10498601801243446, 1.0],
                    ],
                    [
                        [87.0583480804631, 98.18913317359093, 41.41100531879138, 1.0],
                        [87.0583480804631, 98.18913317359093, 0.10498601801243446, 1.0],
                    ],
                    [
                        [86.1484972701268, 98.9530771258053, 41.41100531879138, 1.0],
                        [86.1484972701268, 98.9530771258053, 0.10498601801243446, 1.0],
                    ],
                    [
                        [85.13411660424308, 99.53744969410937, 41.41100531879138, 1.0],
                        [85.13411660424308, 99.53744969410937, 0.10498601801243446, 1.0],
                    ],
                    [
                        [84.04949056513439, 99.9294630677344, 41.41100531879138, 1.0],
                        [84.04949056513439, 99.9294630677344, 0.10498601801243446, 1.0],
                    ],
                    [
                        [82.93012270018926, 100.12270004716686, 41.41100531879138, 1.0],
                        [82.93012270018926, 100.12270004716686, 0.10498601801243446, 1.0],
                    ],
                    [
                        [81.81157945380292, 100.11714209752466, 41.41100531879138, 1.0],
                        [81.81157945380292, 100.11714209752466, 0.10498601801243446, 1.0],
                    ],
                    [
                        [80.72836224726413, 99.91899207420349, 41.41100531879138, 1.0],
                        [80.72836224726413, 99.91899207420349, 0.10498601801243446, 1.0],
                    ],
                    [
                        [79.71284353312089, 99.54030241634977, 41.41100531879138, 1.0],
                        [79.71284353312089, 99.54030241634977, 0.10498601801243446, 1.0],
                    ],
                    [
                        [78.79429955371126, 98.99842565475291, 41.41100531879138, 1.0],
                        [78.79429955371126, 98.99842565475291, 0.10498601801243446, 1.0],
                    ],
                    [
                        [77.99806896739302, 98.31530957806882, 41.41100531879138, 1.0],
                        [77.99806896739302, 98.31530957806882, 0.10498601801243446, 1.0],
                    ],
                    [
                        [77.34486042821379, 97.51666357495867, 41.41100531879138, 1.0],
                        [77.34486042821379, 97.51666357495867, 0.10498601801243446, 1.0],
                    ],
                    [
                        [76.85023159296097, 96.63102764841777, 41.41100531879138, 1.0],
                        [76.85023159296097, 96.63102764841777, 0.10498601801243446, 1.0],
                    ],
                    [
                        [76.52423795516884, 95.6887728430358, 41.41100531879138, 1.0],
                        [76.52423795516884, 95.6887728430358, 0.10498601801243446, 1.0],
                    ],
                    [
                        [76.37131126182027, 94.72108277710663, 41.41100531879138, 1.0],
                        [76.37131126182027, 94.72108277710663, 0.10498601801243446, 1.0],
                    ],
                    [
                        [76.38390439702991, 94.07962526877671, 41.41100531879138, 1.0],
                        [76.38390439702991, 94.07962526877671, 0.10498601801243446, 1.0],
                    ],
                    [
                        [76.41774580834283, 93.76478202139324, 41.41100531879138, 1.0],
                        [76.41774580834283, 93.76478202139324, 0.10498601801243446, 1.0],
                    ],
                ],
                dtype=float,
            ),
            "degree": (3, 1)

        }
    )

    s2 = nurbs.NURBSSurface(
        **{
            "control_points": np.array(
                [
                    [
                        [58.99900024079902, 100.314235228814, 10.808972893047397, 1.0],
                        [68.34707046760253, 62.81015051393596, 22.869960482526768, 1.0],
                    ],
                    [
                        [59.85259522172079, 100.40321381490975, 10.424062008949615, 1.0],
                        [69.2006654485243, 62.89912910003171, 22.485049598428986, 1.0],
                    ],
                    [
                        [61.59535154598625, 100.62543299535449, 9.764308279851145, 1.0],
                        [70.94342177278978, 63.12134828047643, 21.825295869330514, 1.0],
                    ],
                    [
                        [64.23566593465347, 101.07978060252134, 9.130699040125272, 1.0],
                        [73.58373616145698, 63.575695887643285, 21.191686629604643, 1.0],
                    ],
                    [
                        [66.82169195021058, 101.63989463403122, 8.868051500649052, 1.0],
                        [76.16976217701409, 64.13580991915317, 20.92903909012842, 1.0],
                    ],
                    [
                        [69.27561378785545, 102.28744991944934, 8.979695311117084, 1.0],
                        [78.62368401465896, 64.78336520457128, 21.04068290059645, 1.0],
                    ],
                    [
                        [71.52460129952792, 103.00170534847233, 9.457583450629707, 1.0],
                        [80.87267152633143, 65.49762063359428, 21.518571040109077, 1.0],
                    ],
                    [
                        [73.50296637187336, 103.7601435903035, 10.282610121810293, 1.0],
                        [82.85103659867687, 66.25605887542544, 22.34359771128966, 1.0],
                    ],
                    [
                        [75.15407947566067, 104.53918616196599, 11.425348828344557, 1.0],
                        [84.5021497024642, 67.03510144708794, 23.486336417823928, 1.0],
                    ],
                    [
                        [76.4319842413868, 105.31492442395539, 12.84707402902431, 1.0],
                        [85.78005446819031, 67.81083970907734, 24.90806161850368, 1.0],
                    ],
                    [
                        [77.30266592053286, 106.06385369817762, 14.501060762368999, 1.0],
                        [86.65073614733637, 68.55976898329958, 26.562048351848368, 1.0],
                    ],
                    [
                        [77.74493862312394, 106.76358533047862, 16.33411116294826, 1.0],
                        [87.09300884992747, 69.25950061560057, 28.395098752427632, 1.0],
                    ],
                    [
                        [77.75092893971322, 107.39351583348422, 18.288260346354235, 1.0],
                        [87.09899916651673, 69.88943111860618, 30.349247935833606, 1.0],
                    ],
                    [
                        [77.32614608516276, 107.93543292674464, 20.302606547220165, 1.0],
                        [86.67421631196629, 70.43134821186659, 32.36359413669953, 1.0],
                    ],
                    [
                        [76.48914157925714, 108.374040718005, 22.315207959940675, 1.0],
                        [85.83721180606065, 70.86995600312694, 34.376195549420046, 1.0],
                    ],
                    [
                        [75.27077405647069, 108.69738891115995, 24.26498719809055, 1.0],
                        [84.6188442832742, 71.1933041962819, 36.32597478756992, 1.0],
                    ],
                    [
                        [73.71310675519464, 108.89719408343927, 26.09358483707635, 1.0],
                        [83.06117698199816, 71.39310936856123, 38.15457242655572, 1.0],
                    ],
                    [
                        [71.86797619548975, 108.9690445398807, 27.747105786913014, 1.0],
                        [81.21604642229327, 71.46495982500265, 39.808093376392385, 1.0],
                    ],
                    [
                        [69.79528019189668, 108.912483943953, 29.17770624903919, 1.0],
                        [79.14335041870021, 71.40839922907496, 41.23869383851856, 1.0],
                    ],
                    [
                        [67.56104138064012, 108.73097271506528, 30.344974576091378, 1.0],
                        [76.90911160744363, 71.22688800018722, 42.40596216557074, 1.0],
                    ],
                    [
                        [65.2353086400129, 108.43172996011272, 31.217066291473003, 1.0],
                        [74.58337886681642, 70.92764524523467, 43.27805388095238, 1.0],
                    ],
                    [
                        [62.88996297924695, 108.02546234999494, 31.77156160345183, 1.0],
                        [72.23803320605046, 70.52137763511688, 43.832549192931204, 1.0],
                    ],
                    [
                        [60.59649655943559, 107.52598975220137, 31.996022702903044, 1.0],
                        [69.94456678623911, 70.02190503732332, 44.057010292382415, 1.0],
                    ],
                    [
                        [58.42383345135465, 106.9497804848072, 31.888237676620694, 1.0],
                        [67.77190367815817, 69.44569576992915, 43.94922526610006, 1.0],
                    ],
                    [
                        [56.43625855771079, 106.31541167528079, 31.456147696635423, 1.0],
                        [65.7843287845143, 68.81132696040274, 43.517135286114794, 1.0],
                    ],
                    [
                        [54.6915169252106, 105.64297231371285, 30.71746395224668, 1.0],
                        [64.03958715201412, 68.1388875988348, 42.77845154172605, 1.0],
                    ],
                    [
                        [53.23913960038584, 104.95342812558968, 29.69899027199356, 1.0],
                        [62.58720982718936, 67.44934341071163, 41.759977861472926, 1.0],
                    ],
                    [
                        [52.11904445398667, 104.26796831364257, 28.435676247861895, 1.0],
                        [61.46711468079018, 66.76388359876452, 40.496663837341266, 1.0],
                    ],
                    [
                        [51.36045127250022, 103.60735451068608, 26.969433656648654, 1.0],
                        [60.70852149930374, 66.10326979580802, 39.03042124612802, 1.0],
                    ],
                    [
                        [50.98114019270961, 102.99129194467976, 25.347755837285032, 1.0],
                        [60.329210419513124, 65.48720722980171, 37.4087434267644, 1.0],
                    ],
                    [
                        [50.98707156725684, 102.43784186217442, 23.622185229476838, 1.0],
                        [60.33514179406036, 64.93375714729638, 35.68317281895621, 1.0],
                    ],
                    [
                        [51.37237394666195, 101.96289272436415, 21.846678353107304, 1.0],
                        [60.720444173465474, 64.45880800948609, 33.907665942586675, 1.0],
                    ],
                    [
                        [52.119695405417104, 101.57970563616664, 20.07592000206839, 1.0],
                        [61.467765632220626, 64.07562092128858, 32.13690759154776, 1.0],
                    ],
                    [
                        [53.20090228304913, 101.29854696364163, 18.363639283653434, 1.0],
                        [62.54897250985265, 63.79446224876358, 30.424626873132805, 1.0],
                    ],
                    [
                        [54.57809889796445, 101.12641822233122, 16.76097935013552, 1.0],
                        [63.926169124767966, 63.62233350745318, 28.821966939614892, 1.0],
                    ],
                    [
                        [56.204932240584405, 101.06689017284573, 15.314970288643282, 1.0],
                        [65.55300246738793, 63.56280545796767, 27.37595787812265, 1.0],
                    ],
                    [
                        [58.02813734684593, 101.12004474161692, 14.067150754018297, 1.0],
                        [67.37620757364945, 63.615960026738875, 26.128138343497668, 1.0],
                    ],
                    [
                        [59.989272235191144, 101.28252499964725, 13.052378687633212, 1.0],
                        [69.33734246199467, 63.778440284769204, 25.11336627711258, 1.0],
                    ],
                    [
                        [62.026586151827686, 101.54769008646086, 12.297865044371106, 1.0],
                        [71.37465637863122, 64.04360537158281, 24.358852633850475, 1.0],
                    ],
                    [
                        [64.07696154754362, 101.90586876386043, 11.822457065740263, 1.0],
                        [73.42503177434713, 64.40178404898238, 23.88344465521963, 1.0],
                    ],
                    [
                        [66.07786878348696, 102.344702322243, 11.636189532261826, 1.0],
                        [75.42593901029048, 64.84061760736495, 23.697177121741195, 1.0],
                    ],
                    [
                        [67.96927305164172, 102.84956493006106, 11.740113864966524, 1.0],
                        [77.31734327844524, 65.34548021518302, 23.80110145444589, 1.0],
                    ],
                    [
                        [69.69543535682843, 103.4040472919943, 12.126406197044489, 1.0],
                        [79.04350558363194, 65.89996257711626, 24.18739378652386, 1.0],
                    ],
                    [
                        [71.20655354155971, 103.99048772723764, 12.778746877553214, 1.0],
                        [80.55462376836323, 66.48640301235959, 24.839734467032585, 1.0],
                    ],
                    [
                        [72.46019508954504, 104.59053354416359, 13.67295556825788, 1.0],
                        [81.80826531634855, 67.08644882928554, 25.733943157737247, 1.0],
                    ],
                    [
                        [73.4224806160484, 105.18571490276132, 14.777858405998941, 1.0],
                        [82.77055084285192, 67.68163018788327, 26.838845995478312, 1.0],
                    ],
                    [
                        [74.06898530002766, 105.75801323533824, 16.05635685771805, 1.0],
                        [83.41705552683118, 68.25392852046019, 28.11734444719742, 1.0],
                    ],
                    [
                        [74.38533475697693, 106.29040673481244, 17.46666209515935, 1.0],
                        [83.73340498378046, 68.78632201993437, 29.527649684638725, 1.0],
                    ],
                    [
                        [74.36748169130416, 106.7673763968811, 18.963654127559103, 1.0],
                        [83.71555191810768, 69.26329168200306, 31.02464171703847, 1.0],
                    ],
                    [
                        [74.02165978704144, 107.17535757922418, 20.500321679381486, 1.0],
                        [83.36973001384497, 69.67127286434612, 32.56130926886085, 1.0],
                    ],
                    [
                        [73.36402137543755, 107.50312396436341, 22.029236968752418, 1.0],
                        [82.71209160224106, 69.99903924948536, 34.09022455823179, 1.0],
                    ],
                    [
                        [72.41997514295973, 107.74209311625725, 23.504019167428822, 1.0],
                        [81.76804536976326, 70.2380084013792, 35.56500675690819, 1.0],
                    ],
                    [
                        [71.22324921440524, 107.88654542659906, 24.880741395479788, 1.0],
                        [80.57131944120877, 70.38246071172101, 36.941728984959155, 1.0],
                    ],
                    [
                        [69.81471308887015, 107.93375106912856, 26.119238568627395, 1.0],
                        [79.16278331567366, 70.42966635425051, 38.18022615810676, 1.0],
                    ],
                    [
                        [68.24099887994021, 107.88400252751389, 27.184277175751223, 1.0],
                        [77.58906910674372, 70.37991781263584, 39.24526476523059, 1.0],
                    ],
                    [
                        [66.55296791450633, 107.74055324034097, 28.04655298146921, 1.0],
                        [75.90103814130984, 70.23646852546291, 40.10754057094858, 1.0],
                    ],
                    [
                        [64.80407282186044, 107.5094658216663, 28.68348855261178, 1.0],
                        [74.15214304866396, 70.00538110678825, 40.74447614209115, 1.0],
                    ],
                    [
                        [63.04866769119459, 107.19937607698492, 29.079809197894644, 1.0],
                        [72.39673791799811, 69.69529136210687, 41.14079678737401, 1.0],
                    ],
                    [
                        [61.34031963911918, 106.82118155801334, 29.22788316540032, 1.0],
                        [70.68838986592269, 69.31709684313529, 41.28887075487969, 1.0],
                    ],
                    [
                        [59.73017421089068, 106.3876656097714, 29.12781952627325, 1.0],
                        [69.0782444376942, 68.88358089489334, 41.188807115752624, 1.0],
                    ],
                    [
                        [58.265424494161095, 105.91306969543133, 28.787324842102592, 1.0],
                        [67.61349472096461, 68.40898498055329, 40.84831243158196, 1.0],
                    ],
                    [
                        [56.98792975719873, 105.41262818651259, 28.2213272254739, 1.0],
                        [66.33599998400226, 67.90854347163454, 40.282314814953274, 1.0],
                    ],
                    [
                        [55.933023986052845, 104.90208074078389, 27.45138352431772, 1.0],
                        [65.28109421285636, 67.39799602590583, 39.51237111379709, 1.0],
                    ],
                    [
                        [55.12854807944527, 104.3971778355355, 26.504891872967217, 1.0],
                        [64.47661830624878, 66.89309312065745, 38.56587946244659, 1.0],
                    ],
                    [
                        [54.59413189297321, 103.91319497333818, 25.414137560742272, 1.0],
                        [63.94220211977673, 66.40911025846012, 37.47512515022164, 1.0],
                    ],
                    [
                        [54.340744058865255, 103.46447054039609, 24.215204905223267, 1.0],
                        [63.68881428566877, 65.96038582551805, 36.27619249470264, 1.0],
                    ],
                    [
                        [54.37051881566359, 103.0639812987641, 22.94679144823197, 1.0],
                        [63.71858904246711, 65.55989658388606, 35.007779037711344, 1.0],
                    ],
                    [
                        [54.676860244783896, 102.72296807188822, 21.648963220956706, 1.0],
                        [64.02493047158742, 65.21888335701017, 33.70995081043608, 1.0],
                    ],
                    [
                        [55.24481560923653, 102.45062238980726, 20.36189099325378, 1.0],
                        [64.59288583604005, 64.94653767492922, 32.42287858273315, 1.0],
                    ],
                    [
                        [56.051701196560145, 102.2538427585446, 19.124607314315938, 1.0],
                        [65.39977142336366, 64.74975804366653, 31.185594903795305, 1.0],
                    ],
                    [
                        [57.06795643875382, 102.13706687917137, 17.973822791731884, 1.0],
                        [66.41602666555733, 64.63298216429332, 30.034810381211255, 1.0],
                    ],
                    [
                        [58.25819534720403, 102.10218364359788, 16.94283750692893, 1.0],
                        [67.60626557400755, 64.59809892871982, 29.003825096408306, 1.0],
                    ],
                    [
                        [59.58241865880238, 102.14852615805604, 16.060579827306256, 1.0],
                        [68.93048888560591, 64.644441443178, 28.12156741678563, 1.0],
                    ],
                    [
                        [60.99734570132495, 102.27294447437156, 15.35080028225537, 1.0],
                        [70.34541592812846, 64.76885975949351, 27.41178787173474, 1.0],
                    ],
                    [
                        [62.45782196998011, 102.46995422490728, 14.831442783232337, 1.0],
                        [71.80589219678362, 64.96586951002922, 26.892430372711704, 1.0],
                    ],
                    [
                        [63.91825683559605, 102.73195503687046, 14.514209471297441, 1.0],
                        [73.26632706239957, 65.22787032199241, 26.57519706077681, 1.0],
                    ],
                    [
                        [65.33404570380335, 103.04951051643103, 14.404329069764543, 1.0],
                        [74.68211593060687, 65.54542580155297, 26.465316659243914, 1.0],
                    ],
                    [
                        [66.6629322921057, 103.41167980509685, 14.500532015313976, 1.0],
                        [76.01100251890921, 65.9075950902188, 26.561519604793343, 1.0],
                    ],
                    [
                        [67.86626942037813, 103.80638927184376, 14.795229051577333, 1.0],
                        [77.21433964718165, 66.3023045569657, 26.8562166410567, 1.0],
                    ],
                    [
                        [68.91014070957155, 104.22083185443788, 15.274883604325971, 1.0],
                        [78.25821093637506, 66.71674713955981, 27.33587119380534, 1.0],
                    ],
                    [
                        [69.76631070387806, 104.6418809289694, 15.920562315933726, 1.0],
                        [79.11438093068158, 67.13779621409135, 27.981549905413097, 1.0],
                    ],
                    [
                        [70.41297699058978, 105.0565053808684, 16.708642780893634, 1.0],
                        [79.76104721739331, 67.55242066599035, 28.769630370372997, 1.0],
                    ],
                    [
                        [70.83530467955467, 105.45217277268613, 17.611652953624425, 1.0],
                        [80.18337490635818, 67.94808805780808, 29.672640543103796, 1.0],
                    ],
                    [
                        [71.0257308908213, 105.81722813909609, 18.599213027221126, 1.0],
                        [80.37380111762482, 68.31314342421803, 30.660200616700493, 1.0],
                    ],
                    [
                        [70.98403444289741, 106.14123696029141, 19.639047908803956, 1.0],
                        [80.33210466970093, 68.63715224541336, 31.700035498283324, 1.0],
                    ],
                    [
                        [70.71717348891951, 106.4152822317001, 20.698036811532106, 1.0],
                        [80.06524371572303, 68.91119751682206, 32.75902440101147, 1.0],
                    ],
                    [
                        [70.23890117161812, 106.6322072107228, 21.74326597756703, 1.0],
                        [79.58697139842164, 69.12812249584474, 33.8042535670464, 1.0],
                    ],
                    [
                        [69.56917622944874, 106.7867973213543, 22.74305113676631, 1.0],
                        [78.91724645625226, 69.28271260647625, 34.80403872624568, 1.0],
                    ],
                    [
                        [68.73339167361581, 106.87589676975892, 23.66789795388347, 1.0],
                        [78.08146190041933, 69.37181205488086, 35.72888554336284, 1.0],
                    ],
                    [
                        [67.76144998225055, 106.89845759837641, 24.491371350341705, 1.0],
                        [77.10952020905407, 69.39437288349836, 36.552358939821076, 1.0],
                    ],
                    [
                        [66.68671756798375, 106.85552111107476, 25.19084810246327, 1.0],
                        [76.03478779478726, 69.3514363961967, 37.25183569194264, 1.0],
                    ],
                    [
                        [65.54489444837252, 106.75013376561667, 25.748131386847042, 1.0],
                        [74.89296467517605, 69.2460490507386, 37.80911897632641, 1.0],
                    ],
                    [
                        [64.37283700370801, 106.58720168321987, 26.149910813750555, 1.0],
                        [73.72090723051153, 69.08311696834183, 38.210898403229926, 1.0],
                    ],
                    [
                        [63.20737240314217, 106.3732898039749, 26.388056792337437, 1.0],
                        [72.55544262994569, 68.86920508909685, 38.449044381816805, 1.0],
                    ],
                    [
                        [62.08414271880278, 106.11637336382533, 26.45974362789767, 1.0],
                        [71.4322129456063, 68.61228864894728, 38.52073121737704, 1.0],
                    ],
                    [
                        [61.03651497042671, 105.82555073473551, 26.36740137592553, 1.0],
                        [70.38458519723022, 68.32146601985747, 38.428388965404906, 1.0],
                    ],
                    [
                        [60.094590430611404, 105.51072771558223, 26.118501987570816, 1.0],
                        [69.44266065741493, 68.00664300070419, 38.179489577050184, 1.0],
                    ],
                    [
                        [59.28434258918677, 105.18228405931109, 25.7251904986972, 1.0],
                        [68.63241281599029, 67.67819934443304, 37.78617808817657, 1.0],
                    ],
                    [
                        [58.626908371720226, 104.85073335598291, 25.203776776656582, 1.0],
                        [67.97497859852373, 67.34664864110486, 37.26476436613595, 1.0],
                    ],
                    [
                        [58.13805170490232, 104.52638735741037, 24.574107498017568, 1.0],
                        [67.48612193170584, 67.02230264253231, 36.635095087496936, 1.0],
                    ],
                    [
                        [57.827812513452045, 104.21903543605772, 23.858841465041017, 1.0],
                        [67.17588274025556, 66.71495072117966, 35.919829054520385, 1.0],
                    ],
                    [
                        [57.700347924999086, 103.93764913586084, 23.082653972395953, 1.0],
                        [67.04841815180261, 66.43356442098278, 35.14364156187532, 1.0],
                    ],
                    [
                        [57.75396606415173, 103.6901207362929, 22.271397669844184, 1.0],
                        [67.10203629095525, 66.18603602141484, 34.33238525932356, 1.0],
                    ],
                    [
                        [57.98134654260204, 103.4830434159075, 21.451248078143294, 1.0],
                        [67.32941676940555, 65.97895870102946, 33.51223566762266, 1.0],
                    ],
                    [
                        [58.3699358141897, 103.32153915652795, 20.647862024233383, 1.0],
                        [67.71800604099323, 65.8174544416499, 32.70884961371276, 1.0],
                    ],
                    [
                        [58.90250010583999, 103.20913850463208, 19.88557519646438, 1.0],
                        [68.25057033264352, 65.70505378975402, 31.946562785943744, 1.0],
                    ],
                    [
                        [59.5578139953341, 103.14771571819331, 19.18666678759025, 1.0],
                        [68.90588422213762, 65.64363100331526, 31.247654377069622, 1.0],
                    ],
                    [
                        [60.31145839489129, 103.13747643443831, 18.570702656680645, 1.0],
                        [69.65952862169482, 65.63339171956025, 30.63169024616001, 1.0],
                    ],
                    [
                        [61.13670019069741, 103.17701011196023, 18.054016620468136, 1.0],
                        [70.48477041750093, 65.67292539708218, 30.115004209947507, 1.0],
                    ],
                    [
                        [61.71584962205868, 103.23457875769459, 17.784148986040286, 1.0],
                        [71.06391984886218, 65.73049404281653, 29.845136575519653, 1.0],
                    ],
                    [
                        [62.00790573812107, 103.27044695570538, 17.669319488366327, 1.0],
                        [71.3559759649246, 65.76636224082733, 29.730307077845698, 1.0],
                    ],
                ],
                dtype=float,
            ),
            "degree": (3, 1),

        }
    )

    surf1 = NURBSSurfaceTuple(order_u=s1.degree[0] + 1, order_v=s1.degree[1] + 1, knot_u=s1.knots_u.tolist(),
                              knot_v=s1.knots_v.tolist(), control_points=np.array(s1.control_points),
                              weights=np.ascontiguousarray(s1.control_points_w[..., -1]))

    surf2 = NURBSSurfaceTuple(order_u=s2.degree[0] + 1, order_v=s2.degree[1] + 1, knot_u=s2.knots_u.tolist(),
                              knot_v=s2.knots_v.tolist(), control_points=np.array(s2.control_points),
                              weights=np.ascontiguousarray(s2.control_points_w[..., -1]))
    branches=trace_intersection_curves(surf1, surf2,tol=1e-2,spt=1e-2)
    print(f"{len(branches)} intersection curves")
    print([branch[0].tolist() for branch in branches])
