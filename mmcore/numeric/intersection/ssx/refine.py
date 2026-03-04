from __future__ import annotations

from typing import Union

import numpy as np

from mmcore.geom._nurbs_eval import NURBSSurfaceTuple, evaluate_nurbs_surface


def normal_angle_gap(n1, n2):
    """
    Compute sin(theta) between normals n1 and n2:
        sin θ = ||n1 × n2|| / (||n1|| · ||n2||)
    """
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    num = np.linalg.norm(np.cross(n1, n2))
    den = np.linalg.norm(n1) * np.linalg.norm(n2)
    return num / den


def normal_distance_gap(n, S1, S2):
    """
    Compute the scalar projection of the residual (S1 - S2) on normal n:
        |n · (S1 - S2)| / ||n||
    If n is already unit-length you can omit the division by ||n||.
    """

    diff = S1- S2
    return abs(np.dot(n, diff))


def within_normal_gap(n1, n2, S1, S2, eps_theta, eps_n):
    """
    Check whether either gap metrics is below its threshold:
      sin θ ≤ eps_theta
      OR
      |n1·(S1−S2)| ≤ eps_n
    """
    if normal_angle_gap(n1, n2) <= eps_theta:
        return True
    if normal_distance_gap(n1, S1, S2) <= eps_n:
        return True
    return False


def calculate_eps_n(spt, angle_tol):
    return (spt**2)/(angle_tol+10e-12)


def refine_intersection_point(x: np.ndarray, surf1: NURBSSurfaceTuple, surf2: NURBSSurfaceTuple, spt: float = 1e-3, eps_n=None,angle_tol=0.052,max_iter: int = 10,full_outp=False) -> Union[tuple[np.ndarray,dict,dict,float], tuple[bool,np.ndarray,dict,dict,float]]:
    """
    Refines the intersection point of two NURBS surfaces to a higher accuracy using an
    iterative approach. The function computes the intersection refinement by minimizing
    the distance between the evaluated points on the two surfaces while considering normal
    vector alignment, ensuring the refinement achieves geometric consistency and convergence
    within a specified tolerance.

    :param x: Initial guess for the intersection parameter vector, in the form [s, t, u, v].
    :type x: numpy.ndarray
    :param surf1: The first NURBS surface to be used in the intersection refinement.
    :type surf1: NURBSSurfaceTuple
    :param surf2: The second NURBS surface to be used in the intersection refinement.
    :type surf2: NURBSSurfaceTuple
    :param spt: Convergence tolerance for geometric proximity between the surfaces.
    :type spt: float
    :param eps_n: Tolerance for normal vector alignment. If None, it will be computed
                  based on `spt` and `angle_tol`.
    :type eps_n: float or None
    :param angle_tol: Angular tolerance for the alignment of surface normal vectors,
                      given in radians.
    :type angle_tol: float
    :param max_iter: Maximum number of iterations allowed for refining the intersection.
    :type max_iter: int
    :return: A tuple containing:
             - Refined parameter vector `x` ([s, t, u, v]) as a numpy array.
             - Evaluation results for the first surface as a dictionary.
             - Evaluation results for the second surface as a dictionary.
             - Final error metric between the surfaces after refinement.
    :rtype: tuple[numpy.ndarray, dict, dict, float]
    """
    iteration = 0
    x_current = np.array(x, dtype=float)
    p_eval, q_eval=dict(),dict()
    error=-1
    success=False
    if eps_n is None:
        eps_n=calculate_eps_n(spt,angle_tol)

    while iteration < max_iter:
        s, t, u, v = x_current

        # Evaluate surfaces at first derivative level for Jacobian computation.
        p_eval = evaluate_nurbs_surface(surf1, s, t, d_order=1)
        q_eval = evaluate_nurbs_surface(surf2, u, v, d_order=1)
        #print(p_eval,q_eval)
        S0 = np.array(p_eval["S"])
        S1 = np.array(q_eval["S"])
        error = np.linalg.norm(S0 - S1)
        n1=np.cross(p_eval["Su"], p_eval["Sv"])
        n1/=np.linalg.norm(n1)

        n2 = np.cross(q_eval["Su"], q_eval["Sv"])
        n2/=np.linalg.norm(n2)
        # Check convergence.
        if (error<spt) and within_normal_gap(n1,n2, p_eval['S'],q_eval['S'], angle_tol,eps_n) :
            success=True

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
    if full_outp:
        return success,x_current, p_eval, q_eval,error
    return x_current, p_eval, q_eval,error
