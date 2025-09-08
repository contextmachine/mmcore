from __future__ import annotations

import numpy as np


def segment_intersection(p1, p2, q1, q2, eps=1e-5):
    """
    Compute intersection (or overlap) of two line segments [p1→p2] and [q1→q2] in 2D or 3D.
    Returns:
      - (t, u):       parameters if they intersect at a unique point (0 <= t,u <= 1)
      - [(t1,u1),(t2,u2)]:  parameters at the start and end of overlap if colinear and overlapping
      - None:         if no intersection
    """
    dim = len(p1)
    if dim not in (2, 3):
        raise ValueError("Only 2D or 3D points are supported")
    # Convert to floats
    P = np.array(p1,float)
    R =  np.array(p2,float)
    Q = np.array(q1,float)
    S2 =np.array(q2,float)
    # Direction vectors and connector

    d1 = R- P
    d2 = S2 - Q
    C  = Q - P
    nc=np.linalg.norm(C)
    nd1=np.linalg.norm(d1)
    nd2 = np.linalg.norm(d1)
    if np.isclose(nd1,0):
        ud1=np.zeros(dim)
    else:
        # Dot product helper
        ud1=d1/nd1
    if np.isclose(nd2, 0):
        ud2 = np.zeros(dim)
    else:
        # Dot product helper
        ud2 = d2 / nd2
    if np.isclose(nc, 0):
        uC = np.zeros(dim)
    else:
        # Dot product helper
        uC = C / nc
   
   
    # Check for degenerate segments
    denom1 = np.dot(d1, d1)
    denom2 = np.dot(d2, d2)

    if nd1 <= 0.  or nd2 <= 0. :
        return None
    # 2D case
    if dim == 2:
        # Compute determinant of 2x2 system

        det = d1[0] * (-d2[1]) - (-d2[0]) * d1[1]
        if np.abs(det) > eps:

            # Unique intersection
            t = (C[0] * (-d2[1]) - (-d2[0]) * C[1]) / det
            u = (d1[0] * C[1] - C[0] * d1[1]) / det

            if (-eps) <= t <= (1. + eps) and (-eps) <= u <= (1. + eps):
                return (t, u)


            return None
        # Parallel: check colinearity
        if np.abs(ud1[0] * uC[1] - ud1[1] * uC[0]) > eps:
            
            return None
        # Colinear: compute overlap in t-domain
        t0 = np.dot(C, d1) / denom1
        C2_vec = [S2[i] - P[i] for i in range(dim)]
        t1 = np.dot(C2_vec, d1) / denom1
        t_min, t_max = min(t0, t1), max(t0, t1)
        t_start, t_end = max(0., t_min), min(1., t_max)
        if t_start > (t_end + eps):


            return None
        # Compute corresponding points and u-parameters
        I_start = P+d1*t_start
        I_end   = P + d1 * t_end
        u_start = np.dot(I_start - Q, d2) / denom2
        u_end   = np.dot(I_end  - Q , d2) / denom2
        return [(t_start, u_start), (t_end, u_end)]
    # 3D case
    # Cross product helper

    cp = np.cross(d1, d2)
    cp_norm_sq = np.dot(cp, cp)
    # Parallel or colinear if cross-product ~ 0
    if cp_norm_sq < eps:
        # Check colinearity
        cross_C_d1 = np.cross(C, d1)
        if np.dot(cross_C_d1, cross_C_d1) > eps:
            return None
        # Colinear: compute overlap exactly as in 2D
        t0 = np.dot(C, d1) / denom1
        C2_vec = [S2[i] - P[i] for i in range(dim)]
        t1 = np.dot(C2_vec, d1) / denom1
        t_min, t_max = min(t0, t1), max(t0, t1)
        t_start, t_end = max(0, t_min), min(1, t_max)
        if t_start > (t_end + eps):

            return None
        I_start = [P[i] + d1[i] * t_start for i in range(dim)]
        I_end   = [P[i] + d1[i] * t_end   for i in range(dim)]
        u_start = np.dot([I_start[i] - Q[i] for i in range(dim)], d2) / denom2
        u_end   = np.dot([I_end[i]   - Q[i] for i in range(dim)], d2) / denom2
        return [(t_start, u_start), (t_end, u_end)]
    # Non-parallel: check coplanarity
    if abs(np.dot(C, cp)) > eps:
        return None
    # Solve by projecting to the largest cross-product axis
    idx = max(range(3), key=lambda i: abs(cp[i]))
    i1, i2 = [i for i in range(3) if i != idx]
    a, b = d1[i1], -d2[i1]
    c, d = d1[i2], -d2[i2]
    det = a * d - b * c
    if abs(det) < eps:
        return None
    C1, C2p = C[i1], C[i2]
    t = (C1 * d - b * C2p) / det
    u = (a * C2p - C1 * c) / det
    if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
        return (t, u)
    return None
