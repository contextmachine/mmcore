from enum import Enum, auto

import numpy as np
from math import comb
import numpy as np

from mmcore.geom._nurbs_knots import split_curve_multiple,split_curve
from mmcore.numeric.aabb import aabb,aabb_intersect


def origin_in_convex_hull(points: np.ndarray, tol: float = 1e-12, return_witness: bool = False):
    """
    Return True if the convex hull of `points` contains the origin (0,0), else False.
    Optionally also return a separating direction `u` (unit vector) when the origin is outside.

    Parameters
    ----------
    points : (N, 2) array_like
        2D points (floats are fine; integers are cast to float).
    tol : float, optional
        Numerical tolerance around zero and π comparisons. Equality to π counts as "inside".
    return_witness : bool, optional
        If True and the origin is outside, also return a unit vector `u` such that u·p > 0 for all p.

    Returns
    -------
    inside : bool
        True  -> origin is in the convex hull (including boundary).
        False -> origin is outside the convex hull.
    u : np.ndarray or None
        If `return_witness` is True and `inside` is False, a 2-vector giving a separating direction.
        Otherwise None.

    Notes
    -----
    By the strict separation theorem, 0 ∉ conv(P) ⇔ ∃ u with u·p_i > 0 ∀i.
    That happens iff all point directions fit in an open semicircle (< π).
    We detect this by sorting angles and checking the maximum circular gap.
    Complexity: O(N log N) time, O(N) memory (no convex hull construction).
    """
    P = np.asarray(points, dtype=float)
    
    # Handle trivial cases
    if P.size == 0:
        return (False, None) if return_witness else False
    
    # If any point is (near) the origin, the origin is in the convex hull
    if np.any(np.all(np.abs(P) <= tol, axis=1)):
        return (True, None) if return_witness else True
    
    # Drop exact zeros just in case
    nz = np.linalg.norm(P, axis=1) > tol
    if not np.any(nz):
        return (True, None) if return_witness else True
    
    # Angles in [0, 2π)
    ang = np.mod(np.arctan2(P[nz, 1], P[nz, 0]), 2 * np.pi)
    ang.sort()
    
    # Circular gaps between consecutive angles (including wrap-around)
    gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    max_gap_idx = np.argmax(gaps)
    max_gap = gaps[max_gap_idx] if gaps.size else 2 * np.pi
    
    # If there's an empty arc strictly larger than π, all points lie in an open semicircle -> outside
    outside = (max_gap > np.pi + tol)
    
    if not return_witness:
        return not outside
    
    if outside:
        # The points all lie in the complement arc of length (2π - max_gap) < π.
        # A separating direction u is the center angle of that complement arc.
        # The complement arc runs from ang[max_gap_idx+1] to ang[max_gap_idx]+2π.
        left = ang[(max_gap_idx + 1) % len(ang)]
        complement_len = 2 * np.pi - max_gap
        alpha = left + complement_len / 2.0  # center of the arc containing all points
        u = np.array([np.cos(alpha), np.sin(alpha)])
        return False, u / np.linalg.norm(u)
    else:
        return True, None


def _bernstein_product_conv(deg: int) -> np.ndarray:
    """
    Build the exact Bernstein product convolution tensor:
      B_i^deg(t) * B_j^deg(t) = sum_{r=0}^{2deg} Conv[r,i,j] * B_r^{2deg}(t)
    where Conv[r,i,j] = C(deg,i) C(deg,j) / C(2deg, r) if r == i+j else 0.
    Shape: (2deg+1, deg+1, deg+1)
    """
    m = deg
    C2m = np.array([comb(2 * m, r) for r in range(2 * m + 1)], dtype=np.float64)
    Ci = np.array([comb(m, i) for i in range(m + 1)], dtype=np.float64)
    Conv = np.zeros((2 * m + 1, m + 1, m + 1), dtype=np.float64)
    # Fill only entries with r == i+j
    for i in range(m + 1):
        for j in range(m + 1):
            r = i + j
            Conv[r, i, j] = (Ci[i] * Ci[j]) / C2m[r]
    return Conv


def bernstein_distance_squared_net(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Given two Bézier curves in Bernstein form:
        C1(u) = sum_{i=0..p} P[i] * B_i^p(u),  P.shape == (p+1, d)
        C2(v) = sum_{j=0..q} Q[j] * B_j^q(v),  Q.shape == (q+1, d)
    return the bivariate Bernstein control net F of:
        f(u,v) = || C1(u) - C2(v) ||^2
    so that
        f(u,v) = sum_{r=0..2p} sum_{s=0..2q} F[r, s] * B_r^{2p}(u) * B_s^{2q}(v)

    Parameters
    ----------
    P : (p+1, d) float64
    Q : (q+1, d) float64

    Returns
    -------
    F : (2p+1, 2q+1) float64
        Bivariate control net of f(u,v).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.ndim != 2 or Q.ndim != 2:
        raise ValueError("P and Q must be 2D arrays of shapes (p+1,d) and (q+1,d).")
    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same ambient dimension d.")
    p = P.shape[0] - 1
    q = Q.shape[0] - 1
    d = P.shape[1]
    
    # Difference control net D_{i,j,:} = P_i - Q_j (tensor-product net)
    # Shape: (p+1, q+1, d)
    D = P[:, None, :] - Q[None, :, :]
    
    # Exact Bernstein product convolution tensors in u and v
    ConvU = _bernstein_product_conv(p)  # (2p+1, p+1, p+1)
    ConvV = _bernstein_product_conv(q)  # (2q+1, q+1, q+1)
    
    # For efficient contractions, flatten the (i,i') and (j,j') pairs:
    ConvU2 = ConvU.reshape(2 * p + 1, (p + 1) * (p + 1))  # (2p+1, (p+1)^2)
    ConvV2 = ConvV.reshape(2 * q + 1, (q + 1) * (q + 1))  # (2q+1, (q+1)^2)
    
    # First convolve along u (degree p -> 2p), but sum over Cartesian dims right away.
    # For each (j, j'), build A = sum_k D[:,j,k] * D[:,j',k]^T  (shape (p+1, p+1)),
    # then Tu[:, j, j'] = ConvU2 @ A.ravel()
    Tu = np.zeros((2 * p + 1, q + 1, q + 1), dtype=np.float64)
    Dj = D.transpose(1, 0, 2)  # (q+1, p+1, d) to help locality
    
    for j in range(q + 1):
        # (p+1, d)
        Dj_mat = Dj[j]  # rows indexed by i, columns by k
        for jp in range(q + 1):
            Djp_mat = Dj[jp]  # (p+1, d)
            # A[i,i'] = sum_k D[i,j,k] * D[i',j',k]
            A = Dj_mat @ Djp_mat.T  # (p+1, p+1)
            Tu[:, j, jp] = ConvU2 @ A.ravel()
    
    # Then convolve along v (degree q -> 2q) for each r, again exact.
    F = np.zeros((2 * p + 1, 2 * q + 1), dtype=np.float64)
    for r in range(2 * p + 1):
        B = Tu[r, :, :]  # (q+1, q+1)
        F[r, :] = ConvV2 @ B.ravel()
    
    return F


import numpy as np
from mmcore.geom._nurbs_eval import NURBSCurveTuple, evaluate_nurbs_surface, evaluate_nurbs_curve, to_homogeneous_2d
from ._bez_overlap import _bez_curve_overlap, CCXInt
from mmcore.geom._nurbs_knots import subdivide_surface, trim_curve

from mmcore.numeric.sbern import bern_to_nurbs_bezier
from mmcore.numeric.gauss_map import  compute_gauss_map_rational, compute_gauss_map

from mmcore.numeric.newton.cnewton import newtons_method
class BezIntType(int,Enum):
    OVERLAP=auto()
    ISOLATED=auto()

import numpy as np
from math import comb

def _bernstein_product_conv(deg: int) -> np.ndarray:
    """
    Exact Bernstein product convolution tensor for equal degrees:
      B_i^deg(t) * B_j^deg(t) = Conv[r,i,j] * B_r^{2deg}(t), with r = i + j
      Conv[r,i,j] = C(deg,i) C(deg,j) / C(2deg, r)
    Returns array with shape (2*deg+1, deg+1, deg+1).
    """
    m = deg
    C2m = np.array([comb(2*m, r) for r in range(2*m+1)], dtype=np.float64)
    Ci  = np.array([comb(m, i)    for i in range(m+1)],  dtype=np.float64)
    Conv = np.zeros((2*m+1, m+1, m+1), dtype=np.float64)
    for i in range(m+1):
        for j in range(m+1):
            r = i + j
            Conv[r, i, j] = (Ci[i] * Ci[j]) / C2m[r]
    return Conv

def _bivariate_squared_norm_net(E: np.ndarray) -> np.ndarray:
    """
    Given a bivariate Bernstein vector control net E of degree (p,q):
        E.shape == (p+1, q+1, d)
    compute the bivariate control net F of f(u,v) = ||E(u,v)||^2 of degree (2p, 2q):
        F.shape == (2p+1, 2q+1)
    """
    if E.ndim != 3:
        raise ValueError("E must have shape (p+1, q+1, d).")
    p = E.shape[0] - 1
    q = E.shape[1] - 1

    ConvU = _bernstein_product_conv(p)   # (2p+1, p+1, p+1)
    ConvV = _bernstein_product_conv(q)   # (2q+1, q+1, q+1)
    ConvU2 = ConvU.reshape(2*p + 1, (p + 1) * (p + 1))
    ConvV2 = ConvV.reshape(2*q + 1, (q + 1) * (q + 1))

    # Convolve along u: for each (j,j'), A = sum_k E[:,j,k] * E[:,j',k]^T, then apply ConvU
    Tu = np.zeros((2*p + 1, q + 1, q + 1), dtype=np.float64)
    Dj = E.transpose(1, 0, 2)  # (q+1, p+1, d)
    for j in range(q + 1):
        Dj_mat = Dj[j]  # (p+1, d)
        for jp in range(q + 1):
            Djp_mat = Dj[jp]  # (p+1, d)
            A = Dj_mat @ Djp_mat.T  # (p+1, p+1) = sum_k E[i,j,k] E[i',j',k]
            Tu[:, j, jp] = ConvU2 @ A.ravel()

    # Convolve along v
    F = np.zeros((2*p + 1, 2*q + 1), dtype=np.float64)
    for r in range(2 * p + 1):
        B = Tu[r, :, :]  # (q+1, q+1)
        F[r, :] = ConvV2 @ B.ravel()

    return F

def _square_univariate_bernstein_net(w: np.ndarray) -> np.ndarray:
    """
    Given univariate Bernstein control points w (degree p), return control points
    of w(t)^2 in Bernstein form of degree 2p. Shape -> (2p+1,).
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    p = w.shape[0] - 1
    Conv = _bernstein_product_conv(p)  # (2p+1, p+1, p+1)
    # Contraction: sum_{i,i'} Conv[r,i,i'] * w[i] * w[i']
    return np.tensordot(Conv, np.outer(w, w), axes=([1, 2], [0, 1]))

def bernstein_rational_distance_squared_nets(P1: np.ndarray, W1: np.ndarray,
                                             P2: np.ndarray, W2: np.ndarray):
    """
    Compute bivariate Bernstein control nets (G, H) for the squared distance between two rational Bézier curves:
        C1(u) = (Σ_i w1[i] P1[i] B_i^p(u)) / (Σ_i w1[i] B_i^p(u))
        C2(v) = (Σ_j w2[j] P2[j] B_j^q(v)) / (Σ_j w2[j] B_j^q(v))

    Returns G, H such that:
        ||C1(u) - C2(v)||^2 = g(u,v)/h(u,v),
      with
        g(u,v) = Σ_{r,s} G[r,s] B_r^{2p}(u) B_s^{2q}(v),
        h(u,v) = Σ_{r,s} H[r,s] B_r^{2p}(u) B_s^{2q}(v).

    Parameters
    ----------
    P1 : (p+1, d)  control points of curve 1 in Euclidean space
    W1 : (p+1,)    positive weights of curve 1
    P2 : (q+1, d)  control points of curve 2 in Euclidean space
    W2 : (q+1,)    positive weights of curve 2

    Returns
    -------
    G : (2p+1, 2q+1)  bivariate Bernstein control net of the numerator g(u,v)
    H : (2p+1, 2q+1)  bivariate Bernstein control net of the denominator h(u,v)

    Notes
    -----
    Let c1(u) = Σ_i (w1[i] P1[i]) B_i^p(u),  w1(u) = Σ_i w1[i] B_i^p(u),
        c2(v) = Σ_j (w2[j] P2[j]) B_j^q(v),  w2(v) = Σ_j w2[j] B_j^q(v).
    Then
        g(u,v) = || c1(u) w2(v) - c2(v) w1(u) ||^2,
        h(u,v) = w1(u)^2 * w2(v)^2.
    This function constructs both control nets exactly using Bernstein product identities.
    """
    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64).reshape(-1)
    W2 = np.asarray(W2, dtype=np.float64).reshape(-1)

    if P1.ndim != 2 or P2.ndim != 2:
        raise ValueError("P1 and P2 must be 2D arrays of shapes (p+1,d) and (q+1,d).")
    if P1.shape[1] != P2.shape[1]:
        raise ValueError("P1 and P2 must have the same ambient dimension d.")
    if P1.shape[0] != W1.shape[0] or P2.shape[0] != W2.shape[0]:
        raise ValueError("Weights must match the number of control points.")

    # Homogeneous numerators (control nets of c1 and c2)
    C1_num = W1[:, None] * P1  # (p+1, d)
    C2_num = W2[:, None] * P2  # (q+1, d)

    # Build bivariate vector net E_{i,j,:} = c1_i * w2_j - c2_j * w1_i   (degree (p,q))
    E = C1_num[:, None, :] * W2[None, :, None] - C2_num[None, :, :] * W1[:, None, None]  # (p+1,q+1,d)

    # Numerator G: squared norm net of E  (degree (2p,2q))
    G = _bivariate_squared_norm_net(E)

    # Denominator H: (w1(u)^2) * (w2(v)^2) -> outer of the squared-univariate nets
    W1_sq_net = _square_univariate_bernstein_net(W1)  # (2p+1,)
    W2_sq_net = _square_univariate_bernstein_net(W2)  # (2q+1,)
    H = np.outer(W1_sq_net, W2_sq_net)                # (2p+1, 2q+1)

    return G, H

def bernstein_rational_distance_squared_nets_homog(H1: np.ndarray, H2: np.ndarray):
    """
    Variant that accepts *homogeneous* control points directly.
    H1 : (p+1, d+1) with last column weights (w), first d columns = w*P
    H2 : (q+1, d+1)
    Returns (G, H) as above.
    """
    H1 = np.asarray(H1, dtype=np.float64)
    H2 = np.asarray(H2, dtype=np.float64)
    if H1.ndim != 2 or H2.ndim != 2:
        raise ValueError("H1 and H2 must be 2D arrays of shapes (p+1,d+1) and (q+1,d+1).")
    if H1.shape[1] != H2.shape[1]:
        raise ValueError("H1 and H2 must have the same ambient homogeneous dimension (d+1).")

    # Extract numerator control points (w*P) and weights
    C1_num = H1[:, :-1]  # (p+1, d)
    W1     = H1[:, -1]   # (p+1,)
    C2_num = H2[:, :-1]  # (q+1, d)
    W2     = H2[:, -1]   # (q+1,)

    # E_{i,j,:} = c1_i * w2_j - c2_j * w1_i
    E = C1_num[:, None, :] * W2[None, :, None] - C2_num[None, :, :] * W1[:, None, None]

    # Build G, H exactly
    G = _bivariate_squared_norm_net(E)
    W1_sq_net = _square_univariate_bernstein_net(W1)
    W2_sq_net = _square_univariate_bernstein_net(W2)
    H = np.outer(W1_sq_net, W2_sq_net)
    return G, H

MACHINE_EPS_SQRT=np.sqrt(np.finfo(float).eps)
def _bez_int(bez1, bez2, atol=0.001, angle_tol: float = 0.0013704652454261668):
    
    atol_sq = atol * atol
    bern1, bern2 = bez1.control_points, bez2.control_points
    is_overlap, result = _bez_curve_overlap(bez1, bez2, spt=atol, angle_tol=angle_tol)
    (u0, u1) = bez1.interval()
    (v0, v1) = bez2.interval()
    overlaps = []
    if is_overlap:
        u_start: CCXInt
        u_end: CCXInt
        v_start: CCXInt
        v_end: CCXInt
        overlaps.append(np.array(((result.start.s, result.start.t), (result.end.s, result.end.t))))
        u_start, u_end = min(result.start, result.end, key=lambda x: x.s), max(result.start, result.end,
                                                                               key=lambda x: x.s)
        v_start, v_end = min(result.start, result.end, key=lambda x: x.t), max(result.start, result.end,
                                                                               key=lambda x: x.t)
        
        ou0, ou1 = u_start.s, u_end.s
        ov0, ov1 = v_start.t, v_end.t
        full_overlap_u = not (u0 <= (ou0 - u_start.ds) or u1 >= (ou1 + u_end.ds))
        full_overlap_v = not (v0 <= (ov0 - v_start.dt) or v1 >= (ov1 + v_end.dt))
        
        if full_overlap_u or full_overlap_v:
            return [], overlaps
        
        if (ou0 - u0) < u_start.ds:
            u_param = ou1, u1
        
        elif (u1 - ou1) < u_end.ds:
            
            u_param = v0, ov0
        else:
            raise ValueError(f'wtf: {(u0, ou0, ou1, u1)}')
        if (ov0 - v0) < v_start.dt:
            v_param = ov1, v1
        elif (v1 - ov1) < v_end.dt:
            
            v_param = v0, ov0
        
        else:
            raise ValueError(f'wtf: {(v0, ov0, ov1, v1)}')
        
        bez1 = trim_curve(bez1, *u_param)
        bez2 = trim_curve(bez2, *v_param)
    if not aabb_intersect(aabb(bez1.control_points), aabb(bez2.control_points)):
        return [], overlaps
    F = bernstein_distance_squared_net(bern1, bern2)[..., None]  # (N,M,1)
    
    dst_sq = bern_to_nurbs_bezier(F, interval=(bez1.interval(), bez2.interval()),
                                  rational=False)
    (u0, u1), (v0, v1) = dst_sq.interval()
    i, j, _ = dst_sq.control_points.shape
    PF = np.zeros((F.shape[0], F.shape[1], 3))
    PF[..., 2] = np.squeeze(dst_sq.control_points)
    mgr = np.mgrid[u0:u1:complex(i), v0:v1:complex(j)]
    PF[..., 0] = mgr[0, ...]
    PF[..., 1] = mgr[1, ...]
    
    gm_sq = bern_to_nurbs_bezier(compute_gauss_map(PF), interval=(bez1.interval(), bez2.interval()),
                                 rational=False)
    
    stack = [(dst_sq, gm_sq)]
    roots = []
    
    while stack:
        
        sub_dist_sq, gm = stack.pop(0)
        
        (u0, u1), (v0, v1) = sub_dist_sq.interval()
        
        min_f = np.min(sub_dist_sq.control_points[..., -1])
        max_f = np.max(sub_dist_sq.control_points[..., -1])
        # print(min_f,max_f)
        
        if min_f > 0.:
            # print('C0', min_f)
            
            continue
            
            # gm = compute_gauss_map(sub_dist_sq.control_points)
            # if not origin_in_convex_hull(gm.reshape((-1, 3))[...,:-1]):
            #    continue
        _p=gm.control_points.reshape((-1, 3))
        
        if not origin_in_convex_hull(_p/np.linalg.norm(_p, axis=1,keepdims=True), tol=np.finfo(np.float64).eps):
            
            # print('no root', ((u0, u1), (v0, v1)))
            # pts = np.zeros((i, j, 3))
            # print(sub_dist_sq.control_points.shape)
            # print(pts[..., -1].shape)
            # pts[..., -1] = np.squeeze(sub_dist_sq.control_points)
            # pts[..., :-1] = (np.mgrid[u0:u1:complex(i), v0:v1:complex(i)].T) * 10
            # gm=compute_gauss_map(pts)
            # all_patches.append(bern_to_nurbs_bezier(pts, rational=False))
            continue
        
        elif (max_f > atol_sq):
            
            umid = u0 + (u1 - u0) / 2
            vmid = v0 + (v1 - v0) / 2
            
            stack.extend(zip(subdivide_surface(sub_dist_sq, umid, vmid), subdivide_surface(gm, umid, vmid)))
        
        
        
        else:
            
            def eq(uv):
                u, v = uv
                return evaluate_nurbs_surface(sub_dist_sq, u, v)['S'][0]
            
            initial = np.zeros(2)
            initial[0] = u0 + (u1 - u0) / 2
            initial[1] = v0 + (v1 - v0) / 2
            result = newtons_method(eq, initial, max_iter=5)
            
            if result is None:
                raise ValueError('result is None')
            result = np.array(result)
            if not np.all(np.isfinite(result)):
                raise ValueError(f'result is not finite {result}')
            if eq(result) < MACHINE_EPS_SQRT:
              
                roots.append(result)

            # print('C2', )
            # (u0, u1), (v0, v1)=sub_dist_sq.interval()
            # i,j,_=sub_dist_sq.control_points.shape
            # pts=np.zeros((i,j,3))
            # print(sub_dist_sq.control_points.shape)
            # print(pts[..., -1].shape)
            # pts[..., -1]=np.squeeze(sub_dist_sq.control_points)
            # pts[...,:-1]=(np.mgrid[u0:u1:complex(i),v0:v1:complex(i)].T)*10
            # gm=compute_gauss_map(pts)
            
            # patches.append(bern_to_nurbs_bezier(pts,rational=False))
    
    # return roots,patches,all_patches
    roots.sort(key=lambda x: x[0])
    s_roots = []
    
    for i in roots:
        if len(overlaps) > 0:
            
            rru = (
                    (
                            (max(overlaps[0].start.s, i[0]) - min(overlaps[0].start.s, i[0])) < overlaps[0].start.ds
                    ) or (
                            (max(overlaps[0].end.s, i[0]) - min(overlaps[0].end.s, i[0])) < overlaps[0].end.ds
                    )
            )
            rrv = (
                    (
                            (max(overlaps[0].start.t, i[1]) - min(overlaps[0].start.t, i[1])) < overlaps[0].start.dt
                    ) or (
                            (max(overlaps[0].end.t, i[1]) - min(overlaps[0].end.t, i[1])) < overlaps[0].end.dt
                    )
            )
            if rru and rrv:
                continue
        
        if len(s_roots) == 0:
            s_roots.append(i)
            continue
        u_last, v_last = s_roots[-1]
        
        p1 = evaluate_nurbs_curve(bez1, u_last, 0)['C']
        p11 = evaluate_nurbs_curve(bez1, i[0], 0)['C']
        dp1 = np.linalg.norm(p1 - p11)
        if dp1 <= atol:
            
            p2 = evaluate_nurbs_curve(bez2, v_last, 0)['C']
            p22 = evaluate_nurbs_curve(bez2, i[1], 0)['C']
            if np.linalg.norm(p1 - p2) > np.linalg.norm(p11 - p22):
                s_roots[-1] = i
                continue
            else:
                continue
        else:
            s_roots.append(i)
    
    return s_roots, overlaps


def _bez_int_rat(bez1, bez2, atol=0.001,angle_tol:float=0.0013704652454261668):
    atol_sq = atol * atol
    bern1, bern2 = bez1.control_points, bez2.control_points
    is_overlap,result=_bez_curve_overlap(bez1, bez2, spt=atol,angle_tol=angle_tol)
    (u0, u1)=bez1.interval()
    (v0, v1)=bez2.interval()
    overlaps=[]
    if is_overlap:
        u_start:CCXInt
        u_end: CCXInt
        v_start: CCXInt
        v_end: CCXInt
        overlaps.append(np.array(((result.start.s,result.start.t),  (result.end.s,result.end.t))))
        u_start,u_end=min(result.start,result.end,key=lambda x:x.s),max(result.start,result.end,key=lambda x:x.s)
        v_start, v_end = min(result.start, result.end, key=lambda x: x.t), max(result.start, result.end,
                                                                               key=lambda x: x.t)
        
        ou0,ou1=u_start.s,u_end.s
        ov0,ov1=v_start.t,v_end.t
        full_overlap_u= not (u0 <= (ou0 - u_start.ds) or u1 >= (ou1 + u_end.ds))
        full_overlap_v = not (v0 <= (ov0 - v_start.dt) or v1 >= (ov1 + v_end.dt))

        if full_overlap_u or full_overlap_v:
            return [], overlaps
        
        if (ou0-u0)<u_start.ds:
            u_param=ou1,u1
           
        elif (u1-ou1)<u_end.ds:
            
            u_param=v0,ov0
        else:
            raise ValueError(f'wtf: {(u0, ou0, ou1, u1)}')
        if (ov0 - v0) < v_start.dt:
            v_param = ov1,v1
        elif (v1 - ov1) < v_end.dt:
            
            v_param =  v0,ov0
            
        else:
            raise ValueError(f'wtf: {(v0,ov0,ov1,v1)}')
       
        bez1=trim_curve(bez1, *u_param)
        bez2=trim_curve(bez2, *v_param)
    if not aabb_intersect(aabb(bez1.control_points),aabb(bez2.control_points)):
        return [], overlaps
    #F = bernstein_distance_squared_net(bern1, bern2)[..., None]  # (N,M,1)
    G,H=bernstein_rational_distance_squared_nets(bez1.control_points,bez1.weights, bez2.control_points,bez2.weights)
    dst_sq = bern_to_nurbs_bezier(np.stack([G,H],axis=-1), interval=(bez1.interval(), bez2.interval()),
                                  rational=True)
    (u0, u1), (v0, v1) = dst_sq.interval()
    i, j, _ = dst_sq.control_points.shape
    PF = np.zeros((i, j, 3))
    PF[..., 2] = np.squeeze(dst_sq.control_points)
    mgr = np.mgrid[u0:u1:complex(i), v0:v1:complex(j)]
    PF[..., 0] = mgr[0, ...]
    PF[..., 1] = mgr[1, ...]
    
  
    dsqq = bern_to_nurbs_bezier(to_homogeneous_2d(PF
                                                , dst_sq.weights), interval=dst_sq.interval(), rational=True)
  
    
    gm_sq = bern_to_nurbs_bezier(compute_gauss_map_rational(to_homogeneous_2d(dsqq.control_points,dsqq.weights)), interval=(bez1.interval(), bez2.interval()),
                                 rational=True)
    
    stack = [(dst_sq, gm_sq)]
    roots = []
    
    
    while stack:
        
        sub_dist_sq, gm = stack.pop(0)
        
        (u0, u1), (v0, v1) = sub_dist_sq.interval()
        gmin = sub_dist_sq.control_points[..., -1].min()
        gmax = sub_dist_sq.control_points[..., -1].max()
        hmax = sub_dist_sq.weights.max()
        hmin = sub_dist_sq.weights.min()
        max_f =  gmax / hmin
        #min_f = np.min(sub_dist_sq.control_points[..., -1])
        #max_f = np.max(sub_dist_sq.control_points[..., -1])
        min_f =  gmin / hmax
     
        
        # print(min_f,max_f)
        
        if min_f > 0.:
            # print('C0', min_f)
            
            continue
            
            # gm = compute_gauss_map(sub_dist_sq.control_points)
            # if not origin_in_convex_hull(gm.reshape((-1, 3))[...,:-1]):
            #    continue
        _p=(gm.control_points/gm.weights[...,None]).reshape((-1, 3))
       
        if not origin_in_convex_hull( _p/ np.linalg.norm(_p,axis=1,keepdims=True), tol=np.finfo(np.float64).eps):
            
            # print('no root', ((u0, u1), (v0, v1)))
            # pts = np.zeros((i, j, 3))
            # print(sub_dist_sq.control_points.shape)
            # print(pts[..., -1].shape)
            # pts[..., -1] = np.squeeze(sub_dist_sq.control_points)
            # pts[..., :-1] = (np.mgrid[u0:u1:complex(i), v0:v1:complex(i)].T) * 10
            # gm=compute_gauss_map(pts)
            # all_patches.append(bern_to_nurbs_bezier(pts, rational=False))
            continue
        
        elif (max_f > atol_sq):
            
            umid = u0 + (u1 - u0) / 2
            vmid = v0 + (v1 - v0) / 2
            
            stack.extend(zip(subdivide_surface(sub_dist_sq, umid, vmid), subdivide_surface(gm, umid, vmid)))
        
        
        
        else:
            
            def eq(uv):
                u, v = uv
                return evaluate_nurbs_surface(sub_dist_sq, u, v)['S'][0]
            
            initial = np.zeros(2)
            initial[0] = u0 + (u1 - u0) / 2
            initial[1] = v0 + (v1 - v0) / 2
            result = newtons_method(eq, initial, max_iter=5)
            
            if result is None:
                raise ValueError('result is None')
            result = np.array(result)
            if not np.all(np.isfinite(result)):
                raise ValueError(f'result is not finite {result}')
            if eq(result) < MACHINE_EPS_SQRT:
                
                roots.append(result)
            # print('C2', )
            # (u0, u1), (v0, v1)=sub_dist_sq.interval()
            # i,j,_=sub_dist_sq.control_points.shape
            # pts=np.zeros((i,j,3))
            # print(sub_dist_sq.control_points.shape)
            # print(pts[..., -1].shape)
            # pts[..., -1]=np.squeeze(sub_dist_sq.control_points)
            # pts[...,:-1]=(np.mgrid[u0:u1:complex(i),v0:v1:complex(i)].T)*10
            # gm=compute_gauss_map(pts)
            
            # patches.append(bern_to_nurbs_bezier(pts,rational=False))
    
    # return roots,patches,all_patches
    roots.sort(key=lambda x: x[0])
    s_roots = []
    
    for i in roots:
        if len(overlaps)>0:
            
            rru=(
                    (
                     (max(overlaps[0].start.s,i[0])-min(overlaps[0].start.s, i[0]))<overlaps[0].start.ds
                    ) or (
                    (max(overlaps[0].end.s, i[0]) - min(overlaps[0].end.s, i[0])) < overlaps[0].end.ds
                   )
            )
            rrv = (
                (
                        (max(overlaps[0].start.t, i[1]) - min(overlaps[0].start.t, i[1])) < overlaps[0].start.dt
                ) or (
                        (max(overlaps[0].end.t, i[1]) - min(overlaps[0].end.t, i[1])) < overlaps[0].end.dt
                )
            )
            if rru and rrv:
                continue
                
          
        if len(s_roots) == 0:
            s_roots.append(i)
            continue
        u_last, v_last = s_roots[-1]
        
        p1 = evaluate_nurbs_curve(bez1, u_last, 0)['C']
        p11 = evaluate_nurbs_curve(bez1, i[0], 0)['C']
        dp1 = np.linalg.norm(p1 - p11)
        if dp1 <= atol:
            
            p2 = evaluate_nurbs_curve(bez2, v_last, 0)['C']
            p22 = evaluate_nurbs_curve(bez2, i[1], 0)['C']
            if np.linalg.norm(p1 - p2) > np.linalg.norm(p11 - p22):
                s_roots[-1] = i
                continue
            else:
                continue
        else:
            s_roots.append(i)
    
    return  s_roots,overlaps




if __name__ == '__main__':
    import numpy as np
    from mmcore.geom._nurbs_eval import NURBSCurveTuple
    
    val1 = NURBSCurveTuple(
        order=4,
        knot=np.array([0., 0., 0., 0., 1., 1., 1., 1.]),
        control_points=np.array([[-19.77608536, 23.10065701, 0.],
                                 [-14.86834768, 28.69713066, 0.],
                                 [-5.8568525, 25.12677787, 0.],
                                 [-12.62581769, 15.26478654, 0.]]),
        weights=np.array([1., 1., 1., 1.])
    )
    import numpy as np
    from mmcore.geom._nurbs_eval import NURBSCurveTuple
    
    val2 = NURBSCurveTuple(
        order=4,
        knot=np.array([0., 0., 0., 0., 1., 1., 1., 1.]),
        control_points=np.array([[-22.0315362, 18.75969713, 0.],
                                 [-19.42270945, 28.2502867, 0.],
                                 [-8.46791623, 27.56878356, 0.],
                                 [-10.43007782, 19.78973126, 0.]]),
        weights=np.array([1., 1., 1., 1.])
    )
    isolated,overlaps=_bez_int(val1,val2, 1e-3) # [] , [array([[0.        , 0.19069075], [0.82759776, 1.        ]])]
    
    
    # noinspection PyTypeChecker
    val3=val1._replace(control_points=val1.control_points+np.array([-0.0333436 , -0.03435689,  0.        ])) # small translation

    isolated2, overlaps2 = _bez_int(val3, val2, 1e-3)  # [array([0.02160491, 0.20967348])] , []
    

    val0 = NURBSCurveTuple(
        order=4,
        knot=np.array([ 0.        ,  0.        ,  0.        ,  0.        , 75.19404178,
               75.19404178, 75.19404178, 75.19404178]),
        control_points=np.array([[-28.46565557, -11.09883504,   0.        ],
               [-31.79098016,  13.62423043,   0.        ],
               [-12.99566723,  16.66039636,   0.        ],
               [  8.11291498,  -6.32771715,   0.        ]]),
        weights=np.array([1., 1., 1., 1.])
    )
    
    

    
    val1 = NURBSCurveTuple(
        order=4,
        knot=np.array([0., 0., 0., 0.,
                       108.34889276, 108.34889276, 108.34889276, 108.34889276]),
        control_points=np.array([[-45.36434109, -7.12015504, 0.],
                                 [-25.49612403, 13.94186047, 0.],
                                 [-2.13178295, -17.35271318, 0.],
                                 [12.02325581, 20.42248062, 0.]]),
        weights=np.array([1., 1., 1., 1.])
    )
    import time
    s=time.perf_counter()
    isolated12, overlaps12 = _bez_int(val0, val1,
                                     1e-3)  # [array([19.35827625,  0.90777449]), array([62.9638176 ,  0.06147848])] []
    print(time.perf_counter()-s,isolated12, overlaps12 )
    s=time.perf_counter()
    isolated12, overlaps12 = _bez_int_rat(val0, val1,
                                     1e-3)  # [array([19.35827625,  0.90777449]), array([62.9638176 ,  0.06147848])] []
    print(time.perf_counter()-s,isolated12, overlaps12 )
    # RATIONAL CASES:
    val2 = NURBSCurveTuple(
        order=3,
        knot=np.array([0., 0., 0., 1., 1., 1.]),
        control_points=np.array([[-8.33199606, -0.74131171, 0.],
                                 [-22.97431561, 12.65582434, 0.],
                                 [-36.37145166, -1.98649522, 0.]]),
        weights=np.array([1., 0.70710678, 1.])
    )
    import numpy as np
    from mmcore.geom._nurbs_eval import NURBSCurveTuple
    
    val3 = NURBSCurveTuple(
        order=2,
        knot=np.array([0., 0., 16.74026551, 16.74026551]),
        control_points=np.array([[-34.45266686, -0.15233323, 0.],
                                 [-9.67826877, 0.38524533, 0.]]),
        weights=np.array([1., 1.])
    )
    

    
    isolated12, overlaps12 = _bez_int_rat(val1, val2, 1e-3)  # [array([19.35827625,  0.90777449]), array([62.9638176 ,  0.06147848])] []
    print(isolated12, overlaps12)
    isolated32, overlaps32 =  _bez_int_rat(val3, val2, 1e-3)  # [array([0.00000001, 0.90777449]), array([16.74026551,  0.06147848])] []
    print(isolated32, overlaps32)
    isolated13, overlaps13 =  _bez_int_rat(val1, val3, 1e-3)  # [array([19.35827621,  0.        ]), array([57.71658282, 14.79457618]), array([62.96381746, 16.74026546])] []
    print(isolated13, overlaps13)