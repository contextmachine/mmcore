from __future__ import annotations

import numpy as np
import math

from numpy._typing import NDArray

from mmcore.geom._nurbs_eval import NURBSCurveTuple, NURBSSurfaceTuple, from_homogeneous_2d
from mmcore.geom._nurbs_knots import normalize_knots_curve, to_homogeneous_1d, make_curves_compatible, \
    make_curves_compatible_multiple


def circle(radius=1.0, start_angle=0.0, end_angle=2 * math.pi, center=None, normal=None, xaxis=None, yaxis=None):
    """
    Create a NURBS representation of a circle (or circular arc).

    Parameters:
      radius      : The radius of the circle.
      start_angle : The starting angle in radians.
      end_angle   : The ending angle in radians.
      center      : A 3D point (iterable of 3 numbers) representing the center.
                    Defaults to [0,0,0] if not provided.
      normal      : A 3D vector (iterable of 3 numbers) normal to the plane.
                    Used if xaxis and yaxis are not provided.
      xaxis, yaxis: Two 3D vectors (iterable of 3 numbers) defining the in‐plane axes.
                    If provided, these are used to orient the circle.

    Returns:
      A NURBSCurveTuple representing the circle (or arc).
    """
    # Set default center if not provided
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.array(center, dtype=float)

    # Determine the coordinate system.
    # Option 1: Use provided xaxis and yaxis.
    if (xaxis is not None) and (yaxis is not None):
        xaxis = np.array(xaxis, dtype=float)
        yaxis = np.array(yaxis, dtype=float)
        xaxis = xaxis / np.linalg.norm(xaxis)
        yaxis = yaxis / np.linalg.norm(yaxis)
    # Option 2: Use the provided normal vector.
    elif normal is not None:
        normal = np.array(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        # Choose an arbitrary vector that is not parallel to the normal.
        if abs(normal[0]) < 1e-6 and abs(normal[1]) < 1e-6:
            # If normal is (nearly) vertical, choose xaxis = (1,0,0)
            xaxis = np.array([1.0, 0.0, 0.0])
        else:
            # Otherwise, use the cross product with a reference vector (here z-axis).
            xaxis = np.cross([0, 0, 1], normal)
            if np.linalg.norm(xaxis) < 1e-6:
                xaxis = np.array([1.0, 0.0, 0.0])
            else:
                xaxis = xaxis / np.linalg.norm(xaxis)
        # yaxis is then given by normal x xaxis
        yaxis = np.cross(normal, xaxis)
        yaxis = yaxis / np.linalg.norm(yaxis)
    # Option 3: Default to the xy-plane.
    else:
        xaxis = np.array([1.0, 0.0, 0.0])
        yaxis = np.array([0.0, 1.0, 0.0])

    # Ensure that end_angle > start_angle.
    delta = end_angle - start_angle
    if delta <= 0:
        raise ValueError("end_angle must be greater than start_angle")

    # Divide the arc into segments of at most 90 degrees (pi/2 radians)
    n_seg = int(math.ceil(delta / (math.pi / 2)))
    seg_angle = delta / n_seg

    # Build lists to hold the control points (in local 2D coordinates) and weights.
    # For a quadratic (degree-2) NURBS arc segment, three control points are used.
    # When joining segments, the endpoint of one is the start point of the next.
    control_points_local = []
    weights = []

    for i in range(n_seg):
        theta0 = start_angle + i * seg_angle
        theta2 = start_angle + (i + 1) * seg_angle
        theta_mid = (theta0 + theta2) / 2.0
        # Weight for the middle control point.
        w_mid = math.cos(seg_angle / 2)

        # Compute the control points in 2D (local coordinates).
        P0 = np.array([radius * math.cos(theta0), radius * math.sin(theta0)])
        # For the middle control point, scale so that it is off the circle
        # by a factor of 1/cos(seg_angle/2)
        P1 = np.array([radius / w_mid * math.cos(theta_mid), radius / w_mid * math.sin(theta_mid)])
        P2 = np.array([radius * math.cos(theta2), radius * math.sin(theta2)])

        if i == 0:
            control_points_local.append(P0)
            control_points_local.append(P1)
            control_points_local.append(P2)
            weights.append(1.0)
            weights.append(w_mid)
            weights.append(1.0)
        else:
            # Avoid duplicating the shared control point between segments.
            control_points_local.append(P1)
            control_points_local.append(P2)
            weights.append(w_mid)
            weights.append(1.0)

    # Number of control points for a piecewise quadratic curve.
    n_cp = len(control_points_local)
    # For a degree-2 curve, the knot vector has length: n_cp + 3.
    # Construct the knot vector in clamped form:
    # [0, 0, 0, u1, u1, u2, u2, ..., 1, 1, 1] with internal knots equally spaced.
    knot_vector = [0, 0, 0]
    for i in range(1, n_seg):
        u = i / n_seg
        knot_vector.extend([u, u])
    knot_vector.extend([1, 1, 1])

    # Transform the local 2D control points to 3D using the coordinate system.
    control_points_global = []
    for pt in control_points_local:
        # Map a local point (a, b) to 3D: center + a*xaxis + b*yaxis.
        global_pt = center + pt[0] * xaxis + pt[1] * yaxis
        control_points_global.append(global_pt)

    # Convert lists to numpy arrays.
    control_points_global = np.array(control_points_global)
    weights = np.array(weights)
    knot_vector = np.array(knot_vector, dtype=float)

    # Return the NURBS curve tuple.
    # Here, we assume NURBSCurveTuple takes (dimension, knot_vector, control_points, weights).
    return NURBSCurveTuple(3, knot_vector, control_points_global, weights)


def ruled(curve1:NURBSCurveTuple, curve2:NURBSCurveTuple)->NURBSSurfaceTuple:
    """
    Generates a ruled surface between two given NURBS curves. A ruled surface is a
    surface created by linear interpolation between corresponding points on two
    curves. This function assumes that the input curves are NURBS curves and processes
    them to make them compatible before producing the NURBS surface. If the input
    curves have different knot vectors or control points, they will be modified to
    produce a valid ruled surface.

    :param curve1: The first input NURBS curve.
    :type curve1: NURBSCurveTuple
    :param curve2: The second input NURBS curve.
    :type curve2: NURBSCurveTuple

    :return: A NURBS ruled surface created between the two input curves.
    :rtype: NURBSSurfaceTuple
    """
    # Make curves compatible
    curve1=normalize_knots_curve(curve1)
    curve2=normalize_knots_curve(curve2)

    c1, c2 = make_curves_compatible(curve1, curve2)

    # Create surface control points
    n = len(c1.control_points)
    control_points = np.zeros((n, 2, 4))  # nx2x4 array


    # Fill control points

    control_points[:, 0, :] =  to_homogeneous_1d(c1.control_points,c1.weights)
    control_points[:, 1, :] =  to_homogeneous_1d(c2.control_points,c1.weights)

    # Create surface knot vectors
    u_knots = c1.knots  # Same for both curves now
    v_knots = np.array([0., 0., 1., 1.])  # Linear interpolation in v direction

    return NURBSSurfaceTuple( order_u=c1.degree+1,
                              order_v=2,
                              knot_u=u_knots,
                              knot_v=v_knots,
                              control_points=np.ascontiguousarray(control_points[...,:-1]),
                              weights=np.ascontiguousarray(control_points[...,-1]))
from mmcore.geom._nurbs_interp import interpolate_curve

from typing import NamedTuple, Literal, List

LoftType=Literal['normal', 'loose','straight']


class LoftOptions(NamedTuple):
    loft_type:LoftType
def default_loft_options()->LoftOptions:
    return LoftOptions(LoftType.NORMAL)


def loft(curves: list[NURBSCurveTuple],
         degree_v: int = 3,
         v_params: list[float] | None = None
         ) -> NURBSSurfaceTuple:
    """
    Creates a tensor-product NURBS surface whose v-isocurves
    are **exactly** the given curves.
    """

    # -- 0. Compatibility in U
    curves = make_curves_compatible_multiple(curves)
    v_count = len(curves)                       # number of section curves
    u_count = curves[0].control_points.shape[0] # ctrlpts per curve

    order_u  = curves[0].order                  # already degree+1
    knots_u  = curves[0].knot

    # -- 1. Choose parameter values v_i   (Rhino: uniform, chord-length, …)
    if v_params is None:
        v_params = np.linspace(0.0, 1.0, v_count, dtype=float)
    assert len(v_params) == v_count

    # -- 2. Build the control-point *grid* in [v][u][4] order  ★
    grid4 = np.empty((v_count, u_count, 4))
    for i, crv in enumerate(curves):
        grid4[i, :, :] = to_homogeneous_1d(crv.control_points, crv.weights)

    # -- 3. Interpolate every U-column with **one shared** knot vector in V  ★
    order_v = degree_v + 1
    kv_v    = None
    for j in range(u_count):
        ctrl4, kv_v = interpolate_curve(
            grid4[:, j, :],             # data points for this column
            degree_v,
            params=v_params,
            return_knots=True
        )
        grid4[:, j, :] = ctrl4          # overwrite the column with its ctrl pts

    # -- 4. Back to Euclidean ctrl pts + weights
    ctrlpts, wts = from_homogeneous_2d(grid4)

    # -- 5. Assemble the surface  (orders, knots, lattice)
    return NURBSSurfaceTuple(
        order_u, order_v,               # u-order, v-order
        knots_u, kv_v,                  # common knot vectors
        ctrlpts, wts                    # [v][u][3] and [v][u]
    )


# ---------------------------------------------------------------------
#  Basic B-spline / NURBS utilities
# ---------------------------------------------------------------------


from mmcore.geom._nurbs_eval import evaluate_nurbs_curve,bspline_basis
# ---------------------------------------------------------------------
#  Gordon-surface construction
# ---------------------------------------------------------------------
def bspline_basis_vector(knot: NDArray[np.float64],
                         degree: int,
                         u: float) -> NDArray[np.float64]:
    """Return the complete vector [N_0(u), …, N_{K-1}(u)]."""
    K = len(knot) - degree - 1           # number of non-zero one-dim. bases
    return np.fromiter((bspline_basis(j, degree, knot, u) for j in range(K)),
                       dtype=float, count=K)

# ----------------------------------------------------------------------
#  Curve evaluation (unchanged logic)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
#  Gordon surface construction
# ----------------------------------------------------------------------
def construct_gordon_surface(
    curves_u: List[NURBSCurveTuple],
    curves_v: List[NURBSCurveTuple],
    v_params: NDArray[np.float64]=None,
    u_params: NDArray[np.float64]=None,spt=1e-3
) -> NURBSSurfaceTuple:

    # 1 - unify knots/degree separately for each family
    C =curves_u= make_curves_compatible_multiple(curves_u)   # (m+1) curves
    D = curves_v=make_curves_compatible_multiple(curves_v)   # (n+1) curves
    m1, n1 = len(C), len(D)
    dim= dim  = C[0].control_points.shape[1]
    P_corner = np.empty((m1, n1, dim))
    if u_params is None:
        u_params = np.linspace(0.0, 1.0, len(curves_v), dtype=float)
        v_params = np.linspace(0.0, 1.0, len(curves_u), dtype=float)

    uv_params = np.zeros((len(curves_u), len(curves_v), 2))
    pmap_u = np.zeros_like(u_params)
    pmap_v = np.zeros_like(v_params)

    for i, cu in enumerate(curves_u):
        for j, cv in enumerate(curves_v):

            res = nurbs_ccx(cu, cv, spt=spt)
            if len(res) != 1:
                raise ValueError("nurbs_ccx error")
            P_corner[i,j]= res[0][0]
            uv_params[i, j, :] = res[0][1]
            u_params[j]=pmap_u[j] = res[0][1][0]
            v_params[i]=pmap_v[i] = res[0][1][1]
    print('------ DEBUG: uv_params -------')
    print(uv_params)
    print('------ DEBUG: v_params,u_params -------')
    print(v_params,u_params)
    if len(v_params) != m1 or len(u_params) != n1:
        raise ValueError("v_params length must equal len(curves_u) and "
                         "u_params length must equal len(curves_v).")

    p, U = C[0].order - 1, C[0].knot
    q, V = D[0].order - 1, D[0].knot
    K_u  = len(U) - p - 1
    K_v  = len(V) - q - 1
    dim  = C[0].control_points.shape[1]

    # 2 - intersection grid
    #P_corner = np.empty((m1, n1, dim))
    #for i, cur in enumerate(C):
    #    P_corner[i] = [evaluate_nurbs_curve(cur, uj,0)['C'] for uj in u_params]

    # 3 - basis-value matrices  (collocation)

    Bv = np.stack([bspline_basis_vector(V, q, v) for v in v_params])  # (m1, K_v)
    Bu = np.stack([bspline_basis_vector(U, p, u) for u in u_params])  # (n1, K_u)
    print('\n------ DEBUG: Bv,Bu -------')

    print(Bv,Bu,)
    #   Moore–Penrose gives min-norm La,Κ that satisfy interpolation conditions
    La = (np.linalg.pinv(Bv) @ np.eye(m1)).T      # (m1, K_v)
    Κ = (np.linalg.pinv(Bu) @ np.eye(n1)).T      # (n1, K_u)

    # 4 - gather compatible control nets
    CP_u = np.stack([c.control_points for c in C])   # (m1, K_u, dim)
    W_u  = np.stack([c.weights for c in C])          # (m1, K_u)

    CP_v = np.stack([d.control_points for d in D])   # (n1, K_v, dim)
    W_v  = np.stack([d.weights for d in D])          # (n1, K_v)

    # 5 - build partial homogeneous sums
    W_su   = np.zeros((K_u, K_v)); Pw_su  = np.zeros((K_u, K_v, dim))
    W_sv   = np.zeros((K_u, K_v)); Pw_sv  = np.zeros((K_u, K_v, dim))
    W_suv  = np.zeros((K_u, K_v)); Pw_suv = np.zeros((K_u, K_v, dim))

    # 5a  sweep rows
    for i in range(m1):
        lam = La[i]                                # (K_v,)
        w   = W_u[i][:, None]                     # (K_u,1)
        W_su  += w @ lam[None, :]                 # rank-1 outer product
        Pw_su += (w @ lam[None, :])[:, :, None] * CP_u[i][:, None, :]

    # 5b  sweep columns
    for j in range(n1):
        kap = Κ[j]                                # (K_u,)
        w   = W_v[j][None, :]                     # (1,K_v)
        W_sv  += kap[:, None] @ w                 # rank-1 outer product
        Pw_sv += (kap[:, None] @ w)[:, :, None] * CP_v[j][None, :, :]

    # 5c  bilinear correction
    for i in range(m1):
        lam = La[i]                                # (K_v,)
        for j in range(n1):
            kap = Κ[j]                            # (K_u,)
            outer = kap[:, None] * lam[None, :]   # (K_u,K_v)
            W_suv  += outer
            Pw_suv += outer[:, :, None] * P_corner[i, j]

    # 6 - combine and dehomogenise
    W_tot = W_su + W_sv - W_suv
    if np.any(np.abs(W_tot) < np.finfo(float).eps):
        print('\n------ DEBUG: W_tot -------')
        print(W_tot)
        raise ZeroDivisionError("Composite weight vanished — check input data.")
    CP_tot = (Pw_su + Pw_sv - Pw_suv) / W_tot[:, :, None]

    # 7 - package as a NURBS surface
    return NURBSSurfaceTuple(
        order_u       = C[0].order,
        order_v       = D[0].order,
        knot_u        = U,
        knot_v        = V,
        control_points= CP_tot,
        weights       = W_tot
    )

from mmcore.numeric.intersection.ccx._nccx import nurbs_ccx

def network_surface(curves_u:List[NURBSCurveTuple],curves_v: List[NURBSCurveTuple], spt:float=1e-3,**kwargs):
    bvhs_u=[]
    for cu in curves_u:
        bvh1, curves1 = nurbs_curve_bvh(curve1, spt=spt)
        bvhs_u.append((bvh1, curves1))
        
    if bvh1 is None:
        bvh1, curves1 = nurbs_curve_bvh(curve1, spt=spt)
    if bvh2 is None:
        bvh2, curves2 = nurbs_curve_bvh(curve2, spt=spt)

    nurbs_ccx(curves_u,curves_v,spt)
    return NURBSSurfaceTuple(
        order_u       = curves_u[0].order,
        order_v       = curves_v[0].order,
        knot_u        = curves_u[0].knot,
        knot_v        = curves_v[0].knot,
        control_points= curves_u[0].control_points,
        weights       = curves_u[0].weights
    )

if __name__=='__main__':
    pts_v = [
        [[-5.0, 10.0, 2.0], [-10.0, 0.0, 2.0], [0.0, -20.0, -2.0], [10.0, 0.0, 2.0], [5.0, 10.0, 2.0]],
        [[-5.0, 10.0, 0.0], [-10.0, 0.0, 0.0], [0.0, -10.0, 0.0], [10.0, 0.0, 0.0], [5.0, 10.0, 0.0]],
    ]
    w_v = [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
    knots_v = [[0.0, 0.0, 0.0, 0.0, 11.5, 23.0, 23.0, 23.0, 23.0], [0.0, 0.0, 0.0, 0.0, 11.5, 23.0, 23.0, 23.0, 23.0]]
    degs_v = [3, 3]
    v_crvs = [NURBSCurveTuple(d + 1, np.array(k), np.array(p), np.array(w)) for d, k, p, w in zip(degs_v, knots_v, pts_v, w_v)]

    pts_u = [
        [[-5.0, 10.0, 2.0], [-5.0, 10.0, 0.0]],
        [[0.0, -10.0, 0.0], [0.0, -8.0, -1.0], [0.0, -7.0, -1.0], [3.9648774381914339e-11, -5.0, 0.0]],
        [[5.0, 10.0, 2.0], [5.0, 10.0, 0.0]],
    ]
    w_u = [[1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0]]
    knots_u = [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 2.0, 2.0]]
    degs_u = [1, 3, 1]

    u_crvs = [NURBSCurveTuple(d + 1, np.array(k), np.array(p), np.array(w)) for d, k, p, w in zip(degs_u, knots_u, pts_u, w_u)]

    print(u_crvs)

    res = construct_gordon_surface(u_crvs, v_crvs, np.linspace(0, 1, 3), np.linspace(0, 1, 2))

    print(res)
