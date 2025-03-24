from __future__ import annotations

from typing import  List, Tuple

from mmcore.geom._nurbs_eval import *
from mmcore.geom._nurbs_eval import _find_span_linear,_curve_interval,_surface_interval


# -----------------------
# Interpolation routines
# -----------------------
def interpolate_bspline_curve(points: NDArray[float], degree: int, parameters: List[float]) -> BSplineCurveTuple:
    """
    Given a set of points and associated parameter values (e.g. chord–length),
    construct a clamped B–spline curve that interpolates the points.
    """
    n = len(points)
    dim = points.shape[1]
    # Compute knot vector using averaging
    U = np.zeros(n + degree + 1)
    for i in range(degree + 1):
        U[i] = parameters[0]
    for j in range(1, n - degree):
        U[j + degree] = sum(parameters[j : j + degree]) / degree
    for i in range(n, n + degree + 1):
        U[i] = parameters[-1]
    # Build interpolation matrix A: A[i,j] = N_{j,degree}(parameters[i])
    A = np.zeros((n, n))
    for i in range(n):
        u = parameters[i]
        span = _find_span_linear(degree, U, n, u)
        ders = compute_basis_function_derivatives_np(degree, U, span, u, 0)
        N_vals = ders[0]
        for j in range(degree + 1):
            col = span - degree + j
            if col < n:
                A[i, col] = N_vals[j]
    control_points = np.zeros((n, dim))
    for d_idx in range(dim):
        b = points[:, d_idx]
        control_points[:, d_idx] = np.linalg.solve(A, b)
    return BSplineCurveTuple(order=degree + 1, knot=U, control_points=control_points)


def interpolate_bspline_surface(
    grid: NDArray[float],
    degrees: Tuple[int, int],
    us: NDArray[float],
    vs: NDArray[float],
) -> BSplineSurfaceTuple:
    """
    Tensor–product interpolation of a grid of points.
    grid has shape (n_u, n_v, dim); degrees is (deg_u, deg_v).
    We use a two–stage interpolation: first in u then in v.
    """
    n_u, n_v, dim = grid.shape
    deg_u, deg_v = degrees
    # Generate knot vector in u (using averaging)
    U = np.zeros(n_u + deg_u + 1)
    for i in range(deg_u + 1):
        U[i] = us[0]
    for j in range(1, n_u - deg_u):
        U[j + deg_u] = sum(us[j : j + deg_u]) / deg_u
    for i in range(n_u, n_u + deg_u + 1):
        U[i] = us[-1]
    # Similarly in v
    V = np.zeros(n_v + deg_v + 1)
    for i in range(deg_v + 1):
        V[i] = vs[0]
    for j in range(1, n_v - deg_v):
        V[j + deg_v] = sum(vs[j : j + deg_v]) / deg_v
    for i in range(n_v, n_v + deg_v + 1):
        V[i] = vs[-1]
    # First, interpolate in u for each fixed v
    Q = np.zeros((n_u, n_v, dim))
    for j in range(n_v):
        pts = grid[:, j, :]
        curve_u = interpolate_bspline_curve(pts, deg_u, us.tolist())
        Q[:, j, :] = curve_u.control_points
    # Now, for each fixed u, interpolate in v.
    P = np.zeros((n_u, n_v, dim))
    for i in range(n_u):
        pts = Q[i, :, :]
        curve_v = interpolate_bspline_curve(pts, deg_v, vs.tolist())
        P[i, :, :] = curve_v.control_points
    return BSplineSurfaceTuple(order_u=deg_u + 1, order_v=deg_v + 1, knot_u=U, knot_v=V, control_points=P)

'''
# -----------------------
# Knot removal routines (using re–interpolation over a dense set)
# -----------------------
def remove_knots_bspline_curve(curve: BSplineCurveTuple, tol: float) -> BSplineCurveTuple:
    """
    Iteratively attempts to remove each interior knot.
    For each candidate knot, a new curve is re–constructed by least–squares
    interpolation on a dense set of sample parameters. If the maximum error is below tol,
    the knot is removed.
    """
    # Evaluate the original curve densely.
    u0, u1 = nurbs_interval(curve.knot, curve.order - 1)
    dense_params = np.linspace(u0, u1, 100)
    original_points = np.array([evaluate_bspline_curve(curve, u) for u in dense_params])
    U = curve.knot.copy()
    ctrl_pts = curve.control_points.copy()
    p = curve.order - 1
    i = p+1  # first interior knot index
    while i < (len(U) - p-1):
        # Candidate: remove knot at index i (one occurrence)
        new_U = np.delete(U, i)

        m = len(new_U) - (p + 1)  # new number of control points
        # Form new system over the dense parameters
        N_matrix = np.zeros((len(dense_params), m))
        for r, u in enumerate(dense_params):
            span = _find_span_linear(p, new_U, m, u)
            ders = compute_basis_function_derivatives_np(p, new_U, span, u, 0)
            N_vals = ders[0]
            for j in range(p + 1):
                col = span - p + j
                if 0 <= col < m:
                    N_matrix[r, col] = N_vals[j]
        new_ctrl_pts = np.zeros((m, ctrl_pts.shape[1]))
        # Try solving each coordinate's system robustly
        success = True
        for d_idx in range(ctrl_pts.shape[1]):
            b = original_points[:, d_idx]
            # Check condition number to avoid ill-conditioned solves
            cond = np.linalg.cond(N_matrix)
            if cond > 1e12:
                success = False
                break
            try:
                sol, residuals, rank, s = np.linalg.lstsq(N_matrix, b, rcond=None)
                new_ctrl_pts[:, d_idx] = sol
            except np.linalg.LinAlgError as e:
                warnings.warn(f"lstsq failed at interior knot index {i}: {e}")
                success = False
                break
        # If the system was solved successfully, test the candidate curve.
        if success:
            candidate_curve = BSplineCurveTuple(order=curve.order, knot=new_U, control_points=new_ctrl_pts)
            candidate_points = np.array([evaluate_bspline_curve(candidate_curve, u) for u in dense_params])
            error = np.max(np.linalg.norm(candidate_points - original_points, axis=1))
            if error < tol:
                U = new_U
                ctrl_pts = new_ctrl_pts
                # Do not increment i to see if we can remove further knots at the same location.
                continue
        # Otherwise, skip this knot.
        i += 1
    return BSplineCurveTuple(order=curve.order, knot=U, control_points=ctrl_pts)


def remove_knots_bspline_surface(surface: BSplineSurfaceTuple, tol: float) -> BSplineSurfaceTuple:
    """
    A similar strategy is used for surfaces: dense sampling over (u,v),
    then for each candidate interior knot (in u and v) a re–interpolated surface is constructed.
    For brevity, here we apply the curve removal process in each parametric direction sequentially.
    """
    # Remove knots in u direction (for each fixed v, remove on the u–curves)
    new_control_net = surface.control_points.copy()
    U = surface.knot_u.copy()
    p = surface.order_u - 1
    u0, u1 = nurbs_interval(U, p)
    n_u = new_control_net.shape[0]
    dense_u = np.linspace(u0, u1, 50)
    # Process each row (fixed v)
    for j in range(new_control_net.shape[1]):
        # Build a BSplineCurve from the row
        curve_row = BSplineCurveTuple(order=surface.order_u, knot=U, control_points=new_control_net[:, j, :])
        curve_row_new = remove_knots_bspline_curve(curve_row, tol)
        U = curve_row_new.knot  # update knot vector (assumed common to all rows)
        new_control_net[:, j, :] = curve_row_new.control_points
    # Now remove knots in v direction (for each fixed u)
    V = surface.knot_v.copy()
    q = surface.order_v - 1
    v0, v1 = nurbs_interval(V, q)
    dense_v = np.linspace(v0, v1, 50)
    for i in range(new_control_net.shape[0]):
        # Build a BSplineCurve from the column
        curve_col = BSplineCurveTuple(order=surface.order_v, knot=V, control_points=new_control_net[i, :, :])
        curve_col_new = remove_knots_bspline_curve(curve_col, tol)
        V = curve_col_new.knot
        new_control_net[i, :, :] = curve_col_new.control_points
    return BSplineSurfaceTuple(
        order_u=surface.order_u,
        order_v=surface.order_v,
        knot_u=U,
        knot_v=V,
        control_points=new_control_net,
    )

'''
# -----------------------
# Offset routines
# -----------------------
def sample_offset_curve_segment(
    curve: NURBSCurveTuple, u0: float, u1: float, d: float, tol: float, p: int
) -> Tuple[List[List[float]], List[float]]:
    """
    Samples the Bézier segment [u0,u1] of the NURBS curve.
    Initially 2*(p+1) sample points are used to estimate a bound on the offset second derivative.
    Then the number of points n_required = max(p+1, ceil((1/ε)^0.34 * sqrt(M/8))) is used.
    """
    n0 = 2 * (p + 1)
    us_initial = np.linspace(u0, u1, n0)
    offset_pts_initial = []
    for u in us_initial:
        ev = evaluate_nurbs_curve(curve, u, d_order=2)
        C = ev["C"]
        C1 = ev["C1"]
        T = C1 / np.linalg.norm(C1)
        # For a planar curve, choose the unit normal as a 90° rotation of T.
        N = np.array([-T[1], T[0]])
        if len(C) > 2:
            N = np.concatenate((N, np.zeros(len(C) - 2)))
        offset_pt = C + d * N
        offset_pts_initial.append(offset_pt)
    offset_pts_initial = np.array(offset_pts_initial)
    du = (u1 - u0) / (n0 - 1)
    M = 0.0
    for i in range(1, n0 - 1):
        second_diff = (offset_pts_initial[i + 1] - 2 * offset_pts_initial[i] + offset_pts_initial[i - 1]) / (du**2)
        norm_second = np.linalg.norm(second_diff)
        if norm_second > M:
            M = norm_second
    pow_val = 0.34
    n_required = int(np.ceil((1.0 / tol) ** pow_val * np.sqrt(M / 8.0)))
    n_required = max(n_required, p + 1)
    if n_required > n0:
        us = np.linspace(u0, u1, n_required)
        offset_pts = []
        for u in us:
            ev = evaluate_nurbs_curve(curve, u, d_order=2)
            C = ev["C"]
            C1 = ev["C1"]
            T = C1 / np.linalg.norm(C1)
            N = np.array([-T[1], T[0]])
            if len(C) > 2:
                N = np.concatenate((N, np.zeros(len(C) - 2)))
            offset_pt = C + d * N
            offset_pts.append(offset_pt)
        return offset_pts, us.tolist()
    else:
        return offset_pts_initial.tolist(), us_initial.tolist()


def offset_nurbs_curve(curve: NURBSCurveTuple, d: float, tol: float, interp_degree: int = 3) -> BSplineCurveTuple:
    """
    Computes the offset of a NURBS curve.
    First special shapes (lines, circles) could be detected and offset exactly.
    Here we assume a free-form offset computed via sampling, interpolation and knot removal.
    """
    p = curve.order - 1
    u_start, u_end = _curve_interval(curve)
    unique_knots = np.unique(curve.knot)
    seg_knots = unique_knots[(unique_knots >= u_start) & (unique_knots <= u_end)]
    segments = []
    for i in range(len(seg_knots) - 1):
        segments.append((seg_knots[i], seg_knots[i + 1]))
    offset_points: List[List[float]] = []
    parameters: List[float] = []
    for idx, seg in enumerate(segments):
        pts, us_seg = sample_offset_curve_segment(curve, seg[0], seg[1], d, tol, p)
        if idx > 0 and np.allclose(offset_points[-1], pts[0]):
            offset_points.extend(pts[1:])
            parameters.extend(us_seg[1:])
        else:
            offset_points.extend(pts)
            parameters.extend(us_seg)
    offset_curve = interpolate_bspline_curve(np.array(offset_points), interp_degree, parameters)
    #offset_curve_reduced = remove_knots_bspline_curve(offset_curve, tol)
    return offset_curve


def offset_nurbs_surface(
    surface: NURBSSurfaceTuple,
    d: float,
    tol: float,
    interp_degrees: Tuple[int, int] = (3, 3),
) -> BSplineSurfaceTuple:
    """
    Computes the offset of a NURBS surface.
    The surface is sampled on a grid (with density computed from the bound on second partials),
    the offset sample points are interpolated by a tensor–product B–spline surface, and
    finally removable knots are eliminated.
    """
    (u_start, u_end), (v_start, v_end) = _surface_interval(surface)
    p = surface.order_u - 1
    q = surface.order_v - 1
    n0_u = 2 * (p + 1)
    n0_v = 2 * (q + 1)
    us_initial = np.linspace(u_start, u_end, n0_u)
    vs_initial = np.linspace(v_start, v_end, n0_v)
    dim = len(surface.control_points[0][0])
    offset_grid_initial = np.zeros((n0_u, n0_v, dim))
    for i, u in enumerate(us_initial):
        for j, v in enumerate(vs_initial):
            ev = evaluate_nurbs_surface(surface, u, v, d_order=2)
            S_pt = ev["S"]
            Su = ev["Su"]
            Sv = ev["Sv"]
            N = np.cross(Su, Sv)
            norm_N = np.linalg.norm(N)
            if norm_N == 0:
                N = np.zeros_like(N)
            else:
                N = N / norm_N
            offset_grid_initial[i, j, :] = S_pt + d * N
    du = (u_end - u_start) / (n0_u - 1)
    dv = (v_end - v_start) / (n0_v - 1)
    M1 = 0.0
    for j in range(n0_v):
        for i in range(1, n0_u - 1):
            diff_u = (offset_grid_initial[i + 1, j, :] - 2 * offset_grid_initial[i, j, :] + offset_grid_initial[i - 1, j, :]) / (du**2)
            M1 = max(M1, np.linalg.norm(diff_u))
    M2 = 0.0
    for i in range(1, n0_u - 1):
        for j in range(1, n0_v - 1):
            diff_uv = (
                offset_grid_initial[i + 1, j + 1, :]
                - offset_grid_initial[i + 1, j - 1, :]
                - offset_grid_initial[i - 1, j + 1, :]
                + offset_grid_initial[i - 1, j - 1, :]
            ) / (4 * du * dv)
            M2 = max(M2, np.linalg.norm(diff_uv))
    M3 = 0.0
    for i in range(n0_u):
        for j in range(1, n0_v - 1):
            diff_v = (offset_grid_initial[i, j + 1, :] - 2 * offset_grid_initial[i, j, :] + offset_grid_initial[i, j - 1, :]) / (dv**2)
            M3 = max(M3, np.linalg.norm(diff_v))
    pow_val = 0.34
    n_required_u = int(np.ceil((1.0 / tol) ** pow_val * np.sqrt((M1 + 2 * M2 + M3) / 8.0)))
    n_required_u = max(n_required_u, p + 1)
    n_required_v = int(np.ceil((1.0 / tol) ** pow_val * np.sqrt((M1 + 2 * M2 + M3) / 8.0)))
    n_required_v = max(n_required_v, q + 1)
    us = np.linspace(u_start, u_end, n_required_u)
    vs = np.linspace(v_start, v_end, n_required_v)
    offset_grid = np.zeros((n_required_u, n_required_v, dim))
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            ev = evaluate_nurbs_surface(surface, u, v, d_order=2)
            S_pt = ev["S"]
            Su = ev["Su"]
            Sv = ev["Sv"]
            N = np.cross(Su, Sv)
            norm_N = np.linalg.norm(N)
            if norm_N == 0:
                N = np.zeros_like(N)
            else:
                N = N / norm_N
            offset_grid[i, j, :] = S_pt + d * N
    bspline_surface = interpolate_bspline_surface(offset_grid, interp_degrees, us, vs)
    #bspline_surface_reduced = remove_knots_bspline_surface(bspline_surface, tol)
    return bspline_surface


# -----------------------
# Example usage:
# -----------------------
if __name__ == "__main__":
    # For example, one can construct a NURBS circle (or any NURBS curve)
    # and then call offset_nurbs_curve. Similarly for surfaces.
    #
    # Here we create a simple planar NURBS curve (for instance a quarter circle)
    # and then compute its offset.
    #
    # NOTE: In a full application these data would be provided by the CAD system.

    # Example NURBS curve: quarter circle in 2D
    order = 3
    knot = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    # Control points and weights for a quarter circle (approximation)
    control_points = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    weights = np.array([1.0, 1 / np.sqrt(2), 1.0], dtype=float)
    nurbs_curve = NURBSCurveTuple(order=order, knot=knot, control_points=control_points, weights=weights)
    # Compute an offset of 0.1 with tolerance 1e-2.
    offset_curve = offset_nurbs_curve(nurbs_curve, d=0.1, tol=1e-3, interp_degree=3)
    print("Offset curve B-spline control points:")
    print(offset_curve.control_points)
    print("Offset curve knot vector:")
    print(offset_curve.knot)

    # A similar procedure would be applied to NURBS surfaces.
