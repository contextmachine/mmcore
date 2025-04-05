
from ._nurbs_eval import NURBSCurveTuple,NURBSSurfaceTuple,evaluate_nurbs_curve,evaluate_nurbs_surface,nurbs_interval,_surface_interval,_curve_interval

import numpy as np


def extend_nurbs_curve(curve, interval: tuple[float, float], smooth: bool):
    """
    Extend a NURBS curve by constructing extra Bezier segments in homogeneous space.

    This implementation assumes that the input curve is clamped and consists of a single
    Bezier segment over its effective domain [u_start, u_end]. (For a general curve, one
    must first insert knots to decompose it into Bezier segments.)

    The extended curve is built as a concatenation of three segments:
      - a left extension defined on [new_u_start, u_start],
      - the original curve on [u_start, u_end],
      - a right extension defined on [u_end, new_u_end].

    In homogeneous space the extra segments are built so that the endpoint (first and second)
    derivatives match those of the original curve. For a degree-p (order = p+1) Bezier,
    if we denote the homogeneous control points by H[0],...,H[p] (with H[0] at u_start),
    then for the left extension (using a normalized parameter so that the original lies on [0,1]):

         Delta_L = (u_start - new_u_start) / (u_end - u_start)

    and one may set (for p=2):

         L_2 = H[0]
         L_1 = H[0] - (Delta_L/2) * D_left
         L_0 = (if smooth:) 2*L_1 - H[0] + (Delta_L**2/2)*A_left
               (else:) H[0] - Delta_L * D_left

    where
         D_left = 2*(H[1]-H[0])
         A_left = 2*(H[2] - 2*H[1] + H[0])

    A similar construction is used at the right end.

    Finally, the extended homogeneous control points are merged (avoiding duplication of the join)
    and dehomogenized, and a new knot vector is built so that the new parameter domain is
    [new_u_start, new_u_end]. For a quadratic curve one valid choice is:

         knot = [new_u_start, new_u_start, new_u_start,
                 u_start, u_start,
                 u_end, u_end,
                 new_u_end, new_u_end, new_u_end]

    Parameters:
      curve: NURBSCurveTuple with attributes order, knot, control_points, weights.
      interval: (new_u_start, new_u_end) desired extended parameter domain.
      smooth: if True, include second derivative in the extrapolation.

    Returns:
      A new NURBSCurveTuple representing the extended curve.
    """
    p = curve.order - 1  # degree
    # Original effective parameter values:
    orig_knot = curve.knot
    u_start = orig_knot[p]  # left effective value
    u_end = orig_knot[-(p + 1)]  # right effective value
    # For a clamped, single-Bezier, we assume u_start and u_end are the only unique values.

    # --- Convert to homogeneous control points ---
    n = len(curve.control_points)
    H = []
    for i in range(n):
        pt = np.array(curve.control_points[i])
        w = curve.weights[i]
        H.append(np.hstack((pt * w, w)))
    # Now H[0] ... H[p] are the homogeneous control points.

    # --- Reparameterize the original curve to [0,1] ---
    # (We assume u_end > u_start.)
    T = u_end - u_start  # original span
    # In our formulas below the original Bezier is assumed to lie on [0,1].

    # --- Compute endpoint derivatives in homogeneous space ---
    # Left endpoint (at parameter 0):
    D_left = p * (H[1] - H[0])  # first derivative
    if smooth and p >= 2:
        A_left = p * (p - 1) * (H[2] - 2 * H[1] + H[0])
    else:
        A_left = np.zeros_like(H[0])
    # Right endpoint (at parameter 1):
    D_right = p * (H[-1] - H[-2])
    if smooth and p >= 2:
        A_right = p * (p - 1) * (H[-1] - 2 * H[-2] + H[-3])
    else:
        A_right = np.zeros_like(H[-1])

    # --- Determine normalized extension lengths ---
    # We map the original [u_start,u_end] to [0,1]. Then:
    Delta_L = (u_start - interval[0]) / T  # positive number (extension length on left)
    Delta_R = (interval[1] - u_end) / T  # positive number (extension length on right)

    # --- Build left extension (a Bezier segment of degree p) ---
    # It will have p+1 control points L[0],...,L[p] with L[p] = H[0].
    L = [None] * (p + 1)
    L[p] = H[0]
    # For the first derivative: p*(L[p] - L[p-1])/Delta_L = D_left  => L[p-1] = H[0] - (Delta_L/p)*D_left.
    L[p - 1] = H[0] - (Delta_L / p) * D_left
    if p >= 2:
        if smooth:
            # Second derivative: p(p-1)*(L[p] - 2*L[p-1] + L[p-2])/Delta_L^2 = A_left
            L[p - 2] = 2 * L[p - 1] - H[0] + (Delta_L**2 / (p * (p - 1))) * A_left
        else:
            L[p - 2] = H[0] - Delta_L * D_left
        # For degrees higher than 2 one should fill in the remaining control points.
        for i in range(p - 3, -1, -1):
            # Here we simply linearly propagate; a more sophisticated approach is needed for high degree.
            L[i] = L[i + 1]
    elif p == 1:
        L[0] = H[0] - Delta_L * D_left

    # --- Build right extension (a Bezier segment of degree p) ---
    # It will have p+1 control points R[0],...,R[p] with R[0] = H[-1].
    R = [None] * (p + 1)
    R[0] = H[-1]
    R[1] = H[-1] + (Delta_R / p) * D_right
    if p >= 2:
        if smooth:
            R[2] = (
                R[0]
                + 2 * (Delta_R / p) * D_right
                + (Delta_R**2 / (p * (p - 1))) * A_right
            )
        else:
            R[2] = H[-1] + Delta_R * D_right
        for i in range(3, p + 1):
            R[i] = R[i - 1]
    elif p == 1:
        R[1] = H[-1] + Delta_R * D_right

    # --- Merge control points ---
    # Omit the duplicated join endpoints: left extension gives L[0] ... L[p-1],
    # then the original Bezier H[0]...H[p], then right extension R[1] ... R[p].
    ext_H = L[:-1] + H + R[1:]
    # For example, for p=2, this gives 2 + 3 + 2 = 7 control points.

    # --- Build new knot vector ---
    # For a spline of degree p with m control points, there are m+p+1 knots.
    # Here we choose a new knot vector that exactly represents three segments
    # defined on the intervals [interval[0], u_start], [u_start, u_end], [u_end, interval[1]].
    # One acceptable choice (for p=2) is:
    new_knot = (
        [interval[0]] * (p + 1) + [u_start] * p + [u_end] * p + [interval[1]] * (p + 1)
    )
    # (For general p one may use the same pattern.)

    # --- Dehomogenize the extended control points ---
    ext_ctrl_pts = []
    ext_weights = []
    for H_i in ext_H:
        w = H_i[-1]
        pt = H_i[:-1] / w
        ext_ctrl_pts.append(pt)
        ext_weights.append(w)

    # --- Build and return new NURBSCurveTuple ---
    new_curve = NURBSCurveTuple(
        order=curve.order,
        knot=np.array(new_knot, dtype=float),
        control_points=np.array(ext_ctrl_pts, dtype=float),
        weights=np.array(ext_weights, dtype=float),
    )
    return new_curve


def extend_nurbs_surface(
    surface,
    interval_u: tuple[float, float],
    interval_v: tuple[float, float],
    smooth: bool,
):
    """
    Extend a NURBS surface by extending its control net in both u and v directions.

    The strategy is analogous to the curve case. In each parametric direction the boundary curves
    (i.e. the first and last rows and columns) are interpreted as Bezier curves (by assuming the surface
    is defined as a single patch in that direction), and are extended by constructing extra Bezier segments
    in homogeneous space that satisfy the derivative conditions.

    Then the new control net is formed by adding the extra rows/columns on each side and the new knot vectors
    in u and v are built from the corresponding extension intervals.

    Parameters:
      surface: NURBSSurfaceTuple with attributes order_u, order_v, knot_u, knot_v,
               control_points (2D array-like, with shape (nu, nv)), and weights (2D array-like).
      interval_u: (new_u_start, new_u_end) desired u-domain.
      interval_v: (new_v_start, new_v_end) desired v-domain.
      smooth: if True, use quadratic (second-derivative) extrapolation; else, linear.

    Returns:
      A new NURBSSurfaceTuple representing the extended surface.

    Note: This implementation assumes the original surface is a single Bezier patch in each parametric direction.
    For a general surface one would first need to perform knot insertion.
    """
    # Degrees in u and v
    p = surface.order_u - 1
    q = surface.order_v - 1
    # Original effective domains:
    U_orig = surface.knot_u
    V_orig = surface.knot_v
    u_start = U_orig[p]
    u_end = U_orig[-(p + 1)]
    v_start = V_orig[q]
    v_end = V_orig[-(q + 1)]
    T_u = u_end - u_start
    T_v = v_end - v_start

    # --- Convert control net to homogeneous coordinates ---
    nu = len(surface.control_points)
    nv = len(surface.control_points[0])
    H_net = []
    for i in range(nu):
        row = []
        for j in range(nv):
            pt = np.array(surface.control_points[i][j])
            w = surface.weights[i, j]
            row.append(np.hstack((pt * w, w)))
        H_net.append(row)
    # For simplicity we assume the surface is a tensor product Bezier patch.

    # --- Extend in u-direction (rows) ---
    # For each fixed column, extract the curve in u and extend it.
    ext_rows = []
    for j in range(nv):
        # Extract the j-th column (as a list of homogeneous points).
        H_col = [H_net[i][j] for i in range(nu)]
        # Compute left and right derivatives in u (treating the column as a Bezier curve in u).
        D_left_u = p * (H_col[1] - H_col[0])
        if smooth and p >= 2:
            A_left_u = p * (p - 1) * (H_col[2] - 2 * H_col[1] + H_col[0])
        else:
            A_left_u = np.zeros_like(H_col[0])
        D_right_u = p * (H_col[-1] - H_col[-2])
        if smooth and p >= 2:
            A_right_u = p * (p - 1) * (H_col[-1] - 2 * H_col[-2] + H_col[-3])
        else:
            A_right_u = np.zeros_like(H_col[-1])
        Delta_L_u = (u_start - interval_u[0]) / T_u
        Delta_R_u = (interval_u[1] - u_end) / T_u
        # Build left extension for this column:
        L = [None] * (p + 1)
        L[p] = H_col[0]
        L[p - 1] = H_col[0] - (Delta_L_u / p) * D_left_u
        if p >= 2:
            if smooth:
                L[p - 2] = (
                    2 * L[p - 1] - H_col[0] + (Delta_L_u**2 / (p * (p - 1))) * A_left_u
                )
            else:
                L[p - 2] = H_col[0] - Delta_L_u * D_left_u
            for i in range(p - 3, -1, -1):
                L[i] = L[i + 1]
        elif p == 1:
            L[0] = H_col[0] - Delta_L_u * D_left_u
        # Build right extension for this column:
        R = [None] * (p + 1)
        R[0] = H_col[-1]
        R[1] = H_col[-1] + (Delta_R_u / p) * D_right_u
        if p >= 2:
            if smooth:
                R[2] = (
                    R[0]
                    + 2 * (Delta_R_u / p) * D_right_u
                    + (Delta_R_u**2 / (p * (p - 1))) * A_right_u
                )
            else:
                R[2] = H_col[-1] + Delta_R_u * D_right_u
            for i in range(3, p + 1):
                R[i] = R[i - 1]
        elif p == 1:
            R[1] = H_col[-1] + Delta_R_u * D_right_u
        # Merge for this column:
        ext_col = L[:-1] + H_col + R[1:]
        ext_rows.append(ext_col)
    # Now ext_rows is a list (length = nv) of extended columns (each of length n_u_ext).
    nuext = len(ext_rows[0])
    # Reassemble extended control net in u (transpose ext_rows)
    H_net_ext_u = []
    for i in range(nuext):
        row = []
        for j in range(nv):
            row.append(ext_rows[j][i])
        H_net_ext_u.append(row)

    # --- Extend in v-direction (columns) ---
    # Now treat each row of H_net_ext_u as a Bezier curve in v.
    nuext = len(H_net_ext_u)
    ext_net = []
    for i in range(nuext):
        H_row = H_net_ext_u[i]
        D_left_v = q * (H_row[1] - H_row[0])
        if smooth and q >= 2:
            A_left_v = q * (q - 1) * (H_row[2] - 2 * H_row[1] + H_row[0])
        else:
            A_left_v = np.zeros_like(H_row[0])
        D_right_v = q * (H_row[-1] - H_row[-2])
        if smooth and q >= 2:
            A_right_v = q * (q - 1) * (H_row[-1] - 2 * H_row[-2] + H_row[-3])
        else:
            A_right_v = np.zeros_like(H_row[-1])
        Delta_L_v = (v_start - interval_v[0]) / T_v
        Delta_R_v = (interval_v[1] - v_end) / T_v
        # Left extension in v:
        L = [None] * (q + 1)
        L[q] = H_row[0]
        L[q - 1] = H_row[0] - (Delta_L_v / q) * D_left_v
        if q >= 2:
            if smooth:
                L[q - 2] = (
                    2 * L[q - 1] - H_row[0] + (Delta_L_v**2 / (q * (q - 1))) * A_left_v
                )
            else:
                L[q - 2] = H_row[0] - Delta_L_v * D_left_v
            for k in range(q - 3, -1, -1):
                L[k] = L[k + 1]
        elif q == 1:
            L[0] = H_row[0] - Delta_L_v * D_left_v
        # Right extension in v:
        R = [None] * (q + 1)
        R[0] = H_row[-1]
        R[1] = H_row[-1] + (Delta_R_v / q) * D_right_v
        if q >= 2:
            if smooth:
                R[2] = (
                    R[0]
                    + 2 * (Delta_R_v / q) * D_right_v
                    + (Delta_R_v**2 / (q * (q - 1))) * A_right_v
                )
            else:
                R[2] = H_row[-1] + Delta_R_v * D_right_v
            for k in range(3, q + 1):
                R[k] = R[k - 1]
        elif q == 1:
            R[1] = H_row[-1] + Delta_R_v * D_right_v
        # Merge for this row:
        ext_row = L[:-1] + H_row + R[1:]
        ext_net.append(ext_row)
    # ext_net now is the fully extended homogeneous control net.
    nuext = len(ext_net)
    nvext = len(ext_net[0])

    # --- Build new knot vectors ---
    # For u: use
    knot_u_new = (
        [interval_u[0]] * (p + 1)
        + [u_start] * p
        + [u_end] * p
        + [interval_u[1]] * (p + 1)
    )
    # For v: use
    knot_v_new = (
        [interval_v[0]] * (q + 1)
        + [v_start] * q
        + [v_end] * q
        + [interval_v[1]] * (q + 1)
    )

    # --- Dehomogenize the extended net ---
    ext_ctrl_pts = []
    ext_weights = []
    for i in range(nuext):
        row_pts = []
        row_ws = []
        for j in range(nvext):
            H_ij = ext_net[i][j]
            w = H_ij[-1]
            pt = H_ij[:-1] / w
            row_pts.append(pt)
            row_ws.append(w)
        ext_ctrl_pts.append(row_pts)
        ext_weights.append(row_ws)

    new_surface = NURBSSurfaceTuple(
        order_u=surface.order_u,
        order_v=surface.order_v,
        knot_u=np.array(knot_u_new, dtype=float),
        knot_v=np.array(knot_v_new, dtype=float),
        control_points=np.array(ext_ctrl_pts, dtype=float),
        weights=np.array(ext_weights, dtype=float),
    )
    return new_surface

if __name__ == '__main__':
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
    t0,t1=_curve_interval(nurbs_curve)
    res=extend_nurbs_curve(nurbs_curve,(t0-1.0,t1+1.0),True)

