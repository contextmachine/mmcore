import numpy as np
import math

from mmcore.geom._nurbs_construct import circle
from mmcore.geom._nurbs_eval import NURBSSurfaceTuple
__all__=["make_revolved_surf"]
def point_to_line(S, T, P):
    """
    Given a line defined by point S and direction T, compute the projection of point P
    onto the line.

    Parameters:
        S : numpy array of shape (3,)
            A point on the line.
        T : numpy array of shape (3,)
            The direction vector of the line (need not be normalized).
        P : numpy array of shape (3,)
            The point to be projected.

    Returns:
        O : numpy array of shape (3,)
            The projection of P onto the line (i.e. the closest point on the line).
    """
    T_norm = T / np.linalg.norm(T)
    # Projection formula: O = S + dot(P-S, T_norm)*T_norm
    return S + np.dot(P - S, T_norm) * T_norm


def vec_norm_and_normalize(v):
    """
    Normalizes the vector v in place and returns its original norm.

    Parameters:
        v : numpy array of shape (3,)
            The vector to normalize.

    Returns:
        norm : float
            The original length of v.

    Note:
        After calling this function, the vector v is modified so that it becomes a unit vector.
    """
    norm = np.linalg.norm(v)
    if norm != 0:
        v /= norm
    return norm


def intersect_3d_lines(P0, d0, P1, d1):
    """
    Computes the intersection of two (coplanar) lines in 3D.
    The two lines are given in parametric form:

        L0: P0 + t * d0
        L1: P1 + s * d1

    This function assumes that the two lines intersect (which is true for our
    construction) and selects two coordinates (the pair where the cross product
    of d0 and d1 is largest) to solve for the parameter.

    Parameters:
        P0 : numpy array of shape (3,)
            A point on the first line.
        d0 : numpy array of shape (3,)
            The direction vector of the first line.
        P1 : numpy array of shape (3,)
            A point on the second line.
        d1 : numpy array of shape (3,)
            The direction vector of the second line.

    Returns:
        X : numpy array of shape (3,)
            The intersection point.
    """
    cp = np.cross(d0, d1)
    abs_cp = np.abs(cp)
    max_index = np.argmax(abs_cp)
    if max_index == 0:
        a0, a1 = 1, 2  # use y and z coordinates
    elif max_index == 1:
        a0, a1 = 0, 2  # use x and z coordinates
    else:
        a0, a1 = 0, 1  # use x and y coordinates

    # The equations are:
    #    P0[a] + t*d0[a] = P1[a] + s*d1[a]   for a = a0 and a1.
    A = np.array([[d0[a0], -d1[a0]], [d0[a1], -d1[a1]]])
    b = np.array([P1[a0] - P0[a0], P1[a1] - P0[a1]])
    sol = np.linalg.solve(A, b)
    t = sol[0]
    return P0 + t * d0


def make_revolved_surf(S, T, theta, m, Pj, wj):
    """
    Constructs a NURBS surface of revolution by revolving a generating curve.

    The generating curve is given by its control points (Pj) and weights (wj).
    The surface is created by revolving this curve about the line through S in
    the direction T through an angle theta (in degrees).

    Parameters:
        S : numpy array of shape (3,)
            A point on the axis of revolution.
        T : numpy array of shape (3,)
            The direction vector of the axis of revolution (need not be normalized).
        theta : float
            The total angle of revolution in degrees.
        m : int
            The highest index of the control points for the generating curve.
            (There are m+1 control points.)
        Pj : list of numpy arrays
            The control points of the generating curve (each a 3D point).
        wj : list of floats
            The weights corresponding to each control point in Pj.

    Returns:
        n : int
            n = 2*narcs, i.e. there are (n+1) control points in the revolution (u) direction.
        U : list of floats
            The knot vector in the u direction.
        Pij : 2D list of numpy arrays
            The control points of the resulting NURBS surface. It is indexed as Pij[i][j],
            where i = 0,...,n and j = 0,...,m.
        wij : 2D list of floats
            The corresponding weights for Pij.
    """
    # Determine the number of circular arcs (narcs) based on theta.
    if theta <= 90.0:
        narcs = 1
    elif theta <= 180.0:
        narcs = 2
    elif theta <= 270.0:
        narcs = 3
    else:
        narcs = 4

    # Set the interior knots in the knot vector U.
    j_index = 3 + 2 * (narcs - 1)
    U = [None] * (j_index + 3)
    if narcs == 2:
        U[3] = 0.5
        U[4] = 0.5
    elif narcs == 3:
        U[3] = 1.0 / 3.0
        U[4] = 1.0 / 3.0
        U[5] = 2.0 / 3.0
        U[6] = 2.0 / 3.0
    elif narcs == 4:
        U[3] = 0.25
        U[4] = 0.25
        U[5] = 0.5
        U[6] = 0.5
        U[7] = 0.75
        U[8] = 0.75
    j_temp = j_index
    for i in range(3):
        U[i] = 0.0
        U[j_temp] = 1.0
        j_temp += 1

    n = 2 * narcs

    # Compute the arc angle increment.
    dtheta = theta / narcs  # degrees per arc segment
    dtheta_rad = math.radians(dtheta)  # convert to radians
    wm = math.cos(dtheta_rad / 2.0)  # weight for the mid-arc control points

    # Precompute cosine and sine values for each arc segment.
    cosines = [0.0] * (narcs + 1)
    sines = [0.0] * (narcs + 1)
    angle_rad = 0.0
    for i in range(1, narcs + 1):
        angle_rad += dtheta_rad
        cosines[i] = math.cos(angle_rad)
        sines[i] = math.sin(angle_rad)

    # Allocate arrays for the surface control net.
    num_rows = n + 1  # u-direction (revolution direction)
    num_cols = m + 1  # v-direction (generating curve direction)
    Pij = [[None for j in range(num_cols)] for i in range(num_rows)]
    wij = [[None for j in range(num_cols)] for i in range(num_rows)]

    # Loop over each generating curve control point.
    for j in range(num_cols):
        # Compute the projection of Pj[j] onto the axis (S, T).
        O = point_to_line(S, T, Pj[j])
        X = Pj[j] - O  # vector from the axis to the generating point
        r_local = vec_norm_and_normalize(X)  # original length before normalization
        Y = np.cross(T, X)  # perpendicular direction in the plane
        # The first control point (u=0) is the original generating point.
        Pij[0][j] = Pj[j].copy()
        wij[0][j] = wj[j]

        # Set up for constructing the circular arc in the u-direction.
        P0 = Pj[j].copy()  # starting point for this arc
        TO = Y.copy()  # initial tangent direction (in the plane perpendicular to T)
        index = 1
        for i in range(1, narcs + 1):
            # Compute the rotated point P2.
            P2 = O + r_local * (cosines[i] * X + sines[i] * Y)
            Pij[index+1][j] = P2.copy()
            wij[index+1][j] = wj[j]
            # Compute the tangent direction at P2.
            T2 = -sines[i] * X + cosines[i] * Y
            # Insert an extra control point computed as the intersection of two lines.
            Pij[index ][j] = intersect_3d_lines(P0, TO, P2, T2)
            wij[index ][j] = wm * wj[j]
            index += 2
            if i < narcs:
                P0 = P2.copy()
                TO = T2.copy()

    return n, U, Pij, wij



def make_revolved_surface(profile_curve, axis, interval):
    """
    Create a NURBS surface of revolution by rotating a profile (generating) curve about an axis.

    Parameters:
        profile_curve : NURBSCurveTuple
            The generating (or profile) curve. It is assumed to be defined in space and to represent
            the null section at rotation angle "start_angle". It should have attributes:
                - control_points: a numpy array of shape (num_profile, 3)
                - weights: a numpy array of shape (num_profile,)
                - knot: the knot vector in the v-direction (1D numpy array)
                - order: the order (degree + 1) of the curve.
        axis : tuple/list of two points
            Two points defining the axis of revolution. The first point is the base (S) and the second
            point gives the direction (from which a normalized vector T is computed).
        interval : tuple (start_angle, end_angle) in radians
            The angular interval over which the profile curve is revolved.
            (For a full revolution use (0, 2*pi).)

    Returns:
        NURBSSurfaceTuple object representing the revolved surface.
    """
    # Compute the axis: S is the base point and T is the normalized direction.
    S = np.array(axis[0], dtype=float)
    axis_dir = np.array(axis[1], dtype=float) - S
    T = axis_dir / np.linalg.norm(axis_dir)

    # Unpack the interval.
    start_angle, end_angle = interval
    total_angle = end_angle - start_angle
    if total_angle <= 0 or total_angle > 2 * math.pi:
        raise ValueError("Interval must be positive and no more than 2*pi radians.")

    # Determine the number of arc segments (narcs) based on the total revolution angle.
    # We require each arc segment to be at most 90° (pi/2 radians).
    if total_angle <= math.pi / 2:
        narcs = 1
    elif total_angle <= math.pi:
        narcs = 2
    elif total_angle <= (3 * math.pi / 2):
        narcs = 3
    else:
        narcs = 4

    dtheta = total_angle / narcs
    wm = math.cos(dtheta / 2.0)  # weight for the mid-arc control points

    # Precompute angles for the arc segments (including the starting angle)
    angles = [start_angle + i * dtheta for i in range(narcs + 1)]
    cosines = [math.cos(a) for a in angles]
    sines = [math.sin(a) for a in angles]

    # Extract the profile curve data.
    Pj = profile_curve.control_points  # Expected shape: (num_profile, 3)
    wj = profile_curve.weights  # Expected shape: (num_profile,)
    num_profile = Pj.shape[0]

    # In the revolution direction (u-direction) we will have (2*narcs + 1) rows.
    num_rows = 2 * narcs + 1

    # Set up 2D grids (lists) for the surface control points and weights.
    Pij = [[None for _ in range(num_profile)] for _ in range(num_rows)]
    wij = [[None for _ in range(num_profile)] for _ in range(num_rows)]

    # For each control point of the profile curve, build its "swept" curve.
    for j in range(num_profile):
        P_orig = np.array(Pj[j], dtype=float)
        weight_profile = wj[j]
        # Find the projection of the profile point onto the axis.
        O = point_to_line(S, T, P_orig)
        X = P_orig - O
        r = np.linalg.norm(X)
        if r < 1e-8:
            # The point lies on the axis and remains unchanged upon revolution.
            for i in range(num_rows):
                Pij[i][j] = P_orig.copy()
                wij[i][j] = weight_profile
        else:
            # Compute unit vector from the axis to the profile point.
            Ux = X / r
            # Determine Y so that {Ux, Y} form a right-handed basis in the plane perpendicular to T.
            Y = np.cross(T, Ux)
            if np.linalg.norm(Y) < 1e-8:
                # Fallback in the unlikely event that T and Ux are collinear.
                Y = np.cross(T, [1, 0, 0]) if abs(T[0]) < 0.9 else np.cross(T, [0, 1, 0])
            Y = Y / np.linalg.norm(Y)

            # The profile curve is assumed to correspond to the start angle.
            P0 = O + r * (math.cos(start_angle) * Ux + math.sin(start_angle) * Y)
            Pij[0][j] = P0.copy()
            wij[0][j] = weight_profile
            # Compute the tangent direction at P0 with respect to rotation.
            T0 = -math.sin(start_angle) * Ux + math.cos(start_angle) * Y
            index = 1
            for i in range(1, narcs + 1):
                # Compute the control point at the end of the current arc segment.
                P2 = O + r * (cosines[i] * Ux + sines[i] * Y)
                Pij[index + 1][j] = P2.copy()
                wij[index + 1][j] = weight_profile
                # Compute the tangent direction at P2.
                T2 = -sines[i] * Ux + cosines[i] * Y
                # Compute the intermediate control point as the intersection of the tangent lines.
                P1 = intersect_3d_lines(P0, T0, P2, T2)
                Pij[index][j] = P1.copy()
                wij[index][j] = wm * weight_profile
                index += 2
                # Prepare for the next arc segment if any.
                if i < narcs:
                    P0 = P2.copy()
                    T0 = T2.copy()

    # Build the knot vector in the u direction.
    # For a quadratic curve (degree 2, order 3), the u-direction knot vector is constructed in clamped form.
    U = [0.0, 0.0, 0.0]
    for i in range(1, narcs):
        knot_val = i / float(narcs)
        U.extend([knot_val, knot_val])
    U.extend([1.0, 1.0, 1.0])
    knot_u = np.array(U, dtype=float)

    # Use the original profile curve's knot vector and order in the v direction.
    knot_v = profile_curve.knot
    order_u = 3  # Quadratic in the revolution (u) direction.
    order_v = profile_curve.order

    # Convert the control net and weights (currently stored as lists of lists)
    # into numpy arrays. The control net will have shape (num_rows, num_profile, 3)
    control_points = np.empty((num_rows, num_profile, 3), dtype=float)
    weights = np.empty((num_rows, num_profile), dtype=float)
    for i in range(num_rows):
        for j in range(num_profile):
            control_points[i, j, :] = Pij[i][j]
            weights[i, j] = wij[i][j]

    return NURBSSurfaceTuple(order_u, order_v, knot_u, knot_v, control_points, weights)

def make_torus(R, r):
    """
    Constructs a torus centered at the origin with major radius R and tube radius r.

    The torus is created by revolving a circular generating curve about the z-axis.
    The generating curve is taken to be a full circle in the x-z plane, with its center
    at (R, 0, 0) and radius r. This generating circle is represented as a closed NURBS curve
    with 9 control points (the first and last points coincide) and corresponding weights.

    Parameters:
        R : float
            The major radius of the torus (distance from the center of the tube to the z-axis).
        r : float
            The tube (minor) radius of the torus.

    Returns:
        NurbsSurfaceTuple


    """
    # Use the z-axis as the axis of revolution.
    S = np.array([0.0, 0.0, 0.0])
    T = np.array([0.0, 0.0, 1.0])
    theta = 360.0  # full revolution

    # Define the generating circle in the x-z plane.
    # A full circle is typically represented by 9 control points (with the first equal to the last)
    # and corresponding weights. For a circle of radius 1 (centered at the origin) the standard
    # control points (in 2D) and weights are:
    #   P0 = (1, 0),     w0 = 1
    #   P1 = (1, 1),     w1 = 1/sqrt(2)
    #   P2 = (0, 1),     w2 = 1
    #   P3 = (-1, 1),    w3 = 1/sqrt(2)
    #   P4 = (-1, 0),    w4 = 1
    #   P5 = (-1, -1),   w5 = 1/sqrt(2)
    #   P6 = (0, -1),    w6 = 1
    #   P7 = (1, -1),    w7 = 1/sqrt(2)
    #   P8 = (1, 0),     w8 = 1
    # For our generating circle in the x-z plane with center (R,0,0) and radius r, we map (x,z) accordingly.
    sqrt2_inv = 1.0 / math.sqrt(2)
    P0 = np.array([R + r, 0.0, 0.0])
    P1 = np.array([R + r, 0.0, r])
    P2 = np.array([R, 0.0, r])
    P3 = np.array([R - r, 0.0, r])
    P4 = np.array([R - r, 0.0, 0.0])
    P5 = np.array([R - r, 0.0, -r])
    P6 = np.array([R, 0.0, -r])
    P7 = np.array([R + r, 0.0, -r])
    P8 = np.array([R + r, 0.0, 0.0])
    Pj = [P0, P1, P2, P3, P4, P5, P6, P7, P8]
    wj = [1.0, sqrt2_inv, 1.0, sqrt2_inv, 1.0, sqrt2_inv, 1.0, sqrt2_inv, 1.0]
    m = len(Pj) - 1  # highest index (8 in this case)

    # Create the v-direction (generating curve) knot vector.
    # For a degree-2 NURBS curve with 9 control points (m=8), the knot vector has length m+2+1 = 12.
    # Since the circle is divided into 4 quarter arcs, the standard knot vector is:
    V = [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0]

    # Construct the torus as a surface of revolution.
    n, U, Pij, wij = make_revolved_surf(S, T, theta, m, Pj, wj)

    return  NURBSSurfaceTuple(3,3,(np.array(U)*2*np.pi).tolist(), (np.array(V)*2*np.pi).tolist(), Pij, wij)


def torus(major_radius=1.0, minor_radius=0.25,
          start_angle=0.0, end_angle=2 * math.pi,
          center=None, normal=None, xaxis=None, yaxis=None):
    """
    Generates a torus as a NURBS surface.

    Parameters:
      major_radius : float
          The radius of the major (sweeping) circle: the distance from the torus center
          to the center of the tube.
      minor_radius : float
          The radius of the tube (i.e. the profile curve).
      start_angle, end_angle : float
          The angular interval (in radians) over which the torus is generated.
          For a full torus use (0, 2*pi).
      center : array-like of 3 floats, optional
          The center of the torus (i.e. the center of the major circle). Defaults to [0,0,0].
      normal : array-like of 3 floats, optional
          The normal vector defining the plane of the major circle.
      xaxis, yaxis : array-like of 3 floats, optional
          The in‑plane coordinate system for the major circle. If provided, these are used directly.
          Otherwise, if a normal is provided, an orthonormal basis is constructed.
          If neither is provided, defaults to the xy‑plane.

    Returns:
      A NURBSSurfaceTuple representing the torus.

    Process:
      1. Build the axis for revolution from the torus center along the given normal.
      2. Construct the profile (tube) circle. Its center is computed as:
             profile_center = center + major_radius * xaxis
         and its normal is calculated as the normalized vector (profile_center - center),
         so that the profile circle’s plane is perpendicular to the major circle’s plane.
      3. Generate the revolved surface from the profile curve.
    """
    # --- Determine the torus coordinate system ---
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.array(center, dtype=float)

    # Option 1: use provided xaxis and yaxis if available.
    if (xaxis is not None) and (yaxis is not None):
        xaxis = np.array(xaxis, dtype=float)
        yaxis = np.array(yaxis, dtype=float)
        xaxis = xaxis / np.linalg.norm(xaxis)
        yaxis = yaxis / np.linalg.norm(yaxis)
    # Option 2: if normal is provided, generate an in‑plane basis.
    elif normal is not None:
        normal = np.array(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        # Construct an in‑plane xaxis.
        if abs(normal[0]) < 1e-6 and abs(normal[1]) < 1e-6:
            xaxis = np.array([1.0, 0.0, 0.0])
        else:
            xaxis = np.cross([0, 0, 1], normal)
            if np.linalg.norm(xaxis) < 1e-6:
                xaxis = np.array([1.0, 0.0, 0.0])
            else:
                xaxis = xaxis / np.linalg.norm(xaxis)
        # yaxis is chosen to complete the right-handed system.
        yaxis = np.cross(normal, xaxis)
        yaxis = yaxis / np.linalg.norm(yaxis)
    else:
        # Default to the xy-plane.
        xaxis = np.array([1.0, 0.0, 0.0])
        yaxis = np.array([0.0, 1.0, 0.0])
        normal = np.cross(xaxis, yaxis)
        normal = normal / np.linalg.norm(normal)

    # --- Build the axis for the revolution ---
    # The torus is generated by revolving the profile curve about an axis that passes through 'center'
    # in the direction of 'normal'.
    S = center.copy()  # base point on the axis
    T = normal.copy()  # direction of the axis (normalized)
    axis = (S, S + T)  # the axis expressed as two points

    # --- Generate the profile (tube) curve ---
    # The profile curve (a circle) is positioned at:
    profile_center = center + major_radius * xaxis

    # CORRECT THE PROFILE CIRCLE ORIENTATION:
    # The profile circle’s plane should be perpendicular to the radial direction from
    # the torus center to the profile center. Hence its normal is:

    profile_normal = -yaxis

    # Choose an in‑plane basis for the profile circle.
    # A natural choice is to let the first in‑plane direction be T (the torus’s axis);
    # then the second is the cross of profile_normal and T.
    profile_xaxis = T.copy()
    profile_yaxis = np.cross(profile_normal, T)
    profile_yaxis = profile_yaxis / np.linalg.norm(profile_yaxis)

    # Now generate the profile curve (a full circle representing the tube cross‑section)
    profile_curve = circle(radius=minor_radius,
                           start_angle=0.0,
                           end_angle=2 * math.pi,
                           center=profile_center,
                           normal=profile_normal,  # Use the computed profile normal.
                           xaxis=profile_xaxis,
                           yaxis=profile_yaxis)

    # --- Create the torus as a revolved surface ---
    # The profile is swept about the torus axis using the supplied angular interval.
    torus_surface = make_revolved_surface(profile_curve, axis, (start_angle, end_angle))

    return torus_surface


# ==========================
# Example usage (for testing)
# ==========================
if __name__ == "__main__":

    # Example parameters for the torus:
    R = 3.0  # major radius
    r = 1.0  # tube (minor) radius

    n, U, V, Pij, wij = make_torus(R, r)
    print(len(Pij))
    print("Torus NURBS Surface Parameters:")
    print("n (u-direction control net index) =", n)
    print("U (knot vector in u direction) =", U)
    print("V (knot vector in v direction) =", V)

    print("\nControl Points (Pij):")
    for i, row in enumerate(Pij):
        row_str = ", ".join([f"({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})" for pt in row])
        print(f"Row {i}: {row_str}")
    print("\nWeights (wij):")
    for i, row in enumerate(wij):
        row_str = ", ".join([f"{w:.3f}" for w in row])
        print(f"Row {i}: {row_str}")
