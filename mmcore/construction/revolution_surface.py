import numpy as np
import math

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
        n : int
            n = 2*narcs, so that there are (n+1) control points in the u (revolution) direction.
        U : list of floats
            The knot vector in the u direction.
        Pij : 2D list of numpy arrays
            The control points of the NURBS surface of the torus.
        wij : 2D list of floats
            The weights corresponding to each control point.

    The resulting NURBS surface represents a torus with the standard parameterization:
        X(u,v) = ((R + r*cos(v))*cos(u), (R + r*cos(v))*sin(u), r*sin(v)),
    where u and v are the two parametric directions.
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

    return  (np.array(U)*2*np.pi).tolist(), (np.array(V)*2*np.pi).tolist(), Pij, wij


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
