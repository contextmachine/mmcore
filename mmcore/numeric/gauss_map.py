# mmcore/numeric/gauss_map.py
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog

from mmcore.geom.nurbs import NURBSSurface, subdivide_surface, decompose_surface

from mmcore.numeric.algorithms.quicksort import unique

from mmcore.numeric.monomial import bezier_to_monomial, monomial_to_bezier
from mmcore.numeric.vectors import unit, scalar_dot, scalar_norm
from mmcore.numeric.algorithms.cygjk import gjk

from scipy.spatial import ConvexHull
def convex_hull(pts):
    return np.array(pts)[ConvexHull(np.array(pts),qhull_options='QJ' ).vertices]

def is_flat(surf, u_min, u_max, v_min, v_max, tolerance=1e-3):

    corner_points = [surf(u_min, v_min), surf(u_min, v_max), surf(u_max, v_min), surf(u_max, v_max)]
    center_point = surf((u_min + u_max) / 2, (v_min + v_max) / 2)

    # Compute the plane defined by the three corner points
    normal = np.cross(corner_points[1] - corner_points[0], corner_points[2] - corner_points[0])
    normal = normal / scalar_norm(normal)
    d = -scalar_dot(normal, corner_points[0])

    # Check the distance of the center point from the plane
    distance = np.abs(scalar_dot(normal, center_point) + d)
    # Define an appropriate tolerance for flatness

    return distance < tolerance


def compute_partial_derivative(coeffs, variable):
    """Compute partial derivative of monomial coefficients."""
    n, m, dim = coeffs.shape
    deriv = np.zeros((n, m, dim))
    if variable == "u":
        for i in range(1, n):
            deriv[i - 1, :, :] = i * coeffs[i, :, :]
    elif variable == "v":
        for j in range(1, m):
            deriv[:, j - 1, :] = j * coeffs[:, j, :]
    return deriv


def cross_product(a, b):
    """Compute cross product of two 3D polynomial patches."""
    n, m, _ = a.shape
    result = np.zeros((2 * n - 1, 2 * m - 1, 3))
    for i in range(n):
        for j in range(m):
            for k in range(n):
                for l in range(m):
                    result[i + k, j + l, 0] += (
                            a[i, j, 1] * b[k, l, 2] - a[i, j, 2] * b[k, l, 1]
                    )
                    result[i + k, j + l, 1] += (
                            a[i, j, 2] * b[k, l, 0] - a[i, j, 0] * b[k, l, 2]
                    )
                    result[i + k, j + l, 2] += (
                            a[i, j, 0] * b[k, l, 1] - a[i, j, 1] * b[k, l, 0]
                    )

    return result


def compute_gauss_map(control_points):
    """Compute the Gauss map for a Bézier patch with degree elevation."""
    F = bezier_to_monomial(control_points)
    Fu = compute_partial_derivative(F, "u")
    Fv = compute_partial_derivative(F, "v")

    N = cross_product(Fu, Fv)

    # N_normalized = normalize_polynomial(N)
    # print(N_normalized)
    # N_normalized[np.isnan(N_normalized)]=0.
    gauss_map = monomial_to_bezier(N)

    return gauss_map


def is_bezier(surface: NURBSSurface):
    kv, ku = unique(surface.knots_u).shape[0], unique(surface.knots_u).shape[0]
    if kv.shape[0] < 2 or ku.shape[0] < 2:
        raise ValueError("Degenerated patch")

    return kv.shape[0] == 2 and ku.shape[0] == 2

def compute_gauss_mapw(control_points, weights=None):
    """Compute the Gauss map for a rational Bézier patch."""
    if weights is not None:
        # Convert to homogeneous coordinates
        control_points = control_points * weights[:, :, np.newaxis]
        control_points = np.concatenate([control_points, weights[:, :, np.newaxis]], axis=-1)

    F = bezier_to_monomial(control_points)
    Fu = compute_partial_derivative(F, "u")
    Fv = compute_partial_derivative(F, "v")

    if weights is not None:
        # Handle rational case
        w = F[:, :, -1]
        Fu = Fu[:, :, :3] * w[:, :, np.newaxis] - F[:, :, :3] * Fu[:, :, -1:]
        Fv = Fv[:, :, :3] * w[:, :, np.newaxis] - F[:, :, :3] * Fv[:, :, -1:]
        Fu = Fu / (w ** 2)[:, :, np.newaxis]
        Fv = Fv / (w ** 2)[:, :, np.newaxis]

    N = cross_product(Fu, Fv)
    #N_normalized = normalize_polynomial(N)
    gauss_map = monomial_to_bezier(N)

    return gauss_map

def normalize_polynomial(v, epsilon=1e-10):
    """Normalize a 3D vector polynomial with improved stability."""
    norm_squared = v[:, :, 0] ** 2 + v[:, :, 1] ** 2 + v[:, :, 2] ** 2
    max_norm = np.max(norm_squared)
    if max_norm < epsilon:
        return np.zeros_like(v)
    norm = np.sqrt(norm_squared / max_norm)
    return v / (norm[:, :, np.newaxis] * np.sqrt(max_norm))


import numpy as np


# --- Helper functions for polynomial arithmetic in two variables ---

def poly_mult_2d(A, B):
    """
    Multiply two bivariate polynomials A(u,v) and B(u,v) given in the (power) monomial form.
    A: NumPy array of shape (deg_u_A+1, deg_v_A+1) (or with extra trailing dimensions)
    B: NumPy array of shape (deg_u_B+1, deg_v_B+1)
    Returns the product as a NumPy array of shape
        (deg_u_A+deg_u_B+1, deg_v_A+deg_v_B+1) (and extra dimensions preserved).
    """
    a_rows, a_cols = A.shape[:2]
    b_rows, b_cols = B.shape[:2]
    out_rows = a_rows + b_rows - 1
    out_cols = a_cols + b_cols - 1

    # If A and B are scalar (or have last dim = 1) this works component–wise.
    if A.ndim == 2:
        C = np.zeros((out_rows, out_cols))
        for i in range(a_rows):
            for j in range(a_cols):
                C[i:i + b_rows, j:j + b_cols] += A[i, j] * B
    else:
        d = A.shape[2]
        C = np.zeros((out_rows, out_cols, d))
        for i in range(a_rows):
            for j in range(a_cols):
                C[i:i + b_rows, j:j + b_cols, :] += A[i, j, :] * B  # Here we assume B is scalar (or with d=1)
    return C


def poly_subtract(A, B):
    """
    Subtract two polynomials in monomial form.
    If their coefficient arrays have different shapes, the smaller is padded (in the higher degrees) with zeros.
    """
    # Determine target shape (for u and v) as the maximum along each axis.
    shape_A = A.shape[:2]
    shape_B = B.shape[:2]
    target_shape = (max(shape_A[0], shape_B[0]), max(shape_A[1], shape_B[1]))
    A_pad = elevate_polynomial(A, target_shape) if shape_A != target_shape else A
    B_pad = elevate_polynomial(B, target_shape) if shape_B != target_shape else B
    return A_pad - B_pad


def elevate_polynomial(poly, target_shape):
    """
    "Elevate" (pad) a polynomial (given in monomial form) to a higher degree by adding zeros.
    poly: NumPy array of shape (p, q) or (p, q, d)
    target_shape: tuple (P, Q) with P >= p and Q >= q.
    (In the power basis, a polynomial of true degree d has zero coefficients for u^i v^j with i or j beyond d.)
    """
    current_shape = poly.shape[:2]
    if poly.ndim == 2:
        new_poly = np.zeros(target_shape)
        new_poly[:current_shape[0], :current_shape[1]] = poly
    else:
        new_poly = np.zeros((target_shape[0], target_shape[1], poly.shape[2]))
        new_poly[:current_shape[0], :current_shape[1], :] = poly
    return new_poly



def compute_gauss_map_rational(control_points):
    """
    Compute the Gaussian map for a rational Bézier patch.

    The input 'control_points' is assumed to be a NumPy array of shape (N, M, 4) in homogeneous coordinates,
    where each control point is [w*x, w*y, w*z, w] (i.e. the first three coordinates are pre–weighted).

    The algorithm is as follows:
      1. For each scalar field (X, Y, Z, and the weight W) convert from Bernstein to monomial form.
      2. Compute the partial derivatives with respect to u and v in the monomial basis.
      3. For each coordinate X, Y, Z, form the “numerator” of the rational derivative using the quotient rule,
         that is, compute A = (F_u * W - F * W_u) and B = (F_v * W - F * W_v).
      4. Compute the cross product of the vectors A and B (coordinate–wise using the polynomial convolution).
      5. The Gaussian map (in rational form) is then given by the homogeneous patch
             ( N_x, N_y, N_z, W^4 )
         where W^4 is computed by multiplying the weight polynomial with itself four times.
      6. Because the degrees may not match (often the N components come out one degree lower),
         elevate them (i.e. pad with zero–coefficients) so that all four components have the same degree.
      7. Convert the resulting homogeneous polynomial (in monomial form) back to a rational Bézier patch.

    Returns:
       gauss_map: a NumPy array of the Gaussian map’s control net in rational Bézier (homogeneous) form.
    """
    # Split control points into spatial components and weight.
    n, m, _ = control_points.shape
    # Each component is taken as an array of shape (n, m, 1)
    X = control_points[:, :, 0:1]
    Y = control_points[:, :, 1:2]
    Z = control_points[:, :, 2:3]
    W = control_points[:, :, 3:4]

    # Convert each to monomial representation.

    X_mon = bezier_to_monomial(X)
    Y_mon = bezier_to_monomial(Y)
    Z_mon = bezier_to_monomial(Z)
    W_mon = bezier_to_monomial(W)

    # Compute partial derivatives in the monomial basis.
    X_u = compute_partial_derivative(X_mon, "u")
    X_v = compute_partial_derivative(X_mon, "v")
    Y_u = compute_partial_derivative(Y_mon, "u")
    Y_v = compute_partial_derivative(Y_mon, "v")
    Z_u = compute_partial_derivative(Z_mon, "u")
    Z_v = compute_partial_derivative(Z_mon, "v")
    W_u = compute_partial_derivative(W_mon, "u")
    W_v = compute_partial_derivative(W_mon, "v")

    # For each scalar component, compute the rational derivative numerator parts:
    # For example, for X (scalar), the “adjusted” derivative numerator is: X_u*W - X*W_u.
    # (Here we work with the zeroth component of the last dimension because our arrays have shape (..., 1).)
    A_x = poly_subtract(poly_mult_2d(X_u[..., 0], W_mon[..., 0]),
                        poly_mult_2d(X_mon[..., 0], W_u[..., 0]))
    A_y = poly_subtract(poly_mult_2d(Y_u[..., 0], W_mon[..., 0]),
                        poly_mult_2d(Y_mon[..., 0], W_u[..., 0]))
    A_z = poly_subtract(poly_mult_2d(Z_u[..., 0], W_mon[..., 0]),
                        poly_mult_2d(Z_mon[..., 0], W_u[..., 0]))

    B_x = poly_subtract(poly_mult_2d(X_v[..., 0], W_mon[..., 0]),
                        poly_mult_2d(X_mon[..., 0], W_v[..., 0]))
    B_y = poly_subtract(poly_mult_2d(Y_v[..., 0], W_mon[..., 0]),
                        poly_mult_2d(Y_mon[..., 0], W_v[..., 0]))
    B_z = poly_subtract(poly_mult_2d(Z_v[..., 0], W_mon[..., 0]),
                        poly_mult_2d(Z_mon[..., 0], W_v[..., 0]))

    # Compute the cross product in the monomial basis.
    # Remember: (A × B)_x = A_y*B_z - A_z*B_y, etc.
    N_x = poly_subtract(poly_mult_2d(A_y, B_z),
                        poly_mult_2d(A_z, B_y))
    N_y = poly_subtract(poly_mult_2d(A_z, B_x),
                        poly_mult_2d(A_x, B_z))
    N_z = poly_subtract(poly_mult_2d(A_x, B_y),
                        poly_mult_2d(A_y, B_x))

    # Compute the denominator polynomial. Because the quotient rule for each derivative gave you a
    # denominator of W^2, the cross product has denominator W^4.
    # Compute W^2 then square it.
    W2 = poly_mult_2d(W_mon[..., 0], W_mon[..., 0])
    W4 = poly_mult_2d(W2, W2)

    # In general the N (numerator) polynomials come out with a slightly lower degree than W^4.
    # Determine the shape (i.e. number of monomial coefficients) for W4 and elevate the N components to that.
    target_shape = W4.shape  # (rows, cols)
    N_x_elev = elevate_polynomial(N_x, target_shape)
    N_y_elev = elevate_polynomial(N_y, target_shape)
    N_z_elev = elevate_polynomial(N_z, target_shape)

    # Stack the three numerator components and the weight polynomial into one 4–component polynomial.
    # The resulting array has shape (target_shape[0], target_shape[1], 4).
    N_mon = np.stack([N_x_elev, N_y_elev, N_z_elev, W4], axis=-1)

    # Convert from monomial representation back to a rational Bézier patch (i.e. to Bernstein representation).
    gauss_map = monomial_to_bezier(N_mon)

    return gauss_map


class GaussMap:
    def __init__(self, mp: NURBSSurface, surf: NURBSSurface):
        self.surface = surf
        self._map = mp
        self.hull=None
        self._polar_map = None
        self._convex_hull_on_sphere = None
        self.children = []
        self.bezier_patches = []
        #self.compute()

    @classmethod
    def from_surf(cls, surf):
        _map = compute_gauss_map(np.array(surf.control_points))
        #print((_map.tolist(),np.array(surf.control_points).tolist()))
        # Compute convex hull
        return cls(NURBSSurface(np.array(unit(_map.reshape((-1, 3)))).reshape(_map.shape),
                                (_map.shape[0] - 1, _map.shape[1] - 1)), surf)

    def subdivide(self, u=0.5,v=0.5):
        (umin, umax), (vmin, vmax) = self.surface.interval()
        umid = umin+(( umax-umin ) * u)
        vmid = vmin+(( vmax-vmin ) * v)
        (mumin, mumax), (mvmin, mvmax) = self._map.interval()
        mumid = mumin+(( mumax-mumin ) * u)
        mvmid = mvmin+(( mvmax-mvmin ) * v)
        try:

            srf = subdivide_surface(self.surface,umid,vmid,tol=1e-12,normalize_knots=False)
            mp = subdivide_surface(self._map,mumid, mvmid,tol=1e-12, normalize_knots=False)
        except ValueError as err:
            print(self.surface.interval())
            print(self._map.interval())
            raise err
        if len(self.children)==0:
            self.children = []
            for i in range(len(mp)):
                f = mp[i]
                s = srf[i]

                #f.normalize_knots()
                #s.normalize_knots()
                self.children.append(GaussMap(f, s))

            return  self.children
        else:
            #print("SSS")
            return self.children

    def compute(self):
        # Convert NURBS to Bézier patches

        #self.bezier_patches = decompose_surface(self.surface)
        #_map=compute_gauss_map(np.array(self.surface.control_points))
        #self._map=NURBSSurface(_map,(_map.shape[0]-1,_map.shape[1]-1))
        #_polar_map = cartesian_to_spherical(unit(_map.control_points_flat))
        #_polar_convex_hull = ConvexHull(_polar_map, qhull_options='QJ')
        # Compute Gauss map for each Bézier patch
        #gauss_maps = []
        #for patch in self.bezier_patches:
        #    gm = compute_gauss_map(np.array(patch.control_points))
        #    gm=np.array(unit(gm.reshape((-1,3))))
        #    gauss_maps.append(gm)

        # Combine Gauss maps

        # Compute polar representation
        #self._polar_map = cartesian_to_spherical(unit(self._map.control_points_flat))

        # Compute convex hull

        #self._polar_convex_hull = ConvexHull(np.array(unit(self._map.control_points_flat)), qhull_options='QJ')
        self._convex_hull_on_sphere=np.array(convex_hull(unit(self._map.control_points_flat)))
        #self.hull=self._polar_convex_hull.points[self._polar_convex_hull.vertices]
        self.hull =self._convex_hull_on_sphere
    def bounds(self):
        """Compute bounds on the Gauss map."""
        return self.hull

    def intersects(self, other: GaussMap):
        """Check if this Gauss map intersects with another."""

        return gjk(self.bounds(), other.bounds())


def linear_program_solver(c, A_ub, b_ub, A_eq, b_eq):
    """
    Solve a linear programming problem using scipy's linprog function.
    """
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
    return res.x if res.success else None


def separate_gauss_maps(gm1: GaussMap, gm2: GaussMap):
    """
    Attempt to find separating vectors P1 and P2 for two Gauss maps.

    Returns:
        P1, P2: np.ndarray or None, None if separation is not possible
    """
    N1 = gm1.bounds()
    N2 = gm2.bounds()

    # First, try to find P1
    P1 = find_separating_vector(N1, N2)
    if P1 is None:
        return None, None

    # If P1 is found, try to find P2
    P2 = find_common_side_vector(N1, N2)
    if P2 is None:
        return None, None

    return P1, P2


def find_separating_vector(N1, N2):
    """
    Find a vector P1 that satisfies:
    P1 · n1 > 0 for all n1 in N1
    P1 · n2 < 0 for all n2 in N2
    """
    m, n = len(N1), len(N2)

    # Set up the linear programming problem
    c = [0, 0, 0, 1]  # Minimize epsilon

    A_ub = np.zeros((m + n, 4))
    A_ub[:m, :3] = -N1  # For N1: -P1 · n1 + epsilon <= 0
    A_ub[:m, 3] = -1
    A_ub[m:, :3] = N2  # For N2: P1 · n2 + epsilon <= 0
    A_ub[m:, 3] = -1

    b_ub = np.zeros(m + n)

    A_eq = np.array([[0, 0, 0, 1]])  # epsilon <= 0
    b_eq = np.array([0])

    # Solve the linear programming problem
    result = linear_program_solver(c, A_ub, b_ub, A_eq, b_eq)

    if result is not None:
        P1 = result[:3]
        epsilon = result[3]
        if scalar_norm(P1) > 1e-6 and epsilon < 0:  # Check if the solution is valid
            return P1 / scalar_norm(P1)  # Normalize P1

    return None


def find_common_side_vector(N1, N2):
    """
    Find a vector P2 that satisfies:
    P2 · n1 > 0 for all n1 in N1
    P2 · n2 > 0 for all n2 in N2
    """
    m, n = len(N1), len(N2)

    # Set up the linear programming problem
    c = [0, 0, 0, 1]  # Minimize epsilon

    A_ub = np.zeros((m + n, 4))
    A_ub[:m, :3] = -N1  # For N1: -P2 · n1 + epsilon <= 0
    A_ub[:m, 3] = -1
    A_ub[m:, :3] = -N2  # For N2: -P2 · n2 + epsilon <= 0
    A_ub[m:, 3] = -1

    b_ub = np.zeros(m + n)

    A_eq = np.array([[0, 0, 0, 1]])  # epsilon <= 0
    b_eq = np.array([0])

    # Solve the linear programming problem
    result = linear_program_solver(c, A_ub, b_ub, A_eq, b_eq)

    if result is not None:
        P2 = result[:3]
        epsilon = result[3]
        if scalar_norm(P2) > 1e-6 and epsilon < 0:  # Check if the solution is valid
            return P2 / scalar_norm(P2)  # Normalize P2

    return None


# Usage example:
# gm1 = GaussMap(surface1)
# gm2 = GaussMap(surface2)
# P1, P2 = separate_gauss_maps(gm1, gm2)
# if P1 is not None and P2 is not None:
#     print("Gauss maps can be separated")
#     print("P1:", P1)
#     print("P2:", P2)
# else:
#     print("Gauss maps cannot be separated")



if __name__ == "__main__":
    from mmcore._test_data import ssx as td


