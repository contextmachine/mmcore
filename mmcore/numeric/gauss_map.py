# mmcore/numeric/gauss_map.py
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog

from mmcore.geom.nurbs import NURBSSurface, subdivide_surface, decompose_surface

from mmcore.numeric.algorithms.quicksort import unique

from mmcore.numeric.algorithms.surface_area import v_min
from mmcore.numeric.intersection.csx import nurbs_csx
from mmcore.numeric.intersection.ssx.boundary_intersection import extract_isocurve
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


#def decompose_surface(surface, decompose_dir="uv"):
#    def decompose_direction(srf, idx):
#        srf_list = []
#        knots = srf.knots_u if idx == 0 else srf.knots_v
#        degree = srf.degree[idx]
#        unique_knots = sorted(set(knots[degree + 1: -(degree + 1)]))
#
#        while unique_knots:
#            knot = unique_knots[0]
#            if idx == 0:
#                srfs = split_surface_u(srf, knot)
#            else:
#                srfs = split_surface_v(srf, knot)
#            srf_list.append(srfs[0])
#            srf = srfs[1]
#            unique_knots = unique_knots[1:]
#        srf_list.append(srf)
#        return srf_list
#
#    if not isinstance(surface, NURBSSurface):
#        raise ValueError("Input must be an instance of NURBSSurface class")
#
#    surf = surface.copy()
#
#    if decompose_dir == "u":
#        return decompose_direction(surf, 0)
#    elif decompose_dir == "v":
#        return decompose_direction(surf, 1)
#    elif decompose_dir == "uv":
#        multi_surf = []
#        surfs_u = decompose_direction(surf, 0)
#        for sfu in surfs_u:
#            multi_surf += decompose_direction(sfu, 1)
#        return multi_surf
#    else:
#        raise ValueError(
#            f"Cannot decompose in {decompose_dir} direction. Acceptable values: u, v, uv"
#        )


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


def is_bezier(surface: NURBSSurface):
    kv, ku = unique(surface.knots_u).shape[0], unique(surface.knots_u).shape[0]
    if kv.shape[0] < 2 or ku.shape[0] < 2:
        raise ValueError("Degenerated patch")

    return kv.shape[0] == 2 and ku.shape[0] == 2


def normalize_polynomial(v, epsilon=1e-10):
    """Normalize a 3D vector polynomial with improved stability."""
    norm_squared = v[:, :, 0] ** 2 + v[:, :, 1] ** 2 + v[:, :, 2] ** 2
    max_norm = np.max(norm_squared)
    if max_norm < epsilon:
        return np.zeros_like(v)
    norm = np.sqrt(norm_squared / max_norm)
    return v / (norm[:, :, np.newaxis] * np.sqrt(max_norm))

"""
# Example usage and verification
bezier_patch = np.array(
    [
        [[0, 0, 0], [0, 1, 0], [0, 2, 0]],
        [[1, 0, 1], [1, 1, 2], [1, 2, 1]],
        [[2, 0, 0], [2, 1, 0], [2, 2, 0]],
    ]
)

gauss_map, normalized_gauss_map = compute_gauss_map(bezier_patch)
print("Gauss map control points:")
print(gauss_map)

"""


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


