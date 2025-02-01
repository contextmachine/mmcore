# mmcore/numeric/gauss_map.py
from __future__ import annotations

from mmcore.geom.nurbs import NURBSSurface, subdivide_surface

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


import numpy as np


def is_developable_bezier_patch(control_points, tol=1e-12):
    """
    Check whether a single-span Bézier patch is developable
    by verifying that (N_u x N_v) is identically zero in monomial form.

    Args:
        control_points (np.ndarray): shape (n_u, n_v, 3)
            The control points of the Bézier patch in Bernstein form.
        tol (float): Tolerance for considering coefficients "zero."

    Returns:
        bool: True if the patch is (within tol) developable, False otherwise.
    """
    # 1. Convert Bézier patch to monomial coefficients
    F_monomial = bezier_to_monomial(control_points)

    # 2. Compute partial derivatives F_u, F_v (in monomial form)
    Fu = compute_partial_derivative(F_monomial, "u")  # shape (n_u-1, n_v, 3)
    Fv = compute_partial_derivative(F_monomial, "v")  # shape (n_u, n_v-1, 3)

    # 3. Form the unnormalized normal N = Fu x Fv (still in monomial form)
    N = cross_product(Fu, Fv)  # shape will be ( (n_u-1)+(n_u)-1, (n_v)+(n_v-1)-1, 3 ) => (2n_u-2, 2n_v-2, 3)

    # 4. Compute partial derivatives N_u, N_v
    Nu = compute_partial_derivative(N, "u")  # shape (2n_u-3, 2n_v-2, 3)
    Nv = compute_partial_derivative(N, "v")  # shape (2n_u-2, 2n_v-3, 3)

    # 5. Cross them => Nx(u,v) = Nu x Nv  (monomial form)
    Nx = cross_product(Nu, Nv)  # shape ( (2n_u-3)+(2n_u-2)-1, (2n_v-2)+(2n_v-3)-1, 3 ) => (4n_u-5, 4n_v-5, 3)

    # 6. Check if Nx is identically zero (all coefficients ~ 0)
    #    Since Nx is in monomial form, these are its polynomial coefficients.
    #    If *all* are near zero, the surface is developable.
    if np.all(np.abs(Nx) < tol):
        return True

    return False

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

    def subdivide(self):
        (umin, umax), (vmin, vmax) = self.surface.interval()
        umid = (umin + umax) * 0.5
        vmid = (vmin + vmax) * 0.5
        (mumin, mumax), (mvmin, mvmax) = self._map.interval()
        mumid = (mumin + mumax) * 0.5
        mvmid = (mvmin + mvmax) * 0.5
        try:

            srf = subdivide_surface(self.surface,umid,vmid,tol=1e-12,normalize_knots=False)
            mp = subdivide_surface(self._map,mumid, mvmid,tol=1e-12, normalize_knots=False)
        except ValueError as err:
            print(self.surface.interval())
            print(self._map.interval())
            raise err
        if len(self.children)==0:
            self.children = []
            for i in range(4):
                f = mp[i]
                s = srf[i]

                #f.normalize_knots()
                #s.normalize_knots()
                self.children.append(GaussMap(f, s))

            return  self.children
        else:
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




