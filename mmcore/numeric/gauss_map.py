from __future__ import annotations

import numpy as np

from mmcore.numeric.intersection.ssx.cydqr import gjk
from mmcore.numeric.monominal import bezier_to_monomial, monomial_to_bezier
from mmcore.numeric.vectors import unit,cartesian_to_spherical

from scipy.spatial import ConvexHull


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


def normalize_polynomial(v, epsilon=1e-10):
    """Normalize a 3D vector polynomial with improved stability."""
    norm_squared = v[:, :, 0] ** 2 + v[:, :, 1] ** 2 + v[:, :, 2] ** 2
    max_norm = np.max(norm_squared)
    if max_norm < epsilon:
        return np.zeros_like(v)
    norm = np.sqrt(norm_squared / max_norm)
    return v / (norm[:, :, np.newaxis] * np.sqrt(max_norm))



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

    return  gauss_map


class GaussMap:
    #__slots__='_map','_unit_map', '_polar_map','_polar_convex_hull'
    def __init__(self, bern_coeffs):
        self._shape=bern_coeffs.shape
        self._map = bern_coeffs
        self.solve()



    def solve(self):
        self._polar_map_flat = np.array(cartesian_to_spherical(self._map.reshape((-1, 3))))
        self._unit_polar_map = self._polar_map_flat.copy()
        self._unit_polar_map[..., -1] = 1.
        self.polar_convex_hull=ConvexHull(self._unit_polar_map)

    def intersects(self, other:GaussMap):
        gjk(self.polar_convex_hull.points[self.polar_convex_hull.vertices])


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