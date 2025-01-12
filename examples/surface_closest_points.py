import itertools
import numpy as np
from mmcore.geom.curves import NURBSpline
from mmcore.geom.surfaces import Ruled
from mmcore.numeric import scalar_dot
from mmcore.numeric.divide_and_conquer import (
    divide_and_conquer_min_2d,
    divide_and_conquer_min_2d_vectorized,
)
from mmcore.geom.bvh import contains_point, aabb
from mmcore.numeric.vectors import dot, norm


def create_ruled_from_points(points, degree=3):
    """
    Create a ruled surface from given points.

    :param points: A list of two lists or arrays representing the control points for the two curves.
                   Each list or array should contain points in the form of [x, y, z].
    :param degree: An integer specifying the degree of the NURBSpline curves. Defaults to 3.
    :return: A Ruled object representing a ruled surface created by connecting two NURBSpline curves.
    """
    return Ruled(
        NURBSpline(np.array(points[0], dtype=float), degree=degree),
        NURBSpline(np.array(points[1], dtype=float), degree=degree),
    )


def surface_closest_point_classic_approach(surface, pts, tol=1e-6):
    """
    Compute the closest points on a surface to a given set of points using a classic approach.

    :param surface: The surface object.
    :param pts: The set of points as a numpy array.
    :param tol: The tolerance value for the division and conquest algorithm. Default is 1e-6.
    :return: The closest points on the surface corresponding to the given set of points as a numpy array of (u, v) pairs.
    """
    surface.build_tree(10, 10)

    def objective(u, v):
        d = surface.evaluate(np.array((u, v))) - pt
        return scalar_dot(d, d)

    uvs = np.zeros((len(pts), 2))

    for i, pt in enumerate(pts):
        objects = contains_point(surface.tree, pt)
        if len(objects) == 0:
            uvs[i] = np.array(
                divide_and_conquer_min_2d(objective, *surface.interval(), tol=tol)
            )
        else:
            uvs_ranges = np.array(
                list(itertools.chain.from_iterable(o.uvs for o in objects))
            )
            uvs[i] = np.array(
                divide_and_conquer_min_2d(
                    objective,
                    (np.min(uvs_ranges[..., 0]), np.max(uvs_ranges[..., 0])),
                    (np.min(uvs_ranges[..., 1]), np.max(uvs_ranges[..., 1])),
                    tol=tol,
                )
            )
    return uvs


def surface_closest_point_vectorized_approach(surface, pts, tol=1e-6):
    """
    Compute the closest points on a surface to a given set of points using a vectorized approach.

    :param surface: The surface object.
    :param pts: The set of points as a numpy array.
    :param tol: The tolerance value for the division and conquest algorithm. Default is 1e-6.
    :return: The closest points on the surface corresponding to the given set of points as a numpy array of (u, v) pairs.
    """

    def objective(u, v):
        d = surface(np.array((u, v)).T) - pts
        return np.array(dot(d, d))

    (u_min, u_max), (v_min, v_max) = surface.interval()
    x_range = np.empty((2, len(pts)))
    x_range[0] = u_min
    x_range[1] = u_max
    y_range = np.empty((2, len(pts)))
    y_range[0] = v_min
    y_range[1] = v_max

    uvs = np.array(
        divide_and_conquer_min_2d_vectorized(
            objective, x_range=x_range, y_range=y_range, tol=tol
        )
    )
    return uvs.T


# Define control points for the ruled surface
# Define control points and create the ruled surface
cpts = np.array(
    [
        [
            [15.8, 10.1, 0.0],
            [-3.0, 13.0, 0.0],
            [-11.5, 7.7, 0.0],
            [-27.8, 12.8, 0.0],
            [-34.8, 9.3, 0.0],
        ],
        [
            [15.8, 3.6, 10.1],
            [-3.0, 8.6, 15.4],
            [-11.5, 12.3, 19.6],
            [-27.8, 6.3, 16.9],
            [-34.8, 5.0, 16.7],
        ],
    ]
)  # Control points definition
surface = create_ruled_from_points(cpts, degree=3)
# Generate random points within the bounding box of control points
pts_count = 100
min_point, max_point = np.array(aabb(cpts.reshape(-1, 3)) )* 1.25
pts = np.zeros((pts_count, 3))
pts[..., 0] = np.random.uniform(min_point[0], max_point[0], size=pts_count)
pts[..., 1] = np.random.uniform(min_point[1], max_point[1], size=pts_count)
pts[..., 2] = np.random.uniform(min_point[2], max_point[2], size=pts_count)

# Create the ruled surface
surface = create_ruled_from_points(cpts, degree=3)

# Compare performance of classic and vectorized approaches
import time

s1 = time.time()
uvs_v1 = surface_closest_point_classic_approach(surface, pts, tol=1e-5)
print("Classic approach done in:", time.time() - s1)

s2 = time.time()
uvs_v2 = surface_closest_point_vectorized_approach(surface, pts, tol=1e-5)
print("Vectorized approach done in:", time.time() - s2)

# Project points onto the surface
projected_points1 = np.array(surface(uvs_v1))
projected_points2 = np.array(surface(uvs_v2))

# Compare accuracy
classic_error = np.array(norm(projected_points1 - pts))
vectorized_error = np.array(norm(projected_points2 - pts))

if np.all(classic_error < vectorized_error):
    print(f"Classic approach is more accurate by {np.average(classic_error)} units.")
elif np.all(classic_error > vectorized_error):
    print(
        f"Vectorized approach is more accurate by {np.average(vectorized_error)} units."
    )
else:
    print(f"Both approaches are about equally accurate.")

# Export points for further analysis
ptl_classic = pts.tolist(), projected_points1.tolist()
ptl_vectorized = pts.tolist(), projected_points2.tolist()
