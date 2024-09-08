import numpy as np
from scipy.optimize import linprog

__all__=['spherical_separability']
def project_to_sphere(points, center):
    """Project points onto a unit sphere centered at 'center'."""
    vectors = points - center

    return vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]


def find_smallest_wedge(points_2d):
    """Find the smallest wedge containing 2D points."""
    angles = np.arctan2(points_2d[:, 1], points_2d[:, 0])
    sorted_angles = np.sort(angles)
    angle_diffs = np.diff(sorted_angles)

    angle_diffs = np.append(
        angle_diffs, 2 * np.pi - sorted_angles[-1] + sorted_angles[0]
    )
    max_gap = np.argmax(angle_diffs)
    start_angle = sorted_angles[max_gap]
    end_angle = sorted_angles[(max_gap + 1) % len(sorted_angles)]
    return start_angle, end_angle


def spherical_bounding_box_test(points1, points2, center):
    """Perform spherical bounding box test."""
    sphere_points1 = project_to_sphere(points1, center)
    sphere_points2 = project_to_sphere(points2, center)

    for axis in range(3):
        plane_coords1 = np.column_stack(
            (sphere_points1[:, (axis + 1) % 3], sphere_points1[:, (axis + 2) % 3])
        )
        plane_coords2 = np.column_stack(
            (sphere_points2[:, (axis + 1) % 3], sphere_points2[:, (axis + 2) % 3])
        )

        wedge1_start, wedge1_end = find_smallest_wedge(plane_coords1)
        wedge2_start, wedge2_end = find_smallest_wedge(plane_coords2)

        # Check if wedges are separated
        if (wedge1_end < wedge2_start and wedge2_end > wedge1_start) or (
            wedge2_end < wedge1_start and wedge1_end > wedge2_start
        ):
            return True  # Separating circle found

    return False  # No separating circle found


def separating_circles_test(points1, points2, center):
    """Perform separating circles test using linear programming."""
    sphere_points1 = project_to_sphere(points1, center)
    sphere_points2 = project_to_sphere(points2, center)

    m, n = len(sphere_points1), len(sphere_points2)

    # Set up the linear programming problem
    c = [0, 0, 0, 1]  # Objective function coefficients
    A_ub = np.vstack(
        (
            np.hstack((-sphere_points1, -np.ones((m, 1)))),
            np.hstack((sphere_points2, -np.ones((n, 1)))),
        )
    )
    b_ub = np.zeros(m + n)

    # Solve the linear programming problem
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, method="highs")

    return not res.success  # True if no separating plane was found


def spherical_separability_test(surface_points, curve_points, intersection_point):
    """Perform full spherical separability test."""
    # Remove the intersection point from both sets
    surface_points = surface_points[
        ~np.all(surface_points == intersection_point, axis=1)
    ]
    curve_points = curve_points[~np.all(curve_points == intersection_point, axis=1)]

    # First, try the cheaper bounding box test
    if spherical_bounding_box_test(surface_points, curve_points, intersection_point):
        return True

    # If bounding box test fails, use the more expensive separating circles test
    return separating_circles_test(surface_points, curve_points, intersection_point)

spherical_separability=spherical_separability_test