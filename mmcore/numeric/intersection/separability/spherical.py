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


def spherical_separability(surface_points, curve_points, intersection_point):
    """
    Perform a spherical separability test between a NURBS surface and curve, given a known intersection point.
    This test is based on the method described in section 4.3, "Spherical Separability" by Michael Edward Hohmeyer.

    **Overview**:

    This test projects the control points of both the surface and curve onto a unit sphere centered at the intersection
    point and checks if their spherical projections intersect anywhere other than at the given intersection point.

    The algorithm implements two tests:

    1. A fast spherical bounding box test, which provides an initial check for separability.
    2. If the bounding box test fails, a more accurate separating circles test is performed, based on linear programming.

    Parameters
    ----------
    :param surface_points:
        A 2D array of control points representing the surface, excluding the intersection point.
    :type surface_points: np.ndarray

    :param curve_points:
        A 2D array of control points representing the curve, excluding the intersection point.
    :type curve_points: np.ndarray

    :param intersection_point:
        The known intersection point between the surface and the curve. This point will be excluded from the separability tests.
    :type intersection_point: np.ndarray

    Returns
    -------
    :return:
        True if the surface and curve are separable and intersect only at the given intersection point.
        False if they intersect at additional points.
    :rtype: bool

    Notes
    -----
    - The function performs two stages of separability testing: an initial bounding box test and a more accurate
      separating circles test if necessary.
    - The spherical bounding box test is analogous to axis-aligned bounding box tests in 3D space, and is computationally inexpensive.
    - The separating circles test uses a linear programming approach to determine whether the curve and surface projections
      onto the sphere are separable.

    Example
    -------
    .. code-block:: python

        surface_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        curve_points = np.array([[0, 0, 1], [1, 1, 2]])
        intersection_point = np.array([1, 1, 1])

        result = spherical_separability(surface_points, curve_points, intersection_point)
        print(result)  # Output: True or False depending on separability.

    **Algorithm Reference**:

    - "Spherical Separability" as described in section 4.3 of "Robust and Efficient Surface Intersection for Solid Modeling"
      by Michael Edward Hohmeyer (University of California, 1986).

    **Notes**:

    - The function assumes that the `surface_points` and `curve_points` are in 3D space and excludes the `intersection_point`
      before running the separability tests.
    - The bounding box test is significantly faster but less accurate. If this test fails, the more accurate but
      computationally expensive separating circles test is run.
    """

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