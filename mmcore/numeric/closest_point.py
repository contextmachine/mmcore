import itertools
import os

import numpy as np

from mmcore.numeric.vectors import vector_projection, scalar_dot, scalar_norm, dot

from mmcore.geom.bvh import contains_point
from mmcore.geom.surfaces import Surface
from mmcore.numeric import divide_interval
from mmcore.numeric.fdm import PDE, newtons_method
from mmcore.numeric.divide_and_conquer import iterative_divide_and_conquer_min, divide_and_conquer_min_2d, \
    divide_and_conquer_min_2d_vectorized

from scipy.optimize import newton
import multiprocessing as mp

from mmcore.numeric.fdm import bounded_fdm

import math

# Utility function to calculate the Euclidean distance between two points
import math


# Utility function to calculate the Euclidean distance between two points
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Utility function to find the closest distance in a strip
def strip_closest(strip, d):
    min_dist = d[0]
    pair = d[1]

    # Sort the strip according to y-coordinate
    strip.sort(key=lambda p: p[1])

    # Compare each point with the next points within min_dist in the strip
    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and (strip[j][1] - strip[i][1]) < min_dist:
            current_dist = dist(strip[i], strip[j])
            if current_dist < min_dist:
                min_dist = current_dist
                pair = (strip[i][2], strip[j][2])  # Store the indices of the closest pair
            j += 1

    return (min_dist, pair)


# Recursive function to find the smallest distance and the corresponding pair of indices
def closest_util(points_sorted_x, points_sorted_y, n):
    # Base case: Use brute force for 3 or fewer points
    if n <= 3:
        min_dist = float('inf')
        pair = (-1, -1)
        for i in range(n):
            for j in range(i + 1, n):
                current_dist = dist(points_sorted_x[i], points_sorted_x[j])
                if current_dist < min_dist:
                    min_dist = current_dist
                    pair = (points_sorted_x[i][2], points_sorted_x[j][2])  # Store indices
        return (min_dist, pair)

    # Find the middle point
    mid = n // 2
    mid_point = points_sorted_x[mid]

    # Divide points_sorted_y into left and right halves
    points_sorted_y_left = [point for point in points_sorted_y if point[0] <= mid_point[0]]
    points_sorted_y_right = [point for point in points_sorted_y if point[0] > mid_point[0]]

    # Recursively find the smallest distances in the left and right halves
    dl = closest_util(points_sorted_x[:mid], points_sorted_y_left, mid)
    dr = closest_util(points_sorted_x[mid:], points_sorted_y_right, n - mid)

    # Determine the smaller distance and the corresponding pair
    if dl[0] < dr[0]:
        d = dl
    else:
        d = dr

    # Build the strip of points within distance d from the midline
    strip = [point for point in points_sorted_y if abs(point[0] - mid_point[0]) < d[0]]

    # Find the closest pair in the strip
    strip_closest_dist = strip_closest(strip, d)

    # Return the overall minimum distance and the corresponding pair
    if strip_closest_dist[0] < d[0]:
        return strip_closest_dist
    else:
        return d


# Function to find the closest pair of points and their indices
def min_distance(points):
    """
    Finds the smallest distance between any two points in the given list and returns the distance along with the indices of the points.

    # Example usage
    points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    min_dist, pair = min_distance(points)
    print(f"The smallest distance is {min_dist} between points at indices {pair}")
    # Output: The smallest distance is 1.4142135623730951 between points at indices (0, 5)

    :param points: List of tuples representing the points [(x1, y1), (x2, y2), ...]
    :return: A tuple containing the minimum distance and a tuple of the two point indices
    """
    # Enumerate points to keep track of their original indices
    enumerated_points = list(enumerate(points))

    # Represent each point as (x, y, index)
    points_with_index = [(x, y, idx) for idx, (x, y) in enumerated_points]

    # Sort the points based on x and y coordinates
    points_sorted_x = sorted(points_with_index, key=lambda p: p[0])
    points_sorted_y = sorted(points_with_index, key=lambda p: p[1])

    # Use the recursive utility to find the closest pair
    return closest_util(points_sorted_x, points_sorted_y, len(points_sorted_x))


# Example usage
if __name__ == "__main__":
    points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    min_dist, pair = min_distance(points)
    print(f"The smallest distance is {min_dist} between points at indices {pair}")
    # Expected Output: The smallest distance is 1.4142135623730951 between points at indices (0, 5)
def foot_point(S, P, s0, t0, partial_derivatives=None, epsilon=1e-6, alpha_max=20):
    """
    Find the foot point on the parametric surface S(s, t) closest to the given point P.
    """
    if partial_derivatives is None:
        _pde = PDE(S, dim=2)
        partial_derivatives = lambda uv: _pde(uv).T
    s, t = st = np.array([s0, t0])

    while True:
        p_i = S(st)
        e_s, e_t = partial_derivatives(st)
        # Solve the linear system for Δs and Δt
        A = np.array([
            [scalar_dot(e_s, e_s), scalar_dot(e_s, e_t)],
            [scalar_dot(e_s, e_t), scalar_dot(e_t, e_t)]
        ])
        b = np.array([
            scalar_dot(P - p_i, e_s),
            scalar_dot(P - p_i, e_t)
        ])
        delta = np.linalg.solve(A, b)
        delta_s, delta_t = delta
        q_i = p_i + delta_s * e_s + delta_t * e_t
        s_new = s + delta_s
        t_new = t + delta_t
        p_new = S(s_new, t_new)
        f1 = q_i - p_i
        f2 = p_new - q_i
        # Check convergence
        if np.linalg.norm(q_i - p_i) < epsilon:
            break
        # Newton step for the foot point on the tangent parabola
        a0 = scalar_dot(P - p_i, f1)
        a1 = 2 * scalar_dot(f2, P - p_i) - scalar_dot(f1, f1)
        a2 = -3 * scalar_dot(f1, f2)
        a3 = -2 * scalar_dot(f2, f2)
        alpha = 1 - (a0 + a1 + a2 + a3) / (a1 + 2 * a2 + 3 * a3)
        alpha = np.clip(alpha, 0, alpha_max)
        s = s + alpha * delta_s
        t = t + alpha * delta_t
        st[0] = s
        st[1] = t
    return S(s, t), s, t


def closest_point_on_curve_single(curve, point, tol=1e-3):
    """

    :param curve: The curve on which to find the closest point.
    :param point: The point for which to find the closest point on the curve.
    :param tol: The tolerance for the minimum finding algorithm. Defaults to 1e-5.
    :return: The closest point on the curve to the given point, distance.

    """
    _fn = getattr(curve, "evaluate", curve)

    def distance_func(t):
        return scalar_norm(point - _fn(t))

    t0, t1 = curve.interval()

    t_best, d_best = t0, distance_func(t0)
    t, d = t1, distance_func(t1)
    if d < d_best:
        t_best = t
        d_best = d

    for bnds in divide_interval(*curve.interval(), step=0.5):
        # t,d=find_best(distance_func, bnds, tol=tol)
        t, d = iterative_divide_and_conquer_min(distance_func, bnds, tol=tol)
        if d < d_best:
            t_best = t
            d_best = d

    return t_best, d_best


class _ClosestPointSolution:
    def __init__(self, curve, tol=1e-5):
        self.curve = curve
        self.tol = tol

    def __call__(self, point):
        return closest_point_on_curve_single(self.curve, point, tol=self.tol)


def closest_points_on_curve_mp(curve, points, tol=1e-3, workers=1):
    if workers == -1:
        workers = os.cpu_count()
    with mp.Pool(workers) as pool:
        solution = _ClosestPointSolution(curve, tol=tol)
        return list(pool.map(solution, points
                             ))


def closest_point_on_curve(curve, pts, tol=1e-3, workers=1):
    pts = pts if isinstance(pts, np.ndarray) else np.array(pts)

    if pts.ndim == 1:
        return closest_point_on_curve_single(curve, pts, tol=tol)

    if workers == 1:
        return [closest_point_on_curve_single(curve, pt, tol=tol) for pt in pts]
    else:
        return closest_points_on_curve_mp(curve, pts, tol=tol, workers=workers)


def local_closest_point_on_curve(curve, t0, point, tol=1e-3, **kwargs):
    def fun(t):
        # C' (u) •(C(u) - P)
        return scalar_dot(curve.derivative(t), curve.evaluate(t) - point)

    dfun = bounded_fdm(fun, curve.interval())
    res = newton(fun, t0, fprime=dfun, tol=tol, **kwargs)
    return res, np.linalg.norm(curve.evaluate(res) - point)


def closest_point_on_ray(ray, point):
    start, direction = ray

    return start + vector_projection(point - start, direction)


def closest_point_on_line(line, point):
    start, end = line
    direction = end - start
    return start + vector_projection(point - start, direction)


def closest_point_on_surface(self: Surface, pt, tol=1e-3, bounds=None):
    if bounds is None:
        bounds = tuple(self.interval())
    (umin, umax), (vmin, vmax) = bounds

    def wrp1(uv):
        d = self.evaluate(uv) - pt
        return scalar_dot(d, d)

    def wrp(u, v):
        d = self.evaluate(np.array([u, v])) - pt
        return scalar_dot(d, d)

    cpt = contains_point(self.tree, pt)

    if len(cpt) == 0:
        #(umin, umax), (vmin, vmax) = self.interval()
        return np.array(divide_and_conquer_min_2d(wrp, (umin, umax), (vmin, vmax), tol))

    else:

        initial = np.average(min(cpt, key=lambda x: x.bounding_box.volume()).uvs, axis=0)
        uv = newtons_method(wrp1, initial, tol=tol)
        if uv is None:
            raise ValueError('Newtons method failed to converge')
        return uv


def closest_points_on_surface(surface, pts, tol=1e-6):
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


def closest_point_on_surface_batched(surface, pts, tol=1e-6):
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


__all__ = ["closest_point_on_curve",

           "closest_point_on_line",
           "foot_point",
           "closest_point_on_curve_single",
           "closest_points_on_curve_mp",
           "closest_points_on_curve_mp",
           "local_closest_point_on_curve"
           ]
