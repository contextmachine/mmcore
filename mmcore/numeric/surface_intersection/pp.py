import numpy as np
from scipy.optimize import fsolve

TOL = 0.1

def _evaluate_curvature(
    derivative, second_derivative
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Calculates the unit tangent vector, curvature vector, and a recalculate condition for a given derivative and
    second derivative.

    :param derivative: The derivative vector.
    :param second_derivative: The second derivative vector.
    :return: A tuple containing the unit tangent vector, curvature vector, and recalculate condition.

    Example usage:
        derivative = np.array([1, 0, 0])
        second_derivative = np.array([0, 1, 0])
        _evaluate_curvature(derivative, second_derivative)
    """
    norm_derivative = np.linalg.norm(derivative)
    zero_tolerance = 0.0

    if norm_derivative == zero_tolerance:
        norm_derivative = np.linalg.norm(second_derivative)
        if norm_derivative > zero_tolerance:
            unit_tangent_vector = second_derivative / norm_derivative
        else:
            unit_tangent_vector = np.zeros_like(second_derivative)
        curvature_vector = np.zeros_like(second_derivative)
        recalculate_condition = False
    else:
        unit_tangent_vector = derivative / norm_derivative
        negative_second_derivative_dot_tangent = -np.dot(second_derivative, unit_tangent_vector)
        curvature_vector = (second_derivative + negative_second_derivative_dot_tangent * unit_tangent_vector) / (norm_derivative ** 2)
        recalculate_condition = True

    return unit_tangent_vector, curvature_vector, recalculate_condition


def _plane_line_intersect(plane: np.ndarray, line, tol=1e-6, full_return=False):
    ray_origin, ray_direction = line
    ndotu = plane[-1] @ ray_direction
    if abs(ndotu) < tol:
        return (None, None, None) if full_return else None
    w = ray_origin - plane[0]
    si = -plane[-1] @ w / ndotu
    Psi = w + si * ray_direction + plane[0]
    return (w, si, Psi) if full_return else Psi


def _plane_plane_intersect(plane1: np.ndarray, plane2: np.ndarray):
    normal1, normal2 = plane1[3], plane2[3]

    # Stack the normal vectors
    array_normals_stacked = np.vstack((normal1, normal2))

    # Construct the matrix
    matrix = np.block(
        [
            [2 * np.eye(3), array_normals_stacked.T],
            [array_normals_stacked, np.zeros((2, 2))],
        ]
    )

    # Construct the right-hand side of the equation
    dot_a = np.dot(plane1[0], normal1)
    dot_b = np.dot(plane2[0], normal2)
    array_y = np.array([*plane1[0], dot_a, dot_b])

    # Solve the linear system
    solution = np.linalg.solve(matrix, array_y)
    point_line = solution[:3]
    direction_line = np.cross(normal1, normal2)

    return point_line, direction_line


def _plane_plane_plane_intersect(first, second, third, tol=1e-6):
    return _plane_line_intersect(third, _plane_plane_intersect(first, second), tol)


def _vector_projection(a, b):
    bn = np.dot(b, b)
    return np.outer(a, b).sum(axis=-1) / bn


def _closest_point_on_line(line, point):
    start, end = line
    direction = end - start
    return start + _vector_projection(point - start, direction)


def freeform_step_debug(s1, s2, uv1, uv2):
    pl1, pl2 = s1.plane_at(uv1), s2.plane_at(uv2)
    ln = np.array(_plane_plane_intersect(pl1, pl2))
    np1 = _closest_point_on_line((ln[0], ln[0] + ln[1]), pl1[0])
    np2 = _closest_point_on_line((ln[0], ln[0] + ln[1]), pl2[0])

    return np1, np1 + (np2 - np1) / 2, np2


def solve_marching(s1, s2, uv1, uv2, tol):
    pl1, pl2 = s1.plane_at(uv1), s2.plane_at(uv2)
    marching_direction = np.cross(pl1[-1], pl2[-1])
    K = _evaluate_curvature(marching_direction, pl1[-1])[1]
    r = 1 / np.linalg.norm(K)
    step = np.sqrt(r ** 2 - (r - tol) ** 2) * 2
    new_pln = np.array([pl1[0] + marching_direction * step, pl1[-1], pl2[-1], marching_direction])
    return _plane_plane_plane_intersect(pl1, pl2, new_pln), step


def improve_uv(s, uv_old, xyz_better):
    x_old, y_old, z_old = s.evaluate(uv_old)
    dxdu, dydu, dzdu = s.derivative_u(uv_old)
    dxdv, dydv, dzdv = s.derivative_v(uv_old)
    x_better, y_better, z_better = xyz_better

    A = np.array([[dxdu, dxdv], [dydu, dydv], [dzdu, dzdv]])
    B = np.array([x_better - x_old, y_better - y_old, z_better - z_old])
    return np.linalg.lstsq(A, B, rcond=None)[0]


def freeform_step(s1, s2, uvb1, uvb2, tol=TOL):
    xyz_better = freeform_step_debug(s1, s2, uvb1, uvb2)[1]
    uvb1_better = uvb1 + improve_uv(s1, uvb1, xyz_better)
    uvb2_better = uvb2 + improve_uv(s2, uvb2, xyz_better)

    print(uvb1_better, uvb2_better)

    if np.any(uvb1_better < 0) or np.any(uvb2_better < 0):
        return
    xyz1_new = s1(uvb1_better)
    xyz2_new = s2(uvb2_better)

    return (xyz1_new, uvb1_better), (xyz2_new, uvb2_better) if np.linalg.norm(xyz1_new - xyz2_new) < tol else \
        freeform_step(s1, s2, uvb1_better, uvb2_better, tol)


def marching_step(s1, s2, uvb1, uvb2, tol=TOL):
    xyz_better, step = solve_marching(s1, s2, uvb1, uvb2, tol)
    uvb1_better = uvb1 + improve_uv(s1, uvb1, xyz_better)
    uvb2_better = uvb2 + improve_uv(s2, uvb2, xyz_better)
    print(uvb1_better,uvb2_better)
    if np.any(uvb1_better<0) or  np.any(uvb2_better<0):
        return
    xyz1_new = s1(uvb1_better)
    xyz2_new = s2(uvb2_better)

    return (xyz1_new, uvb1_better), (xyz2_new, uvb2_better), step if np.linalg.norm(xyz1_new - xyz2_new) < tol else \
        marching_step(s1, s2, uvb1_better, uvb2_better, tol)


def check_surface_edge(uv, surf_interval):
    bounds_u, bounds_v = surf_interval
    u, v = uv
    return not (0 < u < 1 and 0 < v < 1)


def check_wrap_back(xyz, initial_xyz, step):
    return np.linalg.norm(xyz - initial_xyz) < step / 2


def stop_check(s1, s2, xyz1, xyz2, uv1, uv2, initial_xyz, step, iterations, max_iter=100):
    return iterations >= max_iter or \
           np.linalg.norm(xyz1 - xyz2) >= TOL or \
           check_surface_edge(uv1, s1.interval()) or \
           check_surface_edge(uv2, s2.interval()) or \
           check_wrap_back(xyz1, initial_xyz, step) or \
           check_wrap_back(xyz2, initial_xyz, step)


def marching_method(s1, s2, initial_uv1, initial_uv2, tol=TOL, max_iter=1000, no_ff=False):
    iterations = 0


    (initial_xyz1, uv1), (initial_xyz2, uv2) = freeform_step(s1, s2, initial_uv1, initial_uv2)
    (xyz1, uv1_new), (xyz2, uv2_new), step = marching_step(s1, s2, uv1, uv2)

    pts = [(xyz1, xyz2)]
    uvs = [(uv1_new, uv2_new)]
    steps = [step]

    for i in range(max_iter):

        uv1, uv2 = uv1_new, uv2_new
        res=marching_step(s1, s2, uv1, uv2)
        if res is None:
            break

        (xyz1, uv1_new), (xyz2, uv2_new), step = res



        pts.append((xyz1, xyz2))
        uvs.append((uv1_new.tolist(), uv2_new.tolist()))
        steps.append(step)

    return uvs, pts, steps


def surface_local_cpt(surf1, surf2):
    def fun(t):

        return surf2.evaluate(np.array([t[1], t[2]])) - surf1.evaluate(np.array([0.00, t[0]]))

    return fsolve(fun, [0.5, 0.5, 0.5])


def surface_surface_intersection(surf1, surf2):
    aa = surface_local_cpt(surf1, surf2)
    print(aa)
    uvs, pts, steps = marching_method(surf2, surf1, np.array([aa[1], aa[2]]), np.array([0.0, aa[0]]))
    return uvs, pts, steps