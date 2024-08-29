import numpy as np
import sys

from mmcore.numeric.vectors import scalar_dot,scalar_cross,dot
def dot2(v):
    return scalar_dot(v,v)

def udTriangle(p, a, b, c):
    ba = b - a
    pa = p - a
    cb = c - b
    pb = p - b
    ac = a - c
    pc = p - c

    nor = np.cross(ba, ac)

    sign_sum = (np.sign(np.dot(np.cross(ba, nor), pa)) +
                np.sign(np.dot(np.cross(cb, nor), pb)) +
                np.sign(np.dot(np.cross(ac, nor), pc)))

    if sign_sum < 2.0:
        ba_dot_pa = np.dot(ba, pa)
        cb_dot_pb = np.dot(cb, pb)
        ac_dot_pc = np.dot(ac, pc)

        t1 = np.clip(ba_dot_pa / dot2(ba), 0.0, 1.0)
        t2 = np.clip(cb_dot_pb / dot2(cb), 0.0, 1.0)
        t3 = np.clip(ac_dot_pc / dot2(ac), 0.0, 1.0)

        v1 = ba * t1 - pa
        v2 = cb * t2 - pb
        v3 = ac * t3 - pc

        return np.sqrt(min(scalar_dot(v1,v1), scalar_dot(v2,v2), scalar_dot(v3,v3)))
    else:
        nor_dot_pa = np.dot(nor, pa)
        return np.sqrt(nor_dot_pa * nor_dot_pa / scalar_dot(nor,nor))



def support_vector(vertices: np.ndarray, d: np.ndarray) -> tuple[np.ndarray,int]:
    """Support Vector Method for 2D and 3D

    :param vertices: An array of vertices (N x 2 for 2D or N x 3 for 3D)
    :type vertices: np.ndarray

    :param d: A vector (2D or 3D)
    :type d: np.ndarray

    :return: The support vector
    :rtype: np.ndarray
    """
    highest = -sys.float_info.max
    support = np.zeros(d.shape, dtype=float)
    support_i=-1
    for i,v in enumerate(vertices):
        dot_value = np.dot(v, d)

        if dot_value > highest:
            highest = dot_value
            support = v
            support_i=i

    return support,support_i


def gjk_collision_detection(vertices1: np.ndarray, vertices2: np.ndarray, tol=0., max_iter=None) -> bool:
    """GJK Collision Detection Algorithm

    :param vertices1: Vertices of the first convex shape (N x 3 for 3D)
    :param vertices2: Vertices of the second convex shape (M x 3 for 3D)
    :param tol: Tolerance for floating-point comparisons
    :param max_iter: Maximum number of iterations
    :return: True if the convex shapes intersect, False otherwise
    """
    visited = np.zeros((vertices1.shape[0], vertices2.shape[0]), dtype=bool)
    visited_pts = np.zeros((vertices1.shape[0], vertices2.shape[0], 3))

    def support(vertices1, vertices2, d):
        """Compute the support point in Minkowski difference."""
        p1, i1 = support_vector(vertices1, d)
        p2, i2 = support_vector(vertices2, -d)
        return p1 - p2, (i1, i2)

    # Initial direction (arbitrary)
    d = np.array([0.0, 0.0, 1.0])
    p, (i, j) = support(vertices1, vertices2, d)
    visited[i, j] = True
    visited_pts[i, j] = p
    # First point in the simplex
    simplex = [p]
    if max_iter is None:
        max_iter=len(vertices1)*len(vertices2)
    # Negate direction
    d = -simplex[0]

    for _ in range(max_iter):
        # Get the next support point in the direction of d
        new_point, (i, j) = support(vertices1, vertices2, d)

        if visited[i, j]:
            dd = visited_pts[i, j] - new_point
            if scalar_dot(dd, dd)==0:
               return True
           
           
        else:
            visited[i, j] = True
            visited_pts[i, j] = new_point

        # If the new point doesn't pass the origin, no collision
        if scalar_dot(new_point, d) < 0:
            return False

        # Add new point to the simplex
        simplex.append(new_point)

        # Check the simplex
        if handle_simplex(simplex, d, tol):
            return True
    return False
from mmcore.numeric.closest_point import closest_point_on_line


def handle_simplex(simplex, d, tol=1e-6):
    """Handle the current simplex and update direction d."""
    if len(simplex) == 2:  # Line segment case
        a = simplex[1]
        b = simplex[0]
        ab = b - a
        ao = -a
        if scalar_dot(ab, ao) > tol:
            # New direction is perpendicular to AB towards the origin
            d[:] = scalar_cross(scalar_cross(ab, ao), ab)
        else:
            # Remove point B and update direction towards the origin
            simplex.pop(0)
            d[:] = ao

    elif len(simplex) == 3:  # Triangle case
        a = simplex[2]
        b = simplex[1]
        c = simplex[0]
        ab = b - a
        ac = c - a
        ao = -a

        abc = scalar_cross(ab, ac)

        if scalar_dot(scalar_cross(abc, ac), ao) > tol:
            if scalar_dot(ac, ao) > 0:
                simplex.pop(1)
                d[:] = scalar_cross(scalar_cross(ac, ao), ac)
            else:
                return handle_simplex([simplex[2], simplex[1]], d)
        else:
            if scalar_dot(scalar_cross(ab, abc), ao) > tol:
                return handle_simplex([simplex[2], simplex[1]], d)
            else:
                if scalar_dot(abc, ao) > tol:
                    d[:] = abc
                else:
                    simplex[0], simplex[1] = simplex[1], simplex[0]
                    d[:] = -abc
    elif len(simplex) == 4:  # Tetrahedron case
        a = simplex[3]
        b = simplex[2]
        c = simplex[1]
        d_point = simplex[0]

        ab = b - a
        ac = c - a
        ad = d_point - a
        ao = -a

        abc = scalar_cross(ab, ac)
        acd = scalar_cross(ac, ad)
        adb = scalar_cross(ad, ab)

        if scalar_dot(abc, ao) > tol:
            simplex.pop(0)
            return handle_simplex([simplex[2], simplex[1], simplex[0]], d)
        elif scalar_dot(acd, ao) > tol:
            simplex.pop(1)
            return handle_simplex([simplex[2], simplex[1], simplex[0]], d)
        elif scalar_dot(adb, ao) > tol:
            simplex.pop(2)
            return handle_simplex([simplex[2], simplex[1], simplex[0]], d)
        else:
            return True

    return False
