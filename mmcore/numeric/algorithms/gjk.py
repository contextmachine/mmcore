import numpy as np
import sys

def support_vector(vertices: np.ndarray, d: np.ndarray) -> np.ndarray:
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

    for v in vertices:
        dot_value = np.dot(v, d)

        if dot_value > highest:
            highest = dot_value
            support = v

    return support


def gjk_collision_detection(vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
    """GJK Collision Detection Algorithm

    :param vertices1: Vertices of the first convex shape (N x 3 for 3D)
    :type vertices1: np.ndarray

    :param vertices2: Vertices of the second convex shape (M x 3 for 3D)
    :type vertices2: np.ndarray

    :return: True if the convex shapes intersect, False otherwise
    :rtype: bool
    """

    def support(vertices1, vertices2, d):
        """Compute the support point in Minkowski difference."""
        p1 = support_vector(vertices1, d)
        p2 = support_vector(vertices2, -d)
        return p1 - p2

    # Initial direction (arbitrary)
    d = np.array([1.0, 0.0, 0.0])

    # First point in the simplex
    simplex = [support(vertices1, vertices2, d)]

    # Negate direction
    d = -simplex[0]
    i=0
    l=len(vertices1)
    while True:
        if i>l:
            return False
        # Get the next support point in the direction of d
        new_point = support(vertices1, vertices2, d)

        # If the new point doesn't pass the origin, no collision

        if np.dot(new_point, d) <= 0:
            return False


        # Add new point to the simplex
        simplex.append(new_point)

        # Check the simplex

        if handle_simplex(simplex, d):
            return True
        i += 1

def handle_simplex(simplex, d):
    """Handle the current simplex and update direction d."""
    if len(simplex) == 2:  # Line segment case
        a = simplex[1]
        b = simplex[0]
        ab = b - a
        ao = -a
        if np.dot(ab, ao) > 0:
            # New direction is perpendicular to AB towards the origin
            d[:] = np.cross(np.cross(ab, ao), ab)
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

        abc = np.cross(ab, ac)

        if np.dot(np.cross(abc, ac), ao) > 0:
            if np.dot(ac, ao) > 0:
                simplex.pop(1)
                d[:] = np.cross(np.cross(ac, ao), ac)
            else:
                return handle_simplex([simplex[2], simplex[1]], d)
        else:
            if np.dot(np.cross(ab, abc), ao) > 0:
                return handle_simplex([simplex[2], simplex[1]], d)
            else:
                if np.dot(abc, ao) > 0:
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

        abc = np.cross(ab, ac)
        acd = np.cross(ac, ad)
        adb = np.cross(ad, ab)

        if np.dot(abc, ao) > 0:
            simplex.pop(0)
            return handle_simplex([simplex[2], simplex[1], simplex[0]], d)
        elif np.dot(acd, ao) > 0:
            simplex.pop(1)
            return handle_simplex([simplex[2], simplex[1], simplex[0]], d)
        elif np.dot(adb, ao) > 0:
            simplex.pop(2)
            return handle_simplex([simplex[2], simplex[1], simplex[0]], d)
        else:
            return True

    return False
