from typing import Union

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


from mmcore.geom.nurbs import NURBSCurve, NURBSSurface

from mmcore.collision import CGJK
from mmcore.numeric.aabb import aabb_intersect,aabb
from mmcore.numeric.algorithms.cygjk import gjk


__all__=['spatial_separability','SpatialSeparabilityTest']
class SpatialSeparabilityTest:
    def __init__(self, object1, object2):
        self.object1 = object1
        self.object2 = object2
        self.bounding_points1 = self.get_bounding_points(object1)
        self.bounding_points2 = self.get_bounding_points(object2)
        self.bbox1 = self._compute_bounding_box(self.bounding_points1)
        self.bbox2 = self._compute_bounding_box(self.bounding_points2)


    def _compute_bounding_box(self, points):
        return aabb(points)

    def bounding_box_test(self):
        return not aabb_intersect(self.bbox1, self.bbox2)
    @staticmethod
    def get_bounding_points(self:Union[NURBSSurface,NURBSCurve]):

        return self.control_points_flat[self._vertices]

    @staticmethod
    def get_corner_points(self:NURBSSurface):
        return self.control_points[[0, -1, -1, 0], [0, 0, -1, -1]]

    def bounding_plane_test(self):
        if isinstance(self.object1, Surface) and isinstance(self.object2, Surface):
            normal1 = self._compute_bounding_plane_normal(self.object1)
            normal2 = self._compute_bounding_plane_normal(self.object2)

            return self._plane_separates(
                normal1, self.bounding_points1, self.bounding_points2
            ) or self._plane_separates(
                normal2, self.bounding_points2, self.bounding_points1
            )
        return False

    def _compute_bounding_plane_normal(self, surface):
        A, B, C, D = self.get_corner_points(surface)
        return np.cross(C - A, D - B)

    def _plane_separates(self, normal, points1, points2):
        d = -np.dot(normal, np.mean(points1, axis=0))
        return np.all(np.dot(points1, normal) + d >= 0) and np.all(
            np.dot(points2, normal) + d <= 0
        )

    def separating_plane_test(self, epsilon=1e-8):
        n = len(self.bounding_points1)
        m = len(self.bounding_points2)

        # Construct the linear programming problem
        c = [0, 0, 0, 0, 1]  # Minimize epsilon
        A_ub = np.zeros((n + m, 5))
        b_ub = np.zeros(n + m)

        # Constraints for object1
        A_ub[:n, :3] = -self.bounding_points1
        A_ub[:n, 3] = -1
        A_ub[:n, 4] = -1

        # Constraints for object2
        A_ub[n:, :3] = self.bounding_points2
        A_ub[n:, 3] = 1
        A_ub[n:, 4] = -1

        # Solve the linear programming problem
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')

        if result.success:
            # A solution was found, but we need to check if it's a valid separating plane
            a, b, c, d, lp_epsilon = result.x
            plane_normal = np.array([a, b, c])
            norm = np.linalg.norm(plane_normal)

            if norm < epsilon:  # Check if the normal vector is essentially zero
                return False, None

            plane_normal /= norm  # Normalize safely

            # Check if the plane actually separates the points
            dist1 = np.dot(self.bounding_points1, plane_normal) + d
            dist2 = np.dot(self.bounding_points2, plane_normal) + d

            # Use the provided epsilon for separation check
            if np.all(dist1 >= -epsilon) and np.all(dist2 <= epsilon):
                return True, (plane_normal, d, lp_epsilon)

        # No separating plane found
        return False, None

    def are_separable(self):
        # Perform tests in order of increasing computational cost
        if self.bounding_box_test():
            return True, "Bounding Box"
        if self.bounding_plane_test():
            return True, "Bounding Plane"
        separable, plane_info = self.separating_plane_test()
        if separable:
            return True, "Separating Plane"
        return False, "All"


class GeometricObject:
    def get_bounding_points(self):
        raise NotImplementedError("Subclass must implement abstract method")


class Surface(GeometricObject):
    def __init__(self, control_points):
        self.control_points = np.array(control_points,dtype=float)
        self.control_points_flat =self.control_points.reshape((-1,3))
        self.ch=ConvexHull(self.control_points_flat,qhull_options='QJ'  )

        self._vertices=self.ch.vertices

    def get_bounding_points(self):
        return self.control_points_flat[self._vertices]

    def get_corner_points(self):
        return self.control_points[[0, -1, -1, 0], [0, 0, -1, -1]]


class Curve(GeometricObject):
    def __init__(self, control_points):
        self.control_points = np.array(control_points)

    def get_bounding_points(self):
        return self.control_points


def spatial_separability(points1, points2, tol=1e-6):
    """
    Test for spatial separability of two sets of points using axis-aligned bounding boxes and the GJK algorithm.

    This algorithm is based on the spatial separability method discussed in section 4.2 of
    "Robust and Efficient Surface Intersection for Solid Modeling" by Michael Edward Hohmeyer B.A.

    **Key Differences:**

    - Instead of the linear solver originally proposed by Hohmeyer, this implementation uses the GJK (Gilbert-Johnson-Keerthi) algorithm as a more efficient test.
    - GJK is both faster and can handle more cases of separability compared to the linear solver.
    - The original :class:`SpatialSeparabilityTest` class is included for reference and comparison,
    but is not well-suited for direct API use with flat sets of control points. The `spatial_separability`
    function adapts the approach for simpler use cases.

    :param points1:
        A set of points (control points) representing the first geometric object.
        Expected shape is (n, 3), where n is the number of points.
    :type points1: np.ndarray

    :param points2:
        A set of points (control points) representing the second geometric object.
        Expected shape is (m, 3), where m is the number of points.
    :type points2: np.ndarray

    :param tol:
        Tolerance for the GJK algorithm. Defaults to 1e-8.
    :type tol: float, optional

    :return:
        True if the two sets of points are spatially separable, meaning there is no intersection between them.
    :rtype: bool

    **Notes:**

    - The method first checks the axis-aligned bounding boxes (AABB) of both point sets.
    - If the AABBs do not intersect, the objects are immediately considered separable.
    - If the AABBs intersect, the GJK algorithm is used as an additional, more thorough test for separability.

    **Examples:**

    .. code-block:: python

        >>> points1 = np.random.rand(10, 3)  # Random set of points representing the first object
        >>> points2 = np.random.rand(8, 3)   # Random set of points representing the second object
        >>> result = spatial_separability(points1, points2)
        >>> print(result)
        True
    """
    # Compute the axis-aligned bounding boxes (AABB) for both point sets
    bb1 = aabb(points1)
    bb2 = aabb(points2)

    # If bounding boxes do not intersect, the objects are separable
    if not aabb_intersect(bb1, bb2):
        return True

    # Use the GJK algorithm for a more detailed check of separability
    gjk_res = gjk(points1, points2, tol=tol)
    if not gjk_res:
        return True

    return False


# Example usage
if __name__ == "__main__":
    # Create two non-intersecting surfaces
    import time

    surface1 = Surface(
        [
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            [[0, 1, 0], [1, 1, 0], [2, 1, 0]],
            [[0, 2, 0], [1, 2, 0], [2, 2, 0]],
        ]
    )
    surface2 = Surface(
        [
            [[0, 0, 2], [1, 0, 2], [2, 0, 2]],
            [[0, 1, 2], [1, 1, 2], [2, 1, 2]],
            [[0, 2, 2], [1, 2, 2], [2, 2, 2]],
        ]
    )

    test = SpatialSeparabilityTest(surface1, surface2)
    s = time.time()
    separable, method = test.are_separable()

    e = time.time() - s
    s1 = time.time()
    res = spatial_separability(surface1.control_points_flat, surface2.control_points_flat)
    e1 = time.time() - s1
    print(e / e1)
    print(f"Surfaces are separable: {separable}, Method: {method},gjk: {not res}")
    # Create two intersecting surfaces
    surface3 = Surface(
        [
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            [[0, 1, 1], [1, 1, 1], [2, 1, 1]],
            [[0, 2, 2], [1, 2, 2], [2, 2, 2]],
        ]
    )
    surface4 = Surface(
        [
            [[0, 0, 2], [1, 0, 1], [2, 0, 0]],
            [[0, 1, 2], [1, 1, 1], [2, 1, 0]],
            [[0, 2, 2], [1, 2, 1], [2, 2, 0]],
        ]
    )


    test = SpatialSeparabilityTest(surface3, surface4)
    separable, method = test.are_separable()
    print(f"Surfaces are separable: {separable}, Method: {method}")

    surface5=Surface([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
     [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]]])
    surface6=Surface([[[0.051617261024058142, 0.0, 2.0], [1.0, 0.0, 1.6212732869464466], [2.0, 0.0, 2.0]],
     [[0.0, 1.0, 0.47259775752978284], [1.0, 1.0, -4.4378567664790083], [2.0, 1.0, 0.61138510325914019]],
     [[0.0, 2.0, 2.0], [1.0, 2.0, -0.082598633863934179], [2.0, 2.0, 2.0]]])
    import time

    test = SpatialSeparabilityTest(surface5, surface6)
    s = time.time()
    separable, method = test.are_separable()
    e=time.time()-s
    s1=time.time()
    res = gjk(surface5.control_points_flat, surface6.control_points_flat)
    e1=time.time()-s1
    print(e/e1)
    print(f"Surfaces are separable: {separable}, Method: {method}, gjk: {not res}")

    a,b=[[[[0.0, 0.0, -1.2947487788397152], [1.0, 0.0, -0.81252815149489144], [2.0, 0.0, -1.4446643721252055]], [[0.0, 1.0, -0.98598375375584846], [1.0, 1.0, -0.9921064360866293], [2.0, 1.6473260568266372, 0.17404336620182415]], [[0.0, 2.0, -0.93772529590905662], [1.0, 2.0, -0.43782477195170255], [2.0, 2.0, -1.4881378355792421]]], [[[0.051617261024058142, 0.0, 2.0], [1.0, 0.0, 1.7330775484965009], [2.0, 0.0, 2.0]], [[0.0, 1.0, -0.039388071952865825], [1.0, 1.0, 1.8951728670530650], [2.0, 1.0, 1.9720578133560540]], [[0.0, 2.0, 2.0], [1.0, 2.0, 1.9348661335911177], [2.0, 2.0, 2.0]]]]

    surface7=Surface(a)
    surface8 = Surface(b)


    test = SpatialSeparabilityTest(surface7, surface8)
    s = time.time()
    separable, method = test.are_separable()
    e = time.time() - s
    s1=time.time()

    res=gjk(surface7.control_points_flat,surface8.control_points_flat)
    e1=time.time()-s1
    print(e / e1)
    print(f"Surfaces are separable: {separable}, Method: {method}, gjk: {not res}")