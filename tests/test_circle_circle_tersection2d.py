import numpy as np
import unittest
from mmcore.geom.circle import circle_intersection2d, Circle, tangents_lines, closest_point_on_circle_2d


class TestCircleIntersection2D(unittest.TestCase):
    def setUp(self):
        # Create 3 Circle object instances
        self.c1 = Circle(5, np.array([0, 0, 0]))
        self.c2 = Circle(6, np.array([7, 0, 0]))
        self.c3 = Circle(5, np.array([16, 0, 0]))

    def test_intersects(self):
        # Test when the circles intersect
        expected_result = np.array([[2.71428571, 4.19912527, 0.], [2.71428571, -4.19912527, 0.]], float)
        np.testing.assert_almost_equal(circle_intersection2d(self.c1, self.c2), expected_result, decimal=5)

    def test_no_intersection(self):
        # Test when circles do not intersect
        with self.assertRaises(ValueError):
            circle_intersection2d(self.c1, self.c3)

    def test_circles_same_center(self):
        # Test with two circles of same center
        same_center_circle = Circle(7, [0., 0., 0.])
        with self.assertRaises(ValueError):
            circle_intersection2d(self.c1, same_center_circle)

    # Test the tangents_lines function
    def test_tangents_lines(self):
        point_tangent = np.array([20, 20, 0.])
        expected_result = np.array([[-2.85485273, 4.10485273, 0.], [4.10485273, -2.85485273, 0.]])
        actual_result = tangents_lines(self.c1, point_tangent)

        np.testing.assert_almost_equal(actual_result, expected_result, decimal=5)

    # Test the closest_point_on_circle_2d function
    def test_closest_point_on_circle_2d(self):
        point_closest = np.array([20, 20, 0], float)
        expected_result_t = 0.7853981633974483

        actual_result = closest_point_on_circle_2d(self.c1, point_closest)
        np.testing.assert_almost_equal(actual_result, expected_result_t, decimal=5)

    def test_evaluate_circle(self):
        t = 0.7853981633974483
        expected_result_point = np.array([3.53553391, 3.53553391, 0.])
        np.testing.assert_almost_equal(self.c1(t), expected_result_point, decimal=5)


if __name__ == "__main__":
    unittest.main()
