import unittest
import numpy as np
from mmcore.geom.line import Line, closest_parameter, closest_point, evaluate_line


class TestLine(unittest.TestCase):
    def setUp(self):
        self.line = Line(start=np.array([0.0, 1.0, 2.0]), direction=np.array([1.0, 2.0, 3.0]))

    def test_direction(self):
        self.assertEqual(self.line.direction.tolist(), [1.0, 2.0, 3.0])

    def test_start(self):
        self.assertEqual(self.line.start.tolist(), [0.0, 1.0, 2.0])

    def test_replace_ends(self):
        self.line.replace_ends(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
        self.assertEqual(self.line.start.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(self.line.end.tolist(), [2.0, 3.0, 4.0])


class TestFunctions(unittest.TestCase):
    def test_closest_parameter(self):
        result = closest_parameter(np.array([1.0, 2.0]), np.array([2.0, 3.0]), np.array([1.5, 2.5]))
        self.assertTrue(np.isclose(result, 0.5))

    def test_closest_point(self):
        starts = np.array([[0.0, 0.0], [1.0, 1.0]])
        ends = np.array([[1.0, 1.0], [2.0, 2.0]])
        pts = np.array([[0.5, 0.5], [1.5, 1.5]])
        result = closest_point(starts, ends, pts)
        self.assertTrue(np.allclose(result, np.array([[[0.5, 0.5], [1.5, 1.5]], [[0.5, 0.5], [1.5, 1.5]]])))

    def test_evaluate_line(self):
        result = evaluate_line(np.array([1.0, 2.0]), np.array([2.0, 3.0]), 0.5)
        self.assertEqual(result.tolist(), [1.5, 2.5])


if __name__ == '__main__':
    unittest.main()
