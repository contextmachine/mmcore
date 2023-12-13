import unittest
import numpy as np
from mmcore.geom.parametric.algorithms import polygon_variable_offset


class TestPolygonVariableOffset(unittest.TestCase):

    def setUp(self):
        self.pts = np.array([[33.049793, -229.883303, 0],
                             [132.290583, -165.409427, 0],
                             [48.220282, 27.548631, 0],
                             [-115.077733, -43.599024, 0],
                             [-44.627307, -205.296759, 0]])
        self.dists = np.zeros((5, 2))
        self.dists[0] = -4
        self.dists[2] = -1
        self.dists[-1] = -2

    def test_polygon_variable_offset(self):
        *res, = polygon_variable_offset(self.pts, self.dists)
        np_result = np.array(res)

        expected_result = np.array([[30.28406152, -226.91009151, 0.],
                                    [130.67080779, -161.69172081, 0.],
                                    [48.61970922, 26.63186609, 0.],
                                    [-114.67830577, -44.5157889, 0.],
                                    [-45.68750838, -202.86338603, 0.]])
        np.testing.assert_almost_equal(np_result, expected_result, decimal=5)

    def test_polygon_variable_offset_with_variable_value(self):
        dists = np.array([[-4., -2.],
                          [0., 0.],
                          [-1., -1.],
                          [0., 0.],
                          [-2., -2.]])  # To set a variable offset per side change one of the values.
        *res, = polygon_variable_offset(self.pts, dists)
        np_result = np.array(res)
        expected_result = np.array([[30.26926633, -226.9054085, 0.],
                                    [131.48297158, -163.55579819, 0.],
                                    [48.61970922, 26.63186609, 0.],
                                    [-114.67830577, -44.5157889, 0.],
                                    [-45.68750838, -202.86338603, 0.]])
        np.testing.assert_almost_equal(np_result, expected_result, decimal=5)

    def test_empty_points(self):
        with self.assertRaises(ValueError):
            list(polygon_variable_offset(np.array([]), self.dists))

    def test_non_2D_points(self):
        with self.assertRaises(ValueError):
            from mmcore.geom.parametric.algorithms import polygon_variable_offset
            list(polygon_variable_offset(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), self.dists))

    def test_incorrect_dists(self):
        with self.assertRaises(ValueError):
            list(polygon_variable_offset(self.pts, np.array([1, 2, 3])))
