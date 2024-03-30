import unittest
import numpy as np
from mmcore.numeric.numeric import  evaluate_curvature


class TestEvaluateCurvature2(unittest.TestCase):
    def compromiser(self, one, two):
        for a, b in zip(one, two):
            self.assertTrue(np.allclose(np.atleast_1d(a), np.atleast_1d(b)))

    def test_evaluate_curvature_zero_derivative(self):
        derivative = np.array([0, 0, 0])
        second_derivative = np.array([0, 0, 0])
        expected_output = (np.array([0, 0, 0]), np.array([0, 0, 0]), False)

        self.compromiser(
            evaluate_curvature(derivative, second_derivative), expected_output
        )

    def test_evaluate_curvature_nonzero_derivative(self):
        derivative = np.array([1, 0, 0])
        second_derivative = np.array([0, 1, 0])
        expected_output = (np.array([1, 0, 0]), np.array([0, 1, 0]), True)
        self.compromiser(
            evaluate_curvature(derivative, second_derivative), expected_output
        )

    def test_evaluate_curvature_zero_second_derivative(self):
        derivative = np.array([1, 0, 0])
        second_derivative = np.array([0, 0, 0])
        expected_output = (derivative, np.array([0, 0, 0]), True)
        self.compromiser(
            evaluate_curvature(derivative, second_derivative), expected_output
        )

    def test_evaluate_curvature_unit_normal_curvature(self):
        derivative = np.array([1, 0, 0])
        second_derivative = np.array([0, 1, 0])
        result = evaluate_curvature(derivative, second_derivative)
        norm_curvature_vector = np.linalg.norm(result[1])
        self.assertAlmostEqual(norm_curvature_vector, 1.0)


    def test_with_old(self):
        try:
            from mmcore.numeric import evaluate_curvature2
            derivative = np.random.random(3)

            second_derivative = np.random.random(3)*2
            self.compromiser(
                evaluate_curvature2(derivative, second_derivative), evaluate_curvature(derivative, second_derivative)
            )

        except ImportError:
            pass






if __name__ == "__main__":
    unittest.main()
