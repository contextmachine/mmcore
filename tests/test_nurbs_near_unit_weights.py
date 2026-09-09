"""Knot changes must preserve rational representation, even near unit weights."""
import numpy as np
import pytest

from mmcore.nurbs._nurbs_eval import NURBSCurveTuple, evaluate_nurbs_curve
from mmcore.nurbs._nurbs_knots import insert_knot_curve, split_curve


@pytest.mark.parametrize('delta', [8e-6, 1e-10])
def test_near_unit_rational_curve_survives_knot_insertion_and_split(delta):
    curve = NURBSCurveTuple(
        order=3, knot=np.array([0., 0., 0., 1., 1., 1.]),
        control_points=np.array([[-1., 0., 1.], [0., 0., -1.], [1., 0., 1.]]),
        weights=np.array([1., 1. + delta, 1.]))
    inserted = insert_knot_curve(curve, .5)
    halves = split_curve(curve, .5)
    for t in np.linspace(0., 1., 33):
        # Analytic rational formula, independent of NURBS decomposition.
        b = np.array([(1. - t)**2, 2. * t * (1. - t), t*t])
        expected = (b * curve.weights) @ curve.control_points / (b @ curve.weights)
        for representation in (curve, inserted, halves[0 if t <= .5 else 1]):
            np.testing.assert_allclose(evaluate_nurbs_curve(representation, float(t))['C'],
                                       expected, atol=5e-15, rtol=0.)
