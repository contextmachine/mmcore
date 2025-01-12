import numpy as np
import pytest

from mmcore.geom.nurbs import NURBSCurve
from mmcore.geom.curves.curve_bool import (

    curve_boolean_operation
)

from more_itertools import flatten
def test_boolean_operations():
    # Create two overlapping squares
    square1_pts = np.array([
        [0., 0., 0.],
        [2., 0., 0.],
        [2., 2., 0.],
        [0., 2., 0.],
        [0., 0., 0.]
    ])

    square2_pts = np.array([
        [1., 1., 0.],
        [3., 1., 0.],
        [3., 3., 0.],
        [1., 3., 0.],
        [1., 1., 0.]
    ])

    square1 = NURBSCurve(square1_pts, degree=1)
    square2 = NURBSCurve(square2_pts, degree=1)

    # Test union
    union = curve_boolean_operation(square1, square2, 0)

    assert np.allclose(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0],
     [2.0, 1.0, 0.0], [3.0, 1.0, 0.0], [3.0, 3.0, 0.0], [1.0, 3.0, 0.0], [1.0, 2.0, 0.0]]),
                       list(flatten(c.control_points.tolist() for c in union)))
    # Test intersection
    intersection = curve_boolean_operation(square1, square2, 1)
    assert np.allclose(
        np.array([[2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [1.0, 2.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],
         [1.0, 2.0, 0.0], [1.0, 1.0, 0.0]])
        , list(flatten(c.control_points.tolist() for c in intersection)))
    # Test difference
    difference = curve_boolean_operation(square1, square2, 2)
    assert np.allclose(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [1.0, 1.0, 0.0]])
        , list(flatten(c.control_points.tolist() for c in difference)))
    # Test invalid operation
    with pytest.raises(ValueError):
        curve_boolean_operation(square1, square2, 3)