import numpy as np
import pytest
from mmcore.numeric.intersection.ccx._bez_ccx4 import bez_ccx


def test_two_transversal_crossings():
    # Parabola peaking at y=1 vs horizontal line at y=0.5
    # Two transversal crossings near x=0.29 and x=1.71
    C1 = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]])
    C2 = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0], [2.0, 0.5, 0.0]])
    result = bez_ccx(C1, C2, atol=1e-3, rational=False)
    assert len(result["isolated"]) == 2
    assert len(result["overlaps"]) == 0
    for iso in result["isolated"]:
        pt = iso["point"]
        pt2 = np.array(pt)  # should be valid 3D point
        assert pt2.shape == (3,)
        # Both intersection points should have y ~ 0.5
        assert abs(pt2[1] - 0.5) < 1e-2


def test_no_intersection():
    C1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    C2 = np.array([[0.0, 10.0, 0.0], [1.0, 10.0, 0.0]])
    result = bez_ccx(C1, C2, atol=1e-3, rational=False)
    assert len(result["isolated"]) == 0
    assert len(result["overlaps"]) == 0


def test_identical_curves_overlap():
    C1 = np.array([[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]])
    C2 = C1.copy()
    result = bez_ccx(C1, C2, atol=1e-3, rational=False)
    assert len(result["overlaps"]) >= 1


def test_rational_arc_line():
    w = np.sqrt(0.5)
    arc = np.array([[1.0, 0.0, 1.0], [w, w, w], [0.0, 1.0, 1.0]])
    line = np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 1.0], [1.0, 1.0, 1.0]])
    result = bez_ccx(arc, line, atol=1e-3, rational=True)
    assert len(result["isolated"]) == 1
