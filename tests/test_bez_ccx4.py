import numpy as np
import pytest
from mmcore.numeric.intersection.ccx._bez_ccx4 import bez_ccx
from mmcore.numeric.intersection.ccx._bez_ccx3 import bez_ccx as bez_ccx_old


# ---------------------------------------------------------------------------
# Shared test fixtures (from test_bez_ccx3_cases.py)
# ---------------------------------------------------------------------------
curve1 = np.array([[-19.77608536, 23.10065701, 0.0], [-14.86834768, 28.69713066, 0.0],
                    [-5.8568525, 25.12677787, 0.0], [-12.62581769, 15.26478654, 0.0]])
curve2 = np.array([[-22.0315362, 18.75969713, 0.0], [-19.42270945, 28.2502867, 0.0],
                    [-8.46791623, 27.56878356, 0.0], [-10.43007782, 19.78973126, 0.0]])
curve3 = np.array([[-28.46565557, -11.09883504, 0.0], [-31.79098016, 13.62423043, 0.0],
                    [-12.99566723, 16.66039636, 0.0], [8.11291498, -6.32771715, 0.0]])
curve4 = np.array([[-45.36434109, -7.12015504, 0.0], [-25.49612403, 13.94186047, 0.0],
                    [-2.13178295, -17.35271318, 0.0], [12.02325581, 20.42248062, 0.0]])


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


# ---------------------------------------------------------------------------
# Comparison tests: _bez_ccx4 (new) vs _bez_ccx3 (old)
# ---------------------------------------------------------------------------

def test_compare_case1_overlap():
    """Old finds 1 overlap for curve1 vs curve2. New should also find overlap."""
    old = bez_ccx_old(curve1, curve2)
    new = bez_ccx(curve1, curve2, atol=1e-3, rational=False)
    assert len(old["overlaps"]) == 1
    # New should find overlap (or at worst, multiple isolated points along the overlap)
    has_overlap = len(new["overlaps"]) >= 1
    has_many_isolated = len(new["isolated"]) >= 2
    assert has_overlap or has_many_isolated, (
        f"New found: {len(new['isolated'])} isolated, {len(new['overlaps'])} overlaps"
    )


def test_compare_case2_two_crossings():
    """Old finds 2 isolated for curve3 vs curve4. New should find same count."""
    old = bez_ccx_old(curve3, curve4)
    new = bez_ccx(curve3, curve4, atol=1e-3, rational=False)
    assert len(old["isolated"]) == 2
    assert len(new["isolated"]) == 2


def test_compare_rational_arc_line():
    """Rational quarter-circle vs line -- both should find 1 isolated."""
    w = np.sqrt(0.5)
    arc = np.array([[1.0, 0.0, 1.0], [w, w, w], [0.0, 1.0, 1.0]])
    line = np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 1.0], [1.0, 1.0, 1.0]])
    old = bez_ccx_old(arc, line, rational=True)
    new = bez_ccx(arc, line, atol=1e-3, rational=True)
    assert len(old["isolated"]) == len(new["isolated"])


def test_tangent_touching():
    """Two curves tangent at one point -- should find exactly 1, not 0 or 2."""
    # Degree-2 Bezier parabola that dips to y=0 at t=0.5:
    #   C1(t) = (1-t)^2*(0,0.1,0) + 2t(1-t)*(0.5,-0.1,0) + t^2*(1,0.1,0)
    #   C1(0.5) = (0.5, 0, 0)  and  C1'(0.5) = (1, 0, 0) (horizontal)
    C1 = np.array([[0.0, 0.1, 0.0], [0.5, -0.1, 0.0], [1.0, 0.1, 0.0]])
    C2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = bez_ccx(C1, C2, atol=1e-3, rational=False)
    # Should find exactly 1 intersection point (tangent)
    assert len(result["isolated"]) == 1
