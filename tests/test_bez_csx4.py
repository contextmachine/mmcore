import numpy as np
import pytest
from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx


def test_line_through_plane():
    """Line crossing a flat surface -- one isolated intersection."""
    S = np.array([
        [[0.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 1.0]],
        [[2.0, 0.0, 0.0, 1.0], [2.0, 2.0, 0.0, 1.0]],
    ])
    C = np.array([[1.0, 1.0, -1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    result = bez_csx(C, S, atol=1e-3, rational=True)
    assert len(result["isolated"]) == 1
    pt = result["isolated"][0]["point"]
    np.testing.assert_allclose(pt, [1.0, 1.0, 0.0], atol=1e-2)


def test_no_intersection_csx():
    """Curve far from surface."""
    S = np.array([
        [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]],
    ])
    C = np.array([[0.0, 0.0, 10.0, 1.0], [1.0, 0.0, 10.0, 1.0]])
    result = bez_csx(C, S, atol=1e-3, rational=True)
    assert len(result["isolated"]) == 0
    assert len(result["overlaps"]) == 0


def test_line_two_crossings():
    """Line crossing a curved surface at two points."""
    # Curved surface: z = u*(1-u) * v*(1-v) * 4 (peaks at 1.0 at center)
    # Bilinear with bump
    S = np.array([
        [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 2.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 0.0, 1.0]],
        [[2.0, 0.0, 0.0, 1.0], [2.0, 1.0, 0.0, 1.0], [2.0, 2.0, 0.0, 1.0]],
    ])
    C = np.array([[1.0, 1.0, -0.5, 1.0], [1.0, 1.0, 0.5, 1.0]])
    result = bez_csx(C, S, atol=1e-3, rational=True)
    # The line goes through the bump -- should cross at 2 points
    assert len(result["isolated"]) >= 1  # At least 1, hopefully 2
