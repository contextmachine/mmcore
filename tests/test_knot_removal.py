"""Tests for the A5.8 knot-removal fix in mmcore.nurbs._nurbs_knots.

The previous implementation was an incorrect transcription of Algorithm A5.8
(wrong `while` bound, wrong alpha denominator, broken incremental bookkeeping):
it removed nothing yet reported success.  These tests pin the corrected
behaviour: insert k copies of a knot then remove them must restore the original
curve exactly (at any coordinate scale), a genuine corner must NOT be removed,
and the reported count must equal the number actually removed.
"""
import numpy as np
import pytest

from mmcore.nurbs._nurbs_eval import NURBSCurveTuple, NURBSSurfaceTuple, evaluate_nurbs_curve
from mmcore.nurbs._nurbs_knots import (
    insert_knot_curve, remove_knot_curve_max,
    insert_knot_surface_u, remove_knot_surface_u,
)


def _curve_max_err(a, b, n=21):
    return max(
        np.linalg.norm(np.asarray(evaluate_nurbs_curve(a, t)["C"]) -
                       np.asarray(evaluate_nurbs_curve(b, t)["C"]))
        for t in np.linspace(0, 1, n)
    )


@pytest.mark.parametrize("scale", [1.0, 1e3, 1e6])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_insert_then_remove_roundtrip(scale, k):
    """Insert k copies of an interior knot, then remove them → original curve."""
    base = NURBSCurveTuple(
        order=4, knot=np.array([0, 0, 0, 0, 1, 1, 1, 1.0]),
        control_points=np.array([[0, 0, 0], [1, 2, 0], [2, 2, 0], [3, 0, 0.0]]) * scale,
        weights=np.ones(4),
    )
    inserted = insert_knot_curve(base, 0.5, num=k)
    back, removed = remove_knot_curve_max(inserted, 0.5, num=k)

    assert removed == k                                   # reported count is honest
    assert back.control_points.shape[0] == base.control_points.shape[0]
    assert np.sum(np.isclose(back.knot, 0.5)) == 0        # interior knot fully gone
    assert _curve_max_err(back, base) <= 1e-7 * scale     # geometry unchanged


def test_genuine_corner_is_not_removed():
    """A C0 corner at a full-multiplicity knot must survive removal attempts."""
    kink = NURBSCurveTuple(
        order=3, knot=np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1.0]),
        control_points=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                                 [2, 1, 0], [2, 0, 0], [3, 0, 0.0]]),
        weights=np.ones(6),
    )
    before = kink.control_points.shape[0]
    back, removed = remove_knot_curve_max(kink, 0.5, num=3)
    assert removed == 0
    assert back.control_points.shape[0] == before
    assert _curve_max_err(back, kink) == 0.0


def test_surface_u_knot_roundtrip():
    surf = NURBSSurfaceTuple(
        order_u=3, order_v=3,
        knot_u=np.array([0, 0, 0, 1, 1, 1.0]), knot_v=np.array([0, 0, 0, 1, 1, 1.0]),
        control_points=np.random.RandomState(0).rand(3, 3, 3), weights=np.ones((3, 3)),
    )
    inserted = insert_knot_surface_u(surf, 0.5, num=2)
    assert inserted.control_points.shape[0] == 5
    removed = remove_knot_surface_u(inserted, 0.5, num=2)
    assert removed.control_points.shape[0] == 3            # back to the original count
