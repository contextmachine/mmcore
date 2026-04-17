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


import time


def test_tangent_curve_on_surface():
    """Curve tangent to surface at one point -- should find exactly 1 intersection."""
    S = np.array([
        [[0.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 1.0]],
        [[2.0, 0.0, 0.0, 1.0], [2.0, 2.0, 0.0, 1.0]],
    ])
    # Degree-2 parabola touching z=0 at t=0.5: B(0.5) = (P0 + 2*P1 + P2)/4
    # With P0_z=P2_z=0.5, P1_z=-0.5 => B(0.5)_z = (0.5 - 1.0 + 0.5)/4 = 0
    C = np.array([[1.0, 1.0, 0.5, 1.0], [1.0, 1.0, -0.5, 1.0], [1.0, 1.0, 0.5, 1.0]])
    result = bez_csx(C, S, atol=1e-3, rational=True)
    assert len(result["isolated"]) == 1


def test_rational_arc_surface():
    """Rational arc intersecting a flat surface."""
    w = np.sqrt(0.5)
    C = np.array([[1.0, 0.0, 0.0, 1.0], [w, 0.0, w, w], [0.0, 0.0, 1.0, 1.0]])
    S = np.array([
        [[-1.0, -1.0, 0.5, 1.0], [-1.0, 1.0, 0.5, 1.0]],
        [[2.0, -1.0, 0.5, 1.0], [2.0, 1.0, 0.5, 1.0]],
    ])
    result = bez_csx(C, S, atol=1e-3, rational=True)
    assert len(result["isolated"]) == 1
    pt = result["isolated"][0]["point"]
    assert abs(pt[2] - 0.5) < 0.05


def test_no_false_positives_csx():
    """Every reported intersection must have actual distance < atol."""
    from mmcore.numeric.intersection._bezier_common import eval_surface
    # Line through a curved rational surface — multiple intersections
    w = np.sqrt(0.5)
    S = np.array([
        [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 2.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.5, 1.0], [1.0, 1.0, 0.5, 1.0], [1.0, 2.0, 0.5, 1.0]],
        [[2.0, 0.0, 0.0, 1.0], [2.0, 1.0, 0.0, 1.0], [2.0, 2.0, 0.0, 1.0]],
    ])
    C = np.array([[1.0, 1.0, -0.5, 1.0], [1.0, 1.0, 0.5, 1.0]])
    result = bez_csx(C, S, atol=1e-3, rational=True)
    for iso in result["isolated"]:
        from mmcore.numeric.intersection._bezier_common import eval_curve
        pt_c = eval_curve(C, iso["t"], rational=True)
        pt_s = eval_surface(S, iso["u"], iso["v"], rational=True)
        dist = float(np.linalg.norm(pt_c - pt_s))
        assert dist < 1e-3, f"False positive: t={iso['t']:.4f} u={iso['u']:.4f} v={iso['v']:.4f} dist={dist:.4f}"


def test_csx_cell_count_reasonable():
    """Verify the new CSX doesn't blow up on a simple case."""
    S = np.array([
        [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 2.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.5, 1.0], [1.0, 1.0, 0.5, 1.0], [1.0, 2.0, 0.5, 1.0]],
        [[2.0, 0.0, 0.0, 1.0], [2.0, 1.0, 0.0, 1.0], [2.0, 2.0, 0.0, 1.0]],
    ])
    C = np.array([[1.0, 1.0, -1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    t0 = time.perf_counter()
    result = bez_csx(C, S, atol=1e-3, rational=True)
    elapsed = time.perf_counter() - t0
    assert len(result["isolated"]) >= 1
    assert elapsed < 5.0, f"CSX took {elapsed:.2f}s -- possible infinite subdivision"


# ---------------------------------------------------------------------------
# Line vs plane regression tests
#
# These cover the Newton-stalled-at-stationary-point false positive where
# a degree-1 (line) curve evaluated against a bilinear surface produced an
# "isolated intersection" at the minimum-distance point even when no actual
# intersection existed.  Regression for the bug fix in _phase2_isolated_search.
# ---------------------------------------------------------------------------

def test_line_parallel_to_plane_no_intersection():
    """Line parallel to the x=0 axis at z=5 vs tilted plane z=x.

    Line: (0, 0, 5) -> (0, 10, 5)  — has x=0, z=5 everywhere.
    Surface: bilinear with z = 10u, x = 10u  — requires x=z=5 at u=0.5.
    These never meet: line has x=0 but surface at z=5 needs x=5.
    """
    C = np.array([[0., 0., 5.], [0., 10., 5.]])
    S = np.array([[[0., 0., 0.], [0., 10., 0.]], [[10., 0., 10.], [10., 10., 10.]]])
    result = bez_csx(C, S, atol=1e-3, rational=False)
    assert len(result["isolated"]) == 0, (
        f"False positive — line x=0 z=5 doesn't meet surface z=x: "
        f"got {result['isolated']}"
    )
    assert len(result["overlaps"]) == 0


def test_line_crossing_plane_one_intersection():
    """Line orthogonal to a flat plane — exactly one isolated intersection."""
    # Flat plane at z=5 (bilinear, all corners z=5)
    S = np.array([[[0., 0., 5.], [0., 10., 5.]], [[10., 0., 5.], [10., 10., 5.]]])
    # Vertical line through (5, 5, *) from z=0 to z=10
    C = np.array([[5., 5., 0.], [5., 5., 10.]])
    result = bez_csx(C, S, atol=1e-3, rational=False)
    assert len(result["isolated"]) == 1
    iso = result["isolated"][0]
    # Verify geometric match
    from mmcore.numeric.intersection._bezier_common import eval_curve, eval_surface
    pt_c = eval_curve(C, iso["t"], rational=False)
    pt_s = eval_surface(S, iso["u"], iso["v"], rational=False)
    assert float(np.linalg.norm(pt_c - pt_s)) < 1e-3
    np.testing.assert_allclose(pt_c, [5., 5., 5.], atol=1e-3)


def test_line_lying_on_plane_detected_as_overlap():
    """Line that lies entirely on the surface — detected as overlap, not isolated."""
    # Tilted plane z = x
    S = np.array([[[0., 0., 0.], [0., 10., 0.]], [[10., 0., 10.], [10., 10., 10.]]])
    # Line x=5, z=5, y varies — satisfies z=x at x=z=5 for all y
    C = np.array([[5., 0., 5.], [5., 10., 5.]])
    result = bez_csx(C, S, atol=1e-3, rational=False)
    assert len(result["isolated"]) == 0
    assert len(result["overlaps"]) >= 1, "Line on surface should be detected as overlap"


def test_line_near_plane_not_intersecting_no_false_positive():
    """Line close to but not touching a surface — Newton may stall at minimum distance."""
    # Flat plane at z=5
    S = np.array([[[0., 0., 5.], [0., 10., 5.]], [[10., 0., 5.], [10., 10., 5.]]])
    # Line at z=4.9 — close to the plane but doesn't touch (gap = 0.1 >> atol=1e-3)
    C = np.array([[0., 0., 4.9], [10., 10., 4.9]])
    result = bez_csx(C, S, atol=1e-3, rational=False)
    # Newton will converge to minimum distance = 0.1, but this must NOT be reported
    assert len(result["isolated"]) == 0, (
        f"False positive — line at z=4.9 vs plane z=5: "
        f"gap is 0.1, atol is 1e-3, but got {result['isolated']}"
    )


def test_degree_one_line_no_false_positive_variants():
    """Several line-vs-bilinear configurations with no actual intersection."""
    # Tilted plane z = x
    S = np.array([[[0., 0., 0.], [0., 10., 0.]], [[10., 0., 10.], [10., 10., 10.]]])

    # Line parallel to y-axis at various (x, z) that don't satisfy z=x
    test_lines = [
        ([[0., 0., 5.], [0., 10., 5.]], "x=0 z=5"),   # user's original repro
        ([[2., 0., 7.], [2., 10., 7.]], "x=2 z=7"),
        ([[8., 0., 3.], [8., 10., 3.]], "x=8 z=3"),
        ([[-1., 0., 5.], [-1., 10., 5.]], "x=-1 (outside surface)"),
    ]
    for cpts, label in test_lines:
        C = np.array(cpts)
        result = bez_csx(C, S, atol=1e-3, rational=False)
        assert len(result["isolated"]) == 0, (
            f"False positive for line {label}: got {result['isolated']}"
        )
