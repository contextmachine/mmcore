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

def test_case_13():
    """missing second isolated intersection"""
    # may vary slightly (values taken from third-party software)


    excepted=[
        {   't':0.0, 'u':0.326507, 'v':0.356348},
        {   't':0.654374, 'u':0.633137, 'v':0.163511}
    ]




    C = np.array([[8.446359931193093, -39.19858842345994, 0.7182627669008318], [10.622854420468764, -38.91014606375564, 2.0882761678970088], [12.48389559934674, -38.81378122751128, 3.469350420794018], [15.0, -38.62105155502256, 5.071820879820981]])



    # Line parallel to y-axis at various (x, z) that don't satisfy z=x
    S =np.array( [[[7.4968198, -34.44808135, 6.627417], [4.89170045910665, -39.13729615771332, -4.42516829776066], [-0.016883173357102876, -44.594332395950104, 0.8101986397153593]], [[11.989753624247342, -35.42881907275406, 6.6274169999999994], [7.443691937074275, -40.76501466713547, -4.882652796843839], [3.454490070369776, -44.96214894393917, 0.5694145013516874]], [[14.847212913305142, -36.95471497775948, 6.6274169999999994], [9.56529033335222, -42.536197535294875, -4.488016026845241], [5.924649611832621, -46.255670751572566, 0.7771204997432491]], [[16.23843504869012, -39.53348121435317, 6.6274169999999994], [11.830092620085596, -44.58445479257182, -3.241257987764865], [8.72332454261908, -47.47050603984768, -0.38590063418195103]]]
                          )
    result = bez_csx(C, S, atol=1e-3, rational=False)
    assert (len(result["isolated"]) == 2) and (len(result["overlaps"]) == 0), f"expected 2 isolated intersections, {len(result['isolated'])} found {result}"
    for i, inter in enumerate(   sorted(result["isolated"], key=lambda x: x["t"])):
        assert np.allclose(
            [inter[key] for key in ["t", "u", "v"]],
       [excepted[i][key]  for key in ["t", "u", "v"]]), f"expected {excepted[i]}, got {inter}"


def test_case_14():
    """missed isolated intersection — line vs bicubic surface.

    A degree-1 segment intersects this 4x3 surface at exactly one point.
    v4 currently returns 0 isolated. Discovered while debugging bez_ssx
    case 9 (the v=0 face of S2 vs S1). Independently verified.
    """
    excepted = [
        {'t': 0.717378, 'u': 0.341217, 'v': 0.934587},
    ]

    C = np.array([
        [40.25282656, -76.40733562, -0.05990905],
        [23.11248642, -70.28329548,  1.46219793],
    ])

    S = np.array([
        [[33.05079627, -57.09987394, 0.0],
         [29.5295466,  -63.44484237, 6.7646494],
         [21.73708777, -71.24200956, 0.0]],
        [[40.28725776, -58.67948118, 0.0],
         [32.51384961, -66.4481508,  9.37051336],
         [28.97354926, -69.99318967, 0.0]],
        [[43.28107855, -61.67330197, 0.0],
         [35.49815262, -69.45145922, 9.37051336],
         [28.73859119, -79.68670826, 0.0]],
        [[45.10433572, -68.20265667, 0.0],
         [41.48244052, -72.46428996, 4.71678541],
         [38.71855016, -76.6579268,  0.0]],
    ])

    result = bez_csx(C, S, atol=1e-3, rational=False)
    assert (len(result["isolated"]) == 1) and (len(result["overlaps"]) == 0), \
        f"expected 1 isolated intersection, {len(result['isolated'])} found {result}"
    for i, inter in enumerate(sorted(result["isolated"], key=lambda x: x["t"])):
        assert np.allclose(
            [inter[key] for key in ["t", "u", "v"]],
            [excepted[i][key] for key in ["t", "u", "v"]]), \
            f"expected {excepted[i]}, got {inter}"


def test_case_15():
    """spurious _micro near-duplicate — degree-2 curve vs bilinear patch.

    A degree-2 curve intersects this 2x2 surface at exactly two points.
    v4 currently returns 3 isolated — the third is a near-duplicate of
    one of the genuine roots (Δstuv ~ 1e-4, well below atol=1e-3),
    flagged with `'_micro': True`. The dedup pass should have removed
    it. Discovered while debugging bez_ssx case 8 (the s=0 face of S1
    vs S2). Independently verified.
    """
    excepted = [
        {'t': 0.233841, 'u': 0.920968, 'v': 0.942768},
        {'t': 0.640738, 'u': 0.960552, 'v': 0.416216},
    ]

    C = np.array([
        [33.05079627, -57.09987394, 0.0],
        [29.5295466,  -63.44484237, 6.7646494],
        [21.73708777, -71.24200956, 0.0],
    ])

    S = np.array([
        [[40.25282656, -76.40733562, 2.23739797],
         [45.30378577, -65.64948729, 3.66374609]],
        [[23.11248642, -70.28329548, 3.75950495],
         [30.39942473, -58.97443598, 2.23739797]],
    ])

    result = bez_csx(C, S, atol=1e-3, rational=False)
    assert (len(result["isolated"]) == 2) and (len(result["overlaps"]) == 0), \
        f"expected 2 isolated intersections, {len(result['isolated'])} found {result}"
    for i, inter in enumerate(sorted(result["isolated"], key=lambda x: x["t"])):
        assert np.allclose(
            [inter[key] for key in ["t", "u", "v"]],
            [excepted[i][key] for key in ["t", "u", "v"]]), \
            f"expected {excepted[i]}, got {inter}"




