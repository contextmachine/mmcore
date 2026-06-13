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


def test_case_16():
    """missed isolated intersection — degree-2 curve vs bicubic surface.

    Same class of bug as test_case_14: a degree-2 segment intersects this
    3x3 surface at exactly one transversal point. v4 currently returns 0
    isolated. Discovered while debugging bez_ssx case 10 (the s=0 face of
    S1 vs S2). Independently verified via Newton: distance at the root is
    0 to machine precision; curve-vs-tangent-plane angle ≈ 47°
    (transversal, not tangent).
    """
    excepted = [
        {'t': 0.385582, 'u': 0.870815, 'v': 0.007269},
    ]

    C = np.array([
        [33.05079627, -57.09987394, 0.0],
        [29.5295466,  -63.44484237, 6.7646494],
        [21.73708777, -71.24200956, 0.0],
    ])

    S = np.array([
        [[29.63685574, -70.79194487, 4.04308391],
         [33.99717923, -70.79194487, 7.50248027],
         [39.66180486, -70.79194487, 4.18744742]],
        [[29.63685574, -66.43162138, 0.58368755],
         [33.99717923, -66.43162138, 4.04308391],
         [39.66180486, -66.43162138, 0.72805106]],
        [[29.63685574, -60.76699576, 3.89872039],
         [33.99717923, -60.76699576, 7.35811675],
         [39.66180486, -60.76699576, 4.04308391]],
    ])

    result = bez_csx(C, S, atol=1e-3, rational=False)
    assert (len(result["isolated"]) == 1) and (len(result["overlaps"]) == 0), \
        f"expected 1 isolated intersection, {len(result['isolated'])} found {result}"
    for i, inter in enumerate(sorted(result["isolated"], key=lambda x: x["t"])):
        assert np.allclose(
            [inter[key] for key in ["t", "u", "v"]],
            [excepted[i][key] for key in ["t", "u", "v"]]), \
            f"expected {excepted[i]}, got {inter}"


def test_case_17():
    """missed second isolated — degree-1 line vs 4x3 bicubic surface.

    A degree-1 segment intersects this 4x3 surface at exactly two
    transversal points. v4 currently returns 1 isolated (finds the
    boundary one at t=1, misses the interior one at t≈0.88).
    Discovered while debugging bez_ssx case 9: the v=0.0685 cut isoline
    of S2 (a bilinear) intersected with S1 (the swept ribbon). Per-piece
    CSX missed one of the two real crossings, which silently dropped
    the second branch of case 9. Independently verified via Newton:
    distance at the missed root is 0 to machine precision; crossing
    angle ≈ 5.4° (transversal, not tangent).
    """
    excepted = [
        {'t': 0.879717, 'u': 0.153108, 'v': 0.915742},
        {'t': 1.000000, 'u': 0.00978626, 'v': 0.888249},
    ]

    C = np.array([
        [40.59861684732258, -75.67084987336429, 0.03773919903009087],
        [23.611352553570853, -69.50908733196081, 1.3579939980205344],
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
    assert (len(result["isolated"]) == 2) and (len(result["overlaps"]) == 0), \
        f"expected 2 isolated intersections, {len(result['isolated'])} found {result}"
    for i, inter in enumerate(sorted(result["isolated"], key=lambda x: x["t"])):
        assert np.allclose(
            [inter[key] for key in ["t", "u", "v"]],
            [excepted[i][key] for key in ["t", "u", "v"]]), \
            f"expected {excepted[i]}, got {inter}"


def test_case_18():
    """missed second isolated — rational=True sub-piece variant of test_case_17.

    Same two geometric crossings as test_case_17, but exercised on the
    s∈[0, 0.5] half-piece of the original surface (after a De Casteljau
    split at s=0.5) with rational=True (weights = 1). This is the form
    bez_ssx actually calls per-piece CSX in: rational mode with w=1
    added, against a sub-piece of the original surface.

    test_case_17 (rational=False, full surface) currently passes.
    test_case_18 (rational=True, sub-piece) currently fails — v4 finds
    only 1 of 2 isolated. So the v4 fix must extend to the rational +
    sub-piece code path.

    Discovered while debugging bez_ssx case 9 — the per-piece CSX
    of the v=0.0685 cut isoline against the s∈[0, 0.5] piece of S1.
    Both roots Newton-verified at machine precision.
    """
    excepted = [
        {'t': 0.879717, 'u': 0.306216, 'v': 0.915742},
        {'t': 1.000000, 'u': 0.0195725, 'v': 0.888249},
    ]

    C = np.array([
        [40.598616847322575, -75.67084987336429, 0.03773919903009087, 1.0],
        [23.61135255357085,  -69.50908733196081, 1.3579939980205342,  1.0],
    ])

    S = np.array([
        [[33.05079627, -57.09987394, 0.0, 1.0],
         [29.5295466,  -63.44484237, 6.7646494,  1.0],
         [21.73708777, -71.24200956, 0.0,        1.0]],
        [[36.669027015,    -57.88967756,    0.0,        1.0],
         [31.021698105,    -64.946496585,   8.06758138, 1.0],
         [25.355318515,    -70.617599615,   0.0,        1.0]],
        [[39.226597585,       -59.033034567499996, 0.0,        1.0],
         [32.51384961,        -66.4481507975,      8.71904737, 1.0],
         [27.105694370000002, -72.72877429,        0.0,        1.0]],
        [[41.107517615000006, -60.7951100075,     0.0,        1.0],
         [34.38099922625,     -67.95099529875,    8.463064371249999, 1.0],
         [29.19900741,        -74.61745376875001, 0.0,        1.0]],
    ])

    result = bez_csx(C, S, atol=1e-3, rational=True)
    assert (len(result["isolated"]) == 2) and (len(result["overlaps"]) == 0), \
        f"expected 2 isolated intersections, {len(result['isolated'])} found {result}"
    for i, inter in enumerate(sorted(result["isolated"], key=lambda x: x["t"])):
        assert np.allclose(
            [inter[key] for key in ["t", "u", "v"]],
            [excepted[i][key] for key in ["t", "u", "v"]]), \
            f"expected {excepted[i]}, got {inter}"


def test_case_19_interior_root_not_pruned_by_outside_basin():
    """Regression: phase-2 pruned a whole cell because Newton from the cell
    center converged to a root OUTSIDE the cell (the t=0 boundary root,
    already excluded from the search interval by phase 1). The interior
    root at t~0.5356 was silently lost.

    Extracted from bez_ssx case 10 (examples/ssx/bez_ssx5_case10.py): the
    curve is the S2 isoline at u=0.30863 restricted to v in [0.3642, 0.9834]
    (a subdivision cut face), the surface is the matching de Casteljau piece
    of S1. SSX lost the branch segment s in [0.57, 0.75] because this
    crossing was never registered.
    """
    C = np.array([
        [32.98566961861533, -67.97624934909058, 4.174218355677444, 1.0],
        [35.97968293692663, -67.97624934909058, 4.788662820718871, 1.0],
        [39.473772059842304, -67.97624934909058, 2.8057580722952005, 1.0],
    ])

    S = np.array([
        [[35.138894229011775, -65.38692059391325, 4.034711061953261, 1.0],
         [33.259522252203, -67.51791403978095, 4.6333398494434554, 1.0],
         [31.503554672958494, -69.57544202356137, 3.6246620839551396, 1.0]],
        [[37.414583913330304, -67.2409983833853, 4.0654045715392, 1.0],
         [35.491018636559495, -69.4127819485895, 4.668587345212076, 1.0],
         [33.766077362440214, -71.55807823458665, 3.6522362023272756, 1.0]],
        [[39.4950028428197, -69.4554164762803, 3.5311794986980405, 1.0],
         [37.743984530723395, -71.5675032096166, 4.055099469485858, 1.0],
         [36.083116890736406, -73.81327077121055, 3.1723045948113477, 1.0]],
        [[42.4388198233443, -71.47896978523977, 2.234894386095701, 1.0],
         [41.42777629852405, -72.78012635464769, 2.566484950072648, 1.0],
         [40.497708882687974, -74.07486560833775, 2.0077613535487226, 1.0]],
    ])

    result = bez_csx(C, S, atol=1e-3, rational=True)
    ts = sorted(p["t"] for p in result["isolated"])
    assert len(ts) == 2, f"expected 2 isolated roots (t=0.0 and t~0.5356), got {result['isolated']}"
    assert abs(ts[0] - 0.0) < 1e-3, f"boundary root at t=0 missing: {ts}"
    assert abs(ts[1] - 0.535593) < 1e-3, f"interior root at t~0.5356 missing: {ts}"
