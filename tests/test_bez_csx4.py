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


def test_bernstein_zero_result_budget_stops_recursive_materialization():
    """A full result quota must stop before materializing child roots."""
    from mmcore.numeric.intersection._bern_zero_1d import (
        bernstein_zero_budget,
        find_bernstein_zeros_1d,
    )

    # Both endpoints are exact roots and the derivative has three sign
    # changes, so the uncapped algorithm would recurse into both children.
    coeffs = np.array([0.0, -1.0, 1.0, -1.0, 0.0])
    with bernstein_zero_budget(max_nodes=100, max_results=1) as budget:
        roots = find_bernstein_zeros_1d(
            coeffs, atol=1e-3, max_depth=4,
        )

    assert roots == [0.0]
    assert budget.exhausted is True
    assert budget.nodes == 1


def test_bernstein_zero_unresolved_depth_limit_exhausts_scoped_budget():
    """A Newton fallback cannot certify a multi-minimum interval complete."""
    from mmcore.numeric.intersection._bern_zero_1d import (
        bernstein_zero_budget,
        find_bernstein_zeros_1d,
    )

    coeffs = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    with bernstein_zero_budget(max_nodes=100, max_results=10) as budget:
        roots = find_bernstein_zeros_1d(
            coeffs, atol=1e-3, max_depth=0,
        )

    assert len(roots) <= 1
    assert budget.exhausted is True


def test_csx_boundary_ccx_calls_share_one_remaining_budget(monkeypatch):
    """The four surface-edge CCX calls must not each reset max_cells."""
    import mmcore.numeric.intersection.csx._bez_csx4 as csx_mod

    C = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    S = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    ])
    F = np.zeros((3, 3, 3), dtype=float)
    allowances = []

    # Skip the two endpoint-face root searches so the test isolates the four
    # nested CCX calls.  The diagonal curve's AABB touches every patch edge.
    monkeypatch.setattr(csx_mod, "_check_min_of_net", lambda *args: True)

    def fake_ccx(*args, max_cells, **kwargs):
        allowances.append(max_cells)
        used = min(2, max_cells)
        return {
            "isolated": [], "overlaps": [],
            "budget_exhausted": False, "cells_processed": used,
            "boundary_topology_complete": True,
        }

    monkeypatch.setattr(csx_mod, "bez_ccx_v4", fake_ccx)
    zeros, exhausted, cells = csx_mod._find_csx_boundary_zeros(
        F, C, S, 1e-3, 1e-3, 1e-3, 1e-3, False,
        max_cells=10, max_results=32,
    )

    assert zeros == []
    assert exhausted is False
    assert cells == 10
    assert allowances == [8, 6, 4, 2]


def test_phase2_max_depth_reports_unresolved_cell(monkeypatch):
    """Reaching ``max_depth`` must propagate as a partial CSX result."""
    import mmcore.numeric.intersection._sq_dist_classify as classify
    import mmcore.numeric.intersection.csx._bez_csx4 as csx_mod

    monkeypatch.setattr(classify, "_check_min_of_net", lambda *args: False)
    monkeypatch.setattr(classify, "_check_lipschitz", lambda *args: False)
    monkeypatch.setattr(csx_mod, "_residual_excludes_zero", lambda *args: False)
    monkeypatch.setattr(
        csx_mod, "bernstein_partial_derivative_coeffs",
        lambda *args, **kwargs: np.array([[-1.0], [1.0]]),
    )
    monkeypatch.setattr(
        csx_mod, "newton_csx",
        lambda *args, **kwargs: (
            0.5, 0.5, 0.5, np.ones(3), np.ones(3),
        ),
    )

    C = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    S = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    ])
    roots, exhausted, cells, cause = csx_mod._phase2_isolated_search(
        np.zeros((2, 2, 2)), np.zeros((2, 2, 2, 3)), C,
        S, C, S,
        0.0, 1.0, 1e-6, False, 1e-12, 1e-12, 1e-12,
        max_depth=0, max_cells=10,
    )

    assert roots == []
    assert cells == 1
    assert exhausted is True
    # 2026-07-26: and it must say WHY.  A depth ceiling is structural — no
    # cell allowance can buy more of it — so a consumer must be able to tell
    # it apart from a resource shortfall instead of escalating it to a
    # global work_budget stop.
    assert cause == "depth"


def test_partial_boundary_topology_is_discarded(monkeypatch):
    import mmcore.numeric.intersection.csx._bez_csx4 as csx_mod
    from mmcore.numeric.intersection._sq_dist_classify import BoundaryZero

    C = np.array([[0.5, 0.5, -1.0], [0.5, 0.5, 1.0]])
    S = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    ])

    def partial_boundary(*args, **kwargs):
        return [BoundaryZero(axis=0, side=0, param=0.5, param2=0.5)], True, 1

    monkeypatch.setattr(csx_mod, "_find_csx_boundary_zeros", partial_boundary)
    monkeypatch.setattr(
        csx_mod, "_check_csx_overlap_valley",
        lambda *args, **kwargs: pytest.fail("partial topology reached overlap check"),
    )

    result = csx_mod.bez_csx(
        C, S, atol=1e-3, rational=False, max_cells=1,
    )
    assert result["budget_exhausted"] is True
    assert result["boundary_topology_complete"] is False
    assert result["isolated"] == []


def test_constant_rational_curve_on_surface_is_parameter_fiber():
    """A collapsed rational boundary is a parameter fiber, not 16k roots.

    Case 14's cone apex edge is geometrically one point although its
    homogeneous weights vary with ``t``.  When that point lies on the other
    surface, the CSX zero set contains every curve parameter.  Enumerating
    the fiber as isolated roots is both topologically wrong and quadratic in
    the number of reported samples.
    """
    from examples.ssx.bez_ssx5_case14 import S1, S2
    from mmcore.numeric.intersection._bezier_common import eval_curve, eval_surface

    C = S1[:, 0, :]
    t0 = time.perf_counter()
    result = bez_csx(C, S2, atol=1e-3, rational=True)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0
    assert result["isolated"] == []
    assert result["overlaps"] == []
    fibers = result.get("parameter_fibers", [])
    assert len(fibers) == 1
    fiber = fibers[0]
    assert fiber["t_range"] == (0.0, 1.0)
    p = eval_curve(C, 0.37, rational=True)
    q = eval_surface(S2, fiber["u"], fiber["v"], rational=True)
    assert np.linalg.norm(p - q) <= 1e-3


def test_constant_curve_on_constant_surface_reports_full_parameter_region():
    """A representative (u,v) cannot stand in for the full [0,1]^3 set."""
    point = np.array([2.0, -3.0, 5.0])
    C = np.tile(np.r_[point, 1.0], (2, 1))
    S = np.tile(np.r_[point, 1.0], (2, 2, 1))

    result = bez_csx(C, S, atol=1e-3, rational=True)
    assert result["budget_exhausted"] is False
    assert result["boundary_topology_complete"] is True
    assert result["isolated"] == [] and result["overlaps"] == []
    assert len(result["parameter_fibers"]) == 1
    region = result["parameter_fibers"][0]
    assert region["t_range"] == (0.0, 1.0)
    assert region["u_range"] == (0.0, 1.0)
    assert region["v_range"] == (0.0, 1.0)
    assert np.array_equal(region["point"], point)
    assert region["surface_kind"] == "degenerate_surface"


def test_constant_curve_parameter_fibers_respect_result_cap(monkeypatch):
    """The collapsed-curve fast path shares the public result budget."""
    import mmcore.numeric._bez_closest_point as closest_mod

    curve = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 2.0],
    ])
    surface = np.array([
        [[-1.0, -1.0, 0.0, 1.0], [-1.0, 1.0, 0.0, 1.0]],
        [[1.0, -1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]],
    ])

    def many_closest(_surface, _query, *, stats, **_kwargs):
        stats.update(cells_processed=1, budget_exhausted=False)
        return [
            {
                # Duplicate exact representatives exercise the public result
                # cap without relying on a lying closest-point witness.
                "u": 0.5,
                "v": 0.5,
                "point": np.zeros(3),
                "distance": 0.0,
                "kind": "min",
            }
            for i in range(6)
        ]

    monkeypatch.setattr(
        closest_mod, "bez_surface_closest_points", many_closest)
    result = bez_csx(
        curve, surface, atol=1e-3, rational=True,
        max_cells=100, max_results=2,
    )
    assert len(result["parameter_fibers"]) == 2
    assert result["budget_exhausted"] is True
    assert result["boundary_topology_complete"] is False

    zero = bez_csx(
        curve, surface, atol=1e-3, rational=True,
        max_cells=100, max_results=0,
    )
    assert zero["parameter_fibers"] == []
    assert zero["budget_exhausted"] is True
    assert zero["boundary_topology_complete"] is False


def test_collapsed_curve_detection_is_translation_invariant():
    """Large world coordinates must not turn a moving curve into a fiber."""
    x0 = 1.0e15
    curve = np.array([
        [x0, 0.0, 0.0, 1.0],
        [x0 + 10.0, 0.0, 0.0, 1.0],
    ])
    plane_x = x0 + 5.0
    surface = np.array([
        [[plane_x, -1.0, -1.0, 1.0], [plane_x, -1.0, 1.0, 1.0]],
        [[plane_x, 1.0, -1.0, 1.0], [plane_x, 1.0, 1.0, 1.0]],
    ])

    result = bez_csx(curve, surface, atol=1e-3, rational=True)
    assert result["parameter_fibers"] == []
    assert len(result["isolated"]) == 1
    assert abs(result["isolated"][0]["t"] - 0.5) <= 1e-6


def test_rational_param_tolerance_is_translation_invariant_for_constant_geometry():
    from mmcore.geom._nurbs_param_tol import (
        bez_curve_param_tolerance, bez_surface_param_tolerance,
    )

    weights = np.array([1.0, np.sqrt(0.5), 1.0])
    p = np.array([26.0, -11.0, 46.0])
    C = np.concatenate([weights[:, None] * p, weights[:, None]], axis=1)
    shift = np.array([1000.0, -2000.0, 500.0])
    Ct = C.copy()
    Ct[:, :3] += weights[:, None] * shift

    pc = bez_curve_param_tolerance(C, 1e-3, rational=True)
    pct = bez_curve_param_tolerance(Ct, 1e-3, rational=True)
    assert pc == pytest.approx(1e-3)
    assert pct == pytest.approx(pc)

    W = np.array([[1.0, 0.8], [0.7, 1.2]])
    S = np.concatenate([W[..., None] * p, W[..., None]], axis=-1)
    St = S.copy()
    St[..., :3] += W[..., None] * shift
    ps = bez_surface_param_tolerance(S, 1e-3, rational=True)
    pst = bez_surface_param_tolerance(St, 1e-3, rational=True)
    assert ps == pytest.approx((1e-3, 1e-3))
    assert pst == pytest.approx(ps)

    # Reversing a rational Bezier curve is a pure reparameterization and
    # must not change its resolution.  The endpoint with the small weight
    # is the high-speed side of this degree-one example (max |C'| = 10),
    # so a sound tolerance is at most atol/10.
    Pe = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    we = np.array([10.0, 1.0])
    Ce = np.concatenate([Pe * we[:, None], we[:, None]], axis=1)
    pe = bez_curve_param_tolerance(Ce, 1e-3, rational=True)
    per = bez_curve_param_tolerance(Ce[::-1].copy(), 1e-3, rational=True)
    assert pe == pytest.approx(per)
    assert pe <= 1.01e-4

    # A coordinate-scale roundoff floor must not erase genuine local
    # motion at large world coordinates.
    huge = 1.0e15
    moving = np.array([
        [[huge, 0.0, 0.0, 1.0], [huge, 1.0, 0.0, 1.0]],
        [[huge + 10.0, 0.0, 0.0, 1.0],
         [huge + 10.0, 1.0, 0.0, 1.0]],
    ])
    pu, _ = bez_surface_param_tolerance(
        moving, 1e-3, rational=True)
    assert pu <= 1.01e-4


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
    """A tolerance-only endpoint near miss must not become topology.

    The formerly expected ``t=0`` item came from third-party tolerance
    matching.  With ``t`` fixed at zero, the closest point on this surface
    remains 2.328e-8 away from the curve endpoint, so it is not a root of
    the supplied floating-point coefficients.  The interior root is real.
    """


    expected = {'t': 0.654374, 'u': 0.633137, 'v': 0.163511}




    C = np.array([[8.446359931193093, -39.19858842345994, 0.7182627669008318], [10.622854420468764, -38.91014606375564, 2.0882761678970088], [12.48389559934674, -38.81378122751128, 3.469350420794018], [15.0, -38.62105155502256, 5.071820879820981]])



    # Line parallel to y-axis at various (x, z) that don't satisfy z=x
    S =np.array( [[[7.4968198, -34.44808135, 6.627417], [4.89170045910665, -39.13729615771332, -4.42516829776066], [-0.016883173357102876, -44.594332395950104, 0.8101986397153593]], [[11.989753624247342, -35.42881907275406, 6.6274169999999994], [7.443691937074275, -40.76501466713547, -4.882652796843839], [3.454490070369776, -44.96214894393917, 0.5694145013516874]], [[14.847212913305142, -36.95471497775948, 6.6274169999999994], [9.56529033335222, -42.536197535294875, -4.488016026845241], [5.924649611832621, -46.255670751572566, 0.7771204997432491]], [[16.23843504869012, -39.53348121435317, 6.6274169999999994], [11.830092620085596, -44.58445479257182, -3.241257987764865], [8.72332454261908, -47.47050603984768, -0.38590063418195103]]]
                          )
    result = bez_csx(C, S, atol=1e-3, rational=False)
    assert len(result["isolated"]) == 1, result
    assert result["overlaps"] == []
    inter = result["isolated"][0]
    assert np.allclose(
        [inter[key] for key in ["t", "u", "v"]],
        [expected[key] for key in ["t", "u", "v"]],
    ), f"expected {expected}, got {inter}"

    # Exact-set membership and polishing must not depend on a common world
    # translation: preserve the real interior root without reviving the
    # tolerance-only endpoint near miss.
    translated = bez_csx(
        C + 1.0e6, S + 1.0e6, atol=1e-3, rational=False)
    assert len(translated["isolated"]) == 1, translated
    translated_inter = translated["isolated"][0]
    assert np.allclose(
        [translated_inter[key] for key in ["t", "u", "v"]],
        [expected[key] for key in ["t", "u", "v"]],
    ), f"expected {expected}, got {translated_inter}"


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


def test_exact_corner_root_reaches_resolution_before_depth_stop():
    """A known endpoint root must not leave a false partial Phase-2 tail.

    The remaining interval ends one parameter tolerance before the root.
    Three subdivision axes need 53 levels to reach the same resolution;
    the former depth-50 default stopped just before that certificate.
    """
    curve = np.array([
        [-128.25, -129.86, 0.0],
        [-128.25, 129.86, 0.0],
    ])
    surface = np.array([
        [[-128.25, -129.86, 67.44], [-128.25, 129.86, 0.0]],
        [[128.25, -46.98, 0.0], [128.25, 129.86, 0.0]],
    ])

    result = bez_csx(curve, surface, atol=1e-3, rational=False)

    assert not result["budget_exhausted"], result
    assert result["boundary_topology_complete"]
    assert len(result["isolated"]) == 1
    assert np.allclose(
        [result["isolated"][0][key] for key in ("t", "u", "v")],
        [1.0, 0.0, 1.0], atol=1e-12,
    )


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


def test_bounded_newton_stall_near_tangent_is_not_a_distinct_root():
    """A cutout-wall stall in a tangent valley must polish to one root."""
    eps = 5e-7
    curve = np.array([
        [0.5, -0.125, 0.25],
        [0.5 + 1.0 / 3.0, 0.125 + eps / 3.0, -1.0 / 12.0],
        [0.5 + 2.0 / 3.0, -0.125 + 2.0 * eps / 3.0, -1.0 / 12.0],
        [1.5, 0.125 + eps, 0.25],
    ])
    plane = np.array([
        [[-0.5, -1.0, 0.0], [-0.5, 1.0, 0.0]],
        [[1.5, -1.0, 0.0], [1.5, 1.0, 0.0]],
    ])
    curve_h = np.column_stack([curve, np.ones(len(curve))])
    plane_h = np.concatenate(
        [plane, np.ones(plane.shape[:-1] + (1,))], axis=-1)

    result = bez_csx(
        curve_h, plane_h, atol=1e-3, rational=True,
        max_cells=20_000, max_results=128)

    assert result["budget_exhausted"] is False
    assert len(result["isolated"]) == 1
    root = result["isolated"][0]
    assert root["t"] == pytest.approx(0.5, abs=1e-7)
    assert root["u"] == pytest.approx(0.75, abs=1e-7)


def test_curved_uv_exact_overlap_is_certified_complete():
    """Ledger L42 → L59: the curved-UV EXACT overlap's full history.

    Parabola lying exactly on the bilinear z=0 patch: the uv-preimage is
    curved, so the exact-AFFINE identity correctly refuses. At 5d05ddc
    this flooded Phase 2 (1,679 lattice roots @33,685 cells, claimed
    complete); L42 bounded it into an honest typed-partial; the L59
    theorem-first tier now CERTIFIES it — domain-pinned span ends,
    roundoff-level witnesses (=> 'exact'), no flips — with no Phase-2
    grind at all. Neither wrong topology nor a partial flag remains.
    """
    curve = np.array([[0.2, 0.2, 0.], [1., 1.8, 0.], [1.8, 0.2, 0.]])
    surf = np.array([[[0., 0., 0.], [0., 2., 0.]],
                     [[2., 0., 0.], [2., 2., 0.]]])
    result = bez_csx(curve, surf, atol=1e-3, rational=False)

    assert result["boundary_topology_complete"] is True
    assert result["budget_exhausted"] is False
    assert result["isolated"] == []
    assert len(result["overlaps"]) == 1
    o = result["overlaps"][0]
    assert o["certification"] == "exact"
    assert o["t_range"][0] == pytest.approx(0.0, abs=1e-9)
    assert o["t_range"][1] == pytest.approx(1.0, abs=1e-9)
    assert result["cells_processed"] <= 1_000


def test_boundary_exhaustion_keeps_certified_roots(monkeypatch):
    """Ledger L51: on boundary-phase exhaustion the certified-partial contract
    must keep the strictly certified roots already in hand (CCX keeps its
    validated hits in the same situation). Dropping them returned
    {isolated: [], budget_exhausted: True} with a certified root found and
    ~no Phase-2 budget left to re-find it. Partial boundary topology still
    must not drive overlap classification, and the result stays flagged."""
    import mmcore.numeric.intersection.csx._bez_csx4 as csx_mod
    from mmcore.numeric.intersection._sq_dist_classify import BoundaryZero

    # cubic with its t=0 endpoint exactly ON the plane z=0 at (0.25, 0.25):
    # z(t) = t (2 (t - 0.6)^2 + 0.001) — the ONLY root is t=0, and the
    # sub-atol valley (6e-4 deep at t=0.6) makes a 1-cell Phase 2 provably
    # unable to re-find/exclude anything (sub-atol valleys must be resolved,
    # never merged).
    def _mono2bern3(a):
        from math import comb
        return [sum(comb(i, k) / comb(3, k) * a[k] for k in range(i + 1))
                for i in range(4)]

    x = _mono2bern3([0.25, 0.5, 0.0, 0.0])
    y = _mono2bern3([0.25, 0.5, 0.0, 0.0])
    z = _mono2bern3([0.0, 0.721, -2.4, 2.0])
    curve = np.column_stack([x, y, z])
    surf = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                     [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])

    bz = BoundaryZero(axis=0, side=0, param=0.25, param2=0.25)
    # the exhausted boundary phase consumed ~all of the allowance: Phase 2
    # has 1 cell left and cannot re-find the discarded root
    monkeypatch.setattr(
        csx_mod, "_find_csx_boundary_zeros",
        lambda *a, **k: ([bz], True, 99))

    result = bez_csx(curve, surf, atol=1e-3, rational=False, max_cells=100)

    assert result["budget_exhausted"] is True
    assert result["boundary_topology_complete"] is False
    assert len(result["isolated"]) == 1, result["isolated"]
    iso = result["isolated"][0]
    assert iso["t"] == pytest.approx(0.0, abs=1e-9)
    assert iso["u"] == pytest.approx(0.25, abs=1e-6)
    assert iso["v"] == pytest.approx(0.25, abs=1e-6)
    assert len(result["overlaps"]) == 0


def test_zero_allowance_preflights_before_net_build(monkeypatch):
    """L52 (zero-allowance preflight): bez_csx(max_cells=0) must refuse
    BEFORE building the superlinear distance/residual nets — the SSX top
    entry has preflighted since L32, but the CSX entry still paid the full
    net construction for an allowance it did not have."""
    import mmcore.numeric.intersection.csx._bez_csx4 as csx_mod

    calls = {"n": 0}
    orig = csx_mod.curve_surface_distance_squared_net_homog

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(
        csx_mod, "curve_surface_distance_squared_net_homog", counting)
    curve = np.array([[0.25, 0.25, -1.0], [0.75, 0.75, 1.0]])
    surf = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                     [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    r = bez_csx(curve, surf, atol=1e-3, rational=False, max_cells=0)
    assert r["budget_exhausted"] is True
    assert r["boundary_topology_complete"] is False
    assert r["cells_processed"] == 0
    assert r["isolated"] == [] and r["overlaps"] == []
    assert calls["n"] == 0, "net built despite zero allowance"


def test_short_clipped_overlap_span_is_certified():
    """L52 slice 10a → L59: the domain-clipped short span's full history.

    A coincident span of 4.2*ptol_t ended by DOMAIN CLIPPING at u=1 used
    to ship as 3 lattice roots claimed complete; slice 10a's lattice-
    cluster detection made it a typed uncertified span; the L59 tier now
    CERTIFIES it as one tolerance overlap. The span's upper end extends
    past the exact-coincidence limit t=0.5 to the within-atol fringe of
    the patch edge (~0.58) — tolerance-coincidence semantics (USER
    DECISION 2026-07-12)."""
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    C = np.array([[0.994, 0.5, 0.0],
                  [1.000, 0.5, 0.0],
                  [1.006, 0.506, 0.0]])

    r = bez_csx(C, S, atol=1e-3, rational=False)

    assert r["boundary_topology_complete"] is True
    assert r["budget_exhausted"] is False
    assert "uncertified_overlap_span" not in r
    assert r["isolated"] == []
    assert len(r["overlaps"]) == 1
    o = r["overlaps"][0]
    assert o["certification"] == "tolerance"
    assert o["t_range"][0] == pytest.approx(0.0, abs=1e-9)
    assert 0.5 <= o["t_range"][1] <= 0.62
    assert r["cells_processed"] <= 500


def test_sub_atol_valley_root_chain_is_never_merged_by_the_cluster():
    """Invariant control for the lattice-cluster detection: three strict-
    distinct roots connected by SUB-ATOL valleys (depth ~5e-4 between
    roots ~0.15 apart at ptol_t ~0.04) form a gap-qualified cluster, but
    the STRICT gap-midpoint certificate fails on the valley floors — the
    roots must ship isolated with complete topology (never merged into a
    span: sub-tolerance topology is preserved, the CSX invariant)."""
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    # cubic z(t) = 0.25*(t-0.35)(t-0.5)(t-0.65): roots 0.35/0.5/0.65,
    # valley depth ~4.3e-4 (sub-atol, far above strict roundoff scale);
    # x spans 0.02 so ptol_t ~ 0.045 and the 0.15 gaps qualify (< 4*ptol).
    from numpy.polynomial import polynomial as P
    coeffs = 0.25 * np.array(P.polyfromroots([0.35, 0.5, 0.65]))
    # convert power basis -> Bernstein-3 control values for z(t)
    M = np.array([[1.0, 0.0, 0.0, 0.0],
                  [1.0, 1.0 / 3.0, 0.0, 0.0],
                  [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
                  [1.0, 1.0, 1.0, 1.0]])
    zc = M @ coeffs
    C = np.column_stack([
        np.array([0.49, 0.4967, 0.5033, 0.51]),   # x affine, speed 0.02
        np.full(4, 0.5),
        zc])

    r = bez_csx(C, S, atol=1e-3, rational=False)

    ts = sorted(x["t"] for x in r["isolated"])
    assert len(ts) == 3, r["isolated"]
    assert ts == pytest.approx([0.35, 0.5, 0.65], abs=1e-3)
    assert r["boundary_topology_complete"] is True
    assert "uncertified_overlap_span" not in r
    assert r["budget_exhausted"] is False


def test_near_band_pair_certifies_cheaply():
    """L60: the worst profiled pair (28,961 cells) certifies cheaply.

    Real user geometry (overlap_nurbs_intersection_3_new): CORRECTED
    diagnosis — the curve TOUCHES the patch at its t=0 end (1.6e-9) and
    runs sub-atol (2.5e-5..7.7e-4) until the projected path exits through
    the u=1 patch edge at t~0.44, then departs. A one-side-pinned
    tolerance band whose far end is domain-clipped: under the L59
    theorem-first semantics this is ONE tolerance overlap (the endpoint
    touch is the span's own end; end-adjacent sign flips are the endpoint
    root, not interior crossing structure). Phase 2 then only scans the
    far, empty remainder — with the L60 geometry-aligned exclusion
    (scalar Bernstein net dot(G, n_mean), exact linear combination,
    L1-margined) handling diagonal-residual cells the axis test cannot
    clear until clearance-scale depth."""
    C_NEAR = np.array([
        [92.51428091, 102.781436125, 5.719709275],
        [91.27842704, 104.13179098, 6.58025473],
        [89.7844795, 105.2243694, 7.46164544],
        [88.15200982059918, 105.9854260265158, 8.363880612681537]])
    S_NEAR = np.array([
        [[92.51539241324699, 102.7802214952655, 41.41100531879138],
         [92.51539241324699, 102.7802214952655, 0.10498601801243446]],
        [[91.96619518531281, 103.38044393393321, 41.41100531879138],
         [91.96619518531281, 103.38044393393321, 0.10498601801243446]],
        [[91.36636523847328, 103.93038701100394, 41.41100531879138],
         [91.36636523847328, 103.93038701100394, 0.10498601801243446]],
        [[90.72572752430331, 104.42241106003442, 41.41100531879138],
         [90.72572752430331, 104.42241106003442, 0.10498601801243446]]])

    r = bez_csx(C_NEAR, S_NEAR, atol=1e-3, rational=False)

    assert r["budget_exhausted"] is False
    assert r["boundary_topology_complete"] is True
    assert r["isolated"] == []          # the t=0 touch is the span's end
    assert len(r["overlaps"]) == 1
    o = r["overlaps"][0]
    assert o["certification"] == "tolerance"
    assert o["t_range"][0] == pytest.approx(0.0, abs=1e-9)
    assert 0.40 <= o["t_range"][1] <= 0.50
    assert r["cells_processed"] <= 2_000, r["cells_processed"]


# ---------------------------------------------------------------------------
# Cluster-4 burn-down (2026-07-25): the strict root certificate must not
# depend on where the model sits in world space.
#
# `_strict_csx_root_tol` centers both nets on a common origin, dividing by
# the net scale BEFORE subtracting `origin * w`.  A coordinate that is
# identically zero after that translation is therefore reached only up to
# the subtraction's cancellation, and both the per-axis `component_scale`
# and the evaluated points carry that noise.  Bounding the residual by those
# already-cancelled magnitudes made the certificate refuse TRUE roots of a
# planar pair at ~30% of random world positions (measured, seed below) —
# the same defect the CCX Phase-2 hull prune had, in the certificate that
# gates every CSX boundary root.
#
# Geometry: the s1 v=0 edge and s2 of the user's boundary-coincidence pair.
# Their exact crossing is t = 0.377142857142857 on the curve, (u, v) =
# (0.697142857142857, 0) on the surface.
# ---------------------------------------------------------------------------

_PLANAR_C = np.array([[-16.0, -27.0, 0.0, 1.0], [-36.0, 2.0, 0.0, 1.0]])
_PLANAR_S = np.array([[[-34.0, -7.0, 0.0, 1.0], [-26.0, 2.0, 0.0, 1.0]],
                      [[-19.0, -20.0, 0.0, 1.0], [-17.0, -10.0, 0.0, 1.0]]])
_PLANAR_T = 0.377142857142857
_PLANAR_U = 0.697142857142857
_PLANAR_V = 0.0


def test_strict_csx_certificate_is_translation_invariant():
    from mmcore.numeric.intersection.csx._bez_csx4 import (
        _strict_csx_root_tol, _strict_csx_residual_ok,
    )

    rng = np.random.default_rng(11)
    rejected = []
    for _ in range(300):
        c = rng.uniform(-100.0, 100.0, 3)
        C = _PLANAR_C.copy()
        C[:, :3] -= c * C[:, 3:]
        S = _PLANAR_S.copy()
        S[..., :3] -= c * S[..., 3:]
        ctx = _strict_csx_root_tol(C, S, True)
        assert ctx is not None
        ok, _res = _strict_csx_residual_ok(
            C, S, _PLANAR_T, _PLANAR_U, _PLANAR_V, True, ctx)
        if not ok:
            rejected.append(c)
    assert not rejected, (
        f"{len(rejected)}/300 world positions rejected an exact root; "
        f"first: {rejected[0] if rejected else None}")


def test_strict_csx_certificate_still_rejects_a_real_offset():
    """Anti-loosening guard: a genuine off-surface point stays rejected.

    The point below is the true crossing lifted one model unit in z — far
    outside any roundoff envelope at every world position.
    """
    from mmcore.numeric.intersection.csx._bez_csx4 import (
        _strict_csx_root_tol, _strict_csx_residual_ok,
    )

    rng = np.random.default_rng(11)
    for _ in range(50):
        c = rng.uniform(-100.0, 100.0, 3)
        C = _PLANAR_C.copy()
        C[:, :3] -= c * C[:, 3:]
        C[:, 2] += 1.0 * C[:, 3]          # lift the curve off the plane
        S = _PLANAR_S.copy()
        S[..., :3] -= c * S[..., 3:]
        ctx = _strict_csx_root_tol(C, S, True)
        assert ctx is not None
        ok, _res = _strict_csx_residual_ok(
            C, S, _PLANAR_T, _PLANAR_U, _PLANAR_V, True, ctx)
        assert not ok, c


# ---------------------------------------------------------------------------
# Truncation-cause schema (2026-07-26).
#
# `budget_exhausted` alone cannot distinguish a resource shortfall the caller
# can fix by raising a knob from an internal structural ceiling it cannot.
# SSX escalated the latter into a GLOBAL hard stop blaming `work_budget`:
# measured on harness case 11 at atol<=1e-5, one CSX depth truncation (1,791
# of 100,000 cells used, topology complete) fired at 1.2% of the SSX ledger
# and collapsed the search from 98 cells to 2.
# ---------------------------------------------------------------------------

def test_truncation_cause_is_none_when_complete():
    S = np.array([
        [[0.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 1.0]],
        [[2.0, 0.0, 0.0, 1.0], [2.0, 2.0, 0.0, 1.0]],
    ])
    C = np.array([[1.0, 1.0, -1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    r = bez_csx(C, S, atol=1e-3, rational=True)
    assert r["budget_exhausted"] is False
    assert r["truncation_cause"] is None


def test_truncation_cause_reports_preflight_refusal():
    S = np.array([
        [[0.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 1.0]],
        [[2.0, 0.0, 0.0, 1.0], [2.0, 2.0, 0.0, 1.0]],
    ])
    C = np.array([[1.0, 1.0, -1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    r = bez_csx(C, S, atol=1e-3, rational=True, max_cells=0)
    assert r["budget_exhausted"] is True
    assert r["truncation_cause"] == "preflight"


def test_truncation_cause_reports_cells_when_the_allowance_runs_out():
    """A genuine resource shortfall must still say so."""
    S = np.array([
        [[0.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 1.0]],
        [[2.0, 0.0, 0.0, 1.0], [2.0, 2.0, 0.0, 1.0]],
    ])
    C = np.array([[0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 0.0, 1.0]])   # on-surface
    r = bez_csx(C, S, atol=1e-9, rational=True, max_cells=12)
    if r["budget_exhausted"]:
        assert r["truncation_cause"] in ("cells", "results", "boundary",
                                         "depth"), r["truncation_cause"]
