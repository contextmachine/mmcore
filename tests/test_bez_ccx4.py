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


def test_soft_cell_budget_covers_phase1_and_all_phase2_intervals():
    """The public cap is one shared allowance, not a per-interval reset."""
    C1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    C2 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    empty = bez_ccx(C1, C2, rational=False, max_cells=0)
    assert empty["budget_exhausted"] is True
    assert empty["boundary_topology_complete"] is False
    assert empty["cells_processed"] == 0

    partial = bez_ccx(C1, C2, rational=False, max_cells=5)
    assert partial["budget_exhausted"] is True
    assert partial["cells_processed"] <= 5

    complete = bez_ccx(C1, C2, rational=False, max_cells=32)
    assert complete["budget_exhausted"] is False
    assert complete["cells_processed"] <= 32
    assert len(complete["isolated"]) == 1


def test_phase2_max_depth_reports_unresolved_cell(monkeypatch):
    """Reaching ``max_depth`` is an incomplete search, not a clean prune."""
    import mmcore.numeric.intersection._sq_dist_classify as classify
    import mmcore.numeric.intersection.ccx._bez_ccx4 as ccx_mod

    monkeypatch.setattr(classify, "_check_min_of_net", lambda *args: False)
    monkeypatch.setattr(classify, "_check_lipschitz", lambda *args: False)
    monkeypatch.setattr(
        ccx_mod, "bernstein_partial_derivative_coeffs",
        lambda *args, **kwargs: np.array([[-1.0], [1.0]]),
    )
    monkeypatch.setattr(
        ccx_mod, "newton_ccx",
        lambda *args, **kwargs: (0.5, 0.5, np.ones(3), np.ones(2)),
    )

    C = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    roots, exhausted, cells = ccx_mod._phase2_ccx(
        np.zeros((2, 2)), C, C, C, C,
        0.0, 1.0, 0.0, 1.0,
        1e-6, False, 1e-12, 1e-12,
        max_depth=0, max_cells=10,
    )

    assert roots == []
    assert cells == 1
    assert exhausted is True


def test_boundary_root_cap_never_becomes_partial_overlap_topology():
    C = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    # Identical curves put roots on all four parameter-square boundaries.
    # A cap below that count must yield an explicit partial result, not an
    # overlap inferred from whichever endpoints happened to fit.
    partial = bez_ccx(
        C, C, atol=1e-3, rational=False,
        max_cells=100, max_results=2,
    )
    assert partial["budget_exhausted"] is True
    assert partial["boundary_topology_complete"] is False
    assert partial["isolated"] == []
    assert partial["overlaps"] == []


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
    """The legacy tolerance trace must not force an exact overlap claim.

    These 8-decimal fixtures follow the same path to about 3e-9 but are not
    coefficient-identical.  V4 may return strict roots/an exact certified
    overlap, or conservatively stop partial; it must not grind indefinitely
    trying to discretize the near-overlap valley.
    """
    old = bez_ccx_old(curve1, curve2)
    new = bez_ccx(curve1, curve2, atol=1e-3, rational=False)
    assert len(old["overlaps"]) == 1
    # New should find overlap (or at worst, multiple isolated points along the overlap)
    has_overlap = len(new["overlaps"]) >= 1
    has_many_isolated = len(new["isolated"]) >= 2
    conservative_partial = new.get("budget_exhausted", False)
    assert has_overlap or has_many_isolated or conservative_partial, (
        f"New found: {len(new['isolated'])} isolated, "
        f"{len(new['overlaps'])} overlaps, partial={conservative_partial}"
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


# ---------------------------------------------------------------------------
# Ledger L47: near-coincident / non-affine coincident overlap semantics
# (USER DECISION 2026-07-12: residual-certified tier alongside the exact one)
# ---------------------------------------------------------------------------

def _monomial_to_bernstein(a):
    """Exact monomial -> Bernstein-n coefficient conversion (1-D)."""
    from math import comb
    n = len(a) - 1
    return [sum(comb(i, k) / comb(n, k) * a[k] for k in range(i + 1))
            for i in range(n + 1)]


@pytest.mark.parametrize("reverse", [False, True], ids=["same-dir", "reversed"])
def test_near_coincident_pair_ships_tolerance_overlap(reverse):
    """Monotone-x cubic vs itself offset 1e-9 in y (atol=1e-3): a crossing-
    free coincidence at tolerance (no tangent of the curve is parallel to
    the offset, so the offset twin never crosses the original). The exact-
    affine narrowing lost the overlap AND misbilled the failure to the
    budget; the residual tier must certify it (dense-sample inversion
    pairing, residual <= atol) and ship it complete."""
    C1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.6, 0.0],
                   [2.0, -0.2, 0.0], [3.0, 0.4, 0.0]])
    C2 = C1.copy()
    C2[:, 1] += 1e-9
    if reverse:
        C2 = C2[::-1].copy()
    r = bez_ccx(C1, C2, atol=1e-3, rational=False)
    assert len(r["overlaps"]) == 1, r
    ov = r["overlaps"][0]
    assert ov["certification"] == "tolerance"
    assert ov["u_range"] == pytest.approx((0.0, 1.0), abs=1e-6)
    v0, v1 = ov["v_range"]
    if reverse:
        assert (v0, v1) == pytest.approx((1.0, 0.0), abs=1e-6)
    else:
        assert (v0, v1) == pytest.approx((0.0, 1.0), abs=1e-6)
    assert float(ov["residual_max"]) <= 1e-3
    assert r["isolated"] == []
    assert r["budget_exhausted"] is False
    assert r["boundary_topology_complete"] is True


def test_offset_twin_with_vertical_tangent_is_not_cleanly_promoted():
    """L54[A2-1] corollary, measured during the on-node fix: a y-offset of a
    curve whose tangent turns PARALLEL to the offset (curve1 has a vertical
    tangent near u=0.75) genuinely CROSSES the original there — the offset
    slides the curve along itself locally, so 'offset pair' does NOT imply
    crossing-free. The bridged flip test detects it (the transverse
    direction reverses through the tangent zone) and promotion is refused:
    the honest outcome is the woven-family typed partial, not a clean
    overlap that would merge real crossing structure."""
    C2 = curve1.copy()
    C2[:, 1] += 1e-9
    r = bez_ccx(curve1, C2, atol=1e-3, rational=False)
    assert len(r["overlaps"]) == 0
    assert r["budget_exhausted"] is True
    assert r["boundary_topology_complete"] is False
    assert "uncertified_overlap_span" in r, sorted(r)


def test_exact_affine_overlap_certification_is_exact():
    """Coefficient-identical curves keep the exact certificate (unchanged
    semantics; the new field just names it)."""
    C = np.array([[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]])
    r = bez_ccx(C, C.copy(), atol=1e-3, rational=False)
    assert len(r["overlaps"]) == 1
    assert r["overlaps"][0]["certification"] == "exact"
    assert r["budget_exhausted"] is False


def test_non_affine_reparameterized_exact_overlap_certifies():
    """Same locus, non-affine parameter map q(s) = (s^2+s)/2: a genuine
    exact overlap that no affine identity can certify. The residual tier
    must ship it (residual ~ roundoff) instead of flooding isolated roots
    or stopping partial."""
    # C1(t) = (2t, 2t(1-t), 0), degree 2
    C1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    # C2(s) = C1(q(s)) with q(s) = (s^2+s)/2 (monotone [0,1]->[0,1]):
    #   x = s^2 + s;  y = (s^2+s) - (s^4 + 2 s^3 + s^2)/2
    bx = _monomial_to_bernstein([0.0, 1.0, 1.0, 0.0, 0.0])
    by = _monomial_to_bernstein([0.0, 1.0, 0.5, -1.0, -0.5])
    C2 = np.column_stack([bx, by, np.zeros(5)])
    r = bez_ccx(C1, C2, atol=1e-3, rational=False)
    assert len(r["overlaps"]) == 1, r
    ov = r["overlaps"][0]
    assert ov["certification"] == "tolerance"
    assert ov["u_range"] == pytest.approx((0.0, 1.0), abs=1e-6)
    assert ov["v_range"] == pytest.approx((0.0, 1.0), abs=1e-6)
    assert float(ov["residual_max"]) <= 1e-9
    assert r["isolated"] == []
    assert r["budget_exhausted"] is False


def test_realistic_woven_near_coincident_reports_typed_span():
    """curve1 vs curve2 follow the same path to ~3e-9 but WEAVE across each
    other — genuine crossings at fitting-noise amplitude. Crossing evidence
    blocks tolerance promotion (the approved no-distinct-roots guard: never
    merge crossing structure), yet the crossings sit below the strict
    certification scale (the curves are ~1e-9-parallel there), so they
    cannot ship as isolated roots either. The honest outcome is the typed
    uncertified span with topology incomplete, at bounded fallback cost —
    not a silent bare-budget grind."""
    r = bez_ccx(curve1, curve2, atol=1e-3, rational=False)
    assert len(r["overlaps"]) == 0
    assert r["budget_exhausted"] is True
    assert r["boundary_topology_complete"] is False
    lo, hi = r["uncertified_overlap_span"]
    assert (lo, hi) == pytest.approx((0.0, 0.8276), abs=1e-3)
    assert r["cells_processed"] < 5_000


def test_interior_crossings_inside_tolerance_band_stay_isolated():
    """Sub-atol-topology invariant (the L42/CSX negative result, 1-D form):
    a pair whose ENDS are within tolerance but whose interior CROSSES twice
    must never be merged into a tolerance overlap — the two transversal
    roots are the topology. The residual tier's transverse-direction flip
    test is the guard."""
    # C1 = flat segment y=0 (degree 2); C2 = (t, f(t)) with
    # f(t) = 1e-9 + 5e-4 (t-0.4)(t-0.6): f(0)=f(1)=1.2e-4 (within atol),
    # two sign changes near t=0.4 and t=0.6, |f| <= 1.2e-4 everywhere.
    C1 = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    f = _monomial_to_bernstein([1e-9 + 5e-4 * 0.24, -5e-4, 5e-4])
    C2 = np.column_stack([[0.0, 0.5, 1.0], f, np.zeros(3)])
    r = bez_ccx(C1, C2, atol=1e-3, rational=False)
    assert len(r["overlaps"]) == 0, r["overlaps"]
    assert len(r["isolated"]) == 2, r
    us = sorted(float(i["u"]) for i in r["isolated"])
    assert us[0] == pytest.approx(0.4, abs=5e-3)
    assert us[1] == pytest.approx(0.6, abs=5e-3)
    assert r["budget_exhausted"] is False


def test_uncertifiable_overlap_class_reports_typed_span_not_bare_budget():
    """A valley-confirmed pair that NEITHER certificate can promote must
    name the structure — uncertified_overlap_span + topology incomplete —
    instead of a bare budget_exhausted with topology claimed complete."""
    # C2 = C1 + (0, f(t), 0) on the curved cubic fixture, with
    # f(t) = 1e-9 + 2e-3 t^8: a LONG 1e-9-coincident band near u=0 (an
    # undiscretizable diagonal valley — the curved y makes the residual
    # net straddle zero along it), sub-atol until t ~ 0.92, 2e-3 > atol at
    # t=1 — so only the u=0 end is pairable and no span candidate exists.
    # Same side everywhere (no crossings to lose).
    def _elevate_once(ctrl):
        n = len(ctrl) - 1
        out = [ctrl[0]]
        for i in range(1, n + 1):
            a = i / (n + 1)
            out.append(a * ctrl[i - 1] + (1.0 - a) * ctrl[i])
        out.append(ctrl[-1])
        return np.asarray(out)

    C1 = curve1
    C2 = curve1.copy()
    for _ in range(5):                       # degree 3 -> 8, exact
        C2 = _elevate_once(C2)
    C2[:, 1] += _monomial_to_bernstein([1e-9] + [0.0] * 7 + [2e-3])
    r = bez_ccx(C1, C2, atol=1e-3, rational=False)
    assert len(r["overlaps"]) == 0
    assert r["budget_exhausted"] is True, r
    # the sub-atol band could not be discretized: the typed span must name
    # the uncertifiable structure and topology must not be claimed complete
    assert r["boundary_topology_complete"] is False
    assert "uncertified_overlap_span" in r, sorted(r)
    lo, hi = r["uncertified_overlap_span"]
    assert 0.0 <= lo < hi <= 1.0


@pytest.mark.parametrize("a, b, t_star", [
    (3e-4, 3e-4, 0.50),    # crossing exactly on sample node 32/64
    (3e-4, 9e-4, 0.25),    # crossing exactly on sample node 16/64
    (3e-4, 7e-4, 0.30),    # off-node control (worked before the fix)
], ids=["node-0.5", "node-0.25", "off-node-0.3"])
def test_on_node_interior_crossing_is_never_merged(a, b, t_star):
    """L54[A2-1] (audit-confirmed): a transversal crossing landing exactly ON
    one of the 65 sample nodes made that sample root-like, and the flip loop
    skipped BOTH straddling pairs — the crossing was silently absorbed into a
    certification='tolerance' overlap reported complete. The flip test must
    bridge across root-like runs (compare consecutive GAP samples), with the
    bracket at the intervening root-like node — the root is exactly there.
    Dyadic-64 fractions (0.25, 0.5, 0.75...) are the most common crossing
    locations in CAD by symmetry."""
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # y(t) = 3t(1-t)[a(1-t) - b t]: shares both endpoints with the line and
    # crosses it transversally at t* = a/(a+b), all within a sub-atol band.
    cubic = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, a, 0.0],
                      [2.0 / 3.0, -b, 0.0], [1.0, 0.0, 0.0]])
    r = bez_ccx(line, cubic, atol=1e-3, rational=False)
    assert len(r["overlaps"]) == 0, r["overlaps"]
    us = sorted(float(i["u"]) for i in r["isolated"])
    assert any(abs(u - t_star) < 5e-3 for u in us), (t_star, us)
