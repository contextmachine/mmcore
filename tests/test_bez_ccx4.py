import numpy as np
import pytest
from mmcore.numeric.intersection.ccx._bez_ccx4 import bez_ccx


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
# ---------------------------------------------------------------------------




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


def test_zero_allowance_preflights_before_net_build(monkeypatch):
    """L52 pin: bez_ccx already refuses a zero allowance before building the
    squared-distance net — lock that ordering so a refactor cannot regress
    it to the pre-preflight behavior the CSX twin had."""
    import mmcore.numeric.intersection.ccx._bez_ccx4 as ccx_mod

    calls = {"n": 0}
    orig = ccx_mod.curve_curve_squared_net_homog

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(
        ccx_mod, "curve_curve_squared_net_homog", counting)
    C1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    C2 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    r = bez_ccx(C1, C2, rational=False, max_cells=0)
    assert r["budget_exhausted"] is True
    assert r["cells_processed"] == 0
    assert calls["n"] == 0, "net built despite zero allowance"


# ---------------------------------------------------------------------------
# Cluster-4 burn-down (2026-07-25): the Phase-2 vector-residual hull prune
# must be translation-invariant.
#
# `_vector_residual_hull_excludes_zero` certifies "one Cartesian residual
# component cannot be zero, therefore no intersection".  Its inputs come from
# `_center_curve_homogeneous_for_exactness`, which divides by the net scale
# BEFORE translating, so a coordinate that is identically zero after
# translation is computed as fl(x/scale) - fl(origin*fl(w/scale)) — pure
# cancellation.  The old single-term margin `op_factor*(|lhs|+|rhs|)` was
# built from those already-cancelled values and so sat orders of magnitude
# BELOW the noise it had to absorb: a coplanar pair whose shared coordinate
# is exactly 0 in world space acquires same-sign noise once translated off
# that plane, and the prune deletes the whole domain.
#
# Reaching fixture: the two segments are the s1 v=0 edge and the s2 v=0
# boundary isocurve of the user's bilinear boundary-coincidence pair
# (docs/superpowers/plans/2026-07-25-ssx5-derived-envelopes-kickoff.md).
# They cross transversally at t = 0.377142857..., in-domain for both.
# ---------------------------------------------------------------------------

_COPLANAR_C = np.array([[-16.0, -27.0, 0.0, 1.0], [-36.0, 2.0, 0.0, 1.0]])
_COPLANAR_D = np.array([[-34.0, -7.0, 0.0, 1.0], [-19.0, -20.0, 0.0, 1.0]])
_COPLANAR_T = 122.0 / 175.0          # parameter on D
_COPLANAR_U = 0.377142857142857      # parameter on C


def _shift_homog(net, c):
    out = np.asarray(net, dtype=np.float64).copy()
    out[:, :3] -= np.asarray(c, dtype=np.float64) * out[:, 3:]
    return out


@pytest.mark.parametrize(
    "c",
    [
        (0.0, 0.0, 0.0),               # world frame: z is exactly 0
        (-22.0, -12.5, 2.5),           # the engine's k=2 canonical-frame center
        (0.0, 0.0, 2.5),               # z only
        (0.0, 0.0, 1e-3),              # z only, sub-unit
        (0.0, 0.0, 1e-12),             # z only, roundoff-scale
        (-22.0, -12.5, 0.0),           # in-plane only (z stays exactly 0)
        (1e4, -3e3, 7e2),              # far translation
    ],
    ids=["world", "canonical-k2", "z2.5", "z1e-3", "z1e-12", "in-plane", "far"],
)
def test_coplanar_crossing_survives_translation(c):
    """The crossing is a translation invariant; finding it must be too."""
    r = bez_ccx(_shift_homog(_COPLANAR_C, c), _shift_homog(_COPLANAR_D, c),
                atol=1e-3, rational=True)
    assert not r["budget_exhausted"]
    assert r["boundary_topology_complete"]
    assert len(r["isolated"]) == 1, (c, len(r["isolated"]))
    iso = r["isolated"][0]
    assert abs(float(iso["u"]) - _COPLANAR_U) < 1e-9, (c, iso["u"])
    assert abs(float(iso["v"]) - _COPLANAR_T) < 1e-9, (c, iso["v"])


@pytest.mark.parametrize(
    "c",
    [(0.0, 0.0, 0.0), (-22.0, -12.5, 2.5), (0.0, 0.0, 1e-3), (1e4, -3e3, 7e2)],
    ids=["world", "canonical-k2", "z1e-3", "far"],
)
def test_vector_residual_hull_prune_is_sound_under_translation(c):
    """Unit-level: the prune must never exclude zero for an intersecting pair.

    This is the unsound direction — a wrong True deletes solutions with no
    downstream recourse — so it is pinned separately from the end-to-end
    search above.
    """
    from mmcore.numeric.intersection.ccx._bez_ccx4 import (
        _vector_residual_hull_excludes_zero,
    )

    C = _shift_homog(_COPLANAR_C, c)
    D = _shift_homog(_COPLANAR_D, c)
    for depth in range(6):
        assert not _vector_residual_hull_excludes_zero(C, D, True, depth), (c, depth)


def test_vector_residual_hull_prune_still_separates_genuine_offsets():
    """The margin must not go so loose that a real separation stops pruning.

    A pair offset in z by a full model unit is separated in that component at
    every translation; the prune is what keeps such cells out of Phase 2.
    """
    from mmcore.numeric.intersection.ccx._bez_ccx4 import (
        _vector_residual_hull_excludes_zero,
    )

    for c in [(0.0, 0.0, 0.0), (-22.0, -12.5, 2.5), (1e4, -3e3, 7e2)]:
        C = _shift_homog(_COPLANAR_C, c)
        D = _shift_homog(_COPLANAR_D, c)
        D_off = D.copy()
        D_off[:, 2] += 1.0 * D_off[:, 3]
        assert _vector_residual_hull_excludes_zero(C, D_off, True, 0), c


@pytest.mark.parametrize("offset", [1e-3, 1e-5, 1e-6])
def test_centering_envelope_does_not_swallow_small_real_offsets(offset):
    """The absent-axis envelope must be tight, not merely finite.

    An offset well above the centering's roundoff is real geometry and must
    still separate the pair at every world position.  This is a far tighter
    bar than the model-unit guard above — 1e-6 on a model of extent ~36 is
    3e-8 relative, and ~7 orders below any modelling tolerance the engine is
    ever called with.

    MEASURED LIMIT (documented, not a target): the envelope is built from
    the operand magnitudes of the centering subtraction, and
    `_center_curve_homogeneous_for_exactness` divides by the net scale
    BEFORE translating, so those operands carry the model's WORLD POSITION.
    The prune's separation floor therefore still degrades with distance from
    the origin, but only at the rate the arithmetic actually loses:
    3.6e-15 at |T|=1, 3.6e-12 at 1e3, 3.5e-6 at 1e9 — all far below a
    default atol=1e-3.  (Before the 2026-07-26 review the factor was the
    siblings' `8192*(n1+n2)`, a Bernstein-chain constant misapplied to a
    single subtraction; the floor was then 2.9e-11 / 2.9e-8 / 2.9e-2, the
    last one ABOVE atol.)  This direction is safe — the prune declines to
    fire and the cell goes to the sound subdivision/Newton path, so it can
    never delete a solution.  Removing the residual dependence means
    reordering the centering to subtract before normalizing, which regresses
    `test_ccx4_exactness_contract.py::test_float_built_quadratic_subcurve_remains_an_overlap`
    — a calibrated fixture — so it is a separate, fixture-first change.
    """
    from mmcore.numeric.intersection.ccx._bez_ccx4 import (
        _vector_residual_hull_excludes_zero,
    )

    for c in [(0.0, 0.0, 0.0), (-22.0, -12.5, 2.5), (1e3, -2e3, 5e2)]:
        C = _shift_homog(_COPLANAR_C, c)
        D = _shift_homog(_COPLANAR_D, c)
        D[:, 2] += offset * D[:, 3]
        assert _vector_residual_hull_excludes_zero(C, D, True, 0), (c, offset)


# ---------------------------------------------------------------------------
# Cluster-4 follow-up (adversarial review, 2026-07-26), re-pinned for L62
# (owner decision 2026-08-19): the ACCEPT path needs anti-loosening guards
# too, not just the prune.
#
# The original pins asserted `len(isolated) == 0` for every resolvable gap —
# correct while an accepted root and an exactness claim were the same thing,
# and in direct conflict with the L62 membership contract (d_min <= atol,
# CLOSED: a 1e-8 gap at atol=1e-3 IS one tolerance contact).  What the
# 2026-07-26 guard bought is kept, aimed at the claims that still exist:
# the phantom-root defect class (an axis silently dropped from a
# certificate) now lives in the TAG, where
# test_absent_axis_is_checked_not_skipped pins it at unit level, and here
# end-to-end as "a resolvable nonzero gap never carries
# certification='exact'".  Membership itself gains the anti-loosening
# direction the old form never tested: gap > atol must REJECT at every
# world position.
# ---------------------------------------------------------------------------

def _parallel_planes_case(X0, gap):
    C1 = np.array([[X0, -1.0, -1.0], [X0, 1.0, 1.0]])
    C2 = np.array([[X0 + gap, -1.0, 1.0], [X0 + gap, 1.0, -1.0]])
    # X0 + gap rounds: the engine is judged against the geometry it was
    # actually given, not against the nominal parameter.
    realized = float(C2[0, 0] - C1[0, 0])
    return bez_ccx(C1, C2, atol=1e-3, rational=False), realized


@pytest.mark.parametrize("rel", [1e-12, 1e-10, 1e-8])
@pytest.mark.parametrize("X0", [0.0, 1.0, 1e3, 1e6, 1e9])
def test_parallel_planes_membership_tracks_atol_never_exact(X0, rel):
    """L62 §1 on the old accept-path grid, both directions, tag guarded.

    realized <= atol → exactly ONE isolated contact, tagged 'tolerance'
    (never 'exact' — the re-scoped 2026-07-26 guard); realized > atol →
    none.  At these world positions the polynomial net construction is
    translation-invariant, so no typed cannot-decide outcome may appear.
    """
    gap = rel * max(1.0, abs(X0))
    r, realized = _parallel_planes_case(X0, gap)
    assert "uncertified_contacts" not in r, (X0, rel, r)
    iso = r["isolated"]
    if realized <= 1e-3:
        assert len(iso) == 1, (X0, rel, realized, len(iso))
        assert iso[0]["certification"] == "tolerance", (X0, rel, iso)
        assert float(iso[0]["d_min"]) <= 1e-3
    else:
        assert len(iso) == 0, (X0, rel, realized, len(iso))


@pytest.mark.parametrize("gap", [1e-8, 5e-4, 2e-3])
@pytest.mark.parametrize("X0", [0.0, 1.0, 1e3, 1e6, 1e9])
def test_parallel_plane_verdict_tracks_realized_gap_at_every_position(X0, gap):
    """The invariance object post-L62 is the verdict on a fixed ABSOLUTE
    gap: the same realized geometry must get the same membership verdict
    at every world position.  (The old form held the RELATIVE gap fixed —
    the right invariant for an exactness certificate, structurally wrong
    for absolute-tolerance membership: rel=1e-8 is 1e-8 at the origin and
    10.0 at X0=1e9 — different geometry, different verdict, correctly.)
    Where float construction changes the realized geometry (at X0=1e9 a
    nominal 1e-8 gap rounds to exactly 0 — a transversal exact crossing),
    the expectation follows the realized gap, and the tag follows the
    strict envelope: 'exact' only when the realized gap is exactly zero.
    """
    r, realized = _parallel_planes_case(X0, gap)
    iso = r["isolated"]
    expected = 1 if realized <= 1e-3 else 0
    assert len(iso) == expected, (X0, gap, realized, len(iso))
    if expected:
        want_cert = "exact" if realized == 0.0 else "tolerance"
        assert iso[0]["certification"] == want_cert, (X0, gap, realized, iso)


def test_absent_axis_is_checked_not_skipped():
    """An axis under the centering envelope is still COMPARED against it.

    Regression pin for the review finding: `_eval_curve_scaled_components`
    skips axes whose scale is 0, so declaring an axis absent used to remove
    it from the membership test altogether. It must instead be tested
    against its own roundoff envelope.
    """
    from mmcore.numeric.intersection.ccx._bez_ccx4 import (
        _ccx_exactness_context, _strict_residual_ok,
    )

    X0, d = 1e6, 1e-8
    C1 = np.array([[X0, -1.0, -1.0], [X0, 1.0, 1.0]])
    C2 = np.array([[X0 + d, -1.0, 1.0], [X0 + d, 1.0, -1.0]])
    ctx = _ccx_exactness_context(C1, C2, False)
    assert ctx is not None
    ok, _p1, _p2 = _strict_residual_ok(C1, C2, 0.5, 0.5, False, ctx)
    assert not ok, "a 1e-8 gap in the x coordinate was certified as exact"
