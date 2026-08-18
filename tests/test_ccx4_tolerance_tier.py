"""L62 isolated tolerance tier — the owner membership contract (2026-08-18).

Membership of an isolated curve-curve contact is ``d_min <= tol``, CLOSED,
at every ``tol``: at any ``tol >= d_min`` the pair has exactly one isolated
intersection there; at any ``tol < d_min`` it has none; topology is correct
at every ``tol``.  The parameter values used below are instances of the law,
never special constants.  ``certification`` ('exact' | 'tolerance') and
``d_min`` are metadata — membership never depends on the tag.

Owner decisions recorded here (2026-08-18/19 session):
- there is no "band" outcome: a compact zero-free region of sub-``tol``
  distance is ONE isolated tangent contact at the certified argmin; a
  dip-through is its k exact crossings, distinguished at high precision;
  only a domain-end-anchored overlap is a long touch (L47, unchanged);
- the tier applies in 2D with the same predicate;
- a measurement whose envelope straddles ``tol`` at decision scale is a
  typed ``uncertified_contacts`` outcome — never a guess.
"""

import numpy as np
import pytest

from mmcore.numeric.intersection.ccx._bez_ccx4 import bez_ccx


ATOL = 1e-3


def _line(p0, p1):
    return np.array([p0, p1], dtype=np.float64)


def _crossing_pair(gap, offset=(0.0, 0.0, 0.0)):
    """Two transversal segments, closest approach exactly ``gap`` at
    u = v = 0.5 (the minimal repro of the L62 issue doc)."""
    off = np.asarray(offset, dtype=np.float64)
    C1 = _line([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) + off
    C2 = _line([0.0, -1.0, gap], [0.0, 1.0, gap]) + off
    return C1, C2


# ---------------------------------------------------------------------------
# The (gap, tol) law
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gap", [0.0, 1e-9, 1e-7, 2.5e-4, 5e-4, 9e-4,
                                 1e-3, 2e-3])
@pytest.mark.parametrize("tol", [1e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 1e-2])
def test_membership_tracks_tol_exactly(gap, tol):
    """Exactly one intersection iff tol >= gap (closed — the grid includes
    the equality instances gap == tol == 2.5e-4 / 5e-4 / 1e-3); the count
    never exceeds one."""
    C1, C2 = _crossing_pair(gap)
    r = bez_ccx(C1, C2, atol=tol, rational=False)
    expected = 1 if gap <= tol else 0
    assert len(r["isolated"]) == expected, (gap, tol, r["isolated"])
    assert r["overlaps"] == []
    assert "uncertified_contacts" not in r
    assert r["boundary_topology_complete"] is True
    if expected:
        iso = r["isolated"][0]
        assert iso["certification"] == ("exact" if gap == 0.0
                                        else "tolerance")
        # d_min is the net-certified measurement: equal to the gap down to
        # the net's own resolution (a 1e-9 gap measures as ~0 — its squared
        # trace sits below coefficient roundoff — while still a member).
        assert float(iso["d_min"]) == pytest.approx(gap, abs=1e-7)
        assert float(iso["u"]) == pytest.approx(0.5, abs=1e-3)
        assert float(iso["v"]) == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------------
# Translation invariance of tolerance acceptance (the hole 5d05ddc closed
# must stay closed: acceptance comes from the net measurement, which cannot
# decay with world position for polynomial inputs)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("offset", [
    (0.0, 0.0, 0.0),
    (1.0e4, -3.0e3, 7.0e2),
    (-2.0e6, 1.0e6, 5.0e5),
], ids=["origin", "1e4", "1e6"])
def test_tolerance_contact_is_translation_invariant(offset):
    gap = 5e-4
    C1, C2 = _crossing_pair(gap, offset)
    r = bez_ccx(C1, C2, atol=ATOL, rational=False)
    assert len(r["isolated"]) == 1, (offset, r["isolated"])
    iso = r["isolated"][0]
    assert iso["certification"] == "tolerance"
    assert float(iso["d_min"]) == pytest.approx(gap, rel=1e-3)
    assert float(iso["u"]) == pytest.approx(0.5, abs=1e-3)
    assert float(iso["v"]) == pytest.approx(0.5, abs=1e-3)
    assert r["boundary_topology_complete"] is True
    assert r["budget_exhausted"] is False


@pytest.mark.parametrize("offset", [(0.0, 0.0, 0.0), (1.0e4, -3.0e3, 7.0e2)],
                         ids=["origin", "1e4"])
def test_rejection_is_translation_invariant(offset):
    """The anti-loosening direction: a gap above tol rejects everywhere."""
    C1, C2 = _crossing_pair(2e-3, offset)
    r = bez_ccx(C1, C2, atol=ATOL, rational=False)
    assert r["isolated"] == [], (offset, r["isolated"])
    assert "uncertified_contacts" not in r


# ---------------------------------------------------------------------------
# Endpoint contacts (Phase-1 boundary analysis lifted from level 0 to tol²)
# ---------------------------------------------------------------------------

def test_endpoint_contact_curve_terminus_vs_interior():
    """A curve STARTING gap-above the other curve's interior: the component
    of {D <= tol} touches the v=0 domain edge only — one endpoint contact."""
    gap = 5e-4
    C1 = _line([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    C2 = _line([0.0, 0.0, gap], [0.0, 1.0, gap + 1.0])
    r = bez_ccx(C1, C2, atol=ATOL, rational=False)
    assert len(r["isolated"]) == 1, r["isolated"]
    iso = r["isolated"][0]
    assert iso["certification"] == "tolerance"
    assert float(iso["d_min"]) == pytest.approx(gap, rel=1e-3)
    assert float(iso["u"]) == pytest.approx(0.5, abs=1e-3)
    assert float(iso["v"]) == pytest.approx(0.0, abs=1e-3)
    # membership still tracks tol: below the gap, no contact
    r2 = bez_ccx(C1, C2, atol=1e-4, rational=False)
    assert r2["isolated"] == []


def test_endpoint_contact_corner_to_corner():
    """Both termini within tol of each other: a corner contact, once."""
    gap = 5e-4
    C1 = _line([-1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    C2 = _line([0.0, gap, 0.0], [1.0, 1.0, 0.0])
    r = bez_ccx(C1, C2, atol=ATOL, rational=False)
    assert len(r["isolated"]) == 1, r["isolated"]
    iso = r["isolated"][0]
    assert iso["certification"] == "tolerance"
    assert float(iso["d_min"]) == pytest.approx(gap, rel=1e-3)
    assert float(iso["u"]) == pytest.approx(1.0, abs=2e-3)
    assert float(iso["v"]) == pytest.approx(0.0, abs=2e-3)
    r2 = bez_ccx(C1, C2, atol=1e-4, rational=False)
    assert r2["isolated"] == []


# ---------------------------------------------------------------------------
# Component rules: no bands, no double counts (owner decision 2026-08-18)
# ---------------------------------------------------------------------------

def test_tangent_graze_is_exactly_one_contact():
    """A parabola grazing 5e-5 above a line: the sub-tol region is a long
    interior valley (~200x the param tol), and it is ONE isolated tangent
    contact at the certified argmin — never several points along the
    valley, never a 'band'."""
    a, b = 5e-5, 0.1
    C1 = _line([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    # y(t) = a + b*(t-1/2)^2, apex a at t=1/2; ends at a+b/4 >> tol
    C2 = np.array([
        [-1.0, a + b / 4.0, 0.0],
        [0.0, a - b / 4.0, 0.0],
        [1.0, a + b / 4.0, 0.0],
    ])
    r = bez_ccx(C1, C2, atol=ATOL, rational=False)
    assert len(r["isolated"]) == 1, r["isolated"]
    iso = r["isolated"][0]
    assert iso["certification"] == "tolerance"
    assert float(iso["d_min"]) == pytest.approx(a, rel=1e-2)
    assert float(iso["u"]) == pytest.approx(0.5, abs=5e-3)
    assert float(iso["v"]) == pytest.approx(0.5, abs=5e-3)
    assert r["overlaps"] == []
    assert r["boundary_topology_complete"] is True


def test_dip_through_is_two_exact_roots_no_extra_contact():
    """A curve dipping 5e-5 THROUGH the other and back out inside one
    sub-tol region: the two transversal crossings are the topology — two
    exact roots, distinguished at high precision, and the tolerance tier
    must not add a third contact at the interior saddle (d = 5e-5 there,
    a member by distance, but the component contains certified zeros and
    is resolved by the exact machinery alone)."""
    c = 2e-2
    C1 = _line([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    # y(t) = c*(t-0.45)*(t-0.55): roots at t=0.45/0.55, apex -c*2.5e-3,
    # ends c*0.2475 >> tol (the component is compact-interior)
    C2 = np.array([
        [-1.0, 0.2475 * c, 0.0],
        [0.0, -0.2525 * c, 0.0],
        [1.0, 0.2475 * c, 0.0],
    ])
    r = bez_ccx(C1, C2, atol=ATOL, rational=False)
    us = sorted(float(i["u"]) for i in r["isolated"])
    assert len(us) == 2, r["isolated"]
    assert us[0] == pytest.approx(0.45, abs=5e-3)
    assert us[1] == pytest.approx(0.55, abs=5e-3)
    for iso in r["isolated"]:
        assert iso["certification"] == "exact"
    assert r["overlaps"] == []


# ---------------------------------------------------------------------------
# 2D: same predicate (owner decision 2026-08-19)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gap,expected", [(5e-4, 1), (2e-3, 0)])
def test_tier_applies_in_2d(gap, expected):
    """In 2D two transversal segments always meet exactly, so the canonical
    2D near-miss is the tangent graze: a parabola whose apex passes ``gap``
    above a line without crossing."""
    C1 = np.array([[-1.0, 0.0], [1.0, 0.0]])
    C2 = np.array([
        [-1.0, gap + 0.25], [0.0, gap - 0.25], [1.0, gap + 0.25],
    ])
    r = bez_ccx(C1, C2, atol=ATOL, rational=False)
    assert len(r["isolated"]) == expected, (gap, r["isolated"])
    if expected:
        iso = r["isolated"][0]
        assert iso["certification"] == "tolerance"
        assert float(iso["d_min"]) == pytest.approx(gap, rel=1e-2)


# ---------------------------------------------------------------------------
# The typed cannot-decide tail (never a guess)
# ---------------------------------------------------------------------------

def test_far_translated_rational_boundary_is_typed_not_guessed():
    """Rational curves with unequal weights at |T| = 1e12: the homogeneous
    cross-products round at world scale and the net measurement genuinely
    cannot resolve tolerance-sized structure (eps_d >= atol).  The engine
    must return the typed ``uncertified_contacts`` outcome with topology
    not claimed complete — never a silent accept or reject."""
    X0 = 1.0e12
    gap = 5e-4
    w = np.array([1.0, 2.0])
    C1_xyz = np.array([[X0 - 1.0, 0.0, 0.0], [X0 + 1.0, 0.0, 0.0]])
    C2_xyz = np.array([[X0, -1.0, gap], [X0, 1.0, gap]])
    C1 = np.concatenate([C1_xyz * w[:, None], w[:, None]], axis=1)
    C2 = np.concatenate([C2_xyz * w[:, None], w[:, None]], axis=1)
    r = bez_ccx(C1, C2, atol=ATOL, rational=True)
    assert r["isolated"] == [], r["isolated"]
    assert "uncertified_contacts" in r, sorted(r)
    assert r["boundary_topology_complete"] is False
    entry = r["uncertified_contacts"][0]
    assert float(entry["envelope"]) >= ATOL


# ---------------------------------------------------------------------------
# Review 2026-08-19 regressions (adversarial verification of the L62 commit;
# every fixture below reproduced a confirmed defect before its fix)
# ---------------------------------------------------------------------------

def _extent_crossing(gap, L=3000.0):
    """Transversal quadratic pair of extent ±L with a pure z-gap — the
    configuration where the global (extent²-scaled) net envelope alone
    opened a false-accept window of up to 0.68·atol."""
    C1 = np.array([[-L, 0.0, 0.0], [0.0, 0.0, 0.0], [L, 0.0, 0.0]])
    C2 = np.array([[0.0, -L, gap], [0.0, 0.0, gap], [0.0, L, gap]])
    return C1, C2


@pytest.mark.parametrize("gap,expected", [
    (5e-4, 1), (1e-3, 1),          # members (closed at the boundary)
    (1.2e-3, 0), (1.5e-3, 0),      # the reviewed false accepts — must reject
])
def test_membership_holds_at_large_extent(gap, expected):
    """The law must hold at |ctrl| ~ 3e3 with atol=1e-3 (an ordinary part
    modelled in mm at micron tolerance): acceptance comes from the sharper
    of the net and direct measurements, so the accept window is the
    measurement's true noise floor, never the net's extent² envelope."""
    C1, C2 = _extent_crossing(gap)
    r = bez_ccx(C1, C2, atol=ATOL, rational=False)
    assert len(r["isolated"]) == expected, (gap, r["isolated"])
    assert "uncertified_contacts" not in r, (gap, r)
    if expected:
        assert float(r["isolated"][0]["d_min"]) <= ATOL + 1e-9


@pytest.mark.parametrize("L", [1.0, 300.0])
def test_super_tol_ridge_never_merges_two_contacts(L):
    """Two 8e-4 endpoint contacts separated by a 1.02e-3 ridge stay TWO
    contacts at every extent — the widened connectivity walk used to step
    over the ridge at L=300 and silently merge them."""
    z0 = z2 = 8e-4
    z1 = (4 * 1.02e-3 - z0 - z2) / 2.0
    C1 = np.array([[-L, 0.0, 0.0], [L, 0.0, 0.0]])
    C2 = np.array([[-L, 0.0, z0], [0.0, 0.0, z1], [L, 0.0, z2]])
    r = bez_ccx(C1, C2, atol=ATOL, rational=False)
    us = sorted(round(float(i["u"]), 3) for i in r["isolated"])
    assert us == [0.0, 1.0], (L, r["isolated"])


def test_band_evidence_does_not_stand_down_the_tier():
    """Band evidence at one terminus (a tangential start) must not disarm
    the tier across the whole call: the unrelated far-end contact at
    d = 5e-4 is a member and ships alongside the exact tangent root."""
    C1 = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    C2 = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                   [6.0, -0.5, 0.0], [8.0, -0.0005, 0.0]])
    r = bez_ccx(C1, C2, atol=ATOL, rational=False)
    got = sorted((round(float(i["u"]), 2), i["certification"])
                 for i in r["isolated"])
    assert (0.0, "exact") in got, got
    assert (0.8, "tolerance") in got, got
    assert len(got) == 2, got
    assert r["boundary_topology_complete"] is True


def test_triple_dip_reports_all_three_contacts():
    """h(t) = 5e-4 + A·(t(1-t)(t-1/2))² against a line: two terminus
    contacts and one interior tangent contact, all at d = 5e-4, separated
    by 5e-3 ridges.  The overlap-class stand-down used to return zero of
    them with topology claimed complete; the unclamped minimizer used to
    slide over the ridges and lose the interior one."""
    from math import comb

    def mono_to_bern(a):
        n = len(a) - 1
        return [sum(comb(i, k) / comb(n, k) * a[k] for k in range(i + 1))
                for i in range(n + 1)]

    p = np.array([0.0, 0.5, -1.5, 1.0])          # t(1-t)(t-1/2) monomials
    p2 = np.polynomial.polynomial.polymul(p, p)
    peak = float(np.max(np.polynomial.polynomial.polyval(
        np.linspace(0.0, 1.0, 1001), p2)))
    h = np.zeros(7)
    h[:len(p2)] = (5e-3 / peak) * p2
    h[0] += 5e-4
    C1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    C2 = np.column_stack([mono_to_bern([0, 1, 0, 0, 0, 0, 0]),
                          mono_to_bern(h.tolist()), np.zeros(7)])
    r = bez_ccx(C1, C2, atol=ATOL, rational=False)
    us = sorted(round(float(i["u"]), 2) for i in r["isolated"])
    assert us == [0.0, 0.5, 1.0], r["isolated"]
    for iso in r["isolated"]:
        assert float(iso["d_min"]) == pytest.approx(5e-4, rel=1e-2)


def test_curved_component_is_one_contact_at_the_argmin():
    """Grid-verified fixture with exactly three connected components of
    {D <= atol}: the engine must report exactly one contact per component,
    each at the component argmin — the straight-chord connectivity used to
    ship one component twice, and this cubic pair pins the valley-following
    walk against that."""
    C1 = np.array([
        [0.18704965, -0.0073058, 0.0], [0.398129, -0.57963249, 0.0],
        [-0.25633933, 0.09465561, 0.0], [0.79909922, -0.21653492, 0.0]])
    C2 = np.array([
        [0.01501163, 0.008352, -0.01421305],
        [0.8892442, -0.37590185, 0.0064144],
        [-0.28783153, -0.04980575, 0.0240725],
        [-0.9890063, 0.40801458, 0.02848489]])
    r = bez_ccx(C1, C2, atol=0.05, rational=False)
    got = sorted((round(float(i["u"]), 2), round(float(i["v"]), 2))
                 for i in r["isolated"])
    # 401x401 grid truth: argmins of the three components
    assert got == [(0.05, 0.10), (0.10, 0.41), (0.76, 0.18)], r["isolated"]


def test_disconnected_components_close_in_space_stay_two_contacts():
    """A loop whose two termini pass 3e-4/3.5e-4 above nearly the same
    point of the line: two disconnected components whose witnesses are
    within atol in 3D.  The removed space-radius dedup used to absorb the
    second one; connectivity is the only component discriminator."""
    C1 = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    C2 = np.array([[0.0, 3e-4, 0.0], [2.0, 1.0, 0.0],
                   [-2.0, 1.0, 0.0], [0.0, 3.5e-4, 0.0]])
    r = bez_ccx(C1, C2, atol=ATOL, rational=False)
    vs = sorted(round(float(i["v"]), 2) for i in r["isolated"])
    assert vs == [0.0, 1.0], r["isolated"]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_endpoint_prefilter_survives_rotation_at_gap_equals_tol(seed):
    """Collinear end-to-end pair at gap == atol, rigidly rotated: the
    endpoint pre-filter's bar carries net-construction roundoff and must
    be envelope-slacked like every other level-atol bar (unslacked it
    dropped the contact for 134/300 rotations by a 1-ulp coefficient
    rounding)."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    C1 = np.array([[-1.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]) @ R.T
    C2 = np.array([[ATOL, 0.0, 0.0], [ATOL + 0.5, 0.0, 0.0],
                   [ATOL + 1.0, 0.0, 0.0]]) @ R.T
    r = bez_ccx(C1, C2, atol=ATOL, rational=False)
    assert len(r["isolated"]) == 1, (seed, r["isolated"])
    assert float(r["isolated"][0]["d_min"]) == pytest.approx(ATOL, rel=1e-6)


# ---------------------------------------------------------------------------
# Adapter-level pins (nurbs_ccx / nurbs_ccx_multiple)
# ---------------------------------------------------------------------------

def _ntuple_line(p0, p1):
    from mmcore.nurbs._nurbs_eval import NURBSCurveTuple
    return NURBSCurveTuple(order=2, knot=np.array([0.0, 0.0, 1.0, 1.0]),
                           control_points=np.array([p0, p1], dtype=float),
                           weights=np.array([1.0, 1.0]))


def test_adapter_closed_seam_keeps_gap_equals_tol():
    """Mutation kill for the adapter's re-verification: reverting the
    closed seam check (or dropping its operand slack) loses the
    gap == tol member the engine certified."""
    from mmcore.numeric.intersection.ccx._nccx4 import nurbs_ccx
    c1 = _ntuple_line([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    c2 = _ntuple_line([0.0, -1.0, ATOL], [0.0, 1.0, ATOL])
    iso, _ovl, status = nurbs_ccx(c1, c2, tol=ATOL)
    assert iso is not None and len(iso) == 1, (iso, status)
    assert str(iso[0]["certification"]) == "tolerance"
    assert float(iso[0]["d_min"]) == pytest.approx(ATOL, rel=1e-9)


def test_adapter_continues_past_typed_cannot_decide():
    """A per-candidate typed cannot-decide (the |T|=1e12 unequal-weights
    pair) must not abort the scan: the unrelated clean crossing ships, the
    aggregate is marked incomplete, and the typed payload reaches the
    status ledger with global parameters and curve indices."""
    from mmcore.nurbs._nurbs_eval import NURBSCurveTuple
    from mmcore.numeric.intersection.ccx._nccx4 import nurbs_ccx_multiple
    a = _ntuple_line([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    b = _ntuple_line([0.0, -1.0, 0.0], [0.0, 1.0, 0.0])
    X0 = 1.0e12
    far1 = NURBSCurveTuple(
        order=2, knot=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([[X0 - 1.0, 0.0, 0.0], [X0 + 1.0, 0.0, 0.0]]),
        weights=np.array([1.0, 2.0]))
    far2 = NURBSCurveTuple(
        order=2, knot=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([[X0, -1.0, 5e-4], [X0, 1.0, 5e-4]]),
        weights=np.array([1.0, 2.0]))
    iso, _ovl, status = nurbs_ccx_multiple([a, b, far1, far2], tol=ATOL)
    assert iso is not None and len(iso) == 1, (iso, status)
    assert float(iso[0]["u"]) == pytest.approx(0.5, abs=1e-6)
    assert status["complete"] is False
    payload = status["uncertified_contacts"]
    assert payload and payload[0]["entries"], status
    entry = payload[0]["entries"][0]
    assert {entry["curve1_i"], entry["curve2_i"]} == {2, 3}
