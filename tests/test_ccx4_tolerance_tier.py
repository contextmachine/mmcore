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
