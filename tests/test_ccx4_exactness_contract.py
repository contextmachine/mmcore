"""Exact-set contracts for the public Bezier CCX result schema.

Re-pinned 2026-07-12 (ledger L56): this suite predated the L47 residual-
certified overlap tier and still asserted the pre-L47 semantics
("sub-tolerance sets are never reported as overlaps").  Under the
USER-APPROVED L47 contract a crossing-free pair whose dense inversion
pairs at residual <= atol ships as ONE overlap with
``certification='tolerance'`` — the exactness property survives in
sharpened form: such pairs are NEVER certified ``'exact'`` (measured:
even a 5e-324 offset stays 'tolerance'), and the exact-affine identity
tier remains magnitude/weight-scale independent.  The suite went
stale-red silently because it was not in the kickoff §3 gate list; it is
now part of the unit-batch gate.
"""

import numpy as np
import pytest

from mmcore.numeric.intersection.ccx._bez_ccx4 import (
    _overlap_mapping_is_identity,
    bez_ccx,
)


ATOL = 1e-3


def _subcurve(control, lo, hi):
    """Exact de Casteljau restriction used only to build overlap controls."""
    control = np.asarray(control, dtype=np.float64)

    def split(curve, t):
        work = curve.copy()
        left = [work[0].copy()]
        right = [work[-1].copy()]
        for level in range(1, len(curve)):
            work = (1.0 - t) * work[:-1] + t * work[1:]
            left.append(work[0].copy())
            right.append(work[-1].copy())
        return np.asarray(left), np.asarray(right[::-1])

    if lo > 0.0:
        _, control = split(control, lo)
    if hi < 1.0:
        local_hi = (hi - lo) / (1.0 - lo) if lo > 0.0 else hi
        control, _ = split(control, local_hi)
    return control


def _homogeneous(points, weights, scale=1.0):
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return np.concatenate(
        [points * weights[:, None], weights[:, None]], axis=1) * scale


@pytest.mark.parametrize(
    "dz",
    [0.5 * ATOL, 1e-12, np.nextafter(0.0, 1.0)],
)
def test_sub_tolerance_parallel_polynomial_lines_are_tolerance_never_exact(dz):
    first = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    second = first.copy()
    second[:, 2] = dz

    result = bez_ccx(first, second, atol=ATOL, rational=False)

    assert result["isolated"] == []
    assert result["budget_exhausted"] is False
    assert result["boundary_topology_complete"] is True
    assert len(result["overlaps"]) == 1
    overlap = result["overlaps"][0]
    # The exactness contract proper: a nonzero offset must never be
    # certified 'exact', no matter how far below atol it sits.
    assert overlap["certification"] == "tolerance"
    assert overlap["residual_max"] == pytest.approx(dz, abs=1e-15)
    assert np.allclose(overlap["u_range"], (0.0, 1.0), atol=0.0)
    assert np.allclose(overlap["v_range"], (0.0, 1.0), atol=0.0)


@pytest.mark.parametrize("scale1,scale2", [(1.0, 1.0), (1e-30, 1e30)])
def test_sub_tolerance_parallel_rational_lines_are_weight_scale_invariant(
        scale1, scale2):
    dz = 0.5 * ATOL
    weights = np.array([1.0, 2.0])
    first_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    second_xyz = first_xyz.copy()
    second_xyz[:, 2] = dz
    first = _homogeneous(first_xyz, weights, scale1)
    second = _homogeneous(second_xyz, weights, scale2)

    result = bez_ccx(first, second, atol=ATOL, rational=True)

    # Weight-scale invariance: the (1,1) and (1e-30,1e30) parametrizations
    # must produce the SAME classification — one tolerance overlap whose
    # residual is the geometric gap, never an 'exact' claim.
    assert result["isolated"] == []
    assert result["budget_exhausted"] is False
    assert result["boundary_topology_complete"] is True
    assert len(result["overlaps"]) == 1
    overlap = result["overlaps"][0]
    assert overlap["certification"] == "tolerance"
    assert overlap["residual_max"] == pytest.approx(dz, rel=1e-6)


def test_sub_tolerance_quadratic_hump_is_one_tolerance_overlap():
    # The hump touches the line exactly at both ends and deviates 0.25*atol
    # at its apex with no transverse sign flip — under the L47 contract a
    # non-flipping in-band pair is ONE tolerance overlap whose endpoints
    # carry the exact touches (they must not double-report as isolated).
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    hump = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5 * ATOL, 0.0],
        [1.0, 0.0, 0.0],
    ])

    result = bez_ccx(line, hump, atol=ATOL, rational=False)

    assert result["budget_exhausted"] is False
    assert result["boundary_topology_complete"] is True
    assert result["isolated"] == []
    assert len(result["overlaps"]) == 1
    overlap = result["overlaps"][0]
    assert overlap["certification"] == "tolerance"
    # apex of the quadratic = ctrl_y / 2
    assert overlap["residual_max"] == pytest.approx(0.25 * ATOL, rel=1e-9)
    # The span itself IS where the exact endpoint touches live (slice-5
    # review finding): a mislocated/partial span (e.g. (0, 0.5)) would
    # silently drop coincident range that boolean2d's shared-edge merge
    # relies on, while passing every assertion above.
    assert np.allclose(overlap["u_range"], (0.0, 1.0), atol=0.0)
    assert np.allclose(overlap["v_range"], (0.0, 1.0), atol=0.0)


def test_exact_coincident_straight_curves_remain_an_overlap():
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    result = bez_ccx(line, line.copy(), atol=ATOL, rational=False)

    assert result["isolated"] == []
    assert result["budget_exhausted"] is False
    assert len(result["overlaps"]) == 1
    overlap = result["overlaps"][0]
    assert np.allclose(overlap["u_range"], (0.0, 1.0), atol=0.0)
    assert np.allclose(overlap["v_range"], (0.0, 1.0), atol=0.0)


def test_exact_curved_affine_parameter_subcurve_remains_an_overlap():
    whole = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    part = _subcurve(whole, 0.25, 0.75)

    result = bez_ccx(whole, part, atol=ATOL, rational=False)

    assert result["isolated"] == []
    assert result["budget_exhausted"] is False
    assert len(result["overlaps"]) == 1
    overlap = result["overlaps"][0]
    assert np.allclose(overlap["u_range"], (0.25, 0.75), atol=1e-12)
    assert np.allclose(overlap["v_range"], (0.0, 1.0), atol=1e-12)


@pytest.mark.parametrize("rational", [False, True])
def test_translated_sub_tolerance_parallel_lines_never_certify_exact(rational):
    """Neither strict roots nor overlap identity may use world magnitude.

    The exact-affine identity must keep refusing the 5e-4 gap at a 2e10
    origin (no world-magnitude envelope); the pair then promotes through
    the L47 tolerance tier with the geometric gap as its residual, exactly
    as it does at the origin — translation changes nothing.
    """
    origin = 2.0e10
    gap = 5.0e-4
    first_xyz = np.array([
        [origin, 0.0, 0.0],
        [origin, 1.0, 0.0],
    ])
    second_xyz = first_xyz.copy()
    second_xyz[:, 0] += gap
    if rational:
        weights = np.array([1.0, 2.0])
        first = _homogeneous(first_xyz, weights, 1.0e-30)
        second = _homogeneous(second_xyz, weights, 1.0e30)
    else:
        first, second = first_xyz, second_xyz

    assert not _overlap_mapping_is_identity(
        first, second, (0.0, 1.0), (0.0, 1.0), rational)
    result = bez_ccx(first, second, atol=ATOL, rational=rational)

    assert result["isolated"] == []
    assert result["budget_exhausted"] is False
    assert result["boundary_topology_complete"] is True
    assert len(result["overlaps"]) == 1
    overlap = result["overlaps"][0]
    assert overlap["certification"] == "tolerance"
    # Inversion at a 2e10 origin costs a few float64 ulps of the gap, not
    # more (poly measured 4.997e-4, rational 5.035e-4).
    assert overlap["residual_max"] == pytest.approx(gap, rel=2e-2)


def test_large_translated_exact_lines_remain_an_overlap():
    origin = 2.0e10
    line = np.array([
        [origin, 0.0, 0.0],
        [origin, 1.0, 0.0],
    ])

    result = bez_ccx(line, line.copy(), atol=ATOL, rational=False)

    assert result["isolated"] == []
    assert result["budget_exhausted"] is False
    assert len(result["overlaps"]) == 1


def test_large_translated_exact_crossing_remains_an_isolated_root():
    origin = 1.0e9
    first = np.array([
        [origin, 0.5, 0.0],
        [origin + 1.0, 0.5, 0.0],
    ])
    second = np.array([
        [origin + 0.5, 0.0, 0.0],
        [origin + 0.5, 1.0, 0.0],
    ])

    result = bez_ccx(first, second, atol=ATOL, rational=False)

    assert result["overlaps"] == []
    assert result["budget_exhausted"] is False
    assert len(result["isolated"]) == 1
    assert result["isolated"][0]["u"] == pytest.approx(0.5)
    assert result["isolated"][0]["v"] == pytest.approx(0.5)


def test_float_built_quadratic_subcurve_remains_an_overlap():
    """Restriction roundoff needs a source-operation identity floor."""
    whole = np.array([
        [0.251562918778363, -0.03577202263613945,
         -1941.6385335538441],
        [0.6475957171083957, 0.0150553739084055,
         -1941.6390899194341],
        [0.9438893741924533, 0.00627637665906201,
         -1941.6256997294456],
    ])
    lo = 0.11162826139544185
    hi = 0.7826689880851946
    part = _subcurve(whole, lo, hi)

    assert _overlap_mapping_is_identity(
        whole, part, (lo, hi), (0.0, 1.0), rational=False)


def test_tolerant_non_affine_overlap_candidate_returns_typed_partial():
    # L47: an overlap-class candidate the tolerance certificate cannot
    # promote exhausts its bounded fallback and ships the TYPED span —
    # never a bare budget flag with a complete-looking topology claim.
    first = np.array([
        [-19.77608536, 23.10065701, 0.0],
        [-14.86834768, 28.69713066, 0.0],
        [-5.85685250, 25.12677787, 0.0],
        [-12.62581769, 15.26478654, 0.0],
    ])
    second = np.array([
        [-22.03153620, 18.75969713, 0.0],
        [-19.42270945, 28.25028670, 0.0],
        [-8.46791623, 27.56878356, 0.0],
        [-10.43007782, 19.78973126, 0.0],
    ])

    result = bez_ccx(first, second, atol=ATOL, rational=False)

    assert result["overlaps"] == []
    assert result["budget_exhausted"] is True
    assert result["boundary_topology_complete"] is False
    assert result["cells_processed"] < 5_000
    span = result["uncertified_overlap_span"]
    assert span[0] == pytest.approx(0.0, abs=1e-9)
    assert span[1] == pytest.approx(0.8276, abs=5e-3)


# ---------------------------------------------------------------------------
# Item 1 (2026-07-26): the exactness certificates must not decay with world
# position.
#
# `_overlap_mapping_is_identity` and its csx twin took their source scales
# from the ALREADY-CENTERED nets, so the envelope measured what survived the
# common-origin cancellation rather than what was consumed by it.  A curve
# translated away from the origin loses precision in proportion to its world
# position while those scales stay at the model's own extent, so the bound
# stopped covering the noise.  Measured on the calibrated fixture above:
# certified at its native position, refused 139/200 at |T|=1 and 200/200 at
# |T|=1e6 -- and end-to-end a genuine exact overlap degraded to 'tolerance'
# and then to no overlap at all.
# ---------------------------------------------------------------------------

_WHOLE = np.array([
    [0.251562918778363, -0.03577202263613945, -1941.6385335538441],
    [0.6475957171083957, 0.0150553739084055, -1941.6390899194341],
    [0.9438893741924533, 0.00627637665906201, -1941.6256997294456],
])
_LO = 0.11162826139544185
_HI = 0.7826689880851946


@pytest.mark.parametrize("mag", [0.0, 1.0, 1e2, 1e3, 1e4, 1e6, 1e9])
def test_float_built_subcurve_certifies_at_every_world_position(mag):
    """A true overlap is a true overlap wherever the model sits."""
    rng = np.random.default_rng(3)
    for _ in range(25):
        t = rng.normal(size=3) * mag
        whole = _WHOLE + t
        part = _subcurve(whole, _LO, _HI)
        assert _overlap_mapping_is_identity(
            whole, part, (_LO, _HI), (0.0, 1.0), rational=False), mag


@pytest.mark.parametrize("mag", [0.0, 1.0, 1e3, 1e6, 1e9])
def test_perturbed_subcurve_is_still_refused_everywhere(mag):
    """Anti-loosening guard for the added centering term.

    The envelope grows with the precision a translation destroys; it must
    not grow enough to swallow a real, representable difference.
    """
    rng = np.random.default_rng(4)
    for _ in range(25):
        t = rng.normal(size=3) * mag
        whole = _WHOLE + t
        part = _subcurve(whole, _LO, _HI)
        part[1, 1] += 1e-6 * max(1.0, abs(mag))
        assert not _overlap_mapping_is_identity(
            whole, part, (_LO, _HI), (0.0, 1.0), rational=False), mag


@pytest.mark.parametrize("mag", [0.0, 1e3, 1e6])
def test_exact_overlap_survives_end_to_end(mag):
    """The public consequence: bez_ccx must still SEE the overlap.

    Before the fix this returned 'tolerance' at the model's own origin and
    no overlap at all for 36/40 samples at |T|=1e6.
    """
    rng = np.random.default_rng(9)
    for _ in range(10):
        t = rng.normal(size=3) * mag
        whole = _WHOLE + t
        part = _subcurve(whole, _LO, _HI)
        r = bez_ccx(whole, part, atol=1e-3, rational=False)
        assert r["overlaps"], (mag, "overlap vanished")
        assert r["overlaps"][0]["certification"] == "exact", (
            mag, r["overlaps"][0]["certification"])
