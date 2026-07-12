"""L59 — CSX theorem-first overlap certification (USER DECISION 2026-07-12).

Two rational/polynomial arcs coinciding on any open sub-arc lie on the same
algebraic curve, so a maximal overlap can only terminate at a DOMAIN
boundary of one operand: certify domain-pinned span ends + interior
witnesses numerically and let the theorem carry the interior. Tolerance-
coincidence IS coincidence (endpoint-pinned + witnessed + no crossing
flips => 'tolerance' overlap; 'exact' when the algebraic identity holds).

Fixture 1 is REAL USER DATA (overlap_nurbs_intersection_3_new.py, curve2
span 2 x surface patch 17): the curve lies on the extruded patch for
t in [0.78, 1.0] — pinned at the curve's domain end and the patch
boundary. At the pre-L59 HEAD this pair ground 12,234 cells and returned
EMPTY-COMPLETE (lost geometry, no partial flag). On tiny the full model
certified in 2.7s via the (unsound) valley rule this tier replaces.
"""
import numpy as np
import pytest

from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx

C_REAL = np.array([
    [9.483507654421916e+01, 8.704233407815154e+01, -1.462358240532153e+00],
    [9.565026587000000e+01, 8.877912870999999e+01, -7.268909099999999e-01],
    [9.611300253000000e+01, 9.066371219000000e+01, 2.942412000000000e-02],
    [9.619901324878532e+01, 9.255100227104691e+01, 8.065860404502375e-01]])

S_REAL = np.array([
    [[96.08645179069646, 91.2936352248758, 41.41100531879138],
     [96.08645179069646, 91.2936352248758, 0.10498601801243446]],
    [[96.19819878977371, 92.12868261443225, 41.41100531879138],
     [96.19819878977371, 92.12868261443225, 0.10498601801243446]],
    [[96.23638809593152, 92.97063253198961, 41.41100531879138],
     [96.23638809593152, 92.97063253198961, 0.10498601801243446]],
    [[96.20144519121484, 93.80660802431686, 41.41100531879138],
     [96.20144519121484, 93.80660802431686, 0.10498601801243446]]])


def test_real_data_partial_overlap_span_is_certified():
    r = bez_csx(C_REAL, S_REAL, atol=1e-3, rational=False)

    assert r["budget_exhausted"] is False
    assert r["boundary_topology_complete"] is True
    assert len(r["overlaps"]) >= 1, (
        "the coincident span t in [0.78, 1.0] must ship as an overlap "
        f"(got isolated={len(r['isolated'])}, overlaps=0)")
    spans = [o["t_range"] for o in r["overlaps"]]
    covering = [s for s in spans if s[0] <= 0.80 and s[1] >= 0.98]
    assert covering, spans
    for o in r["overlaps"]:
        assert o["certification"] in ("exact", "tolerance")
    # the arming evidence is endpoint membership — certified spans must
    # BYPASS the subdivision grind (pre-L59: 12,234 cells for nothing)
    assert r["cells_processed"] <= 4000, r["cells_processed"]


def test_planar_quad_edge_overlap_is_certified():
    # L57 (absorbed): the u=0 edge isocurve of the lifted bilinear lies in
    # the plane of a NON-PARALLELOGRAM planar quad (twist != 0) — the
    # exact-affine identity rightly refuses, and pre-L59 the case ground
    # 4k fallback cells into ~250 lattice pseudo-roots with a typed span.
    # Whole-domain coincidence: both curve ends on the surface.
    quad = np.array([[[28.73565361, -57.3828431, 0.], [41.34259183, -50.11361956, 0.]],
                     [[41.34259183, -75.32749601, 0.], [53.84239759, -62.72055778, 0.]]])
    iso_u0 = np.array([[35.58090097, -65.90568734, 0.],
                       [38.10773149, -56.47542745, 0.]])

    r = bez_csx(iso_u0, quad, atol=1e-3, rational=False)

    assert r["budget_exhausted"] is False
    assert r["boundary_topology_complete"] is True
    assert len(r["overlaps"]) == 1, (r["overlaps"], len(r["isolated"]))
    o = r["overlaps"][0]
    assert o["certification"] in ("exact", "tolerance")
    assert o["t_range"][0] == pytest.approx(0.0, abs=1e-6)
    assert o["t_range"][1] == pytest.approx(1.0, abs=1e-6)
    assert r["cells_processed"] <= 500, r["cells_processed"]


def _fixture_A():
    val1 = np.array([[[28.73565361, -57.3828431, 0.], [41.34259183, -50.11361956, 0.]],
                     [[41.34259183, -75.32749601, 0.], [53.84239759, -62.72055778, 0.]]])
    val2 = np.array([[[35.58090097, -65.90568734, 0.], [38.10773149, -56.47542745, 0.]],
                     [[45.01116086, -68.43251786, 2.89525681], [47.53799138, -59.00225797, 0.]]])
    return val1, val2


def test_planar_quad_L_junction_ships_both_full_branches():
    """L57 acceptance (user fixtures A vs C): the L of val2's u=0 and v=1
    edges meeting at the corner tangent point must ship as two FULL
    branches (the parallelogram variant C always did). Pre-L59 the u=0
    branch was truncated to 37% WITH an honesty flag; the tier's first
    integration certified the boundary overlap but the straight-chord
    conversion dropped it, leaving the truncation FALSELY complete."""
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
    from mmcore.numeric.intersection._bezier_common import eval_surface

    val1, val2 = _fixture_A()
    r = bez_ssx(val1, val2, 1e-3, rational=False)

    assert len(r["branches"]) == 2
    tps = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(tps) == 1
    corner = tps[0].xyz

    lengths = []
    reaches_corner = 0
    for b in r["branches"]:
        xyz = np.asarray(b.curve[1])
        seg = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        lengths.append(seg)
        d_end = min(np.linalg.norm(xyz[0] - corner),
                    np.linalg.norm(xyz[-1] - corner))
        if d_end <= 5e-3:
            reaches_corner += 1
    # both edge branches are ~9.76 long and BOTH terminate at the corner
    assert all(l >= 9.0 for l in sorted(lengths)), lengths
    assert reaches_corner == 2, (lengths, reaches_corner)
    # and the completeness claim must be TRUE, not vacuous
    assert r["complete"] is True, r["status"]["reasons"]
