"""Tests for the NURBS SSX adapter over bez_ssx v5 (_nssx5.py).

Spec: docs/superpowers/specs/2026-07-19-nurbs-ssx5-design.md
"""
import pathlib
import pickle

import numpy as np
import pytest

from mmcore.geom._nurbs_eval import (
    NURBSSurfaceTuple,
    evaluate_nurbs_curve,
    evaluate_nurbs_surface,
)
from mmcore.numeric.bern import de_casteljau_split_nd

FIXTURE_DIR = pathlib.Path(__file__).parent.parent / "examples" / "ssx"


# ---------------------------------------------------------------------------
# Self-contained surface builders (exact constructions, no fixtures needed)
# ---------------------------------------------------------------------------

def bezier_surface(cp):
    """Single-span (Bezier) non-rational NURBSSurfaceTuple from a CP grid."""
    cp = np.asarray(cp, dtype=np.float64)
    nu, nv = cp.shape[0], cp.shape[1]
    return NURBSSurfaceTuple(
        order_u=nu, order_v=nv,
        knot_u=np.array([0.0] * nu + [1.0] * nu),
        knot_v=np.array([0.0] * nv + [1.0] * nv),
        control_points=cp, weights=np.ones((nu, nv)))


def insert_midknot(surf, axis=0):
    """Exact re-representation of a SINGLE-SPAN surface with a
    full-multiplicity interior knot at 0.5 (de Casteljau split + C0 join).
    Only valid on surfaces produced by `bezier_surface`."""
    cp = surf.control_points
    left, right = de_casteljau_split_nd(cp, axis=axis, t=0.5)
    if axis == 0:
        joined = np.concatenate([left, right[1:]], axis=0)
        order = surf.order_u
        knot = np.array([0.0] * order + [0.5] * (order - 1) + [1.0] * order)
        return NURBSSurfaceTuple(
            order_u=order, order_v=surf.order_v,
            knot_u=knot, knot_v=surf.knot_v,
            control_points=joined, weights=np.ones(joined.shape[:2]))
    joined = np.concatenate([left, right[:, 1:]], axis=1)
    order = surf.order_v
    knot = np.array([0.0] * order + [0.5] * (order - 1) + [1.0] * order)
    return NURBSSurfaceTuple(
        order_u=surf.order_u, order_v=order,
        knot_u=surf.knot_u, knot_v=knot,
        control_points=joined, weights=np.ones(joined.shape[:2]))


def plane_z0():
    """z = 0 over (x, y) in [-1, 1]^2; s -> x, t -> y."""
    return bezier_surface([[[-1, -1, 0], [-1, 1, 0]],
                           [[1, -1, 0], [1, 1, 0]]])


def plane_tilted():
    """z = x over (x, y) in [-1, 1]^2; crosses plane_z0 along x = 0."""
    return bezier_surface([[[-1, -1, -1], [-1, 1, -1]],
                           [[1, -1, 1], [1, 1, 1]]])


def cylinder():
    """Full circular cylinder, radius 1, axis z, z in [0, 1].
    Standard 9-point rational circle in u (C0-closed seam at u=0/1),
    linear v.  rational=True path."""
    r2 = np.sqrt(2.0) / 2.0
    circle_xy = np.array(
        [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0],
         [-1, -1], [0, -1], [1, -1], [1, 0]], dtype=np.float64)
    w_u = np.array([1, r2, 1, r2, 1, r2, 1, r2, 1], dtype=np.float64)
    cp = np.zeros((9, 2, 3))
    cp[:, 0, :2] = circle_xy
    cp[:, 1, :2] = circle_xy
    cp[:, 1, 2] = 1.0
    return NURBSSurfaceTuple(
        order_u=3, order_v=2,
        knot_u=np.array([0, 0, 0, .25, .25, .5, .5, .75, .75, 1, 1, 1],
                        dtype=np.float64),
        knot_v=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=cp,
        weights=np.stack([w_u, w_u], axis=1))


def big_plane(z):
    """z = const over (x, y) in [-2, 2]^2 (single span)."""
    return bezier_surface([[[-2, -2, z], [-2, 2, z]],
                           [[2, -2, z], [2, 2, z]]])


def paraboloid():
    """z = (2s-1)^2 + (2t-1)^2, x = 2s-1, y = 2t-1 (exact biquadratic
    Bezier).  Touches z=0 tangentially at (s,t) = (0.5, 0.5)."""
    lin = [-1.0, 0.0, 1.0]      # Bezier deg-2 coefficients of 2x-1
    quad = [1.0, -1.0, 1.0]     # Bezier deg-2 coefficients of (2x-1)^2
    cp = [[[lin[i], lin[j], quad[i] + quad[j]] for j in range(3)]
          for i in range(3)]
    return bezier_surface(cp)


# ---------------------------------------------------------------------------
# Task 1: foundations
# ---------------------------------------------------------------------------

def test_reject_unknown_kwargs():
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    with pytest.raises(TypeError, match="atol"):
        nurbs_ssx(plane_z0(), plane_tilted(), tol=1e-3)


def test_rejects_non_surface_input():
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    with pytest.raises(TypeError):
        nurbs_ssx(np.zeros((2, 2, 3)), plane_z0())


def test_domain_ctx_and_wrap_diff():
    from mmcore.numeric.intersection.ssx._nssx5 import _domain_ctx, _axis_diff
    ctx = _domain_ctx(cylinder(), big_plane(0.5), atol=1e-3)
    # cylinder u-axis (stuv axis 0) is C0-closed; its v and the plane are not
    assert ctx.closed[0] is True
    assert ctx.closed[1] is False
    assert ctx.closed[2] is False and ctx.closed[3] is False
    # wrap-aware difference on the closed axis
    assert _axis_diff(0.02, 0.98, 0, ctx) == pytest.approx(0.04)
    # plain difference on an open axis
    assert _axis_diff(0.02, 0.98, 1, ctx) == pytest.approx(0.96)


def test_match_and_dup_predicates():
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _domain_ctx, _match_stuv, _dup_stuv)
    atol = 1e-3
    ctx = _domain_ctx(plane_z0(), plane_tilted(), atol=atol)
    p = np.array([0.5, 0.5, 0.5, 0.5])
    x = np.zeros(3)
    # identical points match under both predicates
    assert _match_stuv(p, p, x, x, ctx, atol)
    assert _dup_stuv(p, p, x, x, ctx, atol)
    # xyz guard: parametric proximity alone must NOT match (ladder rule)
    far = np.array([10 * atol, 0, 0])
    assert not _match_stuv(p, p, x, far, ctx, atol)
    assert not _dup_stuv(p, p, x, x + np.array([2 * atol, 0, 0]), ctx, atol)
    # parametric guard: xyz proximity alone must not match either
    q = p + np.array([50.0 * ctx.ptol[0], 0, 0, 0])
    assert not _match_stuv(p, q, x, x, ctx, atol)


def test_wrap_dup_vs_joint_plain_dup():
    """Periodic vertex-pair contract: opposite-seam preimages are the SAME
    point for destructive dedup (wrap-aware) but DISTINCT at stitch joints
    (no wrap), so both seam vertices survive concatenation."""
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _domain_ctx, _dup_stuv, _joint_plain_dup)
    atol = 1e-3
    ctx = _domain_ctx(cylinder(), big_plane(0.5), atol=atol)
    eps = 0.1 * float(ctx.ptol[0])
    p = np.array([0.0 + eps, 0.5, 0.5, 0.5])
    q = np.array([1.0 - eps, 0.5, 0.5, 0.5])
    x = np.array([1.0, 0.0, 0.5])
    assert _dup_stuv(p, q, x, x, ctx, atol)
    assert not _joint_plain_dup(p, q, x, x, ctx, atol)


def test_remap4():
    from mmcore.numeric.intersection.ssx._nssx5 import _remap4
    rect = (0.0, 0.5, 0.5, 1.0, 0.25, 0.75, 0.0, 1.0)
    out = _remap4(np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
                  rect)
    assert np.allclose(out[0], [0.0, 0.5, 0.25, 0.0])
    assert np.allclose(out[1], [0.5, 1.0, 0.75, 1.0])
    single = _remap4(np.array([0.5, 0.5, 0.5, 0.5]), rect)
    assert np.allclose(single, [0.25, 0.75, 0.5, 0.5])


def test_aggregate_status_consume_and_fields():
    from mmcore.numeric.intersection.ssx._nssx5 import _make_aggregate
    agg = _make_aggregate({}, n_candidates=2)
    assert agg.max_cells == 2 * 250_000
    assert agg.max_csx_calls == 2 * 10_000
    assert agg.max_output_items == 2 * 1_024
    fake = {'status': {'reasons': ['tangential_zone'],
                       'work': {'cells_processed': 100, 'csx_calls': 3,
                                'output_items': 5,
                                'cell_counts': {'csx': 40, 'cells': 60}}}}
    agg.consume(fake)
    agg.consume(fake)
    fields = agg.result_fields()
    assert fields['complete'] is False
    assert fields['status']['reasons'] == ['tangential_zone']
    work = fields['status']['work']
    assert work['cells_processed'] == 200
    assert work['csx_calls'] == 6
    assert work['output_items'] == 10
    assert work['cell_counts'] == {'csx': 80, 'cells': 120}
    assert work['max_cells'] == 2 * 250_000
    # explicit values are absolute aggregates
    agg2 = _make_aggregate({'max_cells': 77, 'max_csx_calls': 5,
                            'max_output_items': 9}, n_candidates=100)
    assert (agg2.max_cells, agg2.max_csx_calls, agg2.max_output_items) \
        == (77, 5, 9)


def test_aggregate_postprocess_cap_reason():
    from mmcore.numeric.intersection.ssx._nssx5 import _make_aggregate
    from mmcore.numeric._work_budget import REASON_POSTPROCESS_CAP
    agg = _make_aggregate({'max_postprocess_work': 3}, n_candidates=1)
    assert agg.charge_postprocess(2)
    assert not agg.charge_postprocess(5)
    fields = agg.result_fields()
    assert fields['complete'] is False
    assert REASON_POSTPROCESS_CAP in fields['status']['reasons']


# ---------------------------------------------------------------------------
# Task 2: pipeline core
# ---------------------------------------------------------------------------

def _branch_paths(res):
    return [(np.asarray(b.curve[0], dtype=float),
             np.asarray(b.curve[1], dtype=float)) for b in res['branches']]


def test_single_patch_parity_with_bez_ssx():
    """One Bezier pair, identity remap: adapter == direct bez_ssx."""
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
    s1, s2 = plane_z0(), plane_tilted()
    res = nurbs_ssx(s1, s2, atol=1e-3)
    ref = bez_ssx(np.asarray(s1.control_points, dtype=float),
                  np.asarray(s2.control_points, dtype=float),
                  atol=1e-3, rational=False)
    assert res['complete'] is True and ref['complete'] is True
    assert len(res['branches']) == len(ref['branches']) == 1
    assert len(res['points']) == len(ref['points'])
    got_s, got_x = _branch_paths(res)[0]
    ref_s = np.asarray(ref['branches'][0].curve[0], dtype=float)
    ref_x = np.asarray(ref['branches'][0].curve[1], dtype=float)
    # identical vertex data up to orientation
    if not np.allclose(got_x[0], ref_x[0]):
        ref_s, ref_x = ref_s[::-1], ref_x[::-1]
    assert np.allclose(got_s, ref_s, atol=1e-9)
    assert np.allclose(got_x, ref_x, atol=1e-9)


def test_empty_result_when_disjoint():
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    far = bezier_surface([[[-1, -1, 5], [-1, 1, 5]],
                          [[1, -1, 5], [1, 1, 5]]])
    res = nurbs_ssx(plane_z0(), far, atol=1e-3)
    assert res['complete'] is True
    assert res['branches'] == [] and res['points'] == []
    assert res['singularities'] == [] and res['overlap_regions'] == []
    assert res['unresolved_regions'] == []
    assert res['status']['reasons'] == []


def test_params_are_global_after_remap():
    """Multi-span surf1 (seam at t=0.5): every branch vertex satisfies
    S1(s,t) == S2(u,v) == xyz at the GLOBAL parameters."""
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    s1 = insert_midknot(plane_z0(), axis=1)
    s2 = plane_tilted()
    res = nurbs_ssx(s1, s2, atol=1e-3)
    assert res['complete'] is True
    assert len(res['branches']) == 1
    for stuv, xyz in _branch_paths(res):
        for k in range(0, len(stuv), max(1, len(stuv) // 8)):
            s, t, u, v = stuv[k]
            p1 = evaluate_nurbs_surface(s1, s, t, d_order=0)['S']
            p2 = evaluate_nurbs_surface(s2, u, v, d_order=0)['S']
            assert np.linalg.norm(np.asarray(p1) - xyz[k]) <= 2e-3
            assert np.linalg.norm(np.asarray(p2) - xyz[k]) <= 2e-3


def test_starvation_is_soft_and_typed():
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    from mmcore.numeric._work_budget import REASON_WORK_BUDGET
    s1 = insert_midknot(plane_z0(), axis=1)
    s2 = insert_midknot(plane_tilted(), axis=1)
    res = nurbs_ssx(s1, s2, atol=1e-3, max_cells=1)
    assert res['complete'] is False
    assert REASON_WORK_BUDGET in res['status']['reasons']
    assert len(res['unresolved_regions']) >= 1
    for entry in res['unresolved_regions']:
        assert 'stuv_min' in entry and 'stuv_max' in entry
        assert 'reason' in entry


def test_aggregate_work_counters_populated():
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    res = nurbs_ssx(plane_z0(), plane_tilted(), atol=1e-3)
    work = res['status']['work']
    assert work['cells_processed'] > 0
    assert work['max_cells'] == 250_000  # 1 candidate pair
    assert work['postprocess_work'] >= 0


# ---------------------------------------------------------------------------
# Task 3: branch assembly
# ---------------------------------------------------------------------------

def test_multispan_crossing_stitches_to_one_branch():
    """Seams on BOTH surfaces cut the intersection line; the wrapper must
    return ONE continuous branch spanning the full line."""
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    s1 = insert_midknot(plane_z0(), axis=1)      # seam crossing the line
    s2 = insert_midknot(plane_tilted(), axis=1)
    res = nurbs_ssx(s1, s2, atol=1e-3)
    assert res['complete'] is True
    assert len(res['branches']) == 1
    stuv, xyz = _branch_paths(res)[0]
    # continuity: no vertex gap wildly above the largest in-fragment step
    gaps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    assert gaps.max() <= 10 * np.median(gaps) + 1e-12
    # full extent: line x=0,z=0 runs y in [-1,1] -> t from 0 to 1
    ts = stuv[:, 1]
    assert ts.min() <= 1e-6 and ts.max() >= 1 - 1e-6
    # geometry: on both planes
    assert np.abs(xyz[:, 0]).max() <= 2e-3
    assert np.abs(xyz[:, 2]).max() <= 2e-3


def test_knot_line_curve_reported_once():
    """SSI curve exactly ON a decomposition seam is traced by both
    adjacent pairs; containment dedup must keep exactly one branch."""
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    s1 = insert_midknot(plane_z0(), axis=0)      # seam s=0.5 IS the curve
    s2 = plane_tilted()
    res = nurbs_ssx(s1, s2, atol=1e-3)
    assert res['complete'] is True
    assert len(res['branches']) == 1
    stuv, xyz = _branch_paths(res)[0]
    assert np.abs(xyz[:, 0]).max() <= 2e-3    # x == 0
    assert np.abs(xyz[:, 2]).max() <= 2e-3    # z == 0
    # spec delta 3: seam-coincident curve carries kind='overlap'
    assert res['branches'][0].kind == 'overlap'


def test_cylinder_plane_closed_loop_with_seam_pair():
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    res = nurbs_ssx(cylinder(), big_plane(0.5), atol=1e-3)
    assert res['complete'] is True
    assert len(res['branches']) == 1
    br = res['branches'][0]
    assert br.closed is True
    stuv, xyz = _branch_paths(res)[0]
    # circle of radius 1 at z=0.5
    r = np.linalg.norm(xyz[:, :2], axis=1)
    assert np.abs(r - 1.0).max() <= 2e-3
    assert np.abs(xyz[:, 2] - 0.5).max() <= 2e-3
    # closed polyline convention: first == last vertex in xyz
    assert np.linalg.norm(xyz[0] - xyz[-1]) <= 2e-3
    # periodic vertex-pair contract: some consecutive pair jumps the u-seam
    # (|ds| ~ full span) while xyz stays put
    ds = np.abs(np.diff(stuv[:, 0]))
    gaps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    seam_jumps = (ds > 0.9) & (gaps <= 2e-3)
    assert seam_jumps.any()


def test_junction_cluster_is_not_chained_through():
    """Unit test on the assembler: 4 fragment ends meeting at one point
    (X-junction) must stay 4 separate branches."""
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _Frag, _assemble_branches, _domain_ctx, _make_aggregate)
    ctx = _domain_ctx(plane_z0(), plane_tilted(), atol=1e-3)
    agg = _make_aggregate({}, 1)
    center_s = np.array([0.5, 0.5, 0.5, 0.5])
    center_x = np.zeros(3)
    frags = []
    for d in (np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]),
              np.array([0, 1.0, 0]), np.array([0, -1.0, 0])):
        stuv = np.vstack([center_s + np.concatenate([0.3 * d[:2], [0, 0]]),
                          center_s])
        xyz = np.vstack([center_x + 0.3 * d, center_x])
        frags.append(_Frag(stuv=stuv, xyz=xyz, kind='transversal',
                           overlap=False))
    out = _assemble_branches(frags, ctx, atol=1e-3, agg=agg)
    assert len(out) == 4
    assert all(not b.closed for b in out)


def test_kind_conflict_blocks_stitching():
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _Frag, _assemble_branches, _domain_ctx, _make_aggregate)
    ctx = _domain_ctx(plane_z0(), plane_tilted(), atol=1e-3)
    agg = _make_aggregate({}, 1)
    a = _Frag(stuv=np.array([[0.0, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]),
              xyz=np.array([[-0.5, 0, 0], [0, 0, 0]]),
              kind='transversal', overlap=False)
    b = _Frag(stuv=np.array([[0.5, 0.5, 0.5, 0.5], [1.0, 0.5, 0.5, 0.5]]),
              xyz=np.array([[0, 0, 0], [0.5, 0, 0]]),
              kind='tangential', overlap=False)
    out = _assemble_branches([a, b], ctx, atol=1e-3, agg=agg)
    assert len(out) == 2


def test_chain_orientation_all_four_combinations():
    """Stitching must produce one contiguous polyline regardless of how
    the two fragments' endpoint orientations pair up."""
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _Frag, _assemble_branches, _domain_ctx, _make_aggregate)
    ctx = _domain_ctx(plane_z0(), plane_tilted(), atol=1e-3)
    A_s = np.array([[0.2, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
    A_x = np.array([[-0.6, 0.0, 0.0], [0.0, 0.0, 0.0]])
    B_s = np.array([[0.5, 0.5, 0.5, 0.5], [0.8, 0.5, 0.5, 0.5]])
    B_x = np.array([[0.0, 0.0, 0.0], [0.0, 0.6, 0.0]])
    for flip_a in (False, True):
        for flip_b in (False, True):
            agg = _make_aggregate({}, 1)
            fa = _Frag(stuv=A_s[::-1].copy() if flip_a else A_s.copy(),
                       xyz=A_x[::-1].copy() if flip_a else A_x.copy(),
                       kind='transversal', overlap=False)
            fb = _Frag(stuv=B_s[::-1].copy() if flip_b else B_s.copy(),
                       xyz=B_x[::-1].copy() if flip_b else B_x.copy(),
                       kind='transversal', overlap=False)
            out = _assemble_branches([fa, fb], ctx, atol=1e-3, agg=agg)
            assert len(out) == 1
            xyz = np.asarray(out[0].curve[1], dtype=float)
            gaps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
            assert gaps.max() <= 0.61  # contiguous: no jump across the joint


def test_containment_dedup_identical_twins_deterministic():
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _Frag, _containment_dedup, _make_aggregate)
    path_s = np.array([[0.1, 0.5, 0.5, 0.5], [0.9, 0.5, 0.5, 0.5]])
    path_x = np.array([[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]])
    twins = [
        _Frag(stuv=path_s.copy(), xyz=path_x.copy(),
              kind='overlap', overlap=True),
        _Frag(stuv=path_s.copy(), xyz=path_x.copy(),
              kind='overlap', overlap=True),
    ]
    for order in (twins, twins[::-1]):
        agg = _make_aggregate({}, 1)
        kept = _containment_dedup(list(order), 1e-3, agg)
        assert len(kept) == 1


# ---------------------------------------------------------------------------
# Task 4: points & singularities
# ---------------------------------------------------------------------------

def test_seam_straddling_tangency_dedups_to_one():
    """Paraboloid touching z=0 at (0.5, 0.5); seam inserted THROUGH the
    tangency: both adjacent pairs certify it; wrapper reports ONE
    tangent_point."""
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    s1 = insert_midknot(paraboloid(), axis=0)
    s2 = big_plane(0.0)
    res = nurbs_ssx(s1, s2, atol=1e-3)
    tps = [s for s in res['singularities'] if s.kind == 'tangent_point']
    assert len(tps) == 1
    tp = tps[0]
    assert np.linalg.norm(np.asarray(tp.xyz) - np.zeros(3)) <= 5e-3
    assert abs(tp.stuv[0] - 0.5) <= 0.05 and abs(tp.stuv[1] - 0.5) <= 0.05
    # The certified tangent_point ships WITH typed structural caveats:
    # bez_ssx reports complete=False + unresolved_multiplicity (split
    # patches; the unsplit single-patch case additionally reports
    # unresolved_tangential_zone) for isolated C2 tangencies — verified
    # against a direct bez_ssx call on the unsplit surfaces. The adapter
    # must surface that honestly (spec: AND of completes, union of
    # reasons), never claim completeness the engine didn't certify.
    assert res['complete'] is False
    assert set(res['status']['reasons']) <= {
        'unresolved_multiplicity', 'unresolved_tangential_zone'}
    assert len(res['status']['reasons']) >= 1


def test_point_on_neighbor_pair_branch_is_filtered():
    """Unit test: a point lying on another pair's branch polyline is
    dropped by the global on-branch filter (4*atol)."""
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _assemble_points, _domain_ctx, _make_aggregate)
    from mmcore.numeric.intersection.ssx._ssx_substrate import SSXPoint, SSXBranch
    ctx = _domain_ctx(plane_z0(), plane_tilted(), atol=1e-3)
    agg = _make_aggregate({}, 1)
    xyz_path = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=float)
    stuv_path = np.array([[.5, 0, .5, 0], [.5, .5, .5, .5],
                          [.5, 1, .5, 1]], dtype=float)
    branch = SSXBranch(curve=(stuv_path, xyz_path))
    on = SSXPoint(stuv=np.array([.5, .25, .5, .25]),
                  xyz=np.array([0.0, -0.5, 0.002]))     # 2e-3 < 4*atol
    off = SSXPoint(stuv=np.array([.9, .9, .9, .9]),
                   xyz=np.array([0.8, 0.8, 0.8]))
    out = _assemble_points([on, off], [branch], ctx, 1e-3, agg)
    assert len(out) == 1
    assert np.allclose(out[0].xyz, off.xyz)


def test_duplicate_points_dedup_wrap_aware():
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _assemble_points, _domain_ctx, _make_aggregate)
    from mmcore.numeric.intersection.ssx._ssx_substrate import SSXPoint
    ctx = _domain_ctx(cylinder(), big_plane(0.5), atol=1e-3)
    agg = _make_aggregate({}, 1)
    x = np.array([1.0, 0.0, 0.5])
    eps_s = 0.1 * float(ctx.ptol[0])
    a = SSXPoint(stuv=np.array([0.0 + eps_s, .5, .5, .5]), xyz=x)
    b = SSXPoint(stuv=np.array([1.0 - eps_s, .5, .5, .5]), xyz=x)  # wrap dup
    out = _assemble_points([a, b], [], ctx, 1e-3, agg)
    assert len(out) == 1


def test_branch_links_recomputed_against_final_branches():
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _assemble_singularities, _domain_ctx, _make_aggregate)
    from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch
    from mmcore.numeric.intersection.ssx._bez_ssx5 import SSXSingularity
    ctx = _domain_ctx(plane_z0(), plane_tilted(), atol=1e-3)
    agg = _make_aggregate({}, 1)
    xyz_path = np.array([[0, -1, 0], [0, -0.25, 0], [0, 0.5, 0],
                         [0, 1, 0]], dtype=float)
    stuv_path = np.array([[.5, 0, .5, 0], [.5, .375, .5, .375],
                          [.5, .75, .5, .75], [.5, 1, .5, 1]], dtype=float)
    br = SSXBranch(curve=(stuv_path, xyz_path))
    sing = SSXSingularity(
        kind='cusp', stuv=np.array([.5, .375, .5, .375]),
        xyz=np.array([0.0, -0.25, 0.0]), branch_links=[])
    out = _assemble_singularities([sing], [br], ctx, 1e-3, agg)
    assert len(out) == 1
    assert out[0].branch_links == [(0, 1)]   # nearest vertex of branch 0


def test_branch_link_anchors_mid_segment_cusp():
    """L12: a cusp ON the polyline but far from every vertex must still
    link (point-to-SEGMENT distance), anchored at the nearer endpoint of
    the nearest segment."""
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _recompute_branch_links, _make_aggregate)
    from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch
    agg = _make_aggregate({}, 1)
    xyz_path = np.array([[0, -1, 0], [0, 1, 0]], dtype=float)  # one segment
    stuv_path = np.array([[.5, 0, .5, 0], [.5, 1, .5, 1]], dtype=float)
    br = SSXBranch(curve=(stuv_path, xyz_path))
    target = np.array([0.0, 0.4, 0.0])   # on-segment, 0.6 from v1, 1.4 from v0
    links = _recompute_branch_links(target, [br], 1e-3, agg)
    assert links == [(0, 1)]


def test_cloud_dedup_uses_surf1_preimage():
    """cusp_curve clouds with identical (s,t) but different (u,v) samples
    evaluate to the same S1 points and must collapse (SSI samples satisfy
    S1(s,t)==S2(u,v), so the s1 preimage determines the 3D locus)."""
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _clouds_near_identical, _make_aggregate)
    agg = _make_aggregate({}, 1)
    s1 = plane_z0()
    a = np.array([[0.2, 0.3, 0.1, 0.9], [0.6, 0.7, 0.2, 0.8]])
    b = np.array([[0.2, 0.3, 0.9, 0.1], [0.6, 0.7, 0.8, 0.2]])
    assert _clouds_near_identical(a, b, s1, 1e-3, agg)
    c = np.array([[0.9, 0.9, 0.1, 0.9]])   # different (s,t): far on S1
    assert not _clouds_near_identical(a, c, s1, 1e-3, agg)


def test_self_intersection_mate_discriminates():
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _assemble_singularities, _domain_ctx, _make_aggregate)
    from mmcore.numeric.intersection.ssx._bez_ssx5 import SSXSingularity
    ctx = _domain_ctx(plane_z0(), plane_tilted(), atol=1e-3)
    xyz = np.zeros(3)
    stuv = np.array([.5, .5, .5, .5])

    def sing(mate):
        return SSXSingularity(kind='self_intersection', stuv=stuv.copy(),
                              xyz=xyz.copy(),
                              stuv_mate=np.asarray(mate, dtype=float),
                              branch_links=[])

    near = stuv + 0.5 * ctx.ptol
    far = stuv + np.array([50.0 * ctx.ptol[0], 0, 0, 0])
    agg = _make_aggregate({}, 1)
    merged = _assemble_singularities(
        [sing(stuv), sing(near)], [], ctx, 1e-3, agg)
    assert len(merged) == 1
    agg = _make_aggregate({}, 1)
    kept = _assemble_singularities(
        [sing(stuv), sing(far)], [], ctx, 1e-3, agg)
    assert len(kept) == 2


def test_points_kept_when_postprocess_starved():
    """Zero postprocess budget: the on-branch filter must OVER-include
    (keep the point) and record the typed reason — never silently drop."""
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _assemble_points, _domain_ctx, _make_aggregate)
    from mmcore.numeric._work_budget import REASON_POSTPROCESS_CAP
    from mmcore.numeric.intersection.ssx._ssx_substrate import SSXPoint, SSXBranch
    ctx = _domain_ctx(plane_z0(), plane_tilted(), atol=1e-3)
    agg = _make_aggregate({'max_postprocess_work': 0}, 1)
    xyz_path = np.array([[0, -1, 0], [0, 1, 0]], dtype=float)
    stuv_path = np.array([[.5, 0, .5, 0], [.5, 1, .5, 1]], dtype=float)
    br = SSXBranch(curve=(stuv_path, xyz_path))
    on = SSXPoint(stuv=np.array([.5, .5, .5, .5]),
                  xyz=np.array([0.0, 0.0, 0.0]))
    out = _assemble_points([on], [br], ctx, 1e-3, agg)
    assert len(out) == 1
    assert REASON_POSTPROCESS_CAP in agg.reasons


# ---------------------------------------------------------------------------
# Task 5: overlap-region unification
# ---------------------------------------------------------------------------

def _assert_unified_plane_twin(res, n_tiles_expected_dissolved_seams):
    """Common asserts for plane-twin unification: ONE region whose outer
    uv1 loop spans the full domain, valid boundary refs, no surviving
    branch pinned to an interior seam, and the honest typed reason from
    the off-diagonal seam-curve pairs (engine truth: adjacent coplanar
    patches overlap along their shared edge and bez_ssx types that as
    overlap_region_unsupported; the wrapper surfaces it — no silent
    completeness claims)."""
    assert len(res['overlap_regions']) == 1
    assert res['complete'] is False
    assert set(res['status']['reasons']) == {'overlap_region_unsupported'}
    region = res['overlap_regions'][0]
    outer = np.asarray(region.uv1_loops[0], dtype=float)
    assert outer[:, 0].min() <= 1e-3 and outer[:, 0].max() >= 1 - 1e-3
    assert outer[:, 1].min() <= 1e-3 and outer[:, 1].max() >= 1 - 1e-3
    assert len(region.uv1_loops) == 1          # no holes
    for loop in region.boundary:
        for bi, _rev in loop:
            assert 0 <= bi < len(res['branches'])
            assert res['branches'][bi].kind == 'overlap'
    for cut in n_tiles_expected_dissolved_seams:
        axis, value = cut
        for b in res['branches']:
            stuv = np.asarray(b.curve[0], dtype=float)
            assert not np.all(np.abs(stuv[:, axis] - value) <= 1e-3), \
                f"interior seam rim (axis {axis} @ {value}) survived"


def test_twin_planes_single_axis_unify_to_one_region():
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    a = insert_midknot(plane_z0(), axis=0)
    b = insert_midknot(plane_z0(), axis=0)
    res = nurbs_ssx(a, b, atol=1e-3)
    _assert_unified_plane_twin(res, [(0, 0.5)])


def test_twin_planes_both_axes_unify_to_one_region():
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    a = insert_midknot(insert_midknot(plane_z0(), axis=0), axis=1)
    b = insert_midknot(insert_midknot(plane_z0(), axis=0), axis=1)
    res = nurbs_ssx(a, b, atol=1e-3)
    _assert_unified_plane_twin(res, [(0, 0.5), (1, 0.5)])


# --- synthetic-tile unit tests: unifier bookkeeping without the engine ---

def _synthetic_edge(p0, p1, n=5):
    """Rim polyline from stuv p0 to p1 with n samples; xyz embeds the
    (s, t) preimage on a curved twin surface z = s*t."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    ts = np.linspace(0.0, 1.0, n)[:, None]
    stuv = p0[None, :] * (1 - ts) + p1[None, :] * ts
    xyz = np.stack([stuv[:, 0], stuv[:, 1],
                    stuv[:, 0] * stuv[:, 1]], axis=1)
    return stuv, xyz


def _synthetic_tile_pair():
    """Two tiles left/right of the s=0.5 seam over a curved coincidence;
    each tile one CCW loop of 4 edge rims; the two seam rims are
    opposite-orientation partners."""
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _Frag, _Tile, _RawResults)

    def frag(p0, p1):
        stuv, xyz = _synthetic_edge(
            [p0[0], p0[1], p0[0], p0[1]], [p1[0], p1[1], p1[0], p1[1]])
        return _Frag(stuv=stuv, xyz=xyz, kind='overlap', overlap=True)

    raw = _RawResults()
    # left tile CCW: bottom, seam (t 0->1), top, left (t 1->0)
    left = [frag((0, 0), (.5, 0)), frag((.5, 0), (.5, 1)),
            frag((.5, 1), (0, 1)), frag((0, 1), (0, 0))]
    # right tile CCW: bottom, right, top, seam (t 1->0)
    right = [frag((.5, 0), (1, 0)), frag((1, 0), (1, 1)),
             frag((1, 1), (.5, 1)), frag((.5, 1), (.5, 0))]
    raw.rim_frags = left + right
    cert = {'boundary_resid_max': 0.1, 'interior_resid': 0.05,
            'n_samples': 20, 'orientation_consistent': True}
    raw.tiles = [
        _Tile(pair=(0, 0), rect=(0, .5, 0, 1, 0, .5, 0, 1),
              loops=[[(0, False), (1, False), (2, False), (3, False)]],
              agreement=1,
              interior_stuv=np.array([.25, .5, .25, .5]),
              certification=dict(cert)),
        _Tile(pair=(1, 1), rect=(.5, 1, 0, 1, .5, 1, 0, 1),
              loops=[[(4, False), (5, False), (6, False), (7, False)]],
              agreement=1,
              interior_stuv=np.array([.75, .5, .75, .5]),
              certification=dict(cert)),
    ]
    return raw


def test_synthetic_tiles_dissolve_and_chain():
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _assemble_regions, _domain_ctx, _make_aggregate)
    ctx = _domain_ctx(plane_z0(), plane_z0(), atol=1e-3)
    agg = _make_aggregate({}, 1)
    raw = _synthetic_tile_pair()
    branches, regions = _assemble_regions(
        raw, [], ctx, 1e-3, agg, (0.5,), (), (0.5,), ())
    assert len(regions) == 1
    region = regions[0]
    # two seam rims dissolved: 6 surviving rim branches
    assert len(branches) == 6
    for b in branches:
        stuv = np.asarray(b.curve[0], dtype=float)
        assert not np.all(np.abs(stuv[:, 0] - 0.5) <= 4e-3)
    # single closed outer loop spanning the union
    assert len(region.uv1_loops) == 1
    outer = np.asarray(region.uv1_loops[0], dtype=float)
    assert np.allclose(outer[0], outer[-1])
    assert outer[:, 0].min() <= 1e-9 and outer[:, 0].max() >= 1 - 1e-9
    # conservative certification merge
    cert = region.certification
    assert cert['n_samples'] == 40
    assert cert['boundary_resid_max'] == 0.1
    assert cert['orientation_consistent'] is True
    # boundary refs valid and pointing at rim branches
    for loop in region.boundary:
        for bi, _rev in loop:
            assert 0 <= bi < len(branches)


def test_synthetic_tiles_agreement_mismatch_never_merges():
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _assemble_regions, _domain_ctx, _make_aggregate)
    ctx = _domain_ctx(plane_z0(), plane_z0(), atol=1e-3)
    agg = _make_aggregate({}, 1)
    raw = _synthetic_tile_pair()
    raw.tiles[1].agreement = -1
    branches, regions = _assemble_regions(
        raw, [], ctx, 1e-3, agg, (0.5,), (), (0.5,), ())
    assert len(regions) == 2
    assert len(branches) == 8          # nothing dissolved
    assert {r.normal_agreement for r in regions} == {1, -1}


def test_synthetic_interior_absorption_and_ref_shift():
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _Frag, _RawResults, _assemble_regions, _domain_ctx,
        _make_aggregate)
    from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch
    ctx = _domain_ctx(plane_z0(), plane_z0(), atol=1e-3)
    agg = _make_aggregate({}, 1)
    raw = _synthetic_tile_pair()
    raw.tiles = raw.tiles[:1]          # single LEFT tile only
    raw.rim_frags = raw.rim_frags[:4]

    def stitched_branch(s_val, kind):
        stuv, xyz = _synthetic_edge(
            [s_val, 0.2, s_val, 0.2], [s_val, 0.8, s_val, 0.8])
        return SSXBranch(curve=(stuv, xyz), kind=kind,
                         overlap=(kind == 'overlap'))

    inside_ovl = stitched_branch(0.25, 'overlap')      # absorbed
    outside_ovl = stitched_branch(0.75, 'overlap')     # outside left tile
    inside_trans = stitched_branch(0.25, 'transversal')  # kind-guarded
    branches, regions = _assemble_regions(
        raw, [inside_ovl, outside_ovl, inside_trans],
        ctx, 1e-3, agg, (0.5,), (), (0.5,), ())
    assert len(regions) == 1
    kept_kinds = [(b.kind, float(np.asarray(b.curve[0])[0, 0]))
                  for b in branches[:2]]
    assert (('overlap', 0.75) in kept_kinds
            and ('transversal', 0.25) in kept_kinds)
    assert len(branches) == 2 + 4      # 2 kept stitched + 4 rims
    for loop in regions[0].boundary:
        for bi, _rev in loop:
            assert 2 <= bi < len(branches)
            assert branches[bi].overlap


def test_synthetic_parallel_seam_rims_dissolve():
    """Fix-B isolation: the engine's rim sampler emits every edge in
    increasing local parameter (loop orientation lives in the rev
    flags), so adjacent tiles' seam rims arrive PARALLEL — dissolution
    must fire on the parallel endpoint pairing too."""
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _Frag, _assemble_regions, _domain_ctx, _make_aggregate)
    ctx = _domain_ctx(plane_z0(), plane_z0(), atol=1e-3)
    agg = _make_aggregate({}, 1)
    raw = _synthetic_tile_pair()
    # re-emit the RIGHT tile's seam rim parallel to the left's (t 0->1),
    # flipping its loop entry so the CCW traversal stays head-to-tail
    stuv, xyz = _synthetic_edge([.5, 0, .5, 0], [.5, 1, .5, 1])
    raw.rim_frags[7] = _Frag(stuv=stuv, xyz=xyz, kind='overlap',
                             overlap=True)
    raw.tiles[1].loops[0][3] = (7, True)
    branches, regions = _assemble_regions(
        raw, [], ctx, 1e-3, agg, (0.5,), (), (0.5,), ())
    assert len(regions) == 1
    assert len(branches) == 6


# --- Fix-C retirement machinery: regression suite ------------------------

def _unified_region_fixture():
    """One unified full-domain region (standard two-tile pair) + ctx,
    for driving the retirement helper directly."""
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _assemble_regions, _domain_ctx, _make_aggregate)
    ctx = _domain_ctx(plane_z0(), plane_z0(), atol=1e-3)
    agg = _make_aggregate({}, 1)
    raw = _synthetic_tile_pair()
    _branches, regions = _assemble_regions(
        raw, [], ctx, 1e-3, agg, (0.5,), (), (0.5,), ())
    assert len(regions) == 1
    return ctx, regions


def test_multiplicity_retired_when_rect_region_interior():
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _RawResults, _make_aggregate,
        _retire_multiplicity_if_region_explained)
    from mmcore.numeric._work_budget import REASON_MULTIPLICITY
    ctx, regions = _unified_region_fixture()
    raw = _RawResults()
    raw.mult_rects.append((0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4))
    agg = _make_aggregate({}, 1)
    agg.mark(REASON_MULTIPLICITY)
    _retire_multiplicity_if_region_explained(raw, regions, ctx, agg)
    assert REASON_MULTIPLICITY not in agg.reasons
    assert agg.result_fields()['complete'] is True


def test_multiplicity_kept_when_any_rect_escapes():
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _RawResults, _make_aggregate,
        _retire_multiplicity_if_region_explained)
    from mmcore.numeric._work_budget import REASON_MULTIPLICITY
    ctx, regions = _unified_region_fixture()
    raw = _RawResults()
    raw.mult_rects.append((0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4))
    raw.mult_rects.append((2.0, 2.4, 2.0, 2.4, 2.0, 2.4, 2.0, 2.4))
    agg = _make_aggregate({}, 1)
    agg.mark(REASON_MULTIPLICITY)
    _retire_multiplicity_if_region_explained(raw, regions, ctx, agg)
    assert REASON_MULTIPLICITY in agg.reasons
    assert agg.result_fields()['complete'] is False


def test_multiplicity_kept_when_one_sided():
    """Two-sided site rule: a rect inside the region in uv1 but outside
    in uv2 must NOT retire."""
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _RawResults, _make_aggregate,
        _retire_multiplicity_if_region_explained)
    from mmcore.numeric._work_budget import REASON_MULTIPLICITY
    ctx, regions = _unified_region_fixture()
    raw = _RawResults()
    raw.mult_rects.append((0.2, 0.4, 0.2, 0.4, 2.0, 2.4, 2.0, 2.4))
    agg = _make_aggregate({}, 1)
    agg.mark(REASON_MULTIPLICITY)
    _retire_multiplicity_if_region_explained(raw, regions, ctx, agg)
    assert REASON_MULTIPLICITY in agg.reasons
    assert agg.result_fields()['complete'] is False


def test_multiplicity_kept_when_postprocess_starved():
    """Zero postprocess budget: containment unverified — never retire on
    unverified evidence; the typed cap reason is recorded."""
    from mmcore.numeric.intersection.ssx._nssx5 import (
        _RawResults, _make_aggregate,
        _retire_multiplicity_if_region_explained)
    from mmcore.numeric._work_budget import (
        REASON_MULTIPLICITY, REASON_POSTPROCESS_CAP)
    ctx, regions = _unified_region_fixture()
    raw = _RawResults()
    raw.mult_rects.append((0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4))
    agg = _make_aggregate({'max_postprocess_work': 0}, 1)
    agg.mark(REASON_MULTIPLICITY)
    _retire_multiplicity_if_region_explained(raw, regions, ctx, agg)
    assert REASON_MULTIPLICITY in agg.reasons
    assert REASON_POSTPROCESS_CAP in agg.reasons
    assert agg.result_fields()['complete'] is False


# ---------------------------------------------------------------------------
# Task 6: pickle fixtures (small cases; the Task-7 harness covers all 10)
# ---------------------------------------------------------------------------

def _load_case(num):
    """Return the (s1, s2) surface pair from fixture `num`.

    Fixtures were saved by earlier nurbs_ssx sessions
    (examples/ssx/nurbs_nurbs_intersection_*.py produced them); the
    surface pairs are the durable content. The stored reference curve
    lists in data[1:] are deliberately ignored (old-engine artifacts —
    see the certificate test docstring).
    """
    path = FIXTURE_DIR / f"nurbs_nurbs_intersection_{num}.pkl"
    if not path.exists():
        pytest.skip(f"fixture {path.name} not present")
    with open(path, "rb") as f:
        data = pickle.load(f)
    s1, s2 = data[0]
    return s1, s2


@pytest.mark.parametrize("case,expect_complete,expect_reasons", [
    (5, True, set()),
    (8, True, set()),
    # pins CURRENT engine truth: if bez_ssx later resolves this tangential
    # zone, update this expectation (an improvement will fail this line, by
    # design)
    (10, False, {'unresolved_tangential_zone'}),
])
def test_fixture_case_residual_certificate(case, expect_complete,
                                           expect_reasons):
    """Real-NURBS fixture gate: sampled branch vertices (every len//10-th)
    must lie on BOTH surfaces (the on-intersection certificate), and the
    status must match per-case engine truth.

    The pickles' stored reference CURVES are deliberately unused: they
    are old-engine artifacts (case 5 stores a non-rational polynomial
    'circle' deviating up to 76*atol from the true circle the new engine
    reproduces exactly; case 8 stores corrupt control points at +-5e6).
    The surface pairs are the valuable fixture. Independent completeness
    coverage runs in examples/ssx/nurbs_ssx5_coverage_check.py (Task 7)
    against an isoline x nurbs_csx reference cloud.

    Case 10 is a genuine tangential contact (s1 has z>=5, s2 has z<=5,
    touching at z=5): typed-partial per established engine truth.
    """
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    atol = 1e-3
    s1, s2 = _load_case(case)
    res = nurbs_ssx(s1, s2, atol=atol)
    assert res['branches'], f"case {case}: no branches"
    assert res['complete'] is expect_complete, res['status']['reasons']
    assert set(res['status']['reasons']) == expect_reasons
    for b in res['branches']:
        stuv = np.asarray(b.curve[0], dtype=float)
        xyz = np.asarray(b.curve[1], dtype=float)
        step = max(1, len(stuv) // 10)
        for k in range(0, len(stuv), step):
            s, t, u, v = stuv[k]
            p1 = np.asarray(evaluate_nurbs_surface(
                s1, s, t, d_order=0)['S'], dtype=float)[:3]
            p2 = np.asarray(evaluate_nurbs_surface(
                s2, u, v, d_order=0)['S'], dtype=float)[:3]
            assert np.linalg.norm(p1 - xyz[k]) <= 2 * atol
            assert np.linalg.norm(p2 - xyz[k]) <= 2 * atol
    if case == 5:
        # geometry invariant: the SSI is the unit circle at z=1 — the
        # new engine reproduces it exactly; the stored reference did not
        xyz_all = np.vstack([np.asarray(b.curve[1], dtype=float)
                             for b in res['branches']])
        assert np.abs(np.linalg.norm(xyz_all[:, :2], axis=1) - 1.0).max() \
            <= 2 * atol
        assert np.abs(xyz_all[:, 2] - 1.0).max() <= 2 * atol


# ---------------------------------------------------------------------------
# P1 invariance acceptance gates (kickoff 2026-07-20 gates 1-2; design
# 2026-07-21).  Case 6: ~100-unit coords; case 11: ~800-unit part at
# ~3000-unit offset — both must certify at ORIGINAL world coordinates.
# These are the committed regressions that FAIL without the canonical
# frame (pre-fix: case 6 lost half its curve with trace_unverified).
# ---------------------------------------------------------------------------


def _load_fixture_pair(num):
    with open(FIXTURE_DIR / f"nurbs_nurbs_intersection_{num}.pkl", "rb") as f:
        return pickle.load(f)[0]


def test_case6_original_coords_complete_at_atol_1e3():
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx

    s1, s2 = _load_fixture_pair(6)
    r = nurbs_ssx(s1, s2, atol=1e-3)
    assert r["complete"], r["status"]["reasons"]
    assert r["status"]["reasons"] == []
    assert len(r["branches"]) == 1
    xyz = np.asarray(r["branches"][0].curve[1], dtype=float)
    # Kickoff engine truth: one x=y-mirror-symmetric arm in the plane z=1
    # from ~[4.37, 75] to ~[75, 4.37] passing through ~[5.47, 5.47].
    assert np.all(np.abs(xyz[:, 2] - 1.0) <= 5e-3)
    lo, hi = (xyz[0], xyz[-1]) if xyz[0][0] < xyz[-1][0] else (xyz[-1], xyz[0])
    assert np.allclose(lo[:2], [4.37, 75.0], atol=1.0)
    assert np.allclose(hi[:2], [75.0, 4.37], atol=1.0)


def test_case11_original_coords_complete_at_atol_0_1():
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx

    s1, s2 = _load_fixture_pair(11)
    r = nurbs_ssx(s1, s2, atol=0.1)
    assert r["complete"], r["status"]["reasons"]
    assert r["status"]["reasons"] == []
    assert len(r["branches"]) == 1


def test_case11_original_coords_certificate_clean_at_atol_1e3():
    # P1 fixes the certificate half; the knob-unreachable tier (P2) may
    # still mark work_budget — trace_unverified specifically must be gone.
    from mmcore.numeric._work_budget import REASON_TRACE_UNVERIFIED
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx

    s1, s2 = _load_fixture_pair(11)
    r = nurbs_ssx(s1, s2, atol=1e-3)
    assert REASON_TRACE_UNVERIFIED not in r["status"]["reasons"], r["status"]


# ---------------------------------------------------------------------------
# Cluster-4 burn-down (2026-07-25): user-authored boundary-coincidence pair.
#
# s1 is a bilinear whose height is z = 5(1-u)v, so its z=0 locus is exactly
# its u=1 and v=0 DOMAIN EDGES; s2 is a planar (z=0) non-parallelogram quad.
# The true intersection is those two straight edges clipped to s2's quad.
# Their shared corner (-36, 2, 0) lies OUTSIDE s2, so the result is two
# separate boundary-coincident branches, not one polyline through the corner.
#
# The pair's joint magnitude is 36 — above the identity window — so the whole
# call runs in the k=2 canonical frame.  Before the Phase-2 hull-prune
# translation fix, the centered frame's cancellation noise made
# `_vector_residual_hull_excludes_zero` delete the v=0 span's boundary
# crossing; the CSX overlap tier then failed to arm and the span evaporated
# into reasons=['overlap_region_unsupported'] at 29% coverage.
#
# Truth below is computed analytically (segment/quad clipping), not recorded
# from the engine.
# ---------------------------------------------------------------------------

_BC_S1_CP = np.array([[[-16.0, -27.0, 0.0], [-8.0, -25.0, 5.0]],
                      [[-36.0, 2.0, 0.0], [-20.0, -3.0, 0.0]]])
_BC_S2_CP = np.array([[[-34.0, -7.0, 0.0], [-26.0, 2.0, 0.0]],
                      [[-19.0, -20.0, 0.0], [-17.0, -10.0, 0.0]]])

# v=0 edge clipped to the quad: t in [0.377142857143, 0.781553398058]
# u=1 edge clipped to the quad: t in [0.489130434783, 0.816326530612]
_BC_TRUTH = [
    (np.array([-23.542857142857, -16.062857142857, 0.0]),
     np.array([-31.631067961165, -4.334951456311, 0.0]), 14.2465057482),
    (np.array([-28.173913043478, -0.445652173913, 0.0]),
     np.array([-22.938775510204, -2.081632653061, 0.0]), 5.4848060240),
]


def _boundary_coincidence_pair():
    def surf(cp, ku, kv):
        return NURBSSurfaceTuple(
            order_u=2, order_v=2,
            knot_u=np.array([0.0, 0.0, ku, ku]),
            knot_v=np.array([0.0, 0.0, kv, kv]),
            control_points=cp, weights=np.ones((2, 2)))

    return (surf(_BC_S1_CP, 29.20616373, 18.68154169),
            surf(_BC_S2_CP, 19.84943324, 12.04159458))


def _polyline_length(xyz):
    xyz = np.asarray(xyz, dtype=float)
    if len(xyz) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))


def test_boundary_coincidence_two_edge_branches():
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx

    s1, s2 = _boundary_coincidence_pair()
    res = nurbs_ssx(s1, s2, atol=1e-3)

    assert res["complete"] is True, res["status"]["reasons"]
    assert res["status"]["reasons"] == []
    assert len(res["branches"]) == 2, [b.kind for b in res["branches"]]

    got = []
    for b in res["branches"]:
        xyz = np.asarray(b.curve[1], dtype=float)
        assert len(xyz) >= 2, b.kind
        # The whole locus is the z=0 plane.
        assert np.max(np.abs(xyz[:, 2])) <= 1e-6, np.max(np.abs(xyz[:, 2]))
        got.append(xyz)

    # Match each analytic segment to one branch by endpoint proximity, then
    # pin its length: a truncated span (the pre-fix failure mode shipped 29%
    # of the v=0 edge) fails the length check even if the endpoints round.
    remaining = list(range(len(got)))
    for a, b_end, length in _BC_TRUTH:
        best, best_d = None, np.inf
        for j in remaining:
            xyz = got[j]
            d = min(np.linalg.norm(xyz[0] - a) + np.linalg.norm(xyz[-1] - b_end),
                    np.linalg.norm(xyz[0] - b_end) + np.linalg.norm(xyz[-1] - a))
            if d < best_d:
                best, best_d = j, d
        assert best_d <= 1e-2, (best_d, a, b_end)
        assert abs(_polyline_length(got[best]) - length) <= 1e-2, (
            _polyline_length(got[best]), length)
        remaining.remove(best)
    assert not remaining


def test_boundary_coincidence_survives_similarity():
    """The class is scale/translation invariant; so must the result be.

    Cells straddle the identity-window cliff in both directions.
    """
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx

    s1, s2 = _boundary_coincidence_pair()
    for c, k in [(np.zeros(3), 1.0),
                 (np.zeros(3), 1.0 / 64.0),
                 (np.zeros(3), 256.0),
                 (np.array([1e3, -2e3, 5e2]), 1.0),
                 (np.array([-5e3, 3e3, 1e4]), 8.0)]:
        t1 = NURBSSurfaceTuple(
            order_u=2, order_v=2, knot_u=s1.knot_u, knot_v=s1.knot_v,
            control_points=s1.control_points * k + c, weights=s1.weights)
        t2 = NURBSSurfaceTuple(
            order_u=2, order_v=2, knot_u=s2.knot_u, knot_v=s2.knot_v,
            control_points=s2.control_points * k + c, weights=s2.weights)
        res = nurbs_ssx(t1, t2, atol=1e-3 * k)
        assert res["complete"] is True, (c, k, res["status"]["reasons"])
        assert len(res["branches"]) == 2, (c, k, [b.kind for b in res["branches"]])
        lengths = sorted(_polyline_length(b.curve[1]) for b in res["branches"])
        expect = sorted(t[2] * k for t in _BC_TRUTH)
        for got_l, exp_l in zip(lengths, expect):
            assert abs(got_l - exp_l) <= 1e-2 * k, (c, k, lengths, expect)


# ---------------------------------------------------------------------------
# P2 (2026-07-25): the public work knobs must actually reach the engine.
#
# `_make_aggregate` documents explicit values as "absolute aggregate
# promises", but every per-pair `bez_ssx` call was then clamped to the
# module default (250k cells) regardless of what the caller asked for.  With
# a single candidate pair that is pure knob-unreachability: measured on
# harness case 11 at atol=2.5e-6, passing max_cells=2_000_000 still stopped
# the engine at 249,657/250,000 and reported reasons=['work_budget'] —
# telling the consumer to raise a knob that provably does nothing.
#
# The default path must stay bit-identical (the per-pair default is a
# fairness share, not a ceiling to be lifted for everyone); only an explicit
# aggregate promise is redistributed across the remaining candidates.
# ---------------------------------------------------------------------------

def _capture_bez_ssx_budgets(monkeypatch, surfs, **kwargs):
    import mmcore.numeric.intersection.ssx._nssx5 as nm

    seen = []
    orig = nm.bez_ssx

    def spy(P1, P2, **kw):
        seen.append({k: kw.get(k) for k in
                     ("max_cells", "max_csx_calls", "max_output_items")})
        return orig(P1, P2, **kw)

    monkeypatch.setattr(nm, "bez_ssx", spy)
    nm.nurbs_ssx(surfs[0], surfs[1], **kwargs)
    return seen


def test_explicit_max_cells_reaches_the_engine(monkeypatch):
    from mmcore.numeric.intersection.ssx._nssx5 import _BEZ_DEFAULT_MAX_CELLS

    pair = _boundary_coincidence_pair()
    seen = _capture_bez_ssx_budgets(
        monkeypatch, pair, atol=1e-3, max_cells=2_000_000)
    assert seen, "no bez_ssx call captured"
    assert max(s["max_cells"] for s in seen) > _BEZ_DEFAULT_MAX_CELLS, seen


def test_default_budget_path_is_unchanged(monkeypatch):
    """The per-pair default is a fairness share; an unset budget must give
    each pair exactly the module default, as before."""
    from mmcore.numeric.intersection.ssx._nssx5 import _BEZ_DEFAULT_MAX_CELLS

    pair = _boundary_coincidence_pair()
    seen = _capture_bez_ssx_budgets(monkeypatch, pair, atol=1e-3)
    assert seen
    for s in seen:
        assert s["max_cells"] == _BEZ_DEFAULT_MAX_CELLS, seen


def test_explicit_budget_is_absolute_for_every_pair(monkeypatch):
    """Multi-candidate coverage — the case the first version got wrong.

    With one candidate pair any per-pair policy looks identical, so the
    original two tests could not see that an even fair-share slice starves
    the hot pair.  Work is not spread evenly over BVH candidates: on harness
    case 1 (43 pairs) slicing turned an explicit max_cells=250_000 from
    complete into reasons=['work_budget'] with 61% of the aggregate unspent.
    The contract (`_make_aggregate`: "explicit values are absolute aggregate
    promises"; _ncsx4/_nccx4 hand each call the whole remainder) requires
    every pair to be offered what is LEFT, not a slice of it.
    """
    import mmcore.numeric.intersection.ssx._nssx5 as nm
    from mmcore.numeric.intersection.ssx._nssx5 import _per_pair_allowance

    class Agg:
        def __init__(self, cells, explicit):
            self.remaining_cells = cells
            self.remaining_csx_calls = 10 ** 9
            self.remaining_output_items = 10 ** 9
            self.explicit_cells = explicit
            self.explicit_csx = False
            self.explicit_output = False

    # explicit: every pair is offered the full remainder, at any n
    for n in (1, 2, 4, 43):
        cells, _csx, _out = _per_pair_allowance(Agg(1_000_000, True), n)
        assert cells == 1_000_000, (n, cells)

    # default: the module default is the per-pair share, unchanged
    for n in (1, 2, 4, 43):
        cells, _csx, _out = _per_pair_allowance(
            Agg(nm._BEZ_DEFAULT_MAX_CELLS * n, False), n)
        assert cells == nm._BEZ_DEFAULT_MAX_CELLS, (n, cells)

    # a nearly-drained explicit ledger still offers exactly what is left
    cells, _csx, _out = _per_pair_allowance(Agg(7, True), 43)
    assert cells == 7


def test_multi_candidate_default_path_grants_are_unchanged(monkeypatch):
    """Regression pin on the DEFAULT path with more than one candidate."""
    import mmcore.numeric.intersection.ssx._nssx5 as nm

    s1, s2 = _boundary_coincidence_pair()
    m1 = insert_midknot(s1, axis=0)
    m2 = insert_midknot(s2, axis=0)
    seen = _capture_bez_ssx_budgets(monkeypatch, (m1, m2), atol=1e-3)
    assert len(seen) >= 2, f"expected a multi-candidate split, got {len(seen)}"
    for s in seen:
        assert s["max_cells"] == nm._BEZ_DEFAULT_MAX_CELLS, seen


# ---------------------------------------------------------------------------
# Case 11 (2026-07-26): the per-march allowance must be the caller's ledger,
# not a hardcoded point count.
#
# The tracer used `trace_limit = 400`, which bounds nothing: the points a
# march needs are (arc length)/(step size), and the step is chosen from atol
# and curvature, so the requirement grows ~1/sqrt(atol) while 400 stayed
# put.  Harness case 11's true intersection is ONE CLOSED LOOP of length
# 1261.25; at atol=1e-3 the cap truncated it to 92.68% coverage AND cost
# extra work (20,811 cells vs 15,077), because the fragments were then
# re-processed.
# ---------------------------------------------------------------------------

_CASE11_TRUE_LENGTH = 1261.25


def _load_case11():
    import pathlib
    import pickle
    with open(FIXTURE_DIR / "nurbs_nurbs_intersection_11.pkl", "rb") as f:
        return pickle.load(f)[0]


@pytest.mark.parametrize("atol", [1e-1, 1e-2, 1e-3])
def test_case11_recovers_the_whole_closed_loop(atol):
    """One closed loop, whole, at every tolerance the budget can afford."""
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx

    s1, s2 = _load_case11()
    r = nurbs_ssx(s1, s2, atol=atol)

    assert r["complete"] is True, r["status"]["reasons"]
    assert r["status"]["reasons"] == []
    assert len(r["branches"]) == 1, [b.kind for b in r["branches"]]
    branch = r["branches"][0]
    assert bool(branch.closed), "the loop must close, not fragment"
    xyz = np.asarray(branch.curve[1], dtype=float)
    length = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
    assert abs(length - _CASE11_TRUE_LENGTH) <= 1.0, length


def test_case11_march_allowance_comes_from_the_ledger(monkeypatch):
    """Pin the derivation, not just its effect.

    A march must be offered what the shared ledger has left. The former
    fixed 400 is what made the stop knob-unreachable: no budget the caller
    could set would change it.
    """
    import mmcore.numeric.intersection.ssx._bez_ssx5 as bm

    seen = []
    orig = bm._march_to_boundary

    def spy(*a, **k):
        if k.get("max_points") is not None:
            seen.append(int(k["max_points"]))
        return orig(*a, **k)

    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx

    monkeypatch.setattr(bm, "_march_to_boundary", spy)
    s1, s2 = _load_case11()
    bm_result = nurbs_ssx(s1, s2, atol=1e-3)
    assert seen, "no march observed"
    # Every allowance must exceed the old hardcoded cap by a wide margin,
    # because it is now a slice of a 250k-cell ledger rather than a constant.
    assert min(seen) > 400, sorted(seen)[:5]
    assert bm_result["complete"] is True


# ---------------------------------------------------------------------------
# Truncation-cause propagation (2026-07-26).  A CSX depth ceiling is local
# and structural; it must stop that face, not the run, and must not be
# reported as a resource shortfall.
#
# Measured before the fix, harness case 11 at atol=1e-5: ONE bez_csx call
# hit its Phase-2 max_depth (1,791 of 100,000 CSX cells used, topology
# complete) and `_run_csx` escalated it to mark_exhausted(work_budget) --
# a global stop at 1.2% of the SSX ledger.  Subdivision collapsed from 98
# cells to 2, 17 marches to 1, and 37.1% of a closed loop was reported as
# complete=False/'work_budget' at 17% ledger utilization.
# ---------------------------------------------------------------------------

def test_depth_ceiling_is_typed_and_local():
    """The reason names the real limit, and the search is not aborted.

    Forced deterministically with the explicit knob rather than by picking a
    tolerance that happens to exceed the ceiling: the derived default now
    tracks atol, so no fixed atol reliably triggers this any more (which is
    the point of the derivation).
    """
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx

    s1, s2 = _load_case11()
    # depth=40 is shallow enough to trip the ceiling but not so shallow that
    # CSX finds nothing at all: it yields a genuinely PARTIAL result, which
    # is what distinguishes "this face stopped" from "the run was aborted".
    r = nurbs_ssx(s1, s2, atol=1e-3, csx_max_depth=40)

    assert r["status"]["reasons"] == ["depth_limit"], r["status"]["reasons"]
    # A structural ceiling must not masquerade as resource exhaustion: the
    # ledger is nowhere near spent, so 'work_budget' would have been a lie.
    w = r["status"]["work"]
    assert w["cells_processed"] < 0.9 * w["max_cells"]
    # ...and the search continued rather than being globally aborted: most
    # of the loop is still traced despite the truncated face.
    assert r["branches"], "a local depth ceiling aborted the whole run"
    total = sum(_polyline_length(b.curve[1]) for b in r["branches"])
    assert total > 0.5 * _CASE11_TRUE_LENGTH, total


def test_derived_depth_needs_no_knob_at_tight_tolerance():
    """The default tracks atol, so tight work completes unconfigured.

    atol=1e-5 was 37.1% covered with a `work_budget` reason when the ceiling
    was the constant 64 (required depth there is 76.7).
    """
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx

    s1, s2 = _load_case11()
    r = nurbs_ssx(s1, s2, atol=1e-5)

    assert r["complete"] is True, r["status"]["reasons"]
    assert r["status"]["reasons"] == []
    total = sum(_polyline_length(b.curve[1]) for b in r["branches"])
    assert abs(total - _CASE11_TRUE_LENGTH) <= 1.0, total


def test_derived_depth_stays_inside_float64_resolution():
    """The ceiling can never exceed what subdivision can actually execute."""
    from mmcore.numeric.intersection.csx._bez_csx4 import (
        _derived_max_depth, _CSX_DEPTH_FLOAT64_CEILING,
    )

    # Absurdly tight tolerances must clamp, not run away.
    assert _derived_max_depth(1e-300, 1e-300, 1e-300) == _CSX_DEPTH_FLOAT64_CEILING
    assert _derived_max_depth(0.0, 0.0, 0.0) == _CSX_DEPTH_FLOAT64_CEILING
    # And a loose tolerance must still clear its measured requirement (56.8
    # at atol=1e-3 on this fixture).
    assert _derived_max_depth(6.3e-6, 5.1e-6, 3.37e-6) >= 57
