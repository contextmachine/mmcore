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
