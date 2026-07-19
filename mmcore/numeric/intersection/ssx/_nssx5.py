"""NURBS surface-surface intersection over the v5 sq-dist Bezier SSX.

Adapter in the `_nccx4.py` / `_ncsx4.py` style (spec:
docs/superpowers/specs/2026-07-19-nurbs-ssx5-design.md): decompose both
NURBS surfaces into Bezier patches, run ``bez_ssx`` per BVH-candidate
patch pair under shared aggregate work ledgers, remap every output into
the surfaces' global knot domains, then assemble one NURBS-level result:
stitched branches (wrap-aware across C0-periodic seams), deduplicated
points and singularities, unified overlap regions, aggregated schema-v2
status.

Contract (native layer — no return-shaping flags, no curve fitting):

    nurbs_ssx(surf1, surf2, atol=1e-3, **expert_knobs) -> dict

with the exact ``bez_ssx`` result schema: ``branches`` (SSXBranch with
``curve=(stuv (N,4), xyz (N,3))`` polylines; ``curve_xyz/st/uv`` stay
None), ``points``, ``singularities``, ``overlap_regions``,
``unresolved_regions``, ``complete``, ``status={'reasons','work'}``.
``stuv=(s,t,u,v)``: (s,t) in surf1.interval(), (u,v) in surf2.interval().

Known representation consequence (spec delta 3): an SSI curve lying
exactly on a decomposition knot line is reported by the adjacent pairs'
boundary CSX as a curve-on-surface overlap and therefore carries
``kind='overlap'`` even when the surfaces cross transversally there.

Incompleteness is always soft: certified partial output is returned with
``complete=False`` and typed ``status['reasons']``; exceptions are for
caller errors only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mmcore.geom import nurbs
from mmcore.geom._nurbs_eval import (
    NURBSSurfaceTuple, _nurbs_to_tuple, to_homogeneous_2d,
)
from mmcore.geom._nurbs_knots import decompose_surface
from mmcore.geom._nurbs_param_tol import nurbs_surface_param_tolerance
from mmcore.geom.bvh.lbvh import AABB, build_bvh, bvh_intersect
from mmcore.numeric._work_budget import (
    SoftWorkBudget,
    REASON_WORK_BUDGET,
    REASON_POSTPROCESS_CAP,
)
from mmcore.numeric.intersection.ssx._bez_ssx5 import (
    bez_ssx, SSXSingularity, _dist_point_polyline,
)
from mmcore.numeric.intersection.ssx._ssx4 import SSXBranch, SSXPoint
from mmcore.numeric.intersection.ssx._ssx5_overlap import (
    SSXOverlapRegion, _point_in_polygon, _dist_point_polyline_2d,
)

# ---------------------------------------------------------------------------
# Knobs (bez_ssx defaults; aggregate ledgers scale with candidate count)
# ---------------------------------------------------------------------------

_BEZ_DEFAULT_MAX_CELLS = 250_000
_BEZ_DEFAULT_MAX_CSX_CALLS = 10_000
_BEZ_DEFAULT_MAX_OUTPUT_ITEMS = 1_024

_AGGREGATE_KWARGS = ('max_cells', 'max_csx_calls', 'max_output_items',
                     'max_postprocess_work')
_FORWARD_KWARGS = ('max_depth', 'max_xyz_step', 'csx_max_cells',
                   'boundary_csx_max_cells', 'csx_max_results')
_ALLOWED_KWARGS = _AGGREGATE_KWARGS + _FORWARD_KWARGS


def _reject_unknown_kwargs(context, kwargs, allowed):
    """Fail fast on unknown kwargs (local: this adapter's tolerance is
    ``atol``, mirroring bez_ssx — the shared adapter helper's message
    would wrongly point at 'tol')."""
    unknown = sorted(set(kwargs) - set(allowed))
    if unknown:
        raise TypeError(
            f"{context}: unexpected keyword argument(s) {unknown}; "
            f"accepted: {sorted(allowed)}. The geometric tolerance is "
            "'atol' (this adapter mirrors the bez_ssx contract).")


# ---------------------------------------------------------------------------
# Input & domain helpers
# ---------------------------------------------------------------------------

def _as_surface_tuple(surf) -> NURBSSurfaceTuple:
    if isinstance(surf, NURBSSurfaceTuple):
        return surf
    if isinstance(surf, nurbs.NURBSSurface):
        return _nurbs_to_tuple(surf)
    raise TypeError(
        "nurbs_ssx: arguments must be NURBSSurfaceTuple or "
        f"mmcore.geom.nurbs.NURBSSurface, not {type(surf).__name__}")


def _is_rational(surf: NURBSSurfaceTuple) -> bool:
    return not np.allclose(surf.weights, 1.0)


def _axis_closed(surf: NURBSSurfaceTuple, axis: int) -> bool:
    """C0-periodicity test per parametric axis (the `_ncsx4` seam rule):
    first and last control-point/weight rows coincide."""
    cp, w = surf.control_points, surf.weights
    if axis == 0:
        return bool(np.allclose(cp[0], cp[-1], atol=1e-10)
                    and np.allclose(w[0], w[-1], atol=1e-10))
    return bool(np.allclose(cp[:, 0], cp[:, -1], atol=1e-10)
                and np.allclose(w[:, 0], w[:, -1], atol=1e-10))


@dataclass
class _DomainCtx:
    """Global stuv-domain metadata: bounds, spans, per-axis parametric
    tolerance, and C0-periodicity flags. Axis order: s, t (surf1), u, v
    (surf2)."""
    lows: NDArray[np.float64]
    highs: NDArray[np.float64]
    spans: NDArray[np.float64]
    ptol: NDArray[np.float64]
    closed: tuple


def _domain_ctx(s1: NURBSSurfaceTuple, s2: NURBSSurfaceTuple,
                atol: float) -> _DomainCtx:
    (a0, a1), (b0, b1) = s1.interval()
    (c0, c1), (d0, d1) = s2.interval()
    p_s, p_t = nurbs_surface_param_tolerance(s1, atol)
    p_u, p_v = nurbs_surface_param_tolerance(s2, atol)
    lows = np.array([a0, b0, c0, d0], dtype=np.float64)
    highs = np.array([a1, b1, c1, d1], dtype=np.float64)
    ptol = np.maximum(
        np.array([p_s, p_t, p_u, p_v], dtype=np.float64), 1e-12)
    closed = (_axis_closed(s1, 0), _axis_closed(s1, 1),
              _axis_closed(s2, 0), _axis_closed(s2, 1))
    return _DomainCtx(lows=lows, highs=highs, spans=highs - lows,
                      ptol=ptol, closed=closed)


def _axis_diff(a: float, b: float, axis: int, ctx: _DomainCtx) -> float:
    """|a-b| per stuv axis, modulo the domain span on C0-closed axes."""
    d = abs(float(a) - float(b))
    if ctx.closed[axis] and ctx.spans[axis] > 0.0:
        d = min(d, float(ctx.spans[axis]) - d)
    return d


def _axis_diff_nowrap(a: float, b: float) -> float:
    return abs(float(a) - float(b))


def _match_stuv(p, q, xyz_p, xyz_q, ctx: _DomainCtx, atol: float) -> bool:
    """Matching/unification predicate (tolerance ladder): per-axis
    4·ptol AND xyz <= 2·atol. Wrap-aware."""
    d = np.asarray(xyz_p, dtype=np.float64) - np.asarray(
        xyz_q, dtype=np.float64)
    if float(np.linalg.norm(d)) > 2.0 * atol:
        return False
    return all(_axis_diff(p[i], q[i], i, ctx) <= 4.0 * float(ctx.ptol[i])
               for i in range(4))


def _dup_stuv(p, q, xyz_p, xyz_q, ctx: _DomainCtx, atol: float) -> bool:
    """Destructive-dedup predicate: per-axis 1·ptol AND xyz <= atol.
    Wrap-aware. Every destructive test carries the xyz guard."""
    d = np.asarray(xyz_p, dtype=np.float64) - np.asarray(
        xyz_q, dtype=np.float64)
    if float(np.linalg.norm(d)) > atol:
        return False
    return all(_axis_diff(p[i], q[i], i, ctx) <= float(ctx.ptol[i])
               for i in range(4))


def _joint_plain_dup(p, q, xyz_p, xyz_q, ctx: _DomainCtx,
                     atol: float) -> bool:
    """Destructive predicate WITHOUT wrap: used at stitch joints to decide
    vertex collapse. A wrap-only match keeps both seam preimages (the
    periodic vertex-pair contract)."""
    d = np.asarray(xyz_p, dtype=np.float64) - np.asarray(
        xyz_q, dtype=np.float64)
    if float(np.linalg.norm(d)) > atol:
        return False
    return all(_axis_diff_nowrap(p[i], q[i]) <= float(ctx.ptol[i])
               for i in range(4))


# ---------------------------------------------------------------------------
# Remapping local Bezier [0,1] params -> global knot-domain params
# ---------------------------------------------------------------------------

def _pair_rect(p1: NURBSSurfaceTuple, p2: NURBSSurfaceTuple):
    (s0, s1), (t0, t1) = p1.interval()
    (u0, u1), (v0, v1) = p2.interval()
    return (float(s0), float(s1), float(t0), float(t1),
            float(u0), float(u1), float(v0), float(v1))


def _remap4(stuv_local, rect):
    """Affine per-axis map of (4,) or (N,4) local stuv into global params."""
    x = np.array(stuv_local, dtype=np.float64, copy=True)
    s0, s1, t0, t1, u0, u1, v0, v1 = rect
    low = np.array([s0, t0, u0, v0], dtype=np.float64)
    scale = np.array([s1 - s0, t1 - t0, u1 - u0, v1 - v0],
                     dtype=np.float64)
    return low + x * scale


def _remap2(uv_local, rect2):
    """Affine map of (N,2) local uv by (lo0, hi0, lo1, hi1)."""
    x = np.array(uv_local, dtype=np.float64, copy=True)
    lo0, hi0, lo1, hi1 = rect2
    x[..., 0] = lo0 + x[..., 0] * (hi0 - lo0)
    x[..., 1] = lo1 + x[..., 1] * (hi1 - lo1)
    return x


# ---------------------------------------------------------------------------
# Aggregate status (schema v2 across pairs)
# ---------------------------------------------------------------------------

@dataclass
class _AggregateStatus:
    """Shared ledgers + status folding across per-pair bez_ssx calls.

    Invariant (same as SoftWorkBudget): ``complete == (not reasons)`` —
    every truncation or partiality records a REASON_* string.
    The wrapper's own assembly work charges ``post`` (a SoftWorkBudget
    used only for its postprocess pool).
    """
    max_cells: int
    max_csx_calls: int
    max_output_items: int
    post: SoftWorkBudget
    cells_processed: int = 0
    csx_calls: int = 0
    output_items: int = 0
    reasons: list = field(default_factory=list)
    cell_counts: dict = field(default_factory=dict)

    def _add(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)

    @property
    def remaining_cells(self) -> int:
        return max(0, self.max_cells - self.cells_processed)

    @property
    def remaining_csx_calls(self) -> int:
        return max(0, self.max_csx_calls - self.csx_calls)

    @property
    def remaining_output_items(self) -> int:
        return max(0, self.max_output_items - self.output_items)

    def consume(self, result: dict) -> None:
        """Fold one bez_ssx result's status into the aggregate."""
        status = result.get('status', {}) or {}
        work = status.get('work', {}) or {}
        self.cells_processed += max(0, int(work.get('cells_processed', 0)))
        self.csx_calls += max(0, int(work.get('csx_calls', 0)))
        self.output_items += max(0, int(work.get('output_items', 0)))
        for key, val in dict(work.get('cell_counts', {}) or {}).items():
            self.cell_counts[key] = self.cell_counts.get(key, 0) + int(val)
        for reason in status.get('reasons', []) or []:
            self._add(reason)

    def mark(self, reason: str) -> None:
        self._add(reason)

    def charge_postprocess(self, amount: int = 1) -> bool:
        ok = self.post.charge_postprocess(amount)
        if not ok:
            self._add(REASON_POSTPROCESS_CAP)
        return ok

    @property
    def postprocess_exhausted(self) -> bool:
        return self.post.postprocess_exhausted

    def result_fields(self) -> dict:
        return {
            'complete': not self.reasons,
            'status': {
                'reasons': sorted(self.reasons),
                'work': {
                    'cells_processed': int(self.cells_processed),
                    'csx_calls': int(self.csx_calls),
                    'max_cells': int(self.max_cells),
                    'max_csx_calls': int(self.max_csx_calls),
                    'output_items': int(self.output_items),
                    'max_output_items': int(self.max_output_items),
                    'postprocess_work': int(self.post.postprocess_work),
                    'max_postprocess_work': int(
                        self.post.max_postprocess_work),
                    'cell_counts': dict(self.cell_counts),
                },
            },
        }


def _make_aggregate(kwargs: dict, n_candidates: int) -> _AggregateStatus:
    """Candidate-scaled aggregate ledgers (the `_ncsx4` L41 rule);
    explicit values are absolute aggregate promises."""
    n = max(1, int(n_candidates))
    agg_cells = kwargs.get('max_cells')
    if agg_cells is None:
        agg_cells = _BEZ_DEFAULT_MAX_CELLS * n
    agg_csx = kwargs.get('max_csx_calls')
    if agg_csx is None:
        agg_csx = _BEZ_DEFAULT_MAX_CSX_CALLS * n
    agg_out = kwargs.get('max_output_items')
    if agg_out is None:
        agg_out = _BEZ_DEFAULT_MAX_OUTPUT_ITEMS * n
    agg_cells = max(0, int(agg_cells))
    agg_csx = max(0, int(agg_csx))
    agg_out = max(0, int(agg_out))
    post = SoftWorkBudget(
        max_cells=agg_cells, max_csx_calls=0, max_output_items=0,
        max_postprocess_work=kwargs.get('max_postprocess_work'))
    return _AggregateStatus(
        max_cells=agg_cells, max_csx_calls=agg_csx,
        max_output_items=agg_out, post=post)


def nurbs_ssx(surf1, surf2, atol=1e-3, **kwargs) -> dict:
    _reject_unknown_kwargs("nurbs_ssx", kwargs, _ALLOWED_KWARGS)
    s1 = _as_surface_tuple(surf1)
    s2 = _as_surface_tuple(surf2)
    raise NotImplementedError("pipeline lands in Task 2")
