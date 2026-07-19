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
        return bool(np.allclose(cp[0], cp[-1], rtol=0.0, atol=1e-10)
                    and np.allclose(w[0], w[-1], rtol=0.0, atol=1e-10))
    return bool(np.allclose(cp[:, 0], cp[:, -1], rtol=0.0, atol=1e-10)
                and np.allclose(w[:, 0], w[:, -1], rtol=0.0, atol=1e-10))


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
        d = min(d, max(0.0, float(ctx.spans[axis]) - d))
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
    Not `_adapter_status` (it emits the older CCX/CSX status shape, not
    schema v2) and not a plain SoftWorkBudget (whose counters are
    check-then-charge, not fold-what-pairs-report).
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


# ---------------------------------------------------------------------------
# Per-pair collection (remap + routing)
# ---------------------------------------------------------------------------

@dataclass
class _Frag:
    """A remapped branch fragment awaiting assembly."""
    stuv: NDArray[np.float64]     # (N,4) global
    xyz: NDArray[np.float64]      # (N,3)
    kind: str
    overlap: bool


@dataclass
class _Tile:
    """One per-pair overlap region awaiting unification (Task 5).

    ``loops``: list of loops, each a list of ``(rim_id, reversed)`` where
    ``rim_id`` indexes the shared ``raw.rim_frags`` list.
    """
    pair: tuple
    rect: tuple
    loops: list
    agreement: int
    interior_stuv: NDArray[np.float64]
    certification: dict


@dataclass
class _RawResults:
    frags: list = field(default_factory=list)        # list[_Frag] (non-rim)
    rim_frags: list = field(default_factory=list)    # list[_Frag]
    tiles: list = field(default_factory=list)        # list[_Tile]
    points: list = field(default_factory=list)       # list[SSXPoint], global
    singularities: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)


def _collect_pair(raw: _RawResults, result: dict, rect, pair) -> None:
    """Remap one bez_ssx result into global params and route entities."""
    rim_local = set()
    for region in result.get('overlap_regions', []) or []:
        for loop in region.boundary:
            for idx, _rev in loop:
                rim_local.add(int(idx))

    branches = result.get('branches', []) or []
    rim_map = {}
    for idx in sorted(rim_local):
        b = branches[idx]
        stuv_g = _remap4(np.asarray(b.curve[0], dtype=np.float64), rect)
        xyz = np.array(b.curve[1], dtype=np.float64, copy=True)
        rim_map[idx] = len(raw.rim_frags)
        raw.rim_frags.append(
            _Frag(stuv=stuv_g, xyz=xyz, kind='overlap', overlap=True))

    for idx, b in enumerate(branches):
        if idx in rim_local:
            continue
        stuv_g = _remap4(np.asarray(b.curve[0], dtype=np.float64), rect)
        xyz = np.array(b.curve[1], dtype=np.float64, copy=True)
        raw.frags.append(_Frag(stuv=stuv_g, xyz=xyz,
                               kind=str(b.kind), overlap=bool(b.overlap)))

    for region in result.get('overlap_regions', []) or []:
        loops = [[(rim_map[int(idx)], bool(rev)) for idx, rev in loop]
                 for loop in region.boundary]
        interior = (None if region.interior_stuv is None
                    else _remap4(np.asarray(region.interior_stuv,
                                            dtype=np.float64), rect))
        raw.tiles.append(_Tile(
            pair=pair, rect=rect, loops=loops,
            agreement=int(region.normal_agreement),
            interior_stuv=interior,
            certification=dict(region.certification)))

    for p in result.get('points', []) or []:
        raw.points.append(SSXPoint(
            stuv=_remap4(np.asarray(p.stuv, dtype=np.float64), rect),
            xyz=np.array(p.xyz, dtype=np.float64, copy=True)))

    for s in result.get('singularities', []) or []:
        raw.singularities.append(SSXSingularity(
            kind=str(s.kind),
            stuv=_remap4(np.asarray(s.stuv, dtype=np.float64), rect),
            xyz=np.array(s.xyz, dtype=np.float64, copy=True),
            stuv_mate=(None if s.stuv_mate is None else _remap4(
                np.asarray(s.stuv_mate, dtype=np.float64), rect)),
            branch_links=[],   # recomputed globally in Task 4
            samples=(None if s.samples is None else _remap4(
                np.asarray(s.samples, dtype=np.float64), rect)),
            surface=s.surface))

    for entry in result.get('unresolved_regions', []) or []:
        mapped = dict(entry)
        if 'stuv_min' in mapped:
            mapped['stuv_min'] = tuple(
                float(x) for x in _remap4(
                    np.asarray(mapped['stuv_min'], dtype=np.float64), rect))
        if 'stuv_max' in mapped:
            mapped['stuv_max'] = tuple(
                float(x) for x in _remap4(
                    np.asarray(mapped['stuv_max'], dtype=np.float64), rect))
        raw.unresolved.append(mapped)


def _skip_box(p1: NURBSSurfaceTuple, p2: NURBSSurfaceTuple) -> dict:
    (s0, s1), (t0, t1) = p1.interval()
    (u0, u1), (v0, v1) = p2.interval()
    return {'stuv_min': (float(s0), float(t0), float(u0), float(v0)),
            'stuv_max': (float(s1), float(t1), float(u1), float(v1)),
            'reason': REASON_WORK_BUDGET}


# ---------------------------------------------------------------------------
# Assembly stage stubs (Tasks 3-5 replace these)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Branch assembly: containment dedup -> endpoint graph -> chains
# ---------------------------------------------------------------------------

def _arc_len(xyz) -> float:
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def _bbox_overlap(xyz_a, xyz_b, pad: float) -> bool:
    a = np.asarray(xyz_a, dtype=np.float64)
    b = np.asarray(xyz_b, dtype=np.float64)
    return bool(np.all(a.min(axis=0) - pad <= b.max(axis=0))
                and np.all(b.min(axis=0) - pad <= a.max(axis=0)))


def _containment_dedup(frags, atol, agg):
    """Drop fragments geometrically contained in a longer kept fragment
    (every sample within 2*atol of its polyline — the Bezier-level rule
    applied cross-pair). Longest-first; deterministic tie-break by index.
    On postprocess exhaustion the remaining fragments are kept
    unexamined (honest: dupes possible, reason already recorded)."""
    if len(frags) <= 1:
        return list(frags)
    order = sorted(range(len(frags)),
                   key=lambda k: (-_arc_len(frags[k].xyz), k))
    kept_idx = []
    for k in order:
        f = frags[k]
        dup = False
        if len(f.xyz) >= 1 and not agg.postprocess_exhausted:
            for m in kept_idx:
                g = frags[m]
                if len(g.xyz) < 2:
                    continue
                if not _bbox_overlap(f.xyz, g.xyz, 2.0 * atol):
                    continue
                if not agg.charge_postprocess(max(1, len(f.xyz))):
                    break
                if all(_dist_point_polyline(
                        np.asarray(p, dtype=np.float64), g.xyz)
                        <= 2.0 * atol for p in f.xyz):
                    dup = True
                    break
        if not dup:
            kept_idx.append(k)
    kept_idx.sort()
    return [frags[k] for k in kept_idx]


def _build_chains(frags, ctx, atol, agg, kind_barrier=True):
    """Endpoint-graph chain assembly.

    Endpoints of all fragments are clustered by the matching predicate
    (wrap-aware). A cluster with EXACTLY two endpoint members becomes an
    edge; >2 members is a junction (never chained through); a cluster
    holding both ends of one fragment is a self-loop (closed).

    Returns list of ``(chain, closed)`` where ``chain`` is an ordered
    list of ``(frag_index, flip)``.
    """
    n = len(frags)
    ends = []
    for fi, f in enumerate(frags):
        if len(f.stuv) < 2:
            continue
        ends.append((fi, 0, f.stuv[0], f.xyz[0]))
        ends.append((fi, 1, f.stuv[-1], f.xyz[-1]))

    parent = list(range(len(ends)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    stop = False
    for a in range(len(ends)):
        if stop:
            break
        for b in range(a + 1, len(ends)):
            fa, fb = ends[a][0], ends[b][0]
            if kind_barrier and frags[fa].kind != frags[fb].kind:
                continue
            if not agg.charge_postprocess(1):
                stop = True
                break
            if _match_stuv(ends[a][2], ends[b][2],
                           ends[a][3], ends[b][3], ctx, atol):
                union(a, b)

    clusters = {}
    for e in range(len(ends)):
        clusters.setdefault(find(e), []).append(e)

    # edge per two-member cluster: (fi, end_i) <-> (fj, end_j)
    edge_of = {}       # (fi, end) -> (fj, end_j)
    self_loops = set()
    for members in clusters.values():
        if len(members) != 2:
            continue
        (fi, ei) = ends[members[0]][0], ends[members[0]][1]
        (fj, ej) = ends[members[1]][0], ends[members[1]][1]
        if fi == fj:
            self_loops.add(fi)
            continue
        edge_of[(fi, ei)] = (fj, ej)
        edge_of[(fj, ej)] = (fi, ei)

    visited = [False] * n
    chains = []

    def _walk(start_fi, start_end):
        """Walk from a fragment oriented so ``start_end`` is its FREE end."""
        chain = [(start_fi, start_end == 1)]
        visited[start_fi] = True
        cur_fi, cur_out = start_fi, 1 - start_end
        while True:
            nxt = edge_of.get((cur_fi, cur_out))
            if nxt is None:
                return chain, False
            nfi, nend = nxt
            if visited[nfi]:
                return chain, nfi == start_fi
            chain.append((nfi, nend == 1))
            visited[nfi] = True
            cur_fi, cur_out = nfi, 1 - nend

    # single-fragment closed loops first
    for fi in sorted(self_loops):
        if len(frags[fi].stuv) >= 2 and not visited[fi]:
            visited[fi] = True
            chains.append(([(fi, False)], True))

    # open chains: start at fragments with a free end
    for fi in range(n):
        if visited[fi] or len(frags[fi].stuv) < 2:
            continue
        for end in (0, 1):
            if (fi, end) not in edge_of:
                chain, closed = _walk(fi, end)
                chains.append((chain, closed))
                break

    # remaining unvisited fragments participate in multi-fragment cycles
    for fi in range(n):
        if visited[fi] or len(frags[fi].stuv) < 2:
            continue
        chain, _ = _walk(fi, 0)
        chains.append((chain, True))

    # degenerate (<2 vertex) fragments pass through untouched
    for fi in range(n):
        if not visited[fi] and len(frags[fi].stuv) < 2:
            visited[fi] = True
            chains.append(([(fi, False)], False))
    return chains


def _concat_chain(frags, chain, closed, ctx, atol):
    """Concatenate an oriented chain into one (stuv, xyz) polyline.

    Joint rule: a plain (non-wrap) destructive duplicate collapses to one
    vertex; a wrap-only or gap joint keeps both vertices (the periodic
    vertex-pair contract / honest small gap <= 2*atol).
    Closed chains end with an explicit copy of the first vertex (or the
    wrapped seam preimage pair when the closure crosses a seam).
    """
    stuv_parts, xyz_parts = [], []
    for fi, flip in chain:
        S = frags[fi].stuv[::-1] if flip else frags[fi].stuv
        X = frags[fi].xyz[::-1] if flip else frags[fi].xyz
        if stuv_parts and len(S) and _joint_plain_dup(
                stuv_parts[-1][-1], S[0], xyz_parts[-1][-1], X[0],
                ctx, atol):
            S, X = S[1:], X[1:]
        if len(S):
            stuv_parts.append(np.asarray(S, dtype=np.float64))
            xyz_parts.append(np.asarray(X, dtype=np.float64))
    stuv = np.concatenate(stuv_parts, axis=0)
    xyz = np.concatenate(xyz_parts, axis=0)
    if closed and len(stuv) >= 2 and not _joint_plain_dup(
            stuv[-1], stuv[0], xyz[-1], xyz[0], ctx, atol):
        stuv = np.concatenate([stuv, stuv[:1]], axis=0)
        xyz = np.concatenate([xyz, xyz[:1]], axis=0)
    return stuv, xyz


def _assemble_branches(frags, ctx, atol, agg):
    """Containment dedup -> chain assembly -> SSXBranch list."""
    frags = _containment_dedup(frags, atol, agg)
    chains = _build_chains(frags, ctx, atol, agg, kind_barrier=True)
    out = []
    for chain, closed in sorted(
            chains, key=lambda c: min(fi for fi, _ in c[0])):
        stuv, xyz = _concat_chain(frags, chain, closed, ctx, atol)
        kind = frags[chain[0][0]].kind
        overlap = any(frags[fi].overlap for fi, _ in chain)
        out.append(SSXBranch(curve=(stuv, xyz), closed=bool(closed),
                             overlap=overlap, kind=kind))
    return out


def _assemble_points(points, branches, ctx, atol, agg):
    """Task 4 replaces this with wrap-aware dedup + on-branch filter."""
    return list(points)


def _assemble_singularities(sings, branches, ctx, atol, agg):
    """Task 4 replaces this with cross-pair dedup + link recompute."""
    return list(sings)


def _assemble_regions(raw, stitched, ctx, atol, agg,
                      s_cuts, t_cuts, u_cuts, v_cuts):
    """Task 5 replaces this with seam-rim dissolution + unification.

    Passthrough: rims append after the stitched branches; per-tile
    regions get their refs offset to the final branch list.
    """
    base = len(stitched)
    rim_branches = [SSXBranch(curve=(f.stuv, f.xyz), kind='overlap',
                              overlap=True) for f in raw.rim_frags]
    regions = []
    for tile in raw.tiles:
        loops = [[(base + rid, rev) for rid, rev in loop]
                 for loop in tile.loops]
        uv1_loops, uv2_loops = [], []
        for loop in tile.loops:
            pts = []
            for rid, rev in loop:
                seg = raw.rim_frags[rid].stuv
                pts.append(seg[::-1] if rev else seg)
            chained = np.concatenate(pts, axis=0)
            closed4 = np.concatenate([chained, chained[:1]], axis=0)
            uv1_loops.append(closed4[:, :2].copy())
            uv2_loops.append(closed4[:, 2:].copy())
        regions.append(SSXOverlapRegion(
            boundary=loops, uv1_loops=uv1_loops, uv2_loops=uv2_loops,
            normal_agreement=tile.agreement,
            interior_stuv=tile.interior_stuv,
            certification=dict(tile.certification)))
    return stitched + rim_branches, regions


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def nurbs_ssx(surf1, surf2, atol=1e-3, **kwargs) -> dict:
    """NURBS x NURBS surface intersection over bez_ssx v5.

    See the module docstring for the contract; expert knobs and their
    aggregate semantics are in the spec's "Expert knobs" table.
    """
    _reject_unknown_kwargs("nurbs_ssx", kwargs, _ALLOWED_KWARGS)
    s1 = _as_surface_tuple(surf1)
    s2 = _as_surface_tuple(surf2)
    atol = float(atol)
    rational = _is_rational(s1) or _is_rational(s2)
    ctx = _domain_ctx(s1, s2, atol)

    patches1 = decompose_surface(s1, "uv")
    patches2 = decompose_surface(s2, "uv")

    def _patch_aabb(patch):
        pts = patch.control_points.reshape(
            -1, patch.control_points.shape[-1])
        bb = AABB.from_points(np.asarray(pts, dtype=np.float64))
        bb.offset_inplace(atol)
        return bb

    tree1 = build_bvh([_patch_aabb(p) for p in patches1])
    tree2 = build_bvh([_patch_aabb(p) for p in patches2])
    candidates = sorted(set(
        (int(a.object), int(b.object))
        for a, b in bvh_intersect(tree1, tree2, exact=False)))

    agg = _make_aggregate(kwargs, len(candidates))
    forward = {k: kwargs[k] for k in _FORWARD_KWARGS if k in kwargs}

    raw = _RawResults()
    for k, (i, j) in enumerate(candidates):
        if (agg.remaining_cells <= 0 or agg.remaining_csx_calls <= 0
                or agg.remaining_output_items <= 0):
            agg.mark(REASON_WORK_BUDGET)
            for a, b in candidates[k:]:
                raw.unresolved.append(_skip_box(patches1[a], patches2[b]))
            break
        p1, p2 = patches1[i], patches2[j]
        if rational:
            P1 = to_homogeneous_2d(p1.control_points, p1.weights)
            P2 = to_homogeneous_2d(p2.control_points, p2.weights)
        else:
            P1 = np.ascontiguousarray(p1.control_points, dtype=np.float64)
            P2 = np.ascontiguousarray(p2.control_points, dtype=np.float64)
        result = bez_ssx(
            P1, P2, atol=atol, rational=rational,
            max_cells=min(_BEZ_DEFAULT_MAX_CELLS, agg.remaining_cells),
            max_csx_calls=min(_BEZ_DEFAULT_MAX_CSX_CALLS,
                              agg.remaining_csx_calls),
            max_output_items=min(_BEZ_DEFAULT_MAX_OUTPUT_ITEMS,
                                 agg.remaining_output_items),
            **forward)
        agg.consume(result)
        _collect_pair(raw, result, _pair_rect(p1, p2), pair=(i, j))

    # Interior decomposition cut coordinates per stuv axis (for Task 5's
    # seam-rim classification).
    def _cuts(patches, side):
        vals = set()
        for p in patches:
            (a0, a1), (b0, b1) = p.interval()
            vals.update((a0, a1) if side == 0 else (b0, b1))
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 1.0
        return tuple(sorted(v for v in vals if lo < v < hi))

    s_cuts, t_cuts = _cuts(patches1, 0), _cuts(patches1, 1)
    u_cuts, v_cuts = _cuts(patches2, 0), _cuts(patches2, 1)

    stitched = _assemble_branches(raw.frags, ctx, atol, agg)
    branches, regions = _assemble_regions(
        raw, stitched, ctx, atol, agg, s_cuts, t_cuts, u_cuts, v_cuts)
    points = _assemble_points(raw.points, branches, ctx, atol, agg)
    singularities = _assemble_singularities(
        raw.singularities, branches, ctx, atol, agg)

    out = {
        'branches': branches,
        'points': points,
        'singularities': singularities,
        'overlap_regions': regions,
        'unresolved_regions': raw.unresolved,
    }
    out.update(agg.result_fields())
    return out
