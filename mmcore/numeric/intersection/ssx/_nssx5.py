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

Unified overlap regions guarantee loop ORDERING (outer first) but not
winding direction on the multi-tile path (see ``_assemble_regions``).
Assembly preserves every unresolved reason reported by the pair solvers:
two UV footprints do not establish exhaustive lifted correspondence.

Incompleteness is always soft: certified partial output is returned with
``complete=False`` and typed ``status['reasons']``; exceptions are for
caller errors only. Source control coordinates must be finite and weights
must be finite and strictly positive; unsupported data raises ``ValueError``
before decomposition or control-hull exclusions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mmcore.nurbs import _core as nurbs
from mmcore.nurbs._nurbs_eval import (
    NURBSSurfaceTuple, _nurbs_to_tuple, to_homogeneous_2d,
    evaluate_nurbs_surface,
)
from mmcore.nurbs._nurbs_knots import decompose_surface
from mmcore.nurbs._nurbs_param_tol import nurbs_surface_param_tolerance
from mmcore.numeric.bvh.lbvh import AABB, build_bvh, bvh_intersect
from mmcore.numeric._work_budget import (
    SoftWorkBudget,
    REASON_WORK_BUDGET,
    REASON_POSTPROCESS_CAP,
    REASON_MULTIPLICITY,
    REASON_PARAMETER_REPRESENTATION,
)
from mmcore.numeric.intersection.ssx._bez_ssx5 import (
    bez_ssx, SSXSingularity, _dist_point_polyline,
)
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch, SSXPoint
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
                   'boundary_csx_max_cells', 'csx_max_results',
                   'csx_max_depth')
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
        f"mmcore.nurbs._core.NURBSSurface, not {type(surf).__name__}")


def _is_rational(surf: NURBSSurfaceTuple) -> bool:
    # Weights are representation data, not measured geometry. Even an
    # arbitrarily small nonuniform perturbation can split a double root.
    return not bool(np.all(np.asarray(surf.weights) == 1.0))


def _axis_closed(surf: NURBSSurfaceTuple, axis: int) -> bool:
    """Sufficient exact C0 seam test for a clamped parameter axis.

    Equal end control rows describe equal boundary curves only when the
    end knots are clamped. An unsupported unclamped seam stays separate.
    """
    order = surf.order_u if axis == 0 else surf.order_v
    knots = np.asarray(surf.knot_u if axis == 0 else surf.knot_v)
    low, high = surf.interval()[axis]
    if not (np.all(knots[:order] == low) and
            np.all(knots[-order:] == high)):
        return False
    cp, w = surf.control_points, surf.weights
    if axis == 0:
        return bool(np.array_equal(cp[0], cp[-1])
                    and np.array_equal(w[0], w[-1]))
    return bool(np.array_equal(cp[:, 0], cp[:, -1])
                and np.array_equal(w[:, 0], w[:, -1]))


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
    source_patches: object = None


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
        d = min(d, abs(float(ctx.spans[axis]) - d))
    return d


def _axis_diff_nowrap(a: float, b: float) -> float:
    return abs(float(a) - float(b))


def _same_axis_preimage(a,b,axis,ctx):
    """Exact coordinate identity, including only actual opposite seam ends."""
    return bool(a == b or (ctx.closed[axis] and (
        (a == ctx.lows[axis] and b == ctx.highs[axis])
        or (b == ctx.lows[axis] and a == ctx.highs[axis]))))


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
    """Identical paired preimages, modulo proven seams, with an xyz guard.

    Modeling tolerance bounds approximation error, not separation of
    distinct isolated roots. In the absence of a shared root certificate,
    retain numerically nearby preimages as distinct output points.
    """
    d = np.asarray(xyz_p, dtype=np.float64) - np.asarray(
        xyz_q, dtype=np.float64)
    if float(np.linalg.norm(d)) > atol:
        return False
    return all(_same_axis_preimage(p[i], q[i], i, ctx)
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
    from mmcore.numeric.intersection._parameter_mapping import affine_parameters
    x = np.array(stuv_local, dtype=np.float64, copy=True)
    bounds = np.asarray(rect).reshape(4, 2)
    if np.all(bounds[:, 0] == 0.) and np.all(bounds[:, 1] == 1.):
        return x
    return np.array([affine_parameters(row, bounds)[0]
                     for row in x.reshape(-1, 4)]).reshape(x.shape)


def _remap4_bound(stuv_local, rect, lower):
    """Outward affine mapping for diagnostic enclosures, not representatives."""
    from fractions import Fraction
    from mmcore.numeric.intersection._parameter_mapping import affine_parameters
    mapped, exact = affine_parameters(stuv_local, np.asarray(rect).reshape(4, 2))
    result = []
    for value, target in zip(mapped, exact):
        rounded = Fraction.from_float(value)
        if (lower and rounded > target) or (not lower and rounded < target):
            value = np.nextafter(value, -np.inf if lower else np.inf)
        result.append(float(value))
    return tuple(result)


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
    # Did the CALLER set this ledger, or is it the candidate-scaled default?
    # An explicit value is an absolute aggregate promise and is redistributed
    # across the remaining pairs; a default is a per-pair fairness share.
    explicit_cells: bool = False
    explicit_csx: bool = False
    explicit_output: bool = False

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
        max_output_items=agg_out, post=post,
        explicit_cells=kwargs.get('max_cells') is not None,
        explicit_csx=kwargs.get('max_csx_calls') is not None,
        explicit_output=kwargs.get('max_output_items') is not None)


def _per_pair_allowance(agg, remaining_candidates):
    """Per-pair grant from the aggregate ledgers.

    P2 (2026-07-25): the per-pair values used to be
    ``min(_BEZ_DEFAULT_MAX_*, remaining)`` — the module default acting as a
    hard ceiling on every call.  With one candidate pair that made the
    public knobs unreachable: an explicit ``max_cells=2_000_000`` still
    handed the engine 250k, which then reported ``work_budget`` and invited
    the caller to raise a knob that could not move.

    An EXPLICIT aggregate is an absolute promise (`_make_aggregate`; ledger
    L41), and the house reading of that promise is the reference adapters':
    `_ncsx4` and `_nccx4` both hand each call the ENTIRE remainder
    (``call_kwargs['max_cells'] = remaining_cells``).  Do the same here.

    An even fair-share slice was tried first and REVERTED (review
    2026-07-26): work is not spread evenly over BVH candidates, so slicing
    starves the hot pair.  Measured on harness case 1 with 43 candidates,
    ``max_cells=250_000`` went from ``complete=True, reasons=[]`` to
    ``work_budget`` with 61% of the caller's explicit aggregate unspent —
    reintroducing, on the explicit path, exactly the misbilled
    knob-unreachability this change exists to remove.

    The DEFAULT path keeps the module default as its per-pair share, which
    is bit-identical to the pre-P2 expression.  Note this is a real
    discontinuity: passing ``max_cells=default*n`` is not the same call as
    omitting it, because the former is a promise the caller may concentrate
    on one pair.  That is the documented meaning of "absolute", not an
    accident of the arithmetic.
    """
    def share(remaining, default, explicit):
        return remaining if explicit else min(default, remaining)

    return (
        share(agg.remaining_cells, _BEZ_DEFAULT_MAX_CELLS,
              agg.explicit_cells),
        share(agg.remaining_csx_calls, _BEZ_DEFAULT_MAX_CSX_CALLS,
              agg.explicit_csx),
        share(agg.remaining_output_items, _BEZ_DEFAULT_MAX_OUTPUT_ITEMS,
              agg.explicit_output),
    )


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
    pair: object = None
    rect: object = None
    endpoint_ids: object = None
    source_path: object = None
    source_boundary_face: object = None
    source_endpoints: object = None


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
    # global rects of pairs whose status carried unresolved_multiplicity
    # retained as diagnostics; projected footprints cannot retire ambiguity.
    mult_rects: list = field(default_factory=list)
    # Delay routing checked pairs until all cross-pair parameter aliases
    # are known; otherwise withdrawing one tile would invalidate rim IDs.
    staged_pairs: dict = field(default_factory=dict)
    parameter_identities: dict = field(default_factory=dict)


def _pair_parameter_samples(result):
    for index, branch in enumerate(result.get('branches', []) or []):
        for vertex, (stuv, xyz) in enumerate(zip(*branch.curve)):
            yield ('branch', index, vertex), stuv, xyz
    for index, point in enumerate(result.get('points', []) or []):
        yield ('point', index), point.stuv, point.xyz
    for index, singularity in enumerate(result.get('singularities', []) or []):
        yield ('singularity', index), singularity.stuv, singularity.xyz
        if singularity.stuv_mate is not None:
            yield ('singularity_mate', index), singularity.stuv_mate, singularity.xyz
        if singularity.samples is not None:
            for sample, stuv in enumerate(np.asarray(singularity.samples).reshape(-1, 4)):
                yield ('singularity_sample', index, sample), stuv, None
    for index, region in enumerate(result.get('overlap_regions', []) or []):
        if region.interior_stuv is not None:
            yield ('region_interior', index), region.interior_stuv, None


def _stage_representable_pair(raw, result, rect, pair, sources, rational, atol, agg):
    from fractions import Fraction
    from mmcore.numeric.intersection._parameter_mapping import (
        affine_parameters, map_isolated, mapping_issue,
        reject_parameter_aliases, _exact_evaluate,
    )
    bounds = tuple(map(tuple, np.asarray(rect).reshape(4, 2)))
    context = f'nurbs_ssx patches {pair}'
    status = {'complete': True, 'boundary_topology_complete': True, 'partial_results': 0}
    record = {'result': result, 'rect': rect, 'issues': []}
    raw.staged_pairs[pair] = record
    entries = []
    evaluation_cost = max(1, (3*sum(np.asarray(net).size for net in sources)+127)//128)
    for entity, stuv, xyz in _pair_parameter_samples(result):
        if not agg.charge_postprocess(1):
            record['reason'] = REASON_POSTPROCESS_CAP
            record['issues'].append({'reason': 'parameter mapping budget exhausted'})
            break
        global_float, global_exact = affine_parameters(stuv, bounds)
        changed = any(Fraction.from_float(value) != exact
                      for value, exact in zip(global_float, global_exact))
        if changed:
            if not agg.charge_postprocess(evaluation_cost):
                record['reason'] = REASON_POSTPROCESS_CAP
                record['issues'].append({'reason': 'parameter source-evaluation budget exhausted'})
                break
            if xyz is None:
                try:
                    local = tuple(Fraction.from_float(float(t)) for t in stuv[:2])
                    xyz = tuple(float(x) for x in _exact_evaluate(sources[0], local, rational))
                except (OverflowError, ValueError, ZeroDivisionError):
                    mapping_issue(status, context, True, {
                        'entity': entity, 'local_parameters': tuple(stuv),
                        'parameter_bounds': bounds,
                        'exact_global_parameters': tuple(str(t) for t in global_exact),
                    }, 'source evaluation cannot represent the local anchor')
                    continue
        entry = dict(zip(('s', 't', 'u', 'v'), stuv), entity=entity)
        if xyz is not None:
            entry['point'] = xyz
        mapped = map_isolated(
            entry, ('s', 't', 'u', 'v'), bounds,
            ((sources[0], (0, 1)), (sources[1], (2, 3))),
            rational, atol, status, context, True)
        if mapped is not None:
            entries.append(mapped)
    reject_parameter_aliases(entries, ('s', 't', 'u', 'v'), status, context, True)
    record['issues'].extend(status.get('unrepresentable_parameters', []))
    parameter_issue = bool(status['partial_results'])
    for entry in entries:
        key = tuple(entry[k] for k in ('s', 't', 'u', 'v'))
        exact, payload = entry['_exact_global_parameters'], entry['_local_parameter_payload']
        previous = raw.parameter_identities.get(key)
        if previous is not None and previous[0] != exact:
            issue = {'reason': 'distinct local solutions share global float parameters',
                     'local_solutions': [previous[2], payload]}
            record['issues'].append(issue)
            raw.staged_pairs[previous[1]]['issues'].append(issue)
            parameter_issue = True
        else:
            raw.parameter_identities[key] = (exact, pair, payload)
    if parameter_issue:
        agg.mark(REASON_PARAMETER_REPRESENTATION)


def _finish_collecting(raw):
    """Route whole valid pairs; retain local geometry for every rejected pair."""
    for pair, record in raw.staged_pairs.items():
        result, rect = record['result'], record['rect']
        if record['issues']:
            raw.unresolved.append({
                'stuv_min': tuple(rect[::2]), 'stuv_max': tuple(rect[1::2]),
                'reason': record.get('reason', REASON_PARAMETER_REPRESENTATION),
                'pair': pair, 'local_result': result, 'parameter_bounds': rect,
                'mapping_issues': record['issues'],
            })
        else:
            _collect_pair(raw, result, rect, pair)
    raw.staged_pairs.clear()
    raw.parameter_identities.clear()


def _collect_pair(raw: _RawResults, result: dict, rect, pair, *,
                  sources=None, rational=False, atol=None, agg=None) -> None:
    """Remap one bez_ssx result into global params and route entities."""
    if sources is not None:
        _stage_representable_pair(raw, result, rect, pair, sources, rational, atol, agg)
        return
    from mmcore.numeric.intersection.ssx._ssx_arc_ownership import map_source_path
    bounds = np.asarray(rect).reshape(4,2)
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
            _Frag(stuv=stuv_g, xyz=xyz, kind='overlap', overlap=True,
                  pair=pair,rect=rect,
                  source_path=map_source_path(getattr(b,'_source_parameter_path',None),bounds),
                  source_boundary_face=getattr(b,'_source_boundary_face',None)))

    for idx, b in enumerate(branches):
        if idx in rim_local:
            continue
        stuv_g = _remap4(np.asarray(b.curve[0], dtype=np.float64), rect)
        xyz = np.array(b.curve[1], dtype=np.float64, copy=True)
        raw.frags.append(_Frag(stuv=stuv_g, xyz=xyz,
                               kind=str(b.kind), overlap=bool(b.overlap),
                               pair=pair, rect=rect,
                               source_path=map_source_path(getattr(b,'_source_parameter_path',None),bounds),
                               source_boundary_face=getattr(b,'_source_boundary_face',None),
                               source_endpoints=getattr(b,'_registered_endpoints',None)))

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
        mapped = SSXPoint(
            stuv=_remap4(np.asarray(p.stuv, dtype=np.float64), rect),
            xyz=np.array(p.xyz, dtype=np.float64, copy=True))
        root = getattr(p,'_registered_root',None)
        if root is not None:
            mapped._source_point_owner = (pair,root)
        raw.points.append(mapped)

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
        if 'candidate' in mapped:
            from mmcore.numeric.intersection._parameter_mapping import affine_parameters
            # An unproved source-root proposal is diagnostic data. Keep its
            # local coordinates explicit and preserve the exact affine map;
            # it must not pass through the published-root representation gate.
            candidate = tuple(mapped.pop('candidate'))
            bounds = tuple(map(tuple, np.asarray(rect).reshape(4, 2)))
            mapped['local_candidate'] = candidate
            mapped['candidate_parameter_bounds'] = bounds
            _, exact = affine_parameters(candidate, bounds)
            mapped['exact_global_candidate'] = tuple(str(value) for value in exact)
        if 'stuv_min' in mapped:
            mapped['stuv_min'] = _remap4_bound(mapped['stuv_min'], rect, lower=True)
        if 'stuv_max' in mapped:
            mapped['stuv_max'] = _remap4_bound(mapped['stuv_max'], rect, lower=False)
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


def _seam_segment_mask(stuv, xyz, ctx, atol):
    """Exclude explicit periodic vertex pairs from linear interpolation."""
    s = np.asarray(stuv, dtype=float)
    x = np.asarray(xyz, dtype=float)
    mask = np.ones(max(0, len(s) - 1), dtype=bool)
    if ctx is None:
        return mask
    for i in range(len(mask)):
        wraps = any(ctx.closed[a] and ctx.spans[a] > 0.0
                    and abs(s[i + 1, a] - s[i, a]) > .5 * ctx.spans[a]
                    for a in range(4))
        if wraps and _match_stuv(s[i], s[i + 1], x[i], x[i + 1], ctx, atol):
            mask[i] = False
    return mask


def _certified_boundary_retrace(frag, keeper, ctx, atol, agg):
    """Coalesce ordinary tracing with a proved boundary-overlap owner."""
    if (frag.source_boundary_face is None or keeper.source_boundary_face is None
            or ctx is None or ctx.source_patches is None
            or frag.pair is None or keeper.pair is None
            or frag.rect is None or keeper.rect is None):
        return False
    from mmcore.numeric.intersection.ssx._ssx_boundary_identity import certified_boundary_retrace
    fa, ka = np.asarray(frag.rect).reshape(4, 2), np.asarray(keeper.rect).reshape(4, 2)
    for owner in (0, 1):
        if frag.pair[1-owner] != keeper.pair[1-owner]:
            continue
        for axis in (2*owner, 2*owner+1):
            value = frag.stuv[0, axis]
            if (not np.all(frag.stuv[:, axis] == value)
                    or not np.all(keeper.stuv[:, axis] == value)):
                continue
            sides_f = np.flatnonzero(fa[axis] == value)
            sides_k = np.flatnonzero(ka[axis] == value)
            curve_axis = 2*owner+(1-axis % 2)
            if (len(sides_f) != 1 or len(sides_k) != 1
                    or not np.array_equal(fa[curve_axis], ka[curve_axis])):
                continue
            if (frag.source_boundary_face != (axis,sides_f[0])
                    or keeper.source_boundary_face != (axis,sides_k[0])):
                continue
            source_f = ctx.source_patches[owner][frag.pair[owner]]
            source_k = ctx.source_patches[owner][keeper.pair[owner]]
            target = ctx.source_patches[1-owner][frag.pair[1-owner]]
            net_f = to_homogeneous_2d(source_f.control_points, source_f.weights)
            net_k = to_homogeneous_2d(source_k.control_points, source_k.weights)
            curve = np.take(net_f, 0 if sides_f[0] == 0 else -1, axis=axis % 2)
            other_curve = np.take(net_k, 0 if sides_k[0] == 0 else -1, axis=axis % 2)
            if not np.array_equal(curve, other_curve):
                continue
            surface = to_homogeneous_2d(target.control_points, target.weights)
            relation = certified_boundary_retrace(
                frag.stuv, frag.xyz, keeper.stuv, keeper.xyz, curve, surface,
                curve_axis, (2*(1-owner), 2*(1-owner)+1), fa[curve_axis],
                4.*ctx.ptol, 2.*atol,
                charge=lambda n: agg.charge_postprocess(max(1, (n+127)//128)))
            if relation is None or relation:
                return relation
    return False


def _containment_dedup(frags, atol, agg, ctx=None):
    """Coalesce only proved source-arc duplicates, longest first.

    Distinct arcs can share endpoints and identical approximation chords.
    Shared fragment provenance or exact shared boundary-curve ownership
    with a unique target preimage is required before deleting a fragment.
    On postprocess exhaustion the remaining fragments are kept
    unexamined (honest: dupes possible, reason already recorded)."""
    if len(frags) <= 1:
        return list(frags)
    order = sorted(range(len(frags)), key=lambda k: (
        frags[k].kind != 'overlap', -_arc_len(frags[k].xyz), k))
    kept_idx = []
    seen_objects = set()
    for position,k in enumerate(order):
        if not agg.charge_postprocess(1):
            kept_idx.extend(order[position:])
            break
        f = frags[k]
        if id(f) in seen_objects:
            continue
        seen_objects.add(id(f))
        if f.source_path is None and f.source_boundary_face is None:
            kept_idx.append(k)
            continue
        dup = False
        if len(f.xyz) >= 1 and not agg.postprocess_exhausted:
            for m in kept_idx:
                g = frags[m]
                if g.source_path is None and g.source_boundary_face is None:
                    continue
                if not agg.charge_postprocess(1):
                    break
                if len(g.xyz) < 2:
                    continue
                if not _bbox_overlap(f.xyz, g.xyz, 2.0 * atol):
                    continue
                if ((f.kind != g.kind or f.overlap != g.overlap)
                        and (f.kind != 'transversal' or g.kind != 'overlap')):
                    continue
                from mmcore.numeric.intersection.ssx._ssx_arc_ownership import source_path_covered
                contained = source_path_covered(f.source_path,[g.source_path],agg.charge_postprocess)
                if contained is False:
                    contained = _certified_boundary_retrace(f, g, ctx, atol, agg)
                if contained is None:
                    break
                if contained:
                    dup = True
                    break
        if not dup:
            kept_idx.append(k)
    kept_idx.sort()
    return [frags[k] for k in kept_idx]


def _seam_box_has_root(net, box, source_scale):
    """Sufficient Krawczyk inclusion, independent of numerical residual size.

    The caller separately certifies uniqueness. Restriction maps the box
    to the unit cube; a fixed preconditioned Newton map sends that cube
    strictly into itself. Unsupported boundary roots remain separate.
    The floating coefficient/operation enclosures follow the shared root
    certificate's arithmetic model, not formal arbitrary-precision proof.
    """
    from mmcore.numeric.bern import (
        bernstein_eval_nd, bernstein_partial_derivative_coeffs,
    )
    from mmcore.numeric._bezier_common import restrict_net_axis_v
    from mmcore.numeric.intersection._root_box_certificate import residual_roundoff_bound

    restricted = net
    for axis, (lo, hi) in enumerate(box):
        restricted = restrict_net_axis_v(restricted, axis, lo, hi, 0., 1.)
    # Each axis restriction can perform two de Casteljau subdivisions.
    error = residual_roundoff_bound(net, depth=2*len(box), source_scale=source_scale)
    axes = tuple(range(len(box)))
    magnitude = np.max(np.abs(restricted), axis=axes)
    eps = np.finfo(float).eps
    lower, upper = [], []
    for axis in axes:
        degree = restricted.shape[axis]-1
        derivative = bernstein_partial_derivative_coeffs(restricted, axis=axis)
        derivative_error = degree*(2.*error + 4.*eps*magnitude)
        lower.append(np.nextafter(derivative.min(axis=axes)-derivative_error, -np.inf))
        upper.append(np.nextafter(derivative.max(axis=axes)+derivative_error, np.inf))
    lower, upper = np.asarray(lower).T, np.asarray(upper).T
    midpoint, radius = .5*(lower+upper), .5*(upper-lower)
    try:
        inverse = np.linalg.inv(midpoint)
    except np.linalg.LinAlgError:
        return False
    if not np.all(np.isfinite(inverse)):
        return False
    value = bernstein_eval_nd(restricted, np.full(len(box), .5))
    gamma = (2*len(box)+2)*eps / (1.-(2*len(box)+2)*eps)
    absolute_inverse = np.abs(inverse)
    arithmetic = gamma*(np.eye(len(box))+absolute_inverse@(np.abs(midpoint)+radius))
    linear_radius = .5*np.sum(
        np.abs(np.eye(len(box))-inverse@midpoint)+absolute_inverse@radius+arithmetic,
        axis=1)
    correction = (np.abs(inverse@value)+absolute_inverse@error
                  + gamma*absolute_inverse@(np.abs(value)+error))
    return bool(np.all(np.nextafter(correction+linear_radius, np.inf) < .5))


def _same_certified_seam_root(fa, pa, fb, pb, ctx, agg):
    """Identify one transverse CSX event shared by adjacent source patches.

    Proximity only selects the candidate box. Exact adjacency, identical
    seam coefficients, interval-Jacobian uniqueness and root inclusion
    authorize the union. Existing branch endpoints are numerical samples
    of that event; the test does not promote arbitrary low residuals into
    roots. Singular/tangent events and missing provenance stay separate.
    """
    if (ctx.source_patches is None or fa.pair is None or fb.pair is None
            or fa.rect is None or fb.rect is None):
        return False
    changed = [owner for owner in (0, 1) if fa.pair[owner] != fb.pair[owner]]
    if len(changed) != 1:
        return False
    owner = changed[0]
    ra, rb = np.asarray(fa.rect).reshape(4, 2), np.asarray(fb.rect).reshape(4, 2)
    for axis in (2*owner, 2*owner+1):
        if ra[axis, 1] == rb[axis, 0]:
            cut, side_a, side_b = ra[axis, 1], -1, 0
        elif rb[axis, 1] == ra[axis, 0]:
            cut, side_a, side_b = ra[axis, 0], 0, -1
        else:
            continue
        other_axes = [i for i in range(4) if i != axis]
        if (pa[axis] != cut or pb[axis] != cut
                or not np.array_equal(ra[other_axes], rb[other_axes])):
            continue
        a = ctx.source_patches[owner][fa.pair[owner]]
        b = ctx.source_patches[owner][fb.pair[owner]]
        surface = ctx.source_patches[1-owner][fa.pair[1-owner]]
        ah = to_homogeneous_2d(a.control_points, a.weights)
        bh = to_homogeneous_2d(b.control_points, b.weights)
        curve = np.take(ah, side_a, axis=axis % 2)
        if not np.array_equal(curve, np.take(bh, side_b, axis=axis % 2)):
            continue
        sh = to_homogeneous_2d(surface.control_points, surface.weights)
        if (np.any(curve[:, -1] <= 0.) or np.any(sh[..., -1] <= 0.)
                or not np.all(np.isfinite(curve)) or not np.all(np.isfinite(sh))):
            continue
        parameter_axes = [2*owner + (1-axis % 2), 2*(1-owner), 2*(1-owner)+1]
        lows = ra[parameter_axes, 0]
        spans = ra[parameter_axes, 1]-lows
        if np.any(spans <= 0.):
            continue
        candidates = (np.asarray([pa, pb])[:, parameter_axes]-lows)/spans
        center = candidates.mean(axis=0)
        radii = .5*np.ptp(candidates, axis=0)+ctx.ptol[parameter_axes]/spans
        if not agg.charge_postprocess(max(1, len(curve)*sh.shape[0]*sh.shape[1])):
            return False
        from mmcore.numeric.intersection.csx._bez_csx4 import (
            _residual_vec_net, _csx_residual_source_scale,
        )
        from mmcore.numeric.intersection._root_box_certificate import unique_root_box
        net = _residual_vec_net(curve, sh, rational=True)
        source_scale = _csx_residual_source_scale(curve, sh, rational=True)
        box = unique_root_box(net, center, radii, source_scale)
        if box is None or not all(
                all(lo <= value <= hi for value, (lo, hi) in zip(candidate, box))
                for candidate in candidates):
            continue
        if _seam_box_has_root(net, box, source_scale):
            return True
    return False


def _build_chains(frags, ctx, atol, agg, kind_barrier=True):
    """Endpoint-graph chain assembly.

    Endpoints of all fragments are clustered by the matching predicate
    (wrap-aware). A cluster with EXACTLY two endpoint members becomes an
    edge; >2 members is a junction (never chained through); a cluster
    holding both ends of one fragment is a self-loop (closed).

    Returns list of ``(chain, closed)`` where ``chain`` is an ordered
    list of ``(frag_index, flip)``.

    On postprocess-cap exhaustion mid-pairing, already-made unions keep
    their edges while untested pairs stay unmatched — stitching may then
    be incomplete or (at a junction) wrong; the recorded
    REASON_POSTPROCESS_CAP marks the result partial, mirroring
    ``_containment_dedup``'s honesty rule.
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
                # Closeness is a candidate search, not root identity.
                # Distinct parallel components can have both endpoint
                # pairs within tolerance; joining those pairs invents a
                # closed loop and deletes one of the components.
                ids_a, ids_b = frags[fa].endpoint_ids, frags[fb].endpoint_ids
                same_vertex = (ids_a is not None and ids_b is not None
                               and ids_a[ends[a][1]] == ids_b[ends[b][1]])
                source_a,source_b = frags[fa].source_endpoints,frags[fb].source_endpoints
                root_a = None if source_a is None else source_a[ends[a][1]]
                root_b = None if source_b is None else source_b[ends[b][1]]
                if (frags[fa].pair == frags[fb].pair
                        and (root_a is not None or root_b is not None)):
                    if root_a is not None and root_a is root_b:
                        union(a,b)
                    # The pair solver already resolved admissible source
                    # identities. Float aliases cannot undo that decision.
                    continue
                if same_vertex or all(
                        _same_axis_preimage(ends[a][2][i],ends[b][2][i],i,ctx)
                        for i in range(4)) or _same_certified_seam_root(
                            frags[fa], ends[a][2], frags[fb], ends[b][2], ctx, agg):
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
    if not stuv_parts:
        return (np.zeros((0, 4), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64))
    stuv = np.concatenate(stuv_parts, axis=0)
    xyz = np.concatenate(xyz_parts, axis=0)
    if closed and len(stuv) >= 2 and not _joint_plain_dup(
            stuv[-1], stuv[0], xyz[-1], xyz[0], ctx, atol):
        stuv = np.concatenate([stuv, stuv[:1]], axis=0)
        xyz = np.concatenate([xyz, xyz[:1]], axis=0)
    return stuv, xyz


def _assemble_branches(frags, ctx, atol, agg):
    """Containment dedup -> chain assembly -> SSXBranch list."""
    frags = _containment_dedup(frags, atol, agg, ctx=ctx)
    chains = _build_chains(frags, ctx, atol, agg, kind_barrier=True)
    out = []
    for chain, closed in sorted(
            chains, key=lambda c: min(fi for fi, _ in c[0])):
        stuv, xyz = _concat_chain(frags, chain, closed, ctx, atol)
        if len(stuv) == 0:
            continue
        kind = frags[chain[0][0]].kind
        overlap = any(frags[fi].overlap for fi, _ in chain)
        branch = SSXBranch(curve=(stuv,xyz),closed=bool(closed),overlap=overlap,kind=kind)
        owners = tuple(frags[fi].source_path for fi,_ in chain)
        if all(path is not None for path in owners):
            branch._source_parameter_paths = owners
        out.append(branch)
    return out


# ---------------------------------------------------------------------------
# Points: wrap-aware dedup; source incidence is resolved before remapping
# ---------------------------------------------------------------------------

def _assemble_points(points, branches, ctx, atol, agg):
    """Coalesce equal point preimages while preserving unknown arc incidence.

    A point on a branch's approximation chord can still be a distinct
    isolated source root. The mapped branches carry no complete source-arc
    ownership certificate, so geometric containment cannot delete it.
    If the postprocess cap fires
    mid-dedup, the unexamined remainder passes through undropped (honest:
    duplicates possible, REASON_POSTPROCESS_CAP already recorded).
    Membership is by object identity — SSXPoint holds ndarrays, so
    equality-based membership would raise."""
    kept = []
    for p in points:
        dup = False
        for q in kept:
            if not agg.charge_postprocess(1):
                break
            owner_p = getattr(p,'_source_point_owner',None)
            owner_q = getattr(q,'_source_point_owner',None)
            if (owner_p is not None and owner_q is not None
                    and owner_p[0] == owner_q[0] and owner_p[1] is not owner_q[1]):
                continue
            if _dup_stuv(p.stuv, q.stuv, p.xyz, q.xyz, ctx, atol):
                dup = True
                break
        if not dup:
            kept.append(p)

    if agg.postprocess_exhausted:
        kept_ids = {id(p) for p in kept}
        pool = kept + [p for p in points if id(p) not in kept_ids]
    else:
        pool = kept

    return pool


# ---------------------------------------------------------------------------
# Singularities: cross-pair dedup + branch_links recompute (L11/L12)
# ---------------------------------------------------------------------------

def _clouds_near_identical(a, b, s1, atol, agg):
    """cusp_curve near-duplicate: every sample of the smaller cloud lies
    within 2*atol (xyz, via surf1 evaluation) of some sample of the
    larger."""
    if a is None or b is None:
        return a is None and b is None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    if len(small) == 0 or len(large) == 0:
        return len(small) == len(large)
    if not agg.charge_postprocess(len(small) + len(large)):
        return False
    def _xyz(cloud):
        return np.array([
            np.asarray(evaluate_nurbs_surface(
                s1, float(x[0]), float(x[1]), d_order=0)['S'],
                dtype=np.float64)
            for x in cloud])
    xs, xl = _xyz(small), _xyz(large)
    for p in xs:
        if float(np.min(np.linalg.norm(xl - p[None, :], axis=1))) \
                > 2.0 * atol:
            return False
    return True


def _recompute_branch_links(target_xyz, branches, atol, agg):
    """L11 vertex contract via L12 point-to-SEGMENT distance (mirrors the
    inline linker in _bez_ssx5's C1 pass): a branch links when its
    polyline passes within 4*atol; the link anchors at the nearer
    endpoint of the nearest segment."""
    links = []
    cost = sum(max(1, len(np.asarray(b.curve[1])) - 1) for b in branches)
    if not agg.charge_postprocess(max(1, cost)):
        return links
    target = np.asarray(target_xyz, dtype=np.float64)
    for bi, b in enumerate(branches):
        xyz = np.asarray(b.curve[1], dtype=np.float64)
        if len(xyz) < 2:
            continue
        if _dist_point_polyline(target, xyz) > 4.0 * atol:
            continue
        a, bseg = xyz[:-1], xyz[1:]
        ab = bseg - a
        den = np.einsum("ij,ij->i", ab, ab)
        den = np.where(den < 1e-30, 1e-30, den)
        tt = np.clip(
            np.einsum("ij,ij->i", target[None, :] - a, ab) / den,
            0.0, 1.0)
        dseg = np.linalg.norm(a + tt[:, None] * ab - target[None, :],
                              axis=1)
        kseg = int(dseg.argmin())
        k = (kseg if np.linalg.norm(xyz[kseg] - target)
             <= np.linalg.norm(xyz[kseg + 1] - target) else kseg + 1)
        links.append((bi, k))
    return links


def _mate_matches(a, b, ctx):
    """Mate-preimage match for self_intersection dedup: parametric-only
    (the mate shares the primary's xyz by construction, so xyz would not
    discriminate)."""
    if a is None or b is None:
        return a is None and b is None
    return all(_same_axis_preimage(a[i],b[i],i,ctx)
               for i in range(4))


def _assemble_singularities(sings, branches, ctx, atol, agg, s1=None):
    """Cross-pair singularity dedup (per kind, keep-first) + L11
    branch_links recompute against the final branch list."""
    kept = []
    for s in sings:
        dup = False
        for q in kept:
            if q.kind != s.kind or q.surface != s.surface:
                continue
            if not agg.charge_postprocess(1):
                break
            if s.kind == 'cusp_curve':
                # Sample clouds do not identify their complete source
                # singular strata, even when the sampled tuples coincide.
                dup = s is q
            else:
                if not _dup_stuv(s.stuv, q.stuv, s.xyz, q.xyz,
                                   ctx, atol):
                    continue
                if (s.kind == 'self_intersection'
                        and not _mate_matches(s.stuv_mate, q.stuv_mate,
                                              ctx)):
                    continue
                dup = True
            if dup:
                break
        if not dup:
            kept.append(s)

    for s in kept:
        if s.kind == 'cusp_curve':
            continue
        s.branch_links = _recompute_branch_links(
            s.xyz, branches, atol, agg)
    return kept


# ---------------------------------------------------------------------------
# Overlap-region unification (spec: seam-rim dissolution + rim chaining,
# certification merged conservatively — no re-verification)
# ---------------------------------------------------------------------------

def _vertex_seam_tags(vertex_stuv, cuts_per_axis, ctx):
    """Exact decomposition seams containing this vertex."""
    tags = set()
    for axis, cuts in enumerate(cuts_per_axis):
        for cut in cuts:
            if float(vertex_stuv[axis]) == cut:
                tags.add((axis, float(cut)))
    return frozenset(tags)


def _segment_seam_tags(v0, v1, cuts_per_axis, ctx):
    """Exact decomposition seams containing an entire linear segment."""
    tags = set()
    for axis, cuts in enumerate(cuts_per_axis):
        for cut in cuts:
            if float(v0[axis]) == cut and float(v1[axis]) == cut:
                tags.add((axis, float(cut)))
    return frozenset(tags)


def _split_rim_at_cut_bands(frag, cuts_per_axis, ctx):
    """Split one rim polyline into parts of CONSTANT segment seam-tag set.

    A per-pair rim loop may be a single polyline around the whole tile
    (several edges); dissolution is per shared edge, so each part must
    lie either fully on one set of seams or fully off them. Membership
    is per SEGMENT (both endpoints in the cut band): a vertex-based
    split is orientation-asymmetric at tile corners — the lone corner
    vertex touching a perpendicular cut splits a corner-at-start rim
    into a stub + remainder but leaves its corner-at-end partner whole,
    desynchronizing partner extents and blocking dissolution. Adjacent
    parts share their boundary vertex. Returns a list of _Frag parts
    (>= 1)."""
    stuv = frag.stuv
    n = len(stuv)
    if n < 2:
        return [frag]
    seg_tags = [_segment_seam_tags(stuv[k], stuv[k + 1], cuts_per_axis,
                                   ctx)
                for k in range(n - 1)]
    spans = []
    start = 0
    for k in range(1, n - 1):
        if seg_tags[k] != seg_tags[k - 1]:
            spans.append((start, k))
            start = k
    spans.append((start, n - 1))
    if len(spans) == 1:
        return [frag]
    return [_Frag(stuv=stuv[a:b + 1].copy(),
                  xyz=frag.xyz[a:b + 1].copy(),
                  kind='overlap', overlap=True,pair=frag.pair,rect=frag.rect,
                  source_boundary_face=frag.source_boundary_face,
                  source_path=(frag.source_path[a:b+1] if frag.source_path is not None
                               and len(frag.source_path) == n else None))
            for a, b in spans]


def _part_seam_tags(frag, cuts_per_axis, ctx):
    """Seam membership of a rim part: the INTERSECTION of its vertices'
    tag sets (empty => off-seam, a true rim of the union)."""
    common = None
    for v in frag.stuv:
        t = _vertex_seam_tags(v, cuts_per_axis, ctx)
        common = t if common is None else (common & t)
        if not common:
            return frozenset()
    return common if common is not None else frozenset()


def _rims_are_partners(fa, fb, ctx, atol, charge=None):
    """Same-locus test between seam rims of ADJACENT tiles: endpoints
    match crosswise (opposite raw orientation) OR parallel, and the
    complete lifted paths coincide. Raw polyline direction is
    sampler bookkeeping, not loop orientation: the per-pair rim sampler
    always emits edges in increasing local parameter (rev flags in the
    loop entries carry orientation), so two adjacent tiles' seam rims
    typically arrive PARALLEL even though their loops traverse the
    shared seam oppositely."""
    cross = (_match_stuv(fa.stuv[0], fb.stuv[-1], fa.xyz[0], fb.xyz[-1],
                         ctx, atol)
             and _match_stuv(fa.stuv[-1], fb.stuv[0], fa.xyz[-1],
                             fb.xyz[0], ctx, atol))
    parallel = (_match_stuv(fa.stuv[0], fb.stuv[0], fa.xyz[0], fb.xyz[0],
                        ctx, atol)
            and _match_stuv(fa.stuv[-1], fb.stuv[-1], fa.xyz[-1],
                            fb.xyz[-1], ctx, atol))
    if not (cross or parallel):
        return False
    # Matching approximations merely propose a seam. Actual exact source
    # parameter paths must prove that the complete seam is shared.
    from mmcore.numeric.intersection.ssx._ssx_arc_ownership import source_path_covered
    return (source_path_covered(fa.source_path,[fb.source_path],charge) is True
            and source_path_covered(fb.source_path,[fa.source_path],charge) is True)


def _shoelace_area(poly2):
    p = np.asarray(poly2, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _merge_certifications(tiles):
    """Conservative certification merge: max residuals, summed
    n_samples, AND of orientation flags — no re-verification.
    orientation_consistent is inherited from the tiles (ANDed), not
    re-verified against the re-chained unified loops."""
    cert = {
        'boundary_resid_max': max(
            float(t.certification.get('boundary_resid_max', 0.0))
            for t in tiles),
        'interior_resid': max(
            float(t.certification.get('interior_resid', 0.0))
            for t in tiles),
        'n_samples': int(sum(
            int(t.certification.get('n_samples', 0)) for t in tiles)),
        'orientation_consistent': all(
            bool(t.certification.get('orientation_consistent', True))
            for t in tiles),
    }
    return cert


def _assemble_regions(raw, stitched, ctx, atol, agg,
                      s_cuts, t_cuts, u_cuts, v_cuts):
    """Unify per-pair region tiles into one region per connected
    coincidence component; dissolve interior-seam rims; drop only stitched
    overlap branches continuously identified with represented rims or
    dissolved seams. Pair-level uncertainty survives assembly.

    Unified regions guarantee loop ORDERING (outer loop first, by |area|
    in the uv1 plane) but not winding direction; the per-tile
    passthrough path preserves the engine's outer-CCW/holes-CW winding,
    the multi-tile path does not re-normalize it (no current consumer
    reads winding — revisit before building region trimming on top)."""
    if not raw.tiles:
        return stitched, []

    cuts_per_axis = (s_cuts, t_cuts, u_cuts, v_cuts)
    n_tiles = len(raw.tiles)

    # --- (0) normalize: split rims at cut bands, rebuild tile loops ----
    split_ids = {}
    rims = []
    for rid, frag in enumerate(raw.rim_frags):
        parts = _split_rim_at_cut_bands(frag, cuts_per_axis, ctx)
        ids = []
        for part in parts:
            ids.append(len(rims))
            rims.append(part)
        split_ids[rid] = ids

    tiles_loops = []
    for tile in raw.tiles:
        loops = []
        for loop in tile.loops:
            entries = []
            for rid, rev in loop:
                ids = split_ids[rid]
                entries.extend(
                    (pid, rev)
                    for pid in (reversed(ids) if rev else ids))
            loops.append(entries)
        tiles_loops.append(loops)

    # tile index per (normalized) rim id
    tile_of_rim = {}
    for ti in range(n_tiles):
        for loop in tiles_loops[ti]:
            for rid, _rev in loop:
                tile_of_rim[rid] = ti

    # --- (a) seam-rim dissolution ------------------------------------
    seam_tags = {rid: _part_seam_tags(rims[rid], cuts_per_axis, ctx)
                 for rid in tile_of_rim}
    dissolved = set()
    dissolved_pairs = []
    tile_parent = list(range(n_tiles))

    def _tfind(x):
        while tile_parent[x] != x:
            tile_parent[x] = tile_parent[tile_parent[x]]
            x = tile_parent[x]
        return x

    def _tunion(a, b):
        ra, rb = _tfind(a), _tfind(b)
        if ra != rb:
            tile_parent[max(ra, rb)] = min(ra, rb)

    rim_ids = sorted(tile_of_rim)
    for ai in range(len(rim_ids)):
        ra = rim_ids[ai]
        if ra in dissolved or not seam_tags[ra]:
            continue
        for bi in range(ai + 1, len(rim_ids)):
            rb = rim_ids[bi]
            if rb in dissolved or tile_of_rim[ra] == tile_of_rim[rb]:
                continue
            if not (seam_tags[ra] & seam_tags[rb]):
                continue
            ta, tb = raw.tiles[tile_of_rim[ra]], raw.tiles[tile_of_rim[rb]]
            if ta.agreement != tb.agreement:      # spec (d): never merge
                continue
            if not agg.charge_postprocess(
                    max(1, len(rims[ra].stuv) // 4)):
                break
            if _rims_are_partners(rims[ra], rims[rb], ctx, atol,
                                  charge=agg.charge_postprocess):
                crossed = (_match_stuv(rims[ra].stuv[0], rims[rb].stuv[-1],
                                       rims[ra].xyz[0], rims[rb].xyz[-1], ctx, atol)
                           and _match_stuv(rims[ra].stuv[-1], rims[rb].stuv[0],
                                           rims[ra].xyz[-1], rims[rb].xyz[0], ctx, atol))
                dissolved.update((ra, rb))
                dissolved_pairs.append((ra, rb, crossed))
                _tunion(tile_of_rim[ra], tile_of_rim[rb])
                break

    # --- (b) per-component loop chaining ------------------------------
    components = {}
    for ti in range(n_tiles):
        components.setdefault(_tfind(ti), []).append(ti)

    # The tile loops already supply vertex identity. Keep that topology
    # through seam dissolution instead of reconstructing it with distance
    # clustering (which would also join two nearby ordinary components).
    vertex_parent = list(range(2 * len(rims)))

    def vertex_find(v):
        while vertex_parent[v] != v:
            vertex_parent[v] = vertex_parent[vertex_parent[v]]
            v = vertex_parent[v]
        return v

    def vertex_union(a, b):
        a, b = vertex_find(a), vertex_find(b)
        vertex_parent[max(a, b)] = min(a, b)

    for loops in tiles_loops:
        for loop in loops:
            for (a, ar), (b, br) in zip(loop, loop[1:] + loop[:1]):
                vertex_union(2 * a + int(not ar), 2 * b + int(br))
    for a, b, crossed in dissolved_pairs:
        vertex_union(2 * a, 2 * b + int(crossed))
        vertex_union(2 * a + 1, 2 * b + int(not crossed))
    for rid, rim in enumerate(rims):
        rim.endpoint_ids = (vertex_find(2 * rid), vertex_find(2 * rid + 1))

    final_rim_branches = []      # SSXBranch, appended after stitched
    final_regions = []

    def _tile_passthrough(ti):
        """Emit one tile verbatim (single-tile component or fallback)."""
        tile = raw.tiles[ti]
        loops_out = []
        uv1_loops, uv2_loops = [], []
        for loop in tiles_loops[ti]:
            entries = []
            parts = []
            for rid, rev in loop:
                bi = len(stitched) + len(final_rim_branches)
                final_rim_branches.append(SSXBranch(
                    curve=(rims[rid].stuv, rims[rid].xyz),
                    kind='overlap', overlap=True))
                entries.append((bi, rev))
                parts.append(rims[rid].stuv[::-1] if rev
                             else rims[rid].stuv)
            chained = np.concatenate(parts, axis=0)
            closed4 = np.concatenate([chained, chained[:1]], axis=0)
            loops_out.append(entries)
            uv1_loops.append(closed4[:, :2].copy())
            uv2_loops.append(closed4[:, 2:].copy())
        final_regions.append(SSXOverlapRegion(
            boundary=loops_out, uv1_loops=uv1_loops, uv2_loops=uv2_loops,
            normal_agreement=tile.agreement,
            interior_stuv=tile.interior_stuv,
            certification=dict(tile.certification)))

    for root in sorted(components):
        member_tiles = components[root]
        if len(member_tiles) == 1:
            _tile_passthrough(member_tiles[0])
            continue
        surviving = [rid for ti in member_tiles
                     for loop in tiles_loops[ti]
                     for rid, _rev in loop if rid not in dissolved]
        surv_frags = [rims[rid] for rid in surviving]
        chains = _build_chains(surv_frags, ctx, atol, agg,
                               kind_barrier=False)
        if not chains or not all(closed for _chain, closed in chains):
            # Rim chaining failed to close every loop — inconsistent
            # dissolution evidence. Fall back to honest per-tile output.
            for ti in member_tiles:
                _tile_passthrough(ti)
            continue
        loops_data = []
        for chain, _closed in chains:
            entries = []
            parts = []
            for li, flip in chain:
                rid = surviving[li]
                bi = len(stitched) + len(final_rim_branches)
                final_rim_branches.append(SSXBranch(
                    curve=(rims[rid].stuv, rims[rid].xyz),
                    kind='overlap', overlap=True))
                entries.append((bi, bool(flip)))
                parts.append(rims[rid].stuv[::-1] if flip
                             else rims[rid].stuv)
            chained = np.concatenate(parts, axis=0)
            closed4 = np.concatenate([chained, chained[:1]], axis=0)
            loops_data.append((entries, closed4))
        # outer loop = largest |shoelace area| in the uv1 plane
        loops_data.sort(
            key=lambda ld: -abs(_shoelace_area(ld[1][:, :2])))
        tiles_objs = [raw.tiles[ti] for ti in member_tiles]
        final_regions.append(SSXOverlapRegion(
            boundary=[entries for entries, _c4 in loops_data],
            uv1_loops=[c4[:, :2].copy() for _e, c4 in loops_data],
            uv2_loops=[c4[:, 2:].copy() for _e, c4 in loops_data],
            normal_agreement=tiles_objs[0].agreement,
            interior_stuv=tiles_objs[0].interior_stuv,
            certification=_merge_certifications(tiles_objs)))

    # --- (c) remove only represented rims and proved dissolved seams --
    # A curve inside projected footprints or on an approximate rim chord
    # can be another component. Exact source-path ownership is required.
    # `rims` includes dissolved seams, which remain represented by the region.
    from mmcore.numeric.intersection.ssx._ssx_arc_ownership import source_path_covered
    rim_paths = [rim.source_path for rim in rims if rim.source_path is not None]
    kept_stitched = []
    for b in stitched:
        absorbed = False
        if b.kind == 'overlap' and final_regions:
            owned = getattr(b,'_source_parameter_paths',None)
            absorbed = bool(owned) and all(source_path_covered(
                path,rim_paths,agg.charge_postprocess) is True for path in owned)
        if not absorbed:
            kept_stitched.append(b)

    # dropping stitched branches shifts rim base indices — rebuild refs
    shift = len(kept_stitched) - len(stitched)
    if shift != 0:
        for reg in final_regions:
            reg.boundary = [[(bi + shift, rev) for bi, rev in loop]
                            for loop in reg.boundary]

    return kept_stitched + final_rim_branches, final_regions


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
    from mmcore.numeric.intersection.ssx._ssx_input import validate_control_data
    for index, surface in enumerate((s1, s2), 1):
        validate_control_data(surface.control_points, surface.weights,
                              context=f'nurbs_ssx surface {index}')
    atol = float(atol)
    rational = _is_rational(s1) or _is_rational(s2)
    ctx = _domain_ctx(s1, s2, atol)

    patches1 = decompose_surface(s1, "uv")
    patches2 = decompose_surface(s2, "uv")
    ctx.source_patches = (patches1, patches2)

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
        _pair_cells, _pair_csx, _pair_out = _per_pair_allowance(
            agg, len(candidates) - k)
        result = bez_ssx(
            P1, P2, atol=atol, rational=rational,
            max_cells=_pair_cells,
            max_csx_calls=_pair_csx,
            max_output_items=min(_pair_out,
                                 agg.remaining_output_items),
            **forward)
        agg.consume(result)
        rect = _pair_rect(p1, p2)
        if REASON_MULTIPLICITY in (
                (result.get('status', {}) or {}).get('reasons', []) or []):
            raw.mult_rects.append(rect)
        _collect_pair(raw, result, rect, pair=(i, j), sources=(P1, P2),
                      rational=rational, atol=atol, agg=agg)

    _finish_collecting(raw)

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
        raw.singularities, branches, ctx, atol, agg, s1=s1)

    out = {
        'branches': branches,
        'points': points,
        'singularities': singularities,
        'overlap_regions': regions,
        'unresolved_regions': raw.unresolved,
    }
    out.update(agg.result_fields())
    return out
