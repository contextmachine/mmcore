"""Bezier surface-surface intersection v5.

Combines three approaches:
1. Sq-dist Bernstein net for Lipschitz pruning (from CCX/CSX v4)
2. TΨᵢ monotonicity criterion for loop-absence certification (Cheng et al. 2023)
3. Domain decomposition at boundary crossing points (Krishnan & Manocha 1997)
4. Deflation for tangential (C₂) cases (Cheng et al. 2023)

Architecture:
  Level 1: Pruning (AABB + sq-dist Lipschitz)
  Level 2: Boundary analysis (8 CSX problems on faces of [0,1]⁴)
  Level 3: Monotonicity classification (TΨᵢ sign-definiteness)
  Level 4a: Domain decomposition + tracing (monotonic cells)
  Level 4b: Deflation (tangent cells)
  Level 5: Output assembly (merge/close/prune)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from mmcore.numeric.bern_sq_dist import surface_surface_distance_squared_net_homog
from mmcore.numeric.intersection._bezier_common import (
    extract_weights, eval_surface, eval_surface_d1, eval_curve,
)
from mmcore.numeric.intersection._sq_dist_classify import (
    _check_min_of_net, _check_lipschitz, _weight_max_product,
)
from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx
from mmcore.numeric.intersection._deflate import minors_Tpsi_from_control_nets
from mmcore.numeric._aabb import aabb, aabb_intersect

from mmcore.numeric.intersection.ssx._ssx4 import (
    SSXBranch, SSXPoint,
    _append_unique_point,
    GaussMapBern,
    separate_gauss_maps,
    _trust_gjk,
    reset_witness_rng,
)
from mmcore.numeric.algorithms.cygjk import gjk


# ---------------------------------------------------------------------------
# Data structures (§5 of design)
# ---------------------------------------------------------------------------

@dataclass
class _SSXSoftBudget:
    """One shared work budget for an entire :func:`bez_ssx` call.

    Local solver limits are still useful backstops, but they do not compose:
    SSX can invoke thousands of CSX/zero-dimensional searches and each search
    used to receive a fresh allowance.  This object is deliberately tiny and
    callback-friendly so nested solvers spend from the same counter.
    """

    max_cells: int
    max_csx_calls: int
    max_output_items: int = 1_024
    max_postprocess_work: Optional[int] = None
    cells_processed: int = 0
    csx_calls: int = 0
    output_items: int = 0
    postprocess_work: int = 0
    postprocess_exhausted: bool = False
    exhausted: bool = False
    incomplete: bool = False
    cell_counts: dict = field(default_factory=dict)
    output_counts: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.max_postprocess_work is None:
            self.max_postprocess_work = max(0, int(self.max_cells))
        else:
            self.max_postprocess_work = max(
                0, int(self.max_postprocess_work))

    def charge_cells(self, amount: int = 1, source: str = "nested") -> bool:
        amount = max(0, int(amount))
        if self.exhausted:
            return False
        if self.cells_processed + amount > self.max_cells:
            self.exhausted = True
            return False
        self.cells_processed += amount
        self.cell_counts[source] = self.cell_counts.get(source, 0) + amount
        return True

    def charge_csx_call(self) -> bool:
        if self.exhausted or self.csx_calls >= self.max_csx_calls:
            self.exhausted = True
            return False
        self.csx_calls += 1
        return True

    def charge_postprocess(self, amount: int = 1) -> bool:
        """Charge bounded assembly/filter work after the search phase.

        This counter is separate from subdivision cells so a hard-stopped
        search can still assemble its certified partial fragments. It is
        nevertheless call-wide and finite, preventing postprocessing from
        becoming a second unbounded phase.
        """
        amount = max(0, int(amount))
        if self.postprocess_exhausted:
            return False
        if self.postprocess_work + amount > self.max_postprocess_work:
            self.postprocess_exhausted = True
            self.exhausted = True
            self.incomplete = True
            return False
        self.postprocess_work += amount
        return True

    @property
    def remaining_cells(self) -> int:
        return max(0, self.max_cells - self.cells_processed)

    @property
    def remaining_postprocess_work(self) -> int:
        return max(0, self.max_postprocess_work - self.postprocess_work)

    def mark_exhausted(self) -> None:
        self.exhausted = True

    def mark_incomplete(self) -> None:
        """Record a partial local result without stopping independent work."""
        self.incomplete = True

    def append_output(self, target: list, value, source: str) -> bool:
        """Append one intermediate/output entity under the global cap."""
        if self.output_items >= self.max_output_items:
            self.mark_incomplete()
            return False
        target.append(value)
        self.output_items += 1
        self.output_counts[source] = self.output_counts.get(source, 0) + 1
        return True

    def extend_output(self, target: list, values, source: str) -> bool:
        complete = True
        for value in values:
            if not self.append_output(target, value, source):
                complete = False
                break
        return complete

    def result_fields(self) -> dict:
        return {
            "budget_exhausted": bool(self.exhausted or self.incomplete),
            "budget_usage": {
                "cells_processed": int(self.cells_processed),
                "csx_calls": int(self.csx_calls),
                "max_cells": int(self.max_cells),
                "max_csx_calls": int(self.max_csx_calls),
                "output_items": int(self.output_items),
                "max_output_items": int(self.max_output_items),
                "postprocess_work": int(self.postprocess_work),
                "max_postprocess_work": int(self.max_postprocess_work),
                "postprocess_exhausted": bool(self.postprocess_exhausted),
                "hard_exhausted": bool(self.exhausted),
                "incomplete": bool(self.incomplete),
                "cell_counts": dict(self.cell_counts),
                "output_counts": dict(self.output_counts),
            },
        }


@dataclass
class IsolineRegistration:
    """One crossing's registration on one isoline, from the owning cell's view.

    Design §5: tuple (isoline, isoline_global_interval, param, direction).
    """
    partition: "PartitionCurve"
    param: float
    direction: str            # "in" or "out"
    owner: "_Cell"
    point: "BoundaryPoint"
    consumed: bool = False    # set True after the tracer uses this registration


@dataclass
class PartitionCurve:
    """An isoline — 1D curve bounding one or more cells (design §5).

    Specified by one fixed axis + value and one free axis + its global extent.
    Outer partitions (top-level [0,1]⁴ faces, i.e. an S1 or S2 boundary isoline)
    have one adjacent cell; internal partitions (created by subdivision) have two.
    """
    axis: int                                   # 0..3, the fixed coordinate
    value: float                                # global value of the fixed coordinate
    free_axis: int                              # 0..3, the varying coordinate
    global_extent: tuple[float, float]          # free-axis range in global coords
    adjacents: list["_Cell"] = field(default_factory=list)
    registrations: list[IsolineRegistration] = field(default_factory=list)


@dataclass
class BoundaryPoint:
    """A 4D boundary crossing (design §5).

    `face` is the legacy (axis, side) tag retained while the old heuristic pairing
    still uses it; §4 classification will produce one `IsolineRegistration` per
    on-boundary axis and replace `face` as the crossing's identity.
    """
    stuv: NDArray[np.float64]                               # (4,) parameter values
    xyz: NDArray[np.float64]                                # (3,) Euclidean point
    face: tuple[int, int]                                   # (axis 0-3, side 0-1)
    tangent_raw: Optional[NDArray[np.float64]] = None       # (4,) unclamped null vec of J_Ψ
    registrations: list[IsolineRegistration] = field(default_factory=list)
    parameter_fiber: bool = False  # unresolved free parameter on a collapsed edge
    multiplicity_polished: bool = False  # nearby CSX samples collapsed to this root


# Back-compat alias — existing code uses BoundaryCrossing in many places and
# the design §5 name is BoundaryPoint. Keep both symbols pointing at the same
# dataclass so the rename can propagate gradually.
BoundaryCrossing = BoundaryPoint


@dataclass
class SSXSingularity:
    """A certified singular feature of the SSI (Cheng et al. 2023 C1/C2/C3).

    kind:
      'tangent_point'      — C2: T_Psi = 0, isolated or on branches
      'cusp'               — C1: surface-parameterization cusp on the curve
      'cusp_curve'         — C1 infinite case: samples of a singular curve
      'self_intersection'  — C3: two 4D preimages, one 3D point

    For 'self_intersection', `stuv` is the primary preimage (s,t,u,v) and
    `stuv_mate` differs from it in the DOUBLED side only: an S1-side double
    carries the second S1 preimage, stuv_mate = (p,q,u,v); an S2-side
    double (umbrella-as-S2 class, ledger L7) carries the second S2
    preimage, stuv_mate = (s,t,u',v').

    branch_links contract (ledger L11): each link is (branch_index,
    vertex_index) where vertex_index addresses the linked branch's
    polyline VERTEX nearest this singularity's xyz (locally — for a
    branch crossing itself, each link anchors on its own pass). It is a
    position on the polyline, not a segment id; the vertex itself can
    still sit up to ~half a chord from xyz at coarse chord spacing.
    """
    kind: str
    stuv: NDArray[np.float64]                    # (4,) primary preimage
    xyz: NDArray[np.float64]                     # (3,)
    stuv_mate: Optional[NDArray[np.float64]] = None   # (4,) C3 second preimage
    branch_links: list = field(default_factory=list)  # [(branch_index, vertex_index)]
    samples: Optional[NDArray[np.float64]] = None     # (N,4) for 'cusp_curve'
    surface: Optional[int] = None    # C1 only (ledger L22): 1|2 — WHICH
    #                                  surface's parameterization is
    #                                  degenerate (Sigma_i = 0) at the
    #                                  cusp / along the cusp curve.


@dataclass
class BoundaryOverlap:
    """A curve segment where S1 and S2 overlap on a boundary face of [0,1]⁴."""
    stuv_start: NDArray[np.float64]  # (4,)
    stuv_end: NDArray[np.float64]    # (4,)
    face: tuple[int, int]            # (axis 0-3, side 0-1)


_POINT_PARAM_NEIGHBOR_OFFSETS = tuple(product((-1, 0, 1), repeat=4))
_POINT_XYZ_NEIGHBOR_OFFSETS = tuple(product((-1, 0, 1), repeat=3))


def _deduplicate_ssx_points(points, unify_tol, atol, stats=None):
    """Stable both-guard SSXPoint dedup with a bounded spatial index.

    The legacy final pass compared every point with every previously kept
    point.  Dense pseudo-root output could therefore spend quadratic work
    after the global solver budget had already stopped the search.  Bin the
    four parameters at ``unify_tol`` and xyz at ``2*atol``; an exact match
    can only live in the 3^7 neighboring bins.  Exact comparisons retain the
    established matching-ladder predicate and insertion order.
    """
    if not points:
        if stats is not None:
            stats["comparisons"] = 0
            stats["bucket_probes"] = 0
        return []

    pstep = np.maximum(np.abs(np.asarray(unify_tol, dtype=np.float64)),
                       np.finfo(float).tiny)
    xstep = max(2.0 * abs(float(atol)), np.finfo(float).tiny)
    buckets = {}
    unique = []
    comparisons = 0
    bucket_probes = 0

    def _keys(p):
        pvalues = np.asarray(p.stuv, dtype=np.float64) / pstep
        xvalues = np.asarray(p.xyz, dtype=np.float64) / xstep
        # math.floor returns an arbitrary-precision Python int, avoiding the
        # int64 overflow of an ndarray cast on large model coordinates.
        return (
            tuple(math.floor(float(x)) for x in pvalues),
            tuple(math.floor(float(x)) for x in xvalues),
        )

    for p in points:
        pkey, xkey = _keys(p)
        duplicate = False
        for poffset in _POINT_PARAM_NEIGHBOR_OFFSETS:
            pneighbor = tuple(k + d for k, d in zip(pkey, poffset))
            bucket_probes += 1
            xyz_buckets = buckets.get(pneighbor)
            if xyz_buckets is None:
                continue
            for xoffset in _POINT_XYZ_NEIGHBOR_OFFSETS:
                xneighbor = tuple(k + d for k, d in zip(xkey, xoffset))
                bucket_probes += 1
                for q in xyz_buckets.get(xneighbor, ()):
                    comparisons += 1
                    if (np.all(np.abs(np.asarray(p.stuv)
                                     - np.asarray(q.stuv)) <= pstep)
                            and float(np.linalg.norm(np.asarray(p.xyz)
                                                     - np.asarray(q.xyz)))
                            <= 2.0 * abs(float(atol))):
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if not duplicate:
            unique.append(p)
            buckets.setdefault(pkey, {}).setdefault(xkey, []).append(p)

    if stats is not None:
        stats["comparisons"] = int(comparisons)
        stats["bucket_probes"] = int(bucket_probes)
    return unique


# ---------------------------------------------------------------------------
# Level 1: Pruning
# ---------------------------------------------------------------------------

def _ssx_control_aabbs_disjoint(S1_h, S2_h, rational=True):
    """Cheap top-level hull rejection before any 4-D net allocation."""
    if rational:
        pts1 = S1_h[..., :-1] / S1_h[..., -1:]
        pts2 = S2_h[..., :-1] / S2_h[..., -1:]
    else:
        pts1 = S1_h
        pts2 = S2_h
    bb1 = np.array(aabb(pts1.reshape(-1, pts1.shape[-1])))
    bb2 = np.array(aabb(pts2.reshape(-1, pts2.shape[-1])))
    return not aabb_intersect(bb1, bb2)


def _prune_ssx_cell(S1_h, S2_h, atol, rational=True, F=None):
    """Return True if this patch pair provably does NOT intersect.

    Checks:
    1. AABB non-overlap (Euclidean control points)
    2. Min-of-net on 4-variate sq-dist Bernstein net
    3. Lipschitz tightening on sq-dist net
    """
    # AABB check on Euclidean control points
    _, S1w = extract_weights(S1_h, rational=rational)
    _, S2w = extract_weights(S2_h, rational=rational)

    if _ssx_control_aabbs_disjoint(S1_h, S2_h, rational=rational):
        return True

    # Sq-dist net pruning
    if F is None:
        F = surface_surface_distance_squared_net_homog(
            S1_h, S2_h, rational=rational)
    sw1 = S1w.ravel()
    sw2 = S2w.ravel()
    w_scale = _weight_max_product(sw1, sw2)

    if _check_min_of_net(F, atol, w_scale):
        return True
    if _check_lipschitz(F, atol, w_scale):
        return True

    return False


def _aabb_disjoint(S1_h, S2_h, atol):
    """Fast AABB-only check: True if the two patches' bounding boxes are
    disjoint (no possible intersection). Cheaper than `_prune_ssx_cell`
    because it skips the sq-dist net computation.

    Roundoff margin (ledger L4 — the L1 pattern in a spot the L1 audit
    missed): guided cuts pass EXACTLY through touch coordinates, so a
    control net whose true extremum is exactly 0 at a box corner drifts by
    ~eps per de Casteljau level — measured on the off-lattice touch+loop
    (touch (0.3,0.3)): at depth 8 the S1 patch's z-hull max drifted to
    -3e-17 against the plane's exact z = [0, 0] and this strict test
    pruned every touch-holding cell (`aabb_intersect` itself is non-strict,
    so ONLY the drift kills). Inflate one box by the shared L1 margin
    (`HULL_MARGIN_K = 128` · eps, scaled to the boxes' coordinate
    magnitude, ~3e-14 at O(1)) — orders below atol and any genuine
    separation, so pruning power is unaffected; inflation only makes the
    prune STRICTER to certify, the sound direction.
    """
    pts1 = S1_h[..., :-1] / S1_h[..., -1:]
    pts2 = S2_h[..., :-1] / S2_h[..., -1:]
    bb1 = np.array(aabb(pts1.reshape(-1, pts1.shape[-1])))
    bb2 = np.array(aabb(pts2.reshape(-1, pts2.shape[-1])))
    from mmcore.numeric.intersection.ssx._ssx5_singular import _HULL_MARGIN_K_EPS
    m = _HULL_MARGIN_K_EPS * max(
        float(np.abs(bb1).max()), float(np.abs(bb2).max()), 1e-30)
    bb1[0] -= m
    bb1[1] += m

    return not aabb_intersect(bb1, bb2)


# ---------------------------------------------------------------------------
# Level 2: Boundary analysis (8 CSX problems)
# ---------------------------------------------------------------------------

def _map_csx_to_stuv(s1_axis, side, t_crv, u_other, v_other, owner_is_s1):
    """Map CSX result parameters to stuv in [0,1]⁴."""
    stuv = np.zeros(4, dtype=np.float64)
    if owner_is_s1:
        if s1_axis == 0:
            stuv[0] = float(side)
            stuv[1] = t_crv
        else:
            stuv[0] = t_crv
            stuv[1] = float(side)
        stuv[2] = u_other
        stuv[3] = v_other
    else:
        stuv[0] = u_other  # s = u on S1
        stuv[1] = v_other  # t = v on S1
        if s1_axis == 0:  # s2_axis == 0 means u is fixed
            stuv[2] = float(side)
            stuv[3] = t_crv
        else:              # s2_axis == 1 means v is fixed
            stuv[2] = t_crv
            stuv[3] = float(side)
    return stuv


def _invert_point_on_surface(S_h, P, rational=True, grid=4, iters=12):
    """(u,v) minimizing |S(u,v) − P| — small Gauss-Newton from the best
    point of a coarse grid, clamped to [0,1]². Used to RESOLVE boundary
    overlap endpoints: CSX reports t/u/v ranges whose index-pairing is
    meaningless for curve-on-surface overlaps (corner-sharing bilinear
    repro: the s=1 edge's claim carried u=(0,0), v=(0.5,0.5) — a single
    garbage surface point for the whole edge)."""
    P = np.asarray(P, dtype=np.float64)
    best, best_d = None, np.inf
    for gu in np.linspace(0.0, 1.0, grid + 1):
        for gv in np.linspace(0.0, 1.0, grid + 1):
            d = float(np.linalg.norm(eval_surface(S_h, gu, gv, rational=rational) - P))
            if d < best_d:
                best_d, best = d, (gu, gv)
    u, v = best
    for _ in range(iters):
        pt, du, dv = eval_surface_d1(S_h, u, v, rational=rational)
        r = pt - P
        J = np.column_stack([du, dv])
        A = J.T @ J + 1e-14 * np.eye(2)
        try:
            step = np.linalg.solve(A, -(J.T @ r))
        except np.linalg.LinAlgError:
            break
        u = float(np.clip(u + step[0], 0.0, 1.0))
        v = float(np.clip(v + step[1], 0.0, 1.0))
        if float(np.linalg.norm(step)) < 1e-14:
            break
    return u, v


def _curve_geometry_collapsed(C, rational=True) -> bool:
    pts = (C[..., :-1] / C[..., -1:]) if rational else np.asarray(C)
    # Use local motion, not absolute coordinate magnitude. At x=1e15 the
    # old global-scale epsilon was ~28 model units and misclassified a
    # 10-unit line as a point/fiber, deleting its isolated intersection.
    delta = pts - pts[0]
    scale = max(1.0, float(np.max(np.abs(delta))))
    eps = 128.0 * np.finfo(float).eps * scale
    return float(np.max(np.linalg.norm(delta, axis=-1))) <= eps


def _weight_net_uniform(S) -> bool:
    """Exact rational-polynomial fast-path predicate."""
    w = np.asarray(S, dtype=np.float64)[..., -1]
    return bool(w.size and np.all(w == w.flat[0]))


def _on_collapsed_boundary_fiber(
        S, u, v, rational=True, *, param_tol: float = 1e-10) -> bool:
    """True only for a parameter point on an identically collapsed edge.

    This is deliberately narrower than ``Sigma=0``: a C1 point can still
    be regular on the 4D Psi curve and may be a required branch seed.  We
    suppress only a certified positive-dimensional boundary preimage.
    """
    eps = max(0.0, float(param_tol))
    if abs(u) <= eps and _curve_geometry_collapsed(S[0, :, :], rational):
        return True
    if abs(u - 1.0) <= eps and _curve_geometry_collapsed(S[-1, :, :], rational):
        return True
    if abs(v) <= eps and _curve_geometry_collapsed(S[:, 0, :], rational):
        return True
    if abs(v - 1.0) <= eps and _curve_geometry_collapsed(S[:, -1, :], rational):
        return True
    return False


def _canonicalize_collapsed_fiber_params(
        S, uv, anchor_uv, *, rational=True, param_tol=1e-10):
    """Choose the limiting branch representative on collapsed edge fibers.

    A collapsed edge maps every value of its free parameter to one xyz
    point. Boundary CSX may therefore return an arbitrary free value, which
    is valid as a set member but not as the limit of the incident 4-D SSI
    branch. ``anchor_uv`` is an interior Delta witness on that branch; its
    free coordinate supplies a deterministic, continuation-consistent
    representative without changing xyz.
    """
    out = np.asarray(uv, dtype=np.float64).copy()
    anchor = np.asarray(anchor_uv, dtype=np.float64)
    eps = max(0.0, float(param_tol))
    if abs(out[0]) <= eps and _curve_geometry_collapsed(S[0, :, :], rational):
        out[1] = anchor[1]
    if (abs(out[0] - 1.0) <= eps
            and _curve_geometry_collapsed(S[-1, :, :], rational)):
        out[1] = anchor[1]
    if abs(out[1]) <= eps and _curve_geometry_collapsed(S[:, 0, :], rational):
        out[0] = anchor[0]
    if (abs(out[1] - 1.0) <= eps
            and _curve_geometry_collapsed(S[:, -1, :], rational)):
        out[0] = anchor[0]
    return out


def _find_ssx_boundary_zeros(
        S1_h, S2_h, atol, rational=True, csx_fn=None, fiber_sink=None):
    """Find all intersection points and overlaps on the boundary of [0,1]⁴.

    Returns (crossings, overlaps).
    """
    crossings = []
    overlaps = []
    if csx_fn is None:
        csx_fn = bez_csx
    if fiber_sink is None:
        fiber_sink = []

    def _process_face(iso, other_surf, axis, side, owner_is_s1):
        result = csx_fn(iso, other_surf, atol=atol, rational=rational)

        # A collapsed owner edge produces a positive-dimensional CSX
        # parameter fiber instead of isolated roots. Preserve one typed,
        # unresolved boundary seed; the free edge parameter is chosen only
        # after an interior Delta witness identifies the limiting 4-D SSI
        # branch. Dropping this metadata deleted one end of case 14's cone
        # generator, while choosing t=.5 here would be arbitrary/unsound.
        for fiber in result.get('parameter_fibers', []):
            u_oth = float(fiber.get('u', 0.5))
            v_oth = float(fiber.get('v', 0.5))
            stuv = _map_csx_to_stuv(
                axis, side, 0.5, u_oth, v_oth, owner_is_s1)
            xyz = np.asarray(fiber.get(
                'point', eval_curve(iso, 0.5, rational=rational)),
                dtype=np.float64)
            p1 = eval_surface(S1_h, stuv[0], stuv[1], rational=rational)
            p2 = eval_surface(S2_h, stuv[2], stuv[3], rational=rational)
            if (float(np.linalg.norm(p1 - xyz)) > 2.0 * atol
                    or float(np.linalg.norm(p2 - xyz)) > 2.0 * atol):
                continue
            face_id = axis if owner_is_s1 else axis + 2
            fiber_sink.append(BoundaryPoint(
                stuv=stuv, xyz=xyz, face=(face_id, side),
                tangent_raw=None, parameter_fiber=True))

        for iso_pt in result.get('isolated', []):
            t_crv = float(iso_pt['t'])
            u_oth = float(iso_pt['u'])
            v_oth = float(iso_pt['v'])
            stuv = _map_csx_to_stuv(axis, side, t_crv, u_oth, v_oth, owner_is_s1)
            raw_stuv = stuv.copy()
            # CSX's geometric tolerance can return several samples around
            # one even-multiplicity boundary root.  In SSX those samples
            # would become registrations and residual-only continuation can
            # trace q**d tolerance valleys many atol from the actual curve.
            # The owning face makes Psi=0 square: polish with that parameter
            # fixed (including the bounded multiplicity fallback), then admit
            # only a roundoff-scale root.  Distinct genuine roots remain
            # distinct; repeated samples collapse in `_dedup_crossings`.
            fixed_axis = axis if owner_is_s1 else axis + 2
            polished, pres, _ = _ssx_correct_fixed(
                S1_h, S2_h, stuv,
                fixed_axis=fixed_axis, fixed_value=float(side),
                rational=rational,
            )
            if pres > _strict_ssx_root_tol(
                    S1_h, S2_h, rational=rational):
                continue
            stuv = polished
            multiplicity_polished = bool(np.max(np.abs(
                stuv - raw_stuv)) > 64.0 * np.finfo(float).eps)
            xyz = eval_surface(
                S1_h, stuv[0], stuv[1], rational=rational)
            # A certified collapsed EDGE is a positive-dimensional
            # parameter fiber, not a regular crossing.  Do not generalize
            # this to every Sigma=0 point: ordinary C1 points can be regular
            # on the 4D Psi curve and remain valid branch seeds.
            if (_on_collapsed_boundary_fiber(
                    S1_h, stuv[0], stuv[1], rational=rational)
                    or _on_collapsed_boundary_fiber(
                        S2_h, stuv[2], stuv[3], rational=rational)):
                continue
            face_id = axis if owner_is_s1 else axis + 2
            tang, _, _ = _ssx_tangent_4d(S1_h, S2_h, stuv[0], stuv[1], stuv[2], stuv[3], rational=rational)
            crossings.append(BoundaryPoint(
                stuv=stuv, xyz=xyz, face=(face_id, side),
                tangent_raw=tang,
                multiplicity_polished=multiplicity_polished))

        for ovl in result.get('overlaps', []):
            tr = ovl.get('t_range', (0.0, 1.0))
            # ENDPOINT RESOLUTION (corner-sharing bilinear repro): the
            # claim's u/v_range endpoints are NOT paired with the t_range
            # endpoints in any meaningful order for a curve-on-surface
            # overlap — resolve each t-endpoint's true surface preimage by
            # point inversion instead of trusting the index pairing.
            uv_pairs = []
            for t_end in (tr[0], tr[1]):
                cpt = eval_curve(iso, float(t_end), rational=rational)
                uv_pairs.append(_invert_point_on_surface(
                    other_surf, cpt, rational=rational))
            stuv_s = _map_csx_to_stuv(axis, side, tr[0],
                                      uv_pairs[0][0], uv_pairs[0][1],
                                      owner_is_s1)
            stuv_e = _map_csx_to_stuv(axis, side, tr[1],
                                      uv_pairs[1][0], uv_pairs[1][1],
                                      owner_is_s1)
            face_id = axis if owner_is_s1 else axis + 2
            # GEOMETRIC VERIFICATION of the overlap claim (corner-sharing
            # bilinear repro: CSX claimed the whole s=1 edge "overlaps" a
            # SINGLE surface point — t_range (0,1) with degenerate
            # u/v_range — and the index-paired endpoints landed ~39 model
            # units off the intersection). The endpoint pairing above is
            # index-based and unverified, and everything downstream trusts
            # it: `_overlaps_to_branches` ships exactly this stuv chord as
            # a 2-point branch, and the crossing filter below DELETES
            # genuine crossings near the claimed endpoints (which starved
            # a genuine branch down to a stub). Accept the claim only if
            # the SHIPPED chord is on both surfaces: residual
            # |S1(s,t) − S2(u,v)| ≤ 2·atol (matching ladder) at 5 chord
            # samples. A rejected claim contributes NOTHING — no overlap,
            # no endpoint crossings, no crossing filtering; the face's
            # 'isolated' roots (reported independently by CSX) remain the
            # source of genuine seeds there.
            chord_ok = True
            for _lam in (0.0, 0.25, 0.5, 0.75, 1.0):
                _sm = (1.0 - _lam) * stuv_s + _lam * stuv_e
                _p1 = eval_surface(S1_h, _sm[0], _sm[1], rational=rational)
                _p2 = eval_surface(S2_h, _sm[2], _sm[3], rational=rational)
                if float(np.linalg.norm(_p1 - _p2)) > 2.0 * atol:
                    chord_ok = False
                    break
            if not chord_ok:
                continue
            overlaps.append(BoundaryOverlap(stuv_start=stuv_s, stuv_end=stuv_e,
                                            face=(face_id, side)))
            # Also add endpoints as crossings (they connect to interior branches)
            xyz_s = eval_surface(S1_h, stuv_s[0], stuv_s[1], rational=rational)
            xyz_e = eval_surface(S1_h, stuv_e[0], stuv_e[1], rational=rational)
            tang_s, _, _ = _ssx_tangent_4d(S1_h, S2_h, stuv_s[0], stuv_s[1], stuv_s[2], stuv_s[3], rational=rational)
            tang_e, _, _ = _ssx_tangent_4d(S1_h, S2_h, stuv_e[0], stuv_e[1], stuv_e[2], stuv_e[3], rational=rational)
            crossings.append(BoundaryPoint(stuv=stuv_s, xyz=xyz_s, face=(face_id, side),
                                           tangent_raw=tang_s))
            crossings.append(BoundaryPoint(stuv=stuv_e, xyz=xyz_e, face=(face_id, side),
                                           tangent_raw=tang_e))

    # Faces from S1 boundaries
    for s1_axis in (0, 1):
        for side in (0, 1):
            iso = S1_h[0 if side == 0 else -1, :, :] if s1_axis == 0 \
                else S1_h[:, 0 if side == 0 else -1, :]
            _process_face(iso, S2_h, s1_axis, side, owner_is_s1=True)

    # Faces from S2 boundaries
    for s2_axis in (0, 1):
        for side in (0, 1):
            iso = S2_h[0 if side == 0 else -1, :, :] if s2_axis == 0 \
                else S2_h[:, 0 if side == 0 else -1, :]
            _process_face(iso, S1_h, s2_axis, side, owner_is_s1=False)

    crossings = _dedup_crossings(crossings, atol)
    overlaps = _dedup_overlaps(overlaps, atol)

    # Remove crossings that are endpoints of overlaps (redundant — overlap covers them)
    if overlaps:
        filtered = []
        for c in crossings:
            is_ovl_endpoint = False
            for ovl in overlaps:
                if (np.linalg.norm(c.stuv - ovl.stuv_start) < atol or
                        np.linalg.norm(c.stuv - ovl.stuv_end) < atol):
                    is_ovl_endpoint = True
                    break
            if not is_ovl_endpoint:
                filtered.append(c)
        crossings = filtered

    return crossings, overlaps


def _dedup_crossings(crossings, atol):
    """Unify crossings with identical stuv.

    Design §5 Invariant C: two crossings with identical `stuv` (within tolerance)
    represent the same 4D point and must be unified. Two crossings with close xyz
    but distinct stuv are legitimate (a self-intersection, a fold, two branches
    crossing in 3-space) and are kept separate.

    The common source of stuv duplicates is a crossing on the shared boundary of
    two adjacent S1 / S2 faces, which CSX finds independently from each side.
    """
    if len(crossings) <= 1:
        return crossings

    deduped = []
    for c in crossings:
        duplicate = next((d for d in deduped
                          if np.linalg.norm(c.stuv - d.stuv) < atol), None)
        if duplicate is None:
            deduped.append(c)
        else:
            # Preserve evidence that CSX returned a tolerance cluster around
            # this repeated root even when the exact member was inserted
            # first.  If no strict branch later leaves the root, topology is
            # still explicitly partial (positive-gap endpoint-touch control).
            duplicate.multiplicity_polished = bool(
                duplicate.multiplicity_polished or c.multiplicity_polished)
    return deduped


def _dedup_overlaps(overlaps, atol):
    """Remove duplicate boundary overlaps.

    An overlap whose endpoints both lie on boundaries of both surfaces
    is found from both sides. Dedup by checking if start/end stuv match
    (in either order).
    """
    if len(overlaps) <= 1:
        return overlaps

    deduped = []
    for ovl in overlaps:
        is_dup = False
        for d in deduped:
            # Check same direction
            same = (np.linalg.norm(ovl.stuv_start - d.stuv_start) < atol and
                    np.linalg.norm(ovl.stuv_end - d.stuv_end) < atol)
            # Check reversed direction
            rev = (np.linalg.norm(ovl.stuv_start - d.stuv_end) < atol and
                   np.linalg.norm(ovl.stuv_end - d.stuv_start) < atol)
            if same or rev:
                is_dup = True
                break
        if not is_dup:
            deduped.append(ovl)
    return deduped


# ---------------------------------------------------------------------------
# Level 3: Monotonicity classification
# ---------------------------------------------------------------------------

def _tpsi_to_numpy(T):
    """Convert TΨᵢ from nested list (as returned by minors_Tpsi_from_control_nets) to numpy array."""
    return np.asarray(T, dtype=np.float64)


def _check_monotonicity(T1, T2, T3, T4):
    """Check if any TΨᵢ has all coefficients of one sign (definite sign).

    If so, the 4D intersection curve is monotonic in that variable,
    which guarantees no interior loops.

    Returns
    -------
    (True, axis_index) if monotonic (axis_index 0-3 tells which TΨᵢ is definite)
    (False, None) if all TΨᵢ straddle zero
    """
    for i, T in enumerate([T1, T2, T3, T4]):
        T_arr = _tpsi_to_numpy(T)
        t_min = float(np.min(T_arr))
        t_max = float(np.max(T_arr))
        # Non-negative or non-positive → no sign change → monotonic
        # (touching zero at a boundary is not a sign change)
        if t_min >= 0 or t_max <= 0:
            return True, i
    return False, None


def _check_tangency(
        T1, T2, T3, T4, S1, S2, box, *, rational=False, atol=1e-8):
    """Check if TΨ=0 has a simultaneous solution in the box.

    Uses the Krawczyk interval-Newton operator from _deflate.py to certify
    whether the 4-equation system {T1=0, T2=0, T3=0, T4=0} has a root.

    Returns
    -------
    True if tangency is confirmed (root certified or witness found),
    False if no tangent point exists in the box.
    None if undetermined (Krawczyk couldn't decide).
    """
    # The interval-capable DeflatedSystem below historically accepted
    # Euclidean polynomial control nets.  For genuinely rational surfaces,
    # per-control-point P/w is the wrong surface, so use the exact float
    # quotient evaluator.  Failure remains inconclusive (sound fall-through
    # to subdivision); only a converged zero returns True.
    if rational:
        try:
            from mmcore.numeric.intersection.ssx._ssx5_singular import (
                hull_excludes_zero)
            if any(hull_excludes_zero(np.asarray(T, dtype=np.float64))
                   for T in (T1, T2, T3, T4)):
                return None
            gn, _ = _delta_float_gn(
                T1, T2, T3, T4, S1, S2, rational=True, atol=atol)
            mid = np.array([0.5 * (lo + hi) for lo, hi in box],
                           dtype=np.float64)
            return True if gn(mid) is not None else None
        except (np.linalg.LinAlgError, FloatingPointError):
            return None

    from mmcore.numeric.bern import bern_eval as _bern_eval
    from mmcore.numeric.ndinterval import interval as iv_interval, get_iarray
    from mmcore.numeric.intersection._deflate import (
        DeflatedSystem, build_square_from_subset, isolate_roots_krawczyk,
        gauss_newton_witness, _box_from_any,
    )

    try:
        # Convert control nets to interval arrays
        P1_iv = get_iarray(S1, S1)
        P2_iv = get_iarray(S2, S2)
        T1i = np.asarray(T1, dtype=iv_interval)
        T2i = np.asarray(T2, dtype=iv_interval)
        T3i = np.asarray(T3, dtype=iv_interval)
        T4i = np.asarray(T4, dtype=iv_interval)

        B_iv = tuple(iv_interval(lo, hi) for lo, hi in box)

        sys = DeflatedSystem(
            P1=P1_iv, P2=P2_iv, T=(T1i, T2i, T3i, T4i),
            bern_eval=_bern_eval, interval_ctor=iv_interval,
        )

        Bf = _box_from_any(B_iv)

        # Quick interval range check: if any TΨᵢ excludes 0 on the box,
        # design §6 step 3 says treat as None at this point — coef-hull
        # monotonicity (§1.2) is looser than polynomial-range interval,
        # so the test *can* fire in practice even after cheap certificates
        # failed. Functionally equivalent to False here (both fall through
        # to subdivision), but None matches the design's wording.
        T_box = sys.T_box(Bf)
        for Ti_range in T_box:
            lo, hi = Ti_range
            if lo > 0 or hi < 0:
                return None

        # Quick Gauss-Newton witness (few iterations — fast rejection for non-tangent)
        ok, xw, fn = gauss_newton_witness(sys, Bf, tol_f=1e-8, max_iter=8)
        if ok:
            return True  # Found a tangent point

        # If Gauss-Newton didn't converge but residual is very large,
        # it's clearly not tangent — no need for expensive Krawczyk
        if fn > 1.0:
            return False

        return None  # Undetermined — let domain decomposition handle it

    except Exception:
        return None  # Undetermined due to error


def _tangency_witness(cell, atol, *, enumerate_all=True):
    """Gauss-Newton witness point(s) of the deflated system Δ = Ψ ∩ TΨ on the
    cell (local [0,1]⁴ coords). Returns (ok, roots, best_residual,
    exhausted) where `roots` is a list of DISTINCT local witness points, the
    box-center start's root first when it converges, and `exhausted`
    surfaces `solve_zero_dim`'s budget flag (always False when
    `enumerate_all=False` — no enumeration ran). Callers must treat
    `exhausted=True` with several roots as the 1-dimensional-Δ signature
    (ledger L6, `_delta_roots_curve_like`), not as a complete enumeration.

    `enumerate_all=False` runs ONLY the center Gauss-Newton witness and
    skips the `solve_zero_dim` enumeration — for call sites where Δ's zero
    set may be 1-dimensional (a tangent CURVE, e.g. the legacy crossed
    saddles): enumerating it burns the whole max_cells budget to return
    ptol-spaced samples of the curve (measured: 2.28 s for 69 samples on
    the crossed-saddles top cell vs ~150 ms for the entire case).

    Mirrors `_check_tangency`'s DeflatedSystem construction exactly (same
    interval nets, same local box) but runs the witness tighter
    (tol_f=1e-10, max_iter=24): `_check_tangency` only needs a fast
    yes/no, these points are EMITTED as typed singularities' coordinates.
    Kept separate so `_check_tangency`'s ternary contract (and the
    diagnostic scripts that monkeypatch it) stay untouched.

    Enumeration: one cell can hold SEVERAL isolated tangencies (e.g.
    z = 16·((s-0.45)(s-0.9))² + (2t-1)² touches z=0 twice, both touches in
    the crossing-less TOP cell). A single center start converges into one
    basin and the emission branch `continue`s, silently dropping the rest —
    and multistart GN cannot fix that: between two touches the height
    function has a critical point where all four TΨ rows vanish but Ψ ≠ 0,
    a genuine local minimum of ‖Δ‖ that traps every start on the far side
    (measured: all s=0.75 lattice starts die there at ‖Δ‖≈4e-2). So after
    the primary center witness, enumerate ALL Δ-roots with `solve_zero_dim`
    on the nets {Ψ·3, TΨ·4} — Bernstein hull exclusion prunes that trap
    sheet outright (its Ψ hull excludes 0). If the budget exhausts (Δ's
    zero set 1-dimensional, e.g. a tangent LOOP — Task 5's territory),
    `roots` is a valid lower bound: every entry is still a converged
    witness. Cost is confined to genuine crossing-less tangent cells (zero
    across coverage cases 5–11 and the legacy tangential case).
    """
    from mmcore.numeric.intersection.ssx._ssx5_singular import (
        BoxNet, psi_vector_net, solve_zero_dim,
    )
    from mmcore.geom._nurbs_param_tol import bez_surface_param_tolerance
    try:
        gn, _ = _delta_float_gn(
            cell.T1, cell.T2, cell.T3, cell.T4,
            cell.g1.surface, cell.g2.surface, rational=True, atol=atol)

        def _gn(x0):
            seed = np.full(4, 0.5) if x0 is None else x0
            xw_ = gn(seed)
            return (np.asarray(xw_, dtype=np.float64)
                    if xw_ is not None else None)

        def _xyz(x):
            return eval_surface(cell.g1.surface, x[0], x[1], rational=True)

        roots = []
        exhausted = False
        first = _gn(None)          # primary witness: box-center start
        if first is not None:
            roots.append(first)
        if enumerate_all:
            ps, pt = bez_surface_param_tolerance(cell.g1.surface, atol, rational=True)
            pu, pv = bez_surface_param_tolerance(cell.g2.surface, atol, rational=True)
            # 1e-9 per-axis floor: guards degenerate nets from unbounded span/ptol ratios
            ptol = np.maximum(
                np.array([float(ps), float(pt), float(pu), float(pv)]), 1e-9)
            G = psi_vector_net(cell.g1.surface, cell.g2.surface)
            nets = [BoxNet(G[..., k:k + 1], axes=(0, 1, 2, 3)) for k in range(3)]
            nets += [BoxNet(np.asarray(T, dtype=np.float64)[..., None],
                            axes=(0, 1, 2, 3))
                     for T in (cell.T1, cell.T2, cell.T3, cell.T4)]
            # max_cells=2000: interim cost cap; 1-dim Δ-sets (tangent curves/loops) are Task 5's territory
            sols, exhausted = solve_zero_dim(nets, _gn, ptol,
                                             max_cells=2000, dedup_xyz=_xyz,
                                             atol=atol, max_results=64,
                                             charge_box=((
                                                 lambda n: cell.work_budget.charge_cells(
                                                     n, "singular"))
                                                 if cell.work_budget is not None
                                                 else None))
            for sol in sols:
                # same destructive-dedup rule as solve_zero_dim's own _dup:
                # 1·ptol per-axis box AND xyz <= atol
                sol_xyz = _xyz(sol)
                if not any(np.all(np.abs(sol - r_) <= ptol)
                           and float(np.linalg.norm(sol_xyz - _xyz(r_))) <= atol
                           for r_ in roots):
                    roots.append(sol)
        best_fn = min((float(np.linalg.norm(gn.residual(r_)))
                       for r_ in roots), default=np.inf)
        return bool(roots), roots, best_fn, exhausted
    except (np.linalg.LinAlgError, FloatingPointError):
        # Numerical failure of the witness/enumeration — an honest "could
        # not certify" (ok=False; the caller falls through to subdivision).
        # Deliberately NOT a blanket `except Exception`: contract errors
        # (e.g. psi_vector_net shape mismatches, solve_zero_dim's
        # empty-nets ValueError) are programming bugs and must propagate,
        # not silently degrade every tangency emission to a miss.
        return False, [], np.inf, False


def _delta_roots_curve_like(roots, exhausted):
    """Ledger L6(i): True when a Δ-root enumeration looks like ptol-ladder
    SAMPLES of a 1-dimensional zero set (a tangent curve/loop) rather than
    isolated tangencies. Two signatures, same convention as `c1_pass`'s
    curve_flag:

    - more distinct roots than any plausible multi-touch cell (> 12), or
    - a blown budget with several roots (`exhausted=True` means the
      subdivision frontier never emptied — on a genuinely 0-dimensional
      set the hull exclusion empties it well within the budget; measured
      on the off-lattice tangent ring (z=(r²-0.04)² about (0.3,0.3)):
      top-cell enumeration 16 deduped roots exhausted, crossing-bearing
      off-curve floods 17-39 sols exhausted, vs 1-2 roots exhausted=False
      on every isolated-touch fixture).

    Consumers emit NO tangent_points from such a cell — 1-dim tangencies
    are owned by the deflation/tracing machinery (Φ-tracer, Φ∩L seeding),
    and the flood samples are exactly what the post-assembly subsumption
    filter would have to delete again (and provably does NOT fully delete
    when the traced polyline is locally sparse or tracing failed: measured
    4 surviving on-ring debris points of ~50 emitted pre-fix).
    """
    return len(roots) > 12 or (exhausted and len(roots) > 1)


def _stuv_in_overlap_boxes(stuv_g, overlap_boxes):
    """Ledger L6(ii): True if the GLOBAL 4D point lies inside any detected
    overlap region's parametric box (see `_overlap_region_boxes`)."""
    if not overlap_boxes:
        return False
    p = np.asarray(stuv_g, dtype=np.float64)
    return any(bool(np.all(p >= B[:, 0]) and np.all(p <= B[:, 1]))
               for B in overlap_boxes)


def _overlap_region_boxes(boundary_overlaps, S1_h, atol, unify_tol):
    """Ledger L6(ii): padded 4D parametric AABBs of the detected coplanar
    overlap REGIONS, for suppressing tangent_point emission in their
    interior (paper Fig. 8: the overlap interior is a 2-dimensional C2 set;
    every point of it is a Δ-root, so any witness converging there emits a
    phantom "isolated" touch — measured: plane_patch(0,2) vs
    plane_patch(1,3) emitted the strip's dead center (0.75,0.5,0.25,0.5)).

    The overlap machinery stores only the region's BOUNDARY segments
    (`BoundaryOverlap` start/end stuv from the 8 boundary-CSX calls), not
    region boxes — reconstruct minimally: group segments into connected
    components (endpoints within the matching ladder: per-axis unify_tol
    AND xyz <= 2*atol), then take each component's stuv AABB padded by
    unify_tol per axis. For a partial-overlap strip the component box IS
    the strip's parametric box (the boundary segments span it).

    Known limits (accepted, documented): (a) segment (u,v)-images are taken
    from endpoints only — a strongly curved overlap boundary can bulge
    outside the endpoint AABB and under-suppress; (b) a genuine isolated
    touch inside the AABB of a non-convex overlap region but outside the
    region itself would be over-suppressed (geometrically exotic: the
    surfaces already coincide on the region); (c) legacy overlap
    bookkeeping can store corrupt other-surface stuv (the L3 4-vs-2 gap),
    which skews those axes' extents. All three degrade toward the
    PRE-EXISTING behaviors (phantom kept / point subsumed), never corrupt
    branch geometry.
    """
    if not boundary_overlaps:
        return []
    segs = []
    for ovl in boundary_overlaps:
        a = np.asarray(ovl.stuv_start, dtype=np.float64)
        b = np.asarray(ovl.stuv_end, dtype=np.float64)
        axyz = eval_surface(S1_h, a[0], a[1], rational=True)
        bxyz = eval_surface(S1_h, b[0], b[1], rational=True)
        segs.append((a, b, axyz, bxyz))
    n = len(segs)
    parent = list(range(n))

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _ends_match(pa, pxyz, qa, qxyz):
        return (np.all(np.abs(pa - qa) <= unify_tol)
                and float(np.linalg.norm(pxyz - qxyz)) <= 2.0 * atol)

    for i in range(n):
        for j in range(i + 1, n):
            ia, ib, iaxyz, ibxyz = segs[i]
            ja, jb, jaxyz, jbxyz = segs[j]
            if (_ends_match(ia, iaxyz, ja, jaxyz) or _ends_match(ia, iaxyz, jb, jbxyz)
                    or _ends_match(ib, ibxyz, ja, jaxyz)
                    or _ends_match(ib, ibxyz, jb, jbxyz)):
                parent[_find(i)] = _find(j)

    comps: dict = {}
    for i in range(n):
        comps.setdefault(_find(i), []).append(i)
    boxes = []
    for idxs in comps.values():
        pts = np.array([p for i in idxs for p in (segs[i][0], segs[i][1])])
        B = np.stack([pts.min(axis=0) - unify_tol,
                      pts.max(axis=0) + unify_tol], axis=1)   # (4, 2)
        boxes.append(B)
    return boxes


def _emit_tangent_roots(cell, atol, unify_tol, all_singularities,
                        *, enumerate_all=True, overlap_boxes=None,
                        defer_inconclusive=False):
    """Run the Δ = Ψ ∩ TΨ witness on a tangent cell and emit every distinct
    root as a 'tangent_point' singularity into `all_singularities`.

    Shared by all THREE tangency emission sites of the subdivision loop —
    crossing-less (isolated touches), loop-free with all-four T hulls
    containing 0 (touches ON the subdivision cut lattice), and
    crossing-bearing (tangent cell whose boundary transversal arms pierce;
    passes enumerate_all=False, see _tangency_witness). The same physical
    touch is re-confirmed by neighbor cells, by tangent descendants at
    several depths, and possibly by several arms; the dedup
    (matching-ladder: unify_tol per-axis box AND 2·atol xyz) collapses all
    re-confirmations onto ONE emitted point per touch.

    Two emission suppressions (ledger L6), neither affecting the return:

    - 1-dim Δ-sets: when the full enumeration carries the curve signature
      (`_delta_roots_curve_like`: > 12 roots, or exhausted with several),
      the roots are ptol-ladder samples of a tangent CURVE, not isolated
      touches — emit NOTHING and let the caller's fall-through subdivide;
      the deflation/tracing machinery owns 1-dim tangencies (descendants
      become crossing-bearing and Φ-trace the curve as a `tangential`
      branch that legitimately subsumes any residual on-curve witness).
    - overlap interiors: a root inside a detected coplanar overlap
      region's parametric box (`overlap_boxes` from
      `_overlap_region_boxes`) is a sample of a 2-dimensional C2 set the
      overlap branches already report — skip it (paper Fig. 8; the strip
      interior is far from every overlap boundary POLYLINE, so the
      post-assembly subsumption filter cannot catch it).

    Returns `(ok, roots)`: `ok` from `_tangency_witness` (True iff at least
    one converged witness exists — the crossing-less arm's size gate needs
    it) and the LOCAL witness points — Task 5's Φ∩L seeding consumes
    `roots[0]` in the crossing-less arm (`_choose_phi_equations` seed).
    Suppressed roots stay in `roots`: they are genuine Δ-roots and valid
    seeds; only their typing as isolated points is wrong.
    """
    ok, roots, _fn, exhausted = _tangency_witness(
        cell, atol, enumerate_all=enumerate_all)
    if exhausted and cell.work_budget is not None:
        # A locally capped 0/1-root enumeration is still partial; only the
        # root itself is certified, not the absence of another Delta root.
        cell.work_budget.mark_incomplete()
    local_ptol4 = _cell_ptol4(cell, atol)
    typed_roots = list(roots)
    inconclusive_roots = []
    if roots:
        try:
            dimension_gn, _ = _delta_float_gn(
                cell.T1, cell.T2, cell.T3, cell.T4,
                cell.g1.surface, cell.g2.surface,
                rational=True, atol=atol)
        except (np.linalg.LinAlgError, FloatingPointError):
            dimension_gn = None
        classified = []
        for root in roots:
            if dimension_gn is None:
                local_dimension = None
            else:
                local_dimension = _delta_root_local_dimension(
                    dimension_gn, root, local_ptol4,
                    charge_work=((lambda n: cell.work_budget.charge_cells(
                        n, "singular_dimension"))
                        if cell.work_budget is not None else None))
            if local_dimension is False:
                classified.append(root)   # full rank => locally isolated
            elif local_dimension is None:
                if defer_inconclusive:
                    # A loop-free crossing cell can resolve this ambiguity
                    # immediately below by tracing a strict tangential path
                    # through the root.  Delay (do not erase) the incomplete
                    # status until that stronger geometric evidence exists.
                    inconclusive_roots.append(
                        np.asarray(root, dtype=np.float64).copy())
                elif cell.work_budget is not None:
                    cell.work_budget.mark_incomplete()
            # local_dimension=True is a certified curve sample: preserve it
            # in the returned roots for Phi seeding, but do not type it as an
            # isolated tangent point.
        typed_roots = classified
    for xw in typed_roots:
        stuv_g = _local_to_global(np.asarray(xw), cell.box)
        if _stuv_in_overlap_boxes(stuv_g, overlap_boxes):
            continue
        if (_on_collapsed_boundary_fiber(
                cell.g1.surface, xw[0], xw[1], rational=True,
                param_tol=float(max(local_ptol4[0], local_ptol4[1])))
                or _on_collapsed_boundary_fiber(
                    cell.g2.surface, xw[2], xw[3], rational=True,
                    param_tol=float(max(local_ptol4[2], local_ptol4[3])))):
            # Every value of the collapsed edge's free parameter maps to
            # this endpoint.  It belongs to a C1/parameter-fiber set, never
            # an isolated C2 tangent point; relative normal tests are
            # unreliable here because both derivatives can be roundoff-size.
            continue
        if _normals_degenerate_at(cell.g1.surface, cell.g2.surface, xw):
            continue    # L15: Sigma=0 root — C1 candidate, not a C2 touch
        xyz_w = eval_surface(cell.g1.surface, xw[0], xw[1], rational=True)
        if not any(g.kind == "tangent_point"
                   and np.all(np.abs(g.stuv - stuv_g) <= unify_tol)
                   and float(np.linalg.norm(g.xyz - xyz_w)) <= 2.0 * atol
                   for g in all_singularities):
            item = SSXSingularity(
                kind="tangent_point", stuv=stuv_g, xyz=xyz_w)
            if cell.work_budget is None:
                all_singularities.append(item)
            else:
                cell.work_budget.append_output(
                    all_singularities, item, "singularity")
    if defer_inconclusive:
        cell._deferred_delta_roots = inconclusive_roots
    return ok, roots


def _normals_degenerate_at(S1h, S2h, x4) -> bool:
    """True when either surface's parameterization is degenerate at the
    4D point (Sigma_i = du_i x dv_i vanishes relative to |du_i||dv_i| —
    reparameterization-invariant, so cell-LOCAL nets are fine).

    Ledger L15: Sigma = 0 makes ALL FOUR T-Psi minors vanish at any
    crossing, so the Delta-witness converges there even when the 3D
    crossing is TRANSVERSAL (bilinear cone apex vs plane). A vanishing
    normal is a C1 parameterization-cusp candidate, not a C2 tangency —
    c1_pass reports it as cusp/cusp_curve with its `surface` tag."""
    _, du1, dv1 = eval_surface_d1(S1h, x4[0], x4[1], rational=True)
    _, du2, dv2 = eval_surface_d1(S2h, x4[2], x4[3], rational=True)
    n1 = float(np.linalg.norm(np.cross(du1, dv1)))
    n2 = float(np.linalg.norm(np.cross(du2, dv2)))
    s1 = float(np.linalg.norm(du1)) * float(np.linalg.norm(dv1))
    s2 = float(np.linalg.norm(du2)) * float(np.linalg.norm(dv2))

    # This predicate is a topological type test (Sigma_i == 0), not a
    # conditioning heuristic.  A geometric-looking tolerance such as 1e-6
    # silently retypes perfectly regular, merely ill-conditioned surface
    # patches as cusps.  Compare the scale-free sine of the derivative angle
    # only at roundoff scale; a zero derivative is degenerate outright.
    roundoff = 128.0 * np.finfo(np.float64).eps

    def _is_degenerate(normal_norm, derivative_scale):
        if not np.isfinite(normal_norm) or not np.isfinite(derivative_scale):
            return True
        if derivative_scale == 0.0:
            return True
        return normal_norm <= roundoff * derivative_scale

    return _is_degenerate(n1, s1) or _is_degenerate(n2, s2)


def _delta_float_gn(
        T1, T2, T3, T4, S1, S2, *, rational=False, atol=1e-8):
    """Plain-float Gauss-Newton factory on Δ = {Ψ(3), TΨ1..4} over a cell's
    LOCAL [0,1]⁴, from its surface control nets and T tensors.

    With ``rational=True``, ``S1``/``S2`` are homogeneous nets and every
    Ψ/Jacobian evaluation uses the true quotient surfaces.  Per-control-point
    dehomogenization is not a Bezier representation of a non-uniformly
    weighted rational surface (ledger L26; case 14 missed the true generator
    by 240-640·atol on that false polynomial pair).

    Same control flow and acceptance ladder as _deflate.gauss_newton_witness
    (tol_f=1e-10 convergence; step < 1e-12 or max_iter=24 accept at
    fnorm < 1e-8; full-box [0,1]^4 clamp) but evaluated via per-axis
    Bernstein basis rows contracted against a stacked T net and
    eval_surface_d1 for the Ψ rows, instead of the generic interval-capable
    evaluator — ~0.1 ms per start vs ~40 ms (the plan's Risk-7 note:
    "plain-float GN in the enumeration's Newton callback"; measured 77% of
    the blind-band repro's runtime on the generic path).

    The four T nets are degree-elevated to one common shape and stacked
    (trailing dim 4): the GN then needs 5 einsums per Jacobian instead of
    20 (elevation coefficients are convex combinations of the originals,
    so the represented polynomials are identical; consumers using Tstack
    for hull tests only tighten).

    NOTE the 1e-8 stall acceptance bounds how deep a sub-tolerance valley
    this GN can REJECT: the TΨ=0, Ψ≠0 trap sheet of a touch-plus-loop
    valley has ‖Δ‖ = |Ψ| = eps²/4, distinguishable from a true root down
    to eps ≈ 2e-4 — below that the whole feature is sub-7·atol anyway.

    Returns `(gn, Tstack)`; `gn(x0) -> Optional[(4,) local root]`.
    Raises np.linalg.LinAlgError / FloatingPointError like the evaluators
    it wraps — callers keep their own numerical-failure policy.
    """
    from math import comb as _comb
    _Tnets = [np.asarray(T, dtype=np.float64) for T in (T1, T2, T3, T4)]
    _ONE1 = np.ones(1)
    _ZERO1 = np.zeros(1)
    _ES = "i,j,k,l,ijklm->m"
    _binoms: dict = {}

    def _bin(n):
        r = _binoms.get(n)
        if r is None:
            r = np.array([_comb(n, i) for i in range(n + 1)],
                         dtype=np.float64)
            _binoms[n] = r
        return r

    def _elev_mat(n, m):
        # Bernstein degree elevation n -> m (m > n):
        # c'_i = sum_j E[i, j] c_j, E[i, j] = C(n,j)*C(m-n,i-j)/C(m,i)
        E = np.zeros((m + 1, n + 1))
        for i in range(m + 1):
            for j in range(max(0, i - (m - n)), min(n, i) + 1):
                E[i, j] = _comb(n, j) * _comb(m - n, i - j) / _comb(m, i)
        return E

    _degs = [max(T.shape[ax] - 1 for T in _Tnets) for ax in range(4)]
    Tstack = np.empty(tuple(d + 1 for d in _degs) + (4,))
    for k, T in enumerate(_Tnets):
        A = T
        for ax in range(4):
            n = A.shape[ax] - 1
            if n < _degs[ax]:
                A = np.moveaxis(
                    np.tensordot(_elev_mat(n, _degs[ax]), A,
                                 axes=(1, ax)), 0, ax)
        Tstack[..., k] = A

    # The rational T-Psi numerator minors can be O(1e4) while Psi is O(10).
    # Positive per-equation scaling preserves the zero set and prevents the
    # least-squares solve from satisfying the large minor rows at the expense
    # of the actual surface residual.  Keep the returned Tstack unscaled: its
    # Bernstein hulls remain the caller's exact exclusion certificates.
    _Tmax = np.max(np.abs(Tstack), axis=(0, 1, 2, 3))
    _Tscale = np.where(_Tmax > 0.0, _Tmax, 1.0)
    if rational:
        _P1 = S1[..., :-1] / S1[..., -1:]
        _P2 = S2[..., :-1] / S2[..., -1:]
        _joint = np.vstack([_P1.reshape(-1, 3), _P2.reshape(-1, 3)])
        _psi_scale = max(
            1.0,
            float(np.linalg.norm(_joint.max(axis=0) - _joint.min(axis=0))),
        )
    else:
        _psi_scale = 1.0
    _physical_tol = max(0.0, float(atol))

    def _basis_pair(n, xval):
        # Bernstein basis row B_{i,n}(x) and its derivative row
        # B'_{i,n} = n * (B_{i-1,n-1} - B_{i,n-1}).
        if n == 0:
            return _ONE1, _ZERO1
        i = np.arange(n + 1)
        b = _bin(n) * xval ** i * (1.0 - xval) ** (n - i)
        j = np.arange(n)
        bl = _bin(n - 1) * xval ** j * (1.0 - xval) ** (n - 1 - j)
        d = np.empty(n + 1)
        d[0] = -n * bl[0]
        d[n] = n * bl[n - 1]
        if n > 1:
            d[1:n] = n * (bl[:n - 1] - bl[1:])
        return b, d

    def _delta_F(x):
        p1 = eval_surface(S1, x[0], x[1], rational=rational)
        p2 = eval_surface(S2, x[2], x[3], rational=rational)
        bb = [_basis_pair(_degs[ax], x[ax])[0] for ax in range(4)]
        F = np.empty(7)
        F[:3] = (p1 - p2) / _psi_scale
        F[3:] = (np.einsum(
            _ES, bb[0], bb[1], bb[2], bb[3], Tstack) / _Tscale)
        return F

    def _delta_F_J(x):
        p1, du1, dv1 = eval_surface_d1(
            S1, x[0], x[1], rational=rational)
        p2, du2, dv2 = eval_surface_d1(
            S2, x[2], x[3], rational=rational)
        pairs = [_basis_pair(_degs[ax], x[ax]) for ax in range(4)]
        bb = [p[0] for p in pairs]
        F = np.empty(7)
        J = np.zeros((7, 4))
        F[:3] = (p1 - p2) / _psi_scale
        J[:3, 0], J[:3, 1], J[:3, 2], J[:3, 3] = (
            du1 / _psi_scale, dv1 / _psi_scale,
            -du2 / _psi_scale, -dv2 / _psi_scale,
        )
        F[3:] = (np.einsum(
            _ES, bb[0], bb[1], bb[2], bb[3], Tstack) / _Tscale)
        for ax in range(4):
            rows = [pairs[q][1] if q == ax else bb[q] for q in range(4)]
            J[3:, ax] = (np.einsum(
                _ES, rows[0], rows[1], rows[2], rows[3], Tstack)
                / _Tscale)
        return F, J

    def gn(x0):
        x = np.clip(np.asarray(x0, dtype=np.float64), 0.0, 1.0)
        fnorm_prev = None
        for _ in range(24):
            F, J = _delta_F_J(x)
            fnorm = float(np.linalg.norm(F))
            if fnorm < 1e-10:
                if (_physical_psi_residual(x) <= _physical_tol
                        and float(np.linalg.norm(F[3:])) <= 1e-8
                        and _physical_tangency_residual(x) <= 1e-3):
                    return x
                return None
            try:
                dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
            except np.linalg.LinAlgError:
                return None
            if float(np.linalg.norm(dx)) < 1e-12:
                return (x if fnorm < 1e-8
                        and _physical_psi_residual(x) <= _physical_tol
                        and float(np.linalg.norm(F[3:])) <= 1e-8
                        and _physical_tangency_residual(x) <= 1e-3 else None)
            alpha = 1.0
            for _ls in range(12):
                xn = np.clip(x + alpha * dx, 0.0, 1.0)
                fnorm_n = float(np.linalg.norm(_delta_F(xn)))
                if fnorm_prev is None or fnorm_n < fnorm:
                    x = xn
                    fnorm_prev = fnorm_n
                    break
                alpha *= 0.5
        F_final = _delta_F(x)
        return (x if float(np.linalg.norm(F_final)) < 1e-8
                and _physical_psi_residual(x) <= _physical_tol
                and float(np.linalg.norm(F_final[3:])) <= 1e-8
                and _physical_tangency_residual(x) <= 1e-3 else None)

    gn.residual = _delta_F
    gn.jacobian = lambda x: _delta_F_J(
        np.asarray(x, dtype=np.float64))[1]
    gn.roundoff_scale = float(_psi_scale)
    def _physical_psi_residual(x):
        p1 = eval_surface(S1, x[0], x[1], rational=rational)
        p2 = eval_surface(S2, x[2], x[3], rational=rational)
        return float(np.linalg.norm(p1 - p2))

    def _physical_tangency_residual(x):
        """Scale-free rank residual for the true quotient surfaces."""
        if _normals_degenerate_at(S1, S2, x):
            return np.inf
        _, du1, dv1 = eval_surface_d1(
            S1, x[0], x[1], rational=rational)
        _, du2, dv2 = eval_surface_d1(
            S2, x[2], x[3], rational=rational)
        n1 = np.cross(du1, dv1)
        n2 = np.cross(du2, dv2)
        denom = float(np.linalg.norm(n1) * np.linalg.norm(n2))
        if denom <= 1e-300:
            return np.inf
        return float(np.linalg.norm(np.cross(n1, n2)) / denom)

    gn.physical_residual = _physical_psi_residual
    gn.physical_tangency_residual = _physical_tangency_residual
    return gn, Tstack


def _delta_root_local_dimension(gn, root, ptol, charge_work=None):
    """Classify a regular Delta root as isolated or locally one-dimensional.

    Returns ``False`` for a full-rank (locally isolated by IFT) root,
    ``True`` only after a rank-3 root continues to two distinct, strict-
    residual rank-3 roots on opposite sides of its null direction, and
    ``None`` when rank/continuation is inconclusive or its shared work charge
    is denied.  In particular, rank deficiency alone is *not* a curve proof:
    an isolated multiple root may also have a deficient Jacobian.
    """
    x = np.asarray(root, dtype=np.float64)
    ptol = np.maximum(np.asarray(ptol, dtype=np.float64), 1e-12)

    def _rank3(xq):
        try:
            _, svals, Vt = np.linalg.svd(
                np.asarray(gn.jacobian(xq), dtype=np.float64),
                full_matrices=True)
        except (np.linalg.LinAlgError, FloatingPointError):
            return None, None
        if len(svals) < 4 or not np.all(np.isfinite(svals)):
            return None, None
        scale = float(svals[0])
        if scale <= 0.0:
            return None, None
        # Empirical conditioning gap on the committed adversaries:
        # tangent ring s4/s1 <= 6e-9, isolated blind-band touch >= 8e-5.
        # The continuation proof below is still mandatory, so this bar only
        # selects roots worth probing; it never alone deletes output.
        tol = 1e-7 * scale
        rank = int(np.count_nonzero(svals > tol))
        if rank == 4:
            return False, None
        if rank != 3:
            return None, None
        return True, np.asarray(Vt[-1], dtype=np.float64)

    rank3, null = _rank3(x)
    if rank3 is False:
        return False
    if rank3 is None:
        return None
    n = float(np.linalg.norm(null))
    if n <= 1e-30:
        return None
    null = null / n
    weighted = float(np.linalg.norm(null / ptol))
    if weighted <= 1e-30:
        return None
    alpha = 8.0 / weighted
    strict_psi = (4096.0 * np.finfo(float).eps
                  * max(1.0, float(getattr(gn, "roundoff_scale", 1.0))))

    for sign in (-1.0, 1.0):
        seed = x + sign * alpha * null
        if np.any(seed <= 0.0) or np.any(seed >= 1.0):
            return None
        # gn has a fixed 24-iteration cap. Reserve that whole allowance so
        # the shared budget is a hard upper bound even when Newton exits early.
        if charge_work is not None and not charge_work(24):
            return None
        neighbour = gn(seed)
        if neighbour is None:
            return None
        neighbour = np.asarray(neighbour, dtype=np.float64)
        delta = neighbour - x
        if (float(np.max(np.abs(delta) / ptol)) <= 2.0
                or sign * float(np.dot(delta, null)) <= 0.0
                or float(gn.physical_residual(neighbour)) > strict_psi
                or float(np.linalg.norm(gn.residual(neighbour))) > 1e-8):
            return None
        neighbour_rank3, _ = _rank3(neighbour)
        if neighbour_rank3 is not True:
            return None
    return True


def _dist_point_polyline_nd(p, poly):
    """Min distance from an n-D point to a polyline's segments (poly: (N,n))."""
    p = np.asarray(p, dtype=np.float64)
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) == 1:
        return float(np.linalg.norm(poly[0] - p))
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    ap = p[None, :] - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom < 1e-30, 1e-30, denom)
    tt = np.clip(np.einsum("ij,ij->i", ap, ab) / denom, 0.0, 1.0)
    proj = a + tt[:, None] * ab
    return float(np.linalg.norm(proj - p[None, :], axis=1).min())


def _emit_offcurve_tangent_roots(cell, fragments_local, atol, unify_tol,
                                 all_singularities, max_cells=2000,
                                 overlap_boxes=None):
    """Enumerate Δ = Ψ ∩ TΨ roots of a crossing-bearing tangent cell that lie
    OFF the already-traced Φ-fragments, and emit them as tangent_points.

    On a cell holding a tangent CURVE, Δ's zero set contains the whole
    curve — a plain full enumeration walks it at ptol resolution (measured
    2.4 s on the legacy crossed-saddles top cell, all of it emitting curve
    samples the post-assembly subsumption filter deletes again). Instead:

    - Newton attempts are SKIPPED for boxes inside the traced fragments'
      tube — center within 4·ptol/axis (scaled, + the box's own radius) of
      a fragment's stuv polyline AND within 4·atol xyz of its xyz polyline
      (the same matching-ladder pair: a parametric box is not a metric
      ball, so a param-close but metric-far root still gets its Newton).
      Skipped boxes still SUBDIVIDE — a coarse near-curve box can also
      contain the coexisting touch.
    - Boxes far from the tube are explored FIRST (max-heap on the scaled
      stuv distance), and `max_cells` charges ONLY actual Newton attempts
      (`solve_zero_dim`'s skip-aware budget): neither the skipped on-curve
      flood nor its hull-excluded siblings can starve the budget before an
      off-curve touch's boxes subdivide out of the tube slack. Under the
      original per-pop charging this starvation was real — a blind band at
      5–15·atol from the curve (4·atol and below is legitimately subsumed
      by the post-assembly filter); measured post-fix: touches at
      5–20·atol all found at the default budget. What remains heuristic:
      the traversal backstop (`max_cells + 16·charged`) can still truncate
      a slow-converging frontier (`exhausted=True`), and a geometry with
      more off-tube Δ structure than the Newton budget covers can exhaust
      `max_cells` itself — roots found are always valid either way.

    With no fragments (nothing traced), this degrades to the plain
    budget-bounded full enumeration.

    KNOWN LIMIT (review 2d030bb+7ed47c0): this enumeration recovers only
    Δ-ROOTS (tangencies). A coexisting TRANSVERSAL feature that is not on
    Δ — e.g. a small transversal LOOP with no boundary crossings sharing
    this crossing-bearing cell — is still lost: the arm `continue`s
    without subdividing, and the Φ∩L loop seeding runs only on the
    crossing-LESS arm (the Mexican-hat treatment). Subdividing instead
    costs 1349x on tangent curves (see the arm's comment); accepted gap.
    """
    from mmcore.numeric.intersection.ssx._ssx5_singular import (
        ShiftedPositiveNet, VectorBoxNet, psi_vector_net, solve_zero_dim,
    )

    ptol4 = _cell_ptol4(cell, atol)
    scale = 4.0 * ptol4                       # tube radius: matching ladder

    # Only a fragment measured tangent along its WHOLE path defines a tube
    # of positive-dimensional Delta roots.  Deflation also traces the arms
    # through an isolated saddle touch and tags them ``tangential`` by
    # provenance; treating those transversal-away-from-the-touch arms as a
    # curve tube suppresses the very isolated root this pass must recover.
    curve_fragments = [
        f for f in fragments_local
        if len(f.stuv_path) >= 2
        and _fragment_normals_aligned(cell, f, stuv_local=True)
    ]
    stuv_polys = [
        np.asarray(f.stuv_path, dtype=np.float64) / scale[None, :]
        for f in curve_fragments
    ]
    xyz_polys = [
        np.asarray(f.xyz_path, dtype=np.float64)
        for f in curve_fragments
    ]

    def _scaled_param_dist(p4):
        if not stuv_polys:
            return np.inf
        q = np.asarray(p4, dtype=np.float64) / scale
        return min(_dist_point_polyline_nd(q, poly) for poly in stuv_polys)

    def _box_center_radius(bx):
        c = np.array([0.5 * (lo + hi) for lo, hi in bx])
        r = np.array([0.5 * (hi - lo) for lo, hi in bx])
        return c, float(np.linalg.norm(r / scale)), float(np.sum(r / scale))

    # `priority` (at push) and `skip_newton` (at pop) both need the center's
    # scaled param distance to the fragment polylines — cache it per box so
    # each box pays the polyline scan once (~tens of thousands of boxes on
    # a tube flood).
    _pdist_cache: dict = {}

    def _center_pdist(bx, c):
        d = _pdist_cache.get(bx)
        if d is None:
            d = _scaled_param_dist(c)
            _pdist_cache[bx] = d
        return d

    def skip_newton(bx):
        # Both tests carry the box's OWN radius as slack: the skip decision
        # is re-made at every level (skipped boxes still subdivide), so it
        # only needs to be accurate at FINE scales — where the center
        # approximates any root the box could hold. Without the slack the
        # center-only xyz test fails for every coarse near-curve box and
        # each pays a Gauss-Newton attempt (measured: 1069 attempts, 6.2 s
        # on the crossed-saddles top cell with the old interval-evaluator
        # witness). The xyz slack uses the tolerance ladder's own
        # compounding bound (per-axis ptol ~ atol of motion, 1-norm over
        # axes).
        if not stuv_polys:
            return False
        c, box_r2, box_r1 = _box_center_radius(bx)
        if _center_pdist(bx, c) > 1.0 + box_r2:
            return False
        cxyz = eval_surface(cell.g1.surface, c[0], c[1], rational=True)
        return min(_dist_point_polyline_nd(cxyz, poly)
                   for poly in xyz_polys) <= 4.0 * atol * (1.0 + box_r1)

    def priority(bx):
        if not stuv_polys:
            return 0.0
        c, _, _ = _box_center_radius(bx)
        return _center_pdist(bx, c)

    try:
        # Dedicated float Gauss-Newton on Δ = {Ψ(3), TΨ1..4} + the stacked
        # T net (see `_delta_float_gn`) — the enumeration calls it per
        # surviving box, and the solver splits ONE VectorBoxNet (Tstack)
        # per box instead of four BoxNets (the elevated hulls are subsets
        # of the originals — exclusion only tightens).
        _gn, Tstack = _delta_float_gn(cell.T1, cell.T2, cell.T3, cell.T4,
                                      cell.g1.surface, cell.g2.surface,
                                      rational=True, atol=atol)

        def _xyz(x):
            return eval_surface(cell.g1.surface, x[0], x[1], rational=True)

        G = psi_vector_net(cell.g1.surface, cell.g2.surface)
        nets = []
        if cell.F_sq is not None:
            # One-sided sq-dist exclusion (same threshold as
            # _check_min_of_net): kills the off-tube boxes where individual
            # Ψ-component hulls are weak — e.g. crossed saddles share their
            # xy control layout, so Ψ_x = Ψ_y = 0 on a whole 2-dim diagonal
            # set and the component nets exclude almost nothing there
            # (measured: 1069 interval-GN attempts, 6.2 s, without this
            # net; a handful with it). Listed FIRST: it is the strongest
            # off-tube pruner and `any()` short-circuits.
            thresh = (atol * cell.w_scale) ** 2
            nets.append(ShiftedPositiveNet(
                np.asarray(cell.F_sq, dtype=np.float64)[..., None] - thresh,
                axes=(0, 1, 2, 3)))
        # Ψ components and the elevated T stack ride as vector bundles:
        # identical exclusion semantics to seven scalar BoxNets, one
        # de Casteljau split each per box instead of seven total.
        nets.append(VectorBoxNet(G, axes=(0, 1, 2, 3)))
        nets.append(VectorBoxNet(Tstack, axes=(0, 1, 2, 3)))
        sols, _exhausted = solve_zero_dim(
            nets, _gn, ptol4, max_cells=max_cells, dedup_xyz=_xyz, atol=atol,
            max_results=64,
            skip_newton=skip_newton, priority=priority,
            # The skip-aware solver normally permits up to 16x cheap
            # traversal boxes per charged Newton attempt.  A traced
            # positive-dimensional tangent curve can fill that allowance
            # without adding information; bound the traversal itself here.
            max_boxes=max(256, 16 * int(max_cells)),
            charge_box=((lambda n: cell.work_budget.charge_cells(
                n, "singular")) if cell.work_budget is not None else None))
    except (np.linalg.LinAlgError, FloatingPointError):
        return

    # A local solver cap is still a completeness boundary even when it
    # returns zero or one valid roots.  Previously only the curve-like
    # (>1-root) signature reacted to ``_exhausted``; a truncated empty
    # frontier was therefore reported as a complete absence of off-curve
    # tangencies.  Preserve any certified roots, but surface partial status
    # through the one shared SSX budget.
    if _exhausted and cell.work_budget is not None:
        cell.work_budget.mark_incomplete()

    curve_signature = _delta_roots_curve_like(sols, _exhausted)
    if curve_signature:
        # Ledger L6(i): the off-tube "roots" carry the 1-dim signature —
        # this happens when the Φ-tracer failed (no fragments, no tube, so
        # this degraded to a plain full enumeration of the tangent CURVE:
        # measured 33- and 17-point floods with nfrags=0 on the off-lattice
        # tangent ring) or when the tube only partially covers the curve.
        # Each root is still classified below.  The old blanket return was
        # unsafe for a mixed cell: a dense tangent-curve sample cloud can
        # coexist with a full-rank isolated touch, which must survive as an
        # outlier.  Rank+continuation suppresses only the locally 1-D roots.
        pass

    isolated_sols = []
    for sol in sols:
        local_dimension = _delta_root_local_dimension(
            _gn, sol, ptol4,
            charge_work=((lambda n: cell.work_budget.charge_cells(
                n, "singular_dimension"))
                if cell.work_budget is not None else None))
        if local_dimension is True:
            continue                    # certified local tangent-curve sample
        if local_dimension is False:
            isolated_sols.append(sol)   # full-rank Delta root: locally isolated
            continue
        # Rank-deficient but no certified two-sided continuation (or shared
        # budget denial): never turn ambiguity into a false isolated point.
        if cell.work_budget is not None:
            cell.work_budget.mark_incomplete()
    sols = isolated_sols

    for xw in sols:
        stuv_g = _local_to_global(np.asarray(xw), cell.box)
        if _stuv_in_overlap_boxes(stuv_g, overlap_boxes):
            continue
        if (_on_collapsed_boundary_fiber(
                cell.g1.surface, xw[0], xw[1], rational=True,
                param_tol=float(max(ptol4[0], ptol4[1])))
                or _on_collapsed_boundary_fiber(
                    cell.g2.surface, xw[2], xw[3], rational=True,
                    param_tol=float(max(ptol4[2], ptol4[3])))):
            continue
        if _normals_degenerate_at(cell.g1.surface, cell.g2.surface, xw):
            continue    # L15: Sigma=0 root — C1 candidate, not a C2 touch
        xyz_w = eval_surface(cell.g1.surface, xw[0], xw[1], rational=True)
        if not any(g.kind == "tangent_point"
                   and np.all(np.abs(g.stuv - stuv_g) <= unify_tol)
                   and float(np.linalg.norm(g.xyz - xyz_w)) <= 2.0 * atol
                   for g in all_singularities):
            item = SSXSingularity(
                kind="tangent_point", stuv=stuv_g, xyz=xyz_w)
            if cell.work_budget is None:
                all_singularities.append(item)
            else:
                cell.work_budget.append_output(
                    all_singularities, item, "singularity")


def _dist_point_polyline(pxyz, poly):
    """Min distance from a 3D point to a polyline's segments (poly: (N,3))."""
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    ap = pxyz[None, :] - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom < 1e-30, 1e-30, denom)
    tt = np.clip(np.einsum("ij,ij->i", ap, ab) / denom, 0.0, 1.0)
    proj = a + tt[:, None] * ab
    return float(np.linalg.norm(proj - pxyz[None, :], axis=1).min())


def _point_on_branch_both_guards(pxyz, pstuv, poly_xyz, poly_stuv,
                                 atol, unify_tol, S1_h, S2_h):
    """Both-guards ON-a-branch test (ledger L3, module tolerance-ladder
    convention: a parametric box is not a metric ball, and 3D proximity is
    not 4D identity). True iff SOME polyline location is close to the point
    in BOTH spaces:

    - xyz: point-to-segment distance <= 4*atol (the branch polyline is a
      chorded approximation — on-curve points sit up to the 2*atol sagitta
      off it), AND
    - stuv: per-axis |pstuv - stuv interpolated at that SAME segment
      location| <= 2*unify_tol (= 8*ptol/axis: the witness and the traced
      samples are each ~ptol-accurate, and the polyline's stuv chord
      interpolation adds the stuv-space sagitta).

    All xyz-close segments are tested, not just the single nearest one — a
    closed/looping polyline can pass the same xyz neighborhood twice (once
    parameter-near, once parameter-far), and keying on whichever pass is
    marginally nearer would make subsumption order-dependent.

    Without the stuv guard a certified touch whose 4D preimage is FAR from
    the branch's preimage was deleted on 3D proximity alone (skew ruled
    patch: touch at (u,v)=(0.8,0.5) sits 3*atol in xyz from the u=0 overlap
    isoline but du=0.8 away in parameters, with a 640*atol z-wall between
    the sheets -> `singularities == []`).

    Bookkeeping self-check with xyz-only FALLBACK: the stuv guard judges
    against the branch's STORED preimages, so it is only meaningful where
    those are self-consistent — both segment vertices must evaluate (on
    BOTH full surfaces S1_h/S2_h, homogeneous) to within 2*atol of their
    stored xyz. Marched/assembled branches always pass (their xyz IS the
    S1 eval and |S1-S2| <= atol at kept samples; measured 1.8e-13 on the
    skew-ruled overlap). Legacy boundary-overlap branches can carry
    corrupted other-surface params (the known-broken 4-vs-2 overlap
    bookkeeping: stored v=0.5 where the true preimage is 1.0, off by
    ~1e5*atol) — an xyz-close segment that FAILS the self-check falls back
    to the pre-L3 xyz-only subsumption instead of leaking every on-region
    witness the broken stuv cannot vouch for.
    """
    a = poly_xyz[:-1]
    b = poly_xyz[1:]
    ab = b - a
    ap = pxyz[None, :] - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom < 1e-30, 1e-30, denom)
    tt = np.clip(np.einsum("ij,ij->i", ap, ab) / denom, 0.0, 1.0)
    proj = a + tt[:, None] * ab
    d = np.linalg.norm(proj - pxyz[None, :], axis=1)
    close = np.nonzero(d <= 4.0 * atol)[0]
    if close.size == 0:
        return False
    for k in close:
        consistent = True
        for vtx in (k, k + 1):
            p1 = eval_surface(S1_h, poly_stuv[vtx, 0], poly_stuv[vtx, 1],
                              rational=True)
            p2 = eval_surface(S2_h, poly_stuv[vtx, 2], poly_stuv[vtx, 3],
                              rational=True)
            if (float(np.linalg.norm(p1 - poly_xyz[vtx])) > 2.0 * atol
                    or float(np.linalg.norm(p2 - poly_xyz[vtx])) > 2.0 * atol):
                consistent = False
                break
        if not consistent:
            return True         # xyz-close + unreliable stuv: legacy behavior
        stuv_near = poly_stuv[k] + tt[k] * (poly_stuv[k + 1] - poly_stuv[k])
        if bool(np.all(np.abs(stuv_near - pstuv) <= 2.0 * unify_tol)):
            return True
    return False


# ---------------------------------------------------------------------------
# Level 4a: Domain decomposition
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Level 4a: Marching with curvature-adaptive step
# ---------------------------------------------------------------------------

def _ssx_tangent_4d(S1, S2, s, t, u, v, rational=True, direction_hint=None):
    """Compute the 4D tangent direction of the intersection curve at (s,t,u,v).

    The Jacobian J = [S1_s, S1_t, -S2_u, -S2_v] is 3×4.

    When the null space is 1D (rank 3), the tangent is unique.
    When the null space is 2D+ (rank ≤ 2, e.g. planar intersections),
    we project direction_hint onto the null space to pick the right direction.

    Returns (tangent_4d, pt1, pt2) or (None, pt1, pt2) if degenerate.
    """
    pt1, du1, dv1 = eval_surface_d1(S1, s, t, rational=rational)
    pt2, du2, dv2 = eval_surface_d1(S2, u, v, rational=rational)

    J = np.column_stack([du1, dv1, -du2, -dv2])  # (3, 4)
    try:
        _, sigma, Vt = np.linalg.svd(J, full_matrices=True)
    except np.linalg.LinAlgError:
        return None, pt1, pt2

    # For a 3×4 Jacobian, the null space is (4 - rank)-dimensional.
    # With rank 3, null_dim = 1 and the last row of Vt is the tangent.
    # With rank < 3, null_dim > 1 and we need direction_hint.
    tol_sv = max(J.shape) * sigma[0] * 1e-10 if sigma[0] > 0 else 1e-10
    rank = int(np.sum(sigma > tol_sv))
    null_dim = 4 - rank  # for a 3×4 matrix

    if null_dim <= 0:
        return None, pt1, pt2

    if null_dim == 1 or direction_hint is None:
        tangent = Vt[-1]
    else:
        # Project direction_hint onto the null space
        null_vecs = Vt[-null_dim:]  # (null_dim, 4)
        coeffs = null_vecs @ direction_hint
        tangent = null_vecs.T @ coeffs
        norm = np.linalg.norm(tangent)
        if norm < 1e-14:
            tangent = Vt[-1]  # fallback
        else:
            tangent = tangent / norm

    return tangent, pt1, pt2


def _ssx_correct(S1, S2, s, t, u, v, rational=True, max_iter=32, tol=1e-14):
    """Newton corrector: project (s,t,u,v) back onto the intersection curve.

    Minimizes ||S1(s,t) - S2(u,v)|| using damped pseudoinverse steps.
    Tangential intersections converge only linearly in the singular normal
    direction.  The former 12-step cap left otherwise valid continuation
    samples at residuals around 1e-8 (the cylinder/plane tangent-line
    control), which is neither a certified root nor evidence that the curve
    is absent.  Thirty-two bounded steps reach the roundoff certificate for
    that family while preserving a hard per-correction termination bound.

    Returns (s, t, u, v, residual, sin_ang) where:
      - residual = ||S1(s,t) - S2(u,v)|| at the corrected point
      - sin_ang = ||N1 × N2|| / (||N1|| ||N2||) — sin of the angle between
        the two surface normals at the corrected point.

    Why sin_ang matters: when the surfaces approach each other slowly
    (sin_ang ≈ 0), a corrector based purely on `residual < atol` accepts
    points anywhere in the "thick gap" between the surfaces — these can be
    far from the true intersection curve. The xyz distance from the true
    SSX curve at the corrected point is approximately `residual / sin_ang`,
    so callers should require `residual < atol * sin_ang` (with a floor)
    to avoid such false positives.
    """
    for _ in range(max_iter):
        pt1, du1, dv1 = eval_surface_d1(S1, s, t, rational=rational)
        pt2, du2, dv2 = eval_surface_d1(S2, u, v, rational=rational)
        G = pt1 - pt2
        g2 = float(np.dot(G, G))
        # `g2` is a SQUARED norm. Comparing it directly to a length
        # tolerance made the actual stopping threshold sqrt(tol)=1e-7 and
        # let near-tangent tolerance valleys masquerade as corrected roots.
        if g2 < tol * tol:
            break

        J = np.column_stack([du1, dv1, -du2, -dv2])  # (3, 4)
        JT = J.T
        A = JT @ J + 1e-12 * np.eye(4)
        b = -JT @ G
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break

        # Clamp to [0,1]⁴
        s = max(0.0, min(1.0, s + delta[0]))
        t = max(0.0, min(1.0, t + delta[1]))
        u = max(0.0, min(1.0, u + delta[2]))
        v = max(0.0, min(1.0, v + delta[3]))

    pt1f, du1f, dv1f = eval_surface_d1(S1, s, t, rational=rational)
    pt2f, du2f, dv2f = eval_surface_d1(S2, u, v, rational=rational)
    G = pt1f - pt2f
    N1 = np.cross(du1f, dv1f)
    N2 = np.cross(du2f, dv2f)
    n1m = float(np.linalg.norm(N1))
    n2m = float(np.linalg.norm(N2))
    if n1m > 1e-30 and n2m > 1e-30:
        sin_ang = float(np.linalg.norm(np.cross(N1, N2))) / (n1m * n2m)
    else:
        sin_ang = 0.0
    return s, t, u, v, float(np.linalg.norm(G)), sin_ang


def _strict_ssx_root_tol(S1, S2, rational=True):
    """Translation-invariant roundoff scale for an exact Psi zero."""
    if rational:
        p1 = S1[..., :-1] / S1[..., -1:]
        p2 = S2[..., :-1] / S2[..., -1:]
    else:
        p1, p2 = S1, S2
    pts = np.vstack([np.asarray(p1).reshape(-1, 3),
                     np.asarray(p2).reshape(-1, 3)])
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    scale = max(1.0, diag)
    return max(1024.0 * np.finfo(np.float64).eps * scale,
               1e-12 * scale)


def _march_intersection_curve(
    S1, S2,
    stuv_start, stuv_end,
    *,
    atol=1e-3,
    rational=True,
    h_max=None,
    min_step=1e-6,
    max_step=0.25,
    angle_threshold=0.1,   # radians — target angle between consecutive tangents
    max_points=2000,
    stats=None,
):
    """March the intersection curve from stuv_start toward stuv_end.

    Predictor-corrector with curvature-adaptive step sizing:
    - Predictor: step along the 4D tangent direction (null space of Jacobian)
    - Corrector: Newton projection back onto S1=S2
    - Step adaptation: reduce step when curvature is high (large angle between
      consecutive tangents), increase when curvature is low

    Parameters
    ----------
    S1, S2 : ndarray
        Bezier surface control nets.
    stuv_start, stuv_end : ndarray (4,)
        Start and end parameters on the intersection curve.
    atol : float
        Geometric tolerance.
    initial_step : float
        Initial step size in 4D parameter space.
    angle_threshold : float
        Target angle (radians) between consecutive tangent vectors for step adaptation.
    max_points : int
        Safety limit on marching points.

    Returns
    -------
    stuv_path : ndarray (N, 4)
    xyz_path : ndarray (N, 3)
    """
    stuv_pts = [stuv_start.copy()]
    xyz_start = eval_surface(S1, stuv_start[0], stuv_start[1], rational=rational)
    xyz_pts = [xyz_start]

    sag_tol = 2.0 * atol
    if h_max is None:
        h_max = max(0.05 * _local_diag(S1, rational=rational), 4.0 * atol)
    h = 0.25 * h_max
    h_floor = 1e-6 * h_max

    current = stuv_start.copy().astype(np.float64)
    target = stuv_end.copy().astype(np.float64)

    # Direction hint: vector from start to end
    hint = target - current
    hint_norm = np.linalg.norm(hint)
    if hint_norm > 1e-15:
        hint = hint / hint_norm

    # Get initial tangent, using hint to resolve null-space ambiguity
    tang_prev, _, _ = _ssx_tangent_4d(S1, S2, *current, rational=rational, direction_hint=hint)
    if tang_prev is None:
        if stats is not None:
            stats["iterations"] = 0
        return np.array(stuv_pts), np.array(xyz_pts)

    # Orient tangent toward the target
    if np.dot(tang_prev, target - current) < 0:
        tang_prev = -tang_prev

    t3_prev, speed = _tangent_3d(S1, current, tang_prev, rational=rational)

    rejects = 0
    iterations = 0
    for _ in range(max_points):
        iterations += 1
        # xyz-target → stuv step via the local speed (see _march_to_boundary)
        step = max(min_step, min(max_step, h / max(speed, 1e-12)))

        # Check if we're close enough to the target
        dist_to_end = float(np.linalg.norm(current - target))
        predicted = None
        if dist_to_end < max(step * 2, min_step * 4):
            # Close enough — commit the endpoint, unless the closing chord
            # deviates from the curve. In that case retarget the predictor
            # at the stuv halfway point toward the target (real progress;
            # merely halving h can leave the arrival window pinned).
            s, t, u, v, res, sin_ang = _ssx_correct(S1, S2, *target, rational=rational)
            if res < atol * max(sin_ang, 1e-3):
                final = np.array([s, t, u, v])
            else:
                final = target
            final_xyz = eval_surface(S1, final[0], final[1], rational=rational)
            if (h > 2.0 * h_floor
                    and dist_to_end > 4.0 * min_step
                    and float(np.linalg.norm(final_xyz - np.asarray(xyz_pts[-1]))) > atol
                    and _mid_chord_deviates(S1, S2, current, final,
                                            xyz_pts[-1], final_xyz,
                                            atol, sag_tol, rational)):
                h = max(h_floor, h * 0.5)
                predicted = np.clip(current + 0.5 * (target - current), 0.0, 1.0)
                # fall through to the corrector on the halfway point
            else:
                stuv_pts.append(final)
                xyz_pts.append(final_xyz)
                break

        if predicted is None:
            # Predictor: step along tangent, clamped to [0,1]⁴
            predicted = np.clip(current + step * tang_prev, 0.0, 1.0)

        # Corrector: project back onto intersection curve
        s, t, u, v, residual, sin_ang = _ssx_correct(
            S1, S2, predicted[0], predicted[1], predicted[2], predicted[3],
            rational=rational,
        )

        eff_atol = atol * max(sin_ang, 1e-3)
        if residual > eff_atol:
            # Corrector failed — reduce step and retry (with a stagnation
            # escape: clamped steps make these iterations bit-identical).
            rejects += 1
            if rejects >= 25:
                break
            h = max(h_floor, h * 0.5)
            continue

        corrected = np.array([s, t, u, v])

        # Wall-park guard: a step below both the parametric and geometric
        # resolution is no progress (e.g. the predictor clipped against a
        # domain wall the curve exits through) — don't append duplicates.
        if (float(np.linalg.norm(corrected - current)) <= min_step
                and float(np.linalg.norm(
                    eval_surface(S1, s, t, rational=rational)
                    - np.asarray(xyz_pts[-1]))) <= atol):
            rejects += 1
            if rejects >= 25:
                break
            continue

        # Update direction hint toward remaining target
        hint = target - corrected
        hn = np.linalg.norm(hint)
        if hn > 1e-15:
            hint = hint / hn

        # Get new tangent
        tang_new, pt1, _ = _ssx_tangent_4d(S1, S2, s, t, u, v, rational=rational, direction_hint=hint)
        if tang_new is None:
            rejects += 1
            if rejects >= 25:
                break
            h = max(h_floor, h * 0.5)
            continue

        # Orient tangent consistently (same direction as previous)
        if np.dot(tang_new, tang_prev) < 0:
            tang_new = -tang_new

        t3_new, speed_new = _tangent_3d(S1, corrected, tang_new, rational=rational)
        if t3_prev is not None and t3_new is not None:
            cos3 = float(np.clip(np.dot(t3_prev, t3_new), -1.0, 1.0))
            angle3 = float(np.arccos(abs(cos3)))
            chord = float(np.linalg.norm(pt1 - xyz_pts[-1]))
            if chord * angle3 / 8.0 > sag_tol and h > 2.0 * h_floor:
                h = max(h_floor, h * 0.5)
                continue
            if _mid_chord_deviates(S1, S2, current, corrected,
                                   xyz_pts[-1], pt1, atol, sag_tol,
                                   rational) and h > 2.0 * h_floor:
                h = max(h_floor, h * 0.5)
                continue
            # Adapt the xyz target by 3D curvature only.
            if angle3 > 1e-10:
                h = h * min(2.0, max(0.25, angle_threshold / angle3))
            else:
                h = h * 1.5
            h = max(atol, min(h_max, h))

        # Accept the point
        rejects = 0
        current = corrected
        tang_prev = tang_new
        if t3_new is not None:
            t3_prev = t3_new
            speed = speed_new
        stuv_pts.append(current.copy())
        xyz_pts.append(pt1.copy())

    if stats is not None:
        stats["iterations"] = int(iterations)
    return np.array(stuv_pts), np.array(xyz_pts)


def _promote_transversal_boundary_fiber_pair(
        S1, S2, fibers, ordinary_crossings, overlaps, *,
        atol, unify_tol, h_max, max_points=512):
    """Turn one certified collapsed-fiber pair into regular branch seeds.

    A collapsed boundary edge has infinitely many parameter preimages for
    one xyz endpoint.  Two such fibers can nevertheless bound a regular,
    transversal SSI component.  This helper is deliberately conservative:
    it only promotes the pair after a corrected interior witness and a
    contiguous, residual-checked march reaches both canonical endpoints.

    The returned crossings are *candidate seeds*, not a topology-completeness
    proof.  The caller therefore keeps the public result explicitly partial
    whenever boundary fibers participated.
    """
    stats = {"iterations": 0}
    if not fibers:
        return [], stats, None
    if ordinary_crossings or overlaps:
        return [], stats, None

    tol4 = np.maximum(np.asarray(unify_tol, dtype=np.float64), 1e-12)
    unique = []
    for fiber in fibers:
        duplicate = any(
            np.all(np.abs(fiber.stuv - other.stuv) <= tol4)
            and float(np.linalg.norm(fiber.xyz - other.xyz)) <= 2.0 * atol
            for other in unique
        )
        if not duplicate:
            unique.append(fiber)
    if len(unique) != 2:
        return [], stats, None

    first, second = unique
    xyz_separation = float(np.linalg.norm(first.xyz - second.xyz))
    if not np.isfinite(xyz_separation) or xyz_separation <= 16.0 * atol:
        return [], stats, None

    midpoint = 0.5 * (first.stuv + second.stuv)
    corrected = _ssx_correct(
        S1, S2, *midpoint, rational=True, max_iter=20)
    anchor = np.asarray(corrected[:4], dtype=np.float64)
    residual, sin_angle = float(corrected[4]), float(corrected[5])
    guard = np.minimum(np.maximum(tol4, 1e-10), 0.1)
    if (not np.all(np.isfinite(anchor))
            or not np.isfinite(residual) or not np.isfinite(sin_angle)
            or np.any(anchor <= guard) or np.any(anchor >= 1.0 - guard)
            or residual > atol * max(sin_angle, 1e-3)):
        return [], stats, None

    # A tangential interior belongs to the regulated Phi/C2 arm (case 14).
    # Promotion here is solely for a regular Psi curve with singular endpoint
    # parameterizations.
    if sin_angle <= 1e-3:
        return [], stats, None
    if (_on_collapsed_boundary_fiber(
            S1, anchor[0], anchor[1], rational=True,
            param_tol=float(max(tol4[0], tol4[1])))
            or _on_collapsed_boundary_fiber(
                S2, anchor[2], anchor[3], rational=True,
                param_tol=float(max(tol4[2], tol4[3])))):
        return [], stats, None

    promoted = []
    canonical_stuv = []
    for fiber, other in ((first, second), (second, first)):
        stuv = np.asarray(fiber.stuv, dtype=np.float64).copy()
        stuv[:2] = _canonicalize_collapsed_fiber_params(
            S1, stuv[:2], anchor[:2], rational=True,
            param_tol=float(max(tol4[0], tol4[1])))
        stuv[2:] = _canonicalize_collapsed_fiber_params(
            S2, stuv[2:], anchor[2:], rational=True,
            param_tol=float(max(tol4[2], tol4[3])))
        if (not np.all(np.isfinite(stuv))
                or np.any(stuv < -tol4) or np.any(stuv > 1.0 + tol4)):
            return [], stats, None

        fixed_axis, fixed_side = fiber.face
        fixed_value = float(fixed_side)
        if abs(float(stuv[fixed_axis]) - fixed_value) > tol4[fixed_axis]:
            return [], stats, None

        p1 = eval_surface(S1, stuv[0], stuv[1], rational=True)
        p2 = eval_surface(S2, stuv[2], stuv[3], rational=True)
        xyz = 0.5 * (p1 + p2)
        if (not np.all(np.isfinite(xyz))
                or float(np.linalg.norm(p1 - p2)) > 2.0 * atol
                or float(np.linalg.norm(xyz - fiber.xyz)) > 2.0 * atol):
            return [], stats, None

        hint = np.asarray(other.stuv - fiber.stuv, dtype=np.float64)
        # Canonical free coordinates and the corrected interior anchor give a
        # better limiting direction than the arbitrary representatives stored
        # by boundary CSX.
        hint = anchor - stuv if np.linalg.norm(anchor - stuv) > 1e-14 else hint
        tangent, _, _ = _ssx_tangent_4d(
            S1, S2, *stuv, rational=True, direction_hint=hint)
        if tangent is None or not np.all(np.isfinite(tangent)):
            return [], stats, None
        tangent = np.asarray(tangent, dtype=np.float64)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-14:
            return [], stats, None
        if float(np.dot(tangent, hint)) < 0.0:
            tangent = -tangent
        if float(np.dot(tangent, hint)) <= 1e-8 * tangent_norm * np.linalg.norm(hint):
            return [], stats, None

        # Every active boundary component must point into the parameter box.
        for axis in range(4):
            if stuv[axis] <= tol4[axis] and tangent[axis] < -1e-10:
                return [], stats, None
            if stuv[axis] >= 1.0 - tol4[axis] and tangent[axis] > 1e-10:
                return [], stats, None

        _, du1, dv1 = eval_surface_d1(
            S1, stuv[0], stuv[1], rational=True)
        _, du2, dv2 = eval_surface_d1(
            S2, stuv[2], stuv[3], rational=True)
        vel1 = tangent[0] * du1 + tangent[1] * dv1
        vel2 = tangent[2] * du2 + tangent[3] * dv2
        speed_floor = max(1e-12, 1e-10 * xyz_separation)
        if (float(np.linalg.norm(vel1)) <= speed_floor
                or float(np.linalg.norm(vel2)) <= speed_floor
                or float(np.linalg.norm(vel1 - vel2)) > 2.0 * atol):
            return [], stats, None

        canonical_stuv.append(stuv)
        promoted.append(BoundaryPoint(
            stuv=stuv, xyz=xyz, face=fiber.face,
            tangent_raw=tangent, parameter_fiber=True))

    # Connectivity is not inferred from a midpoint alone: require one
    # bounded continuation to reach the second canonical endpoint and check
    # every retained vertex against the physical quotient surfaces.
    path_stuv, path_xyz = _march_intersection_curve(
        S1, S2, canonical_stuv[0], canonical_stuv[1],
        atol=atol, rational=True, h_max=h_max,
        min_step=max(float(np.max(tol4)), 1e-9),
        max_points=max(1, int(max_points)), stats=stats)
    if len(path_stuv) < 2:
        return [], stats, None
    if (np.any(np.abs(path_stuv[-1] - canonical_stuv[1]) > 2.0 * tol4)
            or float(np.linalg.norm(path_xyz[-1] - promoted[1].xyz)) > 2.0 * atol):
        return [], stats, None

    saw_regular_interior = False
    for idx, stuv in enumerate(np.asarray(path_stuv, dtype=np.float64)):
        if not np.all(np.isfinite(stuv)) or np.any(stuv < -tol4) or np.any(stuv > 1.0 + tol4):
            return [], stats, None
        p1, du1, dv1 = eval_surface_d1(
            S1, stuv[0], stuv[1], rational=True)
        p2, du2, dv2 = eval_surface_d1(
            S2, stuv[2], stuv[3], rational=True)
        if (float(np.linalg.norm(p1 - p2)) > 2.0 * atol
                or float(np.linalg.norm(p1 - path_xyz[idx])) > 2.0 * atol):
            return [], stats, None
        n1, n2 = np.cross(du1, dv1), np.cross(du2, dv2)
        denom = float(np.linalg.norm(n1) * np.linalg.norm(n2))
        if denom > 1e-30:
            path_sin = float(np.linalg.norm(np.cross(n1, n2))) / denom
            if path_sin > 1e-3:
                saw_regular_interior = True
    if not saw_regular_interior:
        return [], stats, None
    path_stuv = np.asarray(path_stuv, dtype=np.float64)
    path_xyz = np.asarray(path_xyz, dtype=np.float64)
    path_stuv[0], path_stuv[-1] = promoted[0].stuv, promoted[1].stuv
    path_xyz[0], path_xyz[-1] = promoted[0].xyz, promoted[1].xyz
    return promoted, stats, (path_stuv, path_xyz)


def _ssx_correct_fixed_multiplicity(
        S1, S2, stuv_init, fixed_axis, fixed_value, *, rational, max_iter):
    """High-precision Newton fallback for a multiple boundary root.

    With one parameter fixed, ``Psi=0`` is a square 3-by-3 system.  The
    ordinary damped normal equations intentionally suppress singular values
    below the LM floor; at a multiplicity-d contact that parks at
    ``distance**d ~= roundoff`` -- many geometric tolerances from the root.
    Decimal Bernstein evaluation plus pivoted square Newton retains the
    small *correlated* residual/Jacobian ratio (``q**d / (d*q**(d-1))``)
    without relaxing any residual certificate.  Every loop is statically
    bounded and line-searched, and failure simply returns the best input so
    callers can reject it with their existing strict residual gate.

    This is deliberately a rare fallback, invoked only when the float64
    square Jacobian is numerically rank deficient.
    """
    from decimal import Decimal, localcontext

    S1a = np.asarray(S1, dtype=np.float64)
    S2a = np.asarray(S2, dtype=np.float64)
    total_degree = max(
        (S1a.shape[0] - 1) + (S1a.shape[1] - 1),
        (S2a.shape[0] - 1) + (S2a.shape[1] - 1),
    )

    with localcontext() as ctx:
        ctx.prec = min(160, max(64, 6 * total_degree + 24))
        zero = Decimal(0)
        one = Decimal(1)

        def _dec(value):
            return Decimal.from_float(float(value))

        def _basis_pair(n, x):
            if n == 0:
                return [one], [zero]
            def _pow(value, exponent):
                return one if exponent == 0 else value ** exponent

            b = [Decimal(math.comb(n, i)) * _pow(x, i)
                 * _pow(one - x, n - i)
                 for i in range(n + 1)]
            bm = [Decimal(math.comb(n - 1, i)) * _pow(x, i)
                  * _pow(one - x, n - 1 - i) for i in range(n)]
            db = []
            for i in range(n + 1):
                left = bm[i - 1] if i > 0 else zero
                right = bm[i] if i < n else zero
                db.append(Decimal(n) * (left - right))
            return b, db

        control_cache = {}

        def _controls(S):
            key = id(S)
            out = control_cache.get(key)
            if out is None:
                arr = np.asarray(S, dtype=np.float64)
                out = [[[ _dec(arr[i, j, k]) for k in range(arr.shape[2])]
                        for j in range(arr.shape[1])]
                       for i in range(arr.shape[0])]
                control_cache[key] = out
            return out

        def _surface_d1(S, u, v):
            arr = np.asarray(S, dtype=np.float64)
            bu, dbu = _basis_pair(arr.shape[0] - 1, u)
            bv, dbv = _basis_pair(arr.shape[1] - 1, v)
            C = _controls(S)
            dim = arr.shape[2]
            h = [zero for _ in range(dim)]
            hu = [zero for _ in range(dim)]
            hv = [zero for _ in range(dim)]
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    w = bu[i] * bv[j]
                    wu = dbu[i] * bv[j]
                    wv = bu[i] * dbv[j]
                    for k in range(dim):
                        c = C[i][j][k]
                        h[k] += w * c
                        hu[k] += wu * c
                        hv[k] += wv * c
            if not rational:
                return h[:3], hu[:3], hv[:3]
            W = h[-1]
            if W == zero:
                raise ZeroDivisionError("zero rational weight")
            W2 = W * W
            p = [h[k] / W for k in range(3)]
            du = [(hu[k] * W - h[k] * hu[-1]) / W2 for k in range(3)]
            dv = [(hv[k] * W - h[k] * hv[-1]) / W2 for k in range(3)]
            return p, du, dv

        def _solve3(A, b):
            A = [list(row) + [b[i]] for i, row in enumerate(A)]
            for col in range(3):
                pivot = max(range(col, 3), key=lambda r: abs(A[r][col]))
                if A[pivot][col] == zero:
                    return None
                if pivot != col:
                    A[col], A[pivot] = A[pivot], A[col]
                piv = A[col][col]
                for j in range(col, 4):
                    A[col][j] /= piv
                for r in range(3):
                    if r == col:
                        continue
                    factor = A[r][col]
                    if factor == zero:
                        continue
                    for j in range(col, 4):
                        A[r][j] -= factor * A[col][j]
            return [A[i][3] for i in range(3)]

        x = [_dec(value) for value in stuv_init]
        x[fixed_axis] = _dec(fixed_value)
        free = [i for i in range(4) if i != fixed_axis]
        limit = min(160, max(1, int(max_iter)))

        for _ in range(limit):
            p1, du1, dv1 = _surface_d1(S1, x[0], x[1])
            p2, du2, dv2 = _surface_d1(S2, x[2], x[3])
            G = [p1[k] - p2[k] for k in range(3)]
            old = sum(g * g for g in G)
            if old == zero:
                break
            cols = (du1, dv1,
                    [-value for value in du2],
                    [-value for value in dv2])
            A = [[cols[idx][row] for idx in free] for row in range(3)]
            delta = _solve3(A, [-g for g in G])
            if delta is None:
                break
            dmax = max(abs(value) for value in delta)
            if not dmax.is_finite():
                break
            # Stay in the local Newton basin.  Scaling preserves direction;
            # the line search below still owns monotone residual decrease.
            if dmax > Decimal("0.25"):
                factor = Decimal("0.25") / dmax
                delta = [value * factor for value in delta]

            accepted = False
            step = one
            for _ls in range(24):
                cand = list(x)
                for k, axis in enumerate(free):
                    cand[axis] = min(one, max(zero, x[axis] + step * delta[k]))
                cand[fixed_axis] = _dec(fixed_value)
                q1, _, _ = _surface_d1(S1, cand[0], cand[1])
                q2, _, _ = _surface_d1(S2, cand[2], cand[3])
                new = sum((q1[k] - q2[k]) ** 2 for k in range(3))
                if new < old:
                    x = cand
                    accepted = True
                    break
                step *= Decimal("0.5")
            if not accepted:
                break

        return np.array([float(value) for value in x], dtype=np.float64)


def _ssx_correct_fixed(S1, S2, stuv_init, fixed_axis: int, fixed_value: float,
                       rational: bool = True, max_iter: int = 60, tol: float = 1e-14):
    """Damped Newton with line search solving `Ψ(s,t,u,v) = 0` with
    `stuv[fixed_axis] = fixed_value`.

    Three free parameters, three equations (`S1(s,t) - S2(u,v) = 0`). Used
    by `_march_to_boundary` when the predictor has detected a boundary
    crossing: the crossed axis is clamped to the boundary value and the
    remaining three parameters are solved for exactly, giving the precise
    point at which the intersection curve exits the cell.

    Convergence is QUADRATIC in transversal regions, but only LINEAR (with
    factor close to 1) when the surfaces meet at a glancing angle — there
    Newton can need dozens of iterations to reach the same precision. The
    function therefore:
      - line-searches each step so ||G|| is monotone-decreasing;
      - exits as soon as ||G||² < `tol`² (a true zero in machine precision);
      - else returns the best point found within `max_iter`.

    Returns `(params, residual, sin_ang)`:
      - params : (4,) ndarray
      - residual : ‖S1 - S2‖ at params (xyz units)
      - sin_ang : sin of the angle between the two surface normals at
        params; the xyz distance from the true SSX curve at params is
        approximately `residual / sin_ang`, so callers can validate
        precision via the same chain-rule rule used elsewhere.
    """
    params = [float(x) for x in stuv_init]
    params[fixed_axis] = float(fixed_value)
    free_cols = [i for i in range(4) if i != fixed_axis]

    pt1, du1, dv1 = eval_surface_d1(S1, params[0], params[1], rational=rational)
    pt2, du2, dv2 = eval_surface_d1(S2, params[2], params[3], rational=rational)
    G = pt1 - pt2
    g2 = float(np.dot(G, G))

    for _ in range(max_iter):
        if g2 < tol * tol:
            break

        J = np.column_stack([du1, dv1, -du2, -dv2])  # (3, 4)
        J_free = J[:, free_cols]
        A = J_free.T @ J_free + 1e-14 * np.eye(3)
        b = -J_free.T @ G
        try:
            delta_free = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break

        # Line-search with clamping so ||G|| is monotone-decreasing AND the
        # iterates stay in [0,1]⁴ (the cell-local parameter domain). Free
        # axes drifting outside [0,1] would extrapolate the local Bezier
        # patch off-surface — the residual could go to zero against a
        # nonsensical surface continuation, giving a "false" intersection
        # that's nowhere near the true SSX curve.
        accepted = False
        step_scale = 1.0
        new_params = list(params)
        new_g2 = g2
        new_G = G
        new_pt1 = pt1
        new_du1 = du1
        new_dv1 = dv1
        new_pt2 = pt2
        new_du2 = du2
        new_dv2 = dv2
        for _ls in range(20):
            cand = list(params)
            for k, idx in enumerate(free_cols):
                v = params[idx] + step_scale * float(delta_free[k])
                # Clamp free axis to [0,1] — prevents the corrector from
                # wandering outside the local cell parametrization, which
                # would silently evaluate an extrapolated Bezier surface.
                if v < 0.0:
                    v = 0.0
                elif v > 1.0:
                    v = 1.0
                cand[idx] = v
            # Re-pin the fixed axis (in case rounding drifted it).
            cand[fixed_axis] = float(fixed_value)
            p1c, du1c, dv1c = eval_surface_d1(S1, cand[0], cand[1], rational=rational)
            p2c, du2c, dv2c = eval_surface_d1(S2, cand[2], cand[3], rational=rational)
            Gc = p1c - p2c
            gc2 = float(np.dot(Gc, Gc))
            if gc2 < g2:
                new_params = cand
                new_g2 = gc2
                new_G = Gc
                new_pt1, new_du1, new_dv1 = p1c, du1c, dv1c
                new_pt2, new_du2, new_dv2 = p2c, du2c, dv2c
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
        params = new_params
        g2 = new_g2
        G = new_G
        pt1, du1, dv1 = new_pt1, new_du1, new_dv1
        pt2, du2, dv2 = new_pt2, new_du2, new_dv2

    residual = float(g2 ** 0.5)

    # A small square singular value plus a small residual is the repeated-
    # root trap: LM damping has stopped moving, but residual magnitude alone
    # says nothing about root distance.  Re-polish from the caller's seed in
    # high precision, then recompute every returned measurement in float64.
    J = np.column_stack([du1, dv1, -du2, -dv2])
    J_free = J[:, free_cols]
    try:
        svals = np.linalg.svd(J_free, compute_uv=False)
    except np.linalg.LinAlgError:
        svals = np.empty(0)
    ill_conditioned = bool(
        len(svals) == 3 and svals[0] > 0.0
        and svals[-1] <= 1e-7 * svals[0])
    if ill_conditioned:
        degree_bound = max(
            (S1.shape[0] - 1) + (S1.shape[1] - 1),
            (S2.shape[0] - 1) + (S2.shape[1] - 1),
        )
        try:
            polished = _ssx_correct_fixed_multiplicity(
                S1, S2, stuv_init, fixed_axis, fixed_value,
                rational=rational,
                max_iter=max(64, 8 * degree_bound),
            )
        except (ArithmeticError, FloatingPointError, ValueError):
            polished = None
        if polished is not None:
            params = polished.tolist()
            pt1, du1, dv1 = eval_surface_d1(
                S1, params[0], params[1], rational=rational)
            pt2, du2, dv2 = eval_surface_d1(
                S2, params[2], params[3], rational=rational)
            G = pt1 - pt2
            residual = float(np.linalg.norm(G))

    N1 = np.cross(du1, dv1)
    N2 = np.cross(du2, dv2)
    n1m = float(np.linalg.norm(N1))
    n2m = float(np.linalg.norm(N2))
    if n1m > 1e-30 and n2m > 1e-30:
        sin_ang = float(np.linalg.norm(np.cross(N1, N2))) / (n1m * n2m)
    else:
        sin_ang = 0.0
    return np.array(params, dtype=np.float64), residual, sin_ang


def _detect_boundary_crossing(current, predicted):
    """If `predicted` exits `[0,1]⁴` on any axis, return the axis and value
    (0 or 1) of the FIRST boundary crossed along the interval, plus the
    fraction `α ∈ [0, 1]` at which the crossing occurs. Returns `(None,
    None, None)` if `predicted` is entirely inside `[0,1]⁴`.
    """
    crossed_axis = None
    crossed_value = None
    crossed_alpha = None
    for i in range(4):
        ci = float(current[i])
        pi = float(predicted[i])
        if pi < 0.0:
            denom = pi - ci
            if denom == 0.0:
                continue
            alpha = (0.0 - ci) / denom
            if crossed_alpha is None or alpha < crossed_alpha:
                crossed_axis, crossed_value, crossed_alpha = i, 0.0, alpha
        elif pi > 1.0:
            denom = pi - ci
            if denom == 0.0:
                continue
            alpha = (1.0 - ci) / denom
            if crossed_alpha is None or alpha < crossed_alpha:
                crossed_axis, crossed_value, crossed_alpha = i, 1.0, alpha
    return crossed_axis, crossed_value, crossed_alpha


def _on_boundary(stuv, tol=1e-5):
    """Check if any parameter is at 0 or 1 (on domain boundary).

    The default tolerance `1e-5` reflects the parametric distance at which
    the corrector can no longer push a nearly-on-boundary coordinate any
    closer to an exact 0 or 1. The Ψ=0 curve doesn't pass through the cell
    corner at machine precision, so numerical noise parks the corrected
    point ~1e-6 to 1e-4 from the boundary; a stricter tolerance (1e-8) lets
    the marcher loop forever at the 2000-point safety cap, jittering in
    place. 1e-5 is loose enough to catch the park and tight enough to not
    false-trigger on interior points that happen to pass near a boundary.
    """
    for i in range(4):
        if stuv[i] < tol or stuv[i] > 1.0 - tol:
            return True
    return False


def _tangent_3d(S1, stuv, tang4, rational=True):
    """3D direction AND speed of the intersection curve along the (unit)
    4D tangent: dX = S1_s·ds + S1_t·dt.

    Returns (unit_dir, speed) or (None, 0.0) when degenerate. `speed` is
    the xyz arc length per unit stuv step along `tang4` — the local
    reparameterization factor that converts an xyz step target h into a
    stuv step: step_stuv = h / speed. Sizing steps this way makes sampling
    density a property of the GEOMETRY, invariant under reparameterization
    of either surface; parameter-space curvature (which appears and
    vanishes under rescaling) never enters step sizing.
    """
    _, du1, dv1 = eval_surface_d1(S1, stuv[0], stuv[1], rational=rational)
    d3 = du1 * tang4[0] + dv1 * tang4[1]
    n = float(np.linalg.norm(d3))
    if n < 1e-30:
        return None, 0.0
    return d3 / n, n


def _local_diag(S, rational=True):
    """Diagonal of the Euclidean control-net AABB (fallback xyz scale)."""
    pts = S[..., :-1] / S[..., -1:] if rational else S
    flat = pts.reshape(-1, pts.shape[-1])
    return float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0)))


def _mid_chord_deviates(S1, S2, stuv_a, stuv_b, xyz_a, xyz_b, atol, sag_tol,
                        rational):
    """True if the curve deviates more than sag_tol from the chord at its
    parametric midpoint.

    The endpoint-tangent sagitta estimate is blind to S-shaped spans: an
    inflection inside the step leaves both endpoint tangents parallel
    while the middle bulges. Correcting the stuv midpoint onto the curve
    and measuring its distance to the chord catches exactly that case.
    """
    mid = 0.5 * (np.asarray(stuv_a, dtype=np.float64)
                 + np.asarray(stuv_b, dtype=np.float64))
    ms, mt, mu, mv, mres, msin = _ssx_correct(S1, S2, *mid, rational=rational)
    if mres > atol * max(msin, 1e-3):
        return False  # midpoint correction unreliable — don't judge
    xm = eval_surface(S1, ms, mt, rational=rational)
    a3 = np.asarray(xyz_a, dtype=np.float64)
    b3 = np.asarray(xyz_b, dtype=np.float64)
    ab = b3 - a3
    denom = float(np.dot(ab, ab))
    if denom < 1e-30:
        return False
    tt = float(np.clip(np.dot(xm - a3, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(a3 + tt * ab - xm)) > sag_tol


def _march_to_boundary(
    S1, S2, stuv_start,
    *,
    atol=1e-3,
    rational=True,
    h_init=None,
    h_max=None,
    min_step=1e-6,
    max_step=0.25,
    angle_threshold=0.1,
    max_points=400,
    direction_hint=None,
    sag_tol=None,
    stats=None,
):
    """March from stuv_start until the curve hits a domain boundary [0,1]⁴.

    Like _march_intersection_curve but without a known endpoint.
    Stops when any parameter reaches 0 or 1.

    Step control is xyz-driven (local reparameterization): the marcher
    maintains a target xyz chord length `h`, adapted purely by the 3D
    turning angle and a chord-deviation (sagitta) bound, and converts it
    to a stuv step each iteration via the local speed |dX/d(stuv)| along
    the tangent. Parameter-space curvature never sizes steps — it appears
    and vanishes under reparameterization while xyz accuracy is what
    matters. The stuv step is still clamped to [min_step, max_step] as a
    Newton-basin floor and cell-escape ceiling.

    Returns (stuv_path, xyz_path, exit_info) where exit_info is
    (axis, value) of the boundary face the march exited through, or None
    if the march ended in the interior (failure/truncation).
    """
    iterations = 0
    if sag_tol is None:
        sag_tol = 2.0 * atol
    if h_max is None:
        h_max = max(0.05 * _local_diag(S1, rational=rational), 4.0 * atol)
    h = h_init if h_init is not None else 0.25 * h_max
    h_floor = 1e-6 * h_max

    stuv_pts = [stuv_start.copy()]
    xyz_pts = [eval_surface(S1, stuv_start[0], stuv_start[1], rational=rational)]
    exit_info = None

    current = stuv_start.copy().astype(np.float64)

    # Initial tangent
    tang_prev, _, _ = _ssx_tangent_4d(S1, S2, *current, rational=rational,
                                       direction_hint=direction_hint)
    if tang_prev is None:
        if stats is not None:
            stats["iterations"] = 0
        return np.array(stuv_pts), np.array(xyz_pts), exit_info

    # Orient tangent using hint if provided
    if direction_hint is not None and np.dot(tang_prev, direction_hint) < 0:
        tang_prev = -tang_prev

    t3_prev, speed = _tangent_3d(S1, current, tang_prev, rational=rational)

    rejects = 0
    for iter_num in range(max_points):
        iterations += 1
        step = max(min_step, min(max_step, h / max(speed, 1e-12)))
        predicted = current + step * tang_prev

        crossed_axis, crossed_val, crossed_alpha = _detect_boundary_crossing(
            current, predicted)
        if crossed_axis is not None:
            # Curve left the cell somewhere in (current, predicted).
            # Initial guess: linear interpolation at the crossing fraction,
            # with the crossed axis clamped to its exact boundary value.
            # Then Newton-solve Ψ=0 with that axis held fixed → exact
            # boundary-crossing point.
            stuv_init = current + crossed_alpha * (predicted - current)
            stuv_init[crossed_axis] = crossed_val
            final, fres, fsin = _ssx_correct_fixed(
                S1, S2, stuv_init,
                fixed_axis=crossed_axis, fixed_value=crossed_val,
                rational=rational,
            )
            # Angle-aware acceptance: only commit a converged exit. A
            # refused exit leaves exit_info=None and the caller decides
            # what to do with the partial trace.
            eff_atol = atol * max(fsin, 1e-3)
            if fres > eff_atol:
                break

            final_xyz = eval_surface(S1, final[0], final[1], rational=rational)
            # The exit chord gets the same deviation scrutiny as any other
            # step: the fixed-axis Newton can slide along the face far
            # from the interpolated guess (case 10: a 0.47-long exit chord
            # cutting the corner by 6.5mm). Halving h alone CANNOT shrink
            # this chord — the ray-face intersection init (and hence the
            # deterministic fixed-axis Newton result) is independent of
            # the step length. Instead retarget the predictor at the
            # interior halfway point toward the face and run the normal
            # corrector on it: the march makes real progress toward the
            # boundary and the eventual exit chord shrinks geometrically.
            interior_len = 0.5 * crossed_alpha * step
            if (h > 2.0 * h_floor
                    and interior_len > 0.25 * min_step
                    and _mid_chord_deviates(S1, S2, current, final,
                                            xyz_pts[-1], final_xyz,
                                            atol, sag_tol, rational)):
                h = max(h_floor, h * 0.5)
                predicted = current + interior_len * tang_prev
                # fall through to the interior corrector below
            else:
                stuv_pts.append(final)
                xyz_pts.append(final_xyz)
                exit_info = (crossed_axis, crossed_val)
                break

        # Interior corrector path (also handles the retargeted predictor
        # from a rejected exit chord above).
        s, t, u, v, residual, sin_ang = _ssx_correct(
            S1, S2, *predicted, rational=rational,
        )

        # Angle-aware acceptance: ||r|| < atol alone is misleading when the
        # surfaces approach slowly (small sin_ang). The xyz distance from the
        # true SSX curve is ≈ residual / sin_ang, so require residual to be
        # tighter when surfaces are close to parallel. The floor (1e-3) caps
        # how aggressive the tightening can get — without it, slowly
        # converging segments would never accept any correction at all.
        eff_atol = atol * max(sin_ang, 1e-3)

        if residual > eff_atol:
            # Stagnation escape: with the stuv step clamped at min_step,
            # halving h changes nothing and iterations repeat bit-identically
            # — stop burning the budget and return the partial trace.
            rejects += 1
            if rejects >= 25:
                break
            h = max(h_floor, h * 0.5)
            continue

        corrected = np.array([s, t, u, v])

        # New tangent
        tang_new, pt1, _ = _ssx_tangent_4d(S1, S2, *corrected, rational=rational,
                                            direction_hint=tang_prev)
        if tang_new is None:
            rejects += 1
            if rejects >= 25:
                break
            h = max(h_floor, h * 0.5)
            continue

        if np.dot(tang_new, tang_prev) < 0:
            tang_new = -tang_new

        t3_new, speed_new = _tangent_3d(S1, corrected, tang_new, rational=rational)
        if t3_prev is not None and t3_new is not None:
            cos3 = float(np.clip(np.dot(t3_prev, t3_new), -1.0, 1.0))
            angle3 = float(np.arccos(abs(cos3)))
            # Chord-deviation rejection: sagitta ≈ chord·angle/8. If this
            # step's xyz chord deviates more than sag_tol from the curve,
            # redo it with a smaller step.
            chord = float(np.linalg.norm(pt1 - xyz_pts[-1]))
            if chord * angle3 / 8.0 > sag_tol and h > 2.0 * h_floor:
                rejects += 1
                if rejects >= 25:
                    break
                h = max(h_floor, h * 0.5)
                continue
            if _mid_chord_deviates(S1, S2, current, corrected,
                                   xyz_pts[-1], pt1, atol, sag_tol,
                                   rational) and h > 2.0 * h_floor:
                rejects += 1
                if rejects >= 25:
                    break
                h = max(h_floor, h * 0.5)
                continue
            # Adapt the xyz target by 3D curvature only.
            if angle3 > 1e-10:
                h = h * min(2.0, max(0.25, angle_threshold / angle3))
            else:
                h = h * 1.5
            h = max(atol, min(h_max, h))

        rejects = 0
        current = corrected
        tang_prev = tang_new
        if t3_new is not None:
            t3_prev = t3_new
            speed = speed_new
        stuv_pts.append(current.copy())
        xyz_pts.append(pt1.copy())

    if stats is not None:
        stats["iterations"] = int(iterations)
    return np.array(stuv_pts), np.array(xyz_pts), exit_info


# ---------------------------------------------------------------------------
# Φ-tracer crossing pairing — used only inside _deflate_tangent_cell (§8)
# ---------------------------------------------------------------------------

def _pair_crossings_for_tracing(crossings, originals=None, cell=None):
    """Pair boundary crossings for Φ tracing.

    When `originals` and `cell` are supplied, we pair design-§5 "in"
    registrations with "out" registrations in the owning cell's view. A
    through-touch crossing (in on some axes, out on others in the same cell)
    counts as an "in" once, pairable with any remaining "out".

    Falls back to stuv-distance nearest-neighbour pairing when the caller
    doesn't provide cell context — this only runs in legacy call sites.

    Returns `(pairs, unpaired)` where `pairs` is a list of `(i, j)` index
    tuples into `crossings`.
    """
    n = len(crossings)
    if n < 2:
        return [], list(range(n))

    # Registration-based pairing when we have cell context.
    if originals is not None and cell is not None:
        in_ids: list[int] = []
        out_ids: list[int] = []
        for idx, orig in enumerate(originals):
            has_in = any(r.owner is cell and r.direction == "in"
                         for r in orig.registrations)
            has_out = any(r.owner is cell and r.direction == "out"
                          for r in orig.registrations)
            if has_in:
                in_ids.append(idx)
            if has_out and not has_in:
                out_ids.append(idx)
        remaining_out = list(out_ids)
        pairs: list[tuple[int, int]] = []
        for i in in_ids:
            if not remaining_out:
                break
            # Closest remaining "out" by stuv distance — picks the
            # correct pair on the same branch when multiple Φ curves
            # cross the cell.
            j = min(
                remaining_out,
                key=lambda k: float(np.linalg.norm(crossings[i].stuv - crossings[k].stuv)),
            )
            pairs.append((i, j))
            remaining_out.remove(j)
        if not pairs and n == 2:
            # BELT-AND-BRACES — currently unexercised by any test: on the
            # geometries that motivated it the in/out registrations now
            # pair successfully (verified by instrumentation 2026-07-07);
            # kept because registration starvation on rank-deficient
            # endpoints is input-dependent, not structurally impossible.
            # Tangent-curve boundary endpoints have a rank-deficient
            # Ψ-Jacobian (2D null space), so `tangent_raw` is an arbitrary
            # null vector and §4 classification can register BOTH crossings
            # with the same direction — the in/out matching then starves
            # and the whole curve went untraced (measured: the t=0.5
            # tangent-line repro returned 0 branches). With exactly two
            # crossings the pairing is unambiguous — pair them and let the
            # Φ-march itself verify connectivity (the Ψ-validity filter in
            # _deflate_tangent_cell discards the samples if they are not).
            return [(0, 1)], []
        paired_ids = {i for p in pairs for i in p}
        unpaired = [k for k in range(n) if k not in paired_ids]
        return pairs, unpaired

    # No cell context: nothing to pair against — every crossing unpaired.
    # (The legacy stuv-nearest-neighbour fallback was removed; it returned
    # undefined names and would have crashed if this path was ever taken.)
    return [], list(range(n))


# ---------------------------------------------------------------------------
# Level 4b: Φ-tracer for C₂ tangent cells
# ---------------------------------------------------------------------------

def _normalize_t_net_numeric(T):
    """Positive per-equation scaling for numerical T-Psi consumers.

    Homogeneous rescaling of either rational surface multiplies the four
    numerator minors by different positive powers. Their zeros/signs are
    unchanged, but raw magnitudes make equation ranking and least-squares
    correction representation-dependent. Numerical Φ/Δ paths use this
    max-abs-normalized view; proof-oriented hull nets remain untouched.
    """
    arr = np.asarray(T, dtype=np.float64)
    scale = float(np.max(np.abs(arr))) if arr.size else 0.0
    return arr / scale if np.isfinite(scale) and scale > 0.0 else arr.copy()


def _choose_phi_equations(S1, S2, T_arrs, seed_stuv, rational, ranked=False):
    """Choose the best 2 Ψ equations + 1 TΨ equation for the regulated system Φ.

    Picks the combination giving the best-conditioned 3×4 Jacobian at the seed.

    Returns (psi_rows, t_index) — indices into Ψ (0-2) and TΨ (0-3).
    With `ranked=True` returns the whole candidate list ordered best-first
    (used by the Φ∩L loop seeding to retry with the runner-up equations
    when the Ψ-validity filter fragments a Φ-marched loop).
    """
    from itertools import combinations
    from mmcore.numeric.bern import bernstein_eval_nd, bernstein_partial_derivative_coeffs

    s, t, u, v = seed_stuv
    _, du1, dv1 = eval_surface_d1(S1, s, t, rational=rational)
    _, du2, dv2 = eval_surface_d1(S2, u, v, rational=rational)
    J_psi = np.column_stack([du1, dv1, -du2, -dv2])  # (3, 4)

    params = np.array(seed_stuv)
    scored: list[tuple[float, tuple[int, int], int]] = []

    for ti in range(4):
        Tv = _normalize_t_net_numeric(T_arrs[ti])
        grad = np.zeros(4)
        for axis in range(4):
            dT = bernstein_partial_derivative_coeffs(Tv, axis=axis)
            grad[axis] = bernstein_eval_nd(dT, params).item()

        if np.linalg.norm(grad) < 1e-12:
            continue  # This TΨᵢ has zero gradient at seed — skip

        for psi_rows in combinations(range(3), 2):
            J_phi = np.vstack([J_psi[list(psi_rows), :], grad.reshape(1, 4)])
            svals = np.linalg.svd(J_phi, compute_uv=False)
            score = float(svals[-1])  # smallest singular value — want it large
            scored.append((score, psi_rows, ti))

    scored.sort(key=lambda e: -e[0])
    if ranked:
        return [(psi_rows, ti) for _, psi_rows, ti in scored]
    if not scored:
        return (0, 1), 0
    return scored[0][1], scored[0][2]


def _eval_phi(S1, S2, T_arr, psi_rows, s, t, u, v, rational):
    """Evaluate the regulated system Φ at (s,t,u,v). Returns (3,) residual."""
    pt1 = eval_surface(S1, s, t, rational=rational)
    pt2 = eval_surface(S2, u, v, rational=rational)
    psi = pt1 - pt2  # (3,)

    from mmcore.numeric.bern import bernstein_eval_nd
    t_val = bernstein_eval_nd(T_arr, np.array([s, t, u, v])).item()

    return np.array([psi[psi_rows[0]], psi[psi_rows[1]], t_val])


def _jac_phi(S1, S2, T_arr, psi_rows, s, t, u, v, rational):
    """Compute the 3×4 Jacobian of Φ at (s,t,u,v)."""
    from mmcore.numeric.bern import bernstein_eval_nd, bernstein_partial_derivative_coeffs

    _, du1, dv1 = eval_surface_d1(S1, s, t, rational=rational)
    _, du2, dv2 = eval_surface_d1(S2, u, v, rational=rational)
    J_psi = np.column_stack([du1, dv1, -du2, -dv2])  # (3, 4)

    params = np.array([s, t, u, v])
    grad_t = np.zeros(4)
    for axis in range(4):
        dT = bernstein_partial_derivative_coeffs(T_arr, axis=axis)
        grad_t[axis] = bernstein_eval_nd(dT, params).item()

    J = np.vstack([J_psi[list(psi_rows), :], grad_t.reshape(1, 4)])
    return J  # (3, 4)


def _march_phi_curve(
    S1, S2, T_arr, psi_rows,
    stuv_start, stuv_end,
    *,
    atol=1e-3,
    rational=True,
    h_max=None,
    min_step=1e-6,
    max_step=0.25,
    angle_threshold=0.1,
    max_points=2000,
    stats=None,
):
    """March along the Φ-curve from stuv_start toward stuv_end.

    Same xyz-driven predictor-corrector as _march_intersection_curve but on
    the regulated system Φ = {Ψ_i, Ψ_j, TΨ_k} instead of Ψ. Deviation
    checks use the Φ corrector (the Ψ corrector is the wrong system here),
    and the reparameterization speed is the max over BOTH surface images:
    off the Ψ curve the S1 image can stall while (u,v) sweeps, and an
    S1-only speed would peg the stuv step at max_step.
    """
    stuv_pts = [stuv_start.copy()]
    xyz_start = eval_surface(S1, stuv_start[0], stuv_start[1], rational=rational)
    xyz_pts = [xyz_start]

    sag_tol = 2.0 * atol
    if h_max is None:
        h_max = max(0.05 * _local_diag(S1, rational=rational), 4.0 * atol)
    h = 0.25 * h_max
    h_floor = 1e-6 * h_max

    current = stuv_start.copy().astype(np.float64)
    target = stuv_end.copy().astype(np.float64)

    def _phi_correct(x0):
        x = np.asarray(x0, dtype=np.float64).copy()
        for _ in range(5):
            f = _eval_phi(S1, S2, T_arr, psi_rows, *x, rational=rational)
            if np.dot(f, f) < 1e-20:
                break
            Jc = _jac_phi(S1, S2, T_arr, psi_rows, *x, rational=rational)
            A = Jc.T @ Jc + 1e-12 * np.eye(4)
            try:
                delta = np.linalg.solve(A, -Jc.T @ f)
            except np.linalg.LinAlgError:
                break
            x = np.clip(x + delta, 0.0, 1.0)
        res = float(np.linalg.norm(
            _eval_phi(S1, S2, T_arr, psi_rows, *x, rational=rational)))
        return x, res

    def _phi_mid_deviates(a_stuv, b_stuv, a_xyz, b_xyz):
        xm, res = _phi_correct(0.5 * (np.asarray(a_stuv) + np.asarray(b_stuv)))
        if res > atol * 100:
            return False
        pm = eval_surface(S1, xm[0], xm[1], rational=rational)
        a3 = np.asarray(a_xyz, dtype=np.float64)
        b3 = np.asarray(b_xyz, dtype=np.float64)
        ab = b3 - a3
        den = float(np.dot(ab, ab))
        if den < 1e-30:
            return False
        tt = float(np.clip(np.dot(pm - a3, ab) / den, 0.0, 1.0))
        return float(np.linalg.norm(a3 + tt * ab - pm)) > sag_tol

    def _phi_dir_speed(x, tang):
        d1, sp1 = _tangent_3d(S1, x, tang, rational=rational)
        _, du2, dv2 = eval_surface_d1(S2, x[2], x[3], rational=rational)
        sp2 = float(np.linalg.norm(du2 * tang[2] + dv2 * tang[3]))
        return d1, max(sp1, sp2)

    def _null_tangent(J, hint):
        """Project a continuation hint into the full numerical nullspace."""
        _, svals, Vt = np.linalg.svd(J, full_matrices=True)
        scale = float(svals[0]) if len(svals) else 0.0
        tol = max(J.shape) * np.finfo(float).eps * scale
        rank = int(np.count_nonzero(svals > tol)) if scale > 0.0 else 0
        null_rows = Vt[rank:, :]
        if not len(null_rows):
            return None
        hint = np.asarray(hint, dtype=np.float64)
        tang = null_rows.T @ (null_rows @ hint)
        n = float(np.linalg.norm(tang))
        if n <= 1e-14:
            tang = null_rows[-1].copy()
            n = float(np.linalg.norm(tang))
        return tang / n if n > 0.0 else None

    # Get tangent direction from Φ Jacobian
    J = _jac_phi(S1, S2, T_arr, psi_rows, *current, rational=rational)
    tang_prev = _null_tangent(J, target - current)
    if tang_prev is None:
        if stats is not None:
            stats["iterations"] = 0
        return np.asarray(stuv_pts), np.asarray(xyz_pts)

    if np.dot(tang_prev, target - current) < 0:
        tang_prev = -tang_prev

    t3_prev, speed = _phi_dir_speed(current, tang_prev)

    rejects = 0
    iterations = 0
    for _ in range(max_points):
        iterations += 1
        step = max(min_step, min(max_step, h / max(speed, 1e-12)))

        dist_to_end = float(np.linalg.norm(current - target))
        predicted = None
        if dist_to_end < step * 2:
            final_xyz = eval_surface(S1, target[0], target[1], rational=rational)
            if (h > 2.0 * h_floor
                    and dist_to_end > 4.0 * min_step
                    and float(np.linalg.norm(final_xyz - np.asarray(xyz_pts[-1]))) > atol
                    and _phi_mid_deviates(current, target, xyz_pts[-1], final_xyz)):
                h = max(h_floor, h * 0.5)
                predicted = np.clip(current + 0.5 * (target - current), 0.0, 1.0)
                # fall through to the corrector on the halfway point
            else:
                stuv_pts.append(target.copy())
                xyz_pts.append(final_xyz)
                break

        if predicted is None:
            # Predictor
            predicted = np.clip(current + step * tang_prev, 0.0, 1.0)

        # Corrector: Newton on Φ = 0
        x, residual = _phi_correct(predicted)
        if residual > atol * 100:
            rejects += 1
            if rejects >= 25:
                break
            h = max(h_floor, h * 0.5)
            continue

        # New tangent
        J = _jac_phi(S1, S2, T_arr, psi_rows, *x, rational=rational)
        try:
            tang_new = _null_tangent(J, tang_prev)
            if tang_new is None:
                raise np.linalg.LinAlgError("empty Phi nullspace")
        except np.linalg.LinAlgError:
            rejects += 1
            if rejects >= 25:
                break
            h = max(h_floor, h * 0.5)
            continue

        if np.dot(tang_new, tang_prev) < 0:
            tang_new = -tang_new

        pt1 = eval_surface(S1, x[0], x[1], rational=rational)

        t3_new, speed_new = _phi_dir_speed(x, tang_new)
        if t3_prev is not None and t3_new is not None:
            cos3 = float(np.clip(np.dot(t3_prev, t3_new), -1.0, 1.0))
            angle3 = float(np.arccos(abs(cos3)))
            chord = float(np.linalg.norm(pt1 - xyz_pts[-1]))
            if chord * angle3 / 8.0 > sag_tol and h > 2.0 * h_floor:
                rejects += 1
                if rejects >= 25:
                    break
                h = max(h_floor, h * 0.5)
                continue
            if (h > 2.0 * h_floor
                    and _phi_mid_deviates(current, x, xyz_pts[-1], pt1)):
                rejects += 1
                if rejects >= 25:
                    break
                h = max(h_floor, h * 0.5)
                continue
            # Adapt the xyz target by 3D curvature only.
            if angle3 > 1e-10:
                h = h * min(2.0, max(0.25, angle_threshold / angle3))
            else:
                h = h * 1.5
            h = max(atol, min(h_max, h))

        rejects = 0
        current = x
        tang_prev = tang_new
        if t3_new is not None:
            t3_prev = t3_new
        speed = speed_new
        stuv_pts.append(current.copy())
        xyz_pts.append(pt1.copy())

    if stats is not None:
        stats["iterations"] = int(iterations)
    return np.array(stuv_pts), np.array(xyz_pts)


# ---------------------------------------------------------------------------
# Level 4b': Φ∩L loop seeding for crossing-less tangent cells (paper §5.3.2)
# ---------------------------------------------------------------------------

def _cell_ptol4(cell, atol):
    """Per-axis parametric tolerance of the cell's LOCAL sub-surfaces (4,)."""
    from mmcore.geom._nurbs_param_tol import bez_surface_param_tolerance
    ps, pt = bez_surface_param_tolerance(cell.g1.surface, atol, rational=True)
    pu, pv = bez_surface_param_tolerance(cell.g2.surface, atol, rational=True)
    return np.maximum(np.array([float(ps), float(pt), float(pu), float(pv)]), 1e-9)


def _march_closed_from_seed(seed, correct, tangent, midcheck, atol, h_max,
                            displace=0.02, min_step=1e-6, max_step=0.25,
                            angle_threshold=0.1, max_points=2000,
                            sag_tol=None, stats=None):
    """Closed-loop predictor-corrector engine: displace one step off `seed`
    along the curve tangent, then march AWAY until the path returns to the
    seed. System-agnostic — the Ψ and Φ marchers supply the callbacks:

      correct(x4)        -> (x4_on_curve, xyz3, ok)
      tangent(x4, prev4) -> (tang4_unit, dir3, speed)  — oriented to prev
                            when given; (None, None, 0.0) if degenerate
      midcheck(a4, b4, a3, b3) -> True if the curve deviates > sagitta
                            tolerance from the chord at the stuv midpoint

    The target-based marchers orient every step TOWARD their target, so
    seeding them with target == start walks straight back and "arrives"
    after one step. Here the arrival check is ARMED only once the path has
    escaped 3x the displacement radius, forcing the march the long way
    around; `sign(displace)` picks the branch direction at the seed (the
    Risk-2 flip: a through-the-singularity path is retried with the other
    sign by the caller).

    Returns (stuv_path, xyz_path) with path[0] == path[-1] == the corrected
    seed on genuine closure, else None (degenerate tangent, corrector
    failure streak, cell-boundary exit, arming never reached, or
    max_points without closure). Loops whose diameter is below the arming
    radius (~3*|displace| in local parameters) are NOT resolvable by this
    engine — at the size-gated terminal cells where the seeding runs, such
    loops are within tolerance of the emitted tangent point itself.
    """
    iterations = 0

    def _finish(value):
        if stats is not None:
            stats["iterations"] = int(iterations)
        return value

    seed = np.asarray(seed, dtype=np.float64)
    x0, xyz0, ok = correct(seed)
    if not ok:
        return _finish(None)
    seed = x0                                # snap the seed onto the curve
    tang, _d3, speed0 = tangent(seed, None)
    if tang is None or speed0 <= 0.0:
        return _finish(None)
    step0 = abs(float(displace))
    if displace < 0:
        tang = -tang
    x1_pred = np.clip(seed + step0 * tang, 0.0, 1.0)
    x1, xyz1, ok = correct(x1_pred)
    if not ok or float(np.linalg.norm(x1 - seed)) < 0.25 * step0:
        return _finish(None)                 # displacement collapsed back
    away = x1 - seed
    away /= float(np.linalg.norm(away))
    tang_prev, t3_prev, speed = tangent(x1, away)
    if tang_prev is None:
        return _finish(None)

    stuv_pts = [seed.copy(), x1.copy()]
    xyz_pts = [np.asarray(xyz0, dtype=np.float64),
               np.asarray(xyz1, dtype=np.float64)]
    current = x1
    if sag_tol is None:
        sag_tol = 2.0 * atol
    h = 0.25 * h_max
    h_floor = 1e-6 * h_max
    arm_radius = 3.0 * step0
    armed = False
    closed = False
    rejects = 0

    for _ in range(max_points):
        iterations += 1
        step = max(min_step, min(max_step, h / max(speed, 1e-12)))
        dist = float(np.linalg.norm(current - seed))
        if not armed and dist > arm_radius:
            armed = True
        predicted = None
        if armed and dist < max(2.0 * step, 4.0 * min_step):
            # Arrival: commit the seed unless the closing chord deviates —
            # then retarget the predictor at the halfway point (same
            # pattern as the target-based marchers).
            if (h > 2.0 * h_floor
                    and dist > 4.0 * min_step
                    and float(np.linalg.norm(np.asarray(xyz0) - np.asarray(xyz_pts[-1]))) > atol
                    and midcheck(current, seed, xyz_pts[-1], xyz0)):
                h = max(h_floor, h * 0.5)
                predicted = np.clip(current + 0.5 * (seed - current), 0.0, 1.0)
            else:
                stuv_pts.append(seed.copy())
                xyz_pts.append(np.asarray(xyz0, dtype=np.float64))
                closed = True
                break

        if predicted is None:
            predicted = np.clip(current + step * tang_prev, 0.0, 1.0)

        x, xyz, ok = correct(predicted)
        if not ok:
            rejects += 1
            if rejects >= 25:
                break
            h = max(h_floor, h * 0.5)
            continue
        # The loop leaves this cell — not an interior loop; the size-gated
        # subdivision fall-through owns boundary-crossing features.
        if np.any(x <= 1e-9) or np.any(x >= 1.0 - 1e-9):
            return _finish(None)

        tang_new, t3_new, speed_new = tangent(x, tang_prev)
        if tang_new is None:
            rejects += 1
            if rejects >= 25:
                break
            h = max(h_floor, h * 0.5)
            continue

        # No-progress guard (predictor pinned by clipping/corrector).
        if float(np.linalg.norm(x - current)) <= min_step:
            rejects += 1
            if rejects >= 25:
                break
            h = max(h_floor, h * 0.5)
            continue

        if t3_prev is not None and t3_new is not None:
            cos3 = float(np.clip(np.dot(t3_prev, t3_new), -1.0, 1.0))
            angle3 = float(np.arccos(abs(cos3)))
            chord = float(np.linalg.norm(np.asarray(xyz) - np.asarray(xyz_pts[-1])))
            if chord * angle3 / 8.0 > sag_tol and h > 2.0 * h_floor:
                rejects += 1
                if rejects >= 25:
                    break
                h = max(h_floor, h * 0.5)
                continue
            if h > 2.0 * h_floor and midcheck(current, x, xyz_pts[-1], xyz):
                rejects += 1
                if rejects >= 25:
                    break
                h = max(h_floor, h * 0.5)
                continue
            if angle3 > 1e-10:
                h = h * min(2.0, max(0.25, angle_threshold / angle3))
            else:
                h = h * 1.5
            h = max(atol, min(h_max, h))

        rejects = 0
        current = x
        tang_prev = tang_new
        if t3_new is not None:
            t3_prev = t3_new
            speed = speed_new
        stuv_pts.append(current.copy())
        xyz_pts.append(np.asarray(xyz, dtype=np.float64))

    if not closed:
        return _finish(None)
    return _finish((np.array(stuv_pts), np.array(xyz_pts)))


def _march_psi_closed(
        cell, seed_local, atol, h_max, displace=0.02, *,
        max_points=2000, stats=None):
    """Closed-loop march of the ordinary (transversal) Ψ system from a
    full-Ψ seed strictly inside the cell. Backend for Φ∩L seeds that
    refine onto a TRANSVERSAL loop point (Ψ-Jacobian rank 3): the Φ curve
    only MEETS such a loop at its TΨ_k-extremes — between them Φ leaves the
    intersection set (and, measured on the touch-plus-loop test geometry,
    rides the Ψ-valid-at-tolerance touch valley straight through the
    singularity, Risk 2) — while the Ψ marcher follows the actual loop.

    Returns a GLOBAL closed `_Fragment` (start/end = None) or None.
    """
    S1h, S2h = cell.g1.surface, cell.g2.surface

    def correct(x):
        s, t, u, v, res, sin_ang = _ssx_correct(S1h, S2h, *x, rational=True)
        ok = res <= atol * max(sin_ang, 1e-3)
        xc = np.array([s, t, u, v])
        return xc, eval_surface(S1h, s, t, rational=True), ok

    def tangent(x, prev):
        tang, _, _ = _ssx_tangent_4d(S1h, S2h, *x, rational=True,
                                     direction_hint=prev)
        if tang is None:
            return None, None, 0.0
        if prev is not None and float(np.dot(tang, prev)) < 0.0:
            tang = -tang
        d3, sp = _tangent_3d(S1h, x, tang, rational=True)
        return tang, d3, sp

    def midcheck(a4, b4, a3, b3):
        # Half the standard sagitta budget: seeded closed loops are the
        # inputs of every downstream duplicate-containment test (two seeds
        # on one loop each march a full copy), and two 2·atol-sagitta
        # samplings of a high-curvature cap sit up to ~4·atol apart —
        # outside the 2·atol containment that is supposed to collapse
        # them (measured: aspect-16 ellipse caps, κ≈160, copies 1.8·atol
        # off-curve each). At 0.5·atol the copies stay ≤ 1·atol apart.
        return _mid_chord_deviates(S1h, S2h, a4, b4, a3, b3,
                                   atol, 0.5 * atol, True)

    res = _march_closed_from_seed(seed_local, correct, tangent, midcheck,
                                  atol, h_max, displace=displace,
                                  sag_tol=0.5 * atol,
                                  max_points=max_points, stats=stats)
    if res is None:
        return None
    stuv_path, xyz_path = res
    if len(stuv_path) < 6:
        return None
    # Closure is exact by construction: _march_closed_from_seed appends the
    # corrected seed itself on closure (path[-1] == path[0] bitwise), so no
    # epsilon closure re-check is needed here.
    stuv_g = np.array([_local_to_global(x, cell.box) for x in stuv_path])
    return _Fragment(start_point=None, end_point=None,
                     stuv_path=stuv_g, xyz_path=xyz_path, tangential=False)


def _march_phi_closed(cell, seed_local, psi_rows, t_idx, atol, h_max,
                      displace=0.02, *, max_points=2000, stats=None):
    """March Φ = {Ψ_a, Ψ_b, TΨ_k} from a seed with no known endpoint until
    the path returns to its start (closed loop) or exits the cell. Keeps
    only Ψ-valid samples (|S1-S2| < atol); requires >= 6 valid samples
    forming ONE contiguous cyclic run (closure itself is exact by
    construction — the closed-loop engine ends on the seed) to emit a
    closed tangential fragment; otherwise returns None, which triggers the
    caller's ranked-equation retry. Backend for Φ∩L seeds that sit on a
    TANGENT curve (rank-deficient Ψ-Jacobian, where the Ψ marcher's
    tangent is not unique — same reason `_deflate_tangent_cell` traces Φ).

    Displaces one predictor-corrector step off the seed along the Φ tangent
    (SVD null vector of `_jac_phi`), then marches away with the closed-loop
    engine; `sign(displace)` selects the branch (Risk-2 flip).

    Returns a GLOBAL closed `_Fragment` (start/end = None) or None.
    """
    T_arrs = [_normalize_t_net_numeric(T)[..., None]
              for T in (cell.T1, cell.T2, cell.T3, cell.T4)]
    T_arr = T_arrs[t_idx]
    S1h = cell.g1.surface
    S2h = cell.g2.surface

    def correct(x0):
        x = np.asarray(x0, dtype=np.float64).copy()
        for _ in range(5):
            f = _eval_phi(S1h, S2h, T_arr, psi_rows, *x, rational=True)
            if float(np.dot(f, f)) < 1e-20:
                break
            Jc = _jac_phi(S1h, S2h, T_arr, psi_rows, *x, rational=True)
            A = Jc.T @ Jc + 1e-12 * np.eye(4)
            try:
                delta = np.linalg.solve(A, -Jc.T @ f)
            except np.linalg.LinAlgError:
                break
            x = np.clip(x + delta, 0.0, 1.0)
        res = float(np.linalg.norm(
            _eval_phi(S1h, S2h, T_arr, psi_rows, *x, rational=True)))
        # Same acceptance as _march_phi_curve's corrector (T-component units
        # are not xyz units — the 100x slack absorbs the scale mismatch).
        return x, eval_surface(S1h, x[0], x[1], rational=True), res <= atol * 100.0

    def tangent(x, prev):
        J = _jac_phi(S1h, S2h, T_arr, psi_rows, *x, rational=True)
        try:
            _, _, Vt = np.linalg.svd(J, full_matrices=True)
        except np.linalg.LinAlgError:
            return None, None, 0.0
        tang = Vt[-1]
        if prev is not None and float(np.dot(tang, prev)) < 0.0:
            tang = -tang
        d3, sp1 = _tangent_3d(S1h, x, tang, rational=True)
        _, du2, dv2 = eval_surface_d1(S2h, x[2], x[3], rational=True)
        sp2 = float(np.linalg.norm(du2 * tang[2] + dv2 * tang[3]))
        return tang, d3, max(sp1, sp2)

    def midcheck(a4, b4, a3, b3):
        xm, _, okm = correct(0.5 * (np.asarray(a4) + np.asarray(b4)))
        if not okm:
            return False
        pm = eval_surface(S1h, xm[0], xm[1], rational=True)
        a3 = np.asarray(a3, dtype=np.float64)
        b3 = np.asarray(b3, dtype=np.float64)
        ab = b3 - a3
        den = float(np.dot(ab, ab))
        if den < 1e-30:
            return False
        tt = float(np.clip(np.dot(pm - a3, ab) / den, 0.0, 1.0))
        # 0.5·atol: same tightened budget as the Ψ backend — see the
        # comment there (duplicate-loop containment headroom).
        return float(np.linalg.norm(a3 + tt * ab - pm)) > 0.5 * atol

    res = _march_closed_from_seed(seed_local, correct, tangent, midcheck,
                                  atol, h_max, displace=displace,
                                  sag_tol=0.5 * atol,
                                  max_points=max_points, stats=stats)
    if res is None:
        return None
    stuv_path, xyz_path = res
    # Closure is exact by construction: _march_closed_from_seed appends the
    # corrected seed itself on closure (path[-1] == path[0] bitwise), so no
    # epsilon closure re-check is needed here.
    # Ψ-validity filter (same as _deflate_tangent_cell): keep samples on
    # the actual intersection.
    keep = []
    for k in range(len(stuv_path)):
        p1 = eval_surface(S1h, stuv_path[k, 0], stuv_path[k, 1], rational=True)
        p2 = eval_surface(S2h, stuv_path[k, 2], stuv_path[k, 3], rational=True)
        if float(np.linalg.norm(p1 - p2)) < atol:
            keep.append(k)
    if len(keep) < 6:
        return None
    # A CLOSED output must retain the whole closed march.  The older
    # "one contiguous cyclic run" test was unsound because ``stuv_path``
    # stores the seed twice (indices 0 and -1): removing one connected arc
    # could still look like one run under ``np.roll`` and then ship an OPEN
    # half-loop with ``start_point=end_point=None``.  Besides corrupting the
    # geometry, that open fragment failed to subsume the Delta samples and
    # revived the tangent-point flood.  Any Psi-invalid sample invalidates
    # the closure claim; let the ranked-equation/sign retry find a fully
    # valid regulated curve instead.
    if len(keep) != len(stuv_path):
        return None
    stuv_g = np.array([_local_to_global(stuv_path[k], cell.box) for k in keep])
    xyz_g = xyz_path[np.asarray(keep)]
    return _Fragment(start_point=None, end_point=None,
                     stuv_path=stuv_g, xyz_path=xyz_g, tangential=True)


def _fragment_normals_aligned(
        cell, frag, bar=1e-3, *, stuv_local=False):
    """Tangency-by-MEASUREMENT test on a fragment's kept samples: the two
    surface normals must stay aligned, sin(angle) <= the pipeline's own
    1e-3 transversality bar, at EVERY sample. Two consumers:

    - Ledger L2: validate a Φ-marched CLOSED fragment as a genuine TANGENT
      feature. The Φ corrector solves only {Ψ_a, Ψ_b, TΨ_k} and accepts at
      atol*100, and the Ψ-validity keep-filter passes anything within atol
      of the intersection — a SUB-TOLERANCE valley (touch-plus-loop at
      eps=1e-3: valley floor |Ψ_z| = eps²/4 = 2.5e-7) satisfies both, so
      the marcher ships a closed 'tangential' phantom that is
      transversal-normal along most of its arc (measured: sin_ang up to
      2.56e-2 on the phantom vs 3.96e-7 max on the genuine tangent circle
      of z=(q-1/4)² — five decades of separation around the 1e-3 bar). A
      genuine tangent feature has TΨ = 0, i.e. parallel normals, along the
      whole path.

    NOT usable for ledger L5's tagging of Ψ-traced loop-free fragments —
    measured both failure directions there: Ψ-marched samples of a GENUINE
    tangent curve wander in the sub-tolerance valley (cylinder-on-plane
    line: ~1.4e-3 off t=0.5, sin_ang up to 1.08e-2 — an order ABOVE this
    bar), while a genuine TRANSVERSAL ring deep in a valley sits BELOW it
    (touch-plus-loop at eps=1e-3: ring |∇z| = 2·eps^1.5 ≈ 6e-5, all
    samples "aligned"). L5 uses the Δ-snap (`_fragment_on_tangent_locus`)
    instead. Φ-marched fragments (the L2 consumer here) are corrected onto
    Δ itself and sit at sin_ang ~4e-7, so for them this bar separates
    cleanly (phantom max 2.56e-2 — five decades of margin).

    Degenerate-normal samples (‖N‖ < 1e-30) are skipped — cannot decide
    there, same convention as the transversality pre-check on crossings.
    Normals are evaluated on the cell's LOCAL nets: reparameterization
    scales derivatives by positive factors, so sin(angle) is invariant.
    """
    S1h, S2h = cell.g1.surface, cell.g2.surface
    for x_g in np.asarray(frag.stuv_path):
        x = (np.asarray(x_g, dtype=np.float64) if stuv_local
             else _global_to_local(x_g, cell.box))
        _, du1, dv1 = eval_surface_d1(S1h, x[0], x[1], rational=True)
        _, du2, dv2 = eval_surface_d1(S2h, x[2], x[3], rational=True)
        N1 = np.cross(du1, dv1)
        N2 = np.cross(du2, dv2)
        n1m = float(np.linalg.norm(N1))
        n2m = float(np.linalg.norm(N2))
        if n1m < 1e-30 or n2m < 1e-30:
            continue
        if float(np.linalg.norm(np.cross(N1, N2))) > bar * n1m * n2m:
            return False
    return True


def _fragment_on_tangent_locus(cell, frag, atol):
    """Ledger L5: decide by MEASUREMENT whether a Ψ-traced fragment from a
    loop-free cell (tangency hull gate fired) is a tangent CURVE. A tangent
    curve traces fine through the loop-free path (non-strict monotone
    T-hulls certify loop-free) but shipped kind='transversal', breaking the
    output contract AND blinding the kind-keyed subsumption filter (stray
    on-curve tangent_points survived). Tangentiality is decided here by
    measurement, never by provenance — the caller only measures fragments
    from cells whose hull gate fired, and a failing fragment keeps
    kind='transversal' (the zero-risk direction for coverage geometry).

    The test: EVERY sample must Δ-SNAP — Gauss-Newton onto the deflated
    set Δ = Ψ ∩ TΨ (plain-float `_delta_float_gn`, ~0.1 ms/start) moving
    at most 2·atol in xyz (the matching ladder). Normal alignment CANNOT
    decide this either way (measured, both directions):

    - the genuine cylinder-on-plane tangent line's Ψ-marched samples
      WANDER in the sub-tolerance valley (the Ψ corrector is rank-
      deficient transversally and stalls at |Ψ_z| ~ 7e-6 ≈ 1.4e-3 param
      off the line) → sin_ang up to 1.08e-2, an order ABOVE the pipeline's
      1e-3 transversality bar — an "any misaligned ⇒ transversal" rule
      loses the tag;
    - the touch-plus-loop eps=1e-3 ring is genuinely TRANSVERSAL (sign
      change) yet its normals align to |∇z| = 2·eps^1.5 ≈ 6e-5, BELOW the
      bar at every sample — an "all aligned ⇒ tangential" rule flips the
      committed L2 ground truth (tangential == []).

    The Δ-snap separates both: wandered tangent-line samples snap back
    onto the locus (~1·atol motion, on-curve Δ-roots everywhere); the
    eps=1e-3 ring's samples are 15.8·atol from the only Δ-root (the
    touch) → reject; near-tangent TOUCH-FREE cells (case 5: 20 gate-fired
    cells, case 11: 10) have no Δ-root at all → the first sample rejects
    (the GN also rejects the TΨ=0,Ψ≠0 trap sheet down to ‖Ψ‖ ≈ 1e-8, see
    `_delta_float_gn`). A transversal fragment passing near an isolated
    touch cannot pass either: only its sub-2·atol-to-the-touch samples
    snap, the rest move too far. (A micro-fragment entirely within 2·atol
    of a touch can pass, but micro tangential polylines are below the
    subsumption filter's 16·atol arc floor and are owned by the
    micro-branch filter.)

    Numerical failure of the factory/GN (LinAlgError, FloatingPointError)
    keeps the fragment transversal — fail-safe in the coverage-kind
    direction.
    """
    S1h = cell.g1.surface
    xyz_path = np.asarray(frag.xyz_path)
    try:
        gn, _Tstack = _delta_float_gn(cell.T1, cell.T2, cell.T3, cell.T4,
                                      S1h, cell.g2.surface, rational=True,
                                      atol=atol)
        for k, x_g in enumerate(np.asarray(frag.stuv_path)):
            x = _global_to_local(x_g, cell.box)
            xw = gn(x)
            if xw is None:
                return False
            xyz_w = eval_surface(S1h, xw[0], xw[1], rational=True)
            if float(np.linalg.norm(xyz_w - xyz_path[k])) > 2.0 * atol:
                return False
    except (np.linalg.LinAlgError, FloatingPointError):
        return False
    return True


def _phi_slice_loop_fragments(cell, roots, atol, h_max, all_singularities):
    """Paper §5.3.2: tiny Ψ-loops around a tangency can have no boundary
    crossings; the regulated Φ curve meets every such loop >= 2x (Lemma 2:
    along a closed loop the 4D tangent is ∝ (T¹,−T²,T³,−T⁴), and the loop's
    k-th coordinate has >= 2 extremes — TΨ_k = 0 there, i.e. ON Φ). Slice Φ
    with the four deterministic axis mid-planes, refine each Φ∩L seed onto
    the FULL intersection, and march closed loops from the survivors.

    The full-Ψ refinement (Gauss-Newton on all three Ψ components) is the
    SECOND line of defense against phantom geometry — the first is
    upstream: `phi_loop_seeds` carries ALL THREE Ψ components as
    hull-exclusion nets, which already prunes the sub-tolerance-valley
    boxes (e.g. the touch-plus-loop valley-floor ring at |Ψ| = eps^2/4
    < atol, whose third component is bounded away from 0 — review of
    2d030bb verified the valley never seeds). What the refinement still
    owns: Φ∩L Newton solves only TWO Ψ components plus {T_k, L}, so its
    solutions can sit off the true intersection in the third component —
    on symmetric geometries the damped Newton returns ptol-ladder samples
    of a degenerate solution LINE, and any near-loop seed carries O(step)
    error. Refinement converges genuine near-loop seeds onto the loop
    (residual ~1e-12), stalls on critical manifolds of ‖Ψ‖² (the residual
    is orthogonal to the Jacobian's column space), and snaps line samples
    onto the actual intersection set; the 0.01·atol acceptance separates
    the outcomes cleanly.

    Backend choice per refined seed (design latitude, measured): normals
    angle sin_ang > 1e-3 (the pipeline's own transversality bar) => the
    loop is transversal, march Ψ (`_march_psi_closed`); else the seed lies
    on a tangent curve, march Φ (`_march_phi_closed`, retrying with the
    runner-up (psi_rows, t_idx) if Ψ-validity fragments the loop). A
    Ψ-marched path that passes within 2·atol of an emitted tangent point
    while claiming closure is a through-the-singularity artifact (Risk 2) —
    retried with the flipped displacement, then discarded. The Φ branch
    skips that rejection: its seeds are ON the Δ set, whose closed
    components are legitimately covered with emitted Δ-witness samples.
    Every Φ-marched closed fragment must additionally pass the L2 tangency
    validation (`_fragment_normals_aligned`: sin_ang <= 1e-3 at
    every sample) before it is emitted as `tangential` — sub-tolerance
    valleys (floor |Ψ| < atol) are Ψ-valid and Φ-marchable yet
    transversal-normal, and such a phantom both ships wrong geometry AND
    lets the post-assembly subsumption filter delete the genuine touch it
    passes through. A validation failure stops the retry ladder for that
    seed (the phantom is geometry, not an equation-choice artifact).

    Returns a list of GLOBAL closed fragments; duplicates of subdivision-
    traced geometry are absorbed downstream by `_drop_duplicate_fragments`
    containment.
    """
    from mmcore.numeric.intersection.ssx._ssx5_singular import phi_loop_seeds

    S1h = cell.g1.surface
    S2h = cell.g2.surface
    T_numeric = tuple(_normalize_t_net_numeric(T)
                      for T in (cell.T1, cell.T2, cell.T3, cell.T4))
    T_arrs = [T[..., None] for T in T_numeric]
    seed_pt = np.asarray(roots[0]) if roots else np.full(4, 0.5)
    ranked = _choose_phi_equations(S1h, S2h, T_arrs, seed_pt,
                                   rational=True, ranked=True)
    if not ranked:
        return []
    psi_rows, t_idx = ranked[0]
    ptol4 = _cell_ptol4(cell, atol)
    _phi_stats = {}
    try:
        seeds = phi_loop_seeds(
            cell.g1.surface, cell.g2.surface,
            T_numeric,
            psi_rows, t_idx, atol, ptol=ptol4,
            charge_box=((lambda n: cell.work_budget.charge_cells(
                n, "phi")) if cell.work_budget is not None else None),
            stats=_phi_stats)
    except (np.linalg.LinAlgError, FloatingPointError):
        return []
    if (cell.work_budget is not None
            and (_phi_stats.get("budget_exhausted", False)
                 or _phi_stats.get("external_budget_exhausted", False))):
        # Mid-plane seeds are partial evidence when any slice enumeration
        # truncates: a missed seed can own an otherwise boundary-free loop.
        # Keep already certified seeds/fragments, but never call the result
        # globally complete.
        cell.work_budget.mark_incomplete()
        if cell.work_budget.exhausted:
            # A denied shared charge cannot recover during this synchronous
            # call.  Do not spend uncharged corrector/dedup work on partial
            # seeds that cannot be marched under the exhausted budget.
            return []
    if not seeds:
        return []

    tangent_xyz = [np.asarray(g.xyz, dtype=np.float64)
                   for g in all_singularities if g.kind == "tangent_point"]

    def _near_tangent_point(path_xyz):
        return any(float(np.linalg.norm(np.asarray(p) - txyz)) <= 2.0 * atol
                   for p in np.asarray(path_xyz) for txyz in tangent_xyz)

    def _bounded_closed_march(marcher, *args, **kwargs):
        work_budget = cell.work_budget
        if work_budget is None:
            return marcher(*args, **kwargs)
        limit = min(2000, work_budget.remaining_cells)
        if limit <= 0:
            work_budget.mark_exhausted()
            return None
        march_stats = {}
        frag = marcher(
            *args, **kwargs, max_points=limit, stats=march_stats)
        if not work_budget.charge_cells(
                int(march_stats.get("iterations", 0)),
                "singular_trace"):
            return None
        return frag

    refined: list[tuple] = []
    for seed in seeds:
        s, t, u, v, res, sin_ang = _ssx_correct(
            cell.g1.surface, cell.g2.surface, *seed,
            rational=True, max_iter=15, tol=1e-24)
        if res > 0.01 * atol:
            continue                    # not on the intersection set
        x = np.array([s, t, u, v])
        xyz = eval_surface(cell.g1.surface, s, t, rational=True)
        if any(float(np.linalg.norm(xyz - txyz)) <= 2.0 * atol
               for txyz in tangent_xyz):
            continue                    # the tangency itself — nothing to march
        if any(np.all(np.abs(x - rx) <= ptol4)
               and float(np.linalg.norm(xyz - rxyz)) <= atol
               for rx, rxyz, _, _ in refined):
            continue                    # destructive dedup: 1·ptol AND atol xyz
        # Equation conditioning is local along a singular curve.  Reusing
        # the ranking at ``roots[0]`` for every loop seed selected a minor
        # whose regulating gradient collapsed at an off-lattice ring's
        # cardinal point; its null tangent pointed radially inward and made
        # a short, Psi-invalid pseudo-loop.  Rank again at the corrected seed
        # and keep the existing two-candidate retry and full-Psi gates.
        seed_ranked = _choose_phi_equations(
            S1h, S2h, T_arrs, x, rational=True, ranked=True)
        if not seed_ranked:
            continue
        # Backend selection starts from topological rank.  A perfectly regular
        # transversal curve may meet at a very small angle (the eps=1e-3
        # touch-plus-loop ring has sin(angle)=1.26e-4); the old 1e-3 angle
        # heuristic routed it through Phi and lost the real loop.  Conversely
        # an exact tangent curve has rank(Psi') <= 2, but a seed a few ulps off
        # that curve can look numerically rank-3 (off-lattice tangent ring:
        # sin=8.1e-6).  Require both numerical rank 3 and a much narrower
        # 1e-5 conditioning guard.  More ill-conditioned genuine transversal
        # cases conservatively take the regulated/partial path.
        _p1, _ds1, _dt1 = eval_surface_d1(
            S1h, x[0], x[1], rational=True)
        _p2, _du2, _dv2 = eval_surface_d1(
            S2h, x[2], x[3], rational=True)
        _Jpsi = np.column_stack([_ds1, _dt1, -_du2, -_dv2])
        try:
            _svals = np.linalg.svd(_Jpsi, compute_uv=False)
        except np.linalg.LinAlgError:
            _svals = np.empty(0)
        _psi_rank3 = bool(
            len(_svals) == 3 and _svals[0] > 0.0
            and _svals[-1] > max(_Jpsi.shape) * _svals[0] * 1e-10
            and sin_ang > 1e-5)
        refined.append((x, xyz, _psi_rank3, seed_ranked))

    fragments: list[_Fragment] = []
    for refined_idx, (x, xyz, psi_rank3, seed_ranked) in enumerate(refined):
        if any(len(fr.xyz_path) >= 2
               and _dist_point_polyline(xyz, np.asarray(fr.xyz_path)) <= 2.0 * atol
               for fr in fragments):
            continue                    # this loop is already marched
        # The closed-loop engine arms only after travelling 3*displace.
        # A fixed 0.02 local displacement cannot arm on a smaller genuine
        # loop (the eps=1e-3 transversal ring has 0.0316 parameter diameter),
        # even though deterministic Phi slices provide several distinct
        # seeds around it.  Use their nearest-neighbour spacing to size a
        # bounded local displacement, floored at four resolution cells.
        _other_seed_dist = [
            float(np.linalg.norm(x - other[0]))
            for j, other in enumerate(refined) if j != refined_idx
        ]
        if _other_seed_dist:
            _seed_displace = min(
                0.02,
                max(4.0 * float(np.max(ptol4)),
                    0.2 * min(_other_seed_dist)))
        else:
            _seed_displace = 0.02
        frag = None
        if psi_rank3:
            for disp in (_seed_displace, -_seed_displace):
                if (cell.work_budget is not None
                        and cell.work_budget.exhausted):
                    break
                frag = _bounded_closed_march(
                    _march_psi_closed, cell, x, atol, h_max,
                    displace=disp)
                if frag is not None and not _near_tangent_point(frag.xyz_path):
                    break
                frag = None
        else:
            # No tangent-point-proximity rejection here: a Φ-backend seed is
            # by construction ON the Δ set (rank-deficient), so its closed
            # component (a tangent LOOP) is itself peppered with emitted
            # Δ-witness samples — proximity along the path is the expected
            # case, not a through-artifact. (The concrete Risk-2 through-
            # march starts from a TRANSVERSAL seed, which the Ψ branch
            # above owns.)
            phantom = False
            for pr, ti in seed_ranked[:2]:
                for disp in (_seed_displace, -_seed_displace):
                    if (cell.work_budget is not None
                            and cell.work_budget.exhausted):
                        break
                    frag = _bounded_closed_march(
                        _march_phi_closed, cell, x, pr, ti, atol, h_max,
                        displace=disp)
                    if (frag is not None
                            and not _fragment_normals_aligned(cell, frag)):
                        # Ledger L2: transversal-normal somewhere along the
                        # closed path => a sub-tolerance-valley phantom, not
                        # a tangent feature. Normal alignment is a property
                        # of the marched GEOMETRY, not of the (psi_rows,
                        # t_idx) choice — the ranked-equation retry exists
                        # for Ψ-validity FRAGMENTATION and would only
                        # re-march the same phantom here, so stop retrying
                        # this seed outright.
                        frag = None
                        phantom = True
                        break
                    if frag is not None:
                        break
                if (cell.work_budget is not None
                        and cell.work_budget.exhausted):
                    break
                if frag is not None or phantom:
                    break
        if frag is not None:
            fragments.append(frag)
    return fragments


def _deflate_tangent_cell(S1, S2, T1, T2, T3, T4, box, crossings, atol,
                          *, rational=True, originals=None, cell=None,
                          h_max=None):
    """Handle a confirmed-tangent cell by tracing the regulated Φ curve.

    1. Choose the best Φ = {Ψ_i, Ψ_j, TΨ_k} equations
    2. March Φ between boundary crossing pairs selected via in/out
       registrations (design §4/§8) when `originals` and `cell` are supplied.
    3. Filter points that are also on the full intersection (Ψ=0)

    Returns `(fragments, points)`. Fragments carry `start_point` / `end_point`
    references — to `originals[i]` / `originals[j]` when originals are
    provided, otherwise None — so the §9 assembly can chain Φ-fragments
    alongside Ψ-fragments (design §8).
    """
    T_arrs = [_normalize_t_net_numeric(T)[..., np.newaxis]
              for T in (T1, T2, T3, T4)]

    fragments: list[_Fragment] = []
    points: list[SSXPoint] = []

    if len(crossings) < 2:
        for c in crossings:
            points.append(SSXPoint(stuv=c.stuv, xyz=c.xyz))
        return fragments, points

    pairs, unpaired = _pair_crossings_for_tracing(crossings, originals=originals, cell=cell)

    if cell is not None:
        end_tol = 4.0 * _cell_ptol4(cell, atol)
    else:
        from mmcore.geom._nurbs_param_tol import bez_surface_param_tolerance
        ps, pt = bez_surface_param_tolerance(S1, atol, rational=rational)
        pu, pv = bez_surface_param_tolerance(S2, atol, rational=rational)
        end_tol = 4.0 * np.maximum(
            np.array([ps, pt, pu, pv], dtype=np.float64), 1e-9)

    def _pair_alignment(candidate, start, target):
        psi_rows, ti = candidate
        J = _jac_phi(S1, S2, T_arrs[ti], psi_rows,
                     *start, rational=rational)
        try:
            _, svals, Vt = np.linalg.svd(J, full_matrices=True)
        except np.linalg.LinAlgError:
            return -np.inf
        scale = float(svals[0]) if len(svals) else 0.0
        tol = max(J.shape) * np.finfo(float).eps * scale
        rank = int(np.count_nonzero(svals > tol)) if scale > 0.0 else 0
        null_rows = Vt[rank:, :]
        hint = np.asarray(target) - np.asarray(start)
        hnorm = float(np.linalg.norm(hint))
        if not len(null_rows) or hnorm <= 1e-30:
            return -np.inf
        projected = null_rows.T @ (null_rows @ hint)
        return float(np.linalg.norm(projected) / hnorm)

    for i, j in pairs:
        accepted = None
        accepted_indices = None
        forward_start = np.asarray(crossings[i].stuv, dtype=np.float64)
        forward_target = np.asarray(crossings[j].stuv, dtype=np.float64)

        def _valid_endpoint_path(stuv_path, xyz_path, target):
            if len(stuv_path) < 2:
                return False
            if any(float(np.linalg.norm(
                    eval_surface(S1, x[0], x[1], rational=rational)
                    - eval_surface(
                        S2, x[2], x[3], rational=rational))) >= atol
                   for x in stuv_path):
                return False
            target_xyz = eval_surface(
                S1, target[0], target[1], rational=rational)
            return bool(
                np.all(np.abs(
                    np.asarray(stuv_path[-1]) - target) <= end_tol)
                and float(np.linalg.norm(
                    np.asarray(xyz_path[-1]) - target_xyz))
                <= 2.0 * atol)

        # Prefer the full physical Psi system.  With a continuation hint,
        # its full-nullspace projection can trace many tangent curves even
        # though rank(Psi') < 3; every vertex is then constrained by all
        # three surface-residual equations.  The regulated Phi fallback is
        # still required for harder singularities (notably case 14), where
        # the rank-deficient Psi continuation cannot choose the branch.
        for start_idx, target_idx in ((i, j), (j, i)):
            start = np.asarray(
                crossings[start_idx].stuv, dtype=np.float64)
            target = np.asarray(
                crossings[target_idx].stuv, dtype=np.float64)
            trace_limit = 512
            if cell is not None and cell.work_budget is not None:
                trace_limit = min(
                    trace_limit, cell.work_budget.remaining_cells)
                if trace_limit <= 0:
                    cell.work_budget.mark_exhausted()
                    break
            trace_stats = {}
            stuv_path, xyz_path = _march_intersection_curve(
                S1, S2, start, target,
                atol=atol, rational=rational, h_max=h_max,
                max_points=trace_limit, stats=trace_stats,
            )
            if cell is not None and cell.work_budget is not None:
                if not cell.work_budget.charge_cells(
                        int(trace_stats.get("iterations", 0)),
                        "singular_trace"):
                    break
            if _valid_endpoint_path(stuv_path, xyz_path, target):
                accepted = (stuv_path, xyz_path)
                accepted_indices = (start_idx, target_idx)
                break

        # Continuation of a rank-deficient Phi curve can be numerically
        # directional even though the geometric endpoint pair is not.  On a
        # closed tangent ring split into two cells, one half reached its
        # target only when marched out->in; forcing the registration's
        # in->out orientation dropped that half, shipped an open semicircle,
        # and left dozens of Delta samples unsubsumed.  Try the reverse
        # orientation immediately after each ranked equation.  Every attempt
        # still has to pass the full-Psi and endpoint-reach checks below, so
        # this adds no optimistic connectivity assumption.
        if (accepted is None
                and not (cell is not None
                         and cell.work_budget is not None
                         and cell.work_budget.exhausted)):
            candidates = _choose_phi_equations(
                S1, S2, T_arrs, forward_start,
                rational=rational, ranked=True)
            candidates = sorted(
                enumerate(candidates),
                key=lambda entry: (
                    -_pair_alignment(
                        entry[1], forward_start, forward_target),
                    entry[0]),
            )
            for _, (psi_rows, t_idx) in candidates:
                if (cell is not None and cell.work_budget is not None
                        and cell.work_budget.exhausted):
                    break
                for start_idx, target_idx in ((i, j), (j, i)):
                    trace_limit = 512
                    if cell is not None and cell.work_budget is not None:
                        trace_limit = min(
                            trace_limit,
                            cell.work_budget.remaining_cells)
                        if trace_limit <= 0:
                            cell.work_budget.mark_exhausted()
                            break
                    start = np.asarray(
                        crossings[start_idx].stuv, dtype=np.float64)
                    target = np.asarray(
                        crossings[target_idx].stuv, dtype=np.float64)
                    trace_stats = {}
                    stuv_path, xyz_path = _march_phi_curve(
                        S1, S2, T_arrs[t_idx], psi_rows, start, target,
                        atol=atol, rational=rational, h_max=h_max,
                        max_points=trace_limit, stats=trace_stats,
                    )
                    if cell is not None and cell.work_budget is not None:
                        if not cell.work_budget.charge_cells(
                                int(trace_stats.get("iterations", 0)),
                                "singular_trace"):
                            break
                    if _valid_endpoint_path(
                            stuv_path, xyz_path, target):
                        accepted = (stuv_path, xyz_path)
                        accepted_indices = (start_idx, target_idx)
                        break
                if (cell is not None and cell.work_budget is not None
                        and cell.work_budget.exhausted):
                    break
                if accepted is not None:
                    break
        if accepted is None:
            if cell is not None and cell.work_budget is not None:
                cell.work_budget.mark_incomplete()
            continue
        stuv_path, xyz_path = accepted
        start_idx, target_idx = accepted_indices
        start_pt = originals[start_idx] if originals is not None else None
        end_pt = originals[target_idx] if originals is not None else None
        fragments.append(_Fragment(
            start_point=start_pt, end_point=end_pt,
            stuv_path=stuv_path,
            xyz_path=xyz_path,
            tangential=True,
        ))

    for k in unpaired:
        src = originals[k] if originals is not None else crossings[k]
        points.append(SSXPoint(stuv=src.stuv, xyz=src.xyz))

    return fragments, points


# ---------------------------------------------------------------------------
# Newton SSX solver
# ---------------------------------------------------------------------------

def newton_ssx(
    S1, S2,
    s0: float, t0: float, u0: float, v0: float,
    *,
    rational: bool = True,
    tol: float = 1e-12,
    max_iter: int = 30,
    lm_damp: float = 1e-12,
):
    """Pseudoinverse Newton for SSX: S1(s,t) - S2(u,v) = 0.

    This is 3 equations in 4 unknowns (underdetermined). The step is
    computed via the pseudoinverse of the 3×4 Jacobian, which projects
    the correction onto the null space of J — effectively moving along
    the intersection curve while reducing the residual.

    Parameters
    ----------
    S1, S2 : ndarray
        Bezier surface control nets (homogeneous if rational).
    s0, t0, u0, v0 : float
        Initial parameter guess.
    rational : bool
        Whether control nets are homogeneous.

    Returns
    -------
    converged : bool
    stuv : ndarray (4,)
        Best parameters found.
    residual : ndarray (3,)
        Final residual vector S1 - S2.
    """
    s, t, u, v = float(s0), float(t0), float(u0), float(v0)

    def _clamp(x):
        return max(0.0, min(1.0, x))

    for _ in range(max_iter):
        pt1, du1, dv1 = eval_surface_d1(S1, s, t, rational=rational)
        pt2, du2, dv2 = eval_surface_d1(S2, u, v, rational=rational)
        G = pt1 - pt2
        g2 = float(np.dot(G, G))
        if g2 < tol * tol:
            return True, np.array([s, t, u, v]), G

        # Jacobian: J = [dS1/ds, dS1/dt, -dS2/du, -dS2/dv]  (3×4)
        J = np.column_stack([du1, dv1, -du2, -dv2])

        # Damped pseudoinverse step: delta = -J^+ @ G
        # Using (J^T J + λI)^-1 J^T for numerical stability
        JT = J.T
        A = JT @ J + lm_damp * np.eye(4)
        b = -JT @ G
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break

        # Line search
        step = 1.0
        accepted = False
        for _ in range(8):
            sn = _clamp(s + step * delta[0])
            tn = _clamp(t + step * delta[1])
            un = _clamp(u + step * delta[2])
            vn = _clamp(v + step * delta[3])
            Gn = (eval_surface(S1, sn, tn, rational=rational)
                  - eval_surface(S2, un, vn, rational=rational))
            if float(np.dot(Gn, Gn)) <= g2:
                s, t, u, v = sn, tn, un, vn
                accepted = True
                break
            step *= 0.5

        if not accepted:
            break

    G = eval_surface(S1, s, t, rational=rational) - eval_surface(S2, u, v, rational=rational)
    converged = float(np.linalg.norm(G)) < tol ** 0.5  # looser convergence check
    return converged, np.array([s, t, u, v]), G


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _overlaps_to_branches(boundary_overlaps, S1, atol, rational):
    """Convert BoundaryOverlap objects to SSXBranch with overlap=True.

    Filters zero-length overlaps and deduplicates overlaps that
    represent the same 3D geometry (same start/end points in space).
    """
    branches = []
    for ovl in boundary_overlaps:
        xyz_start = eval_surface(S1, ovl.stuv_start[0], ovl.stuv_start[1], rational=rational)
        xyz_end = eval_surface(S1, ovl.stuv_end[0], ovl.stuv_end[1], rational=rational)

        # Skip zero-length overlaps
        if np.linalg.norm(xyz_start - xyz_end) < atol:
            continue

        # Check if we already have an overlap with the same 3D endpoints
        is_dup = False
        for b in branches:
            _, existing_xyz = b.curve
            same = (np.linalg.norm(xyz_start - existing_xyz[0]) < atol and
                    np.linalg.norm(xyz_end - existing_xyz[-1]) < atol)
            rev = (np.linalg.norm(xyz_start - existing_xyz[-1]) < atol and
                   np.linalg.norm(xyz_end - existing_xyz[0]) < atol)
            if same or rev:
                is_dup = True
                break
        if is_dup:
            continue

        stuv_path = np.stack([ovl.stuv_start, ovl.stuv_end], axis=0)
        xyz_path = np.stack([xyz_start, xyz_end], axis=0)
        branches.append(SSXBranch(curve=(stuv_path, xyz_path), overlap=True, kind="overlap"))

    return branches


# ---------------------------------------------------------------------------
# Loop-absence check: TΨᵢ monotonicity OR Gauss map separability
# ---------------------------------------------------------------------------

def _check_loop_free(g1, g2, T1=None, T2=None, T3=None, T4=None):
    """Check if the intersection is provably loop-free.

    Two independent checks — either suffices:
    1. TΨᵢ monotonicity: any TΨᵢ non-negative or non-positive
    2. Gauss map separability: normal cones separated into opposite hemispheres

    Returns True if loop-free.
    """
    if T1 is not None:
        is_mono, _ = _check_monotonicity(T1, T2, T3, T4)
        if is_mono:
            return True

    try:
        p1, p2 = separate_gauss_maps(g1.map_dirs(), g2.map_dirs())
        if p1 is not None and p2 is not None:
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _local_to_global(stuv_local, box):
    """Convert local [0,1]⁴ stuv to global coordinates using the cell's box."""
    g = np.empty(4, dtype=np.float64)
    for i in range(4):
        lo, hi = box[i]
        g[i] = lo + stuv_local[i] * (hi - lo)
    return g


def _global_to_local(stuv_global, box):
    """Convert global stuv to local [0,1]⁴ coordinates for a cell."""
    loc = np.empty(4, dtype=np.float64)
    for i in range(4):
        lo, hi = box[i]
        span = hi - lo
        if span > 1e-15:
            loc[i] = (stuv_global[i] - lo) / span
        else:
            loc[i] = 0.5
    return np.clip(loc, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Registration-based tracing (design §7)
# ---------------------------------------------------------------------------

def _find_exit_registration(cell, stuv_end, tol_param=1e-4):
    """Design §7 Invariant D: locate the unique unconsumed "out" registration
    owned by `cell` that matches the marcher's stopping point.

    The marcher is guaranteed to stop on the cell's boundary (it clamps to
    `[0,1]⁴` in local coords); therefore `stuv_end` must have at least one
    on-boundary axis for this cell. We walk every matching partition on
    every on-boundary axis and return the first unconsumed out-registration
    whose `param` matches `stuv_end[free_axis]` within `tol_param`.
    """
    best: Optional[IsolineRegistration] = None
    best_residual = float("inf")
    for i in range(4):
        local = _on_axis_local(stuv_end[i], cell.box[i][0], cell.box[i][1])
        if local is None:
            continue
        target_value = cell.box[i][local]
        for p in cell.partitions:
            if p.axis != i or abs(p.value - target_value) > 1e-8:
                continue
            target_param = float(stuv_end[p.free_axis])
            for reg in p.registrations:
                if reg.consumed or reg.owner is not cell or reg.direction != "out":
                    continue
                r = abs(reg.param - target_param)
                if r < tol_param and r < best_residual:
                    best = reg
                    best_residual = r
    return best


@dataclass
class _Fragment:
    """A partial branch produced by one cell's tracer.

    `start_point` / `end_point` refer to the actual `BoundaryPoint` objects
    (shared across adjacent cells via shared `PartitionCurve`s when they sit
    on an internal partition), so §9 assembly chains fragments by object
    identity — no xyz proximity involved.
    """
    start_point: Optional[BoundaryPoint]
    end_point: Optional[BoundaryPoint]
    stuv_path: NDArray[np.float64]
    xyz_path: NDArray[np.float64]
    tangential: bool = False


def _consume_cell_directions(point: BoundaryPoint, cell, direction: str) -> None:
    """Mark all `point`'s registrations with the given direction in this cell
    as consumed. A single march at a multi-axis corner represents the whole
    curve's passage through the corner; all its same-direction registrations
    in this cell describe that one passage and must all be consumed together.
    """
    for r in point.registrations:
        if r.owner is cell and r.direction == direction:
            r.consumed = True


def _cell_has_unused_direction(point: BoundaryPoint, cell, direction: str) -> bool:
    return any(
        r.owner is cell and r.direction == direction and not r.consumed
        for r in point.registrations
    )


def _trace_cell_by_registrations(cell, atol, h_max=None):
    """Trace all branch segments inside a certified cell.

    For each boundary crossing, march in one direction. If the march
    immediately exits (corner touch), try the opposite direction.

    Endpoint policy ("trust the marcher's stopping point"): a march that
    reaches the cell boundary ends at a Newton-verified intersection point
    on a face. If an unconsumed crossing matches it within the parametric
    tolerance, the fragment ends at that crossing; otherwise the exit point
    itself becomes a synthesized BoundaryPoint endpoint. Discarding the
    fragment (the old behavior, with a fixed 1e-6 match radius) silently
    deleted real curve segments whenever the partner crossing was missing
    or less accurate than 1e-6 — CSX only guarantees ~ptol accuracy.
    """
    from mmcore.geom._nurbs_param_tol import bez_surface_param_tolerance

    fragments: list[_Fragment] = []
    points: list = []
    used: set[int] = set()
    work_budget = getattr(cell, "work_budget", None)

    def _deny_trace_work():
        if work_budget is not None:
            # `charge_cells` is what turns a zero remaining allowance into a
            # hard exhaustion flag.  The additional incomplete bit records
            # that registrations still existed when continuation stopped.
            work_budget.charge_cells(1, "branch_trace")
            work_budget.mark_incomplete()

    if (work_budget is not None
            and (work_budget.exhausted or work_budget.remaining_cells <= 0)):
        _deny_trace_work()
        return fragments, points

    # Per-axis parametric tolerance for the cell's local sub-surfaces.
    # Sizes the marcher's initial/minimal steps and the endpoint matching
    # radius (in GLOBAL coordinates the local tolerance scales by the
    # cell's span on each axis).
    ptol_s, ptol_t = bez_surface_param_tolerance(cell.g1.surface, atol, rational=True)
    ptol_u, ptol_v = bez_surface_param_tolerance(cell.g2.surface, atol, rational=True)
    ptol_local = np.array([float(ptol_s), float(ptol_t), float(ptol_u), float(ptol_v)])
    ptol_local = np.maximum(ptol_local, 1e-12)
    ptol_min = max(float(ptol_local.max()), 1e-9)
    strict_root_tol = _strict_ssx_root_tol(
        cell.g1.surface, cell.g2.surface, rational=True)
    spans = np.array([cell.box[ax][1] - cell.box[ax][0] for ax in range(4)])
    # Global per-axis matching radius: CSX roots and marcher exits are each
    # accurate to ~ptol, so 4x covers both ends with headroom.
    match_tol_global = 4.0 * ptol_local * np.maximum(spans, 1e-15)

    for i, start_cx in enumerate(cell.crossings):
        if work_budget is not None and work_budget.exhausted:
            work_budget.mark_incomplete()
            break
        if i in used:
            continue

        start_local = _global_to_local(start_cx.stuv, cell.box)

        # XYZ distance to the nearest unused partner crossing bounds the
        # marcher's initial xyz step target: step toward the partner, not
        # past it. (xyz, not stuv — step sizing is geometry-driven.)
        cell_h_max = h_max if h_max is not None else max(
            0.05 * _local_diag(cell.g1.surface, rational=True), 4.0 * atol)
        nearest_xyz = float('inf')
        nearest_hint_local = None
        for j, cx in enumerate(cell.crossings):
            if j == i or j in used:
                continue
            d = float(np.linalg.norm(np.asarray(cx.xyz, dtype=np.float64)
                                     - np.asarray(start_cx.xyz, dtype=np.float64)))
            if d < nearest_xyz:
                nearest_xyz = d
                nearest_hint_local = (
                    _global_to_local(cx.stuv, cell.box) - start_local)

        if nearest_xyz == float('inf'):
            h_init = 0.25 * cell_h_max
        else:
            h_init = min(cell_h_max, max(atol, 0.25 * nearest_xyz))

        # Candidate collection across attempts: commit the FIRST substantial
        # fragment (arc > 16·atol, the micro-branch scale) immediately, but
        # keep trying further attempts while only micro fragments came back.
        # Two graze diseases need the extra attempts:
        #  - both plain marches BOUNCE (grazing corner, off-lattice loop) —
        #    attempts 2/3 march from a displaced, corrected interior seed;
        #  - a plain march makes a little progress and then exits through a
        #    face the curve merely GRAZES (corner-sharing bilinear repro:
        #    the arc from the (1,1,0,0) domain corner dipped out at u=0
        #    after 0.142 and the remaining 32-unit arc was silently lost) —
        #    the displaced attempts march PAST the dip and return the full
        #    arc; the longest candidate wins, and genuine micro-fragments
        #    (case 10's 5.3·atol sliver) keep winning when the extra
        #    attempts find nothing longer.
        candidates = []   # (arc_xyz, stuv_global, xyz_local, matched_j)
        tang_seed = None
        for attempt in range(4):
            # At a rank-deficient tangent crossing, ker(Psi') contains both
            # the actual curve direction and a singular transverse direction.
            # Picking SVD's last vector with no hint is arbitrary: on
            # z=(t-1/2)^d it walks the q^d tolerance valley instead of the
            # tangent line, and a roundoff-small residual is then many atol
            # from the root for large d.  A certified partner registration
            # supplies the missing topological direction.  Project its LOCAL
            # 4-D displacement into the full nullspace on the first attempt;
            # rank-3 curves are unchanged except for orientation.
            hint = nearest_hint_local if attempt == 0 else None
            seed_local = start_local
            prepend_crossing = False
            if attempt >= 1:
                if tang_seed is None:
                    tang_seed, _, _ = _ssx_tangent_4d(
                        cell.g1.surface, cell.g2.surface,
                        *start_local, rational=True,
                        direction_hint=nearest_hint_local)
                if tang_seed is None:
                    break
            if attempt == 1:
                hint = -tang_seed
            elif attempt >= 2:
                # Displaced-seed recovery: step the seed a few percent
                # ALONG the curve tangent, Newton-correct back onto the
                # curve, march from the interior point; the registered
                # crossing is prepended so the fragment still starts at
                # the registered stuv (the first chord skips the graze
                # dip within sagitta h²·kappa/2 << atol).
                sign = 1.0 if attempt == 2 else -1.0
                seed_local = None
                for alpha in (0.02, 0.05, 0.1):
                    if (work_budget is not None
                            and not work_budget.charge_cells(
                                1, "branch_trace")):
                        work_budget.mark_incomplete()
                        break
                    cand = np.clip(start_local + sign * alpha * tang_seed,
                                   1e-6, 1.0 - 1e-6)
                    cs, ct, cu, cv, res, sin_ang = _ssx_correct(
                        cell.g1.surface, cell.g2.surface, *cand,
                        rational=True)
                    corr = np.array([cs, ct, cu, cv])
                    if (res < atol * max(sin_ang, 1e-3)
                            and np.all(corr > 1e-9)
                            and np.all(corr < 1.0 - 1e-9)
                            and np.any(np.abs(corr - start_local)
                                       > 4.0 * ptol_local)):
                        seed_local = corr
                        hint = sign * tang_seed
                        prepend_crossing = True
                        break
                if seed_local is None:
                    if work_budget is not None and work_budget.exhausted:
                        break
                    continue

            trace_limit = 400
            if work_budget is not None:
                if work_budget.exhausted or work_budget.remaining_cells <= 0:
                    _deny_trace_work()
                    break
                trace_limit = min(trace_limit, work_budget.remaining_cells)
            trace_stats = {}
            stuv_local, xyz_local, exit_info = _march_to_boundary(
                cell.g1.surface, cell.g2.surface, seed_local,
                atol=atol, rational=True, direction_hint=hint,
                h_init=h_init, h_max=cell_h_max,
                min_step=ptol_min,
                max_points=trace_limit, stats=trace_stats,
            )
            trace_iterations = int(trace_stats.get("iterations", 0))
            if (work_budget is not None and trace_iterations
                    and not work_budget.charge_cells(
                        trace_iterations, "branch_trace")):
                work_budget.mark_incomplete()
                break
            if (trace_iterations >= trace_limit and exit_info is None):
                if work_budget is not None:
                    work_budget.mark_incomplete()
                    if work_budget.remaining_cells <= 0:
                        work_budget.mark_exhausted()

            if len(stuv_local) < 2:
                if work_budget is not None and work_budget.exhausted:
                    break
                continue
            if prepend_crossing:
                stuv_local = np.vstack([np.asarray(start_local)[None, :],
                                        np.asarray(stuv_local)])
                xyz_local = np.vstack([np.asarray(start_cx.xyz,
                                                  dtype=np.float64)[None, :],
                                       np.asarray(xyz_local)])

            # `atol` controls chord accuracy and matching; it is not an
            # equality certificate.  Every continuation vertex must be a
            # roundoff-scale Psi zero before it can justify a branch or a
            # synthesized endpoint.  Otherwise retain only the already
            # certified registration and surface the topology as partial.
            if (work_budget is not None
                    and not work_budget.charge_cells(
                        len(stuv_local), "branch_trace_verify")):
                work_budget.mark_incomplete()
                break
            strict_path = True
            for q in np.asarray(stuv_local, dtype=np.float64):
                p1 = eval_surface(
                    cell.g1.surface, q[0], q[1], rational=True)
                p2 = eval_surface(
                    cell.g2.surface, q[2], q[3], rational=True)
                if float(np.linalg.norm(p1 - p2)) > strict_root_tol:
                    strict_path = False
                    break
            if not strict_path:
                if work_budget is not None:
                    work_budget.mark_incomplete()
                continue

            # Bounce/degenerate detector, in XYZ over the WHOLE path: a
            # march made no real progress only if EVERY sample stayed
            # within the geometric tolerance of the seed (outward bounce
            # or jitter) — try the opposite direction then. Endpoint-only
            # displacement is not enough: a genuine hairpin arc (thin
            # needle loop) returns to within atol of its start while its
            # interior travels far — real geometry, keep it. (The older
            # stuv-vs-ptol test was worse still: it deleted micro-fragments
            # tiny in parameter space but many atol long in xyz, e.g.
            # case 10's final sliver to the v=1 domain corner.)
            path_xyz = np.asarray(xyz_local, dtype=np.float64)
            disp_xyz = float(np.linalg.norm(
                path_xyz - path_xyz[0][None, :], axis=1).max())
            if disp_xyz <= atol:
                continue

            stuv_global = np.empty((len(stuv_local), 4), dtype=np.float64)
            for j in range(len(stuv_local)):
                stuv_global[j] = _local_to_global(stuv_local[j], cell.box)
            stuv_global[0] = start_cx.stuv.copy()

            # Match the exit against the cell's crossings within the
            # parametric tolerance. Consumed crossings stay eligible as
            # endpoints (a corner can terminate two fragments) but only
            # unconsumed ones are removed from the seed pool. Side effects
            # (endpoint stamping, `used` bookkeeping) are DEFERRED to the
            # winning candidate.
            best_j = None
            best_score = float('inf')
            for j, cx in enumerate(cell.crossings):
                if j == i:
                    continue
                diff = np.abs(cx.stuv - stuv_global[-1])
                score = float(np.max(diff / match_tol_global))
                if score < best_score:
                    best_score = score
                    best_j = j

            # A parametric match must also agree in xyz: within the 4·ptol
            # box two genuinely distinct crossings can be many atol apart
            # in space (large surface derivatives). If the xyz check fails,
            # fall through to endpoint synthesis — recoverable, unlike a
            # wrong match which bends the branch end onto the wrong point.
            # The 2·atol radius matches the unification guard exactly.
            matched_j = None
            if (best_j is not None and best_score <= 1.0
                    and float(np.linalg.norm(
                        np.asarray(cell.crossings[best_j].xyz, dtype=np.float64)
                        - np.asarray(xyz_local[-1], dtype=np.float64))) <= 2.0 * atol):
                matched_j = best_j

            arc_xyz = float(np.linalg.norm(
                np.diff(path_xyz, axis=0), axis=1).sum())
            candidates.append((arc_xyz, stuv_global,
                               np.asarray(xyz_local, dtype=np.float64),
                               matched_j, exit_info,
                               np.asarray(stuv_local[-1], dtype=np.float64)))

            # Graze-exit suspicion: the march ended on a face NO registered
            # crossing accounts for, with the curve tangent nearly PARALLEL
            # to that face — the signature of the curve dipping just outside
            # the box and re-entering (corner-sharing bilinear repro: the
            # arc from the (1,1,0,0) domain corner dipped out at u=0 after
            # 0.142 = 142·atol — arc length alone cannot flag it). Keep
            # attempting; the displaced-seed marches (2/3) start PAST the
            # dip and recover the remaining arc; winner-by-length decides.
            graze_exit = False
            if matched_j is None and exit_info is not None:
                tang_exit, _, _ = _ssx_tangent_4d(
                    cell.g1.surface, cell.g2.surface,
                    *np.asarray(stuv_local[-1], dtype=np.float64),
                    rational=True)
                if tang_exit is not None:
                    _ax = exit_info[0]
                    if (abs(float(tang_exit[_ax]))
                            < 0.1 * float(np.linalg.norm(tang_exit))):
                        graze_exit = True
            if arc_xyz > 16.0 * atol and not graze_exit:
                break     # substantial fragment, honest exit — done

        if candidates:
            candidates.sort(key=lambda c: -c[0])
            _, stuv_global, xyz_local, matched_j, exit_info, exit_local = candidates[0]
            if matched_j is not None:
                end_cx = cell.crossings[matched_j]
                stuv_global[-1] = end_cx.stuv.copy()
                xyz_local[-1] = end_cx.xyz.copy()
                used.add(matched_j)
            elif exit_info is not None:
                # No registered crossing here — the marcher just proved one
                # exists (Newton-converged exit on a face). Synthesize it.
                axis = exit_info[0]
                side = 0 if exit_local[axis] < 0.5 else 1
                tang_end, _, _ = _ssx_tangent_4d(
                    cell.g1.surface, cell.g2.surface,
                    *exit_local, rational=True)
                end_cx = BoundaryPoint(
                    stuv=stuv_global[-1].copy(),
                    xyz=xyz_local[-1].copy(),
                    face=(axis, side),
                    tangent_raw=tang_end,
                )
            else:
                # March ended in the interior (truncation or refused exit).
                # The traced points are still Newton-verified curve samples;
                # keep them as an open fragment rather than deleting real
                # geometry. Assembly treats a None endpoint as terminal.
                end_cx = None

            used.add(i)
            fragments.append(_Fragment(
                start_point=start_cx,
                end_point=end_cx,
                stuv_path=stuv_global,
                xyz_path=xyz_local,
            ))

        if (not candidates and i not in used
                and not (work_budget is not None and work_budget.exhausted)):
            # Both directions failed (genuine corner touch or marcher
            # failure). Surface the crossing as an isolated point instead
            # of silently dropping it.
            if (start_cx.multiplicity_polished
                    and work_budget is not None):
                # High-precision polishing proves this point is a root but
                # collapsing a CSX tolerance cluster does not prove whether
                # a branch leaves it.  A successful strict trace above would
                # resolve that ambiguity; a point-only fallback remains
                # explicitly partial (positive-gap endpoint-touch control).
                work_budget.mark_incomplete()
            points.append(SSXPoint(stuv=start_cx.stuv, xyz=start_cx.xyz))

    return fragments, points


def _assembly_spend(work_budget, amount: int = 1,
                    source: str = "assembly") -> bool:
    """Spend bounded post-processing work from the call-wide allowance."""
    if work_budget is None:
        return True
    if work_budget.charge_postprocess(amount):
        return True
    work_budget.mark_incomplete()
    return False


def _unify_fragment_endpoints(fragments: list[_Fragment], unify_tol,
                              unify_atol: float = 1e-3,
                              work_budget=None) -> None:
    """Replace fragment endpoint objects that represent the same physical
    crossing with one canonical `BoundaryPoint` (in place).

    Fragments from adjacent cells reference DIFFERENT objects for the same
    physical point whenever the crossing was discovered independently
    (corner duplicates, re-found cut-face roots) or one side synthesized
    its exit from the marcher. The id-based chain walker can only connect
    fragments sharing the object, so unify endpoints whose stuv agree
    within the per-axis parametric tolerance — design §4.7.4's 1D
    param-matching on shared partitions, generalized to 4D.
    """
    tol = np.asarray(unify_tol, dtype=np.float64)
    objs: list[BoundaryPoint] = []
    index_of: dict[int, int] = {}
    for f in fragments:
        for p in (f.start_point, f.end_point):
            if p is not None and id(p) not in index_of:
                index_of[id(p)] = len(objs)
                objs.append(p)

    n = len(objs)
    parent = list(range(n))

    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # Cluster bounding boxes — cap the merged diameter at 2·tol per axis so
    # transitive chains (A within tol of B, B within tol of C, A NOT within
    # tol of C) cannot merge arbitrarily distant points and fuse distinct
    # branches.
    box_lo = [objs[k].stuv.astype(np.float64).copy() for k in range(n)]
    box_hi = [objs[k].stuv.astype(np.float64).copy() for k in range(n)]

    stopped = False
    for a in range(n):
        for b in range(a + 1, n):
            if not _assembly_spend(work_budget):
                stopped = True
                break
            if not np.all(np.abs(objs[a].stuv - objs[b].stuv) <= tol):
                continue
            # xyz guard: a parametric box is not a metric ball — where
            # surface derivatives are large, points within the stuv box can
            # be many atol apart in space and are genuinely distinct.
            if float(np.linalg.norm(np.asarray(objs[a].xyz, dtype=np.float64)
                                    - np.asarray(objs[b].xyz, dtype=np.float64))) > 2.0 * unify_atol:
                continue
            ra, rb = _find(a), _find(b)
            if ra == rb:
                continue
            merged_lo = np.minimum(box_lo[ra], box_lo[rb])
            merged_hi = np.maximum(box_hi[ra], box_hi[rb])
            if np.any(merged_hi - merged_lo > 2.0 * tol):
                continue
            parent[rb] = ra
            box_lo[ra] = merged_lo
            box_hi[ra] = merged_hi
        if stopped:
            break

    canon = {id(objs[k]): objs[_find(k)] for k in range(n)}
    for f in fragments:
        if f.start_point is not None:
            f.start_point = canon[id(f.start_point)]
        if f.end_point is not None:
            f.end_point = canon[id(f.end_point)]


def _fragment_contained_in(f: _Fragment, g: _Fragment, tol: float,
                           work_budget=None) -> Optional[bool]:
    """Return containment, or ``None`` when its bounded scan was denied."""
    poly = np.asarray(g.xyz_path, dtype=np.float64)
    if len(poly) < 2:
        return False
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom < 1e-30, 1e-30, denom)
    for p in np.asarray(f.xyz_path, dtype=np.float64):
        # The vectorized point/segment scan does O(len(a)) arithmetic. Charge
        # that amount before allocating/processing it, so a long duplicate
        # family cannot hide quadratic work behind one Python loop turn.
        if not _assembly_spend(work_budget, len(a)):
            return None
        ap = p[None, :] - a
        tt = np.clip(np.einsum("ij,ij->i", ap, ab) / denom, 0.0, 1.0)
        proj = a + tt[:, None] * ab
        if float(np.linalg.norm(proj - p[None, :], axis=1).min()) > tol:
            return False
    return True


def _drop_duplicate_fragments(fragments: list[_Fragment], atol: float,
                              work_budget=None) -> list[_Fragment]:
    """Remove fragments whose geometry is contained in another fragment.

    Duplicates arise when a partner seed re-traces a segment that was
    already traced — including against truncated/open fragments, so the
    test is purely geometric (no endpoint-pair precondition): a fragment
    whose EVERY sample lies within 2·atol of a longer kept fragment's
    polyline duplicates it. Fragments are Newton-corrected onto the same
    curve, so true re-traces sit within ~atol of each other, while a
    genuinely distinct second arc of a thin loop deviates by more than the
    tolerance somewhere along its length and is kept. Sorting by arc
    length keeps the most complete trace of each segment.
    """
    def _arc_len(fr: _Fragment) -> float:
        xyz = np.asarray(fr.xyz_path, dtype=np.float64)
        if len(xyz) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())

    arc_work = sum(max(0, len(fr.xyz_path) - 1) for fr in fragments)
    if arc_work and not _assembly_spend(work_budget, arc_work):
        # Failure-safe direction: an unproved duplicate remains visible in
        # the explicitly partial result; no certified fragment is deleted.
        return list(fragments)

    keep: list[_Fragment] = []
    ordered = sorted(fragments, key=_arc_len, reverse=True)
    for pos, f in enumerate(ordered):
        duplicate = False
        for g in keep:
            contained = _fragment_contained_in(
                f, g, 2.0 * atol, work_budget=work_budget)
            if contained is True:
                duplicate = True
                break
            if contained is None:
                # No allowance remains. Keep the current and all following
                # fragments without further containment scans.
                keep.append(f)
                keep.extend(ordered[pos + 1:])
                return keep
        if duplicate:
            continue
        keep.append(f)
    return keep


def _assemble_fragments(
    fragments: list[_Fragment],
    *,
    S1_full=None, S2_full=None, atol_full: float = 1e-3,
    rational_full: bool = True,
    unify_tol=None,
    h_max=None,
    barrier_xyz=None,
    work_budget=None,
) -> list[SSXBranch]:
    """Design §9: chain fragments that share a `BoundaryPoint` endpoint into
    full branches. Two fragments touching the same `BoundaryPoint` object
    represent the same through-curve crossing an internal partition, one
    from each adjacent cell — by Invariant A/B those are the only partial
    branches the point can connect.

    The `S1_full / S2_full / atol_full / rational_full` keyword args enable
    a final closing-segment march for near-closed loops whose chain has a
    small id-graph gap (different `BoundaryPoint` objects produced for the
    same physical point in different parts of the subdivision tree).

    `barrier_xyz` (optional, (K,3)): emitted tangent-point positions.
    Branches TERMINATE at singular points (paper semantics): two
    TRANSVERSAL fragments are never chained through a junction within
    2·atol of a barrier point — doing so paired arms arbitrarily at
    X-crossings and stitched V-spikes through the touch on thin tangent
    loops. Tangential (Φ-traced) fragments are exempt: a tangent CURVE
    legitimately passes through its own tangency locus.
    """
    from collections import defaultdict

    if unify_tol is not None and len(fragments) > 1:
        _unify_fragment_endpoints(
            fragments, unify_tol, unify_atol=atol_full,
            work_budget=work_budget)
        fragments = _drop_duplicate_fragments(
            fragments, atol_full, work_budget=work_budget)

    barrier = None
    if barrier_xyz is not None and len(barrier_xyz):
        barrier = np.asarray(barrier_xyz, dtype=np.float64).reshape(-1, 3)

    def _at_barrier(pt: BoundaryPoint) -> bool:
        if barrier is None or pt is None:
            return False
        if not _assembly_spend(work_budget, len(barrier)):
            # Unknown means "at a barrier" in the failure-safe direction:
            # do not stitch independently certified arms through it.
            return True
        d = np.linalg.norm(barrier - np.asarray(pt.xyz, dtype=np.float64)[None, :], axis=1)
        return bool(d.min() <= 2.0 * atol_full)

    # Map each BoundaryPoint to the fragments that touch it.
    touches: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for i, f in enumerate(fragments):
        if f.start_point is not None:
            touches[id(f.start_point)].append((i, "start"))
        if f.end_point is not None:
            touches[id(f.end_point)].append((i, "end"))

    consumed = [False] * len(fragments)
    branches: list[SSXBranch] = []

    def _pop_neighbour(pt: BoundaryPoint, self_idx: int) -> Optional[tuple[int, str]]:
        pool = touches.get(id(pt), [])
        block_transversal = (_at_barrier(pt)
                             and not fragments[self_idx].tangential)
        for j, role in pool:
            if not _assembly_spend(work_budget):
                return None
            if j == self_idx or consumed[j]:
                continue
            if block_transversal and not fragments[j].tangential:
                continue    # barrier: transversal chains end at the touch
            return j, role
        return None

    for i, f in enumerate(fragments):
        if consumed[i]:
            continue
        consumed[i] = True
        chain: list[tuple[int, bool]] = [(i, False)]

        # Walk forward: current_end becomes the end of the last fragment.
        current_end = f.end_point
        current_idx = i
        while current_end is not None:
            nb = _pop_neighbour(current_end, current_idx)
            if nb is None:
                break
            j, role = nb
            consumed[j] = True
            # If the neighbour's `end` (not `start`) is the shared point, we
            # must traverse it in reverse.
            reverse = (role == "end")
            chain.append((j, reverse))
            g = fragments[j]
            current_end = g.start_point if reverse else g.end_point
            current_idx = j

        # Walk backward.
        current_start = f.start_point
        current_idx = i
        while current_start is not None:
            nb = _pop_neighbour(current_start, current_idx)
            if nb is None:
                break
            j, role = nb
            consumed[j] = True
            # Neighbour's `start` matching means the neighbour runs INTO our
            # current_start, so we must reverse it to prepend.
            reverse = (role == "start")
            chain.insert(0, (j, reverse))
            g = fragments[j]
            current_start = g.end_point if reverse else g.start_point
            current_idx = j

        # Concatenate chain, dropping the duplicated shared-point sample
        # between adjacent pieces.
        stuv_pieces = []
        xyz_pieces = []
        for k, (idx, rev) in enumerate(chain):
            g = fragments[idx]
            stuv_seg = g.stuv_path[::-1] if rev else g.stuv_path
            xyz_seg = g.xyz_path[::-1] if rev else g.xyz_path
            if k == 0:
                stuv_pieces.append(stuv_seg)
                xyz_pieces.append(xyz_seg)
            else:
                stuv_pieces.append(stuv_seg[1:])
                xyz_pieces.append(xyz_seg[1:])

        stuv_full = np.concatenate(stuv_pieces, axis=0)
        xyz_full = np.concatenate(xyz_pieces, axis=0)

        # March the closing segment of near-closed loops.
        #
        # A loop intersection traversed by subdivisions sometimes ends up
        # with a small chain gap: the SAME geometric point on the loop is
        # independently produced as different `BoundaryPoint` objects in
        # cells from different parts of the subdivision tree, so the
        # id-based chain walker can't bridge them. Symptom: a chain whose
        # two free endpoints (BOTH interior — neither on the [0,1]⁴ box
        # boundary) are much closer to each other than to the rest of the
        # chain. Close the loop by actually marching the missing segment
        # from end_point.stuv to start_point.stuv using the
        # known-endpoint marcher; the resulting samples are real curve
        # points (not a duplicate-start placeholder).
        if S1_full is not None and len(xyz_full) >= 4:
            steps = np.linalg.norm(np.diff(xyz_full, axis=0), axis=1)
            steps = steps[steps > 0]
            if len(steps) > 0:
                gap = float(np.linalg.norm(xyz_full[-1] - xyz_full[0]))
                median_step = float(np.median(steps))
                # Only close when both endpoints are strictly INTERIOR. Open
                # boundary-to-boundary branches naturally end at the [0,1]⁴
                # box boundary; we must NOT join those.
                start_interior = bool(np.all((stuv_full[0] > 1e-9) &
                                             (stuv_full[0] < 1 - 1e-9)))
                end_interior = bool(np.all((stuv_full[-1] > 1e-9) &
                                           (stuv_full[-1] < 1 - 1e-9)))
                # Two retrace guards (off-lattice touch+loop pathology): a
                # SHORT open arc (< ~180 deg of a small loop) whose true
                # complement went untraced ALSO has interior endpoints with
                # a small gap — but there the end->start chord points
                # BACKWARD across the arc's opening, so the known-endpoint
                # marcher walks back along the already-traced arc and
                # manufactures an out-and-back "closed" branch (net angular
                # progress 0, half the samples duplicated).
                #  1. Pre-guard: only attempt the close when the traced
                #     path is most-of-a-loop (path length > 3x gap); a
                #     <180-deg arc has path ~ gap (chord) and must stay
                #     open for assembly/dedup to handle.
                #  2. Post-guard: reject a closing segment whose interior
                #     samples ALL lie on the existing polyline (2*atol) —
                #     that is a retrace, not the missing sliver.
                path_len = float(steps.sum())
                if (start_interior and end_interior
                        and median_step > 0
                        and gap < 10.0 * median_step
                        and path_len > 3.0 * gap):
                    close_limit = 2000
                    if work_budget is not None:
                        close_limit = min(
                            close_limit,
                            work_budget.remaining_postprocess_work)
                    close_stats = {}
                    if (close_limit <= 0
                            or not _assembly_spend(work_budget, 0)):
                        # Spend one denied unit to publish hard exhaustion.
                        _assembly_spend(work_budget, 1, "assembly_trace")
                        closing_stuv = np.empty((0, 4))
                        closing_xyz = np.empty((0, 3))
                    else:
                        closing_stuv, closing_xyz = _march_intersection_curve(
                            S1_full, S2_full,
                            stuv_full[-1], stuv_full[0],
                            atol=atol_full, rational=rational_full,
                            h_max=h_max, max_points=close_limit,
                            stats=close_stats,
                        )
                        close_iterations = int(
                            close_stats.get("iterations", 0))
                        if close_iterations:
                            _assembly_spend(
                                work_budget, close_iterations,
                                "assembly_trace")
                        reached_start = (
                            len(closing_xyz) >= 2
                            and float(np.linalg.norm(
                                np.asarray(closing_xyz[-1])
                                - np.asarray(xyz_full[0])))
                            <= 2.0 * atol_full)
                        if (close_iterations >= close_limit
                                and not reached_start
                                and work_budget is not None):
                            work_budget.mark_incomplete()
                            if (work_budget.remaining_postprocess_work
                                    <= 0):
                                work_budget.postprocess_exhausted = True
                                work_budget.mark_exhausted()
                        if not reached_start:
                            closing_stuv = np.empty((0, 4))
                            closing_xyz = np.empty((0, 3))
                    is_retrace = False
                    if len(closing_xyz) > 3:
                        interior = np.asarray(closing_xyz[1:-1],
                                              dtype=np.float64)
                        is_retrace = True
                        for p in interior:
                            if not _assembly_spend(
                                    work_budget, max(1, len(xyz_full) - 1)):
                                # Unknown retrace: reject the optional close.
                                is_retrace = True
                                break
                            if (_dist_point_polyline(p, xyz_full)
                                    > 2.0 * atol_full):
                                is_retrace = False
                                break
                    if len(closing_stuv) >= 2 and not is_retrace:
                        # Skip the first sample (duplicates xyz_full[-1]).
                        stuv_full = np.concatenate(
                            [stuv_full, closing_stuv[1:]], axis=0)
                        xyz_full = np.concatenate(
                            [xyz_full, closing_xyz[1:]], axis=0)

        branch_kind = ("tangential" if any(fragments[idx].tangential for idx, _ in chain)
                       else "transversal")
        branches.append(SSXBranch(curve=(stuv_full, xyz_full), kind=branch_kind))

    # --- Join open branches across sub-tolerance junction gaps ---
    # Guided cuts pass THROUGH discovered crossings, so a small loop is
    # partitioned into arcs whose junction micro-slivers (~1° of arc) are
    # eaten by the containment dedup (its point-to-polyline distance clamps
    # at a keeper's terminal VERTEX, so a sliver extending a fragment's end
    # by < 2·atol per sample looks "contained" — off-lattice touch+loop:
    # four arcs arrived here open with 1–2.4·atol endpoint gaps and were
    # previously force-"closed" individually into out-and-back doubles).
    # Join open, strictly-interior, tangent-consistent endpoint pairs
    # within 4·atol (junction chord sagitta κL²/8 ≈ 2e-5 ≪ atol); a branch
    # whose own two ends meet closes exactly. Junctions within 2·atol of a
    # barrier point are never joined — branches terminate at singularities.
    if len(branches) >= 1:
        def _ends(b):
            xyz = np.asarray(b.curve[1], dtype=np.float64)
            stuv = np.asarray(b.curve[0], dtype=np.float64)
            return stuv, xyz

        def _interior(p4):
            return bool(np.all(p4 > 1e-9) and np.all(p4 < 1.0 - 1e-9))

        def _near_barrier_xyz(p3):
            if barrier is None:
                return False
            if not _assembly_spend(work_budget, len(barrier)):
                return True
            return bool(np.linalg.norm(
                barrier - np.asarray(p3)[None, :], axis=1).min() <= 2.0 * atol_full)

        def _dir(xyz, at_start):
            # unit direction pointing OUT of the branch at the given end
            if len(xyz) < 2:
                return None
            v = xyz[0] - xyz[1] if at_start else xyz[-1] - xyz[-2]
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-15 else None

        def _chord_is_real(pa4, pc4):
            # Same truth test as the valley-fiction filter: estimated
            # true-curve distance at the junction-chord midpoint must be
            # within tolerance — inside a sub-atol grazing valley every
            # RESIDUAL is small, but junctions between genuine arcs also
            # pass, and outside valleys this rejects fiction chords.
            if S1_full is None:
                return True
            mid = 0.5 * (np.asarray(pa4) + np.asarray(pc4))
            p1, du1, dv1 = eval_surface_d1(S1_full, mid[0], mid[1],
                                           rational=rational_full)
            p2, du2, dv2 = eval_surface_d1(S2_full, mid[2], mid[3],
                                           rational=rational_full)
            res = float(np.linalg.norm(p1 - p2))
            N1 = np.cross(du1, dv1)
            N2 = np.cross(du2, dv2)
            n1 = float(np.linalg.norm(N1))
            n2 = float(np.linalg.norm(N2))
            sin_ang = (float(np.linalg.norm(np.cross(N1, N2))) / (n1 * n2)
                       if n1 > 1e-30 and n2 > 1e-30 else 1.0)
            return res / max(sin_ang, 1e-3) <= 2.0 * atol_full

        changed = True
        while changed:
            if not _assembly_spend(work_budget):
                break
            changed = False
            open_ends = []      # (branch_idx, end_is_start, stuv4, xyz3, out_dir)
            scan_denied = False
            for bi, b in enumerate(branches):
                stuv, xyz = _ends(b)
                if len(xyz) < 2:
                    continue
                if float(np.linalg.norm(xyz[0] - xyz[-1])) <= 1e-9:
                    continue    # already exactly closed
                # Only SUBSTANTIAL arcs participate (> 16·atol — the
                # established micro-branch scale): the pass exists for
                # loop arcs partitioned at guided-cut corners. Near-touch
                # valley junk (2-5·atol arcs) must stay open and fall to
                # the micro-branch/sliver filters — joining it self-closed
                # junk blobs and frankenjoined junk onto genuine ring arcs
                # in sub-tolerance clusters (eps=1e-3 touch+loop), after
                # which containment ate the clean ring.
                if not _assembly_spend(
                        work_budget, max(1, len(xyz) - 1)):
                    scan_denied = True
                    break
                arc_b = float(np.linalg.norm(
                    np.diff(xyz, axis=0), axis=1).sum())
                if arc_b <= 16.0 * atol_full:
                    continue
                for at_start in (True, False):
                    p4 = stuv[0] if at_start else stuv[-1]
                    p3 = xyz[0] if at_start else xyz[-1]
                    if not _interior(p4) or _near_barrier_xyz(p3):
                        continue
                    open_ends.append((bi, at_start, p4, p3,
                                      _dir(xyz, at_start)))
            if scan_denied:
                break
            best = None
            for a in range(len(open_ends)):
                for c in range(a + 1, len(open_ends)):
                    if not _assembly_spend(work_budget):
                        scan_denied = True
                        break
                    ia, sa, _, pa, da = open_ends[a]
                    ic, sc, _, pc, dc = open_ends[c]
                    if ia == ic and sa == sc:
                        continue
                    gap = float(np.linalg.norm(pa - pc))
                    if gap > 4.0 * atol_full:
                        continue
                    # tangent consistency: the two out-directions must be
                    # roughly opposed (the curve continues), and when the
                    # gap is resolvable the chord must agree with both.
                    if da is not None and dc is not None:
                        if float(np.dot(da, dc)) > -0.2:
                            continue
                        if gap > 0.25 * atol_full:
                            chord = (pc - pa) / gap
                            if (float(np.dot(da, chord)) < 0.2
                                    or float(np.dot(dc, -chord)) < 0.2):
                                continue
                    if ia == ic:
                        # self-join → closure; require most-of-a-loop
                        xyz = np.asarray(branches[ia].curve[1], dtype=np.float64)
                        path_len = float(np.linalg.norm(
                            np.diff(xyz, axis=0), axis=1).sum())
                        if path_len <= 3.0 * gap:
                            continue
                    if gap > 0.25 * atol_full and not _chord_is_real(
                            open_ends[a][2], open_ends[c][2]):
                        continue
                    if best is None or gap < best[0]:
                        best = (gap, a, c)
                if scan_denied:
                    break
            if scan_denied:
                break
            if best is None:
                break
            _, a, c = best
            ia, sa, *_ = open_ends[a]
            ic, sc, *_ = open_ends[c]
            sa_stuv, sa_xyz = _ends(branches[ia])
            if ia == ic:
                # close the loop exactly: repeat the start sample at the end
                stuv_j = np.concatenate([sa_stuv, sa_stuv[:1]], axis=0)
                xyz_j = np.concatenate([sa_xyz, sa_xyz[:1]], axis=0)
                branches[ia] = SSXBranch(curve=(stuv_j, xyz_j),
                                         kind=branches[ia].kind)
                changed = True
                continue
            sc_stuv, sc_xyz = _ends(branches[ic])
            # orient A to END at the junction, B to START at it
            if sa:
                sa_stuv, sa_xyz = sa_stuv[::-1], sa_xyz[::-1]
            if not sc:
                sc_stuv, sc_xyz = sc_stuv[::-1], sc_xyz[::-1]
            stuv_j = np.concatenate([sa_stuv, sc_stuv], axis=0)
            xyz_j = np.concatenate([sa_xyz, sc_xyz], axis=0)
            kind_j = ("tangential"
                      if "tangential" in (branches[ia].kind, branches[ic].kind)
                      else "transversal")
            keep_i, drop_i = (ia, ic) if ia < ic else (ic, ia)
            branches[keep_i] = SSXBranch(curve=(stuv_j, xyz_j), kind=kind_j)
            del branches[drop_i]
            changed = True

    # --- Branch-level containment dedup (post-join) ---
    # Partial (non-contained) FRAGMENT overlaps are a known round-1 residue:
    # two traversal families can each cover a loop in pieces that pairwise
    # overlap only partially, so fragment containment keeps both families
    # and the join pass then assembles TWO full copies of the same loop.
    # At branch level the copies ARE mutually contained — apply the same
    # proven 2·atol geometric containment (every sample of the shorter
    # within 2·atol of the longer's polyline), longest kept first.
    if len(branches) > 1:
        order = sorted(range(len(branches)),
                       key=lambda k: -len(branches[k].curve[1]))
        kept_idx: list[int] = []
        for idx in order:
            xyz = np.asarray(branches[idx].curve[1], dtype=np.float64)
            contained = False
            containment_unknown = False
            for kidx in kept_idx:
                poly = np.asarray(branches[kidx].curve[1], dtype=np.float64)
                if len(poly) < 2 or not len(xyz):
                    continue
                inside = True
                for p in xyz:
                    if not _assembly_spend(
                            work_budget, max(1, len(poly) - 1)):
                        containment_unknown = True
                        inside = False
                        break
                    if (_dist_point_polyline(p, poly)
                            > 2.0 * atol_full):
                        inside = False
                        break
                if inside:
                    contained = True
                    break
                if containment_unknown:
                    break
            if not contained:
                kept_idx.append(idx)
        branches = [branches[k] for k in sorted(kept_idx)]

    # --- Drop valley-fiction branches (grazing-gap bridges) ---
    # A march seeded at a near-touch grazing corner can exit-commit a single
    # chord that slides along a sub-atol valley from the touch to the loop
    # (the exit-commit gate bypasses the mid-chord check when there is no
    # interior progress): every SAMPLE is Ψ-valid at tolerance, but the
    # chord is not ON the intersection set. Per _ssx_correct's own contract
    # the true-curve distance at a point is ≈ residual / sin_ang; drop a
    # branch only when EVERY chord midpoint fails at 2·atol — genuine
    # branches have good chords (ring chords measure ~1e-4·atol here),
    # bridges are all-fiction (single chord at 4–12·atol). Cheap: real
    # branches exit at their first good chord.
    if S1_full is not None and branches:
        _kept_v = []
        for b in branches:
            stuv_b = np.asarray(b.curve[0], dtype=np.float64)
            if len(stuv_b) < 2:
                _kept_v.append(b)
                continue
            all_bad = True
            for k in range(len(stuv_b) - 1):
                if not _assembly_spend(work_budget):
                    # This is a soundness filter. Without allowance to
                    # verify at least one real chord, omit the branch from
                    # the explicitly partial result rather than publish a
                    # possible grazing-valley connector.
                    break
                mid = 0.5 * (stuv_b[k] + stuv_b[k + 1])
                p1, du1, dv1 = eval_surface_d1(S1_full, mid[0], mid[1],
                                               rational=rational_full)
                p2, du2, dv2 = eval_surface_d1(S2_full, mid[2], mid[3],
                                               rational=rational_full)
                res = float(np.linalg.norm(p1 - p2))
                N1 = np.cross(du1, dv1)
                N2 = np.cross(du2, dv2)
                n1 = float(np.linalg.norm(N1))
                n2 = float(np.linalg.norm(N2))
                sin_ang = (float(np.linalg.norm(np.cross(N1, N2))) / (n1 * n2)
                           if n1 > 1e-30 and n2 > 1e-30 else 1.0)
                if res / max(sin_ang, 1e-3) <= 2.0 * atol_full:
                    all_bad = False
                    break
            if not all_bad:
                _kept_v.append(b)
        branches = _kept_v

    # --- Drop short slivers that lie on top of another branch ---
    # When the fragment graph has a Y-junction (≥3 fragments meeting at the
    # same BoundaryPoint, e.g. because two adjacent cells redundantly traced
    # the same curve segment), the chain walker peels off the main path and
    # leaves the spurious side fragment as a separate short "branch". Such
    # a sliver lies entirely within another (longer) branch in 3-space.
    #
    # Criterion (intentionally narrow to avoid false positives):
    #   - The candidate branch has FEW points (≤ 5).
    #   - Every one of its xyz samples lies within 4·atol of the other
    #     branch's POLYLINE (segments, not samples). Tolerance-scale only:
    #     a diameter-relative threshold (the previous diam·1e-2) deleted
    #     legitimate short branches that merely passed near a long one.
    SLIVER_MAX_PTS = 5
    if len(branches) > 1:
        sliver_tol = 4.0 * atol_full

        keep = []
        order = sorted(range(len(branches)),
                       key=lambda k: -len(branches[k].curve[1]))
        kept_xyz = []
        for idx in order:
            xyz = branches[idx].curve[1]
            is_sliver = False
            sliver_unknown = False
            if len(xyz) <= SLIVER_MAX_PTS:
                for big in kept_xyz:
                    if len(big) < 2:
                        continue
                    inside = True
                    for p in xyz:
                        if not _assembly_spend(
                                work_budget, max(1, len(big) - 1)):
                            sliver_unknown = True
                            inside = False
                            break
                        if (_dist_point_polyline(np.asarray(p), big)
                                > sliver_tol):
                            inside = False
                            break
                    if inside:
                        is_sliver = True
                        break
                    if sliver_unknown:
                        break
            if not is_sliver:
                keep.append(branches[idx])
                kept_xyz.append(xyz)
        branches = keep

    return branches


# ---------------------------------------------------------------------------
# Domain decomposition helpers
# ---------------------------------------------------------------------------

def _choose_multi_cut(crossings_global, box, min_margin: float = 0.05):
    """Design §6.5 (Krishnan & Manocha 1997) — multi-crossing cut.

    Choose a subdivision axis and the *complete set* of distinct interior
    crossing parameter values on that axis. The cell is then split into
    `len(cuts) + 1` strips by sequential de Casteljau.

    Axis selection: prefer the axis with the most valid cut candidates
    (more strips ⇒ each strip's TΨᵢ coefficient hull is tighter and the
    cheap loop-free certificate fires sooner). Tiebreak: widest spread
    of cut values on that axis.

    A candidate cut is *valid* iff its local position is strictly in
    `(min_margin, 1 − min_margin)` — too close to a cell boundary would
    produce a near-zero-width strip.

    Returns `(axis, sorted_cut_values_global)` or `(None, None)` if no
    axis has any valid interior cut.
    """
    if not crossings_global:
        return None, None

    best_axis: Optional[int] = None
    best_cuts: list[float] = []
    best_spread = -1.0

    for axis in range(4):
        lo, hi = box[axis]
        span = hi - lo
        if span <= 0:
            continue
        seen: dict[float, float] = {}  # rounded key → actual global val
        for c in crossings_global:
            val = float(c.stuv[axis])
            local = (val - lo) / span
            if local <= min_margin or local >= 1.0 - min_margin:
                continue
            key = round(val, 10)
            if key not in seen:
                seen[key] = val
        if not seen:
            continue
        cuts = sorted(seen.values())
        spread = cuts[-1] - cuts[0] if len(cuts) > 1 else 0.0
        if (len(cuts) > len(best_cuts)
                or (len(cuts) == len(best_cuts) and spread > best_spread)):
            best_axis = axis
            best_cuts = cuts
            best_spread = spread

    if best_axis is None:
        return None, None
    return best_axis, best_cuts


def _choose_cut(crossings_global, box, min_margin: float = 0.05):
    """Back-compat single-cut selector: returns the first crossing index on
    the best multi-cut axis. Kept only for callers that still expect the
    (cx_idx, axis) tuple; new code should use `_choose_multi_cut`.
    """
    axis, cuts = _choose_multi_cut(crossings_global, box, min_margin)
    if axis is None or not cuts:
        return None, None
    # Pick the crossing whose value is closest to the cell centre on `axis`.
    lo, hi = box[axis]
    target = 0.5 * (lo + hi)
    best_idx = None
    best_dist = float("inf")
    for ci, c in enumerate(crossings_global):
        if any(abs(c.stuv[axis] - cv) < 1e-12 for cv in cuts):
            d = abs(c.stuv[axis] - target)
            if d < best_dist:
                best_dist = d
                best_idx = ci
    return best_idx, axis


def _extract_isoline(S, axis, value):
    """Extract isoline from Bezier surface at parameter value along axis."""
    from mmcore.numeric.bern import de_casteljau_split_nd
    if axis == 0:
        left, _ = de_casteljau_split_nd(S, axis=0, t=value)
        return left[-1, :, :]
    else:
        left, _ = de_casteljau_split_nd(S, axis=1, t=value)
        return left[:, -1, :]


def _isoline_csx_to_global(csx_result, cut_axis, cut_global_val, cell_box, surf_to_split,
                           S1_local=None, S2_local=None, rational=True):
    """Convert CSX results on an isoline to global BoundaryPoint objects.

    The isoline is in the cell's local coords. CSX returns local params.
    We convert everything to global using the cell's box. If the cell's local
    surface nets are provided we also compute the raw 4D tangent at the
    crossing (design §4 / §5) so downstream classification can use it.
    """
    crossings = []
    local_axis = cut_axis if cut_axis < 2 else cut_axis - 2

    for iso_pt in csx_result.get('isolated', []):
        t_crv = float(iso_pt['t'])
        u_oth = float(iso_pt['u'])
        v_oth = float(iso_pt['v'])

        # Build LOCAL stuv first
        stuv_local = np.zeros(4, dtype=np.float64)
        if surf_to_split == 1:
            # Isoline on S1: local_axis param is the cut value (in local)
            # t_crv is param along the other S1 axis (in local)
            # u_oth, v_oth are S2 params (in local)
            local_cut = (cut_global_val - cell_box[cut_axis][0]) / max(cell_box[cut_axis][1] - cell_box[cut_axis][0], 1e-15)
            if local_axis == 0:
                stuv_local[0] = local_cut
                stuv_local[1] = t_crv
            else:
                stuv_local[0] = t_crv
                stuv_local[1] = local_cut
            stuv_local[2] = u_oth
            stuv_local[3] = v_oth
        else:
            stuv_local[0] = u_oth
            stuv_local[1] = v_oth
            local_cut = (cut_global_val - cell_box[cut_axis][0]) / max(cell_box[cut_axis][1] - cell_box[cut_axis][0], 1e-15)
            if local_axis == 0:
                stuv_local[2] = local_cut
                stuv_local[3] = t_crv
            else:
                stuv_local[2] = t_crv
                stuv_local[3] = local_cut

        # Convert to global
        stuv_global = _local_to_global(stuv_local, cell_box)
        # Force the cut axis to exact global value (avoid rounding)
        stuv_global[cut_axis] = cut_global_val

        xyz = np.asarray(iso_pt['point'], dtype=np.float64)

        # Raw 4D tangent (design §4) computed in the cell's local frame;
        # signs are invariant under the positive affine global↔local rescale.
        tang = None
        if S1_local is not None and S2_local is not None:
            tang, _, _ = _ssx_tangent_4d(
                S1_local, S2_local,
                stuv_local[0], stuv_local[1], stuv_local[2], stuv_local[3],
                rational=rational,
            )

        crossings.append(BoundaryPoint(stuv=stuv_global, xyz=xyz, face=(cut_axis, -1),
                                       tangent_raw=tang))

    return crossings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _partition_free_axis(fixed_axis: int) -> int:
    """Return the isoline's free parameter axis for an SSX partition.

    Partitions are isolines of one of the two surfaces: S1 owns axes 0 (s)
    and 1 (t), S2 owns axes 2 (u) and 3 (v). If one of a surface's two axes
    is held fixed by the partition, the other one is free.
    """
    return {0: 1, 1: 0, 2: 3, 3: 2}[fixed_axis]


def _classify_on_axis(local_param: int, tangent_component: float) -> Optional[str]:
    """Design §4 (local_param, sign) table.

    Returns "in" or "out" for the owning cell, or None if the tangent is
    exactly orthogonal to the axis (degenerate case; caller may still record
    the crossing without a direction or skip it).
    """
    if local_param not in (0, 1):
        raise ValueError(f"local_param must be 0 or 1, got {local_param}")
    if tangent_component > 0:
        return "in" if local_param == 0 else "out"
    if tangent_component < 0:
        return "out" if local_param == 0 else "in"
    return None


def _build_cell_partitions(owner_cell: "_Cell",
                           skip: Optional[tuple[int, int]] = None,
                           skip_faces: Optional[list[tuple[int, int]]] = None,
                           ) -> list[PartitionCurve]:
    """Create the partitions corresponding to a cell's 8 box faces.

    `skip_faces` (or legacy single `skip`) lists `(axis, side)` faces to
    omit — used when the caller will splice in shared internal partitions
    in their place (design §5 Invariant A).
    """
    if skip_faces is None:
        skip_faces = []
    if skip is not None:
        skip_faces = list(skip_faces) + [skip]
    skip_set = set(skip_faces)

    parts: list[PartitionCurve] = []
    for axis in range(4):
        free = _partition_free_axis(axis)
        extent = owner_cell.box[free]
        for side_idx in (0, 1):
            if (axis, side_idx) in skip_set:
                continue
            value = owner_cell.box[axis][side_idx]
            p = PartitionCurve(
                axis=axis, value=float(value),
                free_axis=free, global_extent=(float(extent[0]), float(extent[1])),
                adjacents=[owner_cell], registrations=[],
            )
            parts.append(p)
    return parts


# Back-compat: earlier code references this name for the top-level call.
_build_outer_partitions = _build_cell_partitions


def _on_axis_local(global_val: float, lo: float, hi: float, tol: float = 1e-8) -> Optional[int]:
    """Return 0 if `global_val` equals `lo`, 1 if it equals `hi`, else None."""
    if abs(global_val - lo) < tol:
        return 0
    if abs(global_val - hi) < tol:
        return 1
    return None


def _classify_boundary_point(point: BoundaryPoint, cell: "_Cell") -> None:
    """Design §4: produce one IsolineRegistration per on-boundary axis.

    The raw tangent stored on the point carries sign info that is invariant
    under the positive affine global↔local rescale, so classification works
    directly from `point.tangent_raw[i]` and the point's local param derived
    from the cell's box on axis `i`.
    """
    if point.tangent_raw is None:
        return

    for i in range(4):
        local_param = _on_axis_local(point.stuv[i], cell.box[i][0], cell.box[i][1])
        if local_param is None:
            continue  # axis strictly interior for this cell — no registration (§4)

        direction = _classify_on_axis(local_param, float(point.tangent_raw[i]))
        if direction is None:
            continue  # tangent exactly orthogonal to axis — degenerate, skip

        # Find the cell's partition whose fixed axis is i and whose value
        # equals the cell's box.lo (if local_param==0) or box.hi (local_param==1).
        target_value = cell.box[i][local_param]
        match = None
        for p in cell.partitions:
            if p.axis == i and abs(p.value - target_value) < 1e-8:
                match = p
                break
        if match is None:
            continue  # no partition yet recorded on this axis/value for this cell

        reg = IsolineRegistration(
            partition=match,
            param=float(point.stuv[match.free_axis]),
            direction=direction,
            owner=cell,
            point=point,
        )
        match.registrations.append(reg)
        point.registrations.append(reg)


def _split_bern_scalar_tensor(T, axis, t):
    """Split a scalar 4D Bernstein tensor (no trailing value dim) along `axis` at `t`.

    Used to propagate TΨᵢ tensors through the domain-decomposition splits so that
    sub-cells can run the cheap TΨᵢ monotonicity certificate (design §1.2, §6).
    """
    from mmcore.numeric.bern import de_casteljau_split_nd
    T = np.asarray(T, dtype=np.float64)
    with_val = T[..., None]
    left, right = de_casteljau_split_nd(with_val, axis=axis, t=float(t))
    return left[..., 0], right[..., 0]


def _pinned_count(point: BoundaryPoint, cell_box, tol: float = 1e-8) -> int:
    """Count how many of the 4D stuv coordinates sit on a cell boundary."""
    count = 0
    for i in range(4):
        lo, hi = cell_box[i]
        if abs(point.stuv[i] - lo) < tol or abs(point.stuv[i] - hi) < tol:
            count += 1
    return count


def _is_pinned(val, lo, hi, tol=1e-8):
    return abs(val - lo) < tol or abs(val - hi) < tol

# FIXME: Previously, `min_margin=0.05` was used here. This resulted in some segments of certain intersecting branches being omitted in certain cases. I reduced `min_margin` to 1e-5, and this helped in the specific cases I tested. But this is still a rough approach. We need to calculate `min_margin` based on the parametric tolerance or something similar. To be honest, I still don’t really understand why a high min_margin led to candidates being lost
def _compute_split_plan(crossings, cell_box, min_margin=1e-8,
                        cut_tol=None, max_cuts=8):
    """Determine per-surface split axes and values from productive crossings.

    For each crossing, check S1 pair (s,t) and S2 pair (u,v):
    - If both params in a pair are pinned → skip this crossing entirely.
    - If exactly 1 pinned → the free param gives the split value for that surface.
    - If 0 pinned → pick the param closer to center as the split value.

    Returns (s1_axis, s1_cuts, s2_axis, s2_cuts) where each axis is 0/1 for
    S1 or 2/3 for S2, and cuts are sorted global values. Returns None for
    axis/cuts if no productive crossing provides a split for that surface.
    """
    s1_candidates = {}  # axis -> set of global values
    s2_candidates = {}

    for c in crossings:
        stuv = c.stuv
        # Check S1 pair
        s_pin = _is_pinned(stuv[0], cell_box[0][0], cell_box[0][1])
        t_pin = _is_pinned(stuv[1], cell_box[1][0], cell_box[1][1])
        # Check S2 pair
        u_pin = _is_pinned(stuv[2], cell_box[2][0], cell_box[2][1])
        v_pin = _is_pinned(stuv[3], cell_box[3][0], cell_box[3][1])

        if (s_pin and t_pin) or (u_pin and v_pin):
            continue

        # S1 split: only when exactly 1 of (s,t) is pinned.
        # If 0 pinned → no guided split for S1 from this crossing (→ midpoint).
        if s_pin and not t_pin:
            s1_candidates.setdefault(1, set()).add(float(stuv[1]))
        elif t_pin and not s_pin:
            s1_candidates.setdefault(0, set()).add(float(stuv[0]))

        # S2 split: only when exactly 1 of (u,v) is pinned.
        if u_pin and not v_pin:
            s2_candidates.setdefault(3, set()).add(float(stuv[3]))
        elif v_pin and not u_pin:
            s2_candidates.setdefault(2, set()).add(float(stuv[2]))

    def _pick_best(candidates, box):
        if not candidates:
            return None, None
        best_axis = max(candidates, key=lambda a: len(candidates[a]))
        lo, hi = box[best_axis]
        span = hi - lo
        cuts = sorted(v for v in candidates[best_axis]
                      if min_margin < (v - lo) / span < 1 - min_margin)
        if not cuts:
            return None, None
        # CSX/Newton re-finds the same cut coordinate with a few ULPs of
        # variation. Exact-float sets turned those into dozens of distinct
        # planes and a Cartesian child explosion. Coalescing GUIDE planes
        # is non-destructive (all crossings remain on the cells); if the
        # remaining fanout is still large, use the ordinary midpoint split,
        # which is always a sound subdivision fallback.
        tol = (float(cut_tol[best_axis]) if cut_tol is not None
               else 128.0 * np.finfo(float).eps * max(1.0, abs(lo), abs(hi)))
        clustered = []
        for value in cuts:
            if not clustered or abs(value - clustered[-1]) > tol:
                clustered.append(value)
        cuts = clustered
        if len(cuts) > max_cuts:
            return None, None
        return best_axis, cuts

    s1_axis, s1_cuts = _pick_best(s1_candidates, cell_box)
    s2_axis, s2_cuts = _pick_best(s2_candidates, cell_box)
    return s1_axis, s1_cuts, s2_axis, s2_cuts


def _split_surface_multi(g, axis_4d, cut_values, cell_box):
    """Multi-cut a GaussMapBern along one axis. Returns list of pieces."""
    local_axis = axis_4d if axis_4d < 2 else axis_4d - 2
    lo, hi = cell_box[axis_4d]
    remain = g
    remain_lo = lo
    pieces = []
    for cv in cut_values:
        local_cut = (cv - remain_lo) / (hi - remain_lo)
        left, right = (remain.split_u(local_cut) if local_axis == 0
                       else remain.split_v(local_cut))
        pieces.append(left)
        remain = right
        remain_lo = cv
    pieces.append(remain)
    return pieces


def _split_tensor_multi(T, axis_4d, cut_values, cell_box):
    """Multi-cut a TΨᵢ tensor along one axis. Returns list of pieces."""
    lo, hi = cell_box[axis_4d]
    remain = T
    remain_lo = lo
    pieces = []
    for cv in cut_values:
        local_cut = (cv - remain_lo) / (hi - remain_lo)
        left, right = _split_bern_scalar_tensor(remain, axis=axis_4d, t=local_cut)
        pieces.append(left)
        remain = right
        remain_lo = cv
    pieces.append(remain)
    return pieces


def _csx_on_cut_face(cell, cut_axis: int, cut_global_val: float, atol: float):
    """Run boundary CSX on one cut face of a cell.

    Extracts the isoline of the surface that owns `cut_axis` at the local
    parameter corresponding to `cut_global_val`, runs `bez_csx` against the
    other surface, and returns the results as global `BoundaryPoint` objects.

    This is the paper §5.1 "compute new xsection points on each dividing line".
    """
    surf_to_split = 1 if cut_axis < 2 else 2
    local_axis = cut_axis if cut_axis < 2 else cut_axis - 2
    cell_lo, cell_hi = cell.box[cut_axis]
    cell_span = cell_hi - cell_lo
    if cell_span < 1e-15:
        return []
    cut_local = (cut_global_val - cell_lo) / cell_span

    csx_fn = cell.csx_fn or bez_csx
    if surf_to_split == 1:
        isoline = _extract_isoline(cell.g1.surface, local_axis, cut_local)
        csx_result = csx_fn(isoline, cell.g2.surface, atol=atol, rational=True)
    else:
        isoline = _extract_isoline(cell.g2.surface, local_axis, cut_local)
        csx_result = csx_fn(isoline, cell.g1.surface, atol=atol, rational=True)
    csx_result['isolated'] = list(filter(
        lambda x: not (((1 - x['t']) < 1e-6) or (x['t'] < 1e-6)),
        csx_result.get('isolated', [])))
    return _isoline_csx_to_global(
        csx_result, cut_axis, cut_global_val, cell.box, surf_to_split,
        S1_local=cell.g1.surface, S2_local=cell.g2.surface, rational=True,
    )


def _pick_midpoint_axis(cell) -> int:
    """Pick the axis to split at 0.5 when no productive cuts exist."""
    return cell.depth % 4


def _midpoint_split(cell, axis: int, atol: float):
    """Split cell at midpoint of `axis`. Returns (left, right, mid_global, new_crossings).

    Performs a single binary cut at local 0.5 on the chosen axis, runs CSX on
    the new cut face, deduplicates against inherited crossings, and returns the
    two sub-cells with their crossings and partitions fully set up.
    """
    surf_to_split = 1 if axis < 2 else 2
    local_axis = axis if axis < 2 else axis - 2
    lo, hi = cell.box[axis]
    mid_global = 0.5 * (lo + hi)

    # Split surface / Gauss map
    if surf_to_split == 1:
        left_g1, right_g1 = (cell.g1.split_u(0.5) if local_axis == 0
                             else cell.g1.split_v(0.5))
        left_g2, right_g2 = cell.g2, cell.g2
    else:
        left_g2, right_g2 = (cell.g2.split_u(0.5) if local_axis == 0
                             else cell.g2.split_v(0.5))
        left_g1, right_g1 = cell.g1, cell.g1

    # Split TΨᵢ tensors
    left_T1, right_T1 = _split_bern_scalar_tensor(cell.T1, axis=axis, t=0.5)
    left_T2, right_T2 = _split_bern_scalar_tensor(cell.T2, axis=axis, t=0.5)
    left_T3, right_T3 = _split_bern_scalar_tensor(cell.T3, axis=axis, t=0.5)
    left_T4, right_T4 = _split_bern_scalar_tensor(cell.T4, axis=axis, t=0.5)

    left_box = list(cell.box)
    left_box[axis] = (lo, mid_global)
    left_box = tuple(left_box)

    right_box = list(cell.box)
    right_box[axis] = (mid_global, hi)
    right_box = tuple(right_box)

    # CSX on the new cut face
    new_crossings = _csx_on_cut_face(cell, axis, mid_global, atol)


    # Invariant C dedup: unify with inherited crossings
    deduped_new: list = []
    for nc in new_crossings:
        assert abs(nc.stuv[axis] - mid_global) < 1e-8, \
            f"CSX crossing must be on cut face: stuv[{axis}]={nc.stuv[axis]}, expected {mid_global}"
        match = None
        for ec in cell.crossings:
            if np.linalg.norm(ec.stuv - nc.stuv) < atol:
                match = ec
                break
        if match is None:
            pc = _pinned_count(nc, left_box if nc.stuv[axis] <= mid_global + 1e-10 else right_box)
            assert pc == 1, \
                f"New crossing must be 1-pinned in its strip, got {pc}: stuv={nc.stuv}"
            deduped_new.append(nc)
        else:
            match.stuv[axis] = mid_global

    # Pool = inherited + genuinely new
    all_cx = list(cell.crossings) + deduped_new

    # Distribute crossings to strips
    left_cx = [c for c in all_cx if c.stuv[axis] <= mid_global + 1e-10]
    right_cx = [c for c in all_cx if c.stuv[axis] >= mid_global - 1e-10]

    # Separate genuinely new crossings per strip (for next-level cut decisions)
    left_new = [c for c in deduped_new if c.stuv[axis] <= mid_global + 1e-10]
    right_new = [c for c in deduped_new if c.stuv[axis] >= mid_global - 1e-10]

    # Propagate F_sq alongside TΨᵢ
    if cell.F_sq is not None:
        left_F, right_F = _split_bern_scalar_tensor(cell.F_sq, axis=axis, t=0.5)
    else:
        left_F = right_F = None

    left_cell = _Cell(g1=left_g1, g2=left_g2, crossings=left_cx, box=left_box,
                      depth=cell.depth + 1,
                      T1=left_T1, T2=left_T2, T3=left_T3, T4=left_T4,
                      new_crossings=left_new,
                      F_sq=left_F, w_scale=cell.w_scale,
                      work_budget=cell.work_budget, csx_fn=cell.csx_fn)
    right_cell = _Cell(g1=right_g1, g2=right_g2, crossings=right_cx, box=right_box,
                       depth=cell.depth + 1,
                       T1=right_T1, T2=right_T2, T3=right_T3, T4=right_T4,
                       new_crossings=right_new,
                       F_sq=right_F, w_scale=cell.w_scale,
                       work_budget=cell.work_budget, csx_fn=cell.csx_fn)

    # Partitions: skip the cut-axis face, splice in shared internal partition
    shared_free = _partition_free_axis(axis)
    shared_extent = cell.box[shared_free]
    shared_partition = PartitionCurve(
        axis=axis, value=float(mid_global),
        free_axis=shared_free,
        global_extent=(float(shared_extent[0]), float(shared_extent[1])),
        adjacents=[left_cell, right_cell], registrations=[],
    )

    left_cell.partitions = _build_cell_partitions(left_cell, skip=(axis, 1))
    left_cell.partitions.append(shared_partition)

    right_cell.partitions = _build_cell_partitions(right_cell, skip=(axis, 0))
    right_cell.partitions.append(shared_partition)

    # Classify all crossings per sub-cell
    for c in left_cx:
        _classify_boundary_point(c, left_cell)
    for c in right_cx:
        _classify_boundary_point(c, right_cell)

    return left_cell, right_cell


@dataclass
class _Cell:
    """A sub-problem in the domain decomposition stack.

    Design §6 lists the cell's state as `box, S1, S2, g1, g2, T1..T4,
    partitions, depth`. The implementation also carries a `crossings` list;
    per the design, crossings are derivable from
    `[r.point for p in partitions for r in p.registrations]` (deduped by
    identity). The `crossings` field is kept here as scaffolding used by
    `_choose_cut` and the subdivision's L/R distribution step — removing it
    requires rewriting those in terms of partition registrations, which is
    a separate refactor.
    """
    g1: object                          # GaussMapBern for S1 sub-patch (local [0,1]²)
    g2: object                          # GaussMapBern for S2 sub-patch (local [0,1]²)
    crossings: list                     # BoundaryPoint (global coords) — redundant with partitions[].registrations
    box: tuple                          # 4D parameter range in GLOBAL coords
    depth: int = 0
    # TΨᵢ Bernstein tensors for this sub-cell's local [0,1]⁴ — propagated by
    # de Casteljau-splitting the parent's tensors along the cut axis (never
    # recomputed; see design §1.2).
    T1: Optional[NDArray[np.float64]] = None
    T2: Optional[NDArray[np.float64]] = None
    T3: Optional[NDArray[np.float64]] = None
    T4: Optional[NDArray[np.float64]] = None
    # Isolines bounding this cell (design §5).
    partitions: list[PartitionCurve] = field(default_factory=list)
    # Crossings discovered by CSX on this cell's NEW cut faces (not inherited
    # from parent). Only these drive the next subdivision decision. Inherited
    # crossings are in `crossings` for tracing but should not be re-used for
    # cutting — cutting at inherited coordinates produces zero-info strips.
    new_crossings: list = field(default_factory=list)
    # Squared-distance Bernstein net (4D scalar tensor), propagated from the
    # top level by de Casteljau-splitting alongside TΨᵢ. Avoids reconstructing
    # the net per cell; only the cheap min-of-net check runs per cell.
    F_sq: Optional[NDArray[np.float64]] = None
    # Top-level w_scale (max weight product) — constant across the tree.
    w_scale: float = 1.0
    # Ledger L4: a probe-only cell hunts Δ-touches the parent's center
    # witness missed (loop-free arm, hull gate fired, witness failed). It
    # carries ONLY nets/boxes — no crossings, no partitions — and its
    # lifecycle is the lean probe arm of the queue loop: prunes → hull
    # gate → center witness → lean net-split descent. It never traces
    # (the parent already traced its geometry — that is what keeps the
    # descent duplication-free) and never runs cut-face CSX.
    probe_only: bool = False
    # Every nested solver and every descendant spends from the top-level
    # call's shared budget.  ``csx_fn`` is the bounded adapter installed by
    # bez_ssx; keeping it on the cell also covers helper subdivision paths.
    work_budget: Optional[_SSXSoftBudget] = None
    csx_fn: Optional[object] = None


def _probe_children(cell):
    """Lean 2x2 net-split of a probe(-able) cell (ledger L4).

    Splits each surface at the midpoint of its longest GLOBAL axis and
    de Casteljau-propagates T1..T4 and F_sq — nothing else: no cut-face
    CSX, no crossing distribution, no partitions (the children are
    probe_only and never trace). Cost per level is four de Casteljau
    sweeps vs the main subdivision's CSX-per-cut (measured ~60 ms per
    fall-through on case 5's 20 near-tangent cells — the lean descent is
    what keeps the L4 recovery inside the coverage timing budget).

    Termination of the descent (probe arm): AABB/GJK/F_sq prunes, hull
    gate clearing (near-tangent cells: T hulls exclude 0 within a few
    levels), witness success (touch emitted), or the 4·unify_tol size
    gate — each level halves one axis per surface, longest-first, so all
    four spans reach the gate in finitely many levels.
    """
    b = cell.box
    s1_axis = 0 if (b[0][1] - b[0][0]) >= (b[1][1] - b[1][0]) else 1
    s2_axis = 2 if (b[2][1] - b[2][0]) >= (b[3][1] - b[3][0]) else 3
    g1_lr = cell.g1.split_u(0.5) if s1_axis == 0 else cell.g1.split_v(0.5)
    g2_lr = cell.g2.split_u(0.5) if s2_axis == 2 else cell.g2.split_v(0.5)

    def _split2(T):
        # -> [i1][i2] pieces along (s1_axis, s2_axis), local midpoints
        if T is None:
            return [[None, None], [None, None]]
        a, c = _split_bern_scalar_tensor(T, axis=s1_axis, t=0.5)
        a1, a2 = _split_bern_scalar_tensor(a, axis=s2_axis, t=0.5)
        c1, c2 = _split_bern_scalar_tensor(c, axis=s2_axis, t=0.5)
        return [[a1, a2], [c1, c2]]

    Ts = [_split2(T) for T in (cell.T1, cell.T2, cell.T3, cell.T4)]
    Fs = _split2(cell.F_sq)
    m1 = 0.5 * (b[s1_axis][0] + b[s1_axis][1])
    m2 = 0.5 * (b[s2_axis][0] + b[s2_axis][1])
    out = []
    for i1 in range(2):
        for i2 in range(2):
            sub = list(b)
            sub[s1_axis] = (b[s1_axis][0], m1) if i1 == 0 else (m1, b[s1_axis][1])
            sub[s2_axis] = (b[s2_axis][0], m2) if i2 == 0 else (m2, b[s2_axis][1])
            out.append(_Cell(
                g1=g1_lr[i1], g2=g2_lr[i2], crossings=[], box=tuple(sub),
                depth=cell.depth + 1,
                T1=Ts[0][i1][i2], T2=Ts[1][i1][i2],
                T3=Ts[2][i1][i2], T4=Ts[3][i1][i2],
                new_crossings=[], F_sq=Fs[i1][i2], w_scale=cell.w_scale,
                probe_only=True, work_budget=cell.work_budget,
                csx_fn=cell.csx_fn,
            ))
    return out


def bez_ssx(
    S1,
    S2,
    atol=1e-3,
    rational=True,
    max_depth=13,
    max_xyz_step=None,
    max_cells=250_000,
    max_csx_calls=10_000,
    csx_max_cells=100_000,
    boundary_csx_max_cells=20_000,
    csx_max_results=128,
    max_output_items=1_024,
    max_postprocess_work=None,
) -> dict:
    """Bezier surface-surface intersection v5.

    Iterative stack-based domain decomposition.
    All crossings and branch endpoints are in GLOBAL [0,1]⁴ coordinates.
    Surfaces in sub-cells are in LOCAL [0,1]² (De Casteljau reparameterized).
    Conversion between local and global uses the cell's box.

    ``max_cells`` and ``max_csx_calls`` are shared by the entire search,
    including nested singular solvers and CSX searches. Top-level independent
    boundary probes use ``boundary_csx_max_cells``; topology-critical internal
    cuts use ``csx_max_cells`` directly, so they never repay a discarded
    smaller attempt. Assembly and
    containment use one separate call-wide ``max_postprocess_work`` cap
    (default: ``max_cells``), so a stopped search can still assemble its
    certified partial fragments without opening an unbounded second phase.
    Exhaustion is a soft stop: already-certified output is returned together
    with ``budget_exhausted=True`` and usage counters.

    Returns dict with 'branches', 'points', 'singularities', and budget
    status fields.
    """
    S1 = np.asarray(S1, dtype=np.float64)
    S2 = np.asarray(S2, dtype=np.float64)
    budget = _SSXSoftBudget(
        max_cells=max(0, int(max_cells)),
        max_csx_calls=max(0, int(max_csx_calls)),
        max_output_items=max(0, int(max_output_items)),
        max_postprocess_work=(None if max_postprocess_work is None
                              else max(0, int(max_postprocess_work))),
    )

    def _result(branches=None, points=None, singularities=None):
        result = {
            'branches': [] if branches is None else branches,
            'points': [] if points is None else points,
            'singularities': [] if singularities is None else singularities,
        }
        result.update(budget.result_fields())
        return result

    # A zero public allowance is a hard promise that no expensive solver
    # setup runs.  In particular, the 4-D squared-distance Bernstein net can
    # allocate work quadratic in the tensor-product control count before the
    # first subdivision cell exists.
    if budget.max_cells <= 0 or budget.max_csx_calls <= 0:
        budget.mark_exhausted()
        return _result()

    # Reject disjoint control hulls before preflighting the distance net.
    if _ssx_control_aabbs_disjoint(S1, S2, rational=rational):
        return _result()

    if rational:
        S1_h_top, S2_h_top = S1, S2
    else:
        S1_h_top = np.concatenate(
            [S1, np.ones(S1.shape[:-1] + (1,))], axis=-1)
        S2_h_top = np.concatenate(
            [S2, np.ones(S2.shape[:-1] + (1,))], axis=-1)

    # The current distance-net constructor forms a pairwise Gram tensor over
    # the full four-axis control product.  Charge one shared work unit per
    # 128 pair coefficients before entering it; denial returns immediately
    # instead of allowing a high-degree input to freeze or exhaust memory.
    control_product = (
        int(S1.shape[0]) * int(S1.shape[1])
        * int(S2.shape[0]) * int(S2.shape[1]))
    pair_coefficients = control_product * control_product
    precompute_units = max(1, (pair_coefficients + 127) // 128)
    if not budget.charge_cells(precompute_units, "precompute"):
        return _result()
    F_sq_top = surface_surface_distance_squared_net_homog(
        S1_h_top, S2_h_top, rational=True)

    def _run_csx(
        curve, surface, *, local_truncation_is_soft=False, **kwargs,
    ):
        """Bounded adapter used by every boundary and cut-face CSX call."""
        if budget.remaining_cells <= 0 or not budget.charge_csx_call():
            budget.mark_exhausted()
            return {
                'isolated': [], 'overlaps': [], 'parameter_fibers': [],
                'budget_exhausted': True, 'cells_processed': 0,
            }
        def _attempt(allowance):
            attempt_kwargs = dict(kwargs)
            attempt_kwargs['max_cells'] = allowance
            attempt_kwargs['max_results'] = max(1, int(csx_max_results))
            attempt_result = bez_csx(curve, surface, **attempt_kwargs)
            attempt_used = max(
                0, int(attempt_result.get('cells_processed', 0)))
            if not budget.charge_cells(attempt_used, "csx"):
                budget.mark_exhausted()
            return attempt_result, attempt_used

        per_call_cap = (boundary_csx_max_cells
                        if local_truncation_is_soft else csx_max_cells)
        allowance = min(max(1, int(per_call_cap)), budget.remaining_cells)
        result, used = _attempt(allowance)
        if result.get('budget_exhausted', False):
            # A locally truncated CSX root set is not safe input for SSX
            # topology decisions, even if the outer allowance has room.
            # The eight TOP-LEVEL boundary faces are independent, however:
            # discard this face's partial entities, mark the public result
            # incomplete, and continue to the remaining faces while shared
            # work remains.  This preserves certified output from later
            # faces (case 14's second collapsed apex fiber).  Internal cut
            # faces are dependencies of child topology and keep the hard-stop
            # behavior below.
            if local_truncation_is_soft and not budget.exhausted:
                budget.mark_incomplete()
                return {
                    'isolated': [], 'overlaps': [],
                    'parameter_fibers': [],
                    'budget_exhausted': True,
                    'boundary_topology_complete': False,
                    'cells_processed': used,
                }
            budget.mark_exhausted()
        return result

    def _run_top_boundary_csx(curve, surface, **kwargs):
        return _run_csx(
            curve, surface, local_truncation_is_soft=True, **kwargs)

    # Reproducibility: the Gauss-separability witness search draws from
    # module-global PRNGs; without a per-call reset, bit-identical inputs
    # returned different branch topologies depending on call history
    # (trace-vs-subdivide flips on marginal near-tangent cells).
    reset_witness_rng()

    # --- Level 1: Pruning ---
    if _prune_ssx_cell(
            S1, S2, atol, rational=rational, F=F_sq_top):
        return _result()

    # --- Build GaussMapBern ONCE ---
    if rational:
        g1 = GaussMapBern.from_surf(S1, rational=True)
        g2 = GaussMapBern.from_surf(S2, rational=True)
    else:
        S1_h = np.concatenate([S1, np.ones(S1.shape[:-1] + (1,))], axis=-1)
        S2_h = np.concatenate([S2, np.ones(S2.shape[:-1] + (1,))], axis=-1)
        g1 = GaussMapBern.from_surf(S1_h, rational=True)
        g2 = GaussMapBern.from_surf(S2_h, rational=True)

    # --- Level 2: Boundary CSX (8 calls, once) ---
    # Crossings are already in global [0,1]⁴ coords (top-level box is [0,1]⁴).
    # _find_ssx_boundary_zeros already filters crossings that coincide with
    # overlap endpoints by stuv (design §10.6); nothing more to do here.
    boundary_parameter_fibers = []
    crossings, boundary_overlaps = _find_ssx_boundary_zeros(
        S1, S2, atol, rational=rational, csx_fn=_run_top_boundary_csx,
        fiber_sink=boundary_parameter_fibers)
    _capped_boundary_fibers = []
    budget.extend_output(
        _capped_boundary_fibers, boundary_parameter_fibers,
        "boundary_parameter_fiber")
    boundary_parameter_fibers = _capped_boundary_fibers
    if boundary_parameter_fibers:
        # A fiber is certified output as a set, but its incident SSI branch
        # multiplicity is not.  Preserve useful branches while refusing a
        # complete-topology claim until that limiting topology is proved.
        budget.mark_incomplete()
    _overlap_candidates = _overlaps_to_branches(
        boundary_overlaps, S1, atol, rational)
    overlap_branches = []
    budget.extend_output(overlap_branches, _overlap_candidates, "branch")
    if budget.exhausted:
        return _result(branches=overlap_branches)

    # --- Level 3: TΨᵢ (once at top level) ---
    if rational:
        P1_cart = S1[..., :-1] / S1[..., -1:]
        P2_cart = S2[..., :-1] / S2[..., -1:]
    else:
        P1_cart = S1
        P2_cart = S2
    # TΨᵢ minor nets. For rational input with NON-UNIFORM weights the
    # dehomogenized control net P/w does NOT represent the rational surface
    # R = P/w (control-point quotients != function quotient), so its
    # derivatives — hence its minors — describe the WRONG surface pair, and
    # the whole tangency chain silently loses C2 contacts (ledger L9,
    # measured on a weight-2 rationalized paraboloid). Build the minors from
    # the quotient-rule NUMERATOR columns instead: each numerator minor
    # equals the true rational minor times a strictly positive power-of-W
    # factor, so it shares the true minor's zero set and sign structure —
    # all every consumer needs (hull gates, sign-definiteness certificates,
    # monotonicity, deflation zero-finding). Unit weights (all coverage
    # cases + every polynomial test) take the exact polynomial path,
    # BIT-IDENTICAL to pre-L9. (`S1[..., -1]` is a genuine weight column
    # only when `rational`; the short-circuit keeps it unread otherwise.)
    _w1_uniform = not rational or _weight_net_uniform(S1)
    _w2_uniform = not rational or _weight_net_uniform(S2)
    if rational and not (_w1_uniform and _w2_uniform):
        from mmcore.numeric.intersection.ssx._ssx5_singular import (
            minors_Tpsi_rational)
        T1, T2, T3, T4 = minors_Tpsi_rational(S1, S2)
    else:
        T1, T2, T3, T4 = minors_Tpsi_from_control_nets(P1_cart, P2_cart)

    # --- Top-level cell + outer partitions (design §5) ---
    box = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    T1_arr = _tpsi_to_numpy(T1)
    T2_arr = _tpsi_to_numpy(T2)
    T3_arr = _tpsi_to_numpy(T3)
    T4_arr = _tpsi_to_numpy(T4)

    # The preflighted top-level net is propagated by De Casteljau split.
    _, S1w_top = extract_weights(S1_h_top, rational=True)
    _, S2w_top = extract_weights(S2_h_top, rational=True)
    w_scale_top = _weight_max_product(S1w_top.ravel(), S2w_top.ravel())

    # Global per-axis parametric tolerance (from the full surfaces):
    # crossings/exit points whose stuv agree within this radius are the
    # same physical point. CSX roots and marcher exits are each accurate
    # to ~ptol, so 4x covers both ends.
    from mmcore.geom._nurbs_param_tol import bez_surface_param_tolerance
    _gp_s, _gp_t = bez_surface_param_tolerance(S1_h_top, atol, rational=True)
    _gp_u, _gp_v = bez_surface_param_tolerance(S2_h_top, atol, rational=True)
    unify_tol = 4.0 * np.maximum(
        np.array([float(_gp_s), float(_gp_t), float(_gp_u), float(_gp_v)]), 1e-12)
    # Destructive dedup uses the tight 1·ptol radius (plus an xyz guard at
    # the call sites); the looser 4·ptol box is reserved for matching /
    # unification where a miss is recoverable.
    dedup_tol = unify_tol / 4.0

    # Ledger L6(ii): parametric boxes of the detected coplanar overlap
    # regions — every tangency emission site skips witness roots inside
    # them (the overlap interior is a 2-dim C2 set the overlap branches
    # already report; an "isolated" point there is a phantom).
    overlap_boxes = _overlap_region_boxes(
        boundary_overlaps, S1_h_top, atol, unify_tol)

    # Global xyz step ceiling for all marchers. NOT an accuracy criterion —
    # accuracy is governed by the chord-deviation (sagitta) control at
    # 2·atol, which measurements show is the binding constraint nearly
    # everywhere (point counts and max deviation are almost identical with
    # the cap at 0.01·diag vs 0.5·diag). The cap only bounds how much
    # geometry a single chord can skip, keeping the finite-probe deviation
    # checks (endpoint tangents + one midpoint) inside their trust region:
    # a chord spanning multiple inflections can fool O(1) probes, and the
    # midpoint corrector's Newton seed degrades with chord length.
    # 0.05·diag serves that role while binding almost nowhere.
    _pts_joint = np.vstack([P1_cart.reshape(-1, 3), P2_cart.reshape(-1, 3)])
    _diag = float(np.linalg.norm(_pts_joint.max(axis=0) - _pts_joint.min(axis=0)))
    h_max = max_xyz_step if max_xyz_step is not None else max(0.05 * _diag, 4.0 * atol)

    promoted_fiber_fragment = None
    if boundary_parameter_fibers and not budget.exhausted:
        promotion_limit = min(512, budget.remaining_cells)
        if promotion_limit <= 0:
            budget.mark_exhausted()
        else:
            promoted_fibers, promotion_stats, promoted_path = (
                _promote_transversal_boundary_fiber_pair(
                    S1_h_top, S2_h_top, boundary_parameter_fibers,
                    crossings, boundary_overlaps,
                    atol=atol, unify_tol=unify_tol, h_max=h_max,
                    max_points=promotion_limit))
            promotion_work = max(
                0, int(promotion_stats.get("iterations", 0)))
            if not budget.charge_cells(promotion_work, "fiber_promotion"):
                promoted_fibers = []
                promoted_path = None
            if promoted_fibers and promoted_path is not None:
                promoted_fiber_fragment = _Fragment(
                    start_point=promoted_fibers[0],
                    end_point=promoted_fibers[1],
                    stuv_path=promoted_path[0],
                    xyz_path=promoted_path[1],
                    tangential=False,
                )

    top_cell = _Cell(
        g1=g1, g2=g2, crossings=crossings, box=box, depth=0,
        T1=T1_arr, T2=T2_arr, T3=T3_arr, T4=T4_arr,
        new_crossings=list(crossings),
        F_sq=F_sq_top, w_scale=w_scale_top,
        work_budget=budget, csx_fn=_run_csx,
    )
    top_cell.partitions = _build_outer_partitions(top_cell)

    # §4 classification: one IsolineRegistration per on-boundary axis per
    # boundary crossing.
    for c in crossings:
        _classify_boundary_point(c, top_cell)

    # NOTE: do NOT early-return when there are no boundary crossings.
    # A purely interior intersection (e.g. case 7's closed loop strictly
    # inside [0,1]⁴) has zero boundary crossings on the top-level box but is
    # still a real intersection. The midpoint-fallback path inside the
    # subdivision loop discovers it. Cheap certificates (AABB, GJK, F_sq,
    # loop_free) inside the loop will terminate cells with no actual
    # intersection.

    # --- Iterative domain decomposition (single code path, design §6) ---
    # The top-level cell enters the same stack as any sub-cell and goes
    # through the same 4-step lifecycle: cheap certificates → tangency →
    # subdivision. If it's loop-free at top level, the first iteration
    # traces it and the loop exits.
    from collections import deque
    queue = deque([top_cell])
    all_fragments: list[_Fragment] = []
    all_points = []
    all_singularities: list[SSXSingularity] = []
    if promoted_fiber_fragment is not None:
        budget.append_output(
            all_fragments, promoted_fiber_fragment, "fragment")
    # Crossing-less Phi seeding is complete for the whole cell it slices;
    # descendant reconfirmations must not repay the 4-plane search. Keep
    # ancestor boxes rather than keying on emitted points: a 1-D tangent
    # loop deliberately emits no isolated tangent_point but still needs one
    # Phi seed pass (the case13 dedup fix must not suppress that path).
    phi_seeded_boxes: list[tuple] = []
    phi_seed_attempts: list[NDArray[np.float64]] = []
    # NOTE no per-cell C3 gate here (ledger L8): Theorem 3 is a PER-BOX
    # injectivity certificate — a C3 whose two preimages lie in DIFFERENT
    # traced cells passes every per-cell check (each branch-carrying cell
    # certifies its own image injective, truthfully), so "some traced cell
    # failed the certificate" is NOT a sound trigger for the post-trace
    # c3_pass. The pass now runs unconditionally whenever a collision is
    # possible at all (see the c3_pass call site below).
    from mmcore.numeric.intersection.ssx._ssx5_singular import (
        hull_excludes_zero, psi_vector_net,
    )

    def _available_queue_slots() -> int:
        # Queued cells are already allocated future pop work. Reserving
        # their slots prevents a producer-heavy tree from materializing
        # O(fanout*max_cells) cells while only popped cells are charged.
        return max(0, budget.remaining_cells - len(queue))

    def _enqueue_probe_children(parent) -> bool:
        if _available_queue_slots() < 4:
            budget.mark_exhausted()
            return False
        queue.extend(_probe_children(parent))
        return True

    while queue:
        if not budget.charge_cells(1, "ssx"):
            break
        cell = queue.popleft()

        # Cheap AABB pruning first: if control-point bounding boxes don't
        # overlap, there is no intersection in this cell.
        if _aabb_disjoint(cell.g1.surface, cell.g2.surface, atol):
            continue


        # GJK separability: tighter than AABB, much cheaper than the sq-dist
        # net or Gauss separability. Test the convex hulls of the two control
        # nets — if they're separated, the surfaces don't intersect.
        if _trust_gjk(cell.g1) and _trust_gjk(cell.g2):
            P1_pts = (cell.g1.surface[..., :-1] / cell.g1.surface[..., -1:]).reshape(-1, 3)
            P2_pts = (cell.g2.surface[..., :-1] / cell.g2.surface[..., -1:]).reshape(-1, 3)
            if not gjk(P1_pts, P2_pts, atol, 15):
                continue

        # Sq-dist net pruning using the PROPAGATED F_sq (built once at top,
        # split alongside TΨᵢ at every subdivision — never reconstructed).
        if cell.F_sq is not None:
            if _check_min_of_net(cell.F_sq, atol, cell.w_scale):
                continue
            if _check_lipschitz(cell.F_sq, atol, cell.w_scale):
                continue

        # Ledger L4 probe descent: a probe-only cell exists solely to find
        # the Δ-touch its loop-free ancestor's center witness missed (the
        # off-lattice touch pinned at a box corner, where the center GN
        # lands in the touch/valley trap and diverges). Lifecycle: the
        # sound prunes above → hull gate (some T hull excluding 0 proves
        # no Δ-root — drop) → center witness (success emits and ends this
        # probe line; the emission dedup absorbs re-confirmations from
        # sibling probes) → lean 2x2 net-split descent, bounded by the
        # same 4·unify_tol size gate as the crossing-less arm. Probes
        # never trace and never CSX — the ancestor already traced this
        # geometry, which is what keeps the descent duplication-free.
        if cell.probe_only:
            if (cell.T1 is None or any(
                    hull_excludes_zero(T)
                    for T in (cell.T1, cell.T2, cell.T3, cell.T4))):
                continue
            # Component-wise Ψ hull exclusion (sound: a Δ-root needs Ψ = 0
            # too). The T hulls alone cannot end a probe line along the
            # TRAP SHEET — the critical set where all four TΨ vanish but
            # Ψ != 0 (the touch-plus-loop valley floor: a 1-dim ring with
            # |Ψ_z| = eps²/4, INSIDE the atol tolerance band, so the
            # F_sq-vs-atol prune above keeps it too) — and the descent
            # walked that ring to the size gate (measured +3.7 s on the
            # off-lattice repro). A sign-carrying Ψ component excludes it
            # as soon as the cell-local range drops under the floor value
            # (1-2 probe levels here); touch cells keep Ψ = 0 and survive.
            # Margin at COORDINATE scale, not per-component net scale
            # (`hull_excludes_zero`'s max|c| convention is WRONG here —
            # the L1 drift race): the split drift in a mathematically-zero
            # corner coefficient originates from the O(coordinate)-scale
            # surface nets, while a touch-hugging component's own range
            # shrinks without bound (measured on the touch cell at d=8:
            # G_z hull [-3.9e-4, -9.8e-17] — true max is 0 AT the corner,
            # drift 1e-16, per-component margin 1.1e-17 wrongly excluded
            # and the touch was lost again).
            S1h_p, S2h_p = cell.g1.surface, cell.g2.surface
            G = psi_vector_net(S1h_p, S2h_p)
            from mmcore.numeric.intersection.ssx._ssx5_singular import (
                _HULL_MARGIN_K_EPS as _gK)
            gm = _gK * (float(np.abs(S1h_p[..., :-1]).max())
                        * float(np.abs(S2h_p[..., -1]).max())
                        + float(np.abs(S2h_p[..., :-1]).max())
                        * float(np.abs(S1h_p[..., -1]).max()))
            if any(float(G[..., k].min()) > gm or float(G[..., k].max()) < -gm
                   for k in range(3)):
                continue
            ok, roots = _emit_tangent_roots(cell, atol, unify_tol,
                                            all_singularities,
                                            enumerate_all=False,
                                            overlap_boxes=overlap_boxes)
            if not (ok and roots) and not np.all(
                    np.array([hi - lo for (lo, hi) in cell.box])
                    <= 4.0 * unify_tol):
                _enqueue_probe_children(cell)
            continue

        # Loop-absence on this sub-cell — TΨᵢ monotonicity (cheap) tried first,
        # Gauss map separability as fallback (design §6, §10 principle 8).

        if _check_loop_free(cell.g1, cell.g2,
                            cell.T1, cell.T2, cell.T3, cell.T4):
            # C2 touch ON the subdivision lattice: when the tangent point
            # coincides with cut values (the saddle's X at s=t=1/2 under
            # midpoint cuts), the children are loop-free via a NON-STRICT
            # monotone T net that attains 0 exactly at the touch corner,
            # so no cell holding the touch ever reaches the tangency arms
            # below — the touch surfaces only as a boundary crossing of
            # loop-free cells and would go unreported. Emit the Δ-witness
            # here too, gated by the necessary condition for any TΨ = 0
            # in the cell: ALL FOUR T-net hulls contain 0 (a strictly
            # one-signed net excludes tangency). Regular transversal
            # cells almost always carry a strictly one-signed net, so the
            # gate costs 8 min/max on already-carried nets. Near-tangent
            # (but touch-free) geometries DO pass the hull gate (case 5:
            # 13 cells, case 11's near-tangent loop: 5), so the witness
            # runs center-GN only (enumerate_all=False, ~1 ms/call —
            # full solve_zero_dim enumeration on those cells measured
            # ~2 s per case, a 1.3–1.6x coverage-case regression). The
            # center start suffices here: a lattice touch is pinned to a
            # corner of a small post-subdivision cell (guided cuts pass
            # exactly through the discovered touch crossing), unlike the
            # crossing-less arm whose one large cell can hold several
            # distant touches and keeps the full enumeration.
            # "Contains 0" is margin-consistent with the L1 hull
            # convention: a hull counts as containing 0 unless it CLEARS
            # zero by the roundoff margin (`not hull_excludes_zero`) —
            # a lattice touch's mathematically-zero T coefficient drifts
            # to ~eps/8 after the guided cut through it, and the strict
            # `min <= 0` gate then never fired. The margin makes the
            # probe fire MORE often — the safe direction.
            tangent_gate = (cell.T1 is not None and not any(
                hull_excludes_zero(T)
                for T in (cell.T1, cell.T2, cell.T3, cell.T4)))
            if tangent_gate:
                ok, roots = _emit_tangent_roots(cell, atol, unify_tol,
                                                all_singularities,
                                                enumerate_all=False,
                                                overlap_boxes=overlap_boxes,
                                                defer_inconclusive=bool(
                                                    cell.crossings))
                # Ledger L4: the hull gate says a touch is POSSIBLE but
                # the center-only witness failed (GN diverged or landed
                # outside — e.g. an off-lattice touch sitting at this
                # cell's box corner, where the center start dies in the
                # touch/valley trap: the (0.3,0.3) touch-plus-loop lost
                # its touch through exactly this arm's old unconditional
                # `continue`). Push a lean PROBE descent (probe_only
                # cells: hull gate + center witness + net-split, no
                # tracing/CSX — see the probe arm above) to hunt the
                # missed root, bounded by the same 4·unify_tol size gate.
                # The cell itself still traces below exactly as before —
                # probes never trace, so nothing is double-collected, and
                # the traced geometry is bit-identical to the pre-L4
                # baseline. Guided-cut subdivision instead of probes
                # measured +1.2 s on case 5 (20 near-tangent witness
                # failures x ~60 ms of cut-face CSX) and degraded the
                # off-lattice ring's arc coverage (descendant traces
                # replaced the parent's) — the lean descent costs ~ms and
                # keeps the traced geometry untouched.
                if not (ok and roots) and not np.all(
                        np.array([hi - lo for (lo, hi) in cell.box])
                        <= 4.0 * unify_tol):
                    _enqueue_probe_children(cell)
            if cell.crossings:
                # A loop-free cell whose registrations all collapse onto
                # one isolated tangent witness has no certified through-arc.
                # In this configuration the ordinary tracer's displaced-seed
                # recovery can walk the sub-atol tolerance valley and invent
                # an unregistered partner endpoint (regular isolated touch:
                # two complete-looking 20*atol "transversal" branches).
                # Require a crossing distinct from the tangent cluster before
                # regular tracing.  Both the parameter and xyz guards follow
                # the module's matching ladder; failure stays explicitly
                # partial until a second-order isolation certificate exists.
                tangent_cluster_only = False
                if tangent_gate and ok and roots:
                    root_clusters = []
                    for root in roots:
                        root = np.asarray(root, dtype=np.float64)
                        root_clusters.append((
                            _local_to_global(root, cell.box),
                            eval_surface(
                                cell.g1.surface, root[0], root[1],
                                rational=True),
                        ))
                    tangent_cluster_only = all(
                        any(
                            np.all(np.abs(np.asarray(c.stuv) - rg)
                                   <= unify_tol)
                            and float(np.linalg.norm(
                                np.asarray(c.xyz) - rx)) <= 2.0 * atol
                            for rg, rx in root_clusters)
                        for c in cell.crossings)

                if tangent_cluster_only:
                    budget.mark_incomplete()
                    fr, pt = [], []
                else:
                    fr, pt = _trace_cell_by_registrations(
                        cell, atol, h_max=h_max)
                if tangent_gate:
                    # Ledger L5: a tangent CURVE traces fine through this
                    # loop-free path (non-strict monotone T-hulls) but
                    # shipped kind='transversal'. Tag by MEASUREMENT
                    # (`_fragment_on_tangent_locus`: normal alignment,
                    # escalating to the Δ-snap for valley-wandered
                    # samples) => tangential fragment (propagates to the
                    # branch kind via assembly's any-fragment rule, and
                    # the kind-keyed subsumption filter then eats the
                    # stray on-curve witnesses). Only fired-gate cells are
                    # measured — transversal fragments exit on their first
                    # failing sample.
                    for f in fr:
                        if (not f.tangential and len(f.stuv_path) >= 2
                                and _fragment_on_tangent_locus(cell, f,
                                                               atol)):
                            f.tangential = True

                # A high-multiplicity tangent curve can leave Delta' at
                # rank 2 everywhere, so the local rank-3 continuation test
                # is intentionally inconclusive.  A strict traced tangent
                # path through that same 4-D root is a stronger direct
                # certificate of local one-dimensionality.  Resolve only
                # against the SAME-LENGTH-SCALE stuv and xyz location;
                # otherwise preserve the earlier conservative partial flag
                # (isolated high-order touch beside an unrelated branch).
                deferred = getattr(cell, "_deferred_delta_roots", [])
                if deferred:
                    scale4 = np.maximum(
                        np.asarray(unify_tol, dtype=np.float64), 1e-12)
                    for root in deferred:
                        root_g = _local_to_global(root, cell.box)
                        root_xyz = eval_surface(
                            cell.g1.surface, root[0], root[1],
                            rational=True)
                        covered = False
                        for f in fr:
                            if not f.tangential or len(f.stuv_path) < 2:
                                continue
                            if (_dist_point_polyline(
                                    root_xyz,
                                    np.asarray(f.xyz_path,
                                               dtype=np.float64))
                                    > 2.0 * atol):
                                continue
                            scaled_poly = (np.asarray(
                                f.stuv_path, dtype=np.float64)
                                / scale4[None, :])
                            if (_dist_point_polyline_nd(
                                    root_g / scale4, scaled_poly) <= 2.0):
                                covered = True
                                break
                        if not covered:
                            budget.mark_incomplete()
                            break
                budget.extend_output(all_fragments, fr, "fragment")
                budget.extend_output(all_points, pt, "point")
            continue

        # §6 step 3: Krawczyk-based tangency certification. If TΨ = 0 has a
        # simultaneous root in this cell, the intersection is tangential (C₂)
        # and must be traced via the regulated Φ system (design §1.4, §8),
        # NOT by further subdivision — deflation makes the Φ-curve regular
        # where Ψ is rank-deficient.
        #
        # Geometric tangency pre-check: at a SSX-tangent crossing the two
        # surface normals are parallel, so sin(angle(N1, N2)) ≈ 0. At a
        # transversal crossing they are not parallel, so sin(angle) > 0.
        # We previously used the algebraic |TΨᵢ| pre-check, but its slope is
        # O(10⁴) near the tangent locus — a Newton-precision crossing
        # (~1e-8 off the locus) gives |TΨ| ~ 1e-4, far above any usable
        # threshold, so genuine tangents were misclassified as transversal.
        # The dot/cross-product test scales linearly (slope ~1) with the
        # offset, so a 1e-3 threshold (≈0.06°) gives ~10⁴× headroom.
        _cell_fibers = (boundary_parameter_fibers
                        if cell.depth == 0 else [])
        _has_boundary_seeds = bool(cell.crossings or _cell_fibers)
        is_clearly_transversal = False
        if not _has_boundary_seeds:
            # An isolated tangency or interior tangent loop lives in exactly
            # this kind of cell (no boundary crossings). Whether tangency is
            # even possible is already known for free: _check_monotonicity
            # failed (we are past the loop-free gate), i.e. all four T-Psi
            # hulls straddle zero. Run the tangency check instead of
            # assuming transversality — assuming it silently deleted
            # isolated tangent points (paper Fig. 24/25 class).
            pass
        else:
            for c in cell.crossings:
                s, t, u, v = c.stuv  # global stuv on the original surfaces
                # Near a collapsed apex edge the normal direction is
                # ill-conditioned and can approach any limiting direction.
                # Such a crossing cannot CERTIFY transversality.  Case 14's
                # true tangent generator ended at two apex fibers; CSX placed
                # their representatives ~1.7e-5 inside the face, and the
                # arbitrary near-apex normals falsely sent the top cell down
                # the transversal subdivision path.  Use the same parametric
                # matching ladder as crossing unification and let the exact
                # rational Delta witness decide the cell instead.
                if (_on_collapsed_boundary_fiber(
                        S1_h_top, s, t, rational=True,
                        param_tol=float(max(unify_tol[0], unify_tol[1])))
                        or _on_collapsed_boundary_fiber(
                            S2_h_top, u, v, rational=True,
                            param_tol=float(max(unify_tol[2], unify_tol[3])))):
                    continue
                _, du1, dv1 = eval_surface_d1(S1, s, t, rational=rational)
                _, du2, dv2 = eval_surface_d1(S2, u, v, rational=rational)
                N1 = np.cross(du1, dv1)
                N2 = np.cross(du2, dv2)
                n1m = float(np.linalg.norm(N1))
                n2m = float(np.linalg.norm(N2))
                if n1m < 1e-30 or n2m < 1e-30:
                    # Degenerate parametrization at this point — can't decide,
                    # skip and let the next crossing or _check_tangency decide.
                    continue
                sin_ang = float(np.linalg.norm(np.cross(N1, N2))) / (n1m * n2m)
                if sin_ang > 1e-3:  # ≈ 0.06° off-parallel → clearly transversal
                    is_clearly_transversal = True
                    break

        if is_clearly_transversal:
            tangency = False
        else:
            local_box = ((0.0, 1.0),) * 4
            if rational:
                _tan_S1, _tan_S2, _tan_rat = (
                    cell.g1.surface, cell.g2.surface, True)
            else:
                _tan_S1 = cell.g1.surface[..., :-1]
                _tan_S2 = cell.g2.surface[..., :-1]
                _tan_rat = False
            tangency = _check_tangency(
                cell.T1, cell.T2, cell.T3, cell.T4,
                _tan_S1, _tan_S2, local_box, rational=_tan_rat,
                atol=atol,
            )
        if tangency is True and not _has_boundary_seeds:
            # Isolated tangent point (or tangent feature with no boundary
            # contact). The Gauss-Newton witness from _check_tangency is the
            # point — recompute it here tighter to get coordinates (cheap:
            # often this fires at the TOP cell, before any subdivision) and
            # emit typed singularities. The witness enumerates every distinct
            # Δ-root in the cell (solve_zero_dim hull exclusion): one big
            # cell can hold several isolated tangencies, and `continue`
            # would silently drop the ones the center start missed.
            # Emission + dedup live in _emit_tangent_roots (shared with the
            # loop-free and crossing-bearing sites). enumerate_all=True is
            # spelled out because THIS site depends on full enumeration
            # (multi-touch cells); `roots` feeds Task 5's Φ∩L seeding
            # (_choose_phi_equations takes roots[0]).
            ok, roots = _emit_tangent_roots(cell, atol, unify_tol,
                                            all_singularities,
                                            enumerate_all=True,
                                            overlap_boxes=overlap_boxes)
            _root_globals = [
                _local_to_global(np.asarray(r, dtype=np.float64), cell.box)
                for r in roots]
            _seed_cell = top_cell if cell.depth > 0 else cell
            _seed_roots = (_root_globals if cell.depth > 0 else roots)
            _seed_anchor = (np.asarray(_root_globals[0], dtype=np.float64)
                            if _root_globals else np.array(
                                [0.5 * (lo + hi) for lo, hi in cell.box]))
            _phi_already_seeded = any(all(
                parent[ax][0] <= cell.box[ax][0]
                and cell.box[ax][1] <= parent[ax][1]
                for ax in range(4)) for parent in phi_seeded_boxes)
            _phi_already_attempted = any(
                np.all(np.abs(_seed_anchor - prior) <= unify_tol)
                for prior in phi_seed_attempts)
            if (ok and not _phi_already_seeded
                    and not _phi_already_attempted and not budget.exhausted):
                # Paper §5.3.2: slice the regulated Φ curve with the four
                # deterministic axis mid-planes to seed loops around the
                # tangency that have no boundary crossings, then march each
                # refined seed around its closed loop (Ψ or Φ backend, see
                # _phi_slice_loop_fragments). Loops the subdivision below
                # ALSO finds are absorbed by _drop_duplicate_fragments
                # containment — the seeding can add geometry, never
                # duplicate it; loops inside the size-gated blind window
                # (cells that `continue` below) are found ONLY here.
                _phi_fragments = _phi_slice_loop_fragments(
                    _seed_cell, _seed_roots, atol, h_max, all_singularities)
                phi_seed_attempts.append(_seed_anchor)
                budget.extend_output(
                    all_fragments, _phi_fragments, "fragment")
                # An empty coarse-cell slice is inconclusive: a small loop
                # can miss all four mid-planes until a descendant tightens
                # the box. Cache only a productive pass; shared solver
                # budgets bound unsuccessful retries.
                if _phi_fragments:
                    phi_seeded_boxes.append(_seed_cell.box)
            # Emitting the tangency does NOT resolve the cell: the same
            # crossing-less cell can hold coexisting transversal features —
            # z = q(q-1/2) (Mexican hat) has the touch at the center AND a
            # transversal ring at q = 1/2, and an unconditional `continue`
            # here silently deleted the ring (the ring is transversal and
            # NOT on Φ — even the Φ∩L seeding above only reaches it through
            # the full-Ψ refinement of nearby Φ points, which is
            # opportunistic, not certified). Stop only when the cell is at
            # tolerance scale (all four GLOBAL spans within 4·unify_tol);
            # otherwise fall through to the subdivision path like any other
            # uncertified cell — descendants that re-confirm the same
            # tangency are absorbed by the emission dedup above.
            # A failed witness (ok=False: GN non-convergence or the
            # numerical-failure path) must not vanish either — fall through
            # regardless of size so the cell is never dropped with neither
            # emission nor subdivision.
            spans = np.array([hi - lo for (lo, hi) in cell.box])
            if ok and np.all(spans <= 4.0 * unify_tol):
                continue

        if tangency is True and _has_boundary_seeds:
            # C2 with transversal branches THROUGH the touch (saddle
            # X-crossing): a cell holding such a tangent point is crossing-
            # BEARING (the arms pierce its boundary), so the crossing-less
            # arm above never sees it. Emit the center Δ-witness here too,
            # through the same dedup. enumerate_all=False — this arm is the
            # one that fires on tangent CURVES (the legacy crossed-saddles
            # case traces its curve right below via _deflate_tangent_cell),
            # where Δ's zero set is 1-dimensional and full enumeration
            # burned the whole solve_zero_dim budget to emit ptol-spaced
            # curve samples (measured: 69 tangent_points, 2.43 s vs 0.15 s
            # for the case — a >2x slowdown gate per the Task 4 plan). The
            # center witness costs ~ms.  Do NOT emit it yet: on a tangent
            # curve this is one arbitrary curve sample per descendant cell,
            # and relying on a later globally-complete branch to subsume the
            # samples revived a 29-point flood whenever tracing was partial.
            # `_emit_offcurve_tangent_roots` below is the single emission
            # owner for crossing-bearing cells: it suppresses roots in tubes
            # measured tangent along their whole path, but enumerates and
            # emits an isolated saddle/off-curve touch.
            _ok_curve, _curve_roots, _curve_fn, _curve_exhausted = (
                _tangency_witness(cell, atol, enumerate_all=False))
            if _curve_exhausted and cell.work_budget is not None:
                cell.work_budget.mark_incomplete()
            # Boundary roots on a collapsed apex edge carry an arbitrary
            # free parameter. Canonicalize it to the interior Delta
            # witness before Phi tracing; otherwise identical physical
            # endpoints can be paired as a sub-atol micro-fragment while
            # the actual generator is never marched (case 14 was
            # nondeterministic between those two outcomes).
            _needs_fiber_canonicalization = bool(_cell_fibers) or any(
                _on_collapsed_boundary_fiber(
                    S1_h_top, c.stuv[0], c.stuv[1], rational=True,
                    param_tol=float(max(unify_tol[0], unify_tol[1])))
                or _on_collapsed_boundary_fiber(
                    S2_h_top, c.stuv[2], c.stuv[3], rational=True,
                    param_tol=float(max(unify_tol[2], unify_tol[3])))
                for c in cell.crossings)
            if not _needs_fiber_canonicalization:
                # Preserve the established object/registration topology on
                # ordinary tangent cells; only apex fibers need rewriting.
                trace_crossings = list(cell.crossings)
            else:
                _anchor_local = (
                    np.asarray(_curve_roots[0], dtype=np.float64)
                    if _curve_roots else np.full(4, 0.5))
                _anchor_global = _local_to_global(_anchor_local, cell.box)
                trace_crossings = []
                for c in list(cell.crossings) + list(_cell_fibers):
                    stuv_c = np.asarray(c.stuv, dtype=np.float64).copy()
                    stuv_c[:2] = _canonicalize_collapsed_fiber_params(
                        S1_h_top, stuv_c[:2], _anchor_global[:2], rational=True,
                        param_tol=float(max(unify_tol[0], unify_tol[1])))
                    stuv_c[2:] = _canonicalize_collapsed_fiber_params(
                        S2_h_top, stuv_c[2:], _anchor_global[2:], rational=True,
                        param_tol=float(max(unify_tol[2], unify_tol[3])))
                    canonical = BoundaryPoint(
                        stuv=stuv_c, xyz=np.asarray(c.xyz, dtype=np.float64),
                        face=c.face, tangent_raw=c.tangent_raw)
                    if any(np.all(np.abs(stuv_c - q.stuv) <= unify_tol)
                           and float(np.linalg.norm(canonical.xyz - q.xyz))
                           <= 2.0 * atol for q in trace_crossings):
                        continue
                    trace_crossings.append(canonical)
            # Convert crossings to the cell's local stuv for the Φ tracer.
            crossings_local = [
                BoundaryPoint(
                    stuv=_global_to_local(c.stuv, cell.box),
                    xyz=c.xyz, face=c.face, tangent_raw=c.tangent_raw,
                )
                for c in trace_crossings
            ]
            fr_local, pt_local = _deflate_tangent_cell(
                cell.g1.surface, cell.g2.surface,
                cell.T1, cell.T2, cell.T3, cell.T4,
                local_box, crossings_local, atol,
                rational=True, originals=trace_crossings, cell=cell,
                h_max=h_max,
            )
            for f in fr_local:
                stuv_glob = np.empty_like(f.stuv_path)
                for k in range(len(f.stuv_path)):
                    stuv_glob[k] = _local_to_global(f.stuv_path[k], cell.box)
                _fragment = _Fragment(
                    start_point=f.start_point, end_point=f.end_point,
                    stuv_path=stuv_glob, xyz_path=f.xyz_path,
                    tangential=f.tangential,
                )
                budget.append_output(all_fragments, _fragment, "fragment")
            # pt_local's SSXPoint.stuv is already global — we passed
            # `originals` so _deflate_tangent_cell copied from them.
            budget.extend_output(all_points, pt_local, "point")
            # Tracing the Φ curve between this cell's crossings does NOT by
            # itself resolve the cell: the deflation only reaches features
            # ON Φ through the boundary crossings, and the center witness
            # converges into the CURVE's basin — a coexisting ISOLATED
            # touch in the same cell (z = (2t-1)^2*((s-0.7)^2+(t-0.2)^2):
            # tangent line at t=0.5 PLUS a touch at (0.7,0.2)) is off every
            # traced fragment and the plain `continue` deleted it with NO
            # descendants ever seeing it (this cell was the only holder —
            # there is no "some other cell covers it" on this path).
            # Subdividing instead (the crossing-less arm's e1db506
            # treatment) is correct but measured 1349x slower on the legacy
            # crossed-saddles case (0.15 s -> 200 s): cells along a
            # 1-dimensional tangent curve can never be certified away, so
            # the size gate forces a full dyadic descent along the curve's
            # length. Enumerate the cell's REMAINING Δ-roots here instead:
            # hull-exclusion subdivision with the Newton attempts SKIPPED
            # inside the traced fragments' tube (those roots are curve
            # samples the subsumption filter would delete anyway),
            # far-from-tube boxes explored FIRST, and only Newton attempts
            # charged against the budget (skip-exempt charging — under
            # per-pop charging the flood's excluded siblings starved the
            # budget and touches at 5-15*atol from the curve fell into a
            # blind band).
            # KNOWN LIMIT: this recovers only Δ-roots. A coexisting
            # TRANSVERSAL loop (not on Δ) with no boundary crossings in
            # this cell is still lost — the `continue` below skips
            # subdivision, and the Φ∩L loop seeding runs only on the
            # crossing-LESS arm. Accepted (the subdivision alternative is
            # the 1349x path above).
            _emit_offcurve_tangent_roots(cell, fr_local, atol, unify_tol,
                                         all_singularities,
                                         overlap_boxes=overlap_boxes)
            continue

        if cell.depth >= max_depth:
            for c in cell.crossings:
                budget.append_output(
                    all_points, SSXPoint(stuv=c.stuv, xyz=c.xyz), "point")
            # Reaching the caller's depth ceiling after all sound
            # certificates above failed leaves this cell unresolved.  The
            # boundary samples are useful partial output, not a proof that
            # no interior component exists (case 7 at max_depth=0).
            budget.mark_incomplete()
            continue

        # --- Dual-surface subdivision ---
        # Both surfaces are split at each step. Productive crossings provide
        # per-surface split values; if a surface has no guided split, it gets
        # a midpoint cut on its longest-span axis.

        s1_axis, s1_cuts, s2_axis, s2_cuts = _compute_split_plan(
            cell.new_crossings, cell.box, cut_tol=dedup_tol, max_cuts=8)
        #print(s1_axis, s1_cuts, s2_axis, s2_cuts,cell.box,[nc.xyz.tolist() for nc in cell.new_crossings])
        # Midpoint fallback per surface when no guided cuts
        if s1_axis is None:
            s1_span_s = cell.box[0][1] - cell.box[0][0]
            s1_span_t = cell.box[1][1] - cell.box[1][0]
            s1_axis = 0 if s1_span_s >= s1_span_t else 1
            s1_cuts = [0.5 * (cell.box[s1_axis][0] + cell.box[s1_axis][1])]
        if s2_axis is None:
            s2_span_u = cell.box[2][1] - cell.box[2][0]
            s2_span_v = cell.box[3][1] - cell.box[3][0]
            s2_axis = 2 if s2_span_u >= s2_span_v else 3
            s2_cuts = [0.5 * (cell.box[s2_axis][0] + cell.box[s2_axis][1])]
        # Guard the Cartesian product BEFORE allocating split nets/grids.
        # Productive crossings are only guides; midpoint subdivision is a
        # sound fallback when noisy guides would consume the remaining call
        # budget in one cell.  If even the 2x2 fallback cannot fit, return
        # the partial result instead of allocating work that cannot run.
        projected_children = (len(s1_cuts) + 1) * (len(s2_cuts) + 1)
        if projected_children > _available_queue_slots():
            if _available_queue_slots() < 4:
                budget.mark_exhausted()
                break
            s1_cuts = [0.5 * (cell.box[s1_axis][0] + cell.box[s1_axis][1])]
            s2_cuts = [0.5 * (cell.box[s2_axis][0] + cell.box[s2_axis][1])]
        #print(s1_axis, s1_cuts, s2_axis, s2_cuts, cell.box,[nc.xyz.tolist() for nc in cell.new_crossings])
        # Split S1 (Gauss map) along s1_axis
        g1_pieces = _split_surface_multi(cell.g1, s1_axis, s1_cuts, cell.box)
        # Split S2 (Gauss map) along s2_axis
        g2_pieces = _split_surface_multi(cell.g2, s2_axis, s2_cuts, cell.box)

        # Split TΨᵢ along BOTH axes sequentially
        T_list = [cell.T1, cell.T2, cell.T3, cell.T4]
        T_after_s1 = [_split_tensor_multi(T, s1_axis, s1_cuts, cell.box) for T in T_list]
        T_pieces = []
        F_sq_after_s1 = (_split_tensor_multi(cell.F_sq, s1_axis, s1_cuts, cell.box)
                         if cell.F_sq is not None else None)
        F_sq_pieces = []
        for i1 in range(len(g1_pieces)):
            row = []
            for T_idx in range(4):
                T_s1_piece = T_after_s1[T_idx][i1]
                sub_box = list(cell.box)
                s1_lo = cell.box[s1_axis][0] if i1 == 0 else s1_cuts[i1 - 1]
                s1_hi = s1_cuts[i1] if i1 < len(s1_cuts) else cell.box[s1_axis][1]
                sub_box[s1_axis] = (s1_lo, s1_hi)
                pieces_s2 = _split_tensor_multi(T_s1_piece, s2_axis, s2_cuts, tuple(sub_box))
                row.append(pieces_s2)
            T_pieces.append(row)
            # F_sq propagation (single 4D tensor, same axis convention)
            if F_sq_after_s1 is not None:
                sub_box = list(cell.box)
                s1_lo = cell.box[s1_axis][0] if i1 == 0 else s1_cuts[i1 - 1]
                s1_hi = s1_cuts[i1] if i1 < len(s1_cuts) else cell.box[s1_axis][1]
                sub_box[s1_axis] = (s1_lo, s1_hi)
                F_sq_pieces.append(
                    _split_tensor_multi(F_sq_after_s1[i1], s2_axis, s2_cuts, tuple(sub_box)))
            else:
                F_sq_pieces.append([None] * (len(s2_cuts) + 1))

        # --- CSX: cut_line vs each piece of the opposite surface ---
        # Split first, then CSX. Each cut line is intersected with each
        # piece of the opposite surface separately, so crossings are found
        # on the refined geometry and map deterministically to sub-cells.
        s1_other = 1 - s1_axis if s1_axis < 2 else 1 - (s1_axis - 2)
        s1_local_axis = s1_axis if s1_axis < 2 else s1_axis - 2
        s2_other_global = ({2: 3, 3: 2})[s2_axis]
        s2_local_axis = s2_axis - 2

        # Per-sub-cell new crossings: new_cx_grid[i1][i2] = list
        n1 = len(g1_pieces)
        n2 = len(g2_pieces)
        new_cx_grid = [[[] for _ in range(n2)] for _ in range(n1)]

        # a/b: CSX(cut_line_s1, S2_piece) for each S1 cut × each S2 piece
        for cut_idx, cv in enumerate(s1_cuts):
            if budget.exhausted:
                break
            s1_lo_box, s1_hi_box = cell.box[s1_axis]
            cut_local_s1 = (cv - s1_lo_box) / (s1_hi_box - s1_lo_box)
            isoline_s1 = _extract_isoline(cell.g1.surface, s1_local_axis, cut_local_s1)

            for s2_idx in range(n2):
                if budget.exhausted:
                    break
                s2_piece_surf = g2_pieces[s2_idx].surface
                csx_r = _run_csx(
                    isoline_s1, s2_piece_surf, atol=atol, rational=True)
                #print(csx_r)
                csx_r['isolated'] = list(filter(
                    lambda x: not (((1 - x['t']) < 1e-6) or (x['t'] < 1e-6)), csx_r['isolated']))

                s2_lo = cell.box[s2_axis][0] if s2_idx == 0 else s2_cuts[s2_idx - 1]
                s2_hi = s2_cuts[s2_idx] if s2_idx < len(s2_cuts) else cell.box[s2_axis][1]
                s2_other_lo, s2_other_hi = cell.box[s2_other_global]

                for iso_pt in csx_r.get('isolated', []):
                    stuv = np.zeros(4, dtype=np.float64)
                    stuv[s1_axis] = cv
                    s1_other_lo, s1_other_hi = cell.box[s1_other]
                    stuv[s1_other] = s1_other_lo + float(iso_pt['t']) * (s1_other_hi - s1_other_lo)
                    if s2_local_axis == 0:
                        stuv[s2_axis] = s2_lo + float(iso_pt['u']) * (s2_hi - s2_lo)
                        stuv[s2_other_global] = s2_other_lo + float(iso_pt['v']) * (s2_other_hi - s2_other_lo)
                    else:
                        stuv[s2_other_global] = s2_other_lo + float(iso_pt['u']) * (s2_other_hi - s2_other_lo)
                        stuv[s2_axis] = s2_lo + float(iso_pt['v']) * (s2_hi - s2_lo)

                    xyz = np.asarray(iso_pt['point'], dtype=np.float64)
                    stuv_local = _global_to_local(stuv, cell.box)
                    tang, _, _ = _ssx_tangent_4d(
                        cell.g1.surface, cell.g2.surface,
                        stuv_local[0], stuv_local[1], stuv_local[2], stuv_local[3],
                        rational=True)
                    bp = BoundaryPoint(stuv=stuv, xyz=xyz, face=(s1_axis, -1), tangent_raw=tang)
                    new_cx_grid[cut_idx][s2_idx].append(bp)
                    new_cx_grid[cut_idx + 1][s2_idx].append(bp)

        if budget.exhausted:
            break

        # c/d: CSX(cut_line_s2, S1_piece) for each S2 cut × each S1 piece
        for cut_idx, cv in enumerate(s2_cuts):
            if budget.exhausted:
                break
            s2_lo_box, s2_hi_box = cell.box[s2_axis]
            cut_local_s2 = (cv - s2_lo_box) / (s2_hi_box - s2_lo_box)
            isoline_s2 = _extract_isoline(cell.g2.surface, s2_local_axis, cut_local_s2)

            for s1_idx in range(n1):
                if budget.exhausted:
                    break
                s1_piece_surf = g1_pieces[s1_idx].surface
                csx_r = _run_csx(
                    isoline_s2, s1_piece_surf, atol=atol, rational=True)
                csx_r['isolated'] = list(filter(
                    lambda x: not (((1 - x['t']) < 1e-6) or (x['t'] < 1e-6)), csx_r['isolated']))

                s1_lo = cell.box[s1_axis][0] if s1_idx == 0 else s1_cuts[s1_idx - 1]
                s1_hi = s1_cuts[s1_idx] if s1_idx < len(s1_cuts) else cell.box[s1_axis][1]
                s1_other_lo, s1_other_hi = cell.box[s1_other]

                for iso_pt in csx_r.get('isolated', []):
                    stuv = np.zeros(4, dtype=np.float64)
                    stuv[s2_axis] = cv
                    s2_other_lo2, s2_other_hi2 = cell.box[s2_other_global]
                    stuv[s2_other_global] = s2_other_lo2 + float(iso_pt['t']) * (s2_other_hi2 - s2_other_lo2)
                    if s1_local_axis == 0:
                        stuv[s1_axis] = s1_lo + float(iso_pt['u']) * (s1_hi - s1_lo)
                        stuv[s1_other] = s1_other_lo + float(iso_pt['v']) * (s1_other_hi - s1_other_lo)
                    else:
                        stuv[s1_other] = s1_other_lo + float(iso_pt['u']) * (s1_other_hi - s1_other_lo)
                        stuv[s1_axis] = s1_lo + float(iso_pt['v']) * (s1_hi - s1_lo)

                    xyz = np.asarray(iso_pt['point'], dtype=np.float64)
                    stuv_local = _global_to_local(stuv, cell.box)
                    tang, _, _ = _ssx_tangent_4d(
                        cell.g1.surface, cell.g2.surface,
                        stuv_local[0], stuv_local[1], stuv_local[2], stuv_local[3],
                        rational=True)
                    bp = BoundaryPoint(stuv=stuv, xyz=xyz, face=(s2_axis, -1), tangent_raw=tang)
                    new_cx_grid[s1_idx][cut_idx].append(bp)
                    new_cx_grid[s1_idx][cut_idx + 1].append(bp)

        if budget.exhausted:
            break

        # Cut-face CSX runs after the first fanout check and spends from the
        # same global allowance. Re-check immediately before allocating the
        # child cells so already-queued work plus this product still fits.
        if n1 * n2 > _available_queue_slots():
            budget.mark_exhausted()
            break

        # Build Cartesian product of S1 pieces × S2 pieces
        for i1 in range(n1):
            s1_lo = cell.box[s1_axis][0] if i1 == 0 else s1_cuts[i1 - 1]
            s1_hi = s1_cuts[i1] if i1 < len(s1_cuts) else cell.box[s1_axis][1]
            for i2 in range(n2):
                s2_lo = cell.box[s2_axis][0] if i2 == 0 else s2_cuts[i2 - 1]
                s2_hi = s2_cuts[i2] if i2 < len(s2_cuts) else cell.box[s2_axis][1]

                sub_box = list(cell.box)
                sub_box[s1_axis] = (s1_lo, s1_hi)
                sub_box[s2_axis] = (s2_lo, s2_hi)
                sub_box = tuple(sub_box)

                # Inherited crossings from parent — check ALL 4 axes
                sub_inherited = [c for c in cell.crossings
                                 if all(sub_box[ax][0]<= c.stuv[ax] <= sub_box[ax][1]
                                        for ax in range(4))]

                # New crossings: deterministic from per-piece CSX grid.
                # Dedup against inherited crossings and against each other —
                # duplicates here become duplicate march seeds, duplicate
                # fragments and broken id-chains downstream. The radius is
                # 1·ptol per axis AND atol in xyz: a parametric box alone is
                # not a metric ball (|Δstuv| ≤ 4·ptol admits xyz separations
                # of ~16·atol where derivatives are large), and deleting a
                # genuinely distinct crossing starves its cell of a march
                # seed (case 10 lost its v=1 endpoint segment exactly this
                # way — a crossing 4e-4 away in stuv but 5.3mm away in xyz
                # was merged into the domain-corner crossing).
                sub_new_raw = new_cx_grid[i1][i2]
                sub_new = []
                for nc in sub_new_raw:
                    if any(np.all(np.abs(nc.stuv - ec.stuv) <= dedup_tol)
                           and float(np.linalg.norm(nc.xyz - ec.xyz)) <= atol
                           for ec in sub_inherited):
                        continue
                    if any(np.all(np.abs(nc.stuv - dc.stuv) <= dedup_tol)
                           and float(np.linalg.norm(nc.xyz - dc.xyz)) <= atol
                           for dc in sub_new):
                        continue
                    sub_new.append(nc)

                sub_cx = sub_inherited + sub_new

                scell = _Cell(
                    g1=g1_pieces[i1], g2=g2_pieces[i2],
                    crossings=sub_cx, box=sub_box, depth=cell.depth + 1,
                    T1=T_pieces[i1][0][i2], T2=T_pieces[i1][1][i2],
                    T3=T_pieces[i1][2][i2], T4=T_pieces[i1][3][i2],
                    new_crossings=sub_new,
                    F_sq=F_sq_pieces[i1][i2] if F_sq_pieces[i1] else None,
                    w_scale=cell.w_scale,
                    work_budget=budget, csx_fn=_run_csx,
                )
                scell.partitions = _build_cell_partitions(scell)
                for c in sub_cx:
                    _classify_boundary_point(c, scell)
                queue.append(scell)

    # --- §9 assembly: chain fragments by shared BoundaryPoint endpoints ---
    # Pass the original surfaces so the assembly can march any small chain
    # gap that arises when a closed-loop intersection ends up with the same
    # geometric point produced as different BoundaryPoint instances in
    # cells from different parts of the subdivision tree.
    if rational:
        S1_for_close, S2_for_close, rational_close = S1, S2, True
    else:
        S1_for_close, S2_for_close, rational_close = S1, S2, False
    all_branches = _assemble_fragments(
        all_fragments,
        S1_full=S1_for_close, S2_full=S2_for_close,
        atol_full=atol, rational_full=rational_close,
        unify_tol=unify_tol, h_max=h_max,
        barrier_xyz=[g.xyz for g in all_singularities
                     if g.kind == "tangent_point"],
        work_budget=budget,
    )
    all_branches.extend(overlap_branches)

    # Everything below is post-assembly classification/filtering.  With no
    # remaining allowance, do not enter its endpoint-pair, point/branch, or
    # linkage scans.  Certified exact overlap claims are independently safe
    # to return; ordinary assembled fragments and unfiltered point-like
    # candidates are omitted from the explicit partial result.
    if not _assembly_spend(budget):
        return _result(branches=list(overlap_branches))

    # A certified boundary overlap owns every ordinary traced fragment whose
    # entire 4-D path is a subset of that same overlap.  Boundary endpoints
    # remain regular registrations, so the loop-free tracer can otherwise
    # re-march half of the edge and publish both ``transversal`` and
    # ``overlap`` copies.  Require BOTH the xyz and same-location stuv guards
    # at every vertex and segment midpoint; xyz proximity alone would delete
    # legitimate parameter-far sheets (ledger L3).  On postprocess-budget
    # denial, conservatively keep the fragment and leave the result partial.
    if overlap_branches and len(all_branches) > len(overlap_branches):
        _without_overlap_duplicates = []
        for branch in all_branches:
            if branch.kind == "overlap":
                _without_overlap_duplicates.append(branch)
                continue
            branch_stuv = np.asarray(branch.curve[0], dtype=np.float64)
            branch_xyz = np.asarray(branch.curve[1], dtype=np.float64)
            if len(branch_stuv) >= 2:
                mid_stuv = 0.5 * (branch_stuv[:-1] + branch_stuv[1:])
                mid_xyz = np.array([
                    eval_surface(
                        S1_h_top, x[0], x[1], rational=True)
                    for x in mid_stuv
                ])
                sample_stuv = np.vstack([branch_stuv, mid_stuv])
                sample_xyz = np.vstack([branch_xyz, mid_xyz])
            else:
                sample_stuv, sample_xyz = branch_stuv, branch_xyz

            contained = False
            denied = False
            for overlap in overlap_branches:
                overlap_stuv = np.asarray(
                    overlap.curve[0], dtype=np.float64)
                overlap_xyz = np.asarray(
                    overlap.curve[1], dtype=np.float64)
                inside = True
                for pstuv, pxyz in zip(sample_stuv, sample_xyz):
                    if not _assembly_spend(
                            budget, max(1, len(overlap_xyz) - 1)):
                        denied = True
                        inside = False
                        break
                    if not _point_on_branch_both_guards(
                            pxyz, pstuv, overlap_xyz, overlap_stuv,
                            atol, unify_tol, S1_h_top, S2_h_top):
                        inside = False
                        break
                if inside:
                    contained = True
                    break
                if denied:
                    break
            if not contained:
                _without_overlap_duplicates.append(branch)
        all_branches = _without_overlap_duplicates

    # --- Overlap-curve JUNCTION singularities ---
    # Two 1-dimensional overlap/tangential features meeting at a point are
    # a structural singularity of the SSI, and at such a junction the
    # surfaces are genuinely tangent (measured sin_ang = 0 exactly on both
    # the corner-sharing bilinear repro and the legacy overlaps corner).
    # The Δ-witness cannot report these: junction points sit inside the L6
    # overlap suppression boxes by construction (they are overlap segment
    # ENDPOINTS), which is correct for the witness junk that suppression
    # exists for — so junctions are emitted STRUCTURALLY here instead.
    # Rules: ≥2 DISTINCT overlap/tangential branches meeting within the
    # matching ladder (2·atol xyz), genuinely non-collinear directions
    # (a collinear meeting is one curve artificially split — no feature),
    # verified tangency at the junction (healthy, parallel normals).
    if len(all_branches) >= 2:
        def _sin_ang_at(s4):
            _, du1, dv1 = eval_surface_d1(S1_h_top, s4[0], s4[1], rational=True)
            _, du2, dv2 = eval_surface_d1(S2_h_top, s4[2], s4[3], rational=True)
            N1 = np.cross(du1, dv1)
            N2 = np.cross(du2, dv2)
            return (float(np.linalg.norm(np.cross(N1, N2)))
                    / max(float(np.linalg.norm(N1)) * float(np.linalg.norm(N2)),
                          1e-300))

        def _sin_ang_inward(b, at_start, d_iso):
            # sin_ang at the polyline point d_iso ALONG the branch from
            # the given end (stuv interpolated on the owning segment).
            xyz = np.asarray(b.curve[1], dtype=np.float64)
            stuv = np.asarray(b.curve[0], dtype=np.float64)
            if not at_start:
                xyz = xyz[::-1]
                stuv = stuv[::-1]
            walked = 0.0
            for k in range(len(xyz) - 1):
                seg = float(np.linalg.norm(xyz[k + 1] - xyz[k]))
                if walked + seg >= d_iso and seg > 1e-15:
                    lam = (d_iso - walked) / seg
                    return _sin_ang_at((1.0 - lam) * stuv[k] + lam * stuv[k + 1])
                walked += seg
            return _sin_ang_at(stuv[-1])

        _ends = []       # (branch_idx, xyz, stuv, out_dir, vertex, branch, at_start)
        _junction_scan_denied = False
        for bi, b in enumerate(all_branches):
            if b.kind not in ("overlap", "tangential"):
                continue
            xyz = np.asarray(b.curve[1], dtype=np.float64)
            stuv = np.asarray(b.curve[0], dtype=np.float64)
            if not _assembly_spend(budget, max(1, len(xyz))):
                _junction_scan_denied = True
                break
            if len(xyz) < 2:
                continue
            if float(np.linalg.norm(xyz[0] - xyz[-1])) <= 2.0 * atol:
                continue    # closed branch: its "ends" are a seam, not a junction
            for at_start in (True, False):
                p3 = xyz[0] if at_start else xyz[-1]
                p4 = stuv[0] if at_start else stuv[-1]
                v = (xyz[1] - xyz[0]) if at_start else (xyz[-2] - xyz[-1])
                n = float(np.linalg.norm(v))
                if n < 1e-15:
                    continue
                _ends.append((bi, p3, p4, v / n,
                              0 if at_start else len(xyz) - 1, b, at_start))
        for a in range(0 if _junction_scan_denied else len(_ends)):
            for c in range(a + 1, len(_ends)):
                if not _assembly_spend(budget):
                    _junction_scan_denied = True
                    break
                ba, pa, sa4, da, ka, bra, sta = _ends[a]
                bc, pc, sc4, dc, kc, brc, stc = _ends[c]
                if ba == bc:
                    continue
                if float(np.linalg.norm(pa - pc)) > 2.0 * atol:
                    continue
                # non-collinear meeting (collinear = artificial split)
                if float(np.linalg.norm(np.cross(da, dc))) < 0.1:
                    continue
                if _normals_degenerate_at(S1_h_top, S2_h_top, sa4):
                    continue
                if _sin_ang_at(sa4) > 1e-3:
                    continue
                # ISOLATION: a genuine junction tangency is 0-dimensional —
                # sin_ang must GROW away from the point along BOTH branches
                # (corner-sharing repro: 0 at the junction, ~0.25 at the
                # far edge ends). A junction inside a coplanar overlap
                # strip or on a 1-dim tangent curve has sin_ang ~ 0 in the
                # whole neighborhood (tangency is 1- or 2-dimensional
                # there) and is exactly the class the L6 suppression
                # exists for — skip it.
                _bra_xyz = np.asarray(bra.curve[1], dtype=np.float64)
                _brc_xyz = np.asarray(brc.curve[1], dtype=np.float64)
                if not _assembly_spend(
                        budget, max(1, len(_bra_xyz) + len(_brc_xyz))):
                    _junction_scan_denied = True
                    break
                _arc_a = float(np.linalg.norm(
                    np.diff(_bra_xyz, axis=0), axis=1).sum())
                _arc_c = float(np.linalg.norm(
                    np.diff(_brc_xyz, axis=0), axis=1).sum())
                d_iso_a = max(8.0 * atol, 0.05 * _arc_a)
                d_iso_c = max(8.0 * atol, 0.05 * _arc_c)
                if (_sin_ang_inward(bra, sta, d_iso_a) <= 1e-3
                        or _sin_ang_inward(brc, stc, d_iso_c) <= 1e-3):
                    continue
                if not any(g.kind == "tangent_point"
                           and np.all(np.abs(g.stuv - sa4) <= unify_tol)
                           and float(np.linalg.norm(np.asarray(g.xyz) - pa)) <= 2.0 * atol
                           for g in all_singularities):
                    budget.append_output(all_singularities, SSXSingularity(
                        kind="tangent_point", stuv=np.asarray(sa4, dtype=np.float64),
                        xyz=np.asarray(pa, dtype=np.float64),
                        branch_links=[(ba, ka), (bc, kc)]), "singularity")
            if _junction_scan_denied:
                break

    # A tangent_point ON a 1-dimensional tangential feature is not an
    # isolated C2 touch: overlap regions and traced tangent curves (branch
    # kind 'overlap'/'tangential') consist entirely of Δ-roots, so the
    # witness on a cell holding one converges to an arbitrary sample of the
    # curve (measured: the legacy overlaps case emitted its domain corner;
    # the crossed-saddles center witness lands on the tangent curve). The
    # richer feature already reports the contact — drop the redundant point
    # (Task 5 will type tangent curves explicitly). "ON a branch" is the
    # BOTH-GUARDS test (ledger L3, _point_on_branch_both_guards): xyz
    # point-to-segment ≤ 4·atol — same tolerance as the points-on-branch
    # filter below; the polyline is a chorded approximation, so points ON
    # the true curve sit up to the 2·atol sagitta off it (measured max
    # 1.9e-3 = 1.9·atol) — AND per-axis stuv ≤ 2·unify_tol against the
    # branch's stuv interpolated at the same segment location. xyz-only
    # subsumption deleted a certified touch on a DIFFERENT sheet 3·atol
    # from an overlap isoline (Δu = 0.8 in parameters, 640·atol z-wall
    # between). Genuine on-curve witnesses lie on the branch in stuv too
    # (they and the samples are ~ptol-accurate) and keep subsuming; a
    # branch segment whose stored stuv fails its own xyz self-check (the
    # known-corrupt legacy overlap bookkeeping) falls back to xyz-only —
    # see _point_on_branch_both_guards.
    # Tangent points coexisting with TRANSVERSAL branches (the saddle X)
    # are not affected — this tests tangential/overlap branches only.
    # Runs FIRST so the micro-branch and near-touch point filters below see
    # only genuinely isolated touches. Micro-scale tangential polylines
    # (arc ≤ 16·atol — the micro-branch filter's cap, computed identically)
    # are EXCLUDED from the subsuming set: a Ψ-valid Φ-micro-fragment AT a
    # touch would otherwise eat the typed point here, drop it from
    # _tangent_xyz, and thereby shield ITSELF from the micro-branch filter
    # (junk kept, singularity lost). With the floor, micro-branches can
    # never subsume a tangent_point, so the two filters are
    # order-independent for micro-branches.
    if (all_singularities
            and _assembly_spend(budget, max(1, len(all_branches)))):
        _one_dim_polys = []
        for b in all_branches:
            if b.kind not in ("overlap", "tangential"):
                continue
            _poly = np.asarray(b.curve[1], dtype=np.float64)
            if len(_poly) < 2:
                continue
            _arc = float(np.linalg.norm(np.diff(_poly, axis=0), axis=1).sum())
            if _arc > 16.0 * atol:
                _one_dim_polys.append(
                    (_poly, np.asarray(b.curve[0], dtype=np.float64)))
        _subsumption_cost = (
            len(all_singularities)
            * sum(max(1, len(poly_xyz) - 1)
                  for poly_xyz, _ in _one_dim_polys))
        if (_one_dim_polys
                and _assembly_spend(
                    budget, max(1, _subsumption_cost))):
            def _interior_subsumed(g):
                # JUNCTION EXCEPTION (corner-sharing bilinear repro): a
                # tangent_point at the ENDPOINT of an overlap/tangential
                # branch is a structural feature — two shared-edge overlap
                # curves meeting at a genuine surface-surface tangency
                # (sin_ang = 0 measured) — not an interior re-confirmation
                # of the 1-dim feature. Subsume only points whose nearest
                # branch location is in the polyline INTERIOR (farther
                # than 2·atol from both branch ends).
                gxyz = np.asarray(g.xyz, dtype=np.float64)
                gstuv = np.asarray(g.stuv, dtype=np.float64)
                for poly_xyz, poly_stuv in _one_dim_polys:
                    if not _point_on_branch_both_guards(
                            gxyz, gstuv, poly_xyz, poly_stuv, atol,
                            unify_tol, S1_h_top, S2_h_top):
                        continue
                    # The endpoint exception applies to genuinely OPEN
                    # ends only: a CLOSED branch's start/end is an
                    # assembly seam — interior in curve terms — and
                    # witness debris near the seam must still be
                    # subsumed (tangent-ring regression).
                    branch_open = (float(np.linalg.norm(
                        poly_xyz[0] - poly_xyz[-1])) > 2.0 * atol)
                    near_end = branch_open and (
                        float(np.linalg.norm(gxyz - poly_xyz[0])) <= 2.0 * atol
                        or float(np.linalg.norm(gxyz - poly_xyz[-1])) <= 2.0 * atol)
                    if not near_end:
                        return True
                return False

            all_singularities = [
                g for g in all_singularities
                if not (g.kind == "tangent_point" and _interior_subsumed(g))
            ]

    # Spurious micro-branches at emitted tangent points: subdividing around
    # a touch (the size-gated tangency arm above) re-exposes the old
    # pathology — CSX grazing-valley roots near the touch yield ~2·atol
    # micro-fragments, and crossing-bearing descendant tangent cells can add
    # Ψ-valid Φ-fragments there. Drop a branch only when EVERY polyline
    # vertex lies within 4·atol (xyz) of some emitted tangent point AND its
    # total xyz arc length is ≤ 16·atol — the length cap is a safety net so
    # nothing long can ever be eaten (the Mexican-hat ring's vertices sit
    # ~0.35 from the touch; saddle arms extend far beyond 4·atol). The
    # 16·atol constant is shared with the subsumption filter's floor above:
    # tangential/overlap polylines AT or BELOW it cannot subsume a
    # tangent_point there, so they are still deletable here.
    _tangent_xyz = [g.xyz for g in all_singularities if g.kind == "tangent_point"]
    if _tangent_xyz and all_branches:
        _tp = np.asarray(_tangent_xyz, dtype=np.float64)          # (K, 3)
        _kept_branches = []
        _micro_cost = len(_tp) * sum(
            max(1, len(np.asarray(b.curve[1]))) for b in all_branches)
        if not _assembly_spend(budget, max(1, _micro_cost)):
            # This pass is a soundness filter.  Exact overlap branches were
            # independently certified; omit unchecked ordinary fragments.
            _kept_branches = [
                b for b in all_branches if b.kind == "overlap"]
        else:
            for b in all_branches:
                xyz = np.asarray(b.curve[1], dtype=np.float64)
                if len(xyz):
                    d_min = np.linalg.norm(
                        xyz[:, None, :] - _tp[None, :, :], axis=2).min(axis=1)
                    if np.all(d_min <= 4.0 * atol):
                        arc = (float(np.linalg.norm(
                            np.diff(xyz, axis=0), axis=1).sum())
                               if len(xyz) > 1 else 0.0)
                        if arc <= 16.0 * atol:
                            continue
                _kept_branches.append(b)
        all_branches = _kept_branches

    # --- C1 pass (paper Fig. 5): parameterization cusps ON the SSI ---
    # Cusps are properties of a surface's parameterization (Sigma_i = 0),
    # not of the marching — the 4D curve is REGULAR through a C1 point
    # (T3=T4=0 but T1,T2 != 0 there), so the branches above already walk
    # through it; this pass only locates and types the cusp. Runs AFTER the
    # branch filters so branch_links index the FINAL branch list (the two
    # filters above drop branches; everything below only touches points).
    # Regular surfaces exit via c1_pass's Sigma-hull precheck at the cost
    # of six min/max scans — zero measurable time on all coverage cases.
    from mmcore.numeric.intersection.ssx._ssx5_singular import c1_pass

    ptol4_global = np.maximum(np.array(
        [float(_gp_s), float(_gp_t), float(_gp_u), float(_gp_v)]), 1e-9)
    if budget.exhausted:
        c1_hits, _c1_curve = [], False
    else:
        _c1_stats = {}
        c1_hits, _c1_curve = c1_pass(
            S1_h_top, S2_h_top, atol, ptol4_global,
            max_cells=min(20_000, budget.remaining_cells),
            charge_box=lambda n: budget.charge_cells(n, "c1"),
            stats=_c1_stats)
        if (_c1_stats.get("budget_exhausted", False)
                or _c1_stats.get("external_budget_exhausted", False)
                or _c1_stats.get("incomplete", False)):
            budget.mark_incomplete()
    for hit in c1_hits:
        if not _assembly_spend(budget):
            break
        if "curve_samples" in hit:
            samples = np.asarray(hit["curve_samples"], dtype=np.float64)
            anchor = samples[0] if len(samples) else np.full(4, np.nan)
            xyz_anchor = (eval_surface(S1_h_top, anchor[0], anchor[1],
                                       rational=True)
                          if len(samples) else np.full(3, np.nan))
            budget.append_output(all_singularities, SSXSingularity(
                kind="cusp_curve", stuv=anchor, xyz=xyz_anchor,
                samples=samples, surface=hit.get("surface")), "singularity")
            continue
        links = []
        _link_cost = sum(max(1, len(np.asarray(b.curve[1])) - 1)
                         for b in all_branches)
        if _assembly_spend(budget, max(1, _link_cost)):
            for bi, b in enumerate(all_branches):
                xyz = np.asarray(b.curve[1], dtype=np.float64)
                if len(xyz) < 2:
                    continue
                # Ledger L12: linkage must use point-to-SEGMENT distance — a
                # cusp exactly ON a coarse low-curvature span sits up to
                # half a chord (~h_max/2 >> 4·atol) from every VERTEX while
                # the polyline passes through it. Anchor the link at the
                # nearer endpoint of the nearest segment (same vertex
                # contract as C3's branch_links, ledger L11).
                if _dist_point_polyline(hit["xyz"], xyz) > 4.0 * atol:
                    continue
                a, bseg = xyz[:-1], xyz[1:]
                ab = bseg - a
                den = np.einsum("ij,ij->i", ab, ab)
                den = np.where(den < 1e-30, 1e-30, den)
                tt = np.clip(np.einsum(
                    "ij,ij->i", hit["xyz"][None, :] - a, ab)
                    / den, 0.0, 1.0)
                dseg = np.linalg.norm(
                    a + tt[:, None] * ab - hit["xyz"][None, :], axis=1)
                kseg = int(dseg.argmin())
                k = (kseg if np.linalg.norm(xyz[kseg] - hit["xyz"])
                     <= np.linalg.norm(xyz[kseg + 1] - hit["xyz"])
                     else kseg + 1)
                links.append((bi, k))
        budget.append_output(all_singularities, SSXSingularity(
            kind="cusp", stuv=np.asarray(hit["stuv"], dtype=np.float64),
            xyz=np.asarray(hit["xyz"], dtype=np.float64),
            branch_links=links, surface=hit.get("surface")), "singularity")

    # --- C3 pass (paper §5.4): 3D self-intersections of the SSI image ---
    # Runs AFTER tracing (branch geometry drives the candidate search).
    # Ledger L8: the old trigger — "some traced cell failed the per-cell
    # Theorem-3 certificate" — was semantically wrong: Theorem 3 certifies
    # only that ONE box's image is injective; a C3 whose two preimages lie
    # in DIFFERENT traced cells (figure-eight wall: s≈0.08 and s≈0.92
    # strips, each cell truthfully certified) passes every per-cell check
    # and the pass never ran. Run it whenever a collision is possible at
    # all: >= 2 branches (cross-branch), or a branch long enough to reach
    # its own segments past the broadphase index gap >= 3 (>= 8 segments,
    # within-branch). The vectorized AABB broadphase keeps the fired path
    # free on regular geometry (measured: 0 candidate pairs on coverage
    # case 10's 115 segments, ~0.3 ms; the old gate fired on every regular
    # coverage case anyway — top-cell T-hulls touch zero at domain edges —
    # so this was already the de-facto hot path).
    if (not budget.exhausted and all_branches
            and (len(all_branches) >= 2 or any(
            len(np.asarray(b.curve[1])) >= 9 for b in all_branches))):
        from mmcore.numeric.intersection.ssx._ssx5_singular import c3_pass

        _c3_stats = {}
        _c3_hits = c3_pass(
            S1_h_top, S2_h_top, all_branches, atol, ptol4_global,
            max_work=budget.remaining_cells,
            charge_work=lambda n: budget.charge_cells(n, "c3"),
            stats=_c3_stats,
        )
        for hit in _c3_hits:
            budget.append_output(all_singularities, SSXSingularity(
                kind="self_intersection",
                stuv=np.asarray(hit["stuv"], dtype=np.float64),
                stuv_mate=np.asarray(hit["stuv_mate"], dtype=np.float64),
                xyz=np.asarray(hit["xyz"], dtype=np.float64),
                branch_links=hit["links"]), "singularity")
        if _c3_stats.get("incomplete", False):
            budget.mark_incomplete()

    # A reported point within 2·atol (xyz) of an emitted tangent_point is
    # not a separate intersection — it is the certified tangency itself,
    # re-found by CSX grazing-valley seeds while subdividing around the
    # touch (measured on the paraboloid/Mexican-hat cases: 4 seeds at the
    # touch + 4 at ±1·atol on the grazing valley). Subsume them into the
    # typed singularity. Matching-ladder xyz guard only (2·atol); no param
    # guard needed — any Ψ-point that close to the certified tangency is
    # indistinguishable from it at tolerance.
    if (all_points and _tangent_xyz
            and _assembly_spend(
                budget, max(1, len(all_points) * len(_tangent_xyz)))):
        _tp_pts = np.asarray(_tangent_xyz, dtype=np.float64)      # (K, 3)
        all_points = [
            p for p in all_points
            if float(np.linalg.norm(
                _tp_pts - np.asarray(p.xyz, dtype=np.float64)[None, :],
                axis=1).min()) > 2.0 * atol
        ]

    # A reported point that lies ON a found branch is not an isolated
    # intersection — it is a corner-touch seed whose curve was traced by a
    # neighboring cell. Keep only genuinely isolated points.
    _point_branch_cost = (len(all_points) * sum(
        max(1, len(np.asarray(b.curve[1])) - 1)
        for b in all_branches))
    if (all_points and all_branches
            and _assembly_spend(budget, max(1, _point_branch_cost))):
        kept_points = []
        for p in all_points:
            pxyz = np.asarray(p.xyz, dtype=np.float64)
            on_branch = False
            for b in all_branches:
                poly = np.asarray(b.curve[1])
                if len(poly) < 2:
                    continue
                if _dist_point_polyline(pxyz, poly) <= 4.0 * atol:
                    on_branch = True
                    break
            if not on_branch:
                kept_points.append(p)
        all_points = kept_points

    # The same isolated point can be reported by several cells/arms (e.g.
    # touch-plus-loop at eps=0.02 surfaced two coincident SSXPoints ~2·atol
    # from the touch). Standard matching-ladder dedup: unify_tol per-axis
    # stuv box AND xyz <= 2·atol.
    if (all_points
            and _assembly_spend(
                budget, max(1, len(all_points) * len(all_points)))):
        all_points = _deduplicate_ssx_points(all_points, unify_tol, atol)

    return _result(all_branches, all_points, all_singularities)
