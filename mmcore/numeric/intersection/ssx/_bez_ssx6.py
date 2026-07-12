"""Bezier surface-surface intersection v5 — STALE PRE-BUDGET FORK (ledger L53).

.. warning::
   **Not maintained. Superseded by `_bez_ssx5.py`** (user decision
   2026-07-12, ledger L53: repair + document). This module is a frozen
   comparison fork from before the budget/status/singularity work:

   - NO work budgets: every nested CSX call gets a fresh 100k-cell
     allowance; there is no call-wide no-hang guarantee, no ``complete`` /
     ``status.reasons`` schema, and exhaustion semantics differ from the
     maintained engine.
   - NO typed singularities (``result['singularities']`` / ``SSXBranch.kind``
     do not exist here); none of the C1/C2/C3 machinery, the L-series
     soundness fixes, or the overlap-region contract ever landed.
   - Its CSX-contract guard (``_require_complete_csx_result``) hard-RAISES
     on any incomplete or fiber-bearing nested result instead of degrading
     honestly.

   Use it only as a historical comparison baseline; do not extend it, and
   do not file review findings against it beyond keeping it importable
   (its contract test pins the guard + the interior cut-face path).

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

import dataclasses
import sys
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Optional, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from mmcore.geom._nurbs_interp import interpolate_nurbs_curve
from mmcore.geom._nurbs_join import join_curves,join_curves_by_spec
from mmcore.numeric.bern import eval_bezier
from mmcore.numeric.bern_sq_dist import surface_surface_distance_squared_net_homog
from mmcore.numeric.intersection._bezier_common import (
    extract_weights, eval_surface, eval_surface_d1,eval_bezier_homogeneous_curve,eval_curve
)
from mmcore.numeric.intersection._sq_dist_classify import (
    _check_min_of_net, _check_lipschitz, _weight_max_product,
)
from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx
from mmcore.numeric.intersection._deflate import minors_Tpsi_from_control_nets
from mmcore.numeric._aabb import aabb, aabb_intersect, aabb_intersection

from mmcore.numeric.intersection.ssx._ssx4 import (
    SSXBranch, SSXPoint,
    _append_unique_point,
    GaussMapBern,
    separate_gauss_maps,
    _trust_gjk,
)
from mmcore.numeric.algorithms.cygjk import gjk
from mmcore.numeric.length import nurbs_length


# ---------------------------------------------------------------------------
# Data structures (§5 of design)
# ---------------------------------------------------------------------------

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

    def _replace(self, **kwargs):

        """Return a new BoundaryPoint with updated fields."""

        return dataclasses.replace(self, **kwargs)

# Back-compat alias — existing code uses BoundaryCrossing in many places and
# the design §5 name is BoundaryPoint. Keep both symbols pointing at the same
# dataclass so the rename can propagate gradually.
BoundaryCrossing = BoundaryPoint


@dataclass
class BoundaryOverlap:
    """A curve segment where S1 and S2 overlap on a boundary face of [0,1]⁴."""
    stuv_start: NDArray[np.float64]  # (4,)
    stuv_end: NDArray[np.float64]    # (4,)
    stuv: NDArray[np.float64]
    xyz: NDArray[np.float64]
    face: tuple[int, int]            # (axis 0-3, side 0-1)

    def _replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

# ---------------------------------------------------------------------------
# Level 1: Pruning
# ---------------------------------------------------------------------------

def _prune_ssx_cell(S1_h, S2_h, atol, rational=True):
    """Return True if this patch pair provably does NOT intersect.

    Checks:
    1. AABB non-overlap (Euclidean control points)
    2. Min-of-net on 4-variate sq-dist Bernstein net
    3. Lipschitz tightening on sq-dist net
    """
    # AABB check on Euclidean control points
    _, S1w = extract_weights(S1_h, rational=rational)
    _, S2w = extract_weights(S2_h, rational=rational)

    if rational:
        pts1 = S1_h[..., :-1] / S1_h[..., -1:]
        pts2 = S2_h[..., :-1] / S2_h[..., -1:]
    else:
        pts1 = S1_h
        pts2 = S2_h

    bb1 = np.array(aabb(pts1.reshape(-1, pts1.shape[-1])))
    bb2 = np.array(aabb(pts2.reshape(-1, pts2.shape[-1])))
    #bb1[0] -= atol; bb1[1] += atol
    #bb2[0] -= atol; bb2[1] += atol
    if not aabb_intersect(bb1, bb2):
        return True

    # Sq-dist net pruning
    F = surface_surface_distance_squared_net_homog(S1_h, S2_h, rational=rational)
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
    because it skips the sq-dist net computation."""
    pts1 = S1_h[..., :-1] / S1_h[..., -1:]
    pts2 = S2_h[..., :-1] / S2_h[..., -1:]
    bb1 = np.array(aabb(pts1.reshape(-1, pts1.shape[-1])))
    bb2 = np.array(aabb(pts2.reshape(-1, pts2.shape[-1])))
    ##print('aabb_intersect', aabb_intersect(bb1, bb2))
    res=aabb_intersect(bb1, bb2)
    if res:
        bbi=np.array(aabb_intersection(bb1, bb2))
        d=bbi[1,:]-bbi[0,:]

        ##print('d',d)
        if np.sum(np.abs(d)<atol)>=2:

            ##print('pdd',  np.prod(d))
            return True

    return not res


# ---------------------------------------------------------------------------
# Level 2: Boundary analysis (8 CSX problems)
# ---------------------------------------------------------------------------

def _require_complete_csx_result(result, context):
    """Reject CSX output that is unsafe to use as SSX boundary topology."""
    if (bool(result.get('budget_exhausted', False)) or
            not bool(result.get('boundary_topology_complete', True))):
        raise RuntimeError(
            f"{context}: incomplete Bezier CSX result cannot drive SSX "
            "topology (budget exhausted or boundary topology incomplete)"
        )
    if result.get('parameter_fibers'):
        raise RuntimeError(
            f"{context}: positive-dimensional parameter fiber requires "
            "explicit SSX overlap-region handling"
        )
    return result

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


def _find_ssx_boundary_zeros(S1_h, S2_h, atol, rational=True):
    """Find all intersection points and overlaps on the boundary of [0,1]⁴.

    Returns (crossings, overlaps).
    """
    crossings = []
    overlaps = []
    ptol_s,ptol_t=bez_surface_param_tolerance(S1_h,rational=rational,tol=atol)
    ptol_u,ptol_v=bez_surface_param_tolerance(S2_h,rational=rational,tol=atol)
    ptol=np.array([ptol_s,ptol_t,ptol_u,ptol_v])
    def _process_face(iso, other_surf, axis, side, owner_is_s1):
        result = bez_csx(iso, other_surf, atol=atol, rational=rational)
        _require_complete_csx_result(result, 'SSX boundary face')

        for iso_pt in result.get('isolated', []):
            t_crv = float(iso_pt['t'])
            u_oth = float(iso_pt['u'])
            v_oth = float(iso_pt['v'])
            stuv = _map_csx_to_stuv(axis, side, t_crv, u_oth, v_oth, owner_is_s1)
            xyz = np.asarray(iso_pt['point'], dtype=np.float64)
            face_id = axis if owner_is_s1 else axis + 2
            tang, _, _ = _ssx_tangent_4d(S1_h, S2_h, stuv[0], stuv[1], stuv[2], stuv[3], rational=rational)
            crossings.append(BoundaryPoint(stuv=stuv, xyz=xyz, face=(face_id, side),
                                           tangent_raw=tang))

        for ovl in result.get('overlaps', []):
            tr = ovl.get('t_range', (0.0, 1.0))
            ur = ovl.get('u_range', (0.0, 1.0))
            vr = ovl.get('v_range', (0.0, 1.0))

            # Overlap start
            stuv_s = _map_csx_to_stuv(axis, side, tr[0], ur[0], vr[0], owner_is_s1)
            stuv_e = _map_csx_to_stuv(axis, side, tr[1], ur[1], vr[1], owner_is_s1)
            face_id = axis if owner_is_s1 else axis + 2
            overlaps.append(BoundaryOverlap(stuv_start=stuv_s, stuv_end=stuv_e,stuv=np.array([stuv_s,stuv_e]),xyz=np.array([eval_curve(iso,stuv_s[0]),eval_curve(iso,stuv_e[0])]),
                                            face=(face_id, side)))
            # Also add endpoints as crossings (they connect to interior branches)
            xyz_s = eval_surface(S1_h, stuv_s[0], stuv_s[1], rational=rational)
            xyz_e = eval_surface(S1_h, stuv_e[0], stuv_e[1], rational=rational)
            #tang_s, _, _ = _ssx_tangent_4d(S1_h, S2_h, stuv_s[0], stuv_s[1], stuv_s[2], stuv_s[3], rational=rational)
            #tang_e, _, _ = _ssx_tangent_4d(S1_h, S2_h, stuv_e[0], stuv_e[1], stuv_e[2], stuv_e[3], rational=rational)
            crossings.append(BoundaryPoint(stuv=stuv_s, xyz=xyz_s, face=(face_id, side),
                                           tangent_raw=None))
            crossings.append(BoundaryPoint(stuv=stuv_e, xyz=xyz_e, face=(face_id, side),
                                           tangent_raw=None))

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

    crossings = _dedup_crossings(crossings,ptol)
    overlaps = _dedup_overlaps(overlaps, ptol)

    # Remove crossings that are endpoints of overlaps (redundant — overlap covers them)
    if overlaps:
        filtered = []
        for c in crossings:
            is_ovl_endpoint = False
            for ovl in overlaps:
                if (np.all(np.abs(c.stuv - ovl.stuv_start) < ptol) or
                        np.all(np.abs(c.stuv - ovl.stuv_end) < ptol)):
                    is_ovl_endpoint = True
                    break
            if not is_ovl_endpoint:
                filtered.append(c)
        crossings = filtered

    return crossings, overlaps


def _dedup_crossings(crossings, ptol):
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
        is_dup = any(np.all(np.abs(c.stuv - d.stuv) < ptol ) for d in deduped)
        if not is_dup:
            deduped.append(c)
    return deduped


def _dedup_overlaps(overlaps, ptol):
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
            same =np.all(np.abs(ovl.stuv_start - d.stuv_start) < ptol) and      np.all(np.abs(ovl.stuv_end - d.stuv_end) < ptol)

            # Check reversed direction
            rev = (np.all(np.abs(ovl.stuv_start - d.stuv_end) < ptol) and
                   np.all(np.abs(ovl.stuv_end - d.stuv_start) < ptol))
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


def _check_tangency(T1, T2, T3, T4, P1_cart, P2_cart, box):
    """Check if TΨ=0 has a simultaneous solution in the box.

    Uses the Krawczyk interval-Newton operator from _deflate.py to certify
    whether the 4-equation system {T1=0, T2=0, T3=0, T4=0} has a root.

    Returns
    -------
    True if tangency is confirmed (root certified or witness found),
    False if no tangent point exists in the box.
    None if undetermined (Krawczyk couldn't decide).
    """
    from mmcore.numeric.bern import bern_eval as _bern_eval
    from mmcore.numeric.ndinterval import interval as iv_interval, get_iarray
    from mmcore.numeric.intersection._deflate import (
        DeflatedSystem, build_square_from_subset, isolate_roots_krawczyk,
        gauss_newton_witness, _box_from_any,
    )

    try:
        # Convert control nets to interval arrays
        P1_iv = get_iarray(P1_cart, P1_cart)
        P2_iv = get_iarray(P2_cart, P2_cart)
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
    rank=np.linalg.matrix_rank(J)
    try:
        _, sigma, Vt = np.linalg.svd(J, full_matrices=True)
    except np.linalg.LinAlgError as e:
        print(f"SVD failed: {e}")
        return None, pt1, pt2

    # For a 3×4 Jacobian, the null space is (4 - rank)-dimensional.
    # With rank 3, null_dim = 1 and the last row of Vt is the tangent.
    # With rank < 3, null_dim > 1 and we need direction_hint.

    #tol_sv = max(J.shape) * sigma[0] * 1e-10 if sigma[0] > 0 else 1e-10
    #rank = int(np.sum(sigma > tol_sv))
    null_dim = 4 - rank  # for a 3×4 matrix

    if null_dim <= 0:
        return None, pt1, pt2

    if null_dim == 1 or direction_hint is None:
        tangent = Vt[-1]
        #print("tang",tangent)
        norm = np.linalg.norm(tangent)
        tangent = tangent / norm
    else:
        # Project direction_hint onto the null space
        null_vecs = Vt[-null_dim:]  # (null_dim, 4)
        coeffs = null_vecs @ direction_hint
        tangent = null_vecs.T @ coeffs
        norm = np.linalg.norm(tangent)
        if norm < 1e-14:
            tangent = Vt[-1]  # fallback
            norm = np.linalg.norm(tangent)
            tangent = tangent / norm
        else:
            tangent = tangent / norm

    return tangent, pt1, pt2


def _ssx_correct(S1, S2, s, t, u, v, rational=True, max_iter=5, tol=1e-14):
    """Newton corrector: project (s,t,u,v) back onto the intersection curve.

    Minimizes ||S1(s,t) - S2(u,v)|| using damped pseudoinverse steps.
    Only a few iterations — the predictor should be close.

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
        if g2 < tol:
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


def _march_intersection_curve(
    S1, S2,
    stuv_start, stuv_end,
    *,
    atol=1e-3,
    rational=True,
    initial_step=0.05,
    min_step=1e-6,
    max_step=0.25,
    angle_threshold=0.1,   # radians — target angle between consecutive tangents
    max_points=2000,
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

    current = stuv_start.copy().astype(np.float64)
    target = stuv_end.copy().astype(np.float64)
    step = initial_step

    # Direction hint: vector from start to end
    hint = target - current
    hint_norm = np.linalg.norm(hint)
    if hint_norm > 1e-15:
        hint = hint / hint_norm

    # Get initial tangent, using hint to resolve null-space ambiguity
    tang_prev, _, _ = _ssx_tangent_4d(S1, S2, *current, rational=rational, direction_hint=hint)
    if tang_prev is None:
        return np.array(stuv_pts), np.array(xyz_pts)

    # Orient tangent toward the target
    if np.dot(tang_prev, target - current) < 0:
        tang_prev = -tang_prev

    for _ in range(max_points):
        ##print("M@",_,step)
        # Check if we're close enough to the target
        dist_to_end = float(np.linalg.norm(current - target))
        if dist_to_end < max(step * 2, min_step * 4):
            # Close enough — add the endpoint and stop
            s, t, u, v, res, sin_ang = _ssx_correct(S1, S2, *target, rational=rational)
            if res < atol * max(sin_ang, 1e-3):
                final = np.array([s, t, u, v])
            else:
                final = target
            stuv_pts.append(final)
            xyz_pts.append(eval_surface(S1, final[0], final[1], rational=rational))
            break

        # Predictor: step along tangent
        predicted = current + step * tang_prev

        # Clamp to [0,1]⁴
        predicted = np.clip(predicted, 0.0, 1.0)

        # Corrector: project back onto intersection curve
        s, t, u, v, residual, sin_ang = _ssx_correct(
            S1, S2, predicted[0], predicted[1], predicted[2], predicted[3],
            rational=rational,
        )

        eff_atol = atol * max(sin_ang, 1e-3)
        if residual > eff_atol:
            # Corrector failed — reduce step and retry
            step = max(min_step, step * 0.5)
            continue

        corrected = np.array([s, t, u, v])

        corrected = np.array([s, t, u, v])

        # Update direction hint toward remaining target
        hint = target - corrected
        hn = np.linalg.norm(hint)
        if hn > 1e-15:
            hint = hint / hn

        # Get new tangent
        tang_new, pt1, _ = _ssx_tangent_4d(S1, S2, s, t, u, v, rational=rational, direction_hint=hint)
        if tang_new is None:
            ##print('ttth')
            step = max(min_step, step * 0.5)
            continue

        # Orient tangent consistently (same direction as previous)
        if np.dot(tang_new, tang_prev) < 0:
            tang_new = -tang_new

        # Compute angle between consecutive tangents
        cos_angle = np.clip(np.dot(tang_prev, tang_new), -1.0, 1.0)
        angle = np.arccos(abs(cos_angle))

        # Adapt step size based on curvature
        if angle > 1e-10:
            step = step * min(2.0, max(0.25, angle_threshold / angle))
        else:
            step = min(max_step, step * 1.5)  # very low curvature — grow step
        step = max(min_step, min(max_step, step))

        # Accept the point
        current = corrected
        tang_prev = tang_new
        stuv_pts.append(current.copy())
        xyz_pts.append(pt1.copy())

    return np.array(stuv_pts), np.array(xyz_pts)


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
      - exits as soon as ||G||² < `tol` (a true zero in machine precision);
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
        if g2 < tol:
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


def _march_to_boundary(
    S1, S2, stuv_start,
    *,
    atol=1e-3,
    rational=True,
    initial_step=0.05,
    min_step=1e-6,
    max_step=0.25,
    angle_threshold=0.1,
    max_points=40,
    direction_hint=None,
    no_progress_tol=1e-8,
    max_no_progress=3,
):
    """March from stuv_start until the curve hits a domain boundary [0,1]⁴.

    Like _march_intersection_curve but without a known endpoint.
    Stops when any parameter reaches 0 or 1.

    Returns (stuv_path, xyz_path).
    """
    stuv_pts = [stuv_start.copy()]
    xyz_pts = [eval_surface(S1, stuv_start[0], stuv_start[1], rational=rational)]

    current = stuv_start.copy().astype(np.float64)
    step = initial_step

    # Initial tangent
    tang_prev, _, _ = _ssx_tangent_4d(S1, S2, *current, rational=rational,
                                       direction_hint=direction_hint)
    if tang_prev is None:
        ##print('tamgg')
        return np.array(stuv_pts), np.array(xyz_pts)

    # Orient tangent using hint if provided
    if direction_hint is not None and np.dot(tang_prev, direction_hint) < 0:
        tang_prev = -tang_prev

    _first_step = True
    for iter_num in range(max_points):
        predicted = current + step * tang_prev

        #if _first_step:
            # On the first step, clamp prediction to [0,1]⁴ instead of
            # triggering a boundary event. This prevents the marcher from
            # bouncing off a face that the start point sits on: the tiny
            # outward component in the tangent is numerical noise, and the
            # corrector will pull the point back onto the true curve.
        #    #predicted = np.clip(predicted, 0.0, 1.0)

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
            # Angle-aware acceptance: if the corrector didn't converge
            # tightly enough for the local angle, the exit is unreliable.
            # The marcher silently appended such points before, then a
            # downstream tight match (1e-6 stuv) would fail. Now we either
            # accept a precise exit or REFUSE to commit a sloppy one — the
            # caller (the simplified tracer) treats a returned trace whose
            # last step doesn't extend the path as "no fragment", so this
            # naturally cascades to a retry / further subdivision.
            eff_atol = atol * max(fsin, 1e-3)
            if fres <= eff_atol:
                final_xyz = eval_surface(S1, final[0], final[1], rational=rational)
                stuv_pts.append(final)
                xyz_pts.append(final_xyz)
            break

        # No crossing: predicted stays inside [0,1]⁴. Normal corrector path.
        s, t, u, v, residual, sin_ang = _ssx_correct(
            S1, S2, *predicted, rational=rational,
        )
        ##print('_ssx_correct', predicted,s,t,u,v,residual,sin_ang)

        # Angle-aware acceptance: ||r|| < atol alone is misleading when the
        # surfaces approach slowly (small sin_ang). The xyz distance from the
        # true SSX curve is ≈ residual / sin_ang, so require residual to be
        # tighter when surfaces are close to parallel. The floor (1e-3) caps
        # how aggressive the tightening can get — without it, slowly
        # converging segments would never accept any correction at all.
        eff_atol = atol * max(sin_ang, 1e-3)
        ##print('eff_atol', eff_atol, residual,step, iter_num)

        if residual > eff_atol:
            step = max(min_step, step * 0.5)

            continue

        corrected = np.array([s, t, u, v])

        # New tangent
        tang_new, pt1, _ = _ssx_tangent_4d(S1, S2, *corrected, rational=rational,
                                            direction_hint=tang_prev)
        if tang_new is None:
            ##print('tamgg')
            step = max(min_step, step * 0.5)
            continue

        if np.dot(tang_new, tang_prev) < 0:
            tang_new = -tang_new

        # Step adaptation
        cos_angle = np.dot(tang_prev/    np.linalg.norm(tang_prev), tang_new/np.linalg.norm(tang_new))
        angle = np.arccos(abs(cos_angle))
        ##print('cos_angel',cos_angle,'angle',angle)
        if angle > 1e-10:
            step = step * min(2.0, max(0.25, angle_threshold / angle))
        else:
            step = min(max_step, step * 1.5)
        step = max(min_step, min(max_step, step))

        current = corrected
        tang_prev = tang_new
        _first_step = False
        stuv_pts.append(current.copy())
        xyz_pts.append(pt1.copy())

    return np.array(stuv_pts), np.array(xyz_pts)


# ---------------------------------------------------------------------------
# Φ-tracer crossing pairing — used only inside _deflate_tangent_cell (§8)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Level 4b: Φ-tracer for C₂ tangent cells
# ---------------------------------------------------------------------------

def _choose_phi_equations(S1, S2, T_arrs, seed_stuv, rational):
    """Choose the best 2 Ψ equations + 1 TΨ equation for the regulated system Φ.

    Picks the combination giving the best-conditioned 3×4 Jacobian at the seed.

    Returns (psi_rows, t_index) — indices into Ψ (0-2) and TΨ (0-3).
    """
    from itertools import combinations
    from mmcore.numeric.bern import bernstein_eval_nd, bernstein_partial_derivative_coeffs

    s, t, u, v = seed_stuv
    _, du1, dv1 = eval_surface_d1(S1, s, t, rational=rational)
    _, du2, dv2 = eval_surface_d1(S2, u, v, rational=rational)
    J_psi = np.column_stack([du1, dv1, -du2, -dv2])  # (3, 4)

    params = np.array(seed_stuv)
    best_score = -1.0
    best_choice = ((0, 1), 0)

    for ti in range(4):
        Tv = T_arrs[ti]
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
            if score > best_score:
                best_score = score
                best_choice = (psi_rows, ti)

    return best_choice


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
    initial_step=0.05,
    min_step=1e-6,
    max_step=0.25,
    angle_threshold=0.1,
    max_points=2000,
):
    """March along the Φ-curve from stuv_start toward stuv_end.

    Same predictor-corrector as _march_intersection_curve but on the
    regulated system Φ = {Ψ_i, Ψ_j, TΨ_k} instead of Ψ.

    At each point, also records whether the full Ψ is satisfied (indicating
    the point is on the actual intersection curve, not just on Φ).
    """
    stuv_pts = [stuv_start.copy()]
    xyz_start = eval_surface(S1, stuv_start[0], stuv_start[1], rational=rational)
    xyz_pts = [xyz_start]

    current = stuv_start.copy().astype(np.float64)
    target = stuv_end.copy().astype(np.float64)
    step = initial_step

    # Get tangent direction from Φ Jacobian
    J = _jac_phi(S1, S2, T_arr, psi_rows, *current, rational=rational)
    _, _, Vt = np.linalg.svd(J, full_matrices=True)
    tang_prev = Vt[-1]

    if np.dot(tang_prev, target - current) < 0:
        tang_prev = -tang_prev

    for _ in range(max_points):
        dist_to_end = float(np.linalg.norm(current - target))
        if dist_to_end < step * 2:
            stuv_pts.append(target.copy())
            xyz_pts.append(eval_surface(S1, target[0], target[1], rational=rational))
            break

        # Predictor
        predicted = np.clip(current + step * tang_prev, 0.0, 1.0)

        # Corrector: Newton on Φ = 0
        x = predicted.copy()
        for _ in range(5):
            f = _eval_phi(S1, S2, T_arr, psi_rows, *x, rational=rational)
            if np.dot(f, f) < 1e-20:
                break
            Jc = _jac_phi(S1, S2, T_arr, psi_rows, *x, rational=rational)
            JT = Jc.T
            A = JT @ Jc + 1e-12 * np.eye(4)
            delta = np.linalg.solve(A, -JT @ f)
            x = np.clip(x + delta, 0.0, 1.0)

        residual = np.linalg.norm(_eval_phi(S1, S2, T_arr, psi_rows, *x, rational=rational))
        if residual > atol * 100:
            step = max(min_step, step * 0.5)
            continue

        # New tangent
        J = _jac_phi(S1, S2, T_arr, psi_rows, *x, rational=rational)
        try:
            _, _, Vt = np.linalg.svd(J, full_matrices=True)
            tang_new = Vt[-1]
        except np.linalg.LinAlgError:
            step = max(min_step, step * 0.5)
            continue

        if np.dot(tang_new, tang_prev) < 0:
            tang_new = -tang_new

        cos_angle = np.clip(np.dot(tang_prev, tang_new), -1.0, 1.0)
        angle = np.arccos(abs(cos_angle))

        if angle > 1e-10:
            step = step * min(2.0, max(0.25, angle_threshold / angle))
        else:
            step = min(max_step, step * 1.5)
        step = max(min_step, min(max_step, step))

        current = x
        tang_prev = tang_new
        pt1 = eval_surface(S1, x[0], x[1], rational=rational)
        stuv_pts.append(current.copy())
        xyz_pts.append(pt1.copy())

    return np.array(stuv_pts), np.array(xyz_pts)
def _march_phi_curve_to_boundary(
    S1, S2, T_arr, psi_rows,
    stuv_start, stuv_end,
    *,
    atol=1e-3,
    rational=True,
    initial_step=0.05,
    min_step=1e-6,
    max_step=0.25,
    angle_threshold=0.1,
    max_points=2000,
):
    raise NotImplementedError

def _deflate_tangent_cell(P1_cart, P2_cart, T1, T2, T3, T4, box, crossings, atol,
                          *, originals=None, cell=None):
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
    from mmcore.numeric.bern import bernstein_partial_derivative_coeffs

    T_arrs = [np.asarray(T, dtype=np.float64)[..., np.newaxis] for T in [T1, T2, T3, T4]]

    fragments: list[_Fragment] = []
    points: list[SSXPoint] = []

    if len(crossings) < 2:
        for c in crossings:
            points.append(SSXPoint(stuv=c.stuv, xyz=c.xyz))
        return fragments, points

    # Choose Φ equations from the first crossing
    psi_rows, t_idx = _choose_phi_equations(
        P1_cart, P2_cart, T_arrs, crossings[0].stuv, rational=False,
    )
    T_chosen = T_arrs[t_idx]

    pairs, unpaired = _pair_crossings_for_tracing(crossings, originals=originals, cell=cell)
    build_branches(cell, crossings,atol=atol,marcher_2pt=lambda start,end,**kwargs:_march_phi_curve(P1_cart,P2_cart,T_chosen,psi_rows,start,end),marcher_to_boundary=_march_phi_curve_to_boundary,all_points=points,all_fragments=fragments )
    for i, j in pairs:
        stuv_path, xyz_path = _march_phi_curve(
            P1_cart, P2_cart, T_chosen, psi_rows,
            crossings[i].stuv, crossings[j].stuv,
            atol=atol, rational=False,
        )
        if len(stuv_path) < 2:
            continue
        # Check that points lie on the actual intersection (full Ψ=0).
        #valid_mask = np.zeros(len(stuv_path), dtype=bool)
        #
        #for k in range(len(stuv_path)):
        #    p1 = eval_surface(P1_cart, stuv_path[k, 0], stuv_path[k, 1], rational=False)
        #    p2 = eval_surface(P2_cart, stuv_path[k, 2], stuv_path[k, 3], rational=False)
        #    if np.linalg.norm(p1 - p2) < atol:
        #
        #        valid_mask[k] = True
        #
        #if not np.any(valid_mask):
        #    continue



        fragments.append(BezSSXBranch(

            stuv=np.array([_local_to_global(pt,   box)  for pt in stuv_path ])     ,
            xyz=xyz_path,atol=atol
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
            _, existing_xyz = b.xyz
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
        branches.append(BezSSXBranch(stuv_path, xyz_path,atol=atol, is_overlap=True))

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
        ##print('is_mono',is_mono,_)
        if is_mono:
            return True

    try:
        p1, p2 = separate_gauss_maps(g1.map_dirs(), g2.map_dirs())
        ##print('is_mono',    p1, p2)
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
def _compute_split_plan(crossings, cell_box, min_margin=1e-8):
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

    if surf_to_split == 1:
        isoline = _extract_isoline(cell.g1.surface, local_axis, cut_local)
        csx_result = bez_csx(isoline, cell.g2.surface, atol=atol, rational=True)
    else:
        isoline = _extract_isoline(cell.g2.surface, local_axis, cut_local)
        csx_result = bez_csx(isoline, cell.g1.surface, atol=atol, rational=True)
    _require_complete_csx_result(csx_result, 'SSX cut face')
    # Ledger L53: this was `list((lambda, seq))` — a 2-element list, not a
    # filter call — crashing the first interior cut of any run. Same
    # t-endpoint filter as the maintained v5 engine.
    csx_result['isolated'] = list(filter(
        lambda x: not (((1 - x['t']) < 1e-6) or (x['t'] < 1e-6)),
        csx_result['isolated']))
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
                      F_sq=left_F, w_scale=cell.w_scale)
    right_cell = _Cell(g1=right_g1, g2=right_g2, crossings=right_cx, box=right_box,
                       depth=cell.depth + 1,
                       T1=right_T1, T2=right_T2, T3=right_T3, T4=right_T4,
                       new_crossings=right_new,
                       F_sq=right_F, w_scale=cell.w_scale)

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
from mmcore.geom._nurbs_eval import NURBSCurveTuple

class BezSSXBranch:
        stuv: NDArray[np.float64]
        xyz: NDArray[np.float64]
        atol:float=1e-12
        is_overlap:bool=False
        xyz_curve:NURBSCurveTuple=field(init=False,default=None)
        st_curve:NURBSCurveTuple=field(init=False,default=None)
        uv_curve:NURBSCurveTuple=field(init=False,default=None)

        def __post_init__(self):
            mask=    np.linalg.norm(        self.xyz-np.roll(     self.xyz,1,axis=0),axis=1)>self.atol

            self.xyz=np.ascontiguousarray(self.xyz[mask])
            self.stuv = np.ascontiguousarray(self.stuv[mask])



        def build_interp(self):
            deg = self.degree
            if deg > 0:


                self.xyz_curve = interpolate_nurbs_curve(self.xyz, degree=deg, tol=self.atol)
                self.st_curve = interpolate_nurbs_curve(self.stuv[..., :2], degree=deg, tol=self.atol)
                self.uv_curve = interpolate_nurbs_curve(self.stuv[..., 2:], degree=deg, tol=self.atol)
                return True
            return False




        def closed(self,rtol=1.e-5, atol=1.e-8 ):

            return np.allclose(self.stuv[0]-self.stuv[-1],0.,rtol=rtol, atol=atol)

        @property
        def degree(self):
            return min(len(self.xyz)-1,3)

        @property
        def is_valid(self):
            if self.degree<1:
                return False
            if any([self.xyz_curve is None,self.st_curve is None,self.uv_curve is None]):
                return False
            return (self.xyz_curve.control_points.shape[0]>=2) and  (self.xyz_curve.control_points.shape[0]==self.st_curve.control_points.shape[0]==self.uv_curve.control_points.shape[0])
if sys.version_info >= (3, 11):

    BezSSXBranch=dataclass(slots=True,unsafe_hash=True)(BezSSXBranch)
else:
    BezSSXBranch = dataclass(unsafe_hash=True)(BezSSXBranch)


from mmcore.geom._nurbs_param_tol import bez_surface_param_tolerance



from dataclasses import FrozenInstanceError

from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

# =============================================================================

# Requires the join algorithm from the previous block:

#

#   join_curves(...)

#   join_curves_by_spec(...)

#   JoinChainSpec

#   NURBSCurveTuple

#

# This adapter joins BezSSXBranch-like objects and keeps:

#   - xyz_curve

#   - st_curve

#   - uv_curve

#   - raw xyz samples

#   - raw stuv samples

#

# in the same source order / reversal / parameterisation / knot-removal pattern.

# =============================================================================

_CURVE_ATTR_ALIASES = {

    "xyz": "xyz_curve",

    "xyz_curve": "xyz_curve",

    "st": "st_curve",

    "st_curve": "st_curve",

    "uv": "uv_curve",

    "uv_curve": "uv_curve",

}

def _resolve_curve_attr(selector: str) -> str:

    try:

        return _CURVE_ATTR_ALIASES[selector]

    except KeyError:

        raise ValueError(

            f"Unknown curve selector {selector!r}. "

            f"Expected one of {sorted(_CURVE_ATTR_ALIASES)}."

        )

def _safe_setattr(obj: Any, name: str, value: Any):

    """

    Works for normal dataclasses and frozen dataclasses.

    """

    try:

        setattr(obj, name, value)

    except (FrozenInstanceError, AttributeError):

        object.__setattr__(obj, name, value)

def _branch_curve(branch: Any, selector: str | Callable[[Any], Any]):

    if callable(selector):

        crv = selector(branch)

        if crv is None:

            raise ValueError("Custom curve selector returned None")

        return crv

    attr = _resolve_curve_attr(selector)

    crv = getattr(branch, attr, None)

    if crv is None:

        raise ValueError(f"Branch has no built curve at attribute {attr!r}")

    return crv

def _ensure_branch_interpolated(

    branch: Any,

    *,

    build_interp: bool = True,

    check_valid: bool = True,

):

    """

    Ensures branch.xyz_curve, branch.st_curve and branch.uv_curve exist.

    The supplied BezSSXBranch has build_interp(), so this calls it when needed.

    """

    missing = any(

        getattr(branch, attr, None) is None

        for attr in ("xyz_curve", "st_curve", "uv_curve")

    )

    if missing and build_interp:

        ok = branch.build_interp()

        if ok is False:

            raise ValueError("branch.build_interp() returned False")

    if check_valid and hasattr(branch, "is_valid"):

        if not branch.is_valid:

            raise ValueError("Invalid BezSSXBranch: branch.is_valid is False")

    for attr in ("xyz_curve", "st_curve", "uv_curve"):

        if getattr(branch, attr, None) is None:

            raise ValueError(f"Branch has no {attr}; call build_interp() first")

def _validate_branch_samples(branch: Any, index: int):

    xyz = np.asarray(branch.xyz, dtype=float)

    stuv = np.asarray(branch.stuv, dtype=float)

    if xyz.ndim != 2:

        raise ValueError(f"Branch {index}: xyz must be a 2D array")

    if stuv.ndim != 2:

        raise ValueError(f"Branch {index}: stuv must be a 2D array")

    if len(xyz) != len(stuv):

        raise ValueError(

            f"Branch {index}: len(xyz)={len(xyz)} but len(stuv)={len(stuv)}"

        )

    if len(xyz) < 2:

        raise ValueError(f"Branch {index}: at least two raw samples are required")

    if stuv.shape[1] < 4:

        raise ValueError(

            f"Branch {index}: stuv must have at least 4 columns: s, t, u, v"

        )

def _copy_branch_samples(branch: Any, *, reversed_: bool):

    xyz = np.ascontiguousarray(np.asarray(branch.xyz, dtype=float).copy())

    stuv = np.ascontiguousarray(np.asarray(branch.stuv, dtype=float).copy())

    if reversed_:

        xyz = np.ascontiguousarray(xyz[::-1].copy())

        stuv = np.ascontiguousarray(stuv[::-1].copy())

    return xyz, stuv

def _snap_sample_endpoint(

    A: np.ndarray,

    a_idx: int,

    B: np.ndarray,

    b_idx: int,

    *,

    policy: str,

):

    """

    Snap raw sample endpoints.

    policy:

      - "average": both endpoints become 0.5 * (A + B)

      - "left":    B endpoint becomes A endpoint

      - "right":   A endpoint becomes B endpoint

      - "none":    no snapping

    """

    if policy == "none":

        return

    if policy == "average":

        p = 0.5 * (A[a_idx] + B[b_idx])

        A[a_idx] = p

        B[b_idx] = p

        return

    if policy == "left":

        B[b_idx] = A[a_idx]

        return

    if policy == "right":

        A[a_idx] = B[b_idx]

        return

    raise ValueError(

        f"Unknown raw endpoint snap policy {policy!r}. "

        "Expected 'average', 'left', 'right' or 'none'."

    )

def merge_branch_samples_by_spec(

    branches: Sequence[Any],

    spec,

    *,

    endpoint_policy: str = "average",

    drop_join_duplicates: bool = True,

    keep_cycle_closure_sample: bool = True,

):

    """

    Merge raw branch.xyz and branch.stuv arrays according to one JoinChainSpec.

    This does not resample. It only:

      - orders source branches

      - reverses raw arrays when the curve segment was reversed

      - snaps join endpoints using endpoint_policy

      - removes duplicate join samples between consecutive segments

    Returns

    -------

    xyz, stuv

    """

    if not spec.segments:

        raise ValueError("Cannot merge samples from an empty JoinChainSpec")

    xyz_pieces = []

    stuv_pieces = []

    for seg in spec.segments:

        branch = branches[int(seg.source_index)]

        xyz, stuv = _copy_branch_samples(

            branch,

            reversed_=bool(seg.is_reversed),

        )

        xyz_pieces.append(xyz)

        stuv_pieces.append(stuv)

    # Snap consecutive joins.

    for i in range(len(xyz_pieces) - 1):

        _snap_sample_endpoint(

            xyz_pieces[i],

            -1,

            xyz_pieces[i + 1],

            0,

            policy=endpoint_policy,

        )

        _snap_sample_endpoint(

            stuv_pieces[i],

            -1,

            stuv_pieces[i + 1],

            0,

            policy=endpoint_policy,

        )

    # Snap cycle closure last -> first.

    if spec.is_cycle and keep_cycle_closure_sample:

        if len(xyz_pieces) == 1:

            _snap_sample_endpoint(

                xyz_pieces[0],

                -1,

                xyz_pieces[0],

                0,

                policy=endpoint_policy,

            )

            _snap_sample_endpoint(

                stuv_pieces[0],

                -1,

                stuv_pieces[0],

                0,

                policy=endpoint_policy,

            )

        else:

            _snap_sample_endpoint(

                xyz_pieces[-1],

                -1,

                xyz_pieces[0],

                0,

                policy=endpoint_policy,

            )

            _snap_sample_endpoint(

                stuv_pieces[-1],

                -1,

                stuv_pieces[0],

                0,

                policy=endpoint_policy,

            )

    xyz_parts = [xyz_pieces[0]]

    stuv_parts = [stuv_pieces[0]]

    for i in range(1, len(xyz_pieces)):

        if drop_join_duplicates:

            xyz_parts.append(xyz_pieces[i][1:])

            stuv_parts.append(stuv_pieces[i][1:])

        else:

            xyz_parts.append(xyz_pieces[i])

            stuv_parts.append(stuv_pieces[i])

    xyz_merged = np.ascontiguousarray(np.vstack(xyz_parts))

    stuv_merged = np.ascontiguousarray(np.vstack(stuv_parts))

    if spec.is_cycle and not keep_cycle_closure_sample and len(xyz_merged) > 1:

        xyz_merged = np.ascontiguousarray(xyz_merged[:-1])

        stuv_merged = np.ascontiguousarray(stuv_merged[:-1])

    if len(xyz_merged) != len(stuv_merged):

        raise RuntimeError("Internal error: merged xyz/stuv lengths differ")

    return xyz_merged, stuv_merged

def _default_branch_factory(

    *,

    template: Any,

    xyz: np.ndarray,

    stuv: np.ndarray,

    xyz_curve,

    st_curve,

    uv_curve,

    source_branches: Sequence[Any],

    spec,

):

    """

    Create a new object of the same class as template.

    Important:

    The supplied BezSSXBranch.__post_init__ removes duplicate points. For joined

    closed cycles we often intentionally keep the final closure sample, so this

    factory restores xyz/stuv after construction.

    """

    cls = type(template)

    atol = max(float(getattr(b, "atol", 1e-12)) for b in source_branches)

    is_overlap = any(bool(getattr(b, "is_overlap", False)) for b in source_branches)

    obj = None

    # Most likely dataclass constructor.

    try:

        obj = cls(

            stuv=np.ascontiguousarray(stuv.copy()),

            xyz=np.ascontiguousarray(xyz.copy()),

            atol=atol,

            is_overlap=is_overlap,

        )

    except TypeError:

        pass

    # Fallback constructor without optional fields.

    if obj is None:

        try:

            obj = cls(

                stuv=np.ascontiguousarray(stuv.copy()),

                xyz=np.ascontiguousarray(xyz.copy()),

            )

        except TypeError:

            pass

    # Last-resort construction without __init__.

    if obj is None:

        obj = cls.__new__(cls)

    # Restore exactly the merged raw arrays, regardless of __post_init__.

    _safe_setattr(obj, "stuv", np.ascontiguousarray(stuv.copy()))

    _safe_setattr(obj, "xyz", np.ascontiguousarray(xyz.copy()))

    _safe_setattr(obj, "atol", atol)

    _safe_setattr(obj, "is_overlap", is_overlap)

    _safe_setattr(obj, "xyz_curve", xyz_curve)

    _safe_setattr(obj, "st_curve", st_curve)

    _safe_setattr(obj, "uv_curve", uv_curve)

    return obj

def _materialize_joined_branches_from_specs(

    branches: Sequence[Any],

    specs: Sequence[Any],

    *,

    precomputed_joined_curves: Optional[dict[str, Sequence[Any]]] = None,

    branch_factory: Optional[Callable[..., Any]] = None,

    raw_endpoint_policy: str = "average",

    drop_join_duplicates: bool = True,

    keep_cycle_closure_sample: bool = True,

    snap_curves: bool = True,

    strict_order: bool = True,

    apply_recorded_knot_removals: bool = True,

    validate_replayed_knot_removal: bool = False,

    replay_knot_removal_tolerance: float = 1e-4,

    knot_remover=None,

):

    """

    Builds output BezSSXBranch-like objects from specs.

    precomputed_joined_curves can contain any of:

      - "xyz_curve"

      - "st_curve"

      - "uv_curve"

    Missing curve families are replayed with join_curves_by_spec().

    """

    precomputed_joined_curves = dict(precomputed_joined_curves or {})

    joined_curves_by_attr = {}

    for attr in ("xyz_curve", "st_curve", "uv_curve"):

        if attr in precomputed_joined_curves:

            joined_curves_by_attr[attr] = list(precomputed_joined_curves[attr])

        else:

            source_curves = [getattr(b, attr) for b in branches]

            joined_curves_by_attr[attr] = join_curves_by_spec(

                source_curves,

                specs,

                snap=snap_curves,

                strict_order=strict_order,

                apply_recorded_knot_removals=apply_recorded_knot_removals,

                validate_replayed_knot_removal=validate_replayed_knot_removal,

                replay_knot_removal_tolerance=replay_knot_removal_tolerance,

                knot_remover=knot_remover,

            )

    factory = branch_factory or _default_branch_factory

    out = []

    for out_i, spec in enumerate(specs):

        xyz_raw, stuv_raw = merge_branch_samples_by_spec(

            branches,

            spec,

            endpoint_policy=raw_endpoint_policy,

            drop_join_duplicates=drop_join_duplicates,

            keep_cycle_closure_sample=keep_cycle_closure_sample,

        )

        source_branches = [branches[int(i)] for i in spec.source_indices]

        template = source_branches[0]

        obj = factory(

            template=template,

            xyz=xyz_raw,

            stuv=stuv_raw,

            xyz_curve=joined_curves_by_attr["xyz_curve"][out_i],

            st_curve=joined_curves_by_attr["st_curve"][out_i],

            uv_curve=joined_curves_by_attr["uv_curve"][out_i],

            source_branches=source_branches,

            spec=spec,

        )

        out.append(obj)

    return out

def join_bezssx_branches(

    branches: Iterable[Any],

    *,

    driver: str | Callable[[Any], Any] = "xyz_curve",

    tol: float = 1e-6,

    build_interp: bool = True,

    check_valid: bool = True,

    branch_factory: Optional[Callable[..., Any]] = None,

    raw_endpoint_policy: str = "average",

    drop_join_duplicates: bool = True,

    keep_cycle_closure_sample: bool = True,

    snap_replayed_curves: bool = True,

    strict_replayed_order: bool = True,

    validate_replayed_knot_removal: bool = False,

    replay_knot_removal_tolerance: Optional[float] = None,

    **join_kwargs,

):

    """

    Join BezSSXBranch-like objects.

    Parameters

    ----------

    branches:

        Iterable of BezSSXBranch-like objects.

    driver:

        Which interpolated curve family should determine topology.

        Accepted strings:

          - "xyz" or "xyz_curve"

          - "st"  or "st_curve"

          - "uv"  or "uv_curve"

        You may also pass a callable:

            driver=lambda branch: branch.xyz_curve

    tol:

        Topological endpoint tolerance used by join_curves().

    raw_endpoint_policy:

        How to snap raw xyz/stuv join samples:

          - "average": same idea as curve endpoint snapping

          - "left": keep previous segment endpoint

          - "right": keep next segment endpoint

          - "none": do not alter raw endpoints

    keep_cycle_closure_sample:

        If True, closed joined chains keep final sample equal to first sample.

    join_kwargs:

        Forwarded to join_curves(), for example:

          - reparameterize_c1=True

          - c1_direction_tol=1e-6

          - remove_c1_knots=True

          - knot_removal_tolerance=...

          - knot_remover=...

    Returns

    -------

    joined_branches, specs

    """

    branches = list(branches)

    if not branches:

        raise ValueError("Empty branch list")

    for i, b in enumerate(branches):

        _validate_branch_samples(b, i)

        _ensure_branch_interpolated(

            b,

            build_interp=build_interp,

            check_valid=check_valid,

        )

    driver_curves = [_branch_curve(b, driver) for b in branches]

    joined_driver_curves, specs = join_curves(

        driver_curves,

        tol=tol,

        **join_kwargs,

    )

    if replay_knot_removal_tolerance is None:

        replay_knot_removal_tolerance = join_kwargs.get("knot_removal_tolerance", tol)

        if replay_knot_removal_tolerance is None:

            replay_knot_removal_tolerance = tol

    precomputed = {}

    if not callable(driver):

        driver_attr = _resolve_curve_attr(driver)

        precomputed[driver_attr] = joined_driver_curves

    joined_branches = _materialize_joined_branches_from_specs(

        branches,

        specs,

        precomputed_joined_curves=precomputed,

        branch_factory=branch_factory,

        raw_endpoint_policy=raw_endpoint_policy,

        drop_join_duplicates=drop_join_duplicates,

        keep_cycle_closure_sample=keep_cycle_closure_sample,

        snap_curves=snap_replayed_curves,

        strict_order=strict_replayed_order,

        apply_recorded_knot_removals=True,

        validate_replayed_knot_removal=validate_replayed_knot_removal,

        replay_knot_removal_tolerance=float(replay_knot_removal_tolerance),

        knot_remover=join_kwargs.get("knot_remover", None),

    )

    return joined_branches, specs

def join_bezssx_branches_by_spec(

    branches: Iterable[Any],

    specs: Sequence[Any],

    *,

    build_interp: bool = True,

    check_valid: bool = True,

    branch_factory: Optional[Callable[..., Any]] = None,

    raw_endpoint_policy: str = "average",

    drop_join_duplicates: bool = True,

    keep_cycle_closure_sample: bool = True,

    snap_curves: bool = True,

    strict_order: bool = True,

    apply_recorded_knot_removals: bool = True,

    validate_replayed_knot_removal: bool = False,

    replay_knot_removal_tolerance: float = 1e-4,

    knot_remover=None,

):

    """

    Replay a previously computed BezSSXBranch join operation on another

    corresponding branch set.

    This uses the same:

      - source branch indices

      - segment reversals

      - degree elevations

      - C1 reparameterised segment lengths

      - accepted knot removals

      - raw xyz/stuv merge order

    """

    branches = list(branches)

    if not branches:

        raise ValueError("Empty branch list")

    for i, b in enumerate(branches):

        _validate_branch_samples(b, i)

        _ensure_branch_interpolated(

            b,

            build_interp=build_interp,

            check_valid=check_valid,

        )

    return _materialize_joined_branches_from_specs(

        branches,

        specs,

        precomputed_joined_curves=None,

        branch_factory=branch_factory,

        raw_endpoint_policy=raw_endpoint_policy,

        drop_join_duplicates=drop_join_duplicates,

        keep_cycle_closure_sample=keep_cycle_closure_sample,

        snap_curves=snap_curves,

        strict_order=strict_order,

        apply_recorded_knot_removals=apply_recorded_knot_removals,

        validate_replayed_knot_removal=validate_replayed_knot_removal,

        replay_knot_removal_tolerance=float(replay_knot_removal_tolerance),

        knot_remover=knot_remover,

    )

# =============================================================================

# Example usage

# =============================================================================

# Join using xyz geometry as the topology driver:

#

# joined, specs = join_bezssx_branches(

#     branches,

#     driver="xyz",

#     tol=1e-6,

#     reparameterize_c1=True,

#     c1_direction_tol=1e-6,

#     remove_c1_knots=True,

#     knot_removal_tolerance=1e-5,

# )

#

# Replay exactly the same join on a corresponding branch set:

#

# joined_other = join_bezssx_branches_by_spec(

#     other_branches,

#     specs,

#     validate_replayed_knot_removal=False,

# )

#

# Join based on parameter-space ST instead:

#

# joined_by_st, specs_st = join_bezssx_branches(

#     branches,

#     driver="st",

#     tol=1e-6,

# )

#

# Join based on UV instead:

#

# joined_by_uv, specs_uv = join_bezssx_branches(

#     branches,

#     driver="uv",

#     tol=1e-6,

# )
def build_branches(cell, isolated_boundary_inters, atol, all_points=None, all_fragments=None ):
    if all_points is None:
        all_points=[]
    if all_fragments is None:
        all_fragments=[]
    isol=list(isolated_boundary_inters)

    if len(isol) == 1:
        all_points.append(_local_to_global(isol[0].stuv,cell.box))

    elif len(isol) >= 2:
        s1_ptol = bez_surface_param_tolerance(cell.g1.surface, atol, rational=True)
        s2_ptol = bez_surface_param_tolerance(cell.g2.surface, atol, rational=True)
        stuv_ptol = np.array([*s1_ptol, *s2_ptol], dtype=float)
        if len(isol) == 2:
            if np.all(np.abs(isol[0].stuv-isol[1].stuv)<stuv_ptol):
                all_fragments.append(BezSSXBranch(np.array([_local_to_global(i, cell.box) for i in [isol[0].stuv,isol[1].stuv]]),  np.array([isol[0].xyz,isol[1].xyz]),atol=atol))
            else:
                res_stuv, res_xyz = _march_intersection_curve(cell.g1.surface, cell.g2.surface, isol[0].stuv, isol[1].stuv, atol=atol,
                                                          rational=True,max_points=32,min_step=min(stuv_ptol))

                all_fragments.append(BezSSXBranch(np.array([_local_to_global(i,cell.box)for i in res_stuv]), res_xyz))
        else:
            isol_stack = list(isol)



            maby_ends = []

            while isol_stack :
                #print(len(isol_stack))
                stuv = isol_stack.pop()

                res_stuv, res_xyz = _march_to_boundary(cell.g1.surface, cell.g2.surface, stuv.stuv, atol=atol,max_points=32
                                                     )
                #print( res_stuv, res_xyz )

                if isol_stack:

                    kd1 = np.array([i.stuv for i in isol_stack])
                    mask1 = np.all(np.abs(kd1 - res_stuv[None, -1, :]) < stuv_ptol, axis=1)
                    if np.any(mask1):
                        isol_stack.pop(next(np.nditer(np.arange(len(isol_stack))[mask1])))
                        all_fragments.append(BezSSXBranch(np.array([_local_to_global(i,cell.box)for i in res_stuv]), res_xyz,atol=atol*2))
                        continue
                if maby_ends:
                    kd1 = np.array([i.stuv for i in maby_ends])
                    mask1 = np.all(np.abs(kd1 - res_stuv[None, -1, :]) < stuv_ptol, axis=1)
                    if np.any(mask1):
                        maby_ends.pop(next(np.nditer(np.arange(len(maby_ends))[mask1])))
                        all_fragments.append(
                            BezSSXBranch(np.array([_local_to_global(i, cell.box) for i in res_stuv]), res_xyz,atol=atol*2))
                        continue
                maby_ends.append(stuv)

    else:
        #print('no boundary intersections')
        ...

    return all_points,all_fragments


def build_tangential_branches(cell, isolated_boundary_inters, atol,all_points=None, all_fragments=None ):

    print('tan branches')
    if all_points is None:
        all_points=[]
    if all_fragments is None:

        all_fragments=[]
    P1_cart = cell.g1.surface[..., :-1] / cell.g1.surface[..., None, -1]
    P2_cart = cell.g2.surface[..., :-1] / cell.g2.surface[..., None, -1]
    # Choose Φ equations from the first crossing

    T_arrs = [np.asarray(T, dtype=np.float64)[..., np.newaxis] for T in [cell.T1, cell.T2, cell.T3, cell.T4]]
    psi_rows, t_idx = _choose_phi_equations(
        P1_cart, P2_cart, T_arrs, isolated_boundary_inters[0].stuv, rational=False,
    )
    T_chosen = T_arrs[t_idx]
    isol=list(isolated_boundary_inters)

    if len(isol) == 1:
        all_points.append(_local_to_global(isol[0],cell.box))

    elif len(isol) >= 2:
        if len(isol) == 2:

            res_stuv, res_xyz = _march_phi_curve(cell.g1.surface, cell.g2.surface,T_chosen,psi_rows , isol[0].stuv, isol[1].stuv,atol=atol,
                                                          rational=True)

            all_fragments.append(BezSSXBranch(np.array([_local_to_global(i,cell.box)for i in res_stuv]), res_xyz,atol=atol*2))
        else:
            raise NotImplementedError("More than two isolated boundary intersections not supported")
            isol_stack = list(isol)

            s1_ptol = bez_surface_param_tolerance(cell.g1.surface, atol, rational=True)
            s2_ptol = bez_surface_param_tolerance(cell.g2.surface, atol, rational=True)
            stuv_ptol = np.array([*s1_ptol, *s2_ptol], dtype=float)
            maby_ends = []

            while isol_stack:
                stuv = isol_stack.pop()

                res_stuv, res_xyz = _march_to_boundary(cell.g1.surface, cell.g2.surface, stuv, atol=atol,
                                                     )
                if maby_ends:
                    kd1 = np.array([i.stuv for i in maby_ends])
                    mask1 = np.all(np.abs(kd1 - res_stuv[None, -1, :]) < stuv_ptol, axis=1)
                    if np.any(mask1):
                        maby_ends.pop(next(np.nditer(np.arange(len(maby_ends))[mask1])))
                        all_fragments.append(BezSSXBranch(np.array([_local_to_global(i,cell.box)for i in res_stuv]), res_xyz))
                        continue
                if isol_stack:

                    kd1 = np.array([i.stuv for i in isol_stack])
                    mask1 = np.all(np.abs(kd1 - res_stuv[None, -1, :]) < stuv_ptol, axis=1)
                    if np.any(mask1):
                        isol_stack.pop(next(np.nditer(np.arange(len(isol_stack))[mask1])))
                        all_fragments.append(BezSSXBranch(res_stuv, res_xyz))
                        continue
                maby_ends.append(stuv)
    else:
        print('no boundary intersections')

    return all_points,all_fragments
def bez_ssx(
    S1,
    S2,
    atol=1e-3,
    rational=True,
    max_depth=24,
) -> dict:
    """Bezier surface-surface intersection v5.

    Iterative stack-based domain decomposition.
    All crossings and branch endpoints are in GLOBAL [0,1]⁴ coordinates.
    Surfaces in sub-cells are in LOCAL [0,1]² (De Casteljau reparameterized).
    Conversion between local and global uses the cell's box.

    Returns dict with 'branches' and 'points'.
    """
    S1 = np.asarray(S1, dtype=np.float64)
    S2 = np.asarray(S2, dtype=np.float64)

    # --- Level 1: Pruning ---
    if _prune_ssx_cell(S1, S2, atol, rational=rational):
        return {'branches': [], 'points': []}

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
    crossings, boundary_overlaps = _find_ssx_boundary_zeros(S1, S2, atol, rational=rational)
    overlap_branches = _overlaps_to_branches(boundary_overlaps, S1, atol, rational)

    # --- Level 3: TΨᵢ (once at top level) ---
    if rational:
        P1_cart = S1[..., :-1] / S1[..., -1:]
        P2_cart = S2[..., :-1] / S2[..., -1:]
    else:
        P1_cart = S1
        P2_cart = S2
    T1, T2, T3, T4 = minors_Tpsi_from_control_nets(P1_cart, P2_cart)

    # --- Top-level cell + outer partitions (design §5) ---
    box = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    T1_arr = _tpsi_to_numpy(T1)
    T2_arr = _tpsi_to_numpy(T2)
    T3_arr = _tpsi_to_numpy(T3)
    T4_arr = _tpsi_to_numpy(T4)

    # Build sq-dist net once at top level; propagated by de Casteljau split.
    if rational:
        S1_h_top = S1
        S2_h_top = S2
    else:
        S1_h_top = np.concatenate([S1, np.ones(S1.shape[:-1]+(1,))], axis=-1)
        S2_h_top = np.concatenate([S2, np.ones(S2.shape[:-1]+(1,))], axis=-1)
    F_sq_top = surface_surface_distance_squared_net_homog(
        S1_h_top, S2_h_top, rational=True)
    _, S1w_top = extract_weights(S1_h_top, rational=True)
    _, S2w_top = extract_weights(S2_h_top, rational=True)
    w_scale_top = _weight_max_product(S1w_top.ravel(), S2w_top.ravel())

    top_cell = _Cell(
        g1=g1, g2=g2, crossings=crossings, box=box, depth=0,
        T1=T1_arr, T2=T2_arr, T3=T3_arr, T4=T4_arr,
        new_crossings=list(crossings),
        F_sq=F_sq_top, w_scale=w_scale_top,
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
    all_fragments: list[BezSSXBranch] = []
    all_points = []

    while queue:
        ##print(len(queue), len(all_points), len(all_fragments))
        cell = queue.popleft()

        # Cheap AABB pruning first: if control-point bounding boxes don't
        # overlap, there is no intersection in this cell.
        if _aabb_disjoint(cell.g1.surface, cell.g2.surface, atol):
            continue


        # GJK separability: tighter than AABB, much cheaper than the sq-dist
        # net or Gauss separability. Test the convex hulls of the two control
        # nets — if they're separated, the surfaces don't intersect.

        P1_pts = np.ascontiguousarray((cell.g1.surface[..., :-1] / cell.g1.surface[..., -1:]).reshape(-1, 3),dtype=float)
        P2_pts = np.ascontiguousarray((cell.g2.surface[..., :-1] / cell.g2.surface[..., -1:]).reshape(-1, 3),dtype=float)
        if  not gjk(P1_pts, P2_pts, atol, 15):
                continue


        # Sq-dist net pruning using the PROPAGATED F_sq (built once at top,
        # split alongside TΨᵢ at every subdivision — never reconstructed).
        if cell.F_sq is not None:
            if _check_min_of_net(cell.F_sq, atol, cell.w_scale):
                continue
            if _check_lipschitz(cell.F_sq, atol, cell.w_scale):
                continue

        # Loop-absence on this sub-cell — TΨᵢ monotonicity (cheap) tried first,
        # Gauss map separability as fallback (design §6, §10 principle 8).


        if _check_loop_free(cell.g1, cell.g2,
                            cell.T1, cell.T2, cell.T3, cell.T4):

                    isol, over = _find_ssx_boundary_zeros(cell.g1.surface, cell.g2.surface, atol=atol, rational=True)
                    if len(isol) == 0:
                        continue
                    #print('crossings:',len(cell.crossings),'isol:',len(isol),'over:',len(over))
                    #print('crossings:', [c.stuv.tolist() for c in cell.crossings], '\nisol:',


                    #isol_global=[isol[i]._replace(stuv=_local_to_global(isol[i].stuv,cell.box)) for i in range(len(isol))]
                    #over_global= [o._replace(stuv_start=_local_to_global(o.stuv_start,cell.box),stuv_end=_local_to_global(o.stuv_end,cell.box) ) for o in over]

                    #all_fragments.extend(over_global)


                    build_branches(cell, isol, atol, all_points=all_points, all_fragments=all_fragments )













                    #fr, pt = _trace_cell_by_registrations(cell, atol)
                    #all_fragments.extend(fr)
                    #all_points.extend(pt)
                    continue


        # §6 step 3: Krawczyk-based tangency certification. If TΨ = 0 has a
        # simultaneous root in this cell, the intersection is tangential (C₂)
        # and must be traced via the regulated Φ system (design §1.4, §8),
        # NOT by further subdivision — deflation makes the Φ-curve regular
        # where Ψ is rank-deficient.
        #
        # Skip when there are no crossings: deflation requires `cell.crossings`
        # to be non-empty to produce fragments (`if tangency is True and
        # cell.crossings` below). A True result without crossings is wasted.
        # Cells with no crossings simply fall through to subdivision either way.
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
        P1_cart_local = cell.g1.surface[..., :-1] / cell.g1.surface[..., -1:]
        P2_cart_local = cell.g2.surface[..., :-1] / cell.g2.surface[..., -1:]
        local_box = ((0.0, 1.0),) * 4
        tangency = _check_tangency(
                cell.T1, cell.T2, cell.T3, cell.T4,
                P1_cart_local, P2_cart_local, local_box,
            )
        #print('tangent')


        if tangency is True :
            #print('tangent')
            # Convert crossings to the cell's local stuv for the Φ tracer.

            isol, over = _find_ssx_boundary_zeros(cell.g1.surface, cell.g2.surface, atol, rational=True)
            #print('crossings:', len(cell.crossings), 'isol:', len(isol), 'over:', len(over))
            #print('crossings:',[c.stuv.tolist() for c in cell.crossings], 'isol:',[c.stuv.tolist() for c in cell.crossings])
            #all_fragments.extend(over)
            if isol:
                fr_local, pt_local = build_tangential_branches(cell,isol,atol,all_points=all_points, all_fragments=all_fragments)



            # pt_local's SSXPoint.stuv is already global — we passed
            # `originals` so _deflate_tangent_cell copied from them.

            continue

        if cell.depth >= max_depth:
            print('max depth reached')
            for c in cell.crossings:
                all_points.append(SSXPoint(stuv=c.stuv, xyz=c.xyz))
            continue

        # --- Dual-surface subdivision ---
        # Both surfaces are split at each step. Productive crossings provide
        # per-surface split values; if a surface has no guided split, it gets
        # a midpoint cut on its longest-span axis.

        s1_axis, s1_cuts, s2_axis, s2_cuts = _compute_split_plan(
            cell.crossings, cell.box)
        ##print(s1_axis, s1_cuts, s2_axis, s2_cuts,cell.box,[nc.xyz.tolist() for nc in cell.new_crossings])
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
        ##print(s1_axis, s1_cuts, s2_axis, s2_cuts, cell.box,[nc.xyz.tolist() for nc in cell.new_crossings])
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
        ''''''
        for cut_idx, cv in enumerate(s1_cuts):
            s1_lo_box, s1_hi_box = cell.box[s1_axis]
            cut_local_s1 = (cv - s1_lo_box) / (s1_hi_box - s1_lo_box)
            isoline_s1 = _extract_isoline(cell.g1.surface, s1_local_axis, cut_local_s1)

            for s2_idx in range(n2):
                s2_piece_surf = g2_pieces[s2_idx].surface
                csx_r = bez_csx(isoline_s1, s2_piece_surf, atol=atol, rational=True)
                _require_complete_csx_result(csx_r, 'SSX S1 multi-cut face')
                ##print(csx_r)
                csx_r['isolated'] = list(csx_r['isolated'])

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
                    #tang, _, _ = _ssx_tangent_4d(
                    #    cell.g1.surface, cell.g2.surface,

                    #    stuv_local[0], stuv_local[1], stuv_local[2], stuv_local[3],
                    #    rational=True)
                    bp = BoundaryPoint(stuv=stuv, xyz=xyz, face=(s1_axis, -1), tangent_raw=None)
                    new_cx_grid[cut_idx][s2_idx].append(bp)
                    new_cx_grid[cut_idx + 1][s2_idx].append(bp)

        # c/d: CSX(cut_line_s2, S1_piece) for each S2 cut × each S1 piece
        
        for cut_idx, cv in enumerate(s2_cuts):
            s2_lo_box, s2_hi_box = cell.box[s2_axis]
            cut_local_s2 = (cv - s2_lo_box) / (s2_hi_box - s2_lo_box)
            isoline_s2 = _extract_isoline(cell.g2.surface, s2_local_axis, cut_local_s2)

            for s1_idx in range(n1):
                s1_piece_surf = g1_pieces[s1_idx].surface
                csx_r = bez_csx(isoline_s2, s1_piece_surf, atol=atol, rational=True)
                _require_complete_csx_result(csx_r, 'SSX S2 multi-cut face')
                csx_r['isolated'] = list(csx_r['isolated'])

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
                    #stuv_local = _global_to_local(stuv, cell.box)
                    #tang, _, _ = _ssx_tangent_4d(
                    #    cell.g1.surface, cell.g2.surface,
                    #    stuv_local[0], stuv_local[1], stuv_local[2], stuv_local[3],
                    #    rational=True)
                    bp = BoundaryPoint(stuv=stuv, xyz=xyz, face=(s2_axis, -1), tangent_raw=None)
                    new_cx_grid[s1_idx][cut_idx].append(bp)
                    new_cx_grid[s1_idx][cut_idx + 1].append(bp)

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

                # New crossings: deterministic from per-piece CSX grid
                sub_new_raw = new_cx_grid[i1][i2]
                ##print('sub_new_raw',[nc.xyz.tolist() for nc in sub_new_raw])
                # Dedup new against inherited
                sub_new = []
                for nc in sub_new_raw:

                    #if not any(np.linalg.norm(nc.stuv - ec.stuv) < 1e-6 for ec in sub_inherited):
                    #    if not any(np.linalg.norm(nc.stuv - dc.stuv) <  1e-6 for dc in sub_new):
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

    #all_branches = _assemble_fragments(
    #    all_fragments,
    #    S1_full=S1_for_close, S2_full=S2_for_close,
    #    atol_full=atol, rational_full=rational_close,
    #)
    #all_branches.extend(overlap_branches)
    #frags=all_fragments
    #if frags:
    #    #mcc=[interpolate_nurbs_curve(b.xyz,degree=min(len(b.xyz)-1,3), tol=1e-12) for b in frags]
   #     #mccc = [interpolate_nurbs_curve(b.stuv, degree=min(len(b.stuv) - 1, 3), tol=1e-12) for b in frags]
    closed_frags=[]
    frags=[]
    st_curves=[]
    uv_curves=[]
    xyz_curves=[]
    if all_fragments:
        for frag in all_fragments:
            if frag.build_interp():
                if frag.is_valid:
                    if frag.closed():
                        closed_frags.append(frag)
                        continue
                    frags.append(frag)

        if len(frags)>0:
            frags,_=join_bezssx_branches(frags,tol=2*atol)


        return {'branches':     frags+closed_frags , 'points': all_points}


    return {'branches': all_fragments, 'points': all_points}
