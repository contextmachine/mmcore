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
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from mmcore.numeric.bern_sq_dist import surface_surface_distance_squared_net_homog
from mmcore.numeric.intersection._bezier_common import (
    extract_weights, eval_surface, eval_surface_d1,
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
)


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


# Back-compat alias — existing code uses BoundaryCrossing in many places and
# the design §5 name is BoundaryPoint. Keep both symbols pointing at the same
# dataclass so the rename can propagate gradually.
BoundaryCrossing = BoundaryPoint


@dataclass
class BoundaryOverlap:
    """A curve segment where S1 and S2 overlap on a boundary face of [0,1]⁴."""
    stuv_start: NDArray[np.float64]  # (4,)
    stuv_end: NDArray[np.float64]    # (4,)
    face: tuple[int, int]            # (axis 0-3, side 0-1)


@dataclass
class SubdomainCell:
    """A sub-box of [0,1]⁴ produced by domain decomposition."""
    box: tuple[tuple[float, float], ...]  # 4 axis ranges
    crossings: list[BoundaryPoint] = field(default_factory=list)
    is_monotonic: bool = False
    mono_axis: Optional[int] = None


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
    bb1[0] -= atol; bb1[1] += atol
    bb2[0] -= atol; bb2[1] += atol
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


def _find_ssx_boundary_zeros(S1_h, S2_h, atol, rational=True):
    """Find all intersection points and overlaps on the boundary of [0,1]⁴.

    Returns (crossings, overlaps).
    """
    crossings = []
    overlaps = []

    def _process_face(iso, other_surf, axis, side, owner_is_s1):
        result = bez_csx(iso, other_surf, atol=atol, rational=rational)

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


def _is_on_both_boundaries(stuv, tol=1e-10):
    """Check if a point in [0,1]⁴ lies on a boundary of BOTH S1 and S2.

    S1 params: (s, t) = stuv[0:2], boundary if any is near 0 or 1.
    S2 params: (u, v) = stuv[2:4], boundary if any is near 0 or 1.

    Returns True if the point is on at least one S1 boundary AND at least one S2 boundary.
    """
    on_s1 = any(abs(stuv[i]) < tol or abs(stuv[i] - 1.0) < tol for i in (0, 1))
    on_s2 = any(abs(stuv[i]) < tol or abs(stuv[i] - 1.0) < tol for i in (2, 3))
    return on_s1 and on_s2


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
        is_dup = any(np.linalg.norm(c.stuv - d.stuv) < atol for d in deduped)
        if not is_dup:
            deduped.append(c)
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

        # Quick interval range check: if any TΨᵢ excludes 0 on the box → no tangency
        T_box = sys.T_box(Bf)
        for Ti_range in T_box:
            lo, hi = Ti_range
            if lo > 0 or hi < 0:
                return False

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

def _crossing_on_box_boundary(c, box, tol=1e-10):
    """Check if a BoundaryCrossing lies on the boundary of a sub-box."""
    stuv = c.stuv
    for axis in range(4):
        lo, hi = box[axis]
        if abs(stuv[axis] - lo) < tol or abs(stuv[axis] - hi) < tol:
            # At least one coordinate is on the boundary
            pass
        elif stuv[axis] < lo - tol or stuv[axis] > hi + tol:
            return False
    # All coordinates within [lo-tol, hi+tol]
    return True


def _crossing_in_box_interior(c, box, tol=1e-10):
    """Check if a crossing is strictly inside a box (not on its boundary)."""
    stuv = c.stuv
    for axis in range(4):
        lo, hi = box[axis]
        if stuv[axis] < lo + tol or stuv[axis] > hi - tol:
            return False
    return True


def _domain_decompose(crossings, box):
    """Subdivide the 4D box at isoparametric lines through crossing points.

    From Krishnan & Manocha (1997): place cuts at the parameter values of
    boundary crossings along the monotonic axis. Since the curve is monotonic
    in that axis, each sub-cell between two consecutive crossing values
    contains at most one curve segment.

    For 4 crossings with monotonic axis i, we sort by axis i values and
    cut between consecutive pairs. Each resulting sub-cell gets exactly the
    crossings on its boundary.

    Parameters
    ----------
    crossings : list of BoundaryCrossing
    box : tuple of (lo, hi) for 4 axes

    Returns
    -------
    list of SubdomainCell
    """
    if len(crossings) <= 2:
        return [SubdomainCell(box=box, crossings=list(crossings), is_monotonic=True)]

    # Find the best axis to cut on: the one that separates crossings most evenly.
    # For each axis, collect the crossing values and see if cuts produce good cells.
    best_axis = None
    best_score = -1

    for axis in range(4):
        vals = sorted(set(round(c.stuv[axis], 10) for c in crossings))
        # Skip if all crossings have the same value on this axis
        if len(vals) <= 1:
            continue
        # Score: number of distinct cut values (more = better separation)
        score = len(vals)
        if score > best_score:
            best_score = score
            best_axis = axis

    if best_axis is None:
        # All crossings have identical coordinates — can't decompose
        return [SubdomainCell(box=box, crossings=list(crossings), is_monotonic=True)]

    # Sort crossings by the chosen axis
    sorted_crossings = sorted(crossings, key=lambda c: c.stuv[best_axis])
    cut_values = sorted(set(round(c.stuv[best_axis], 10) for c in sorted_crossings))

    # Create sub-boxes by cutting at midpoints between consecutive crossing values
    sub_cells = []
    lo_orig, hi_orig = box[best_axis]

    # Build cut boundaries: [lo, mid01, mid12, ..., hi]
    boundaries = [lo_orig]
    for i in range(len(cut_values) - 1):
        mid = 0.5 * (cut_values[i] + cut_values[i + 1])
        boundaries.append(mid)
    boundaries.append(hi_orig)

    for i in range(len(boundaries) - 1):
        sub_lo = boundaries[i]
        sub_hi = boundaries[i + 1]
        if sub_hi - sub_lo < 1e-15:
            continue

        sub_box = list(box)
        sub_box[best_axis] = (sub_lo, sub_hi)
        sub_box = tuple(sub_box)

        # Find crossings that lie on this sub-box's boundary
        cell_crossings = [c for c in crossings if _crossing_on_box_boundary(c, sub_box, tol=1e-6)]

        sub_cells.append(SubdomainCell(
            box=sub_box,
            crossings=cell_crossings,
            is_monotonic=True,
        ))

    return sub_cells


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


def _ssx_correct(S1, S2, s, t, u, v, rational=True, max_iter=5, tol=1e-14):
    """Newton corrector: project (s,t,u,v) back onto the intersection curve.

    Minimizes ||S1(s,t) - S2(u,v)|| using damped pseudoinverse steps.
    Only a few iterations — the predictor should be close.

    Returns refined (s, t, u, v) and the residual norm.
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

    G = eval_surface(S1, s, t, rational=rational) - eval_surface(S2, u, v, rational=rational)
    return s, t, u, v, float(np.linalg.norm(G))


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
        # Check if we're close enough to the target
        dist_to_end = float(np.linalg.norm(current - target))
        if dist_to_end < max(step * 2, min_step * 4):
            # Close enough — add the endpoint and stop
            s, t, u, v, res = _ssx_correct(S1, S2, *target, rational=rational)
            if res < atol:
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
        s, t, u, v, residual = _ssx_correct(
            S1, S2, predicted[0], predicted[1], predicted[2], predicted[3],
            rational=rational,
        )

        if residual > atol:
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


def _on_boundary(stuv, tol=1e-8):
    """Check if any parameter is at 0 or 1 (on domain boundary)."""
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
    max_points=2000,
    direction_hint=None,
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
        return np.array(stuv_pts), np.array(xyz_pts)

    # Orient tangent using hint if provided
    if direction_hint is not None and np.dot(tang_prev, direction_hint) < 0:
        tang_prev = -tang_prev

    for _ in range(max_points):
        # Predictor
        predicted = current + step * tang_prev
        predicted = np.clip(predicted, 0.0, 1.0)

        # Corrector
        s, t, u, v, residual = _ssx_correct(
            S1, S2, *predicted, rational=rational,
        )

        if residual > atol:
            step = max(min_step, step * 0.5)
            continue

        corrected = np.array([s, t, u, v])
        corrected = np.clip(corrected, 0.0, 1.0)

        # New tangent
        tang_new, pt1, _ = _ssx_tangent_4d(S1, S2, *corrected, rational=rational,
                                            direction_hint=tang_prev)
        if tang_new is None:
            step = max(min_step, step * 0.5)
            continue

        if np.dot(tang_new, tang_prev) < 0:
            tang_new = -tang_new

        # Step adaptation
        cos_angle = np.clip(np.dot(tang_prev, tang_new), -1.0, 1.0)
        angle = np.arccos(abs(cos_angle))
        if angle > 1e-10:
            step = step * min(2.0, max(0.25, angle_threshold / angle))
        else:
            step = min(max_step, step * 1.5)
        step = max(min_step, min(max_step, step))

        current = corrected
        tang_prev = tang_new
        stuv_pts.append(current.copy())
        xyz_pts.append(pt1.copy())

        # Check if we hit a boundary
        if _on_boundary(current):
            break

    return np.array(stuv_pts), np.array(xyz_pts)


def _trace_segment(S1_h, S2_h, stuv_start, stuv_end, box, atol, rational=True):
    """Trace an intersection curve segment between two boundary crossings.

    Returns an SSXBranch with the traced curve, or None if tracing fails.
    """
    stuv_path, xyz_path = _march_intersection_curve(
        S1_h, S2_h, stuv_start, stuv_end,
        atol=atol, rational=rational,
    )

    if len(stuv_path) < 2:
        return None

    return SSXBranch(curve=(stuv_path, xyz_path))


# ---------------------------------------------------------------------------
# Level 5: Assemble + pair crossings for tracing
# ---------------------------------------------------------------------------

def _pair_crossings_for_tracing(crossings):
    """Pair boundary crossings for tracing.

    Each pair represents entry/exit of one curve component through the domain.
    Uses nearest-neighbor matching in 3D space.
    """
    if len(crossings) < 2:
        return [], list(range(len(crossings)))

    remaining = list(range(len(crossings)))
    pairs = []

    while len(remaining) >= 2:
        best_i, best_j = 0, 1
        best_dist = float('inf')
        for ii in range(len(remaining)):
            for jj in range(ii + 1, len(remaining)):
                ci = crossings[remaining[ii]]
                cj = crossings[remaining[jj]]
                # Don't pair crossings on the same face
                if ci.face == cj.face:
                    continue
                d = float(np.linalg.norm(ci.xyz - cj.xyz))
                if d < best_dist:
                    best_dist = d
                    best_i, best_j = ii, jj

        if best_dist == float('inf'):
            break

        pairs.append((remaining[best_i], remaining[best_j]))
        # Remove in reverse order to keep indices valid
        remaining.pop(best_j)
        remaining.pop(best_i)

    return pairs, remaining


def _process_monotonic_case(S1_h, S2_h, crossings, box, atol, rational, mono_axis=None):
    """Process a monotonic cell: sort by monotonic axis, trace consecutive pairs.

    Since the curve is monotonic in one variable, all crossings on the same
    curve component are ordered by that variable. Sorting and chaining
    consecutive pairs produces the correct topology.

    Adjacent segments are merged into a single branch.

    Returns (branches, points).
    """
    if not crossings:
        return [], []

    if len(crossings) == 1:
        return [], [SSXPoint(stuv=crossings[0].stuv, xyz=crossings[0].xyz)]

    # Sort crossings by the monotonic axis parameter
    if mono_axis is not None:
        sorted_cx = sorted(crossings, key=lambda c: c.stuv[mono_axis])
    else:
        # Fallback: sort by the axis with the widest spread
        spreads = []
        for axis in range(4):
            vals = [c.stuv[axis] for c in crossings]
            spreads.append(max(vals) - min(vals))
        best_axis = int(np.argmax(spreads))
        sorted_cx = sorted(crossings, key=lambda c: c.stuv[best_axis])

    # Trace between consecutive pairs and merge into a single branch
    all_stuv = [sorted_cx[0].stuv.copy()]
    all_xyz = [sorted_cx[0].xyz.copy()]

    for k in range(len(sorted_cx) - 1):
        stuv_a = sorted_cx[k].stuv
        stuv_b = sorted_cx[k + 1].stuv
        seg = _trace_segment(S1_h, S2_h, stuv_a, stuv_b, box, atol, rational)
        if seg is not None:
            stuv_path, xyz_path = seg.curve
            # Append all points except the first (it's the previous endpoint)
            all_stuv.extend(stuv_path[1:])
            all_xyz.extend(xyz_path[1:])
        else:
            # Tracing failed — just add the endpoint directly
            all_stuv.append(stuv_b.copy())
            all_xyz.append(sorted_cx[k + 1].xyz.copy())

    branch = SSXBranch(curve=(np.array(all_stuv), np.array(all_xyz)))
    return [branch], []


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


def _deflate_tangent_cell(P1_cart, P2_cart, T1, T2, T3, T4, box, crossings, atol):
    """Handle a confirmed-tangent cell by tracing the regulated Φ curve.

    1. Choose the best Φ = {Ψ_i, Ψ_j, TΨ_k} equations
    2. March Φ between boundary crossing pairs
    3. Filter points that are also on the full intersection (Ψ=0)

    Returns (branches, points).
    """
    from mmcore.numeric.bern import bernstein_partial_derivative_coeffs

    T_arrs = [np.asarray(T, dtype=np.float64)[..., np.newaxis] for T in [T1, T2, T3, T4]]

    branches = []
    points = []

    if len(crossings) < 2:
        for c in crossings:
            points.append(SSXPoint(stuv=c.stuv, xyz=c.xyz))
        return branches, points

    # Choose Φ equations from the first crossing
    psi_rows, t_idx = _choose_phi_equations(
        P1_cart, P2_cart, T_arrs, crossings[0].stuv, rational=False,
    )
    T_chosen = T_arrs[t_idx]

    # Pair crossings and trace Φ between each pair
    pairs, unpaired = _pair_crossings_for_tracing(crossings)

    for i, j in pairs:
        stuv_path, xyz_path = _march_phi_curve(
            P1_cart, P2_cart, T_chosen, psi_rows,
            crossings[i].stuv, crossings[j].stuv,
            atol=atol, rational=False,
        )
        if len(stuv_path) >= 2:
            # Check that points lie on the actual intersection (full Ψ=0)
            valid_mask = np.zeros(len(stuv_path), dtype=bool)
            for k in range(len(stuv_path)):
                p1 = eval_surface(P1_cart, stuv_path[k, 0], stuv_path[k, 1], rational=False)
                p2 = eval_surface(P2_cart, stuv_path[k, 2], stuv_path[k, 3], rational=False)
                if np.linalg.norm(p1 - p2) < atol:
                    valid_mask[k] = True

            if np.any(valid_mask):
                branches.append(SSXBranch(curve=(stuv_path[valid_mask], xyz_path[valid_mask])))

    for k in unpaired:
        points.append(SSXPoint(stuv=crossings[k].stuv, xyz=crossings[k].xyz))

    return branches, points


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
        branches.append(SSXBranch(curve=(stuv_path, xyz_path), overlap=True))

    # Remove sub-segment overlaps: if a shorter overlap is geometrically
    # contained within a longer one (both endpoints of the shorter lie on
    # the longer's line segment), remove the shorter.
    if len(branches) > 1:
        to_remove = set()
        for i in range(len(branches)):
            if i in to_remove:
                continue
            _, xyz_i = branches[i].curve
            a_i, b_i = xyz_i[0], xyz_i[-1]
            len_i = np.linalg.norm(b_i - a_i)
            for j in range(len(branches)):
                if i == j or j in to_remove:
                    continue
                _, xyz_j = branches[j].curve
                a_j, b_j = xyz_j[0], xyz_j[-1]
                len_j = np.linalg.norm(b_j - a_j)
                if len_i >= len_j:
                    continue  # only check if i is shorter than j
                # Check if both endpoints of i lie on segment j
                # Point p is on segment (a, b) if: |a-p| + |p-b| ≈ |a-b|
                for p in [a_i, b_i]:
                    d_ap = np.linalg.norm(p - a_j)
                    d_pb = np.linalg.norm(p - b_j)
                    if abs(d_ap + d_pb - len_j) > atol:
                        break  # This endpoint is NOT on segment j
                else:
                    # Both endpoints of i lie on segment j — i is contained
                    to_remove.add(i)
                    break
        if to_remove:
            branches = [b for k, b in enumerate(branches) if k not in to_remove]

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
# Trace all branches: marcher-driven topology discovery
# ---------------------------------------------------------------------------

def _is_cell_corner(stuv_global, box, tol=1e-8):
    """Check if a crossing is at a corner of the cell (on 2+ boundaries)."""
    n_on_boundary = 0
    for i in range(4):
        lo, hi = box[i]
        if abs(stuv_global[i] - lo) < tol or abs(stuv_global[i] - hi) < tol:
            n_on_boundary += 1
    return n_on_boundary >= 2


def _tangent_enters_cell(g1_surf, g2_surf, stuv_local, box, tol=1e-8):
    """Check if the intersection curve tangent at a corner points INTO the cell.

    For a corner crossing, the marcher would immediately leave the cell
    if the tangent points outward. Returns True if the tangent enters.
    """
    tang, _, _ = _ssx_tangent_4d(g1_surf, g2_surf, *stuv_local, rational=True)
    if tang is None:
        return False

    # Check: does a small step along the tangent stay inside [0,1]⁴?
    # Try both directions
    for sign in [1.0, -1.0]:
        stepped = stuv_local + sign * 0.01 * tang
        inside = all(0.0 - tol <= stepped[i] <= 1.0 + tol for i in range(4))
        if inside:
            return True

    return False


def _filter_corner_touches(crossings_global, g1_surf, g2_surf, box):
    """Remove crossings that merely touch a cell corner without entering.

    A crossing at a corner of the cell (on 2+ cell boundaries) may be
    a "touch point" where the intersection curve passes through but
    doesn't enter this particular sub-cell. We check by testing if
    the tangent direction points into the cell interior.
    """
    filtered = []
    for c in crossings_global:
        if not _is_cell_corner(c.stuv, box):
            filtered.append(c)
            continue

        # Corner crossing — check tangent
        stuv_local = _global_to_local(c.stuv, box)
        if _tangent_enters_cell(g1_surf, g2_surf, stuv_local, box):
            filtered.append(c)
        # else: touch point, discard

    return filtered


def _trace_all_branches(g1_surf, g2_surf, crossings_global, box, atol):
    """Trace all branches in a loop-free cell.

    Crossings are in GLOBAL coords. The marcher works in LOCAL [0,1]⁴
    on the cell's surfaces. Results are converted back to global.

    Returns (branches, points) in global coords.
    """
    if not crossings_global:
        return [], []

    # Filter corner touch points before tracing
    crossings_global = _filter_corner_touches(crossings_global, g1_surf, g2_surf, box)

    if not crossings_global:
        return [], []

    if len(crossings_global) % 2 != 0:
        import warnings
        warnings.warn(f"Odd crossing count ({len(crossings_global)})")

    branches = []
    points = []
    unvisited = list(range(len(crossings_global)))

    while unvisited:
        start_idx = unvisited.pop(0)
        start_global = crossings_global[start_idx]

        start_local = _global_to_local(start_global.stuv, box)

        stuv_local, xyz_local = _march_to_boundary(
            g1_surf, g2_surf, start_local,
            atol=atol, rational=True,
        )

        stuv_global_path = np.empty((len(stuv_local), 4), dtype=np.float64)
        for j in range(len(stuv_local)):
            stuv_global_path[j] = _local_to_global(stuv_local[j], box)
        stuv_global_path[0] = start_global.stuv.copy()

        # Find which unvisited crossing the marcher reached
        end_global = stuv_global_path[-1]
        best_k = None
        best_dist = float('inf')
        for k, idx in enumerate(unvisited):
            d = float(np.linalg.norm(end_global - crossings_global[idx].stuv))
            if d < best_dist:
                best_dist = d
                best_k = k

        if best_k is not None and best_dist < 0.1:
            matched_idx = unvisited.pop(best_k)
            stuv_global_path[-1] = crossings_global[matched_idx].stuv.copy()
            xyz_local[-1] = crossings_global[matched_idx].xyz.copy()

        # Record branch if nonzero length
        if len(stuv_global_path) >= 2 and np.linalg.norm(stuv_global_path[-1] - stuv_global_path[0]) > 1e-10:
            branches.append(SSXBranch(curve=(stuv_global_path, xyz_local)))

        # Remove remaining crossings that the branch passed through.
        # These are partition touch-points: on a cell corner (2+ boundary faces)
        # and near the traced path. They don't form separate branches.
        still_unvisited = []
        for idx in unvisited:
            c = crossings_global[idx]
            if not _is_cell_corner(c.stuv, box):
                still_unvisited.append(idx)
                continue
            # Corner crossing — check if it's near the traced path
            near_path = any(
                np.linalg.norm(c.xyz - xyz_local[j]) < atol * 5
                for j in range(len(xyz_local))
            )
            if not near_path:
                still_unvisited.append(idx)
        unvisited = still_unvisited

    return branches, points


# ---------------------------------------------------------------------------
# Domain decomposition helpers
# ---------------------------------------------------------------------------

def _choose_cut(crossings_global, box):
    """Choose a crossing and axis to cut through.

    Crossings are in GLOBAL coords. We look for an interior value
    (not on the cell boundary) that separates crossings into balanced groups.

    Returns (crossing_index, axis) or (None, None).
    """
    if len(crossings_global) <= 2:
        return None, None

    best_score = -1.0
    best_cx_idx = None
    best_axis = None

    for ci, c in enumerate(crossings_global):
        for axis in range(4):
            val = c.stuv[axis]
            lo, hi = box[axis]
            # Skip if the value is on the cell boundary
            if abs(val - lo) < 1e-8 or abs(val - hi) < 1e-8:
                continue
            if val <= lo or val >= hi:
                continue

            n_left = sum(1 for c2 in crossings_global if c2.stuv[axis] < val - 1e-10)
            n_right = sum(1 for c2 in crossings_global if c2.stuv[axis] > val + 1e-10)
            n_on = len(crossings_global) - n_left - n_right

            balance = min(n_left + n_on, n_right + n_on)
            if balance == 0:
                continue

            score = balance - 0.1 * n_on
            if score > best_score:
                best_score = score
                best_cx_idx = ci
                best_axis = axis

    return best_cx_idx, best_axis


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


def _build_cell_partitions(owner_cell: "_Cell") -> list[PartitionCurve]:
    """Create the 8 partitions corresponding to a cell's 8 box faces.

    Each partition is the isoline fixing one of the cell's axes at its lower
    or upper box bound; the free axis is the owning surface's other axis,
    with extent equal to the cell's box range on that axis.

    For the top-level cell (box = [0,1]⁴) this yields the 8 outer faces of
    the full parameter domain; for a sub-cell it yields its own 8 faces in
    global coordinates. Partitions are unshared at this stage — cross-cell
    matching across internal partitions is handled in a later iteration.
    """
    parts: list[PartitionCurve] = []
    for axis in range(4):
        free = _partition_free_axis(axis)
        extent = owner_cell.box[free]
        for side_idx in (0, 1):
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


@dataclass
class _Cell:
    """A sub-problem in the domain decomposition stack."""
    g1: object                          # GaussMapBern for S1 sub-patch (local [0,1]²)
    g2: object                          # GaussMapBern for S2 sub-patch (local [0,1]²)
    crossings: list                     # BoundaryPoint in GLOBAL coords
    box: tuple                          # 4D parameter range in GLOBAL coords
    depth: int = 0
    # TΨᵢ Bernstein tensors for this sub-cell's local [0,1]⁴ — propagated by
    # de Casteljau-splitting the parent's tensors along the cut axis (never
    # recomputed; see design §1.2).
    T1: Optional[NDArray[np.float64]] = None
    T2: Optional[NDArray[np.float64]] = None
    T3: Optional[NDArray[np.float64]] = None
    T4: Optional[NDArray[np.float64]] = None
    # Isolines bounding this cell (design §5). Top-level cell owns 8 outer
    # partitions; sub-cells will inherit/create their own in a later iteration.
    partitions: list[PartitionCurve] = field(default_factory=list)


def bez_ssx(
    S1,
    S2,
    atol=1e-3,
    rational=True,
    max_depth=12,
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
    # Crossings are already in global [0,1]⁴ coords (top-level box is [0,1]⁴)
    crossings, boundary_overlaps = _find_ssx_boundary_zeros(S1, S2, atol, rational=rational)
    overlap_branches = _overlaps_to_branches(boundary_overlaps, S1, atol, rational)

    # Filter crossings that coincide with overlap endpoints
    if overlap_branches:
        filtered = []
        for c in crossings:
            on_ovl = any(
                np.linalg.norm(c.xyz - b.curve[1][0]) < atol or
                np.linalg.norm(c.xyz - b.curve[1][-1]) < atol
                for b in overlap_branches
            )
            if not on_ovl:
                filtered.append(c)
        crossings = filtered

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
    top_cell = _Cell(
        g1=g1, g2=g2, crossings=crossings, box=box, depth=0,
        T1=T1_arr, T2=T2_arr, T3=T3_arr, T4=T4_arr,
    )
    top_cell.partitions = _build_outer_partitions(top_cell)

    # §4 classification: one IsolineRegistration per on-boundary axis per
    # boundary crossing. Still produced unconditionally — consumers land in
    # later iterations.
    for c in crossings:
        _classify_boundary_point(c, top_cell)

    # --- Loop-absence check (top level) ---
    if _check_loop_free(g1, g2, T1, T2, T3, T4):
        if not crossings and not overlap_branches:
            return {'branches': [], 'points': []}
        branches, points = _trace_all_branches(g1.surface, g2.surface, crossings, box, atol)
        branches.extend(overlap_branches)
        return {'branches': branches, 'points': points}

    # --- ITERATIVE DOMAIN DECOMPOSITION ---
    stack = [top_cell]
    all_branches = list(overlap_branches)
    all_points = []

    while stack:
        cell = stack.pop()

        # Loop-absence on this sub-cell — TΨᵢ monotonicity (cheap) tried first,
        # Gauss map separability as fallback (design §6, §10 principle 8).
        if _check_loop_free(cell.g1, cell.g2,
                            cell.T1, cell.T2, cell.T3, cell.T4):
            if cell.crossings:
                br, pt = _trace_all_branches(
                    cell.g1.surface, cell.g2.surface,
                    cell.crossings, cell.box, atol,
                )
                all_branches.extend(br)
                all_points.extend(pt)
            continue

        if cell.depth >= max_depth:
            for c in cell.crossings:
                all_points.append(SSXPoint(stuv=c.stuv, xyz=c.xyz))
            continue

        # --- Choose cut: through a crossing's parameter value ---
        cx_idx, cut_axis = _choose_cut(cell.crossings, cell.box)

        if cx_idx is None:
            # Can't cut — trace directly
            if cell.crossings:
                br, pt = _trace_all_branches(
                    cell.g1.surface, cell.g2.surface,
                    cell.crossings, cell.box, atol,
                )
                all_branches.extend(br)
                all_points.extend(pt)
            continue

        cut_global_val = cell.crossings[cx_idx].stuv[cut_axis]

        # Convert cut to LOCAL parameter for the surface being split
        cell_lo, cell_hi = cell.box[cut_axis]
        cut_local = (cut_global_val - cell_lo) / max(cell_hi - cell_lo, 1e-15)
        cut_local = max(0.01, min(0.99, cut_local))  # safety clamp

        # Which surface to split
        surf_to_split = 1 if cut_axis < 2 else 2
        local_axis = cut_axis if cut_axis < 2 else cut_axis - 2

        # Extract isoline at LOCAL param, run CSX BEFORE splitting
        if surf_to_split == 1:
            isoline = _extract_isoline(cell.g1.surface, local_axis, cut_local)
            csx_result = bez_csx(isoline, cell.g2.surface, atol=atol, rational=True)
        else:
            isoline = _extract_isoline(cell.g2.surface, local_axis, cut_local)
            csx_result = bez_csx(isoline, cell.g1.surface, atol=atol, rational=True)

        # Convert CSX results to global crossings. Pass the cell's local
        # homogeneous surface nets so the raw 4D tangent (§4) is populated.
        new_crossings = _isoline_csx_to_global(
            csx_result, cut_axis, cut_global_val, cell.box, surf_to_split,
            S1_local=cell.g1.surface, S2_local=cell.g2.surface, rational=True,
        )

        # Split GaussMapBern at LOCAL param
        if surf_to_split == 1:
            g1_L, g1_R = (cell.g1.split_u(cut_local) if local_axis == 0
                          else cell.g1.split_v(cut_local))
            g2_L = g2_R = cell.g2
        else:
            g2_L, g2_R = (cell.g2.split_u(cut_local) if local_axis == 0
                          else cell.g2.split_v(cut_local))
            g1_L = g1_R = cell.g1

        # Propagate TΨᵢ to sub-cells by de Casteljau on the T tensors along
        # the same cut_axis and cut_local parameter (design §1.2, §6).
        T1_L, T1_R = _split_bern_scalar_tensor(cell.T1, axis=cut_axis, t=cut_local)
        T2_L, T2_R = _split_bern_scalar_tensor(cell.T2, axis=cut_axis, t=cut_local)
        T3_L, T3_R = _split_bern_scalar_tensor(cell.T3, axis=cut_axis, t=cut_local)
        T4_L, T4_R = _split_bern_scalar_tensor(cell.T4, axis=cut_axis, t=cut_local)

        # Distribute crossings (all in GLOBAL coords) to left/right
        all_cx = list(cell.crossings) + new_crossings
        left_cx = []
        right_cx = []
        for c in all_cx:
            v = c.stuv[cut_axis]
            if v < cut_global_val - 1e-10:
                left_cx.append(c)
            elif v > cut_global_val + 1e-10:
                right_cx.append(c)
            else:
                # On the cut — belongs to both sides
                left_cx.append(c)
                right_cx.append(c)

        # Sub-boxes in GLOBAL coords
        box_L = list(cell.box)
        box_L[cut_axis] = (cell.box[cut_axis][0], cut_global_val)
        box_L = tuple(box_L)

        box_R = list(cell.box)
        box_R[cut_axis] = (cut_global_val, cell.box[cut_axis][1])
        box_R = tuple(box_R)

        if left_cx:
            L_cell = _Cell(g1=g1_L, g2=g2_L, crossings=left_cx,
                           box=box_L, depth=cell.depth + 1,
                           T1=T1_L, T2=T2_L, T3=T3_L, T4=T4_L)
            L_cell.partitions = _build_cell_partitions(L_cell)
            for c in left_cx:
                _classify_boundary_point(c, L_cell)
            stack.append(L_cell)
        if right_cx:
            R_cell = _Cell(g1=g1_R, g2=g2_R, crossings=right_cx,
                           box=box_R, depth=cell.depth + 1,
                           T1=T1_R, T2=T2_R, T3=T3_R, T4=T4_R)
            R_cell.partitions = _build_cell_partitions(R_cell)
            for c in right_cx:
                _classify_boundary_point(c, R_cell)
            stack.append(R_cell)

    # --- Post-processing: merge + dedup ---
    all_branches = _merge_adjacent_branches(all_branches, atol)
    all_branches = _dedup_branches(all_branches, atol)
    return {'branches': all_branches, 'points': all_points}


def _merge_adjacent_branches(branches, atol):
    """Merge branches whose endpoints match (from adjacent sub-cells sharing a partition)."""
    if len(branches) <= 1:
        return branches

    merged = True
    while merged:
        merged = False
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                stuv_i, xyz_i = branches[i].curve
                stuv_j, xyz_j = branches[j].curve
                if len(stuv_i) < 1 or len(stuv_j) < 1:
                    continue

                # Check if end of i matches start of j
                if np.linalg.norm(stuv_i[-1] - stuv_j[0]) < atol:
                    new_stuv = np.concatenate([stuv_i, stuv_j[1:]], axis=0)
                    new_xyz = np.concatenate([xyz_i, xyz_j[1:]], axis=0)
                    branches[i] = SSXBranch(curve=(new_stuv, new_xyz))
                    branches.pop(j)
                    merged = True
                    break
                # Check if end of j matches start of i
                if np.linalg.norm(stuv_j[-1] - stuv_i[0]) < atol:
                    new_stuv = np.concatenate([stuv_j, stuv_i[1:]], axis=0)
                    new_xyz = np.concatenate([xyz_j, xyz_i[1:]], axis=0)
                    branches[i] = SSXBranch(curve=(new_stuv, new_xyz))
                    branches.pop(j)
                    merged = True
                    break
                # Check if start matches start (reverse one)
                if np.linalg.norm(stuv_i[0] - stuv_j[0]) < atol:
                    new_stuv = np.concatenate([stuv_i[::-1], stuv_j[1:]], axis=0)
                    new_xyz = np.concatenate([xyz_i[::-1], xyz_j[1:]], axis=0)
                    branches[i] = SSXBranch(curve=(new_stuv, new_xyz))
                    branches.pop(j)
                    merged = True
                    break
                # Check if end matches end (reverse one)
                if np.linalg.norm(stuv_i[-1] - stuv_j[-1]) < atol:
                    new_stuv = np.concatenate([stuv_i, stuv_j[-2::-1]], axis=0)
                    new_xyz = np.concatenate([xyz_i, xyz_j[-2::-1]], axis=0)
                    branches[i] = SSXBranch(curve=(new_stuv, new_xyz))
                    branches.pop(j)
                    merged = True
                    break
            if merged:
                break

    return branches


def _dedup_branches(branches, atol):
    """Remove duplicate branches (same start AND end points)."""
    if len(branches) <= 1:
        return branches

    to_remove = set()
    for i in range(len(branches)):
        if i in to_remove:
            continue
        stuv_i, xyz_i = branches[i].curve
        if len(stuv_i) < 2:
            continue
        for j in range(i + 1, len(branches)):
            if j in to_remove:
                continue
            stuv_j, xyz_j = branches[j].curve
            if len(stuv_j) < 2:
                continue
            # Same direction
            same = (np.linalg.norm(xyz_i[0] - xyz_j[0]) < atol and
                    np.linalg.norm(xyz_i[-1] - xyz_j[-1]) < atol)
            # Reversed
            rev = (np.linalg.norm(xyz_i[0] - xyz_j[-1]) < atol and
                   np.linalg.norm(xyz_i[-1] - xyz_j[0]) < atol)
            if same or rev:
                # Keep the one with more points
                if len(stuv_j) > len(stuv_i):
                    to_remove.add(i)
                else:
                    to_remove.add(j)

    if to_remove:
        branches = [b for k, b in enumerate(branches) if k not in to_remove]

    # Remove zero-length branches
    branches = [b for b in branches if len(b.curve[0]) > 1 and
                np.linalg.norm(b.curve[1][0] - b.curve[1][-1]) > atol]

    return branches
