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
            crossings.append(BoundaryPoint(stuv=stuv, xyz=xyz, face=(face_id, side)))

        for ovl in result.get('overlaps', []):
            tr = ovl.get('t_range', (0.0, 1.0))
            ur = ovl.get('u_range', (0.0, 1.0))
            vr = ovl.get('v_range', (0.0, 1.0))
            stuv_s = _map_csx_to_stuv(axis, side, tr[0], ur[0], vr[0], owner_is_s1)
            stuv_e = _map_csx_to_stuv(axis, side, tr[1], ur[1], vr[1], owner_is_s1)
            face_id = axis if owner_is_s1 else axis + 2
            overlaps.append(BoundaryOverlap(stuv_start=stuv_s, stuv_end=stuv_e,
                                            face=(face_id, side)))
            # Also add endpoints as crossings (they connect to interior branches);
            # tangent_raw is populated later by _classify_boundary_point via the
            # cofactor formula (design §4.1).
            xyz_s = eval_surface(S1_h, stuv_s[0], stuv_s[1], rational=rational)
            xyz_e = eval_surface(S1_h, stuv_e[0], stuv_e[1], rational=rational)
            crossings.append(BoundaryPoint(stuv=stuv_s, xyz=xyz_s, face=(face_id, side)))
            crossings.append(BoundaryPoint(stuv=stuv_e, xyz=xyz_e, face=(face_id, side)))

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


# ---------------------------------------------------------------------------
# Φ-tracer crossing pairing — used only inside _deflate_tangent_cell (§8)
# ---------------------------------------------------------------------------

def _pair_crossings_for_tracing(crossings, originals, cell):
    """Pair boundary crossings for Φ tracing (design §8).

    Pairs design-§5 "in" registrations with "out" registrations in the
    owning cell's view. A through-touch crossing (in on some axes, out on
    others in the same cell) counts as an "in" once, pairable with any
    remaining "out".

    When the cell is tangent (C₂) the cofactor tangent is identically zero
    on the intersection curve, classification produces no registrations,
    and this function returns `(pairs=[], unpaired=all)`. That is the
    correct signal that a Φ-side classifier (design §4.2 deferred) is
    required. No heuristic fallback.

    Returns `(pairs, unpaired)` with `pairs` a list of `(i, j)` index
    tuples into `crossings`.
    """
    n = len(crossings)
    if n < 2:
        return [], list(range(n))

    in_ids: list[int] = []
    out_ids: list[int] = []
    for idx, orig in enumerate(originals):
        cell_regs = [r for r in orig.registrations if r.owner is cell]
        has_in = any(r.direction == "in" for r in cell_regs)
        has_out = any(r.direction == "out" for r in cell_regs)
        if has_in:
            in_ids.append(idx)
        if has_out and not has_in:
            out_ids.append(idx)

    remaining_out = list(out_ids)
    pairs: list[tuple[int, int]] = []
    for i in in_ids:
        if not remaining_out:
            break
        j = min(
            remaining_out,
            key=lambda k: float(np.linalg.norm(crossings[i].stuv - crossings[k].stuv)),
        )
        pairs.append((i, j))
        remaining_out.remove(j)
    paired_ids = {i for p in pairs for i in p}
    unpaired = [k for k in range(n) if k not in paired_ids]
    return pairs, unpaired


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


def _deflate_tangent_cell(P1_cart, P2_cart, T1, T2, T3, T4, box, crossings, atol,
                          *, originals, cell):
    """Handle a confirmed-tangent cell by tracing the regulated Φ curve.

    1. Choose the best Φ = {Ψ_i, Ψ_j, TΨ_k} equations.
    2. Pair crossings by in/out registrations in the owning cell
       (design §4 / §8); if the cell's registrations are empty (C₂ whole-
       curve-tangent case), no pairs are produced — this surfaces the
       design §4.2 deferred work (Φ-side classifier) rather than falling
       back to a heuristic.
    3. March Φ between each pair. Filter points that are also on the full
       intersection (Ψ = 0).

    Returns `(fragments, points)`. Fragments carry `start_point` /
    `end_point` references to `originals[i]` / `originals[j]` so the §9
    assembly can chain Φ-fragments alongside Ψ-fragments.
    """
    T_arrs = [np.asarray(T, dtype=np.float64)[..., np.newaxis] for T in [T1, T2, T3, T4]]

    fragments: list[_Fragment] = []
    points: list[SSXPoint] = []

    if len(crossings) < 2:
        for c in originals:
            points.append(SSXPoint(stuv=c.stuv, xyz=c.xyz))
        return fragments, points

    psi_rows, t_idx = _choose_phi_equations(
        P1_cart, P2_cart, T_arrs, crossings[0].stuv, rational=False,
    )
    T_chosen = T_arrs[t_idx]

    pairs, unpaired = _pair_crossings_for_tracing(crossings, originals, cell)

    for i, j in pairs:
        stuv_path, xyz_path = _march_phi_curve(
            P1_cart, P2_cart, T_chosen, psi_rows,
            crossings[i].stuv, crossings[j].stuv,
            atol=atol, rational=False,
        )
        if len(stuv_path) < 2:
            continue
        valid_mask = np.zeros(len(stuv_path), dtype=bool)
        for k in range(len(stuv_path)):
            p1 = eval_surface(P1_cart, stuv_path[k, 0], stuv_path[k, 1], rational=False)
            p2 = eval_surface(P2_cart, stuv_path[k, 2], stuv_path[k, 3], rational=False)
            if np.linalg.norm(p1 - p2) < atol:
                valid_mask[k] = True
        if not np.any(valid_mask):
            continue
        fragments.append(_Fragment(
            start_point=originals[i], end_point=originals[j],
            stuv_path=stuv_path[valid_mask],
            xyz_path=xyz_path[valid_mask],
            owner_cell=cell,
        ))

    for k in unpaired:
        points.append(SSXPoint(stuv=originals[k].stuv, xyz=originals[k].xyz))

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
        branches.append(SSXBranch(curve=(stuv_path, xyz_path), overlap=True))

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

def _find_exit_registration(cell, stuv_end, tol_param):
    """Design §7 Invariant D: locate the unique unconsumed "out" registration
    owned by `cell` that matches the marcher's stopping point.

    The marcher is guaranteed to stop on the cell's boundary (it clamps to
    `[0,1]⁴` in local coords); therefore `stuv_end` must have at least one
    on-boundary axis for this cell. We walk every matching partition on
    every on-boundary axis and return the best unconsumed out-registration
    whose `param` matches `stuv_end[free_axis]` within `tol_param`
    (design §9 Invariant B: residual is numerical-noise-sized, ≤ atol).
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

    `start_point` / `end_point` are the actual `BoundaryPoint` objects
    (shared across adjacent cells' views when the point sits on a shared
    partition). `owner_cell` is the cell whose interior was marched to
    produce `stuv_path` / `xyz_path` — §9 adjacency walk uses it to cross
    to the next cell.
    """
    start_point: Optional[BoundaryPoint]
    end_point: Optional[BoundaryPoint]
    stuv_path: NDArray[np.float64]
    xyz_path: NDArray[np.float64]
    owner_cell: Optional["_Cell"] = None


def _trace_cell_by_registrations(cell, atol):
    """Design §7: march each unique `in`-registered BoundaryPoint of `cell`
    through the cell's interior, producing one fragment per starting point.

    Registration consumption is NOT performed here — the §9 adjacency walk
    in `_assemble_branches_by_adjacency` consumes them as it chains. The
    dedup by `id(point)` is sufficient to avoid marching the same start
    point twice within one cell.

    Returns `(fragments, points)`. Each fragment carries its `start_point`
    and (when the marcher's stopping point matched a registration) its
    `end_point`, plus `owner_cell` — all needed by the adjacency walk.
    """
    fragments: list[_Fragment] = []
    points: list = []

    start_points: list[BoundaryPoint] = []
    seen: set[int] = set()
    for p in cell.partitions:
        for reg in p.registrations:
            if reg.owner is cell and reg.direction == "in":
                key = id(reg.point)
                if key in seen:
                    continue
                seen.add(key)
                start_points.append(reg.point)

    for start_point in start_points:
        start_local = _global_to_local(start_point.stuv, cell.box)
        # Design §4.1: the cofactor tangent (cached on start_point by
        # _classify_boundary_point) has a consistent sign — at an "in"
        # registration it points INTO the cell by construction.
        hint = start_point.tangent_raw
        stuv_local, xyz_local = _march_to_boundary(
            cell.g1.surface, cell.g2.surface, start_local,
            atol=atol, rational=True,
            direction_hint=hint,
        )

        if len(stuv_local) < 2:
            continue

        stuv_global = np.empty((len(stuv_local), 4), dtype=np.float64)
        for j in range(len(stuv_local)):
            stuv_global[j] = _local_to_global(stuv_local[j], cell.box)
        stuv_global[0] = start_point.stuv.copy()

        end_reg = _find_exit_registration(cell, stuv_global[-1], tol_param=atol)
        end_point: Optional[BoundaryPoint] = None
        if end_reg is not None:
            end_point = end_reg.point
            stuv_global[-1] = end_point.stuv.copy()
            xyz_local[-1] = end_point.xyz.copy()

        if np.linalg.norm(stuv_global[-1] - stuv_global[0]) <= 1e-10:
            continue  # zero-length through-touch; no fragment recorded,
            # the §9 walk handles it via registrations alone.

        fragments.append(_Fragment(
            start_point=start_point,
            end_point=end_point,
            stuv_path=stuv_global,
            xyz_path=xyz_local,
            owner_cell=cell,
        ))

    return fragments, points


def _assemble_branches_by_adjacency(all_fragments: list[_Fragment],
                                    all_cells: list["_Cell"]) -> list[SSXBranch]:
    """Design §9: walk the partition adjacency graph to build full branches.

    A chain extends one step at a time:
      1. In the current cell, find the exit out-reg for the current point.
         If the point has an unconsumed `"out"` reg on a partition different
         from the entry partition, that's a through-touch — no fragment.
         Otherwise look up the pre-traced fragment that starts at this point
         in this cell and take its stuv_path + xyz_path; its end_point
         defines the new current point.
      2. Cross the exit partition: find the other adjacent cell and locate
         the matching `"in"` reg on that same partition at that same point.
         If the partition is an outer face (one adjacent only), the chain
         ends.

    All registrations are consumed as the walk traverses them — the
    pre-tracer does NOT consume; consumption happens exclusively here.

    Primary chain starts are every unconsumed `"in"` reg on an outer face
    (a partition with `len(adjacents) == 1`). A second pass iterates any
    remaining unconsumed `"in"` regs, which by construction seed closed
    branches (branches entirely interior to `[0,1]⁴`).
    """
    frag_by_start: dict[tuple[int, int], _Fragment] = {}
    for f in all_fragments:
        if f.start_point is not None and f.owner_cell is not None:
            frag_by_start[(id(f.owner_cell), id(f.start_point))] = f

    branches: list[SSXBranch] = []
    # Global set of BoundaryPoint ids already covered by some emitted chain.
    # A fresh walk whose start_point is already in here would only retrace a
    # curve another chain already captured — skip to avoid duplicates. This
    # is the "don't re-emit the same branch via a sibling cell" rule.
    emitted_point_ids: set[int] = set()

    def _consume_all(point, cell, direction: str) -> None:
        """A multi-axis entry or exit (curve entering/leaving a cell via
        several faces at a corner) is one physical event. All same-
        direction registrations at that (cell, point) describe that one
        event and are consumed together.
        """
        for r in point.registrations:
            if r.owner is cell and r.direction == direction:
                r.consumed = True

    def _walk(start_reg: IsolineRegistration) -> tuple[Optional[SSXBranch], set[int]]:
        if start_reg.consumed:
            return None, set()
        if id(start_reg.point) in emitted_point_ids:
            # Another chain already covers this point — consume this cell's
            # in-regs at the point so later iterations don't keep looking
            # here, but produce no branch.
            _consume_all(start_reg.point, start_reg.owner, "in")
            return None, set()

        current_cell = start_reg.owner
        current_point = start_reg.point
        entry_partition = start_reg.partition
        _consume_all(current_point, current_cell, "in")

        stuv_pieces: list[np.ndarray] = []
        xyz_pieces: list[np.ndarray] = []
        visited_point_ids: set[int] = {id(current_point)}

        while True:
            # Phase A: find an "out" reg at current_point in current_cell.
            # A through-touch surfaces here as an out reg already at the
            # entry point; a normal march yields a new end point Y from
            # the pre-traced fragment and we look for the out at Y.
            exit_reg: Optional[IsolineRegistration] = None
            for r in current_point.registrations:
                if r.owner is current_cell and r.direction == "out" and not r.consumed:
                    exit_reg = r
                    break

            if exit_reg is None:
                frag = frag_by_start.get((id(current_cell), id(current_point)))
                if frag is None:
                    return _concat(stuv_pieces, xyz_pieces), visited_point_ids
                stuv_pieces.append(frag.stuv_path)
                xyz_pieces.append(frag.xyz_path)
                Y = frag.end_point
                if Y is None:
                    return _concat(stuv_pieces, xyz_pieces), visited_point_ids
                current_point = Y
                for r in current_point.registrations:
                    if r.owner is current_cell and r.direction == "out" and not r.consumed:
                        exit_reg = r
                        break
                if exit_reg is None:
                    return _concat(stuv_pieces, xyz_pieces), visited_point_ids

            _consume_all(current_point, current_cell, "out")
            exit_partition = exit_reg.partition

            # Phase B: cross the partition.
            if len(exit_partition.adjacents) < 2:
                return _concat(stuv_pieces, xyz_pieces), visited_point_ids  # outer face

            next_cell = None
            for adj in exit_partition.adjacents:
                if adj is not current_cell:
                    next_cell = adj
                    break
            if next_cell is None:
                return _concat(stuv_pieces, xyz_pieces), visited_point_ids

            # Require a matching "in" reg on the same partition at this point.
            has_match = any(
                r.owner is next_cell and r.direction == "in"
                and r.point is current_point and not r.consumed
                for r in exit_partition.registrations
            )
            if not has_match:
                return _concat(stuv_pieces, xyz_pieces), visited_point_ids

            _consume_all(current_point, next_cell, "in")
            current_cell = next_cell
            entry_partition = exit_partition

            if id(current_point) in visited_point_ids:
                return _concat(stuv_pieces, xyz_pieces), visited_point_ids
            visited_point_ids.add(id(current_point))

    def _concat(stuv_pieces, xyz_pieces) -> Optional[SSXBranch]:
        if not stuv_pieces:
            return None
        # Adjacent pieces share an endpoint — drop the duplicated sample.
        stuv_out = [stuv_pieces[0]]
        xyz_out = [xyz_pieces[0]]
        for k in range(1, len(stuv_pieces)):
            stuv_out.append(stuv_pieces[k][1:])
            xyz_out.append(xyz_pieces[k][1:])
        stuv_full = np.concatenate(stuv_out, axis=0)
        xyz_full = np.concatenate(xyz_out, axis=0)
        return SSXBranch(curve=(stuv_full, xyz_full))

    def _try_walk(reg: IsolineRegistration) -> None:
        br, visited = _walk(reg)
        if br is not None:
            branches.append(br)
            emitted_point_ids.update(visited)

    # Primary chains: outer-face "in" regs (partition with one adjacent).
    for cell in all_cells:
        for p in cell.partitions:
            if len(p.adjacents) != 1:
                continue
            for reg in list(p.registrations):
                if (reg.direction == "in" and reg.owner is cell
                        and not reg.consumed):
                    _try_walk(reg)

    # Secondary: remaining unconsumed "in" regs — closed/interior branches.
    for cell in all_cells:
        for p in cell.partitions:
            for reg in list(p.registrations):
                if (reg.direction == "in" and reg.owner is cell
                        and not reg.consumed):
                    _try_walk(reg)

    return branches


# ---------------------------------------------------------------------------
# Domain decomposition helpers
# ---------------------------------------------------------------------------

def _choose_cut(crossings_global, box, min_margin: float = 0.05):
    """Choose a crossing and axis for the next subdivision cut.

    Design principle §10.4: cuts go *through* an existing crossing's parameter
    value, never at a midpoint. We prefer the crossing whose local cut position
    is closest to the cell's center — a cut close to the cell boundary would
    produce a sub-patch of near-zero width, which is useless for further
    subdivision (we'd just keep cutting down the same sliver).

    If every candidate crossing sits within `min_margin` of the cell's
    boundary on every axis (e.g. all crossings are at local corners), no
    productive cut exists and we return `(None, None)`; the caller should
    stop subdividing this cell.

    Returns `(crossing_index, axis)` or `(None, None)`.
    """
    if len(crossings_global) <= 2:
        return None, None

    best_center_dist = float("inf")
    best_cx_idx = None
    best_axis = None

    for ci, c in enumerate(crossings_global):
        for axis in range(4):
            val = c.stuv[axis]
            lo, hi = box[axis]
            span = hi - lo
            if span <= 0:
                continue
            local = (val - lo) / span
            # Reject cuts too close to the cell's own boundaries — they'd
            # produce a near-zero-width sub-patch.
            if local < min_margin or local > 1.0 - min_margin:
                continue
            center_dist = abs(local - 0.5)
            if center_dist < best_center_dist:
                best_center_dist = center_dist
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

        # Tangent is populated by _classify_boundary_point via the cofactor
        # formula (design §4.1); no SVD at construction.
        crossings.append(BoundaryPoint(stuv=stuv_global, xyz=xyz, face=(cut_axis, -1)))

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


def _ssx_tangent_cofactor(T1, T2, T3, T4, stuv):
    """Design §4.1: raw 4D tangent via the cofactor / adjugate formula.

    The null vector of the 3×4 Jacobian `J_Ψ` is the cofactor column:
    `T[i] = (−1)^i · det(J_Ψ without column i)`. The minors are exactly
    the `TΨᵢ` Bernstein tensors (design §1.2), so the tangent is just
    those four tensors evaluated at `stuv` with alternating signs.

    Sign is a fixed function of the surface pair, not of the solver,
    so `(local_param, sign(T[i]))` classification is consistent across
    every cell and every crossing. Zero at tangent points (Ψ = 0 AND
    TΨ = 0, design §1.4) — such points get no registration and cannot
    seed a march; the Krawczyk / Φ path handles them.
    """
    from mmcore.numeric.bern import bernstein_eval_nd
    params = np.asarray(stuv, dtype=np.float64)
    vals = []
    for Ti in (T1, T2, T3, T4):
        arr = np.asarray(Ti, dtype=np.float64)[..., None]
        vals.append(bernstein_eval_nd(arr, params).item())
    return np.array([vals[0], -vals[1], vals[2], -vals[3]], dtype=np.float64)


def _build_cell_partitions(owner_cell: "_Cell",
                           skip: Optional[tuple[int, int]] = None) -> list[PartitionCurve]:
    """Create the partitions corresponding to a cell's 8 box faces.

    Each partition is the isoline fixing one of the cell's axes at its lower
    or upper box bound; the free axis is the owning surface's other axis,
    with extent equal to the cell's box range on that axis.

    If `skip` is provided as `(axis, side_idx)` that one face is omitted —
    used when the caller will splice in a shared internal partition in its
    place (design §5 invariants: an internal partition's object is unique
    and adjacent to exactly two cells).
    """
    parts: list[PartitionCurve] = []
    for axis in range(4):
        free = _partition_free_axis(axis)
        extent = owner_cell.box[free]
        for side_idx in (0, 1):
            if skip is not None and skip == (axis, side_idx):
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
    """Design §4 + §4.1: produce one IsolineRegistration per on-boundary axis,
    using the cofactor tangent evaluated in the cell's local frame.

    Tangent signs are a fixed function of `(S1, S2)` (not of a solver), and
    are invariant under the positive affine global↔local rescale — every
    cell agrees on the direction the curve is moving at `point`.
    """
    if cell.T1 is None:
        return

    local_stuv = _global_to_local(point.stuv, cell.box)
    tangent = _ssx_tangent_cofactor(cell.T1, cell.T2, cell.T3, cell.T4, local_stuv)
    # Cache on the point for later reuse (e.g. marcher direction hint).
    point.tangent_raw = tangent

    for i in range(4):
        local_param = _on_axis_local(point.stuv[i], cell.box[i][0], cell.box[i][1])
        if local_param is None:
            continue  # axis strictly interior for this cell — no registration (§4)

        direction = _classify_on_axis(local_param, float(tangent[i]))
        if direction is None:
            continue  # tangent exactly orthogonal to axis — at-or-near tangent point

        target_value = cell.box[i][local_param]
        match = None
        for p in cell.partitions:
            if p.axis == i and abs(p.value - target_value) < 1e-8:
                match = p
                break
        if match is None:
            continue

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
    top_cell = _Cell(
        g1=g1, g2=g2, crossings=crossings, box=box, depth=0,
        T1=T1_arr, T2=T2_arr, T3=T3_arr, T4=T4_arr,
    )
    top_cell.partitions = _build_outer_partitions(top_cell)

    # §4 classification: one IsolineRegistration per on-boundary axis per
    # boundary crossing.
    for c in crossings:
        _classify_boundary_point(c, top_cell)

    if not crossings and not overlap_branches:
        return {'branches': [], 'points': []}

    # --- Iterative domain decomposition (single code path, design §6) ---
    # The top-level cell enters the same stack as any sub-cell and goes
    # through the same 4-step lifecycle: cheap certificates → tangency →
    # subdivision. If it's loop-free at top level, the first iteration
    # traces it and the loop exits.
    stack = [top_cell]
    # `all_cells` is the list of every _Cell ever created — needed by the
    # §9 adjacency walk because cells are not reachable from `top_cell` via
    # partition-adjacency alone (top_cell's outer partitions aren't shared
    # with its children).
    all_cells: list[_Cell] = [top_cell]
    all_fragments: list[_Fragment] = []
    all_points = []

    while stack:
        cell = stack.pop()

        # Loop-absence on this sub-cell — TΨᵢ monotonicity (cheap) tried first,
        # Gauss map separability as fallback (design §6, §10 principle 8).
        if _check_loop_free(cell.g1, cell.g2,
                            cell.T1, cell.T2, cell.T3, cell.T4):
            if cell.crossings:
                fr, pt = _trace_cell_by_registrations(cell, atol)
                all_fragments.extend(fr)
                all_points.extend(pt)
            continue

        # §6 step 3: Krawczyk-based tangency certification. If TΨ = 0 has a
        # simultaneous root in this cell, the intersection is tangential (C₂)
        # and must be traced via the regulated Φ system (design §1.4, §8),
        # NOT by further subdivision — deflation makes the Φ-curve regular
        # where Ψ is rank-deficient.
        P1_cart_local = cell.g1.surface[..., :-1] / cell.g1.surface[..., -1:]
        P2_cart_local = cell.g2.surface[..., :-1] / cell.g2.surface[..., -1:]
        local_box = ((0.0, 1.0),) * 4
        tangency = _check_tangency(
            cell.T1, cell.T2, cell.T3, cell.T4,
            P1_cart_local, P2_cart_local, local_box,
        )
        if tangency is True and cell.crossings:
            # Convert crossings to the cell's local stuv for the Φ tracer.
            crossings_local = [
                BoundaryPoint(
                    stuv=_global_to_local(c.stuv, cell.box),
                    xyz=c.xyz, face=c.face, tangent_raw=c.tangent_raw,
                )
                for c in cell.crossings
            ]
            fr_local, pt_local = _deflate_tangent_cell(
                P1_cart_local, P2_cart_local,
                cell.T1, cell.T2, cell.T3, cell.T4,
                local_box, crossings_local, atol,
                originals=cell.crossings, cell=cell,
            )
            for f in fr_local:
                stuv_glob = np.empty_like(f.stuv_path)
                for k in range(len(f.stuv_path)):
                    stuv_glob[k] = _local_to_global(f.stuv_path[k], cell.box)
                all_fragments.append(_Fragment(
                    start_point=f.start_point, end_point=f.end_point,
                    stuv_path=stuv_glob, xyz_path=f.xyz_path,
                ))
            # pt_local's SSXPoint.stuv is already global — we passed
            # `originals` so _deflate_tangent_cell copied from them.
            all_points.extend(pt_local)
            continue

        if cell.depth >= max_depth:
            for c in cell.crossings:
                all_points.append(SSXPoint(stuv=c.stuv, xyz=c.xyz))
            continue

        # --- Choose cut: through a crossing's parameter value ---
        cx_idx, cut_axis = _choose_cut(cell.crossings, cell.box)

        if cx_idx is None:
            # Can't cut — trace directly via registrations (§7).
            if cell.crossings:
                fr, pt = _trace_cell_by_registrations(cell, atol)
                all_fragments.extend(fr)
                all_points.extend(pt)
            continue

        cut_global_val = cell.crossings[cx_idx].stuv[cut_axis]

        # Convert cut to LOCAL parameter for the surface being split.
        # _choose_cut already guaranteed `cut_local` is within [min_margin,
        # 1 - min_margin]; no artificial clamp.
        cell_lo, cell_hi = cell.box[cut_axis]
        cut_local = (cut_global_val - cell_lo) / (cell_hi - cell_lo)

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

        # --- Shared internal partition at the cut (design §5 Inv. A) ---
        # The new face created by this cut is a single PartitionCurve shared
        # by both sub-cells. Its adjacents list will collect L and R below.
        shared_free = _partition_free_axis(cut_axis)
        shared_extent = cell.box[shared_free]
        internal_partition = PartitionCurve(
            axis=cut_axis, value=float(cut_global_val),
            free_axis=shared_free,
            global_extent=(float(shared_extent[0]), float(shared_extent[1])),
            adjacents=[], registrations=[],
        )

        if left_cx:
            L_cell = _Cell(g1=g1_L, g2=g2_L, crossings=left_cx,
                           box=box_L, depth=cell.depth + 1,
                           T1=T1_L, T2=T2_L, T3=T3_L, T4=T4_L)
            L_cell.partitions = _build_cell_partitions(L_cell, skip=(cut_axis, 1))
            L_cell.partitions.append(internal_partition)
            internal_partition.adjacents.append(L_cell)
            for c in left_cx:
                _classify_boundary_point(c, L_cell)
            stack.append(L_cell)
            all_cells.append(L_cell)
        if right_cx:
            R_cell = _Cell(g1=g1_R, g2=g2_R, crossings=right_cx,
                           box=box_R, depth=cell.depth + 1,
                           T1=T1_R, T2=T2_R, T3=T3_R, T4=T4_R)
            R_cell.partitions = _build_cell_partitions(R_cell, skip=(cut_axis, 0))
            R_cell.partitions.append(internal_partition)
            internal_partition.adjacents.append(R_cell)
            for c in right_cx:
                _classify_boundary_point(c, R_cell)
            stack.append(R_cell)
            all_cells.append(R_cell)

    # --- §9 assembly: walk the partition adjacency graph ---
    all_branches = _assemble_branches_by_adjacency(all_fragments, all_cells)
    all_branches.extend(overlap_branches)
    return {'branches': all_branches, 'points': all_points}


