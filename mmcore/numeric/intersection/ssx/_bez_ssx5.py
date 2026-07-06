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
    _trust_gjk,
)
from mmcore.numeric.algorithms.cygjk import gjk


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
class SSXSingularity:
    """A certified singular feature of the SSI (Cheng et al. 2023 C1/C2/C3).

    kind:
      'tangent_point'      — C2: T_Psi = 0, isolated or on branches
      'cusp'               — C1: surface-parameterization cusp on the curve
      'cusp_curve'         — C1 infinite case: samples of a singular curve
      'self_intersection'  — C3: two 4D preimages, one 3D point
    """
    kind: str
    stuv: NDArray[np.float64]                    # (4,) primary preimage
    xyz: NDArray[np.float64]                     # (3,)
    stuv_mate: Optional[NDArray[np.float64]] = None   # (4,) C3 second preimage
    branch_links: list = field(default_factory=list)  # [(branch_index, vertex_index)]
    samples: Optional[NDArray[np.float64]] = None     # (N,4) for 'cusp_curve'


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


def _tangency_witness(cell, atol, *, enumerate_all=True):
    """Gauss-Newton witness point(s) of the deflated system Δ = Ψ ∩ TΨ on the
    cell (local [0,1]⁴ coords). Returns (ok, roots, best_residual) where
    `roots` is a list of DISTINCT local witness points, the box-center
    start's root first when it converges.

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
    from mmcore.numeric.bern import bern_eval as _bern_eval
    from mmcore.numeric.ndinterval import interval as iv_interval, get_iarray
    from mmcore.numeric.intersection._deflate import (
        DeflatedSystem, gauss_newton_witness, _box_from_any,
    )
    from mmcore.numeric.intersection.ssx._ssx5_singular import (
        BoxNet, psi_vector_net, solve_zero_dim,
    )
    from mmcore.geom._nurbs_param_tol import bez_surface_param_tolerance
    P1c = cell.g1.surface[..., :-1] / cell.g1.surface[..., -1:]
    P2c = cell.g2.surface[..., :-1] / cell.g2.surface[..., -1:]
    try:
        sys_ = DeflatedSystem(
            P1=get_iarray(P1c, P1c), P2=get_iarray(P2c, P2c),
            T=tuple(np.asarray(T, dtype=iv_interval)
                    for T in (cell.T1, cell.T2, cell.T3, cell.T4)),
            bern_eval=_bern_eval, interval_ctor=iv_interval,
        )
        Bf = _box_from_any(tuple(iv_interval(0.0, 1.0) for _ in range(4)))

        def _gn(x0):
            ok_, xw_, _fn = gauss_newton_witness(sys_, Bf, x0=x0,
                                                 tol_f=1e-10, max_iter=24)
            return np.asarray(xw_, dtype=np.float64) if ok_ else None

        def _xyz(x):
            return eval_surface(cell.g1.surface, x[0], x[1], rational=True)

        roots = []
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
            sols, _exhausted = solve_zero_dim(nets, _gn, ptol,
                                              max_cells=2000, dedup_xyz=_xyz,
                                              atol=atol)
            for sol in sols:
                # same destructive-dedup rule as solve_zero_dim's own _dup:
                # 1·ptol per-axis box AND xyz <= atol
                sol_xyz = _xyz(sol)
                if not any(np.all(np.abs(sol - r_) <= ptol)
                           and float(np.linalg.norm(sol_xyz - _xyz(r_))) <= atol
                           for r_ in roots):
                    roots.append(sol)
        best_fn = min((float(np.linalg.norm(sys_.delta_point(r_)))
                       for r_ in roots), default=np.inf)
        return bool(roots), roots, best_fn
    except Exception:
        return False, [], np.inf


def _emit_tangent_roots(cell, atol, unify_tol, all_singularities,
                        *, enumerate_all=True):
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

    Returns `(ok, roots)`: `ok` from `_tangency_witness` (True iff at least
    one converged witness exists — the crossing-less arm's size gate needs
    it) and the LOCAL witness points — Task 5's Φ∩L seeding consumes
    `roots[0]` in the crossing-less arm (`_choose_phi_equations` seed).
    """
    ok, roots, _fn = _tangency_witness(cell, atol, enumerate_all=enumerate_all)
    for xw in roots:
        stuv_g = _local_to_global(np.asarray(xw), cell.box)
        xyz_w = eval_surface(cell.g1.surface, xw[0], xw[1], rational=True)
        if not any(g.kind == "tangent_point"
                   and np.all(np.abs(g.stuv - stuv_g) <= unify_tol)
                   and float(np.linalg.norm(g.xyz - xyz_w)) <= 2.0 * atol
                   for g in all_singularities):
            all_singularities.append(SSXSingularity(
                kind="tangent_point", stuv=stuv_g, xyz=xyz_w))
    return ok, roots


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
    h_max=None,
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
        return np.array(stuv_pts), np.array(xyz_pts)

    # Orient tangent toward the target
    if np.dot(tang_prev, target - current) < 0:
        tang_prev = -tang_prev

    t3_prev, speed = _tangent_3d(S1, current, tang_prev, rational=rational)

    rejects = 0
    for _ in range(max_points):
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
        return np.array(stuv_pts), np.array(xyz_pts), exit_info

    # Orient tangent using hint if provided
    if direction_hint is not None and np.dot(tang_prev, direction_hint) < 0:
        tang_prev = -tang_prev

    t3_prev, speed = _tangent_3d(S1, current, tang_prev, rational=rational)

    rejects = 0
    for iter_num in range(max_points):
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
    h_max=None,
    min_step=1e-6,
    max_step=0.25,
    angle_threshold=0.1,
    max_points=2000,
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

    # Get tangent direction from Φ Jacobian
    J = _jac_phi(S1, S2, T_arr, psi_rows, *current, rational=rational)
    _, _, Vt = np.linalg.svd(J, full_matrices=True)
    tang_prev = Vt[-1]

    if np.dot(tang_prev, target - current) < 0:
        tang_prev = -tang_prev

    t3_prev, speed = _phi_dir_speed(current, tang_prev)

    rejects = 0
    for _ in range(max_points):
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
            _, _, Vt = np.linalg.svd(J, full_matrices=True)
            tang_new = Vt[-1]
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

    return np.array(stuv_pts), np.array(xyz_pts)


def _deflate_tangent_cell(P1_cart, P2_cart, T1, T2, T3, T4, box, crossings, atol,
                          *, originals=None, cell=None, h_max=None):
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

    for i, j in pairs:
        stuv_path, xyz_path = _march_phi_curve(
            P1_cart, P2_cart, T_chosen, psi_rows,
            crossings[i].stuv, crossings[j].stuv,
            atol=atol, rational=False, h_max=h_max,
        )
        if len(stuv_path) < 2:
            continue
        # Check that points lie on the actual intersection (full Ψ=0).
        valid_mask = np.zeros(len(stuv_path), dtype=bool)
        for k in range(len(stuv_path)):
            p1 = eval_surface(P1_cart, stuv_path[k, 0], stuv_path[k, 1], rational=False)
            p2 = eval_surface(P2_cart, stuv_path[k, 2], stuv_path[k, 3], rational=False)
            if np.linalg.norm(p1 - p2) < atol:
                valid_mask[k] = True
        if not np.any(valid_mask):
            continue
        start_pt = originals[i] if originals is not None else None
        end_pt = originals[j] if originals is not None else None
        fragments.append(_Fragment(
            start_point=start_pt, end_point=end_pt,
            stuv_path=stuv_path[valid_mask],
            xyz_path=xyz_path[valid_mask],
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

    # Per-axis parametric tolerance for the cell's local sub-surfaces.
    # Sizes the marcher's initial/minimal steps and the endpoint matching
    # radius (in GLOBAL coordinates the local tolerance scales by the
    # cell's span on each axis).
    ptol_s, ptol_t = bez_surface_param_tolerance(cell.g1.surface, atol, rational=True)
    ptol_u, ptol_v = bez_surface_param_tolerance(cell.g2.surface, atol, rational=True)
    ptol_local = np.array([float(ptol_s), float(ptol_t), float(ptol_u), float(ptol_v)])
    ptol_local = np.maximum(ptol_local, 1e-12)
    ptol_min = max(float(ptol_local.max()), 1e-9)
    spans = np.array([cell.box[ax][1] - cell.box[ax][0] for ax in range(4)])
    # Global per-axis matching radius: CSX roots and marcher exits are each
    # accurate to ~ptol, so 4x covers both ends with headroom.
    match_tol_global = 4.0 * ptol_local * np.maximum(spans, 1e-15)

    for i, start_cx in enumerate(cell.crossings):
        if i in used:
            continue

        start_local = _global_to_local(start_cx.stuv, cell.box)

        # XYZ distance to the nearest unused partner crossing bounds the
        # marcher's initial xyz step target: step toward the partner, not
        # past it. (xyz, not stuv — step sizing is geometry-driven.)
        cell_h_max = h_max if h_max is not None else max(
            0.05 * _local_diag(cell.g1.surface, rational=True), 4.0 * atol)
        nearest_xyz = float('inf')
        for j, cx in enumerate(cell.crossings):
            if j == i or j in used:
                continue
            d = float(np.linalg.norm(np.asarray(cx.xyz, dtype=np.float64)
                                     - np.asarray(start_cx.xyz, dtype=np.float64)))
            if d < nearest_xyz:
                nearest_xyz = d

        if nearest_xyz == float('inf'):
            h_init = 0.25 * cell_h_max
        else:
            h_init = min(cell_h_max, max(atol, 0.25 * nearest_xyz))

        traced = False
        for attempt in range(2):
            hint = None
            if attempt == 1:
                tang, _, _ = _ssx_tangent_4d(
                    cell.g1.surface, cell.g2.surface,
                    *start_local, rational=True)
                if tang is None:
                    break
                hint = -tang

            stuv_local, xyz_local, exit_info = _march_to_boundary(
                cell.g1.surface, cell.g2.surface, start_local,
                atol=atol, rational=True, direction_hint=hint,
                h_init=h_init, h_max=cell_h_max,
                min_step=ptol_min,
            )

            if len(stuv_local) < 2:
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
            # unconsumed ones are removed from the seed pool.
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
            # The 2·atol radius matches the unification guard exactly:
            # anything accepted here can also be unified across cells (a
            # looser match created a chain-break window at (2, 4]·atol).
            if (best_j is not None and best_score <= 1.0
                    and float(np.linalg.norm(
                        np.asarray(cell.crossings[best_j].xyz, dtype=np.float64)
                        - np.asarray(xyz_local[-1], dtype=np.float64))) <= 2.0 * atol):
                end_cx = cell.crossings[best_j]
                stuv_global[-1] = end_cx.stuv.copy()
                xyz_local[-1] = end_cx.xyz.copy()
                used.add(best_j)
            elif exit_info is not None:
                # No registered crossing here — the marcher just proved one
                # exists (Newton-converged exit on a face). Synthesize it.
                axis = exit_info[0]
                side = 0 if stuv_local[-1][axis] < 0.5 else 1
                tang_end, _, _ = _ssx_tangent_4d(
                    cell.g1.surface, cell.g2.surface,
                    *stuv_local[-1], rational=True)
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
            traced = True
            break

        if not traced and i not in used:
            # Both directions failed (genuine corner touch or marcher
            # failure). Surface the crossing as an isolated point instead
            # of silently dropping it.
            points.append(SSXPoint(stuv=start_cx.stuv, xyz=start_cx.xyz))

    return fragments, points


def _unify_fragment_endpoints(fragments: list[_Fragment], unify_tol,
                              unify_atol: float = 1e-3) -> None:
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

    for a in range(n):
        for b in range(a + 1, n):
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

    canon = {id(objs[k]): objs[_find(k)] for k in range(n)}
    for f in fragments:
        if f.start_point is not None:
            f.start_point = canon[id(f.start_point)]
        if f.end_point is not None:
            f.end_point = canon[id(f.end_point)]


def _fragment_contained_in(f: _Fragment, g: _Fragment, tol: float) -> bool:
    """True if EVERY xyz sample of `f` lies within `tol` of `g`'s polyline."""
    poly = np.asarray(g.xyz_path, dtype=np.float64)
    if len(poly) < 2:
        return False
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom < 1e-30, 1e-30, denom)
    for p in np.asarray(f.xyz_path, dtype=np.float64):
        ap = p[None, :] - a
        tt = np.clip(np.einsum("ij,ij->i", ap, ab) / denom, 0.0, 1.0)
        proj = a + tt[:, None] * ab
        if float(np.linalg.norm(proj - p[None, :], axis=1).min()) > tol:
            return False
    return True


def _drop_duplicate_fragments(fragments: list[_Fragment], atol: float) -> list[_Fragment]:
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

    keep: list[_Fragment] = []
    for f in sorted(fragments, key=_arc_len, reverse=True):
        if any(_fragment_contained_in(f, g, 2.0 * atol) for g in keep):
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
    """
    from collections import defaultdict

    if unify_tol is not None and len(fragments) > 1:
        _unify_fragment_endpoints(fragments, unify_tol, unify_atol=atol_full)
        fragments = _drop_duplicate_fragments(fragments, atol_full)

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
        for j, role in pool:
            if j == self_idx or consumed[j]:
                continue
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
                if (start_interior and end_interior
                        and median_step > 0
                        and gap < 10.0 * median_step):
                    closing_stuv, closing_xyz = _march_intersection_curve(
                        S1_full, S2_full,
                        stuv_full[-1], stuv_full[0],
                        atol=atol_full, rational=rational_full,
                        h_max=h_max,
                    )
                    if len(closing_stuv) >= 2:
                        # Skip the first sample (duplicates xyz_full[-1]).
                        stuv_full = np.concatenate(
                            [stuv_full, closing_stuv[1:]], axis=0)
                        xyz_full = np.concatenate(
                            [xyz_full, closing_xyz[1:]], axis=0)

        branch_kind = ("tangential" if any(fragments[idx].tangential for idx, _ in chain)
                       else "transversal")
        branches.append(SSXBranch(curve=(stuv_full, xyz_full), kind=branch_kind))

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
            if len(xyz) <= SLIVER_MAX_PTS:
                for big in kept_xyz:
                    if len(big) < 2:
                        continue
                    if all(_dist_point_polyline(np.asarray(p), big) <= sliver_tol
                           for p in xyz):
                        is_sliver = True
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
    csx_result['isolated'] = list((lambda x: not (((1 - x['t']) < 1e-6) or (x['t'] < 1e-6)), csx_result['isolated']))
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


def bez_ssx(
    S1,
    S2,
    atol=1e-3,
    rational=True,
    max_depth=12,
    max_xyz_step=None,
) -> dict:
    """Bezier surface-surface intersection v5.

    Iterative stack-based domain decomposition.
    All crossings and branch endpoints are in GLOBAL [0,1]⁴ coordinates.
    Surfaces in sub-cells are in LOCAL [0,1]² (De Casteljau reparameterized).
    Conversion between local and global uses the cell's box.

    Returns dict with 'branches', 'points', and 'singularities'.
    """
    S1 = np.asarray(S1, dtype=np.float64)
    S2 = np.asarray(S2, dtype=np.float64)

    # --- Level 1: Pruning ---
    if _prune_ssx_cell(S1, S2, atol, rational=rational):
        return {'branches': [], 'points': [], 'singularities': []}

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
    all_fragments: list[_Fragment] = []
    all_points = []
    all_singularities: list[SSXSingularity] = []

    while queue:
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
                if (cell.T1 is not None and all(
                        float(np.min(T)) <= 0.0 <= float(np.max(T))
                        for T in (cell.T1, cell.T2, cell.T3, cell.T4))):
                    _emit_tangent_roots(cell, atol, unify_tol,
                                        all_singularities,
                                        enumerate_all=False)
                if cell.crossings:
                    fr, pt = _trace_cell_by_registrations(cell, atol, h_max=h_max)
                    all_fragments.extend(fr)
                    all_points.extend(pt)
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
        is_clearly_transversal = False
        if not cell.crossings:
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
            P1_cart_local = cell.g1.surface[..., :-1] / cell.g1.surface[..., -1:]
            P2_cart_local = cell.g2.surface[..., :-1] / cell.g2.surface[..., -1:]
            local_box = ((0.0, 1.0),) * 4
            tangency = _check_tangency(
                cell.T1, cell.T2, cell.T3, cell.T4,
                P1_cart_local, P2_cart_local, local_box,
            )
        if tangency is True and not cell.crossings:
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
                                            enumerate_all=True)
            # Emitting the tangency does NOT resolve the cell: the same
            # crossing-less cell can hold coexisting transversal features —
            # z = q(q-1/2) (Mexican hat) has the touch at the center AND a
            # transversal ring at q = 1/2, and an unconditional `continue`
            # here silently deleted the ring (Task 5's Φ∩L seeding cannot
            # recover it: the ring is transversal, not on Φ). Stop only when
            # the cell is at tolerance scale (all four GLOBAL spans within
            # 4·unify_tol); otherwise fall through to the subdivision path
            # like any other uncertified cell — descendants that re-confirm
            # the same tangency are absorbed by the emission dedup above.
            # A failed witness (ok=False: the blanket exception path) must
            # not vanish either — fall through regardless of size so the
            # cell is never dropped with neither emission nor subdivision.
            spans = np.array([hi - lo for (lo, hi) in cell.box])
            if ok and np.all(spans <= 4.0 * unify_tol):
                continue

        if tangency is True and cell.crossings:
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
            # center witness costs ~ms; a witness landing ON a traced
            # tangential/overlap branch is dropped by the post-assembly
            # subsumption filter, so the curve case emits nothing. Task 5
            # owns typed 1-dim tangencies.
            _emit_tangent_roots(cell, atol, unify_tol, all_singularities,
                                enumerate_all=False)
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
                originals=cell.crossings, cell=cell, h_max=h_max,
            )
            for f in fr_local:
                stuv_glob = np.empty_like(f.stuv_path)
                for k in range(len(f.stuv_path)):
                    stuv_glob[k] = _local_to_global(f.stuv_path[k], cell.box)
                all_fragments.append(_Fragment(
                    start_point=f.start_point, end_point=f.end_point,
                    stuv_path=stuv_glob, xyz_path=f.xyz_path,
                    tangential=f.tangential,
                ))
            # pt_local's SSXPoint.stuv is already global — we passed
            # `originals` so _deflate_tangent_cell copied from them.
            all_points.extend(pt_local)
            continue

        if cell.depth >= max_depth:
            for c in cell.crossings:
                all_points.append(SSXPoint(stuv=c.stuv, xyz=c.xyz))
            continue

        # --- Dual-surface subdivision ---
        # Both surfaces are split at each step. Productive crossings provide
        # per-surface split values; if a surface has no guided split, it gets
        # a midpoint cut on its longest-span axis.

        s1_axis, s1_cuts, s2_axis, s2_cuts = _compute_split_plan(
            cell.new_crossings, cell.box)
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
            s1_lo_box, s1_hi_box = cell.box[s1_axis]
            cut_local_s1 = (cv - s1_lo_box) / (s1_hi_box - s1_lo_box)
            isoline_s1 = _extract_isoline(cell.g1.surface, s1_local_axis, cut_local_s1)

            for s2_idx in range(n2):
                s2_piece_surf = g2_pieces[s2_idx].surface
                csx_r = bez_csx(isoline_s1, s2_piece_surf, atol=atol, rational=True)
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

        # c/d: CSX(cut_line_s2, S1_piece) for each S2 cut × each S1 piece
        for cut_idx, cv in enumerate(s2_cuts):
            s2_lo_box, s2_hi_box = cell.box[s2_axis]
            cut_local_s2 = (cv - s2_lo_box) / (s2_hi_box - s2_lo_box)
            isoline_s2 = _extract_isoline(cell.g2.surface, s2_local_axis, cut_local_s2)

            for s1_idx in range(n1):
                s1_piece_surf = g1_pieces[s1_idx].surface
                csx_r = bez_csx(isoline_s2, s1_piece_surf, atol=atol, rational=True)
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
    )
    all_branches.extend(overlap_branches)

    # A tangent_point ON a 1-dimensional tangential feature is not an
    # isolated C2 touch: overlap regions and traced tangent curves (branch
    # kind 'overlap'/'tangential') consist entirely of Δ-roots, so the
    # witness on a cell holding one converges to an arbitrary sample of the
    # curve (measured: the legacy overlaps case emitted its domain corner;
    # the crossed-saddles center witness lands on the tangent curve). The
    # richer feature already reports the contact — drop the redundant point
    # (Task 5 will type tangent curves explicitly). Same ON-a-branch
    # semantics and 4·atol tolerance as the points-on-branch filter below;
    # the polyline is a chorded approximation, so points ON the true curve
    # sit up to the 2·atol sagitta off it (measured max 1.9e-3 = 1.9·atol).
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
    if all_singularities:
        _one_dim_polys = []
        for b in all_branches:
            if b.kind not in ("overlap", "tangential"):
                continue
            _poly = np.asarray(b.curve[1], dtype=np.float64)
            if len(_poly) < 2:
                continue
            _arc = float(np.linalg.norm(np.diff(_poly, axis=0), axis=1).sum())
            if _arc > 16.0 * atol:
                _one_dim_polys.append(_poly)
        if _one_dim_polys:
            all_singularities = [
                g for g in all_singularities
                if not (g.kind == "tangent_point" and any(
                    _dist_point_polyline(np.asarray(g.xyz, dtype=np.float64),
                                         poly) <= 4.0 * atol
                    for poly in _one_dim_polys))
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
        for b in all_branches:
            xyz = np.asarray(b.curve[1], dtype=np.float64)
            if len(xyz):
                d_min = np.linalg.norm(
                    xyz[:, None, :] - _tp[None, :, :], axis=2).min(axis=1)
                if np.all(d_min <= 4.0 * atol):
                    arc = (float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
                           if len(xyz) > 1 else 0.0)
                    if arc <= 16.0 * atol:
                        continue
            _kept_branches.append(b)
        all_branches = _kept_branches

    # A reported point within 2·atol (xyz) of an emitted tangent_point is
    # not a separate intersection — it is the certified tangency itself,
    # re-found by CSX grazing-valley seeds while subdividing around the
    # touch (measured on the paraboloid/Mexican-hat cases: 4 seeds at the
    # touch + 4 at ±1·atol on the grazing valley). Subsume them into the
    # typed singularity. Matching-ladder xyz guard only (2·atol); no param
    # guard needed — any Ψ-point that close to the certified tangency is
    # indistinguishable from it at tolerance.
    if all_points and _tangent_xyz:
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
    if all_points and all_branches:
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

    return {'branches': all_branches, 'points': all_points, 'singularities': all_singularities}


