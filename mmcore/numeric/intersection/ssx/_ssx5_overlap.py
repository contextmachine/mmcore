"""2-D surface-overlap regions for bez_ssx (ledger L28, approved Option C).

A C2 positive-dimensional component — S1 coincides with S2 (within atol)
over a 2-D region (Cheng et al. 2023 Fig. 8, the ``#(Δ_B)=∞`` /
2-dimensional full- or partial-overlap sub-case) — cannot be represented by
the branch/point schema.  This module assembles the approved structured
region: closed rim loops that REFERENCE ``kind='overlap'`` branches, paired
sample-synchronized parameter-space loops on both surfaces, a certified
interior witness, and the normal agreement booleans/trimming need.

Rim discovery is deliberately self-contained: each of the eight domain
edges is SAMPLED and point-inverted onto the opposite surface, and a rim
span is kept only where every sample's residual stays within ``atol`` (the
certification records the worst one in atol units).  This uniformly covers
both the affine rims that CSX already claims and the curved-UV rims whose
exact-affine certificate correctly fails (ledger L42) — the L42 fallback
stops the completeness lie at the CSX level, and this assembler is what
finally REPRESENTS the rim.

Tolerance ladder (review doc §8; also the L25 hinge): a coincidence band
admits a region only if an interior witness exists at ≥ 4·ptol (per axis,
in both parameter planes) from every rim loop with residual ≤ atol;
anything thinner stays curve-only (L27's shared-edge fixture is the
negative control).  With ptol ≈ atol/|dS| this is the "band of width
< atol stays a curve" rule, a factor ~4 stricter — the sound direction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mmcore.numeric.intersection._bezier_common import (
    eval_curve, eval_surface, eval_surface_d1,
)
from mmcore.numeric.intersection.ssx._ssx4 import SSXBranch

__all__ = ["SSXOverlapRegion", "assemble_overlap_regions"]


@dataclass
class SSXOverlapRegion:
    """C2 positive-dimensional component: S1 ≡ S2 (within atol) over a
    2-D region (Cheng et al. Fig. 8, #(Δ_B)=∞ / 2-dimensional).

    ``boundary`` holds one inner list per closed rim loop; entries
    ``(branch_index, reversed)`` reference ``result['branches']``
    (kind='overlap' rim curves), ordered head-to-tail; loop 0 is the outer
    loop, later loops are holes (islands where the surfaces depart).
    ``uv1_loops[i][k]`` and ``uv2_loops[i][k]`` are preimages of the same
    3-D point (sample-synchronized closed polylines, first == last).
    ``normal_agreement`` is +1 for aligned normals over the region, −1 for
    opposed (constant over a connected coincidence region).
    ``interior_stuv`` is a certified interior witness (point-in-region
    seed); ``certification`` records {'boundary_resid_max',
    'interior_resid', 'n_samples', 'orientation_consistent'} with the
    residuals in atol units.
    """

    boundary: list = field(default_factory=list)
    uv1_loops: list = field(default_factory=list)
    uv2_loops: list = field(default_factory=list)
    normal_agreement: int = 1
    interior_stuv: NDArray[np.float64] = None
    certification: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Point inversion (2x2 Gauss-Newton, seeded from a coarse grid)
# ---------------------------------------------------------------------------

def _invert_point(S_h, xyz, seed=None, max_iter=30):
    """Project ``xyz`` onto the surface: returns (u, v, residual).

    Plain damped Gauss-Newton on ||S(u,v) - xyz||², clamped to [0,1]².
    A 5x5 seed grid keeps the start inside the right monotone basin for
    the low-degree patches this assembler certifies; the caller judges
    acceptance purely by the returned residual, so a failed inversion is
    always safe (it only loses a rim sample).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if seed is None:
        best = None
        for u in np.linspace(0.0, 1.0, 5):
            for v in np.linspace(0.0, 1.0, 5):
                d = float(np.linalg.norm(
                    eval_surface(S_h, u, v, rational=True) - xyz))
                if best is None or d < best[0]:
                    best = (d, u, v)
        u, v = best[1], best[2]
    else:
        u, v = float(seed[0]), float(seed[1])

    for _ in range(max_iter):
        p, du, dv = eval_surface_d1(S_h, u, v, rational=True)
        r = p - xyz
        J = np.stack([du, dv], axis=1)          # (3, 2)
        JtJ = J.T @ J
        rhs = J.T @ r
        try:
            step = np.linalg.solve(JtJ + 1e-14 * np.eye(2), rhs)
        except np.linalg.LinAlgError:
            break
        u_new = min(1.0, max(0.0, u - float(step[0])))
        v_new = min(1.0, max(0.0, v - float(step[1])))
        if abs(u_new - u) < 1e-15 and abs(v_new - v) < 1e-15:
            u, v = u_new, v_new
            break
        u, v = u_new, v_new
    resid = float(np.linalg.norm(
        eval_surface(S_h, u, v, rational=True) - xyz))
    return u, v, resid


# ---------------------------------------------------------------------------
# Rim discovery: domain edges sampled onto the opposite surface
# ---------------------------------------------------------------------------

_EDGES = (
    # (owner, axis, side): owner surface index, fixed axis (0=first param),
    # fixed value.  Edge parameter runs over the free axis.
    (1, 0, 0.0), (1, 0, 1.0), (1, 1, 0.0), (1, 1, 1.0),
    (2, 0, 0.0), (2, 0, 1.0), (2, 1, 0.0), (2, 1, 1.0),
)


def _edge_own_uv(axis, side, w):
    return (side, w) if axis == 0 else (w, side)


def _stuv_sample(owner, own_uv, other_uv):
    if owner == 1:
        return (own_uv[0], own_uv[1], other_uv[0], other_uv[1])
    return (other_uv[0], other_uv[1], own_uv[0], own_uv[1])


def _sample_edge_rims(S_own_h, S_other_h, owner, axis, side, atol,
                      coarse=33, dense=17):
    """On-surface spans of one domain edge, as fully-sampled rim paths.

    Returns a list of rims, each a dict with synchronized arrays:
    ``stuv`` (N,4) global parameters, ``xyz`` (N,3), ``resid_max``.
    """
    ws = np.linspace(0.0, 1.0, coarse)
    onsurf = np.zeros(coarse, dtype=bool)
    inv = np.zeros((coarse, 2), dtype=np.float64)
    seed = None
    for k, w in enumerate(ws):
        uo, vo = _edge_own_uv(axis, side, float(w))
        xyz = eval_surface(S_own_h, uo, vo, rational=True)
        u, v, resid = _invert_point(S_other_h, xyz, seed=seed)
        # A wandering warm start can hand the next sample a foreign basin
        # after an off-surface stretch; only propagate on-surface seeds.
        seed = (u, v) if resid <= atol else None
        onsurf[k] = resid <= atol
        inv[k] = (u, v)

    rims = []
    k = 0
    while k < coarse:
        if not onsurf[k]:
            k += 1
            continue
        j = k
        while j + 1 < coarse and onsurf[j + 1]:
            j += 1
        w_lo, w_hi = float(ws[k]), float(ws[j])
        # Refine both ends by bisection against the on-surface predicate.
        w_lo = _refine_end(S_own_h, S_other_h, axis, side, atol,
                           w_lo, w_lo - (ws[1] - ws[0]), inv[k])
        w_hi = _refine_end(S_own_h, S_other_h, axis, side, atol,
                           w_hi, w_hi + (ws[1] - ws[0]), inv[j])
        rim = _resample_rim(S_own_h, S_other_h, owner, axis, side,
                            atol, w_lo, w_hi, dense)
        if rim is not None:
            rims.append(rim)
        k = j + 1
    return rims


def _refine_end(S_own_h, S_other_h, axis, side, atol, w_in, w_out, seed):
    """Bisect the on-surface span end between an inside and outside sample."""
    w_out = min(1.0, max(0.0, w_out))
    if w_in == w_out:
        return w_in
    lo, hi = (w_in, w_out)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        uo, vo = _edge_own_uv(axis, side, mid)
        xyz = eval_surface(S_own_h, uo, vo, rational=True)
        _, _, resid = _invert_point(S_other_h, xyz, seed=tuple(seed))
        if resid <= atol:
            lo = mid
        else:
            hi = mid
    return lo


def _resample_rim(S_own_h, S_other_h, owner, axis, side, atol,
                  w_lo, w_hi, dense):
    if w_hi - w_lo <= 1e-12:
        return None
    ws = np.linspace(w_lo, w_hi, dense)
    stuv = np.zeros((dense, 4), dtype=np.float64)
    xyz = np.zeros((dense, 3), dtype=np.float64)
    resid_max = 0.0
    seed = None
    for k, w in enumerate(ws):
        uo, vo = _edge_own_uv(axis, side, float(w))
        p = eval_surface(S_own_h, uo, vo, rational=True)
        u, v, resid = _invert_point(S_other_h, p, seed=seed)
        if resid > atol:
            return None       # the refined span must certify end-to-end
        seed = (u, v)
        resid_max = max(resid_max, resid)
        stuv[k] = _stuv_sample(owner, (uo, vo), (u, v))
        xyz[k] = p
    if float(np.linalg.norm(xyz[-1] - xyz[0])) < atol and dense > 2:
        return None           # degenerate (corner-touch) span
    return {"stuv": stuv, "xyz": xyz, "resid_max": resid_max,
            "owner": owner, "axis": axis, "side": side}


# ---------------------------------------------------------------------------
# Loop assembly
# ---------------------------------------------------------------------------

def _dist_point_polyline(p, poly):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom < 1e-30, 1e-30, denom)
    t = np.clip(np.einsum("ij,ij->i", p[None, :] - a, ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.linalg.norm(proj - p[None, :], axis=1).min())


def _dedup_rims(rims, atol):
    """Drop rims geometrically contained in an earlier rim (shared edges
    are discovered from both surfaces' boundaries)."""
    kept = []
    for rim in sorted(rims, key=lambda r: -len(r["xyz"])):
        dup = False
        for other in kept:
            if len(other["xyz"]) < 2:
                continue
            if all(_dist_point_polyline(p, other["xyz"]) <= 2.0 * atol
                   for p in rim["xyz"]):
                dup = True
                break
        if not dup:
            kept.append(rim)
    return kept


def _peel_dangling_rims(rims, atol):
    """Drop rims that can never participate in a closed loop.

    Cluster rim endpoints into nodes (2*atol) and iteratively remove rims
    touching a degree-1 node.  This peels the tolerance-band corner stubs
    (a stretch of one edge passing WITHIN atol of the other surface near a
    shared corner without lying on the region rim — measured residual ≈
    atol vs ≤ 1e-13 on genuine exact rims): their far end has no
    continuation, so they are graph-theoretically incapable of closing,
    yet a first-match walk would happily wander into them and dead-end.
    """
    alive = list(range(len(rims)))
    while True:
        pts = []
        keys = []
        for i in alive:
            for e in (rims[i]["xyz"][0], rims[i]["xyz"][-1]):
                node = None
                for k, p in enumerate(pts):
                    if float(np.linalg.norm(p - e)) <= 2.0 * atol:
                        node = k
                        break
                if node is None:
                    pts.append(np.asarray(e, dtype=np.float64))
                    node = len(pts) - 1
                keys.append((i, node))
        degree = {}
        for _i, node in keys:
            degree[node] = degree.get(node, 0) + 1
        drop = {i for i, node in keys if degree[node] < 2}
        if not drop:
            return [rims[i] for i in alive]
        alive = [i for i in alive if i not in drop]
        if not alive:
            return []


def _assemble_loops(rims, atol):
    """Connect rims head-to-tail into closed loops (xyz endpoint match)."""
    n = len(rims)
    if n == 0:
        return []
    ends = [(rim["xyz"][0], rim["xyz"][-1]) for rim in rims]
    used = [False] * n
    loops = []
    for start in range(n):
        if used[start]:
            continue
        chain = [(start, False)]
        used[start] = True
        loop_start = ends[start][0]
        cur = ends[start][1]
        closed = False
        while True:
            if float(np.linalg.norm(cur - loop_start)) <= 2.0 * atol and (
                    len(chain) > 1 or float(np.linalg.norm(
                        ends[chain[0][0]][1] - ends[chain[0][0]][0]))
                    <= 2.0 * atol):
                closed = True
                break
            found = None
            for j in range(n):
                if used[j]:
                    continue
                if float(np.linalg.norm(ends[j][0] - cur)) <= 2.0 * atol:
                    found = (j, False)
                elif float(np.linalg.norm(ends[j][1] - cur)) <= 2.0 * atol:
                    found = (j, True)
                if found is not None:
                    break
            if found is None:
                break
            j, rev = found
            used[j] = True
            chain.append((j, rev))
            cur = ends[j][0 if rev else 1]
        if closed and len(chain) >= 1:
            loops.append(chain)
        # non-closing chains simply stay unreferenced rims (curve-only)
    return loops


def _loop_paths(rims, loop):
    """Concatenate a loop's rims into closed synchronized stuv/xyz paths."""
    stuv_parts, xyz_parts = [], []
    for idx, (ri, rev) in enumerate(loop):
        stuv = rims[ri]["stuv"][::-1] if rev else rims[ri]["stuv"]
        xyz = rims[ri]["xyz"][::-1] if rev else rims[ri]["xyz"]
        if idx > 0:
            stuv, xyz = stuv[1:], xyz[1:]
        stuv_parts.append(stuv)
        xyz_parts.append(xyz)
    stuv = np.concatenate(stuv_parts, axis=0)
    xyz = np.concatenate(xyz_parts, axis=0)
    # close exactly
    stuv = np.concatenate([stuv, stuv[:1]], axis=0)
    xyz = np.concatenate([xyz, xyz[:1]], axis=0)
    return stuv, xyz


def _signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _point_in_polygon(pt, poly):
    x, y = float(pt[0]), float(pt[1])
    inside = False
    for k in range(len(poly) - 1):
        x1, y1 = poly[k]
        x2, y2 = poly[k + 1]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def _dist_point_polyline_2d(p, poly):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom < 1e-30, 1e-30, denom)
    t = np.clip(np.einsum("ij,ij->i", p[None, :] - a, ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.linalg.norm(proj - p[None, :], axis=1).min())


def _interior_witness(S1_h, S2_h, uv1_loops, uv2_loops, atol, ptol4):
    """Certified interior seed at >= 4*ptol from every rim loop (§8)."""
    outer1 = uv1_loops[0]
    holes1 = uv1_loops[1:]
    lo = outer1.min(axis=0)
    hi = outer1.max(axis=0)
    p_bar12 = 4.0 * max(float(ptol4[0]), float(ptol4[1]))
    p_bar34 = 4.0 * max(float(ptol4[2]), float(ptol4[3]))

    candidates = [outer1[:-1].mean(axis=0)]
    for gu in np.linspace(0.15, 0.85, 8):
        for gv in np.linspace(0.15, 0.85, 8):
            candidates.append(lo + np.array([gu, gv]) * (hi - lo))

    best = None
    for cand in candidates:
        if not _point_in_polygon(cand, outer1):
            continue
        if any(_point_in_polygon(cand, h) for h in holes1):
            continue
        d1 = min(_dist_point_polyline_2d(cand, lp) for lp in uv1_loops)
        if d1 < p_bar12:
            continue
        p1 = eval_surface(S1_h, float(cand[0]), float(cand[1]),
                          rational=True)
        u, v, resid = _invert_point(S2_h, p1)
        if resid > atol:
            continue
        d2 = min(_dist_point_polyline_2d(np.array([u, v]), lp)
                 for lp in uv2_loops)
        if d2 < p_bar34:
            continue
        score = min(d1 / max(p_bar12, 1e-15), d2 / max(p_bar34, 1e-15))
        if best is None or score > best[0]:
            best = (score, np.array([cand[0], cand[1], u, v]), resid)
    if best is None:
        return None, None
    return best[1], best[2]


# ---------------------------------------------------------------------------
# Public assembler
# ---------------------------------------------------------------------------

def assemble_overlap_regions(
    S1_h, S2_h, *, atol, ptol4,
    existing_overlap_branches=(),
    uncertified_spans=(),
    overlap_boxes=(),
    charge=None,
):
    """Assemble certified SSXOverlapRegion entities from rim evidence.

    Returns a dict: ``regions`` (boundary indices are RELATIVE to the
    returned ``rim_branches`` list), ``rim_branches`` (canonical, properly
    sampled kind='overlap' SSXBranch objects), ``unmatched_branches``
    (pre-existing overlap branches not part of any region rim — curve-only
    overlaps, kept verbatim), and ``covered`` (True iff every piece of
    overlap evidence — parametric overlap boxes and uncertified CSX spans —
    is explained by a certified region, so the caller may retire the
    structural incompleteness reason).
    """
    S1_h = np.asarray(S1_h, dtype=np.float64)
    S2_h = np.asarray(S2_h, dtype=np.float64)
    ptol4 = np.asarray(ptol4, dtype=np.float64)
    existing = list(existing_overlap_branches)

    def _charge(n):
        return charge(n) if charge is not None else True

    empty = {"regions": [], "rim_branches": [],
             "unmatched_branches": existing, "covered": False}

    # 8 edges x (coarse + dense) inversions, each a bounded GN solve.
    if not _charge(8 * 33 + 8 * 17):
        return empty

    rims = []
    for owner, axis, side in _EDGES:
        own = S1_h if owner == 1 else S2_h
        other = S2_h if owner == 1 else S1_h
        rims.extend(_sample_edge_rims(own, other, owner, axis, side, atol))
    rims = _dedup_rims(rims, atol)
    rims = _peel_dangling_rims(rims, atol)
    if not rims:
        return empty

    loops_raw = _assemble_loops(rims, atol)
    if not loops_raw:
        return empty

    # Build loop paths + orientation bookkeeping in S1's (u,v).
    loops = []
    for chain in loops_raw:
        stuv, xyz = _loop_paths(rims, chain)
        area1 = _signed_area(stuv[:, :2])
        loops.append({"chain": chain, "stuv": stuv, "xyz": xyz,
                      "area1": area1})
    loops.sort(key=lambda L: -abs(L["area1"]))

    def _contains(La, Lb):
        return _point_in_polygon(Lb["stuv"][0, :2], La["stuv"][:, :2])

    # Outer loops = loops contained in no other loop; each hole is
    # assigned to its smallest containing outer.  Disjoint coincidence
    # patches therefore become SEPARATE regions instead of silently
    # dropping (no-silent-caps).
    outer_ids = [i for i, L in enumerate(loops)
                 if not any(_contains(loops[j], L)
                            for j in range(len(loops)) if j != i)]
    hole_map = {i: [] for i in outer_ids}
    for i, L in enumerate(loops):
        if i in outer_ids:
            continue
        containers = [j for j in outer_ids if _contains(loops[j], L)]
        if containers:
            hole_map[containers[-1]].append(i)   # smallest (sorted by area)

    def _oriented(L, ccw):
        if (L["area1"] > 0) != ccw:
            chain = [(ri, not rev) for (ri, rev) in reversed(L["chain"])]
            stuv = L["stuv"][::-1].copy()
            xyz = L["xyz"][::-1].copy()
            return {"chain": chain, "stuv": stuv, "xyz": xyz,
                    "area1": -L["area1"]}
        return L

    regions = []
    referenced = []            # rim ids in first-reference order
    for oi in outer_ids:
        # Region on the LEFT in S1's (u,v): outer CCW, holes CW.
        region_loops = ([_oriented(loops[oi], ccw=True)]
                        + [_oriented(loops[h], ccw=False)
                           for h in hole_map[oi]])
        uv1_loops = [L["stuv"][:, :2].copy() for L in region_loops]
        uv2_loops = [L["stuv"][:, 2:].copy() for L in region_loops]

        if not _charge(64 + 16):
            return empty
        witness, w_resid = _interior_witness(
            S1_h, S2_h, uv1_loops, uv2_loops, atol, ptol4)
        if witness is None:
            # Band rule: no interior clear of every rim by 4*ptol — this
            # candidate stays curve-only (L27 negative control).
            continue

        # Normal agreement at the witness (constant over a connected
        # coincidence region).
        _, du1, dv1 = eval_surface_d1(S1_h, witness[0], witness[1],
                                      rational=True)
        _, du2, dv2 = eval_surface_d1(S2_h, witness[2], witness[3],
                                      rational=True)
        n1 = np.cross(du1, dv1)
        n2 = np.cross(du2, dv2)
        agreement = 1 if float(np.dot(n1, n2)) >= 0.0 else -1
        # Redundant orientation check: with agreeing normals the uv2 loop
        # turns the same way as uv1 (§8 assert-consistency).
        area2 = _signed_area(uv2_loops[0])
        orientation_consistent = (
            (area2 > 0) == ((region_loops[0]["area1"] > 0)
                            == (agreement == 1)))

        resid_max = max(rims[ri]["resid_max"]
                        for L in region_loops for (ri, _rev) in L["chain"])
        n_samples = sum(len(rims[ri]["xyz"])
                        for L in region_loops for (ri, _rev) in L["chain"])
        for L in region_loops:
            for ri, _rev in L["chain"]:
                if ri not in referenced:
                    referenced.append(ri)
        regions.append((region_loops, SSXOverlapRegion(
            boundary=[],           # filled below once rim indices exist
            uv1_loops=uv1_loops,
            uv2_loops=uv2_loops,
            normal_agreement=agreement,
            interior_stuv=np.asarray(witness, dtype=np.float64),
            certification={
                "boundary_resid_max": resid_max / max(atol, 1e-300),
                "interior_resid": w_resid / max(atol, 1e-300),
                "n_samples": int(n_samples),
                "orientation_consistent": bool(orientation_consistent),
            },
        )))

    if not regions:
        return empty

    # Canonical rim branches: every rim referenced by a region loop, in
    # first-reference order; their sampled paths REPLACE the L27 2-point
    # chords (the §8 sampling upgrade).  Existing overlap branches that
    # match a rim are absorbed; the rest stay verbatim (curve-only).
    rim_index = {ri: k for k, ri in enumerate(referenced)}
    rim_branches = [
        SSXBranch(curve=(rims[ri]["stuv"].copy(), rims[ri]["xyz"].copy()),
                  overlap=True, kind="overlap")
        for ri in referenced
    ]
    for region_loops, region in regions:
        region.boundary = [[(rim_index[ri], rev)
                            for (ri, rev) in L["chain"]]
                           for L in region_loops]

    unmatched = []
    for b in existing:
        bxyz = np.asarray(b.curve[1], dtype=np.float64)
        absorbed = any(
            all(_dist_point_polyline(p, rims[ri]["xyz"]) <= 2.0 * atol
                for p in bxyz)
            for ri in referenced)
        if not absorbed:
            unmatched.append(b)

    # Evidence coverage: every overlap box and every uncertified CSX span
    # must be explained by some certified region before the caller may
    # retire the structural reason.
    covered = True
    all_rim_xyz = [rims[ri]["xyz"] for ri in referenced]
    p_bar12 = 8.0 * max(float(ptol4[0]), float(ptol4[1]))
    p_bar34 = 8.0 * max(float(ptol4[2]), float(ptol4[3]))

    def _half_explained(pt2, loops, bar):
        in_region = (_point_in_polygon(pt2, loops[0])
                     and not any(_point_in_polygon(pt2, h)
                                 for h in loops[1:]))
        near_rim = min(_dist_point_polyline_2d(pt2, lp)
                       for lp in loops) <= bar
        return in_region or near_rim

    for box in overlap_boxes or ():
        b = np.asarray(box, dtype=np.float64)
        center = 0.5 * (b[:, 0] + b[:, 1])
        st, uv = center[:2], center[2:]
        explained = False
        for _loops, region in regions:
            # BOTH parameter planes must be explained (adversarial-review
            # confirmed finding, 2026-07-12): a box on a DIFFERENT S2
            # sheet sharing an S1 footprint (folded/self-overlapping S2)
            # must not count as covered by the sheet the region actually
            # represents — same two-sided rule as `_site_in_regions`.
            if (_half_explained(st, region.uv1_loops, p_bar12)
                    and _half_explained(uv, region.uv2_loops, p_bar34)):
                explained = True
                break
        if not explained:
            covered = False
            break
    if covered:
        for curve_ctrl, (t_lo, t_hi), span_rational in (
                uncertified_spans or ()):
            for t in np.linspace(t_lo, t_hi, 9):
                p = eval_curve(np.asarray(curve_ctrl, dtype=np.float64),
                               float(t), rational=span_rational)
                if min(_dist_point_polyline(
                        np.asarray(p, dtype=np.float64), rx)
                       for rx in all_rim_xyz) > 2.0 * atol:
                    covered = False
                    break
            if not covered:
                break

    return {"regions": [r for _loops, r in regions],
            "rim_branches": rim_branches,
            "unmatched_branches": unmatched, "covered": covered}
