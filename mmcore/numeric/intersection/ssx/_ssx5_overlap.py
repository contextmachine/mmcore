"""CAD surface-overlap regions at the requested modeling tolerance.

Closed rim loops reference ``kind='overlap'`` branches and carry paired
parameter paths, an interior sample and normal orientation. Convex planar
bilinear charts use polygon clipping with shared corner inverses and
adaptive, whole-chord checks on both surfaces. Other patches use sampled
domain edges and numerical point inversion.

A coincidence band becomes a region only when an interior sample stays
at least 4*ptol from its rims in both parameter planes. Thinner evidence
remains curve-only. The ``certification`` fields report numerical checks
at atol; they are not an exact-algebraic intersection contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np
from numpy.typing import NDArray

from mmcore.numeric._bezier_common import (
    eval_curve, eval_surface, eval_surface_d1, restrict_net_axis_v,
)
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch

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


class _OverlapWorkStopped(Exception):
    """The caller's shared overlap allowance was exhausted."""


def _bilinear_chord_error(S_h, start, end, xyz):
    """Bernstein hull bound for a bilinear chart along one parameter chord.

    Substituting affine u and v gives a degree-two homogeneous curve. Its
    difference from the reported XYZ chord is a degree-three numerator;
    positive weights bound the rational error over the whole interval.
    This is floating point geometry at modeling tolerance, not an exact
    identity predicate.
    """
    H = np.asarray(S_h, dtype=float).copy()
    H /= np.max(H[..., 3])
    H[..., :3] -= np.asarray(xyz[0])*H[..., 3, None]

    def value(u, v):
        return ((1.-u)*((1.-v)*H[0, 0]+v*H[0, 1])
                + u*((1.-v)*H[1, 0]+v*H[1, 1]))

    u0, v0 = start
    u1, v1 = end
    c = np.array([value(u0, v0), .5*(value(u0, v1)+value(u1, v0)),
                  value(u1, v1)])
    if not np.isfinite(c).all() or np.min(c[:, 3]) <= 0.:
        return np.inf
    elevated = np.array([c[0, :3], (c[0, :3]+2.*c[1, :3])/3.,
                         (2.*c[1, :3]+c[2, :3])/3., c[2, :3]])
    delta = np.asarray(xyz[1])-xyz[0]
    line = np.array([np.zeros(3), c[0, 3]*delta/3.,
                     2.*c[1, 3]*delta/3., c[2, 3]*delta])
    return float(np.max(np.linalg.norm(elevated-line, axis=1))/np.min(c[:, 3]))


def _surface_roundoff(surface, xyz=()):
    """Arithmetic scale for closed CAD-distance comparisons."""
    net = np.asarray(surface, dtype=float)
    coordinates = net[..., :3]/net[..., 3, None]
    return 128.*np.finfo(float).eps*max(
        1., float(np.max(abs(coordinates))),
        float(np.max(np.abs(xyz))) if np.size(xyz) else 0.)


def _surface_chord_error(surface, start, end, xyz):
    """Bound an arbitrary-degree rational chart over one affine UV chord.

    Restriction to the UV rectangle followed by its diagonal gives a
    degree p+q homogeneous curve. Subtracting the reported XYZ chord times
    its weight gives a Bernstein numerator bound over the whole segment.
    """
    net = np.asarray(surface, dtype=float).copy()
    if not np.isfinite(net).all() or np.min(net[..., 3]) <= 0.:
        return np.inf
    net /= np.max(net[..., 3])
    origin = np.asarray(xyz[0], dtype=float)
    net[..., :3] -= origin * net[..., 3, None]
    for axis in range(2):
        a, b = float(start[axis]), float(end[axis])
        if not (0. <= a <= 1. and 0. <= b <= 1.):
            return np.inf
        net = restrict_net_axis_v(net, axis, min(a, b), max(a, b), 0., 1.)
        if b < a:
            net = np.flip(net, axis=axis)
    p, q = np.asarray(net.shape[:2])-1
    degree = p+q
    curve = np.zeros((degree+1, 4))
    for i in range(p+1):
        for j in range(q+1):
            curve[i+j] += (comb(p, i)*comb(q, j)/comb(degree, i+j))*net[i, j]
    if not np.isfinite(curve).all() or np.min(curve[:, 3]) <= 0.:
        return np.inf
    fraction = np.arange(degree+2)/(degree+1.)
    elevated = np.zeros((degree+2, 3))
    elevated[:-1] += (1.-fraction[:-1, None])*curve[:, :3]
    elevated[1:] += fraction[1:, None]*curve[:, :3]
    chord = np.zeros_like(elevated)
    chord[1:] = (fraction[1:]*curve[:, 3])[:, None]*(np.asarray(xyz[1])-origin)
    bound = float(np.max(np.linalg.norm(elevated-chord, axis=1))/np.min(curve[:, 3]))
    return bound+_surface_roundoff(surface, xyz)


def _planar_convex_rims(S1_h, S2_h, atol, charge, context, *, include_boundary_contacts=False):
    """Clip convex bilinear chart images before inverting their shared rims.

    Positive-weight planar bilinear charts map the parameter square onto
    their corner quadrilateral. Clipping those two quadrilaterals supplies
    common corner locations, including oblique intersections which separate
    edge-by-edge tolerance searches can miss. The calculation is numerical:
    planarity, inverse residuals and both lifted chord images are checked at
    the caller's modeling tolerance. Unsupported charts return ``None`` so
    the general sampled-rim path remains available.
    """
    if S1_h.shape != (2, 2, 4) or S2_h.shape != (2, 2, 4):
        return None

    def spend(n=1):
        if not charge(n):
            raise _OverlapWorkStopped

    spend(16)
    if any(not np.isfinite(S).all() or np.any(S[..., 3] <= 0.)
           for S in (S1_h, S2_h)):
        return None
    cart = [S[..., :3] / S[..., 3, None] for S in (S1_h, S2_h)]
    points = np.concatenate([p.reshape(-1, 3) for p in cart])
    origin = points[0].copy()
    scale = float(np.max(np.ptp(points, axis=0)))
    if not np.isfinite(scale) or scale <= 0.:
        return None
    centered = (points - origin) / scale
    try:
        _u, singular, frame = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if singular[1] <= 128. * np.finfo(float).eps * singular[0]:
        return None
    # Reserve the rest of atol for inversion and chord representation.
    if np.max(np.abs(centered @ frame[2])) * scale > .25 * atol:
        return None
    basis = frame[:2]
    projected = [((p - origin) / scale) @ basis.T for p in cart]
    eps = 128. * np.finfo(float).eps

    def cross(a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    def quad(p):
        q = p[[0, 1, 1, 0], [0, 0, 1, 1]].copy()
        turns = np.array([cross(q[(i+1) % 4]-q[i],
                                q[(i+2) % 4]-q[(i+1) % 4])
                          for i in range(4)])
        if np.all(turns < -eps):
            q = q[::-1].copy()
        elif not np.all(turns > eps):
            return None
        return q

    first, second = map(quad, projected)
    if first is None or second is None:
        return None

    # Sutherland-Hodgman intersection. The epsilon only handles roundoff
    # on a shared supporting line; it does not expand either CAD polygon.
    polygon = list(first)
    for a, b in zip(second, np.roll(second, -1, axis=0)):
        clipped = []
        if not polygon:
            break
        before = polygon[-1]
        db = cross(b-a, before-a)
        for after in polygon:
            spend()
            da = cross(b-a, after-a)
            if (db >= -eps) != (da >= -eps):
                denominator = db-da
                if denominator != 0.:
                    fraction = np.clip(db/denominator, 0., 1.)
                    clipped.append(before + fraction*(after-before))
            if da >= -eps:
                clipped.append(after)
            before, db = after, da
        polygon = []
        for p in clipped:
            if not polygon or np.linalg.norm(p-polygon[-1]) > eps:
                polygon.append(p)
        if len(polygon) > 1 and np.linalg.norm(polygon[0]-polygon[-1]) <= eps:
            polygon.pop()
    polygon = np.asarray(polygon).reshape(-1, 2)
    area = abs(sum(cross(a, b) for a, b in
                   zip(polygon, np.roll(polygon, -1, axis=0))))
    if len(polygon) >= 3 and area > eps:
        dimension = 2
    else:
        dimension = 0
        if len(polygon) > 1:
            distance = np.linalg.norm(polygon[:, None]-polygon[None, :], axis=-1)
            a, b = np.unravel_index(np.argmax(distance), distance.shape)
            if distance[a, b] > eps:
                polygon = polygon[[a, b]]
                dimension = 1
            else:
                polygon = polygon[:1]
        context.update(origin=origin, scale=scale, basis=basis,
                       polygon=polygon, dimension=dimension)
        if not include_boundary_contacts or not len(polygon):
            return []

    # Use normalized projected homogeneous coordinates for Newton. This
    # avoids a world translation or units choice entering its conditioning.
    weights = [S[..., 3]/np.max(S[..., 3]) for S in (S1_h, S2_h)]
    projected_h = [np.concatenate((p*w[..., None], w[..., None]), axis=2)
                   for p, w in zip(projected, weights)]

    def inverse(H, target, seed):
        uv = np.array([.5, .5]) if seed is None else np.asarray(seed).copy()
        for _ in range(30):
            spend()
            u, v = uv
            row0 = (1.-v)*H[0, 0]+v*H[0, 1]
            row1 = (1.-v)*H[1, 0]+v*H[1, 1]
            value = (1.-u)*row0+u*row1
            if not np.isfinite(value).all() or value[2] <= 0.:
                return None
            p = value[:2]/value[2]
            error = p-target
            if np.linalg.norm(error) <= eps:
                return uv
            hu = row1-row0
            hv = (1.-u)*(H[0, 1]-H[0, 0])+u*(H[1, 1]-H[1, 0])
            jac = np.column_stack(((hu[:2]-p*hu[2])/value[2],
                                   (hv[:2]-p*hv[2])/value[2]))
            try:
                step = np.linalg.solve(jac, error)
            except np.linalg.LinAlgError:
                return None
            new_uv = np.clip(uv-step, 0., 1.)
            if np.array_equal(new_uv, uv):
                return uv if np.linalg.norm(error)*scale <= .05*atol else None
            uv = new_uv
        return None

    def sample(p, seed=None):
        pair = [inverse(H, p, None if seed is None else seed[2*i:2*i+2])
                for i, H in enumerate(projected_h)]
        if any(q is None for q in pair):
            return None
        q = np.concatenate(pair)
        spend(2)
        x1 = eval_surface(S1_h, *q[:2], rational=True)
        x2 = eval_surface(S2_h, *q[2:], rational=True)
        residual = float(np.linalg.norm(x1-x2))
        if not np.isfinite(residual) or residual > .6*atol:
            return None
        return q, .5*(x1+x2), residual

    corners = [sample(p) for p in polygon]
    if any(p is None for p in corners):
        return None
    if dimension == 0:
        context['contact'] = corners[0]
        return []
    rims = []
    for edge, (a, b) in enumerate(zip(polygon, np.roll(polygon, -1, axis=0))):
        if dimension == 1 and edge > 0:
            break
        samples = [corners[edge]]
        residual = samples[0][2]

        def append_interval(lo, hi, left, right):
            nonlocal residual
            pending = [(lo, hi, left, right)]
            while pending:
                low, high, start, end = pending.pop()
                spend(2)
                good = all(_bilinear_chord_error(S, start[0][off:off+2],
                                                 end[0][off:off+2],
                                                 (start[1], end[1])) <= .8*atol
                           for S, off in ((S1_h, 0), (S2_h, 2)))
                if good:
                    samples.append(end)
                    residual = max(residual, end[2])
                    continue
                middle = .5*(low+high)
                if middle == low or middle == high:
                    return False
                mid = sample(a+middle*(b-a), .5*(start[0]+end[0]))
                if mid is None:
                    return False
                pending.extend(((middle, high, mid, end), (low, middle, start, mid)))
            return True

        for k in range(1, 17):
            end = (corners[(edge+1) % len(corners)] if k == 16
                   else sample(a+(k/16.)*(b-a), samples[-1][0]))
            if end is None or not append_interval((k-1)/16., k/16., samples[-1], end):
                return None
        rims.append({"stuv": np.asarray([s[0] for s in samples]),
                     "xyz": np.asarray([s[1] for s in samples]),
                     "resid_max": residual})
    context.update(origin=origin, scale=scale, basis=basis, polygon=polygon,
                   dimension=dimension)
    return rims


def try_planar_intersection(S1_h, S2_h, atol, budget):
    """Resolve convex planar chart intersections in dimensions zero to two.

    A verified positive-area polygon owns the whole pair's intersection.
    Its interior is not a one-dimensional tangency remainder to subdivide.
    Unsupported charts still return ``None`` to the general SSX search.
    """
    from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch, SSXPoint
    context = {}
    try:
        rims = _planar_convex_rims(
            S1_h, S2_h, atol,
            lambda n: budget.charge_cells(n, 'planar_contact'), context,
            include_boundary_contacts=True)
    except _OverlapWorkStopped:
        return None
    if rims is None or not context:
        return None
    result = dict(branches=[], points=[], singularities=[], overlap_regions=[])

    def retain_rims():
        for rim in rims:
            branch = SSXBranch(curve=(rim['stuv'], rim['xyz']),
                               kind='overlap', overlap=True)
            budget.append_output(result['branches'], branch, 'planar_contact')

    if context['dimension'] == 2:
        from mmcore.nurbs._nurbs_param_tol import bez_surface_param_tolerance
        ptol4 = np.array([*bez_surface_param_tolerance(S1_h, atol, rational=True),
                         *bez_surface_param_tolerance(S2_h, atol, rational=True)])
        assembled = assemble_overlap_regions(
            S1_h, S2_h, atol=atol, ptol4=ptol4,
            charge=lambda n: budget.charge_cells(n, 'planar_contact'),
            _planar_data=(context, rims))
        if not assembled['regions'] or not assembled['planar_pair_covered']:
            if budget.exhausted:
                # Every rim already passed paired source and chord checks.
                # A denied interior-witness step cannot erase that geometry.
                retain_rims()
                return result
            return None
        budget.extend_output(result['branches'], assembled['rim_branches'], 'planar_contact')
        # A region's boundary indices refer to this complete rim list.
        # Preserve partial rim geometry on output denial, without emitting
        # a region whose references would point beyond the returned list.
        if len(result['branches']) == len(assembled['rim_branches']):
            budget.extend_output(result['overlap_regions'], assembled['regions'], 'planar_contact')
        return result
    retain_rims()
    if 'contact' in context:
        q, xyz, _ = context['contact']
        budget.append_output(result['points'], SSXPoint(q, xyz), 'planar_contact')
    return result


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
    membership_tol = atol+_surface_roundoff(S_own_h)+_surface_roundoff(S_other_h)
    if not np.isfinite(membership_tol):
        return []
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
        seed = (u, v) if resid <= membership_tol else None
        onsurf[k] = np.isfinite(resid) and resid <= membership_tol
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
    membership_tol = atol+_surface_roundoff(S_own_h)+_surface_roundoff(S_other_h)
    lo, hi = (w_in, w_out)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        uo, vo = _edge_own_uv(axis, side, mid)
        xyz = eval_surface(S_own_h, uo, vo, rational=True)
        _, _, resid = _invert_point(S_other_h, xyz, seed=tuple(seed))
        if np.isfinite(resid) and resid <= membership_tol:
            lo = mid
        else:
            hi = mid
    return lo


def _resample_rim(S_own_h, S_other_h, owner, axis, side, atol,
                  w_lo, w_hi, dense):
    if w_hi - w_lo <= 1e-12:
        return None
    membership_tol = atol+_surface_roundoff(S_own_h)+_surface_roundoff(S_other_h)
    if not np.isfinite(membership_tol):
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
        if not (np.isfinite(resid) and resid <= membership_tol):
            return None       # the refined span must certify end-to-end
        seed = (u, v)
        resid_max = max(resid_max, resid)
        stuv[k] = _stuv_sample(owner, (uo, vo), (u, v))
        # Use the same paired-point convention at discovery, corrected
        # corners, and adaptive midpoints. Mixing owner points with averages
        # can leave an irreducible bound at an old sample even when the
        # complete two-surface gap is smaller than the requested tolerance.
        xyz[k] = .5*(p+eval_surface(S_other_h, u, v, rational=True))
    if float(np.linalg.norm(xyz[-1] - xyz[0])) < atol and dense > 2:
        return None           # degenerate (corner-touch) span
    return {"stuv": stuv, "xyz": xyz, "resid_max": resid_max,
            "owner": owner, "axis": axis, "side": side}


# ---------------------------------------------------------------------------
# Loop assembly
# ---------------------------------------------------------------------------

def _joint_rim_corner(first, second, rim_a, end_a, rim_b, end_b,
                      atol, ptol4, charge):
    """Correct a pair of rim ends on their actual source-domain edges.

    Opposite-chart clamping identifies the other incident edge. Native
    source corners stay fixed. Other coordinates stay in the endpoint's
    half of its own observed rim span, so Newton cannot jump to the rim's
    other corner or to a disjoint contact span.
    """
    if rim_a['owner'] == rim_b['owner']:
        return None
    rims, ends = ((rim_a, rim_b), (end_a, end_b)) if rim_a['owner'] == 1 else ((rim_b, rim_a), (end_b, end_a))
    endpoint = [np.asarray(rim['stuv'][end], dtype=float) for rim, end in zip(rims, ends)]
    tolerance = np.minimum(.25, np.maximum(np.asarray(ptol4), 64.*np.finfo(float).eps))
    for owner in (0, 1):
        axis = 2*owner+rims[owner]['axis']
        if abs(endpoint[1-owner][axis]-rims[owner]['side']) > tolerance[axis]:
            return None
    point = np.r_[endpoint[0][:2], endpoint[1][2:]]
    fixed = np.zeros(4, dtype=bool)
    limits = np.array([[0., 1.]]*4)
    for owner, (rim, end) in enumerate(zip(rims, ends)):
        axis = 2*owner+rim['axis']
        free = 2*owner+1-rim['axis']
        point[axis] = rim['side']
        fixed[axis] = True
        low, high = sorted(rim['stuv'][[0, -1], free])
        middle = .5*(low+high)
        limits[free] = (low, middle) if end == 0 else (middle, high)
        if point[free] in (0., 1.):
            fixed[free] = True
    free_axes = np.flatnonzero(~fixed)
    surfaces = (first, second)

    def evaluate(parameters):
        values = [eval_surface_d1(surface, *parameters[2*i:2*i+2], rational=True)
                  for i, surface in enumerate(surfaces)]
        residual = values[0][0]-values[1][0]
        jacobian = np.column_stack((values[0][1], values[0][2],
                                    -values[1][1], -values[1][2]))[:, free_axes]
        return values, residual, jacobian

    converged = not len(free_axes)
    for _ in range(32):
        if not charge(1):
            raise _OverlapWorkStopped
        values, residual, jacobian = evaluate(point)
        if not len(free_axes):
            break
        try:
            singular = np.linalg.svd(jacobian, compute_uv=False)
            if singular[-1] <= 64.*np.finfo(float).eps*singular[0]:
                return None  # parallel/coincident edges do not define a corner
            step = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        projected = np.clip(point[free_axes]+step, 0., 1.)-point[free_axes]
        if np.max(abs(projected)) <= 128.*np.finfo(float).eps:
            converged = True
            break
        previous = float(residual@residual)
        advanced = False
        for exponent in range(12):
            candidate = point.copy()
            candidate[free_axes] = np.clip(point[free_axes]+step*2.**(-exponent),
                                            limits[free_axes, 0], limits[free_axes, 1])
            other = [eval_surface(surface, *candidate[2*i:2*i+2], rational=True)
                     for i, surface in enumerate(surfaces)]
            error = other[0]-other[1]
            if float(error@error) < previous:
                point = candidate
                advanced = True
                break
        if not advanced:
            break
    if not converged:
        return None
    values, residual, _ = evaluate(point)
    membership_tol = atol+_surface_roundoff(first)+_surface_roundoff(second)
    if (not np.isfinite(point).all() or not np.isfinite(residual).all()
            or not np.isfinite(membership_tol)
            or np.linalg.norm(residual) > membership_tol):
        return None
    xyz = .5*(values[0][0]+values[1][0])
    score = sum(float(np.linalg.norm(point[2*i:2*i+2]-endpoint[i][2*i:2*i+2]))
                for i in (0, 1))
    return point, xyz, score


def _refine_curved_rim_corners(rims, first, second, atol, ptol4, charge):
    """Share jointly corrected corners before endpoint graph construction."""
    proposals = []
    for i, rim in enumerate(rims):
        for j in range(i+1, len(rims)):
            other = rims[j]
            if rim['owner'] == other['owner']:
                continue
            for end_i in (0, -1):
                for end_j in (0, -1):
                    result = _joint_rim_corner(first, second, rim, end_i, other,
                                               end_j, atol, ptol4, charge)
                    if result is not None:
                        point, xyz, score = result
                        proposals.append((score, i, end_i, j, end_j, point, xyz))
    used = set()
    for _, i, end_i, j, end_j, point, xyz in sorted(proposals, key=lambda p: p[0]):
        if (i, end_i) in used or (j, end_j) in used:
            continue
        for index, end in ((i, end_i), (j, end_j)):
            rim = rims[index]
            rim['stuv'][end] = point
            rim['xyz'][end] = xyz
            used.add((index, end))
    return rims


def _adaptive_curved_rim(rim, first, second, atol, charge):
    """Refine a paired rim until both lifted parameter chords meet atol."""
    owner = rim['owner']-1
    own, other = (first, second) if owner == 0 else (second, first)
    roundoff = _surface_roundoff(first, rim['xyz'])+_surface_roundoff(second, rim['xyz'])
    if not np.isfinite(roundoff) or roundoff >= atol:
        return None  # this arithmetic cannot resolve the requested accuracy
    free = 2*owner+1-rim['axis']
    start, end = np.asarray(rim['stuv'][0]), np.asarray(rim['stuv'][-1])
    # Correcting an endpoint retracts its tolerance fringe. Remove old
    # samples outside the new interval before refining the retained path.
    low, high = sorted((start[free], end[free]))
    samples = [(q.copy(), x.copy()) for q, x in zip(rim['stuv'], rim['xyz'])
               if low <= q[free] <= high]
    if len(samples) < 2:
        return None
    result = [samples[0]]
    pending = list(reversed(list(zip(samples[:-1], samples[1:]))))
    residual_max = 0.
    while pending:
        a, b = pending.pop()
        if not charge(1):
            raise _OverlapWorkStopped
        errors = [_surface_chord_error(surface, a[0][2*i:2*i+2], b[0][2*i:2*i+2],
                                         np.array([a[1], b[1]]))
                  for i, surface in enumerate((first, second))]
        # The sum also bounds the distance between the two lifted source
        # chords. Individual source errors alone could add up to 2*atol.
        # Each bound already includes its arithmetic pad. A second pad
        # covers evaluation/comparison rounding at the CLOSED atol boundary;
        # otherwise a constant gap equal to atol subdivides forever.
        if not np.isfinite(errors).all():
            return None
        if sum(errors) <= atol+2.*roundoff:
            result.append(b)
            continue
        value = .5*(a[0][free]+b[0][free])
        if value == a[0][free] or value == b[0][free]:
            return None
        own_uv = _edge_own_uv(rim['axis'], rim['side'], value)
        xyz = eval_surface(own, *own_uv, rational=True)
        opposite = slice(2*(1-owner), 2*(1-owner)+2)
        seed = .5*(a[0][opposite]+b[0][opposite])
        u, v, residual = _invert_point(other, xyz, seed=seed)
        if not (np.isfinite(residual) and residual <= atol+roundoff):
            return None
        point = _stuv_sample(owner+1, own_uv, (u, v))
        projected = eval_surface(other, u, v, rational=True)
        middle = (np.asarray(point), .5*(xyz+projected))
        residual_max = max(residual_max, residual)
        pending.extend(((middle, b), (a, middle)))
    path = np.asarray([q for q, _ in result])
    xyz = np.asarray([x for _, x in result])
    for parameters, point in zip(path, xyz):
        source_points = []
        for side, surface in enumerate((first, second)):
            source_point = eval_surface(
                surface, *parameters[2*side:2*side+2], rational=True)
            source_points.append(source_point)
            residual_max = max(residual_max, float(np.linalg.norm(source_point-point)))
        residual_max = max(residual_max,
                           float(np.linalg.norm(source_points[0]-source_points[1])))
    return dict(rim, stuv=path, xyz=xyz, resid_max=residual_max)


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
        if (idx > 0 and np.array_equal(stuv_parts[-1][-1], stuv[0])
                and np.array_equal(xyz_parts[-1][-1], xyz[0])):
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
    existing_intersection_branches=(),
    uncertified_spans=(),
    overlap_boxes=(),
    charge=None,
    _planar_data=None,
    _completed_rims=None,
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
    intersections = list(existing_intersection_branches)

    def _charge(n):
        return charge(n) if charge is not None else True

    empty = {"regions": [], "rim_branches": [],
             "unmatched_branches": existing,
             "unmatched_intersection_branches": intersections,
             "planar_pair_covered": False, "covered": False}

    if _planar_data is None:
        planar = {}
        try:
            rims = _planar_convex_rims(S1_h, S2_h, atol, _charge, planar)
        except _OverlapWorkStopped:
            return empty
    else:
        # The early planar entry in this module has already validated and
        # sampled these exact sources. Reuse its work, not a second clip.
        planar, rims = _planar_data
    if rims is None:
        # 8 edges x (coarse + dense) inversions, each a bounded GN solve.
        if not _charge(8 * 33 + 8 * 17):
            return empty
        rims = []
        for owner, axis, side in _EDGES:
            own = S1_h if owner == 1 else S2_h
            other = S2_h if owner == 1 else S1_h
            rims.extend(_sample_edge_rims(own, other, owner, axis, side, atol))
        try:
            rims = _refine_curved_rim_corners(rims, S1_h, S2_h, atol, ptol4, _charge)
            rims = _dedup_rims(rims, atol)
            rims = _peel_dangling_rims(rims, atol)
            refined = []
            for rim in rims:
                validated = _adaptive_curved_rim(rim, S1_h, S2_h, atol, _charge)
                if validated is None:
                    return empty
                refined.append(validated)
                if _completed_rims is not None:
                    # Private progress sink: only whole-chord-validated
                    # paired paths survive a later work-budget denial.
                    _completed_rims.append(validated)
        except _OverlapWorkStopped:
            return empty
        if any(rim is None for rim in refined):
            return empty
        rims = refined
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

    def _inside_planar_pair(branch):
        """A checked paired path inside the complete convex planar region."""
        if not planar:
            return False
        stuv, xyz = map(lambda a: np.asarray(a, dtype=float), branch.curve)
        if (stuv.ndim != 2 or stuv.shape != (len(xyz), 4)
                or xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2
                or not np.isfinite(stuv).all() or not np.isfinite(xyz).all()
                or np.any(stuv < 0.) or np.any(stuv > 1.)):
            return False
        if not _charge(2*len(stuv)):
            return False
        p = ((xyz-planar['origin'])/planar['scale']) @ planar['basis'].T
        polygon = planar['polygon']
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = b-a
            signed = edge[0]*(p[:, 1]-a[1])-edge[1]*(p[:, 0]-a[0])
            if np.any(signed < -atol/planar['scale']*np.linalg.norm(edge)):
                return False
        # These are discovery samples of a 2-D component, not a separate
        # 1-D topology. Validate their paired membership; their obsolete
        # interpolation is replaced by the region's mapped boundary. In
        # particular a coarse diagonal parameter chord need not accurately
        # represent a curved interior trace to identify the owned component.
        for q, x in zip(stuv, xyz):
            for surface, off in ((S1_h, 0), (S2_h, 2)):
                error = np.linalg.norm(eval_surface(surface, *q[off:off+2],
                                                    rational=True)-x)
                if not np.isfinite(error) or error > atol:
                    return False
        return True

    unmatched_intersections = [b for b in intersections if not _inside_planar_pair(b)]
    unmatched = []
    for b in existing:
        bxyz = np.asarray(b.curve[1], dtype=np.float64)
        absorbed = _inside_planar_pair(b) if planar else any(
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
            "unmatched_branches": unmatched,
            "unmatched_intersection_branches": unmatched_intersections,
            "planar_pair_covered": bool(planar and not unmatched
                                         and not unmatched_intersections),
            "covered": covered}
