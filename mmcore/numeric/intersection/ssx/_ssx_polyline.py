"""Paired-parameter checks for SSX output coalescing.

World-space proximity alone loses distinct preimages on folded surfaces.
These checks require one correspondence in (s,t,u,v,x,y,z), and test whole
segments rather than only vertices. They concern the returned polylines;
they are not certificates for the topology of the underlying zero set.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np


def _point_matches_exact_parameters(p, q, s, x, xyz_tol, segment_mask):
    """Exact segment incidence and a geometric bound at that correspondence."""
    if (not all(np.all(np.isfinite(a)) for a in (p,q,s,x))
            or not np.isfinite(xyz_tol) or xyz_tol < 0.):
        return False
    floating_p = p
    exact = lambda row: tuple(Fraction.from_float(float(value)) for value in row)
    p,q = exact(p),exact(q)
    limit = Fraction.from_float(float(xyz_tol))**2

    def close(value):
        return sum((a-b)**2 for a,b in zip(value,q)) <= limit

    for i in np.flatnonzero(np.all(s == floating_p,axis=1)):
        if close(exact(x[i])):
            return True
    candidates = np.all((np.minimum(s[:-1],s[1:]) <= floating_p)
                        & (floating_p <= np.maximum(s[:-1],s[1:])),axis=1)
    if segment_mask is not None:
        candidates &= np.asarray(segment_mask,dtype=bool)
    for i in np.flatnonzero(candidates):
        start,end = exact(s[i]),exact(s[i+1])
        delta = tuple(b-a for a,b in zip(start,end))
        world = exact(x[i])
        dx = tuple(b-a for a,b in zip(world,exact(x[i+1])))
        axis = next((j for j,d in enumerate(delta) if d),None)
        if axis is None:
            squared = sum(d*d for d in dx)
            t = (sum((b-a)*d for a,b,d in zip(world,q,dx))/squared
                 if squared else Fraction(0))
            t = max(Fraction(0),min(Fraction(1),t))
        else:
            t = (p[axis]-start[axis])/delta[axis]
            if not 0 <= t <= 1 or any(a+t*d != b for a,d,b in zip(start,delta,p)):
                continue
        if close(tuple(a+t*d for a,d in zip(world,dx))):
            return True
    return False


def point_matches_polyline(stuv, xyz, path_stuv, path_xyz, param_tol,
                           xyz_tol, segment_mask=None):
    """Whether a single point of the polyline meets BOTH error bounds.

Intersect four parameter slabs along each segment, then minimize world
distance over the surviving segment interval. Separate nearest-point
queries in parameter and world space would allow different correspondences.
Periodic jumps must remain separate preimages, not interpolated shortcuts.
"""
    s = np.asarray(path_stuv, dtype=float)
    x = np.asarray(path_xyz, dtype=float)
    if len(s) == 0 or len(s) != len(x):
        return False
    p = np.asarray(stuv, dtype=float)
    q = np.asarray(xyz, dtype=float)
    tol = np.broadcast_to(np.asarray(param_tol, dtype=float), (4,))
    if np.all(tol == 0.):
        return _point_matches_exact_parameters(p,q,s,x,xyz_tol,segment_mask)
    # Explicit vertices remain meaningful on both sides of a seam break.
    if np.any(np.all(np.abs(s - p) <= tol, axis=1)
              & (np.linalg.norm(x - q, axis=1) <= xyz_tol)):
        return True
    if len(s) == 1:
        return bool(np.all(np.abs(s[0] - p) <= tol)
                    and np.linalg.norm(x[0] - q) <= xyz_tol)
    ds = np.diff(s, axis=0)
    lo = np.zeros(len(ds))
    hi = np.ones(len(ds))
    for axis in range(4):
        d = ds[:, axis]
        base = s[:-1, axis] - p[axis]
        moving = d != 0.0
        denom = np.where(moving, d, 1.0)
        a = (-tol[axis] - base) / denom
        b = (tol[axis] - base) / denom
        lo = np.maximum(lo, np.where(moving, np.minimum(a, b), 0.0))
        hi = np.minimum(hi, np.where(moving, np.maximum(a, b), 1.0))
        hi[~moving & (np.abs(base) > tol[axis])] = -1.0
    valid = lo <= hi
    if segment_mask is not None:
        valid &= np.asarray(segment_mask, dtype=bool)
    if not np.any(valid):
        return False
    a = x[:-1][valid]
    dx = np.diff(x, axis=0)[valid]
    squared = np.einsum('ij,ij->i', dx, dx)
    t = np.einsum('ij,ij->i', q - a, dx) / np.where(squared > 0, squared, 1.0)
    t = np.clip(t, lo[valid], hi[valid])
    return bool(np.any(np.linalg.norm(a + t[:, None] * dx - q, axis=1)
                       <= xyz_tol))


def _covered_fraction(p, dp, q, dq, tolerances, exact=False):
    """Project a segment-pair's convex free space onto the first segment.

The free space is the unit square in (a,b) intersected with the linear
inequalities abs(p + a*dp - q - b*dq) <= tolerances. Its projection is an
interval, found by clipping the polygon. Empty means no correspondence.
"""
    zero,one = (Fraction(0),Fraction(1)) if exact else (0.,1.)
    polygon = [(zero,zero),(one,zero),(one,one),(zero,one)]
    for offset, da, db, tol in zip(p - q, dp, -dq, tolerances):
        for sign in (1, -1):
            aa, bb, cc = sign * da, sign * db, tol - sign * offset
            result = []
            previous = polygon[-1]
            old = aa * previous[0] + bb * previous[1] - cc
            for current in polygon:
                new = aa * current[0] + bb * current[1] - cc
                if (old <= 0.0) != (new <= 0.0):
                    fraction = old / (old - new)
                    result.append((previous[0] + fraction * (current[0] - previous[0]),
                                   previous[1] + fraction * (current[1] - previous[1])))
                if new <= 0.0:
                    result.append(current)
                previous, old = current, new
            polygon = result
            if not polygon:
                return None
    values = [v[0] for v in polygon]
    return min(values), max(values)


def polyline_contained(stuv, xyz, keeper_stuv, keeper_xyz, param_tol,
                       xyz_tol, charge=None, source_mask=None, keeper_mask=None):
    """Conservative continuous lifted-polyline containment, or None on cap.

Each source segment must be covered by the union of target-segment free
spaces. The xyz cube has radius xyz_tol/sqrt(3), so acceptance implies the
Euclidean xyz bound. A failure retains the source, including numerical
ambiguities and gaps; there is no gap-closing epsilon. Periodic parameter
seams are intentionally not wrapped into interpolated segments.
"""
    s = np.asarray(stuv, dtype=float)
    x = np.asarray(xyz, dtype=float)
    ks = np.asarray(keeper_stuv, dtype=float)
    kx = np.asarray(keeper_xyz, dtype=float)
    if len(s) == 0 or len(ks) < 2 or len(s) != len(x) or len(ks) != len(kx):
        return False
    ptol = np.broadcast_to(np.asarray(param_tol, dtype=float), (4,))
    bounds = np.concatenate((ptol, np.full(3, xyz_tol / np.sqrt(3.0))))
    p = np.concatenate((s, x), axis=1)
    q = np.concatenate((ks, kx), axis=1)
    if (not np.all(np.isfinite(p)) or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(bounds)) or np.any(bounds < 0)):
        return False
    exact_parameters = bool(np.all(ptol == 0.))
    if exact_parameters:
        if s.shape == ks.shape:
            # Identical sampled parameters provide the complete affine
            # segment correspondence directly. Convexity extends exact
            # endpoint Euclidean error bounds along each matched chord.
            for reverse in (False,True):
                target_s,target_x = (ks[::-1],kx[::-1]) if reverse else (ks,kx)
                if not np.array_equal(s,target_s):
                    continue
                sm = np.ones(len(s)-1,dtype=bool) if source_mask is None else np.asarray(source_mask,dtype=bool)
                km = np.ones(len(s)-1,dtype=bool) if keeper_mask is None else np.asarray(keeper_mask,dtype=bool)
                if reverse:
                    km = km[::-1]
                if np.any(sm & ~km):
                    continue
                if charge is not None and not charge(max(1,len(s))):
                    return None
                limit = Fraction.from_float(float(xyz_tol))**2
                if np.array_equal(x,target_x) or all(
                        sum((Fraction.from_float(float(a))-Fraction.from_float(float(b)))**2
                            for a,b in zip(left,right)) <= limit
                        for left,right in zip(x,target_x)):
                    return True
        # Keep the geometric cube inside the caller's Euclidean ball
        # despite the rounded sqrt/division used to propose its radius.
        limit = Fraction.from_float(float(xyz_tol))**2
        radius = float(bounds[-1])
        while 3*Fraction.from_float(radius)**2 > limit:
            radius = float(np.nextafter(radius,0.))
        bounds[4:] = radius
        exact_bounds = np.array([Fraction.from_float(float(value)) for value in bounds],dtype=object)
        exact_keeper = {}
    if len(s) == 1:
        if charge is not None and not charge(len(ks) - 1):
            return None
        return point_matches_polyline(s[0], x[0], ks, kx, ptol, xyz_tol,
                                      segment_mask=keeper_mask)
    qlo = np.minimum(q[:-1], q[1:]) - bounds
    qhi = np.maximum(q[:-1], q[1:]) + bounds
    for i, (start, end) in enumerate(zip(p[:-1], p[1:])):
        if charge is not None and not charge(len(ks) - 1):
            return None
        if source_mask is not None and not source_mask[i]:
            if not all(point_matches_polyline(s[k], x[k], ks, kx, ptol, xyz_tol,
                                               segment_mask=keeper_mask)
                       for k in (i, i + 1)):
                return False
            continue
        candidate_mask = np.all(
            (np.minimum(start, end) <= qhi) & (np.maximum(start, end) >= qlo), axis=1)
        if keeper_mask is not None:
            candidate_mask &= np.asarray(keeper_mask, dtype=bool)
        candidates = np.flatnonzero(candidate_mask)
        intervals = []
        reach = Fraction(0) if exact_parameters else 0.0
        if exact_parameters:
            exact_start,exact_end = (np.array([Fraction.from_float(float(v)) for v in row],dtype=object)
                                     for row in (start,end))
        for j in candidates:
            # Polygon clipping is scalar work (14 half-planes), unlike
            # the vectorized bbox scan. Charge one 128-operation block.
            if charge is not None and not charge(128):
                return None
            if exact_parameters:
                if j not in exact_keeper:
                    exact_keeper[j] = tuple(np.array([Fraction.from_float(float(v)) for v in row],dtype=object)
                                            for row in (q[j],q[j+1]))
                a,b = exact_keeper[j]
                covered = _covered_fraction(exact_start,exact_end-exact_start,a,b-a,exact_bounds,exact=True)
            else:
                covered = _covered_fraction(start, end - start, q[j], q[j + 1] - q[j], bounds)
            if covered is not None:
                intervals.append(covered)
            reach = Fraction(0) if exact_parameters else 0.0
            for lo, hi in sorted(intervals):
                if lo > reach:
                    break
                reach = max(reach, hi)
            if reach >= 1.0:
                break
        if reach < 1.0:
            return False
    return True
