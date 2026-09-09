"""Exact boundary correspondences forced by opposed supporting halfspaces.

The certificate is a separating plane and one-sign Bernstein heights on
both source charts. Every common point must have height zero on each
chart. Nonzero one-sign heights vanish only on zero edges and corners;
this bounded tier intersects all such affine strata in lifted parameters.
"""
from fractions import Fraction
from itertools import combinations

import numpy as np

from mmcore.numeric._work_budget import REASON_PARAMETER_REPRESENTATION, REASON_WORK_BUDGET
from mmcore.numeric.intersection.ssx._ssx_affine_path import affine_path_representation_bounded
from mmcore.numeric.intersection.ssx._ssx_singular_candidate import _cross, _evaluate
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch, SSXPoint


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _strata(source, heights):
    """Each stratum is (parameter start/direction, XYZ start/direction)."""
    zero = Fraction(0)
    strata, edges = [], []
    for axis in (0, 1):
        for side in (0, 1):
            edge = (source[0 if side == 0 else -1] if axis == 0
                    else [row[0 if side == 0 else -1] for row in source])
            height = (heights[0 if side == 0 else -1] if axis == 0
                      else [row[0 if side == 0 else -1] for row in heights])
            if any(height):
                continue
            delta = _sub(edge[-1], edge[0])
            degree = len(edge)-1
            if not any(delta) or any(
                    point != tuple(a+Fraction(i, degree)*b for a, b in zip(edge[0], delta))
                    for i, point in enumerate(edge)):
                return None  # A collapsed/curved edge needs its own solver.
            start_uv = (Fraction(side), zero) if axis == 0 else (zero, Fraction(side))
            delta_uv = (zero, Fraction(1)) if axis == 0 else (Fraction(1), zero)
            strata.append((start_uv, delta_uv, edge[0], delta))
            edges.append((axis, side))
    for s, t in ((0, 0), (0, 1), (1, 0), (1, 1)):
        i, j = (0 if s == 0 else -1), (0 if t == 0 else -1)
        if heights[i][j] or any((s, t)[axis] == side for axis, side in edges):
            continue
        strata.append(((Fraction(s), Fraction(t)), (zero, zero), source[i][j], (zero,)*3))
    return strata


def _intersect_strata(first, second):
    """Exact parameter intervals, including singleton intersections."""
    _, _, p, d = first
    _, _, q, e = second
    zero, one = Fraction(0), Fraction(1)
    offset = _sub(q, p)
    if not any(d):
        if not any(e):
            return ((zero, zero), (zero, zero)) if p == q else None
        axis = max(range(3), key=lambda k: abs(e[k]))
        v = -offset[axis]/e[axis]
        return ((zero, zero), (v, v)) if 0 <= v <= 1 and all(
            p[k] == q[k]+v*e[k] for k in range(3)) else None
    if not any(e):
        reverse = _intersect_strata(second, first)
        return None if reverse is None else reverse[::-1]
    cross = _cross(d, e)
    if any(cross):
        omitted = max(range(3), key=lambda k: abs(cross[k]))
        i, j = [k for k in range(3) if k != omitted]
        determinant = d[i]*e[j]-d[j]*e[i]
        u = (offset[i]*e[j]-offset[j]*e[i])/determinant
        v = (offset[i]*d[j]-offset[j]*d[i])/determinant
        if 0 <= u <= 1 and 0 <= v <= 1 and all(p[k]+u*d[k] == q[k]+v*e[k] for k in range(3)):
            return (u, u), (v, v)
        return None
    if any(_cross(d, offset)):
        return None
    axis = max(range(3), key=lambda k: abs(d[k]))
    a, b = offset[axis]/d[axis], e[axis]/d[axis]
    lo, hi = max(zero, min(a, a+b)), min(one, max(a, a+b))
    return None if hi < lo else ((lo, hi), ((lo-a)/b, (hi-a)/b))


def append_exact_junctions(result, exact, budget):
    """Classify shared exact endpoints; proximity is never root identity."""
    from mmcore.numeric.intersection.ssx._bez_ssx5 import SSXSingularity

    endpoints = {}
    float_keys, aliases = {}, set()
    for index, branch in enumerate(result['branches']):
        q, xyz = branch.curve
        keys = getattr(branch, '_exact_endpoint_keys', None)
        if keys is None:
            continue
        for key, end in zip(keys, (0, len(q)-1)):
            floating = tuple(q[end])
            if floating in float_keys and float_keys[floating] != key:
                aliases.add(floating)
            float_keys[floating] = key
            endpoints.setdefault(key, []).append((index, end, xyz[end]))
    for point in result['points']:
        key = getattr(point, '_exact_parameter_key', None)
        if key is None:
            continue
        floating = tuple(point.stuv)
        if floating in float_keys and float_keys[floating] != key:
            aliases.add(floating)
        float_keys[floating] = key
    if aliases:
        budget.mark_incomplete(REASON_PARAMETER_REPRESENTATION)
        budget.append_output(result['unresolved_regions'], dict(
            stuv_min=(0.,)*4, stuv_max=(1.,)*4,
            reason=REASON_PARAMETER_REPRESENTATION), 'unresolved_region')
    nets = [np.asarray(source, dtype=object) for source in exact]
    for key, ports in endpoints.items():
        if len({index for index, *_ in ports}) < 2:
            continue
        parameters = tuple(Fraction.from_float(float(x)) for x in key)
        if parameters != key or tuple(map(float, key)) in aliases:
            continue  # Publication rounding is not shared root identity.
        cost = sum(net.size*(1+sum(n-1 for n in net.shape[:2])) for net in nets)
        if not budget.charge_cells(max(1, cost), 'boundary_junction'):
            budget.mark_incomplete(REASON_WORK_BUDGET)
            return
        values, normals = [], []
        for owner, net in enumerate(nets):
            uv = parameters[2*owner:2*owner+2]
            values.append(_evaluate(net, uv))
            jets = [_evaluate((net.shape[axis]-1)*np.diff(net, axis=axis), uv) for axis in (0, 1)]
            normals.append(_cross(*jets))
        if values[0] != values[1] or not all(any(normal) for normal in normals) or any(_cross(*normals)):
            continue
        g = SSXSingularity(kind='tangent_point', stuv=np.asarray(key),
                           xyz=np.asarray(ports[0][2]),
                           branch_links=[(index, end) for index, end, _ in ports])
        budget.append_output(result['singularities'], g, 'singularity')


def opposed_boundary_strata_ssx(nets, exact, atol, rational, budget):
    corners = list(dict.fromkeys(source[i][j] for source in exact
                   for i in (0, -1) for j in (0, -1)))
    strata = None
    for origin, second, third in combinations(corners, 3):
        normal = _cross(_sub(second, origin), _sub(third, origin))
        if not any(normal):
            continue
        if not budget.charge_cells(sum(len(row) for source in exact for row in source), 'boundary_halfspace'):
            return None
        heights = [[[_dot(normal, _sub(point, origin)) for point in row] for row in source] for source in exact]
        flat = [[value for row in height for value in row] for height in heights]
        if not all(any(values) for values in flat):
            continue  # A plane target has a two-dimensional zero stratum.
        if not ((all(x >= 0 for x in flat[0]) and all(x <= 0 for x in flat[1]))
                or (all(x <= 0 for x in flat[0]) and all(x >= 0 for x in flat[1]))):
            continue
        strata = [_strata(source, height) for source, height in zip(exact, heights)]
        if all(item is not None for item in strata):
            break
        strata = None
    if strata is None:
        return None
    result = dict(branches=[], points=[], singularities=[], overlap_regions=[], unresolved_regions=[])
    point_candidates = {}
    branch_endpoint_keys = set()

    def unresolved(reason):
        budget.mark_incomplete(reason)
        budget.append_output(result['unresolved_regions'], dict(
            stuv_min=(0.,)*4, stuv_max=(1.,)*4, reason=reason), 'unresolved_region')

    for first in strata[0]:
        for second in strata[1]:
            if not budget.charge_cells(1, 'boundary_correspondence'):
                unresolved(REASON_WORK_BUDGET)
                return result
            ranges = _intersect_strata(first, second)
            if ranges is None:
                continue
            lifted, image = [], []
            for endpoint in (0, 1):
                q = tuple(a+ranges[owner][endpoint]*b for owner, stratum in enumerate((first, second))
                          for a, b in zip(stratum[0], stratum[1]))
                xyz = tuple(a+ranges[0][endpoint]*b for a, b in zip(first[2], first[3]))
                lifted.append(q)
                image.append(xyz)
            try:
                q, xyz = np.asarray(lifted, dtype=float), np.asarray(image, dtype=float)
            except (OverflowError, ValueError):
                unresolved(REASON_PARAMETER_REPRESENTATION)
                continue
            if (not np.all(np.isfinite(q)) or not np.all(np.isfinite(xyz))
                    or (lifted[0] != lifted[1] and np.array_equal(q[0], q[1]))):
                unresolved(REASON_PARAMETER_REPRESENTATION)
                continue
            if not affine_path_representation_bounded(*nets, q[0], q[1], xyz, atol, rational=rational,
                    charge=lambda amount: budget.charge_cells(amount, 'boundary_representation')):
                unresolved(REASON_WORK_BUDGET if budget.exhausted else REASON_PARAMETER_REPRESENTATION)
                continue
            if lifted[0] == lifted[1]:
                point = SSXPoint(q[0], xyz[0])
                point._exact_parameter_key = lifted[0]
                point_candidates[lifted[0]] = point
            else:
                branch_endpoint_keys.update(lifted)
                branch = SSXBranch((q, xyz), overlap=True, kind='overlap')
                branch._exact_endpoint_keys = tuple(lifted)
                budget.append_output(result['branches'], branch, 'branch')
    for key, point in point_candidates.items():
        if key not in branch_endpoint_keys:
            budget.append_output(result['points'], point, 'point')
    append_exact_junctions(result, exact, budget)
    return result
