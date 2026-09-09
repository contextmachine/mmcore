"""Exact boundary-only SSX when a surface stays on one side of a plane.

A nonzero, one-sign Bernstein height polynomial is strictly nonzero in
the open parameter square. Its zeros are exactly its identically zero
boundary edges and any remaining zero corners. This tier supports affine
source edges and an injective convex polynomial bilinear target plane.
"""
from fractions import Fraction

import numpy as np

from mmcore.numeric._work_budget import REASON_PARAMETER_REPRESENTATION, REASON_WORK_BUDGET
from mmcore.numeric.intersection.csx._planar_overlap import _convex_chart, _sub, _dot, _cross2
from mmcore.numeric.intersection.csx._planar_roots import _bilinear_inverse
from mmcore.numeric.intersection.ssx._ssx_extrusion import _uniform_cartesian
from mmcore.numeric.intersection.ssx._ssx_affine_path import affine_path_representation_bounded
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch, SSXPoint
from mmcore.numeric.intersection.ssx._ssx_singular_candidate import _cross, _evaluate
from mmcore.numeric.intersection.ssx._ssx_opposed_strata import (
    opposed_boundary_strata_ssx)


def _append_planar_junctions(result, source, owner, chart, budget):
    """Shared exact source endpoints with a unique convex target inverse.

    The target UV representative may be algebraic and cannot be treated
    as an exact binary root. Its existence and uniqueness instead follow
    from exact planar inclusion in the already validated convex chart.
    """
    from mmcore.numeric.intersection.ssx._bez_ssx5 import SSXSingularity
    origin, target_normal, axes, quad, edges, orientation = chart
    endpoints, published, aliases = {}, {}, set()
    for index, branch in enumerate(result['branches']):
        stuv, xyz = branch.curve
        for key, end in zip(branch._boundary_source_endpoints, (0,len(stuv)-1)):
            floating = tuple(stuv[end])
            if floating in published and published[floating] != key:
                aliases.add(floating)
            published[floating] = key
            endpoints.setdefault(key, []).append((index,end,stuv[end],xyz[end]))
    for point in result['points']:
        key = point._boundary_source_parameter
        floating = tuple(point.stuv)
        if floating in published and published[floating] != key:
            aliases.add(floating)
        published[floating] = key
    if aliases:
        budget.mark_incomplete(REASON_PARAMETER_REPRESENTATION)
        budget.append_output(result['unresolved_regions'], dict(
            stuv_min=(0.,)*4,stuv_max=(1.,)*4,
            reason=REASON_PARAMETER_REPRESENTATION), 'unresolved_region')
    net = np.asarray(source,dtype=object)
    for key, ports in endpoints.items():
        if len({index for index,*_ in ports}) < 2 or any(tuple(port[2]) in aliases for port in ports):
            continue
        cost = net.size*(1+sum(n-1 for n in net.shape[:2]))
        if not budget.charge_cells(max(1,cost), 'boundary_junction'):
            return
        point = _evaluate(net,key)
        jets = [_evaluate((net.shape[axis]-1)*np.diff(net,axis=axis),key) for axis in (0,1)]
        normal = _cross(*jets)
        projected = tuple(point[k] for k in axes)
        if (not any(normal) or any(_cross(normal,target_normal))
                or _dot(target_normal,_sub(point,origin))
                or any(orientation*_cross2(edge,_sub(projected,anchor)) < 0
                       for edge,anchor in zip(edges,quad))):
            continue
        junction = SSXSingularity(
            kind='tangent_point', stuv=np.asarray(ports[0][2]), xyz=np.asarray(ports[0][3]),
            branch_links=[(index,end) for index,end,*_ in ports])
        junction._source_corner = owner,key
        budget.append_output(result['singularities'],junction,'singularity')


def exact_boundary_strata_ssx(first, second, atol, rational, budget):
    """Return an exhaustive supported result, or None for general search."""
    nets = tuple(np.asarray(net, dtype=float) for net in (first, second))
    if (any(net.ndim != 3 or net.shape[-1] != (4 if rational else 3)
            or min(net.shape[:2]) < 2 or not np.all(np.isfinite(net)) for net in nets)
            or not any(net.shape[:2] == (2, 2) for net in nets)):
        return None
    if not budget.charge_cells(max(1, sum(net.size for net in nets)), 'boundary_strata'):
        return None
    exact = tuple(_uniform_cartesian(net, rational) for net in nets)
    if any(points is None for points in exact):
        return None
    for owner in (0, 1):
        if nets[1-owner].shape[:2] != (2, 2):
            continue
        target = [point for row in exact[1-owner] for point in row]
        chart = _convex_chart(target)
        if chart is None:
            continue
        origin, normal, axes, quad, edges, orientation = chart
        source = exact[owner]
        heights = [[_dot(normal, _sub(point, origin)) for point in row] for row in source]
        flat = [value for row in heights for value in row]
        if not any(flat) or not (all(value >= 0 for value in flat) or all(value <= 0 for value in flat)):
            continue
        zero_edges = []
        unsupported = False
        for fixed_axis in (0, 1):
            for side in (0, 1):
                edge = (source[0 if side == 0 else -1] if fixed_axis == 0
                        else [row[0 if side == 0 else -1] for row in source])
                values = (heights[0 if side == 0 else -1] if fixed_axis == 0
                          else [row[0 if side == 0 else -1] for row in heights])
                if any(values):
                    continue
                degree = len(edge)-1
                direction = _sub(edge[-1], edge[0])
                if not any(direction) or any(
                        any(value != start+Fraction(i, degree)*delta
                            for value, start, delta in zip(point, edge[0], direction))
                        for i, point in enumerate(edge)):
                    unsupported = True
                    break
                zero_edges.append((fixed_axis, side, edge[0], direction))
        if unsupported:
            continue

        p00, p01, p10, p11 = target
        du, dv = _sub(p10, p00), _sub(p01, p00)
        affine = not any(z-a-b+o for z, a, b, o in zip(p11, p10, p01, p00))
        i, j = axes
        determinant = du[i]*dv[j]-du[j]*dv[i]
        target_float = None

        def inverse(point):
            nonlocal target_float
            displacement = _sub(point, p00)
            if affine:
                return ((displacement[i]*dv[j]-displacement[j]*dv[i])/determinant,
                        (du[i]*displacement[j]-du[j]*displacement[i])/determinant)
            projected = tuple(point[k] for k in axes)
            for index, (edge, start) in enumerate(zip(edges, quad)):
                offset = _sub(projected, start)
                if _cross2(edge, offset) == 0:
                    coordinate = max(range(2), key=lambda k: abs(edge[k]))
                    t = offset[coordinate]/edge[coordinate]
                    if 0 <= t <= 1:
                        return ((t, Fraction(0)), (Fraction(1), t),
                                (1-t, Fraction(1)), (Fraction(0), 1-t))[index]
            try:
                if target_float is None:
                    target_float = np.asarray([[list(map(float, p00)), list(map(float, p01))],
                                               [list(map(float, p10)), list(map(float, p11))]])
                with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                    uv = _bilinear_inverse(np.asarray(list(map(float, point))), target_float, axes)
            except (OverflowError, ValueError):
                return None
            return None if uv is None else tuple(Fraction.from_float(value) for value in uv)

        def inside(point):
            projected = tuple(point[k] for k in axes)
            return all(orientation*_cross2(edge, _sub(projected, start)) >= 0
                       for edge, start in zip(edges, quad))

        result = dict(branches=[], points=[], singularities=[], overlap_regions=[], unresolved_regions=[])
        point_keys = set()

        def emit_point(stuv, xyz, exact_key):
            # Clipping two incident edges can discover the same corner.
            # The exact source parameter and target inverse uniqueness
            # establish identity, including a nonbinary target preimage.
            key = tuple(exact_key[2*owner:2*owner+2])
            if key not in point_keys:
                point_keys.add(key)
                point = SSXPoint(stuv, xyz)
                point._boundary_source_parameter = key
                budget.append_output(result['points'], point, 'point')

        def unresolved(reason, detail=None):
            budget.mark_incomplete(reason)
            record = dict(stuv_min=(0.,)*4, stuv_max=(1.,)*4, reason=reason)
            if detail is not None:
                record['boundary_stratum'] = detail
            budget.append_output(result['unresolved_regions'], record, 'unresolved_region')

        for fixed_axis, side, start, direction in zero_edges:
            lo, hi = Fraction(0), Fraction(1)
            projected_start = tuple(start[k] for k in axes)
            projected_direction = tuple(direction[k] for k in axes)
            for edge, anchor in zip(edges, quad):
                value = orientation*_cross2(edge, _sub(projected_start, anchor))
                slope = orientation*_cross2(edge, projected_direction)
                if slope > 0:
                    lo = max(lo, -value/slope)
                elif slope < 0:
                    hi = min(hi, -value/slope)
                elif value < 0:
                    lo, hi = Fraction(1), Fraction(0)
                    break
            if hi < lo:
                continue
            samples, sample_keys = {}, {}
            def sample(parameter):
                if parameter not in samples:
                    point = tuple(value+parameter*delta for value, delta in zip(start, direction))
                    uv = inverse(point)
                    if uv is None:
                        return None
                    own = (Fraction(side), parameter) if fixed_axis == 0 else (parameter, Fraction(side))
                    q = own+uv if owner == 0 else uv+own
                    try:
                        sample_q, sample_xyz = np.asarray(list(map(float, q))), np.asarray(list(map(float, point)))
                    except (OverflowError, ValueError):
                        return None
                    if not np.all(np.isfinite(sample_q)) or not np.all(np.isfinite(sample_xyz)):
                        return None
                    samples[parameter] = sample_q, sample_xyz
                    sample_keys[parameter] = q
                return samples[parameter]
            pending, accepted = [(lo, hi)], []
            while pending:
                if not budget.charge_cells(1, 'boundary_representation'):
                    unresolved(REASON_WORK_BUDGET, (owner, fixed_axis, side))
                    break
                a, b = pending.pop()
                first_sample, second_sample = sample(a), sample(b)
                if first_sample is None or second_sample is None:
                    unresolved(REASON_PARAMETER_REPRESENTATION, (owner, fixed_axis, side))
                    break
                qa, xa = first_sample
                qb, xb = second_sample
                if a < b and float(a) == float(b):
                    # A nonempty exact source interval cannot be published
                    # as a single floating parameter value, even when its
                    # physical length happens to fit inside atol.
                    unresolved(REASON_PARAMETER_REPRESENTATION, (owner, fixed_axis, side))
                    break
                valid = affine_path_representation_bounded(
                    *nets, qa, qb, np.array([xa, xb]), atol, rational=rational,
                    charge=lambda amount: budget.charge_cells(amount, 'boundary_representation'))
                if valid:
                    accepted.append((a, b))
                else:
                    if budget.exhausted:
                        unresolved(REASON_WORK_BUDGET, (owner, fixed_axis, side))
                        break
                    midpoint = (a+b)/2
                    if a == b or float(midpoint) in (float(a), float(b)):
                        unresolved(REASON_PARAMETER_REPRESENTATION, (owner, fixed_axis, side))
                        break
                    pending.extend(((midpoint, b), (a, midpoint)))
            # Already validated intervals remain useful when later work
            # stops. Join only adjacent exact source intervals; never
            # bridge an unprocessed part of the edge.
            groups = []
            for a, b in sorted(accepted):
                if groups and groups[-1][-1] == a:
                    groups[-1].append(b)
                else:
                    groups.append([a, b] if a < b else [a])
            for parameters in groups:
                pairs = [sample(value) for value in parameters]
                stuv = np.array([pair[0] for pair in pairs])
                xyz = np.array([pair[1] for pair in pairs])
                if len(parameters) == 1:
                    emit_point(stuv[0], xyz[0], sample_keys[parameters[0]])
                else:
                    branch = SSXBranch((stuv, xyz), overlap=True, kind='overlap')
                    # The exact source parameter and the convex target's
                    # unique inverse identify the endpoint. Numerical UV
                    # inversion does not supply an exact full 4-D key.
                    branch._boundary_source_endpoints = tuple(
                        sample_keys[value][2*owner:2*owner+2]
                        for value in (parameters[0],parameters[-1]))
                    budget.append_output(result['branches'], branch, 'branch')
        for s, t in ((0, 0), (0, 1), (1, 0), (1, 1)):
            point = source[0 if s == 0 else -1][0 if t == 0 else -1]
            if (_dot(normal, _sub(point, origin)) != 0 or not inside(point)
                    or any((s, t)[axis] == side for axis, side, *_ in zero_edges)):
                continue
            uv = inverse(point)
            own = (Fraction(s), Fraction(t))
            if uv is None:
                unresolved(REASON_PARAMETER_REPRESENTATION, (owner, s, t))
                continue
            q = own+uv if owner == 0 else uv+own
            exact_key = q
            try:
                q, xyz = np.asarray(list(map(float, q))), np.asarray(list(map(float, point)))
            except (OverflowError, ValueError):
                unresolved(REASON_PARAMETER_REPRESENTATION, (owner, s, t))
                continue
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(xyz)):
                unresolved(REASON_PARAMETER_REPRESENTATION, (owner, s, t))
                continue
            if not affine_path_representation_bounded(*nets, q, q, np.array([xyz, xyz]),
                    atol, rational=rational,
                    charge=lambda amount: budget.charge_cells(amount, 'boundary_representation')):
                unresolved(REASON_PARAMETER_REPRESENTATION, (owner, s, t))
                continue
            emit_point(q, xyz, exact_key)
        _append_planar_junctions(result, source, owner, chart, budget)
        return result
    return opposed_boundary_strata_ssx(nets, exact, atol, rational, budget)
