"""Complete intersections of injective affine charts in one exact plane.

The source coefficients, plane identity, convex polygon clipping, and both
parameter inverses use exact rational arithmetic. This tier owns the entire
common set: a convex region, one segment, one point, or the empty set.
Unsupported charts return None; floating representation failures retain the
exact component in a typed partial result instead of changing its dimension.
"""
from fractions import Fraction

import numpy as np

from mmcore.numeric._work_budget import REASON_OUTPUT_CAP, REASON_PARAMETER_REPRESENTATION
from mmcore.numeric.intersection.csx._planar_overlap import (
    _exact_points, _convex_chart, _sub, _dot, _cross2,
)
from mmcore.numeric.intersection.ssx._ssx5_overlap import SSXOverlapRegion
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch, SSXPoint


def _affine_chart(net, rational):
    data = _exact_points(net, rational)
    if data is None:
        return None
    points, weights = data
    if any(w != weights[0] for w in weights):
        return None
    m, n = (size-1 for size in net.shape[:2])
    p00, p01, p10, p11 = points[0], points[n], points[m*(n+1)], points[-1]
    du, dv = _sub(p10, p00), _sub(p01, p00)
    for i in range(m+1):
        for j in range(n+1):
            expected = tuple(o+Fraction(i, m)*a+Fraction(j, n)*b for o, a, b in zip(p00, du, dv))
            if points[i*(n+1)+j] != expected:
                return None
    chart = _convex_chart([p00, p01, p10, p11])
    if chart is None:
        return None
    return p00, du, dv, chart, [p00, p10, p11, p01]


def _clean_polygon(points, axes):
    clean = []
    for point in points:
        if not clean or point != clean[-1]:
            clean.append(point)
    if len(clean) > 1 and clean[-1] == clean[0]:
        clean.pop()
    if len(clean) < 3:
        return clean
    area = sum(_cross2(tuple(a[k] for k in axes), tuple(b[k] for k in axes))
               for a, b in zip(clean, clean[1:]+clean[:1]))
    if area == 0:
        axis = max(axes, key=lambda k: max(p[k] for p in clean)-min(p[k] for p in clean))
        ends = min(clean, key=lambda p: p[axis]), max(clean, key=lambda p: p[axis])
        return [ends[0]] if ends[0] == ends[1] else list(ends)
    while len(clean) > 3:
        collinear = next((i for i in range(len(clean)) if _cross2(
            tuple(clean[i][k]-clean[i-1][k] for k in axes),
            tuple(clean[(i+1) % len(clean)][k]-clean[i][k] for k in axes)) == 0), None)
        if collinear is None:
            break
        clean.pop(collinear)
    return clean


def _clip_polygon(points, chart):
    _, _, axes, quad, edges, orientation = chart
    for edge, origin in zip(edges, quad):
        if not points:
            break
        output = []
        previous = points[-1]
        previous_value = orientation*_cross2(edge, _sub(tuple(previous[k] for k in axes), origin))
        for current in points:
            current_value = orientation*_cross2(edge, _sub(tuple(current[k] for k in axes), origin))
            if (previous_value >= 0) != (current_value >= 0):
                t = previous_value/(previous_value-current_value)
                output.append(tuple(a+t*(b-a) for a, b in zip(previous, current)))
            if current_value >= 0:
                output.append(current)
            previous, previous_value = current, current_value
        points = _clean_polygon(output, axes)
    return points


def _inverse(chart, point):
    origin, du, dv, data, _ = chart
    i, j = data[2]
    p = _sub(point, origin)
    determinant = du[i]*dv[j]-du[j]*dv[i]
    return ((p[i]*dv[j]-p[j]*dv[i])/determinant,
            (du[i]*p[j]-du[j]*p[i])/determinant)


def _value(chart, uv):
    origin, du, dv = chart[:3]
    return tuple(o+uv[0]*a+uv[1]*b for o, a, b in zip(origin, du, dv))


def _strict_polygon(points):
    # Test all supporting halfspaces. Consistent consecutive turns alone
    # do not exclude a multiply wound polygon such as a pentagram.
    sides = [_cross2(_sub(points[(i+1) % len(points)], points[i]), _sub(point, points[i]))
             for i in range(len(points)) for j, point in enumerate(points)
             if j not in (i, (i+1) % len(points))]
    return all(value > 0 for value in sides) or all(value < 0 for value in sides)


def _strict_inside(point, polygon):
    sides = [_cross2(_sub(b, a), _sub(point, a)) for a, b in zip(polygon, polygon[1:]+polygon[:1])]
    return all(t > 0 for t in sides) or all(t < 0 for t in sides)


def exact_affine_plane_region_ssx(first, second, atol, rational, budget):
    """Return all components for supported original source nets, or None.

    Inputs can be degree-elevated affine charts with any uniform positive
    homogeneous weights. The return uses world XYZ and must bypass a
    normalizing caller's denormalizer. Output capacity is reserved for the
    complete region and its rims together, so no dangling references ship.
    """
    nets = tuple(np.asarray(net, dtype=float) for net in (first, second))
    if any(net.ndim != 3 or net.shape[-1] != (4 if rational else 3)
           or min(net.shape[:2]) < 2 or not np.all(np.isfinite(net)) for net in nets):
        return None
    if not budget.charge_cells(max(1, sum(net.size for net in nets)), 'affine_region_identity'):
        return None
    charts = tuple(_affine_chart(net, rational) for net in nets)
    if any(chart is None for chart in charts):
        return None
    origin, normal = charts[0][3][:2]
    if any(_dot(normal, _sub(point, origin)) for point in charts[1][-1]):
        return None
    if not budget.charge_cells(64, 'affine_region_clip'):
        return None
    polygon = _clip_polygon(list(charts[0][-1]), charts[1][3])
    result = dict(branches=[], points=[], singularities=[], overlap_regions=[], unresolved_regions=[])
    if not polygon:
        return result
    dimension = 0 if len(polygon) == 1 else 1 if len(polygon) == 2 else 2
    exact_stuv = [_inverse(charts[0], p)+_inverse(charts[1], p) for p in polygon]

    def partial(reason):
        budget.mark_incomplete(reason)
        budget.append_output(result['unresolved_regions'], {
            'stuv_min': (0.,)*4, 'stuv_max': (1.,)*4, 'reason': reason,
            'exact_dimension': dimension,
            'exact_stuv': tuple(tuple(str(x) for x in p) for p in exact_stuv),
            'exact_xyz': tuple(tuple(str(x) for x in p) for p in polygon),
        }, 'unresolved_region')
        return result

    needed = len(polygon)+1 if dimension == 2 else 1
    if budget.output_items+needed > budget.max_output_items:
        return partial(REASON_OUTPUT_CAP)
    if not budget.charge_cells(16*(len(polygon)+1), 'affine_region_representation'):
        return None
    try:
        limit = Fraction.from_float(float(atol))**2
        stuv = np.array([[float(x) for x in p] for p in exact_stuv])
        xyz = np.array([[float(x) for x in p] for p in polygon])
        rounded_stuv = [tuple(Fraction.from_float(float(x)) for x in p) for p in stuv]
        rounded_xyz = [tuple(Fraction.from_float(float(x)) for x in p) for p in xyz]
        if (len(set(rounded_stuv)) != len(polygon)
                or any(not 0 <= x <= 1 for p in rounded_stuv for x in p)):
            return partial(REASON_PARAMETER_REPRESENTATION)
        if dimension == 1 and any(
                rounded_stuv[0][2*i:2*i+2] == rounded_stuv[1][2*i:2*i+2]
                for i in range(2)):
            # Each affine chart is injective. Collapsing either chart's
            # endpoint parameters loses its segment preimage even when
            # a loose XYZ tolerance would accept that constant image.
            return partial(REASON_PARAMETER_REPRESENTATION)

        def valid_pair(uv, point):
            values = [_value(chart, uv[2*i:2*i+2]) for i, chart in enumerate(charts)]
            return all(sum((a-b)**2 for a, b in zip(left, right)) <= limit
                       for left, right in ((values[0], values[1]), (point, values[0]), (point, values[1])))

        if not all(valid_pair(uv, point) for uv, point in zip(rounded_stuv, rounded_xyz)):
            return partial(REASON_PARAMETER_REPRESENTATION)
        if dimension == 2:
            center = tuple(sum(p[k] for p in polygon)/len(polygon) for k in range(3))
            center_stuv = _inverse(charts[0], center)+_inverse(charts[1], center)
            interior = np.array([float(x) for x in center_stuv])
            rounded_interior = tuple(Fraction.from_float(float(x)) for x in interior)
            center_xyz = tuple(Fraction.from_float(float(x)) for x in map(float, center))
            if not valid_pair(rounded_interior, center_xyz):
                return partial(REASON_PARAMETER_REPRESENTATION)
            for owner in range(2):
                true_loop = [p[2*owner:2*owner+2] for p in exact_stuv]
                public_loop = [p[2*owner:2*owner+2] for p in rounded_stuv]
                witness = rounded_interior[2*owner:2*owner+2]
                if (not _strict_polygon(public_loop) or not _strict_inside(witness, true_loop)
                        or not _strict_inside(witness, public_loop)):
                    return partial(REASON_PARAMETER_REPRESENTATION)
    except (OverflowError, ValueError, ZeroDivisionError):
        return partial(REASON_PARAMETER_REPRESENTATION)

    if dimension == 0:
        budget.append_output(result['points'], SSXPoint(stuv[0], xyz[0]), 'point')
    elif dimension == 1:
        branch = SSXBranch((stuv,xyz),overlap=True,kind='overlap')
        branch._source_parameter_path = tuple(exact_stuv)
        budget.append_output(result['branches'],branch,'branch')
    else:
        for i in range(len(polygon)):
            indices = [i, (i+1) % len(polygon)]
            branch = SSXBranch((stuv[indices],xyz[indices]),overlap=True,kind='overlap')
            branch._source_parameter_path = tuple(exact_stuv[k] for k in indices)
            budget.append_output(result['branches'],branch,'branch')
        closed = np.vstack((stuv, stuv[0]))
        agreement = 1 if _dot(charts[0][3][1], charts[1][3][1]) > 0 else -1
        region = SSXOverlapRegion(
            boundary=[[(i, False) for i in range(len(polygon))]],
            uv1_loops=[closed[:, :2]], uv2_loops=[closed[:, 2:]],
            normal_agreement=agreement, interior_stuv=interior,
            certification={'image_identity': 'exact_affine_plane', 'injective_charts': True,
                           'boundary_resid_max': 1., 'interior_resid': 1.,
                           'residual_values_are_bounds': True,
                           'n_samples': len(polygon)+1, 'orientation_consistent': True,
                           'complete_preimage': True})
        budget.append_output(result['overlap_regions'], region, 'overlap_region')
    return result
