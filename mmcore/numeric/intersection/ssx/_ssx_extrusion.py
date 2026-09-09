"""Exact univariate SSX for polynomial rulings parallel to an affine plane.

If a surface is affine in one parameter and its plane-distance Bernstein
coefficients are independent of that parameter, every component is a ruling
at a zero of one scalar polynomial. Sturm isolation supplies an exhaustive
root census; exact halfspace clipping supplies every ruling's finite extent.
Currently isolation must expose a singleton (dyadic roots, or an arbitrary
rational root of a linear square-free polynomial). Other algebraic roots,
including some non-dyadic rational roots, return None for the general solver.
"""
from fractions import Fraction

import numpy as np

from mmcore.numeric._work_budget import REASON_PARAMETER_REPRESENTATION
from mmcore.numeric.intersection._exact_univariate import (
    _power, _derivative, _value, _refine, isolate_root_intervals,
    coefficient_build_work,
)
from mmcore.numeric.intersection._root_box_certificate import exact_bernstein_value
from mmcore.numeric.intersection.csx._planar_overlap import (
    _exact_points, _convex_chart, _sub, _dot, _cross2,
)
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch, SSXPoint
from mmcore.numeric.intersection.ssx._ssx_singular_candidate import _cross


class _Stopped(Exception):
    pass


def _evaluate_fraction_curve(curve, t):
    work = list(curve)
    while len(work) > 1:
        work = [tuple((1-t)*a+t*b for a, b in zip(left, right))
                for left, right in zip(work, work[1:])]
    return work[0]


def _uniform_cartesian(net, rational):
    data = _exact_points(net, rational)
    if data is None:
        return None
    points, weights = data
    if any(w != weights[0] for w in weights):
        return None
    width = net.shape[1]
    return [points[i:i+width] for i in range(0, len(points), width)]


def _affine_axis(points, axis):
    if axis == 1:
        points = list(zip(*points))
    degree = len(points)-1
    if degree < 1:
        return None
    for i, row in enumerate(points):
        t = Fraction(i, degree)
        if any(any(p != (1-t)*a+t*b for p, a, b in zip(value, first, last))
               for value, first, last in zip(row, points[0], points[-1])):
            return None
    return points[0], points[-1]


def _ruling_normal_zeros(first_curve, last_curve, root, lo, hi):
    """Exact normal zero set along an affine, noncollapsed ruling.

Uniform positive homogeneous weights were reduced exactly to Cartesian
coefficients before this function. The free derivative is constant and
the other derivative is affine in the free parameter, so their cross
product is affine too. Return whole-curve degeneracy or an isolated zero.
"""
    start = _evaluate_fraction_curve(first_curve, root)
    end = _evaluate_fraction_curve(last_curve, root)
    free = _sub(end, start)
    degree = len(first_curve)-1
    derivatives = []
    for curve in (first_curve, last_curve):
        net = [tuple(degree*(b-a) for a, b in zip(left, right))
               for left, right in zip(curve[:-1], curve[1:])]
        derivatives.append(_evaluate_fraction_curve(net, root))
    first, last = [_cross(free, derivative) for derivative in derivatives]
    if not any(first) and not any(last):
        return True, None
    delta = _sub(last, first)
    if not any(delta):
        return False, None
    axis = next(k for k, value in enumerate(delta) if value)
    candidate = -first[axis]/delta[axis]
    if lo <= candidate <= hi and not any(a+candidate*b for a, b in zip(first, delta)):
        return False, candidate
    return False, None


def exact_extrusion_plane_ssx(first, second, atol, rational, budget):
    """Return an exhaustive supported result, or None after charged predicates."""
    nets = (np.asarray(first, dtype=float), np.asarray(second, dtype=float))
    if (any(net.ndim != 3 or net.shape[-1] != (4 if rational else 3)
            or min(net.shape[:2]) < 2 or not np.all(np.isfinite(net)) for net in nets)
            or not any(net.shape[:2] == (2, 2) for net in nets)):
        return None
    # This is a preflight for exact coefficient construction, not a degree
    # limit chosen to fit examples. Unsupported candidates still account for
    # the predicates they execute before general subdivision continues.
    work = max(1, sum(net.size for net in nets))
    if not budget.charge_cells(work, 'extrusion_certificate'):
        return None
    exact = [_uniform_cartesian(net, rational) for net in nets]
    if any(points is None for points in exact):
        return None

    def tick():
        if not budget.charge_cells(1, 'extrusion_roots'):
            raise _Stopped

    for owner in (0, 1):
        target = exact[1-owner]
        if nets[1-owner].shape[:2] != (2, 2):
            continue
        target_flat = [p for row in target for p in row]
        chart = _convex_chart(target_flat)
        if chart is None:
            continue
        origin, normal, axes, quad, edges, orientation = chart
        p00, p01, p10, p11 = target_flat
        du, dv = _sub(p10, p00), _sub(p01, p00)
        if any(z-a-b+o for z, a, b, o in zip(p11, p10, p01, p00)):
            continue  # Non-affine inverse parameter curves need another tier.
        i, j = axes
        determinant = du[i]*dv[j]-du[j]*dv[i]

        def inverse(point):
            p = _sub(point, p00)
            return ((p[i]*dv[j]-p[j]*dv[i])/determinant,
                    (du[i]*p[j]-du[j]*p[i])/determinant)

        for axis in (0, 1):
            rulings = _affine_axis(exact[owner], axis)
            if rulings is None:
                continue
            first_curve, last_curve = rulings
            heights = [_dot(normal, _sub(p, origin)) for p in first_curve]
            if heights != [_dot(normal, _sub(p, origin)) for p in last_curve]:
                continue
            if not any(heights):
                continue  # A two-dimensional coincidence, not isolated rulings.
            if not budget.charge_cells(coefficient_build_work(len(heights)), 'extrusion_roots'):
                return None
            polynomial = _power(heights)
            try:
                squarefree, repeated, sequence, intervals = isolate_root_intervals(polynomial, tick)
                roots = []
                for interval in sorted(intervals):
                    while interval[0] != interval[1]:
                        if np.nextafter(float(interval[0]), np.inf) >= float(interval[1]):
                            return None  # Keep the algebraic case in the general solver.
                        interval = _refine(squarefree, sequence, interval, tick)
                    roots.append(interval[0])
            except _Stopped:
                return None
            result = dict(branches=[], points=[], singularities=[], overlap_regions=[],
                          unresolved_regions=[])
            evaluation_work = max(1, (6*sum((1+sum(n-1 for n in net.shape[:2]))*net.size
                                           for net in nets)+127)//128)
            if not budget.charge_cells(evaluation_work*len(roots), 'extrusion_geometry'):
                return None
            rounded_roots = {}
            for root in roots:
                # A generator collapsing to a point has a parameter fiber;
                # never turn that different topology into a simple line.
                start = _evaluate_fraction_curve(first_curve, root)
                end = _evaluate_fraction_curve(last_curve, root)
                direction = _sub(end, start)
                if not any(direction):
                    return None
                lo, hi = Fraction(0), Fraction(1)
                for edge, q in zip(edges, quad):
                    a = orientation*_cross2(edge, _sub(tuple(start[k] for k in axes), q))
                    d = orientation*_cross2(edge, tuple(direction[k] for k in axes))
                    if d > 0:
                        lo = max(lo, -a/d)
                    elif d < 0:
                        hi = min(hi, -a/d)
                    elif a < 0:
                        lo, hi = Fraction(1), Fraction(0)
                        break
                if hi < lo:
                    continue
                degenerate_ruling, isolated_cusp = _ruling_normal_zeros(
                    first_curve, last_curve, root, lo, hi)
                if isolated_cusp is not None:
                    # The whole SSI is still a line, but a point on it has
                    # a different source-chart stratum. Leave that mixed
                    # classification to the general solver for now.
                    return None
                locations, points = [], []
                for value in (lo, hi):
                    point = tuple(a+value*d for a, d in zip(start, direction))
                    uv = inverse(point)
                    own = (value, root) if axis == 0 else (root, value)
                    locations.append(own+uv if owner == 0 else uv+own)
                    points.append(point)
                try:
                    stuv = np.asarray([[float(x) for x in p] for p in locations])
                    xyz = np.asarray([[float(x) for x in p] for p in points])
                    reported = [tuple(Fraction.from_float(float(x)) for x in p) for p in xyz]
                    limit = Fraction.from_float(float(atol))**2
                    valid = all(np.all(np.isfinite(p)) for p in (stuv, xyz))
                    for uv4, point in zip(stuv, reported):
                        values = [exact_bernstein_value(net, uv4[2*k:2*k+2]) for k, net in enumerate(nets)]
                        if rational:
                            values = [tuple(x/value[-1] for x in value[:3]) for value in values]
                        valid &= all(sum((a-b)**2 for a, b in zip(left, right)) <= limit
                                     for left, right in ((values[0], values[1]),
                                                         (point, values[0]), (point, values[1])))
                    # Source and target maps are affine along each emitted
                    # parameter segment. Their residuals to the reported XYZ
                    # chord are bounded by the endpoint errors just checked.
                    valid &= (lo == hi or not np.array_equal(stuv[0], stuv[1]))
                    valid &= float(root) not in rounded_roots
                except (ValueError, OverflowError, ZeroDivisionError):
                    valid = False
                if not valid:
                    result['unresolved_regions'].append(dict(
                        stuv_min=(0.,)*4, stuv_max=(1.,)*4,
                        reason=REASON_PARAMETER_REPRESENTATION,
                        exact_ruling_parameter=str(root),
                        exact_stuv=tuple(tuple(str(x) for x in p) for p in locations)))
                    continue
                rounded_roots[float(root)] = root
                multiplicity, derivative = 0, polynomial
                while len(derivative) > 1 and _value(derivative, root) == 0:
                    multiplicity += 1
                    derivative = _derivative(derivative)
                if lo == hi:
                    result['points'].append(SSXPoint(stuv[0], xyz[0]))
                else:
                    target_locations = [p[2*(1-owner):2*(1-owner)+2] for p in locations]
                    boundary = (root in (0, 1) or any(
                        target_locations[0][k] == target_locations[1][k]
                        and target_locations[0][k] in (0, 1) for k in range(2)))
                    branch = SSXBranch((stuv, xyz), closed=False, overlap=boundary,
                                       kind=('overlap' if boundary else
                                             'tangential' if multiplicity > 1 and not degenerate_ruling
                                             else 'transversal'))
                    result['branches'].append(branch)
                    if degenerate_ruling:
                        from mmcore.numeric.intersection.ssx._bez_ssx5 import SSXSingularity
                        result['singularities'].append(SSXSingularity(
                            kind='cusp_curve', stuv=stuv[0].copy(), xyz=xyz[0].copy(),
                            branch_links=[(len(result['branches'])-1, 0)],
                            samples=stuv.copy(), surface=owner+1))
            if result['unresolved_regions']:
                budget.mark_incomplete(REASON_PARAMETER_REPRESENTATION)
            for key in ('branches', 'points', 'singularities', 'unresolved_regions'):
                entries, result[key] = result[key], []
                if key == 'singularities':
                    for singularity in entries:
                        singularity.branch_links = [(index, vertex) for index, vertex in singularity.branch_links
                                                    if index < len(result['branches'])]
                budget.extend_output(result[key], entries, key)
            return result
    return None
