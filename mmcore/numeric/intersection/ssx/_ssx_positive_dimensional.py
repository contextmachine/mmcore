"""Bounded ownership of one unsupported convex planar 2D relation.

Strictly convex positive-weight rational bilinear charts are homeomorphisms
onto their corner quadrilaterals. We verify the homogeneous planar Jacobian
numerator exactly: det(H, H_s, H_t) has degree at most (1,1), and its four
Bernstein coefficients have one strict sign. The boundary is a simple
quadrilateral traversed monotonically because its edge weights are positive.
The nonzero interior Jacobian and this degree-one boundary establish global
injectivity and surjectivity onto that quadrilateral.

Positive-area intersection of two such exact coplanar quadrilaterals is one
convex region. The entire paired zero set is its unique lifted image; there
are no additional ordinary components or isolated fibers to search. This
module records that whole owner as unresolved because it does not yet build
a certified paired UV region representation. It never declares it complete.
Higher-degree/folded charts, zero-area contacts, and plane near-coincidences
remain with the general solver.
"""
from fractions import Fraction
from itertools import permutations

import numpy as np

from mmcore.numeric._work_budget import REASON_OVERLAP_REGION
from mmcore.numeric.intersection.csx._planar_overlap import (
    _exact_points, _convex_chart, _dot, _sub,
)


class _Stopped(Exception):
    pass


def _add(a, b, sign, spend):
    spend(len(a)+len(b))
    result = dict(a)
    for index, value in b.items():
        result[index] = result.get(index, 0)+sign*value
        if result[index] == 0:
            del result[index]
    return result


def _multiply(a, b, spend):
    spend(2*len(a)*len(b))
    result = {}
    for (i, j), x in a.items():
        for (k, ell), y in b.items():
            index = (i+k, j+ell)
            result[index] = result.get(index, 0)+x*y
    return {index: value for index, value in result.items() if value}


def _strict_rational_jacobian(points, weights, axes, spend):
    """Exact projected determinant, with no floating normal threshold."""
    spend(8+3*4*4)
    homogeneous = [tuple(p[k]*w for k in axes)+(w,)
                   for p, w in zip(points, weights)]
    rows = []
    for axis in range(3):
        a, c, b, d = (point[axis] for point in homogeneous)
        h = {(0, 0): a, (1, 0): b-a,
             (0, 1): c-a, (1, 1): d-b-c+a}
        h = {index: value for index, value in h.items() if value}
        hs = {(0, j): value for (i, j), value in h.items() if i}
        ht = {(i, 0): value for (i, j), value in h.items() if j}
        rows.append((h, hs, ht))
    determinant = {}
    for permutation in permutations(range(3)):
        inversions = sum(permutation[i] > permutation[j]
                         for i in range(3) for j in range(i+1, 3))
        term = {(0, 0): Fraction(1)}
        for row, column in enumerate(permutation):
            term = _multiply(term, rows[row][column], spend)
        determinant = _add(determinant, term, (-1)**inversions, spend)
    # This identity is also checked rather than merely assumed from degree.
    if any(i > 1 or j > 1 for i, j in determinant):
        return False
    spend(4*max(1, len(determinant)))
    corners = [sum(value for (i, j), value in determinant.items()
                   if (s or not i) and (t or not j))
               for s, t in ((0, 0), (0, 1), (1, 0), (1, 1))]
    return all(value > 0 for value in corners) or all(value < 0 for value in corners)


def _clip_polygon(points, chart, spend):
    _, _, axes, quad, edges, orientation = chart

    def side(point, edge, origin):
        spend(6)
        x, y = (point[k]-origin[i] for i, k in enumerate(axes))
        return orientation*(edge[0]*y-edge[1]*x)

    for edge, origin in zip(edges, quad):
        if not points:
            return []
        output = []
        previous = points[-1]
        previous_value = side(previous, edge, origin)
        for current in points:
            current_value = side(current, edge, origin)
            if (previous_value >= 0) != (current_value >= 0):
                spend(11)
                t = previous_value/(previous_value-current_value)
                output.append(tuple(a+t*(b-a) for a, b in zip(previous, current)))
            if current_value >= 0:
                output.append(current)
            previous, previous_value = current, current_value
        spend(3*len(output))
        points = []
        for point in output:
            if not points or point != points[-1]:
                points.append(point)
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
    return points


def exact_convex_bilinear_region_owner(first, second, rational, budget):
    """Return an explicit 2D unresolved owner, or None without a proof.

    The inputs are the original world source nets. No subdivision or
    normalization can alter the exact coefficients used by this predicate.
    Output contains exact image-polygon rationals, not invented UV samples.
    """
    nets = tuple(np.asarray(net, dtype=float) for net in (first, second))
    dimension = 4 if rational else 3
    if any(net.shape != (2, 2, dimension) or not np.all(np.isfinite(net))
           for net in nets):
        return None

    def spend(amount):
        if not budget.charge_cells(max(1, int(amount)), 'positive_dimensional_owner'):
            raise _Stopped

    try:
        # Original Fraction conversion and Cartesian division, once.
        spend(sum(net.size for net in nets)+24)
        data = tuple(_exact_points(net, rational) for net in nets)
        if any(item is None for item in data):
            return None
        charts = []
        for points, weights in data:
            # The four-corner chart routine uses fewer than128 scalar
            # differences/products/comparisons; reserve before it runs.
            spend(128)
            chart = _convex_chart(points)
            if chart is None:
                return None
            charts.append(chart)
        origin, normal = charts[0][:2]
        spend(4*9)
        if any(_dot(normal, _sub(point, origin)) for point in data[1][0]):
            return None
        for (points, weights), chart in zip(data, charts):
            if not _strict_rational_jacobian(points, weights, chart[2], spend):
                return None
        polygon = _clip_polygon([data[0][0][i] for i in (0, 2, 3, 1)], charts[1], spend)
        if len(polygon) < 3:
            return None
        i, j = charts[0][2]
        spend(4*len(polygon))
        area = sum(a[i]*b[j]-a[j]*b[i]
                   for a, b in zip(polygon, polygon[1:]+polygon[:1]))
        if area == 0:
            return None
        spend(3*len(polygon))
        diagnostic = {
            'stuv_min': (0.,)*4, 'stuv_max': (1.,)*4,
            'reason': REASON_OVERLAP_REGION,
            'exact_dimension': 2,
            'proof': 'exact_convex_rational_bilinear_region',
            'exact_image_polygon': tuple(tuple(str(x) for x in point) for point in polygon),
            'unresolved_obligation': 'paired_region_representation',
        }
    except _Stopped:
        return None
    result = dict(branches=[], points=[], singularities=[], overlap_regions=[], unresolved_regions=[])
    budget.mark_incomplete(REASON_OVERLAP_REGION)
    budget.append_output(result['unresolved_regions'], diagnostic, 'unresolved_region')
    return result
