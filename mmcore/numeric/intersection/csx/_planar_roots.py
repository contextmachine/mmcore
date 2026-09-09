"""Exact univariate reduction for a curve against a convex planar chart.

The supplied binary coefficients define rational polynomials exactly.
Square-free factorization and Sturm counts isolate every plane-distance
root, including multiple roots. Exact halfspace signs classify the root
against the convex quadrilateral. Only the reported representative and
the unique bilinear inverse are floating-point approximations.
"""
from fractions import Fraction

import numpy as np

from ._planar_overlap import _exact_points, _convex_chart, _dot, _sub, _cross2


from mmcore.numeric.intersection._exact_univariate import (
    coefficient_build_work, isolate_root_intervals,
    _value, _derivative, _power, _range, _refine, _sign_at_root,
)


def _exact_curve_point(points, weights, t):
    work = [tuple(x*w for x in point)+(w,) for point, w in zip(points, weights)]
    while len(work) > 1:
        work = [tuple((1-t)*a+t*b for a, b in zip(left, right))
                for left, right in zip(work, work[1:])]
    return tuple(x/work[0][3] for x in work[0][:3])


def _exact_surface_point(points, u, v):
    factors = ((1-u)*(1-v), (1-u)*v, u*(1-v), u*v)
    return tuple(sum(factor*point[i] for factor, point in zip(factors, points))
                 for i in range(3))


def _outward_interval(interval):
    lo, hi = interval
    lower, upper = float(lo), float(hi)
    if Fraction.from_float(lower) > lo:
        lower = np.nextafter(lower, -np.inf)
    if Fraction.from_float(upper) < hi:
        upper = np.nextafter(upper, np.inf)
    return float(lower), float(upper)


def _exact_parameter_root_box(interval, points, weights, surface_points, axes):
    """Exact affine inverse enclosure; general convex charts retain full UV."""
    a, c_end, b_end, opposite = surface_points
    b, c = _sub(b_end, a), _sub(c_end, a)
    if any(opposite[i]-a[i]-b[i]-c[i] for i in range(3)):
        return (interval, (Fraction(0), Fraction(1)), (Fraction(0), Fraction(1)))
    i, j = axes
    determinant = b[i]*c[j]-b[j]*c[i]
    u_coeff = [w*((p[i]-a[i])*c[j]-(p[j]-a[j])*c[i])/determinant
               for p, w in zip(points, weights)]
    v_coeff = [w*(b[i]*(p[j]-a[j])-b[j]*(p[i]-a[i]))/determinant
               for p, w in zip(points, weights)]
    wlo, whi = _range(_power(weights), *interval)
    wlo, whi = max(wlo, min(weights)), min(whi, max(weights))
    box = [interval]
    for coefficients in (u_coeff, v_coeff):
        lower, upper = _range(_power(coefficients), *interval)
        values = (lower/wlo, lower/whi, upper/wlo, upper/whi)
        box.append((max(Fraction(0), min(values)),
                    min(Fraction(1), max(values))))
    return tuple(box)


def _parameter_root_box(interval, points, weights, surface_points, axes):
    return tuple(_outward_interval(bounds) for bounds in _exact_parameter_root_box(
        interval, points, weights, surface_points, axes))


def _bilinear_inverse(point, surface, axes):
    """Compute a representative in a chart already certified injective."""
    projected = surface[..., list(axes)]
    projected_point = point[list(axes)]
    if not np.all(np.isfinite(projected)) or not np.all(np.isfinite(projected_point)):
        return None
    scale = max(float(np.max(np.abs(projected))), float(np.max(np.abs(projected_point))))
    if scale == 0.:
        return None
    projected, projected_point = projected/scale, projected_point/scale
    a = projected[0, 0]
    b = projected[1, 0]-a
    c = projected[0, 1]-a
    d = projected[1, 1]-a-b-c
    target = projected_point-a
    # Eliminate v from cross(target-b*u, c+d*u)=0.
    cross = lambda x, y: x[0]*y[1]-x[1]*y[0]
    coefficients = [-cross(b, d), cross(target, d)-cross(b, c), cross(target, c)]
    try:
        candidates = np.roots(np.trim_zeros(coefficients, 'f'))
    except np.linalg.LinAlgError:
        return None
    best = None
    for candidate in candidates:
        if abs(candidate.imag) > 32*np.finfo(float).eps*max(1., abs(candidate.real)):
            continue
        u = float(candidate.real)
        direction = c+d*u
        v = float(np.dot(target-b*u, direction)/np.dot(direction, direction))
        if not np.isfinite(u+v):
            continue
        error = float(np.linalg.norm(b*u+c*v+d*u*v-target))
        outside = max(0., -u, u-1., -v, v-1.)
        rank = (outside, error)
        if best is None or rank < best[0]:
            best = (rank, u, v)
    if best is None:
        return None
    _, u, v = best
    u, v = float(np.clip(u, 0., 1.)), float(np.clip(v, 0., 1.))
    residual = b*u+c*v+d*u*v-target
    operand_scale = np.abs(projected).max(axis=(0, 1))+np.abs(projected_point)
    # This validates the numerical representative only. The exact
    # halfspace predicates above establish existence and uniqueness.
    if np.any(np.abs(residual) > 64*np.finfo(float).eps*operand_scale):
        return None
    return u, v


def exact_planar_bilinear_roots(C, S, rational=False, max_cells=100_000,
                               max_results=4096, atol=1e-3):
    """Return an exhaustive scalar-root result, or None for unsupported charts.

    Polynomial remainder steps and isolating interval refinements share
    ``max_cells``. A cap gives an explicit partial result. Identically
    planar curves are left to the separate exact overlap certificate.
    """
    curve, surface = np.asarray(C, dtype=float), np.asarray(S, dtype=float)
    dimension = 4 if rational else 3
    if (curve.ndim != 2 or curve.shape[1] != dimension or
            surface.shape != (2, 2, dimension) or
            not np.all(np.isfinite(curve)) or not np.all(np.isfinite(surface))):
        return None
    construction_work = coefficient_build_work(len(curve))
    if construction_work > max_cells:
        return {'isolated': [], 'overlaps': [], 'parameter_fibers': [],
                'budget_exhausted': True, 'boundary_topology_complete': False,
                'cells_processed': 0, 'truncation_cause': 'preflight',
                'required_coefficient_work': construction_work}
    curve_data, surface_data = _exact_points(curve, rational), _exact_points(surface, rational)
    if curve_data is None or surface_data is None:
        return None
    result = _exact_planar_bilinear_roots_data(
        *curve_data, *surface_data, max_cells=max_cells,
        max_results=max_results, atol=atol, construction_work=construction_work)
    if result is not None:
        for root in result['isolated']:
            root.pop('exact_parameter_root_box', None)
    return result


def _exact_planar_bilinear_roots_data(points, weights, surface_points, surface_weights,
                                     *, max_cells=100_000, max_results=4096,
                                     atol=1e-3, construction_work=0):
    """Private exact-coefficient engine; callers must prepay construction.

    All coordinates and weights are Fraction values, including restrictions
    of original sources which cannot be represented by binary floats.
    """
    if any(weight != surface_weights[0] for weight in surface_weights):
        return None
    chart = _convex_chart(surface_points)
    if chart is None:
        return None
    origin, normal, axes, quad, edges, orientation = chart
    polynomial = _power([weight*_dot(normal, _sub(point, origin))
                         for point, weight in zip(points, weights)])
    if not any(polynomial):
        return None
    result = {'isolated': [], 'overlaps': [], 'parameter_fibers': [],
              'budget_exhausted': False, 'cells_processed': construction_work,
              'boundary_topology_complete': True}

    class BudgetExceeded(Exception):
        pass

    def tick():
        if result['cells_processed'] >= max_cells:
            raise BudgetExceeded
        result['cells_processed'] += 1

    try:
        squarefree, repeated_factor, sequence, intervals = isolate_root_intervals(polynomial, tick)
        if not intervals:
            return result

        halfspaces = [_power([
            weight*orientation*_cross2(edge, _sub(tuple(point[i] for i in axes), quad[index]))
            for point, weight in zip(points, weights)]) for index, edge in enumerate(edges)]
        try:
            cartesian_surface = np.array(surface_points, dtype=float).reshape(2, 2, 3)
        except OverflowError:
            cartesian_surface = np.full((2, 2, 3), np.inf)
        for interval in sorted(intervals):
            inside = True
            for halfspace in halfspaces:
                sign, interval = _sign_at_root(squarefree, sequence, interval, halfspace, tick)
                if sign < 0:
                    inside = False
                    break
            if not inside:
                continue
            while interval[0] != interval[1] and np.nextafter(float(interval[0]), np.inf) < float(interval[1]):
                interval = _refine(squarefree, sequence, interval, tick)
            t = float(sum(interval)/2)
            exact_point = _exact_curve_point(points, weights, Fraction.from_float(t))
            try:
                point = np.array([float(x) for x in exact_point])
            except OverflowError:
                point = np.full(3, np.inf)
            a, c_end, b_end, opposite = surface_points
            b, c = _sub(b_end, a), _sub(c_end, a)
            if not any(opposite[i]-a[i]-b[i]-c[i] for i in range(3)):
                # Exact affine inversion also avoids losing a very narrow
                # source rectangle when its float vertices round together.
                i, j = axes
                determinant = b[i]*c[j]-b[j]*c[i]
                target = _sub(exact_point, a)
                uv = tuple(float(max(Fraction(0), min(Fraction(1), value))) for value in (
                    (target[i]*c[j]-target[j]*c[i])/determinant,
                    (b[i]*target[j]-b[j]*target[i])/determinant))
            else:
                uv = _bilinear_inverse(point, cartesian_surface, axes)
            exact_surface_point = (_exact_surface_point(
                surface_points, *(Fraction.from_float(x) for x in uv))
                if uv is not None else None)
            distance_squared = (sum((a-b)**2 for a, b in zip(exact_point, exact_surface_point))
                                if exact_surface_point is not None else None)
            if distance_squared is not None and np.all(np.isfinite(point)):
                rounded_point = tuple(Fraction.from_float(float(x)) for x in point)
                distance_squared = max(distance_squared, *(
                    sum((a-b)**2 for a, b in zip(rounded_point, exact_value))
                    for exact_value in (exact_point, exact_surface_point)))
            if (distance_squared is None or not np.all(np.isfinite(point))
                    or distance_squared > Fraction.from_float(float(atol))**2
                    or any(entry['t'] == t for entry in result['isolated'])):
                result.update(budget_exhausted=True, boundary_topology_complete=False,
                              truncation_cause='resolution')
                result.setdefault('unresolved_parameter_boxes', []).append({
                    't_range': tuple(float(x) for x in interval),
                    'exact_t_interval': tuple((str(x.numerator), str(x.denominator)) for x in interval),
                    'reason': 'parameter_representation',
                })
                continue
            if len(result['isolated']) >= max_results:
                result.update(budget_exhausted=True, boundary_topology_complete=False,
                              truncation_cause='max_results')
                break
            entry = {
                't': t, 'u': uv[0], 'v': uv[1], 'point': point,
                'parameter_root_certification': 'exact_sturm_isolation',
                'exact_t_interval': tuple((str(x.numerator), str(x.denominator)) for x in interval),
                'parameter_root_box': _parameter_root_box(
                    interval, points, weights, surface_points, axes),
                'exact_parameter_root_box': _exact_parameter_root_box(
                    interval, points, weights, surface_points, axes),
            }
            if len(repeated_factor) == 1:
                entry['root_multiplicity'] = 1
            elif interval[0] == interval[1]:
                derivative, multiplicity = polynomial, 0
                while len(derivative) > 1 and _value(derivative, interval[0]) == 0:
                    derivative = _derivative(derivative)
                    multiplicity += 1
                entry['root_multiplicity'] = multiplicity
            result['isolated'].append(entry)
    except BudgetExceeded:
        result.update(budget_exhausted=True, boundary_topology_complete=False,
                      truncation_cause='max_cells')
    return result
