"""Exact isolated intersections of a positive-weight Bezier curve and line."""
from fractions import Fraction

import numpy as np

from mmcore.numeric.intersection._exact_univariate import (
    coefficient_build_work, isolate_root_intervals,
    _power, _gcd, _derivative, _value,
    _refine, _sign_at_root, _range,
)
from mmcore.numeric.intersection._root_box_certificate import exact_bernstein_value


def _cartesian(net, rational):
    rows = [tuple(Fraction.from_float(float(x)) for x in row) for row in net]
    weights = [row[-1] if rational else Fraction(1) for row in rows]
    if min(weights) <= 0:
        return None
    return [tuple(x/w for x in row[:3]) for row, w in zip(rows, weights)], weights


def _point(net, parameter, rational):
    value = exact_bernstein_value(net, (parameter,))
    return tuple(x/value[-1] for x in value[:3]) if rational else value


def _outward(interval):
    lo, hi = interval
    a, b = float(lo), float(hi)
    if Fraction.from_float(a) > lo:
        a = np.nextafter(a, -np.inf)
    if Fraction.from_float(b) < hi:
        b = np.nextafter(b, np.inf)
    return float(a), float(b)


def exact_curve_line_ccx(C1, C2, rational=False, max_cells=100000,
                         max_results=4096, atol=1e-3):
    """GCD/Sturm proves all isolated roots; an identity leaves overlap to its tier."""
    dimension = 4 if rational else 3
    first, second = np.asarray(C1, dtype=float), np.asarray(C2, dtype=float)
    if (first.ndim != 2 or second.ndim != 2 or first.shape[-1] != dimension
            or second.shape[-1] != dimension or
            not np.all(np.isfinite(first)) or not np.all(np.isfinite(second))):
        return None
    swapped = len(first) == 2 and len(second) > 2
    curve, line = (second, first) if swapped else (first, second)
    if len(line) != 2 or len(curve) < 2:
        return None
    construction_work = coefficient_build_work(len(curve))
    if construction_work > max_cells:
        return {'isolated': [], 'overlaps': [], 'parameter_fibers': [],
                'budget_exhausted': True, 'boundary_topology_complete': False,
                'cells_processed': 0, 'truncation_cause': 'preflight',
                'required_coefficient_work': construction_work}
    curve_data, line_data = _cartesian(curve, rational), _cartesian(line, rational)
    if curve_data is None or line_data is None:
        return None
    points, weights = curve_data
    line_points, line_weights = line_data
    origin = line_points[0]
    direction = tuple(b-a for a, b in zip(*line_points))
    norm2 = sum(x*x for x in direction)
    if not norm2:
        return None
    differences = [tuple(w*(p-o) for p, o in zip(point, origin))
                   for point, w in zip(points, weights)]
    cross = [(a[1]*direction[2]-a[2]*direction[1],
              a[2]*direction[0]-a[0]*direction[2],
              a[0]*direction[1]-a[1]*direction[0]) for a in differences]
    polynomials = [_power([value[i] for value in cross]) for i in range(3)]
    nonzero = [p for p in polynomials if any(p)]
    if not nonzero:
        return None  # A positive-dimensional collinear relation, not isolated roots.
    result = {'isolated': [], 'overlaps': [], 'budget_exhausted': False,
              'boundary_topology_complete': True, 'cells_processed': construction_work}

    class BudgetExceeded(Exception):
        pass

    def tick():
        if result['cells_processed'] >= max_cells:
            raise BudgetExceeded
        result['cells_processed'] += 1

    try:
        tick()
        polynomial = nonzero[0]
        for component in nonzero[1:]:
            polynomial = _gcd(polynomial, component, tick)
            if len(polynomial) == 1:
                return result
        squarefree, repeated, sequence, intervals = isolate_root_intervals(polynomial, tick)
        if not intervals:
            return result
        zero, one = Fraction(0), Fraction(1)
        projection = _power([sum(x*d for x, d in zip(value, direction)) for value in differences])
        denominator = _power([norm2*w for w in weights])
        upper_halfspace = [a-b for a, b in zip(
            denominator+[Fraction(0)]*max(0, len(projection)-len(denominator)),
            projection+[Fraction(0)]*max(0, len(denominator)-len(projection)))]

        def line_parameter(value):
            w0, w1 = line_weights
            return value*w0/(w1*(1-value)+value*w0)

        for interval in sorted(intervals):
            inside = True
            for halfspace in (projection, upper_halfspace):
                sign, interval = _sign_at_root(squarefree, sequence, interval, halfspace, tick)
                if sign < 0:
                    inside = False
                    break
            if not inside:
                continue
            while interval[0] != interval[1] and np.nextafter(float(interval[0]), np.inf) < float(interval[1]):
                interval = _refine(squarefree, sequence, interval, tick)
            p_lo, p_hi = _range(projection, *interval)
            d_lo, d_hi = _range(denominator, *interval)
            d_lo, d_hi = max(d_lo, norm2*min(weights)), min(d_hi, norm2*max(weights))
            quotients = (p_lo/d_lo, p_lo/d_hi, p_hi/d_lo, p_hi/d_hi)
            line_interval = (line_parameter(max(zero, min(quotients))),
                             line_parameter(min(one, max(quotients))))
            box = (_outward(interval), _outward(line_interval))
            reported_box = box[::-1] if swapped else box
            t = float(sum(interval)/2)
            exact_curve_point = _point(curve, Fraction.from_float(t), rational)
            projected = sum((p-o)*d for p, o, d in zip(exact_curve_point, origin, direction))/norm2
            v = float(line_parameter(min(one, max(zero, projected))))
            exact_line_point = _point(line, Fraction.from_float(v), rational)
            try:
                point = np.array([float(x) for x in exact_curve_point])
                rounded = tuple(Fraction.from_float(float(x)) for x in point)
                distance = max(sum((a-b)**2 for a, b in zip(left, right))
                               for left, right in ((exact_curve_point, exact_line_point),
                                                   (rounded, exact_curve_point), (rounded, exact_line_point)))
            except (OverflowError, ValueError):
                distance = None
            key = 'v' if swapped else 'u'
            if (distance is None or distance > Fraction.from_float(float(atol))**2
                    or any(entry[key] == t for entry in result['isolated'])):
                result.update(budget_exhausted=True, boundary_topology_complete=False,
                              truncation_cause='resolution')
                result.setdefault('unresolved_parameter_boxes', []).append({
                    'u_range': reported_box[0], 'v_range': reported_box[1],
                    'parameter_root_box': reported_box,
                    'candidate': (v, t) if swapped else (t, v),
                    'exact_curve_parameter_interval': tuple((str(x.numerator), str(x.denominator)) for x in interval),
                    'reason': 'parameter_representation'})
                continue
            if len(result['isolated']) >= max_results:
                result.update(budget_exhausted=True, boundary_topology_complete=False,
                              truncation_cause='max_results')
                break
            entry = {'u': v if swapped else t, 'v': t if swapped else v,
                     'point': point, 'certification': 'exact', 'd_min': 0.,
                     'parameter_root_certification': 'exact_curve_line_sturm',
                     'parameter_root_box': reported_box,
                     'exact_curve_parameter_interval': tuple((str(x.numerator), str(x.denominator)) for x in interval)}
            if len(repeated) == 1:
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
