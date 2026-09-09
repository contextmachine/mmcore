"""Exact intersection of two nonconstant degree-one positive-weight curves."""
from fractions import Fraction
from itertools import combinations

import numpy as np


def exact_linear_ccx(C1, C2, rational=False, max_results=4096, atol=1e-3):
    dimension = 4 if rational else 3
    curves = [np.asarray(curve, dtype=float) for curve in (C1, C2)]
    if any(curve.shape != (2, dimension) or not np.all(np.isfinite(curve))
           for curve in curves):
        return None
    points, weights = [], []
    for curve in curves:
        rows = [tuple(Fraction.from_float(float(x)) for x in row) for row in curve]
        weight = [row[3] if rational else Fraction(1) for row in rows]
        if min(weight) <= 0:
            return None
        weights.append(weight)
        points.append([tuple(x/w for x in row[:3]) for row, w in zip(rows, weight)])
    a, b = points
    da = tuple(y-x for x, y in zip(*a))
    db = tuple(y-x for x, y in zip(*b))
    if not any(da) or not any(db):
        return None
    offset = tuple(y-x for x, y in zip(a[0], b[0]))
    result = {'isolated': [], 'overlaps': [], 'budget_exhausted': False,
              'cells_processed': 1, 'boundary_topology_complete': True}

    def parameter(value, weight):
        w0, w1 = weight
        return value*w0/(w1*(1-value)+value*w0)

    def enclosure(value):
        rounded = float(value)
        exact_rounded = Fraction.from_float(rounded)
        return (float(np.nextafter(rounded, -np.inf)) if exact_rounded > value else rounded,
                float(np.nextafter(rounded, np.inf)) if exact_rounded < value else rounded)

    def represented_pair(x, y):
        parameters = [parameter(value, weight) for value, weight in zip((x, y), weights)]
        rounded = [Fraction.from_float(float(t)) for t in parameters]
        values = [weight[1]*t/(weight[0]*(1-t)+weight[1]*t)
                  for t, weight in zip(rounded, weights)]
        point_a = tuple(p+values[0]*d for p, d in zip(a[0], da))
        point_b = tuple(p+values[1]*d for p, d in zip(b[0], db))
        try:
            rounded_point = tuple(Fraction.from_float(float(p+x*d)) for p, d in zip(a[0], da))
            distance = max(sum((p-q)**2 for p, q in zip(left, right))
                           for left, right in ((point_a, point_b), (rounded_point, point_a),
                                               (rounded_point, point_b)))
        except (OverflowError, ValueError):
            distance = None
        if distance is not None and distance <= Fraction.from_float(float(atol))**2:
            return parameters
        result.update(budget_exhausted=True, boundary_topology_complete=False,
                      truncation_cause='resolution')
        result.setdefault('unresolved_parameter_boxes', []).append({
            'u_range': enclosure(parameters[0]), 'v_range': enclosure(parameters[1]),
            'parameter_root_box': tuple(enclosure(t) for t in parameters),
            'candidate': tuple(float(t) for t in parameters),
            'exact_parameters': tuple((str(t.numerator), str(t.denominator)) for t in parameters),
            'reason': 'parameter_representation',
        })
        return None

    def isolated(x, y):
        if max_results <= 0:
            result.update(budget_exhausted=True, boundary_topology_complete=False,
                          truncation_cause='max_results')
            return
        parameters = represented_pair(x, y)
        if parameters is None:
            return
        result['isolated'].append({
            'u': float(parameters[0]), 'v': float(parameters[1]),
            'point': np.array([float(p+x*d) for p, d in zip(a[0], da)]),
            'certification': 'exact', 'd_min': 0.,
            'parameter_root_certification': 'exact_linear_elimination',
            'parameter_root_box': tuple(enclosure(t) for t in parameters),
            'exact_parameters': tuple((str(t.numerator), str(t.denominator)) for t in parameters),
        })

    for i, j in combinations(range(3), 2):
        determinant = da[j]*db[i]-da[i]*db[j]
        if determinant:
            x = (offset[j]*db[i]-offset[i]*db[j])/determinant
            y = (da[i]*offset[j]-da[j]*offset[i])/determinant
            if (0 <= x <= 1 and 0 <= y <= 1 and
                    all(x*d-y*e == h for d, e, h in zip(da, db, offset))):
                isolated(x, y)
            return result
    # Parallel directions: exact collinearity, then one-dimensional clipping.
    axis = next(i for i, value in enumerate(da) if value)
    if any(offset[i]*da[axis] != offset[axis]*da[i] for i in range(3)):
        return result
    start, end = offset[axis]/da[axis], (offset[axis]+db[axis])/da[axis]
    lo, hi = max(Fraction(0), min(start, end)), min(Fraction(1), max(start, end))
    if hi < lo:
        return result
    ylo, yhi = (lo-start)/(end-start), (hi-start)/(end-start)
    if lo == hi:
        isolated(lo, ylo)
    elif max_results <= 0:
        result.update(budget_exhausted=True, boundary_topology_complete=False,
                      truncation_cause='max_results')
    else:
        if represented_pair(lo, ylo) is None or represented_pair(hi, yhi) is None:
            u_endpoints = [parameter(x, weights[0]) for x in (lo, hi)]
            v_endpoints = [parameter(y, weights[1]) for y in (ylo, yhi)]
            result.setdefault('unresolved_parameter_boxes', []).append({
                'u_range': (enclosure(min(u_endpoints))[0], enclosure(max(u_endpoints))[1]),
                'v_range': (enclosure(min(v_endpoints))[0], enclosure(max(v_endpoints))[1]),
                'reason': 'overlap_parameter_representation',
            })
            return result
        result['overlaps'].append({
            'u_range': tuple(float(parameter(x, weights[0])) for x in (lo, hi)),
            'v_range': tuple(float(parameter(y, weights[1])) for y in (ylo, yhi)),
            'boundary_zeros': [], 'overlap_endpoints': [], 'certification': 'exact',
            'proof': 'exact_collinear_line_clipping',
        })
    return result
