"""Straight source-isocurve paths with an exact planar inclusion proof.

This tier avoids numerical continuation along a high-multiplicity zero:
a small residual there gives no useful bound on distance to the locus.
Only the prescribed isocurve is discharged; other surface components remain
the caller's responsibility.
"""
from fractions import Fraction
from math import comb

import numpy as np

from mmcore.numeric.intersection.csx._planar_overlap import (
    _exact_points, _convex_chart, _sub, _cross, _dot, _cross2,
)
from mmcore.numeric.intersection._root_box_certificate import exact_bernstein_value


def _split(net, t):
    rows, left, right = list(net), [], []
    while rows:
        left.append(rows[0])
        right.append(rows[-1])
        rows = [tuple((1-t)*a+t*b for a, b in zip(x, y))
                for x, y in zip(rows, rows[1:])]
    return left, right[::-1]


def _isocurve(net, start, end, rational):
    changing = [axis for axis in range(2) if start[axis] != end[axis]]
    if len(changing) != 1:
        return None
    axis = changing[0]
    fixed = 1-axis
    source = np.asarray(net).swapaxes(0, 1) if fixed == 1 else np.asarray(net)
    # De Casteljau in the fixed parameter, independently for each remaining
    # control point. Preserve the source binary coefficients exactly.
    curve = [exact_bernstein_value(source[:, i], (start[fixed],))
             for i in range(source.shape[1])]
    if not rational:
        curve = [tuple(row)+(Fraction(1),) for row in curve]
    lo, hi = start[axis], end[axis]
    if hi < lo:
        curve, lo, hi = curve[::-1], 1-lo, 1-hi
    if not (0 <= lo < hi <= 1):
        return None
    if lo:
        _, curve = _split(curve, lo)
    if hi < 1:
        curve, _ = _split(curve, (hi-lo)/(1-lo))
    return curve


def _restrict_exact_curve(curve, start, end):
    if start == end:
        left, _ = _split(curve, start)
        return [left[-1]]
    if end < start:
        curve, start, end = curve[::-1], 1-start, 1-end
    if start:
        _, curve = _split(curve, start)
    if end < 1:
        curve, _ = _split(curve, (end-start)/(1-start))
    return curve


def _surface_rectangle(net, start, end, rational):
    """Exact homogeneous restriction, also allowing reversed/fixed axes."""
    source = np.asarray(net, dtype=float)
    work = np.empty(source.shape, dtype=object)
    for index in np.ndindex(source.shape):
        work[index] = Fraction.from_float(float(source[index]))
    if not rational:
        work = np.concatenate((work, np.full(work.shape[:-1]+(1,), Fraction(1), dtype=object)), axis=-1)
    return _restrict_homogeneous_rectangle(work, start, end)


def _restrict_homogeneous_rectangle(work, start, end):
    """Restrict exact homogeneous coefficients without a float conversion."""
    for axis in range(2):
        oriented = work.swapaxes(0, axis)
        curves = [_restrict_exact_curve(list(map(tuple, oriented[:, j])), start[axis], end[axis])
                  for j in range(oriented.shape[1])]
        work = np.asarray(curves, dtype=object).swapaxes(0, 1).swapaxes(0, axis)
    return work


def _interval_product(a, b):
    values = [x*y for x in a for y in b]
    return min(values), max(values)


def _interval_difference(a, b):
    return a[0]-b[1], a[1]-b[0]


def _coefficient_ranges(net):
    rows = net.reshape(-1, net.shape[-1])
    return [(min(rows[:, k]), max(rows[:, k])) for k in range(net.shape[-1])]


def _source_jet_work(source, rational):
    """Scalar conversion/first-difference operations for one original net."""
    m, n = source.shape[:2]
    coefficients = source.size+(0 if rational else m*n)
    differences = 4*((m-1)*n+m*(n-1))
    return coefficients+2*differences  # one subtraction and one degree product


def _exact_source_jet(source, rational):
    exact = np.empty(source.shape, dtype=object)
    for index in np.ndindex(source.shape):
        exact[index] = Fraction.from_float(float(source[index]))
    if not rational:
        exact = np.concatenate((exact, np.full(exact.shape[:2]+(1,), Fraction(1), dtype=object)), axis=-1)
    derivatives = tuple((exact.shape[axis]-1)*np.diff(exact, axis=axis)
                        if exact.shape[axis] > 1 else None for axis in (0, 1))
    exact.flags.writeable = False
    for derivative in derivatives:
        if derivative is not None:
            derivative.flags.writeable = False
    return exact, derivatives


def _restriction_work(shape, start, end):
    """Operations actually scheduled by exact tensor De Casteljau trims.

Each interpolation performs two products, an addition and ``1-t``.
Both optional splits and the reduced size after a fixed axis are counted;
coefficient visits account for arranging the returned tensor even on an
unrestricted axis. Query pricing uses 128 scalar-operation blocks.
    """
    m, n, width = shape
    dimensions, operations = [m, n], 0
    for axis in (0, 1):
        length, curves = dimensions[axis], dimensions[1-axis]
        fixed = start[axis] == end[axis]
        splits = 1 if fixed else int(start[axis] != 0)+int(end[axis] < 1)
        interpolations = length*(length-1)//2
        operations += length*curves*width+4*interpolations*curves*width*splits
        if not fixed and end[axis] < 1:
            operations += 3*curves  # (end-start)/(1-start) for each curve
        if fixed:
            dimensions[axis] = 1
    return operations


class SourceArcChordBounds:
    """Prepaid immutable source jets reused by many exact arc bounds.

Existence, connectedness and enclosure in each query box remain caller
premises. GLOBAL source cofactors bound parameter rates along a strict
monotone coordinate; exact rational jets and source endpoint enclosures
then bound the entire arc's deviation from its reported XYZ chord.
The public linear STUV path still requires its separate certificate.
    """
    def __init__(self, first, second, *, rational=True, charge=None):
        self.charge = charge
        self.jets = None
        self.exhausted = False
        sources = tuple(np.asarray(net) for net in (first, second))
        if any(net.ndim != 3 or net.shape[-1] != (4 if rational else 3)
               or min(net.shape[:2]) < 1 for net in sources):
            return
        if not self._spend(sum(_source_jet_work(net, rational) for net in sources)):
            return
        try:
            self.jets = tuple(_exact_source_jet(net, rational) for net in sources)
        except (ValueError, OverflowError):
            self.jets = None

    def _spend(self, operations):
        if self.exhausted:
            return False
        if self.charge is not None and not self.charge(max(1, (operations+127)//128)):
            self.exhausted = True
            return False
        return True

    def _restrict(self, net, start, end):
        if not self._spend(_restriction_work(net.shape, start, end)):
            return None
        return _restrict_homogeneous_rectangle(net, start, end)

    def _centered_ranges(self, restricted, origin):
        rows = restricted.shape[0]*restricted.shape[1]
        # Three products/subtractions per coefficient, then two extrema
        # comparisons per homogeneous coefficient.
        if not self._spend((6+8)*rows):
            return None
        centered = restricted.copy()
        for k in range(3):
            centered[..., k] -= origin[k]*centered[..., 3]
        return _coefficient_ranges(centered)

    def bounded(self, box, root_boxes, xyz, monotone_axis, cofactor_bounds, atol):
        if self.jets is None or self.exhausted:
            return False
        # Input Fraction conversions, interval validation and four signed
        # cofactor rate products are query work, not source construction.
        if not self._spend(38+48+48+4):
            return False
        try:
            bounds = [tuple(Fraction.from_float(float(t)) for t in pair) for pair in box]
            roots = [[tuple(Fraction.from_float(float(t)) for t in pair) for pair in root]
                     for root in root_boxes]
            endpoints = [tuple(Fraction.from_float(float(x)) for x in p) for p in xyz]
            if (monotone_axis not in range(4) or len(bounds) != 4 or len(roots) != 2
                    or len(endpoints) != 2 or any(len(root) != 4 for root in roots)
                    or any(not 0 <= lo <= hi <= 1 for lo, hi in bounds)
                    or any(not 0 <= lo <= hi <= 1 for root in roots for lo, hi in root)):
                return False
            minors = []
            for k, (lo, hi) in enumerate(zip(*cofactor_bounds)):
                lo, hi = Fraction.from_float(float(lo)), Fraction.from_float(float(hi))
                minors.append((lo, hi) if k % 2 == 0 else (-hi, -lo))
            denominator = minors[monotone_axis]
            if denominator[0] <= 0 <= denominator[1]:
                return False
            reciprocal = tuple(sorted(1/x for x in denominator))
            rates = [_interval_product(minor, reciprocal) for minor in minors]
            rates[monotone_axis] = (Fraction(1), Fraction(1))
            delta = roots[1][monotone_axis][1]-roots[0][monotone_axis][0]
            if delta <= 0 or roots[0][monotone_axis][1] >= roots[1][monotone_axis][0]:
                return False
            limit = Fraction.from_float(float(atol))**2
            for owner, (exact, source_derivatives) in enumerate(self.jets):
                intervals = bounds[2*owner:2*owner+2]
                start, end = tuple(x[0] for x in intervals), tuple(x[1] for x in intervals)
                restricted = self._restrict(exact, start, end)
                if restricted is None:
                    return False
                # Restriction, differentiation and translation are exact
                # linear operations. Centering the restricted cached jets
                # gives exactly the former centered-source coefficients.
                values = self._centered_ranges(restricted, endpoints[0])
                if values is None:
                    return False
                weight = values[3]
                if weight[0] <= 0:
                    continue
                if not self._spend(4+2*3*32+3*22):
                    return False  # quotient interval arithmetic and velocity sums
                inverse_weight_squared = (1/weight[1]**2, 1/weight[0]**2)
                derivatives = []
                for derivative in source_derivatives:
                    if derivative is not None:
                        derivative = self._restrict(derivative, start, end)
                        if derivative is None:
                            return False
                        derivative_ranges = self._centered_ranges(derivative, endpoints[0])
                        if derivative_ranges is None:
                            return False
                    else:
                        derivative_ranges = [(Fraction(0), Fraction(0))]*4
                    derivatives.append([_interval_product(_interval_difference(
                        _interval_product(derivative_ranges[k], weight),
                        _interval_product(values[k], derivative_ranges[3])),
                        inverse_weight_squared) for k in range(3)])
                velocity = []
                for k in range(3):
                    first_term = _interval_product(derivatives[0][k], rates[2*owner])
                    second_term = _interval_product(derivatives[1][k], rates[2*owner+1])
                    velocity.append((first_term[0]+second_term[0], first_term[1]+second_term[1]))
                endpoint_error = [Fraction(0)]*3
                for root, endpoint in zip(roots, endpoints):
                    pair = root[2*owner:2*owner+2]
                    enclosure = self._restrict(exact, tuple(x[0] for x in pair), tuple(x[1] for x in pair))
                    if enclosure is None or not self._spend(13*enclosure.shape[0]*enclosure.shape[1]):
                        return False
                    rows = enclosure.reshape(-1, 4)
                    if any(row[3] <= 0 for row in rows):
                        return False
                    for k in range(3):
                        endpoint_error[k] = max(endpoint_error[k],
                            max(abs(row[k]/row[3]-endpoint[k]) for row in rows))
                if not self._spend(20):
                    return False
                errors = [delta*(hi-lo)/4+error for (lo, hi), error in zip(velocity, endpoint_error)]
                if sum(error**2 for error in errors) <= limit:
                    return True
        except (OverflowError, ValueError, ZeroDivisionError, IndexError):
            return False
        return False


def source_arc_chord_error_bounded(first, second, box, root_boxes, xyz,
                                   monotone_axis, cofactor_bounds, atol,
                                   rational=True, charge=None):
    """One-shot compatibility wrapper for the exact source secant bound."""
    return SourceArcChordBounds(first, second, rational=rational, charge=charge).bounded(
        box, root_boxes, xyz, monotone_axis, cofactor_bounds, atol)


def _surface_affine_path(net, start, end, rational):
    """Exact homogeneous Bernstein coefficients of S(a+t*(b-a))."""
    work = _surface_rectangle(net, start, end, rational)
    m, n = (size-1 for size in work.shape[:2])
    coefficients = [[Fraction(0)]*4 for _ in range(m+n+1)]
    # On the diagonal, B_i^m(t) B_j^n(t) is a positive multiple of
    # B_(i+j)^(m+n)(t); no sampled fitting or rounded restriction is used.
    for i in range(m+1):
        for j in range(n+1):
            factor = Fraction(comb(m, i)*comb(n, j), comb(m+n, i+j))
            for axis in range(4):
                coefficients[i+j][axis] += factor*work[i, j, axis]
    return list(map(tuple, coefficients))


def source_box_image_diameter_bounded(first, second, box, xyz, atol,
                                      rational=True, charge=None):
    """Bound an entire source-box image together with the output endpoints.

    Once a separate proof places a connected intersection arc in ``box``,
    either surface's image contains that arc. The positive-weight Cartesian
    control hull of an exact restriction contains its complete image. A
    coordinate-box diameter <=atol, including both reported endpoints,
    therefore bounds distance in both directions between arc and chord.
    This proves approximation only, never existence or connectedness.
    """
    sources = (np.asarray(first), np.asarray(second))
    work = sum((1+sum(n-1 for n in net.shape[:2]))*net.size for net in sources)
    if charge is not None and not charge(max(1, (12*work+127)//128)):
        return False
    try:
        bounds = [tuple(Fraction.from_float(float(t)) for t in interval) for interval in box]
        endpoints = [tuple(Fraction.from_float(float(x)) for x in p) for p in xyz]
        if len(bounds) != 4 or len(endpoints) != 2 or any(not 0 <= lo <= hi <= 1 for lo, hi in bounds):
            return False
        limit = Fraction.from_float(float(atol))**2
        for owner, net in enumerate(sources):
            intervals = bounds[2*owner:2*owner+2]
            restricted = _surface_rectangle(
                net, tuple(i[0] for i in intervals), tuple(i[1] for i in intervals), rational)
            rows = restricted.reshape(-1, 4)
            if any(row[-1] <= 0 for row in rows):
                continue
            points = [tuple(x/row[-1] for x in row[:3]) for row in rows]+endpoints
            diameter_squared = sum((max(p[i] for p in points)-min(p[i] for p in points))**2
                                   for i in range(3))
            if diameter_squared <= limit:
                return True
    except (OverflowError, ValueError, ZeroDivisionError):
        return False
    return False


def affine_path_representation_bounded(first, second, start, end, xyz,
                                       atol, rational=True, charge=None):
    """Bound the complete lifted chord against both supplied source surfaces.

    This is an approximation certificate only; root existence and ownership
    require separate evidence. Bounding just the smaller surface image can
    hide a large off-surface excursion in the other parameterization.
    """
    sources = (np.asarray(first), np.asarray(second))
    work = sum((1+sum(n-1 for n in net.shape[:2]))*net.size for net in sources)
    if charge is not None and not charge(max(1, (12*work+127)//128)):
        return False
    try:
        start, end = [tuple(Fraction.from_float(float(t)) for t in p) for p in (start, end)]
        points = [tuple(Fraction.from_float(float(x)) for x in p) for p in xyz]
        if len(points) != 2 or any(t < 0 or t > 1 for p in (start, end) for t in p):
            return False
        tolerance = Fraction.from_float(float(atol))/2
        return all(_chord_error_bounded(
            _surface_affine_path(net, start[2*i:2*i+2], end[2*i:2*i+2], rational),
            points, tolerance) for i, net in enumerate(sources))
    except (OverflowError, ValueError, ZeroDivisionError):
        return False


def _bilinear_path(net, start, end, rational):
    values = [exact_bernstein_value(net, tuple((1-t)*a+t*b for a, b in zip(start, end)))
              for t in (Fraction(0), Fraction(1, 2), Fraction(1))]
    if not rational:
        values = [tuple(row)+(Fraction(1),) for row in values]
    return [values[0], tuple(2*m-(a+b)/2 for a, m, b in zip(*values)), values[2]]


def _chord_error_bounded(curve, endpoints, tolerance):
    """Exact homogeneous Bernstein hull bound against a Cartesian chord."""
    degree = len(curve)-1
    minimum_weight = min(row[-1] for row in curve)
    if minimum_weight <= 0:
        return False
    limit = tolerance**2 * minimum_weight**2
    zero = (Fraction(0),)*4
    for i in range(degree+2):
        a = curve[i-1] if i else zero
        b = curve[i] if i <= degree else zero
        alpha = Fraction(i, degree+1)
        residual = tuple(alpha*a[j]+(1-alpha)*b[j]
                         - alpha*endpoints[1][j]*a[-1]
                         - (1-alpha)*endpoints[0][j]*b[-1] for j in range(3))
        if sum(x*x for x in residual) > limit:
            return False
    return True


def certified_straight_isocurve_path(first, second, start, end, atol,
                                    rational=True, charge=None):
    """Return a two-vertex path when exact source geometry proves the arc.

    A positive-weight straight isocurve with ordered Cartesian control
    points covers its endpoint segment exactly once. Exact plane identity
    and convex quadrilateral inclusion give a unique preimage on the other
    patch. Bernstein bounds also verify the linear public UV interpolation
    and reported XYZ against both source surfaces within ``atol``.
    """
    source = (np.asarray(first), np.asarray(second))
    parameters = [tuple(Fraction.from_float(float(x)) for x in p) for p in (start, end)]
    if any(x < 0 or x > 1 for p in parameters for x in p):
        return None
    tolerance = Fraction.from_float(float(atol))/2
    for owner in (0, 1):
        target = source[1-owner]
        if target.shape[:2] != (2, 2):
            continue
        a, b = [p[2*owner:2*owner+2] for p in parameters]
        if sum(x != y for x, y in zip(a, b)) != 1:
            continue
        work = sum((1+sum(n-1 for n in net.shape[:2]))*net.size for net in source)
        cost = max(1, (6*work+127)//128)
        if charge is not None and not charge(cost):
            return None
        curve = _isocurve(source[owner], a, b, rational)
        if curve is None or any(row[-1] <= 0 for row in curve):
            continue
        points = [tuple(x/row[-1] for x in row[:3]) for row in curve]
        direction = _sub(points[-1], points[0])
        if not any(direction) or any(any(_cross(_sub(p, points[0]), direction)) for p in points):
            continue
        axis = max(range(3), key=lambda i: abs(direction[i]))
        positions = [(p[axis]-points[0][axis])/direction[axis] for p in points]
        if any(x > y for x, y in zip(positions, positions[1:])):
            continue
        target_data = _exact_points(target, rational)
        if target_data is None:
            continue
        target_points, weights = target_data
        if any(w != weights[0] for w in weights):
            continue
        chart = _convex_chart(target_points)
        if chart is None:
            continue
        origin, normal, axes, quad, edges, orientation = chart
        if any(_dot(normal, _sub(p, origin)) for p in points):
            continue
        if any(orientation*_cross2(edge, _sub(tuple(p[i] for i in axes), q)) < 0
               for p in (points[0], points[-1]) for edge, q in zip(edges, quad)):
            continue
        try:
            xyz = np.asarray([[float(x) for x in p] for p in (points[0], points[-1])])
            represented = [tuple(Fraction.from_float(float(x)) for x in p) for p in xyz]
        except (OverflowError, ValueError):
            continue
        uv_start, uv_end = [p[2*(1-owner):2*(1-owner)+2] for p in parameters]
        target_curve = _bilinear_path(target, uv_start, uv_end, rational)
        if not (_chord_error_bounded(curve, represented, tolerance)
                and _chord_error_bounded(target_curve, represented, tolerance)):
            continue
        return np.asarray([start, end], dtype=float), xyz
    return None
