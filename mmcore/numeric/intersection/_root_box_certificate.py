"""Bernstein residual, root-existence, and interval-Jacobian certificates.

Uniqueness tests establish at most one root. Separate exact evaluation or
Krawczyk inclusion is required for existence; a small residual is not proof.
"""
from itertools import combinations
from fractions import Fraction

import numpy as np

from mmcore.numeric.bern import bernstein_partial_derivative_coeffs
from mmcore.numeric._bezier_common import restrict_net_axis_v


def exact_bernstein_value(net, parameters):
    """Evaluate supplied binary coefficients at exact rational parameters."""
    source = np.asarray(net)
    work = np.empty(source.shape, dtype=object)
    for index in np.ndindex(source.shape):
        work[index] = Fraction.from_float(float(source[index]))
    for parameter in parameters:
        t = parameter if isinstance(parameter, Fraction) else Fraction.from_float(float(parameter))
        while len(work) > 1:
            work = (1-t)*work[:-1]+t*work[1:]
        work = work[0]
    return tuple(work)


def exact_residual_value(first, second, parameters, rational=False):
    a = exact_bernstein_value(first, parameters[:1])
    b = exact_bernstein_value(second, parameters[1:])
    if rational:
        return tuple(x*b[-1]-y*a[-1] for x, y in zip(a[:-1], b[:-1]))
    return tuple(x-y for x, y in zip(a, b))


def affine_residual_is_zero(first, second, start, end, rational=False):
    """Exact homogeneous polynomial identity along an affine parameter map."""
    degree = len(first)-1+sum(n-1 for n in second.shape[:-1])
    a = tuple(Fraction.from_float(float(x)) for x in start)
    b = tuple(Fraction.from_float(float(x)) for x in end)
    for index in range(degree+1):
        t = Fraction(index, max(1, degree))
        parameters = tuple((1-t)*x+t*y for x, y in zip(a, b))
        if any(exact_residual_value(first, second, parameters, rational)):
            return False
    return True


def _common_plane_projection(first, second, rational):
    points = []
    for curve in (first, second):
        for row in curve:
            values = tuple(Fraction.from_float(float(x)) for x in row)
            weight = values[-1] if rational else Fraction(1)
            if weight <= 0:
                return None
            points.append(tuple(x/weight for x in values[:3]))
    origin = points[0]
    differences = [tuple(x-y for x, y in zip(point, origin)) for point in points[1:]]
    for a, b in combinations(differences, 2):
        normal = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
        if any(normal):
            if any(sum(x*y for x, y in zip(normal, d)) for d in differences):
                return None
            drop = max(range(3), key=lambda i: abs(normal[i]))
            return tuple(i for i in range(3) if i != drop)
    return None


def root_box_enclosure(net, box, source_scale=None, coefficient_error=None):
    """Return a source-proved Krawczyk enclosure, or None.

    The image encloses every root in the input box. Strict image inclusion
    establishes existence independently of the representative supplied by
    any numerical corrector.
    """
    from mmcore.numeric.bern import bernstein_eval_nd
    n = len(box)
    if net.shape[-1] != n or any(hi <= lo for lo, hi in box):
        return None
    restricted = net
    for axis, (lo, hi) in enumerate(box):
        restricted = restrict_net_axis_v(restricted, axis, lo, hi, 0., 1.)
    error = residual_roundoff_bound(net, depth=2*n, source_scale=source_scale)
    if coefficient_error is not None:
        error = np.nextafter(error+np.asarray(coefficient_error), np.inf)
    axes = tuple(range(n))
    magnitude = np.max(np.abs(restricted), axis=axes)
    eps = np.finfo(float).eps
    lower, upper = [], []
    for axis in axes:
        degree = restricted.shape[axis]-1
        if degree == 0:
            return None
        derivative = bernstein_partial_derivative_coeffs(restricted, axis=axis)
        derivative_error = degree*(2.*error+4.*eps*magnitude)
        lower.append(np.nextafter(derivative.min(axis=axes)-derivative_error, -np.inf))
        upper.append(np.nextafter(derivative.max(axis=axes)+derivative_error, np.inf))
    lower, upper = np.asarray(lower).T, np.asarray(upper).T
    midpoint, radius = .5*(lower+upper), .5*(upper-lower)
    try:
        inverse = np.linalg.inv(midpoint)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(inverse)):
        return None
    value = bernstein_eval_nd(restricted, np.full(n, .5))
    gamma = (2*n+2)*eps/(1.-(2*n+2)*eps)
    absolute_inverse = np.abs(inverse)
    arithmetic = gamma*(np.eye(n)+absolute_inverse@(np.abs(midpoint)+radius))
    linear_radius = .5*np.sum(np.abs(np.eye(n)-inverse@midpoint)+absolute_inverse@radius+arithmetic, axis=1)
    correction = (np.abs(inverse@value)+absolute_inverse@error
                  + gamma*absolute_inverse@(np.abs(value)+error))
    image_radius = np.nextafter(correction+linear_radius, np.inf)
    if not np.all(image_radius < .5):
        return None
    result = []
    for (lo, hi), radius in zip(box, image_radius):
        local_lo = max(0., np.nextafter(.5-radius, -np.inf))
        local_hi = min(1., np.nextafter(.5+radius, np.inf))
        span_lo, span_hi = np.nextafter(hi-lo, -np.inf), np.nextafter(hi-lo, np.inf)
        lower = np.nextafter(lo+np.nextafter(span_lo*local_lo, -np.inf), -np.inf)
        upper = np.nextafter(lo+np.nextafter(span_hi*local_hi, np.inf), np.inf)
        # Intersecting the Krawczyk image with its original existence box
        # is safe: the same root is independently known to belong to both.
        result.append((max(lo, lower), min(hi, upper)))
    return tuple(result)


def root_box_contains_zero(net, box, source_scale=None, coefficient_error=None):
    """Sufficient Krawczyk existence test for a square residual system."""
    return root_box_enclosure(net, box, source_scale, coefficient_error) is not None


def root_existence_certificate(first, second, parameters, box, net,
                               rational=False, source_scale=None):
    """Certify a source root, never infer existence from near-zero residual."""
    if any(t < 0. or t > 1. for t in parameters):
        return None
    if not any(exact_residual_value(first, second, parameters, rational)):
        return 'exact_parameter_identity'
    if box is None:
        return None
    if len(parameters) == net.shape[-1]:
        return 'krawczyk_inclusion' if root_box_contains_zero(net, box, source_scale) else None
    if len(parameters) == 2 and net.shape[-1] == 3:
        projection = _common_plane_projection(first, second, rational)
        if projection is not None:
            projected_scale = None if source_scale is None else source_scale[list(projection)]
            if root_box_contains_zero(net[..., list(projection)], box, projected_scale):
                return 'planar_krawczyk_inclusion'
    return None


def residual_roundoff_bound(net, depth=0, source_scale=None):
    """Carry original operand error through bounded de Casteljau chains."""
    eps = float(np.finfo(float).eps)
    axes = tuple(range(net.ndim-1))
    operations = 1 + 6*sum(n-1 for n in net.shape[:-1])*(int(depth)+1)
    error = (operations*eps/(1.0-operations*eps))*np.max(np.abs(net), axis=axes)
    # Relative gamma bounds assume normal arithmetic. Price one minimum
    # subnormal quantum per counted operation as a conservative additive
    # rounding allowance; a single final nextafter cannot cover a chain.
    quantum = float(np.nextafter(0., 1.))
    error += operations*quantum
    if source_scale is not None:
        error += (3.0*eps/(1.0-3.0*eps))*source_scale
        error += 3.0*quantum
    return np.nextafter(error, np.inf)


def jacobian_is_injective(net, axes, source_error):
    """Prove injectivity in selected parameters uniformly over the others.

    For a fixed left preconditioner R, ||I-R J||_infinity < 1 bounds
    the derivative of x-R G(x) by a contraction on the convex box.
    Thus two roots cannot share the same remaining parameters.
    """
    eps = float(np.finfo(float).eps)
    reduce_axes = tuple(range(net.ndim-1))
    n = len(axes)
    m = net.shape[-1]
    if n > m:
        return False
    magnitude = np.max(np.abs(net), axis=reduce_axes)
    lower = np.empty((m, n))
    upper = np.empty((m, n))
    for column, axis in enumerate(axes):
        degree = net.shape[axis]-1
        if degree == 0:
            return False
        derivative = bernstein_partial_derivative_coeffs(net, axis=axis)
        error = degree*(2.0*source_error + 4.0*eps*magnitude)
        lower[:, column] = np.nextafter(derivative.min(axis=reduce_axes)-error, -np.inf)
        upper[:, column] = np.nextafter(derivative.max(axis=reduce_axes)+error, np.inf)
    mid = .5*(lower+upper)
    radius = .5*(upper-lower)

    def contracts(preconditioner):
        if not np.all(np.isfinite(preconditioner)):
            return False
        residual = np.eye(n)-preconditioner@mid
        uncertainty = np.abs(preconditioner)@radius
        operations = 2*m+2
        gamma = operations*eps/(1.0-operations*eps)
        arithmetic = gamma*(np.eye(n)+np.abs(preconditioner)@(np.abs(mid)+radius))
        contraction = np.nextafter(
            np.sum(np.abs(residual)+uncertainty+arithmetic, axis=1), np.inf)
        return float(np.max(contraction)) < 1.0

    # A coordinate chart can be stronger than a least-squares inverse:
    # graph surfaces retain their exact (u,v) chart despite steep heights.
    for selected in combinations(range(m), n):
        try:
            inverse = np.linalg.inv(mid[list(selected), :])
        except np.linalg.LinAlgError:
            continue
        preconditioner = np.zeros((n, m))
        preconditioner[:, list(selected)] = inverse
        if contracts(preconditioner):
            return True
    if m != n:
        try:
            return contracts(np.linalg.pinv(mid))
        except np.linalg.LinAlgError:
            pass
    return False


def unique_root_box(net, root, radii, source_scale=None, coefficient_error=None):
    """Return a clipped box containing at most one root, or None."""
    box = tuple((max(0., x-r), min(1., x+r)) for x, r in zip(root, radii))
    if any(hi <= lo for lo, hi in box):
        return None
    restricted = net
    for axis, (lo, hi) in enumerate(box):
        restricted = restrict_net_axis_v(restricted, axis, lo, hi, 0., 1.)
    error = residual_roundoff_bound(net, depth=2*len(box), source_scale=source_scale)
    if coefficient_error is not None:
        error = np.nextafter(error+np.asarray(coefficient_error), np.inf)
    return box if jacobian_is_injective(restricted, tuple(range(len(box))), error) else None


def root_boxes_have_same_root(net, first, second, source_scale=None, coefficient_error=None):
    """Prove two boxes with independently established roots share that root."""
    box = tuple((min(a[0], b[0]), max(a[1], b[1])) for a, b in zip(first, second))
    restricted = net
    for axis, (lo, hi) in enumerate(box):
        restricted = restrict_net_axis_v(restricted, axis, lo, hi, 0., 1.)
    error = residual_roundoff_bound(net, depth=2*len(box), source_scale=source_scale)
    if coefficient_error is not None:
        error = np.nextafter(error+np.asarray(coefficient_error), np.inf)
    return jacobian_is_injective(restricted, tuple(range(len(box))), error)


def residual_hull_excludes_zero(net, error):
    """Exclude zero using source-bounded component and fixed projections.

    The caller supplies the error of the original residual construction
    and its restrictions. A direction only proposes a scalar Bernstein
    hull; its dot-product rounding is charged before any sign decision.
    """
    flat = np.asarray(net).reshape(-1, net.shape[-1])
    error = np.asarray(error)
    if (np.any(flat.min(axis=0) > error)
            or np.any(flat.max(axis=0) < -error)):
        return True
    eps = np.finfo(float).eps

    def excludes(direction):
        scale = float(np.max(np.abs(direction)))
        if scale == 0. or not np.isfinite(scale):
            return False
        direction = direction/scale
        projected = flat@direction
        operands = np.max(np.abs(flat), axis=0)@np.abs(direction)
        n = flat.shape[1]
        gamma = (2*n-1)*eps/(1.-(2*n-1)*eps)
        margin = np.nextafter(error@np.abs(direction)+gamma*operands, np.inf)
        return bool(np.min(projected) > margin or np.max(projected) < -margin)

    mean = flat.mean(axis=0)
    if excludes(mean):
        return True
    centered = flat-mean
    scale = float(np.max(np.abs(centered)))
    if scale > 0. and np.isfinite(scale):
        centered = centered/scale
    try:
        _, directions = np.linalg.eigh(centered.T@centered)
    except np.linalg.LinAlgError:
        return False
    return excludes(directions[:, 0])
