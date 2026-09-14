"""Numerical plane intersections that reduce to complete surface isolines.

Surfaces whose homogeneous plane height varies in one parameter admit
a one-dimensional root search. The free parameter remains a curve, even
at a multiple root or a singular surface ruling. Unsupported charts use
the ordinary SSX search.
"""
from __future__ import annotations

import numpy as np

from mmcore.numeric._bern_roots import bernstein_roots
from mmcore.numeric._bezier_common import (
    bernstein_product_1d, eval_surface, subdivide_curve, geometry_collapsed,
)
from mmcore.numeric._work_budget import REASON_TRACE_UNVERIFIED, REASON_PARAMETER_FIBER


def _evaluate(net, parameter):
    values = np.array(net, dtype=float, copy=True)
    for length in range(len(values) - 1, 0, -1):
        values[:length] = ((1. - parameter) * values[:length]
                           + parameter * values[1:length + 1])
    return values[0]


def _restrict(net, lo, hi):
    if lo == hi:
        return _evaluate(net, lo)[None, ...]
    if hi < 1.:
        net, _ = subdivide_curve(net, hi)
    if lo > 0.:
        _, net = subdivide_curve(net, lo / hi)
    return net


def _polynomial(net):
    weights = np.asarray(net[..., -1])
    if not np.all(weights == weights.flat[0]):
        return None
    points = np.asarray(net[..., :3], dtype=float) / weights.flat[0]
    return points if np.all(np.isfinite(points)) else None


def _arithmetic_scale(points):
    return (64. * np.finfo(float).eps * sum(points.shape[:2])
            * max(float(np.max(np.abs(points))), np.finfo(float).tiny))


def _root_tolerance(speed, atol):
    """Leave room for geometric interpolation after locating a scalar root.

    The absolute parameter error is multiplied by the speed of the
    rendered curve or surface. A fixed parameter tolerance can therefore
    move an otherwise valid small-residual root by many modeling units.
    If the requested location is below double precision's parameter
    resolution, let the general search handle the chart instead.
    """
    if not np.isfinite(speed):
        return None
    if speed <= 0.:
        return 1e-13
    tolerance = .25 * float(atol) / float(speed)
    if tolerance < np.finfo(float).eps:
        return None
    return min(1e-13, tolerance)


def _surface_speed_bound(net, axis):
    """Bound a positive-weight chart derivative in physical coordinates."""
    weights = np.asarray(net[..., 3], dtype=float)
    points = np.asarray(net[..., :3], dtype=float) / weights[..., None]
    weights = weights / np.max(weights)
    # Center before differentiating so a world translation does not
    # inflate the rational quotient-rule bound.
    numerator = (points - points[0, 0]) * weights[..., None]
    degree = net.shape[axis] - 1
    if degree <= 0:
        return 0.
    derivative = degree * np.diff(numerator, axis=axis)
    weight_derivative = degree * np.diff(weights, axis=axis)
    minimum_weight = float(np.min(weights))
    radius = float(np.max(np.linalg.norm(numerator, axis=-1))) / minimum_weight
    return ((float(np.max(np.linalg.norm(derivative, axis=-1)))
             + radius * float(np.max(np.abs(weight_derivative))))
            / minimum_weight)


def _affine_plane(points, atol):
    if points.shape[0] < 2 or points.shape[1] < 2:
        return None
    origin = points[0, 0]
    first, second = points[-1, 0] - origin, points[0, -1] - origin
    axes = np.column_stack((first, second))
    lengths = np.linalg.svd(axes, compute_uv=False)
    if lengths[-1] <= 64. * np.finfo(float).eps * lengths[0]:
        return None
    u = np.linspace(0., 1., len(points))[:, None, None]
    v = np.linspace(0., 1., points.shape[1])[None, :, None]
    expected = origin + u * first + v * second
    error = np.max(np.linalg.norm(points - expected, axis=-1))
    if error > min(_arithmetic_scale(points), .01 * atol):
        return None
    normal = np.cross(first, second)
    normal /= np.linalg.norm(normal)
    inverse = np.linalg.pinv(axes)
    return origin, normal, inverse


def _normal_is_zero(first, second, scale):
    coefficients = []
    for axis in range(3):
        a, b = (axis + 1) % 3, (axis + 2) % 3
        coefficients.append(bernstein_product_1d(first[:, a], second[:, b])
                            - bernstein_product_1d(first[:, b], second[:, a]))
    return np.max(np.abs(coefficients)) <= scale


def try_plane_coincidence(first_h, second_h, atol, budget):
    """Represent a full affine projected patch within the plane's CAD tolerance.

    The source may bend in height. Its control hull must remain within
    ``atol`` and its affine projection must fit inside the planar chart.
    This keeps a tolerance-coincident area from collapsing to one isoline.
    """
    nets = (first_h, second_h)
    points = [_polynomial(net) for net in nets]
    for target in (1, 0):
        if points[target] is None or points[1-target] is None:
            continue
        plane = _affine_plane(points[target], atol)
        if plane is None:
            continue
        origin, normal, inverse = plane
        source = points[1-target]
        height = (source-origin) @ normal
        if np.max(np.abs(height)) > atol:
            continue
        projected = source-height[..., None]*normal
        if _affine_plane(projected, atol) is None:
            continue
        uv = (projected-origin) @ inverse.T
        margin = 128*np.finfo(float).eps*max(1., np.max(np.abs(uv)))
        if np.any(uv < -margin) or np.any(uv > 1.+margin):
            continue
        from mmcore.numeric.intersection.ssx._ssx5_overlap import assemble_overlap_regions
        from mmcore.nurbs._nurbs_param_tol import bez_surface_param_tolerance
        tolerances = np.array([*bez_surface_param_tolerance(first_h, atol, rational=True),
                               *bez_surface_param_tolerance(second_h, atol, rational=True)])
        assembled = assemble_overlap_regions(
            first_h, second_h, atol=atol, ptol4=tolerances,
            charge=lambda n: budget.charge_cells(n, 'coincidence'))
        if not assembled['regions'] or budget.exhausted:
            return None
        out = dict(branches=[], points=[], singularities=[], overlap_regions=[])
        budget.extend_output(out['branches'], assembled['rim_branches'], 'coincidence')
        budget.extend_output(out['overlap_regions'], assembled['regions'], 'coincidence')
        return out
    return None


def try_isoline_ssx(first_h, second_h, atol, budget):
    """Return ordinary SSX output lists, or None for the general search.

    Work is charged to the caller's shared ledger. Root enumeration and
    clipping must finish before this tier replaces the general search;
    rendering a known curve can retain its completed prefix on exhaustion.
    No algebraic diagnostics or root certificates are exposed.
    """
    from mmcore.numeric.intersection.ssx._bez_ssx5 import (
        SSXBranch, SSXPoint, SSXSingularity,
    )

    nets = [np.asarray(first_h), np.asarray(second_h)]
    if not budget.charge_cells(sum(np.prod(net.shape[:2]) for net in nets), "isoline_setup"):
        return None
    points = [_polynomial(net) for net in nets]
    source_index = None
    for target_index in (1, 0):
        if points[target_index] is None:
            continue
        plane = _affine_plane(points[target_index], atol)
        if plane is None:
            continue
        source = nets[1 - target_index]
        origin, normal, inverse = plane
        height = np.einsum('ijk,k->ij',
                           source[..., :3] - origin * source[..., 3:], normal)
        if np.max(np.abs(height) / source[..., 3]) <= atol:
            # The whole patch is within the modeling tolerance of the
            # plane. Let overlap assembly represent that coincidence;
            # returning only the polynomial's zero isoline loses its area.
            continue
        noise = (_arithmetic_scale(source[..., :3])
                 + _arithmetic_scale(points[target_index]) * np.max(source[..., 3]))
        height_allowance = min(noise, .01 * atol * float(np.min(source[..., 3])))
        for fixed_axis in (0, 1):
            ordered = np.moveaxis(height, fixed_axis, 0)
            profile = ordered[:, 0]
            profile_variation = float(np.max(np.abs(ordered - profile[:, None])))
            ordered_weights = np.moveaxis(source[..., 3], fixed_axis, 0)
            if (len(profile) > 1 and np.ptp(profile) > noise
                    and profile_variation <= height_allowance
                    and np.all(ordered_weights == ordered_weights[:, :1])):
                source_index = 1 - target_index
                break
        if source_index is not None:
            break
    if source_index is None:
        return None

    def roots(coefficients, parameter_tol):
        return bernstein_roots(
            coefficients, parameter_tol=parameter_tol,
            charge=lambda amount: budget.charge_cells(amount, "isoline_roots"))

    fixed_tolerance = _root_tolerance(_surface_speed_bound(source, fixed_axis), atol)
    if fixed_tolerance is None:
        return None
    census = roots(profile, fixed_tolerance)
    if not census.complete or census.constant_zero:
        return None
    if (not census.roots and profile_variation > 0.
            and np.min(height) <= 0. <= np.max(height)):
        # An approximately independent profile cannot exclude roots of
        # the full height net when that net still straddles the plane.
        return None
    output = dict(branches=[], points=[], singularities=[], overlap_regions=[])
    ordered_source = np.moveaxis(source, fixed_axis, 0)
    degree = len(profile) - 1
    plans = []
    for fixed in census.roots:
        if not budget.charge_cells(max(1, int(source.size // 3)), "isoline_setup"):
            return None
        homogeneous_curve = _evaluate(ordered_source, fixed)
        weights = homogeneous_curve[:, 3:]
        curve = homogeneous_curve[:, :3] / weights
        # A stationary numerical root must still satisfy the caller's
        # geometric accuracy; a tiny polynomial residual alone is not an
        # acceptable rendered curve at an arbitrarily tighter tolerance.
        if np.max(np.abs((curve - origin) @ normal)) > .25 * atol:
            return None
        uv = (curve - origin) @ inverse.T
        free_speed = ((len(curve)-1) * float(np.max(np.linalg.norm(np.diff(curve, axis=0), axis=1)))
                      if len(curve) > 1 else 0.)
        free_tolerance = _root_tolerance(free_speed, atol)
        if free_tolerance is None:
            return None
        breaks = {0., 1.}
        for axis in (0, 1):
            for boundary in (0., 1.):
                clipped = roots(uv[:, axis] - boundary, free_tolerance)
                if not clipped.complete:
                    return None
                breaks.update(clipped.roots)
        breaks = sorted(breaks)
        intervals = []
        for lo, hi in zip(breaks, breaks[1:]):
            midpoint = _evaluate(uv, .5 * (lo + hi))
            uv_slack = 64*np.finfo(float).eps*len(uv)*max(1., np.max(np.abs(uv)))
            if np.all((midpoint >= -uv_slack) & (midpoint <= 1.+uv_slack)):
                if intervals and intervals[-1][1] == lo:
                    intervals[-1] = intervals[-1][0], hi
                else:
                    intervals.append((lo, hi))
        derivative_net = degree * np.diff(ordered_source, axis=0)
        homogeneous_derivative = _evaluate(derivative_net, fixed)
        derivative = ((homogeneous_derivative[:, :3]
                       - curve * homogeneous_derivative[:, 3:]) / weights)
        free_derivative = ((len(curve) - 1) * np.diff(curve, axis=0)
                           if len(curve) > 1 else np.zeros((1, 3)))
        derivative_scale = (np.max(np.abs(derivative_net[..., :3]))
                            + np.max(np.abs(curve)) * np.max(np.abs(derivative_net[..., 3]))) / np.min(weights)
        free_scale = max(np.max(np.abs(free_derivative)), np.finfo(float).tiny)
        normal_scale = 64. * np.finfo(float).eps * source.size * derivative_scale * free_scale
        cusp = _normal_is_zero(derivative, free_derivative, normal_scale)
        height_derivative = float(_evaluate(degree * np.diff(profile), fixed))
        tangent = abs(height_derivative) <= 64. * np.finfo(float).eps * degree * np.max(np.abs(profile))
        plans.append((fixed, curve, uv, intervals, breaks, cusp, tangent))

    def coordinates(fixed, free, xyz):
        source_uv = np.array([fixed, free]) if fixed_axis == 0 else np.array([free, fixed])
        target_uv = (xyz - origin) @ inverse.T
        if np.any(target_uv < -1e-11) or np.any(target_uv > 1. + 1e-11):
            return None
        target_uv = np.clip(target_uv, 0., 1.)
        paired = np.concatenate((source_uv, target_uv) if source_index == 0
                                else (target_uv, source_uv))
        if any(np.linalg.norm(eval_surface(net, *pair, rational=True) - xyz) > .25 * atol
               for net, pair in zip(nets, (paired[:2], paired[2:]))):
            return None
        return paired

    for fixed, curve, uv, intervals, breaks, cusp, tangent in plans:
        for lo, hi in intervals:
            controls = _restrict(curve, lo, hi)
            start = coordinates(fixed, lo, controls[0])
            if start is None:
                budget.mark_incomplete(REASON_TRACE_UNVERIFIED)
                continue
            parameters, xyz = [start], [controls[0]]
            stack = [(lo, hi, controls)]
            while stack:
                if not budget.charge_cells(max(1, len(curve)**2 // 128), "isoline_trace"):
                    break
                a, b, net = stack.pop()
                linear = np.linspace(net[0], net[-1], len(net))
                error = float(np.max(np.linalg.norm(net - linear, axis=1)))
                if error <= .5 * atol:
                    q = coordinates(fixed, b, net[-1])
                    if q is None:
                        budget.mark_incomplete(REASON_TRACE_UNVERIFIED)
                        break
                    parameters.append(q)
                    xyz.append(net[-1])
                else:
                    middle = .5 * (a + b)
                    if not a < middle < b:
                        budget.mark_incomplete(REASON_TRACE_UNVERIFIED)
                        break
                    left, right = subdivide_curve(net, .5)
                    stack.extend(((middle, b, right), (a, middle, left)))
            if len(parameters) >= 2:
                parameters, xyz = np.asarray(parameters), np.asarray(xyz)
                # A whole free parameter interval can map to one spatial
                # point. Preserve its parameter samples as singularity
                # evidence, but do not publish a zero-length curve.
                collapsed = (geometry_collapsed(controls)
                             and np.max(np.linalg.norm(controls-controls[0], axis=1))
                             <= .25*atol)
                links = []
                if collapsed:
                    if not budget.append_output(output['points'],
                                                SSXPoint(parameters[0], xyz[0]), 'isoline'):
                        return output
                    budget.mark_incomplete(REASON_PARAMETER_FIBER)
                else:
                    boundary = fixed in (0., 1.)
                    branch = SSXBranch(
                        curve=(parameters, xyz), overlap=boundary,
                        kind='overlap' if boundary else 'tangential' if tangent else 'transversal')
                    if not budget.append_output(output['branches'], branch, 'isoline'):
                        return output
                    links = [(len(output['branches']) - 1, 0)]
                if cusp:
                    singularity = SSXSingularity(
                        'cusp_curve', parameters[0], xyz[0], samples=parameters,
                        surface=source_index + 1,
                        branch_links=links)
                    if not budget.append_output(output['singularities'], singularity, 'isoline'):
                        return output
            if budget.exhausted:
                return output
        # Clipping can leave a single corner contact with no interval.
        for free in breaks:
            if any(lo <= free <= hi for lo, hi in intervals):
                continue
            xyz = _evaluate(curve, free)
            q = coordinates(fixed, free, xyz)
            if q is not None:
                if not budget.append_output(output['points'], SSXPoint(q, xyz), 'isoline'):
                    return output
    return output
