"""Numerical isolines on surfaces sharing an injective projected chart."""
from itertools import product

import numpy as np

from mmcore.numeric._bern_roots import bernstein_roots
from mmcore.numeric._bezier_common import eval_surface, subdivide_curve
from mmcore.numeric._work_budget import REASON_TRACE_UNVERIFIED
from mmcore.numeric.intersection.ssx._ssx_isolines import (
    _arithmetic_scale, _evaluate, _polynomial, _root_tolerance,
    _surface_speed_bound,
)


def try_matched_isoline_ssx(first_h, second_h, atol, budget):
    """Resolve a one-parameter height difference on a common chart.

    The projected chart must be affine in the fixed parameter and strictly
    monotone in a perpendicular coordinate of the free parameter. Thus
    equal projected points have the same paired parameters. No sampled
    proximity test is used to assume that correspondence.
    """
    from mmcore.numeric.intersection.ssx._bez_ssx5 import SSXBranch

    if not budget.charge_cells(int(np.prod(first_h.shape[:2]) + np.prod(second_h.shape[:2])),
                               'isoline_setup'):
        return None
    first, second = _polynomial(first_h), _polynomial(second_h)
    if first is None or second is None:
        return None
    noise = _arithmetic_scale(first) + _arithmetic_scale(second)
    selected = None
    for transpose, flip_first, flip_second in product((False, True), repeat=3):
        aligned = second.swapaxes(0, 1) if transpose else second
        if flip_first:
            aligned = aligned[::-1]
        if flip_second:
            aligned = aligned[:, ::-1]
        if aligned.shape != first.shape:
            continue
        if not budget.charge_cells(int(np.prod(first.shape[:2])), 'isoline_setup'):
            return None
        difference = first - aligned
        if np.max(np.linalg.norm(difference, axis=-1)) <= atol:
            continue  # Within-tolerance coincidence belongs to region assembly.
        if np.max(np.abs(difference)) <= noise:
            continue  # Coincident surfaces need a region representation.
        _, singular_values, directions = np.linalg.svd(difference.reshape(-1, 3), full_matrices=False)
        normal = directions[0]
        height = difference @ normal
        remainder = difference - height[..., None] * normal
        if np.max(np.linalg.norm(remainder, axis=-1)) > min(noise, .01 * atol):
            continue
        projected = first - (first @ normal)[..., None] * normal
        for fixed_axis in (0, 1):
            ordered = np.moveaxis(projected, fixed_axis, 0)
            heights = np.moveaxis(height, fixed_axis, 0)
            profile = heights[:, 0]
            if (len(profile) < 2 or ordered.shape[1] < 2
                    or np.ptp(profile) <= noise
                    or np.max(np.abs(heights - profile[:, None])) > min(noise, .01 * atol)):
                continue
            direction = ordered[-1, 0] - ordered[0, 0]
            if np.linalg.norm(direction) <= noise:
                continue
            levels = np.linspace(0., 1., len(ordered))[:, None, None]
            expected = ordered[0][None, ...] + levels * direction
            if np.max(np.linalg.norm(ordered - expected, axis=-1)) > min(noise, .01 * atol):
                continue
            perpendicular = np.cross(normal, direction)
            perpendicular /= np.linalg.norm(perpendicular)
            free_coordinate = ordered[0] @ perpendicular
            derivative = (len(free_coordinate) - 1) * np.diff(free_coordinate)
            if not (np.all(derivative > noise) or np.all(derivative < -noise)):
                continue
            selected = aligned, fixed_axis, profile, (transpose, flip_first, flip_second)
            break
        if selected is not None:
            break
    if selected is None:
        return None
    aligned, fixed_axis, profile, orientation = selected
    # Both charts use the same aligned parameters. The larger derivative
    # controls how far a root-location error can move their common curve.
    aligned_h = np.concatenate((aligned, np.ones(aligned.shape[:2] + (1,))), axis=2)
    parameter_tol = _root_tolerance(max(
        _surface_speed_bound(first_h, fixed_axis),
        _surface_speed_bound(aligned_h, fixed_axis)), atol)
    if parameter_tol is None:
        return None
    census = bernstein_roots(
        profile, parameter_tol=parameter_tol,
        charge=lambda amount: budget.charge_cells(amount, 'isoline_roots'))
    if not census.complete or census.constant_zero:
        return None
    output = dict(branches=[], points=[], singularities=[], overlap_regions=[])
    degree = len(profile) - 1

    def parameters(fixed, free):
        first_uv = np.array([fixed, free]) if fixed_axis == 0 else np.array([free, fixed])
        second_uv = first_uv.copy()
        transpose, flip_first, flip_second = orientation
        if flip_first:
            second_uv[0] = 1. - second_uv[0]
        if flip_second:
            second_uv[1] = 1. - second_uv[1]
        if transpose:
            second_uv = second_uv[::-1]
        return np.concatenate((first_uv, second_uv))

    curves = []
    for fixed in census.roots:
        if not budget.charge_cells(int(np.prod(first.shape[:2])), 'isoline_setup'):
            return None
        a = _evaluate(np.moveaxis(first, fixed_axis, 0), fixed)
        b = _evaluate(np.moveaxis(aligned, fixed_axis, 0), fixed)
        if np.max(np.linalg.norm(a - b, axis=1)) > .25 * atol:
            return None
        curves.append((fixed, a, b))
    for fixed, a, b in curves:
        q0 = parameters(fixed, 0.)
        xyz0 = .5 * (a[0] + b[0])
        if any(np.linalg.norm(eval_surface(net, *uv, rational=True) - xyz0) > .25 * atol
               for net, uv in ((first_h, q0[:2]), (second_h, q0[2:]))):
            budget.mark_incomplete(REASON_TRACE_UNVERIFIED)
            continue
        qs, xyz = [q0], [xyz0]
        stack = [(0., 1., a, b)]
        while stack:
            if not budget.charge_cells(max(1, 2 * len(a)**2 // 128), 'isoline_trace'):
                break
            lo, hi, left, right = stack.pop()
            ends = .5 * (left[[0, -1]] + right[[0, -1]])
            linear = np.linspace(ends[0], ends[1], len(left))
            error = max(np.max(np.linalg.norm(left - linear, axis=1)),
                        np.max(np.linalg.norm(right - linear, axis=1)))
            if error <= .5 * atol:
                q = parameters(fixed, hi)
                if any(np.linalg.norm(eval_surface(net, *uv, rational=True) - ends[1]) > .25 * atol
                       for net, uv in ((first_h, q[:2]), (second_h, q[2:]))):
                    budget.mark_incomplete(REASON_TRACE_UNVERIFIED)
                    break
                qs.append(q)
                xyz.append(ends[1])
            else:
                mid = .5 * (lo + hi)
                if not lo < mid < hi:
                    budget.mark_incomplete(REASON_TRACE_UNVERIFIED)
                    break
                a0, a1 = subdivide_curve(left, .5)
                b0, b1 = subdivide_curve(right, .5)
                stack.extend(((mid, hi, a1, b1), (lo, mid, a0, b0)))
        if len(qs) >= 2:
            height_derivative = float(_evaluate(degree * np.diff(profile), fixed))
            tangent = abs(height_derivative) <= 64. * np.finfo(float).eps * degree * np.max(np.abs(profile))
            boundary = fixed in (0., 1.)
            branch = SSXBranch(
                curve=(np.asarray(qs), np.asarray(xyz)), overlap=boundary,
                kind='overlap' if boundary else 'tangential' if tangent else 'transversal')
            if not budget.append_output(output['branches'], branch, 'isoline'):
                return output
        if budget.exhausted:
            return output
    return output
