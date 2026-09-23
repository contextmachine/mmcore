"""Numerical necessary boxes for a surface residual in Bernstein form.

Clipping narrows a copy of the residual net. It does not assign roots to
cells, prove that roots exist, or decide whether a CAD contact is absent.
Callers must validate their geometric coverage on both restricted sources.
"""
from __future__ import annotations

import numpy as np

from mmcore.numeric._bezier_common import restrict_net_axis_v
from mmcore.numeric.intersection.ssx._ssx5_singular import (
    _HULL_MARGIN_K_EPS,
    psi_vector_net,
)

_EPS = np.finfo(float).eps
_UNIT_BOX = ((0.0, 1.0),) * 4


def residual_coordinate_scale(first, second):
    """Scalar operand scale for forming the homogeneous XYZ residual.

    Preserve this value from the original surfaces when clipping a child.
    A child's small z coefficients do not bound cancellation inherited from
    forming and restricting the original surface pair.
    """
    first, second = np.asarray(first), np.asarray(second)
    return float(
        np.max(np.abs(first[..., :3])) * np.max(np.abs(second[..., 3]))
        + np.max(np.abs(second[..., :3])) * np.max(np.abs(first[..., 3]))
    )


def _zero_hull_interval(lower, upper, margin):
    """Axis interval where a scalar coefficient hull can meet zero."""
    lower = np.asarray(lower, dtype=float) - margin
    upper = np.asarray(upper, dtype=float) + margin
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        return 0.0, 1.0
    if float(lower.min()) > 0.0 or float(upper.max()) < 0.0:
        return None
    if len(lower) == 1:
        return 0.0, 1.0

    coordinate = np.linspace(0.0, 1.0, len(lower))
    zeros = coordinate[(lower <= 0.0) & (upper >= 0.0)]
    values = np.concatenate((lower, upper))
    coordinates = np.tile(coordinate, 2)
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    negative, positive = values < 0.0, values > 0.0
    bounds = []
    if len(zeros):
        bounds.extend((float(zeros.min()), float(zeros.max())))
    if np.any(negative) and np.any(positive):
        a = (-values[negative] / scale)[:, None]
        b = (values[positive] / scale)[None, :]
        intersections = (
            coordinates[negative, None] * b
            + coordinates[None, positive] * a
        ) / (a + b)
        bounds.extend((float(intersections.min()),
                       float(intersections.max())))
    if not bounds:
        return 0.0, 1.0
    # These are unit-coordinate convex interpolations. Expand their bounds
    # rather than letting arithmetic roundoff exclude an endpoint root.
    return (max(0.0, min(bounds) - 64.0 * _EPS),
            min(1.0, max(bounds) + 64.0 * _EPS))


def clip_residual_box(first, second, *, source_scale=None, max_passes=12,
                      resolution=None, charge=None):
    """Return a necessary local STUV box and bounded-work diagnostics.

    Inputs are homogeneous surface nets. For each residual component and
    axis, the graph (parameter, residual) lies in the convex hull of the
    coefficient points (k/degree, coefficient). Intersecting that hull with
    zero gives a necessary axis interval. Restriction propagates the new
    interval to the other axes before another pass.

    ``source_scale`` may carry ``residual_coordinate_scale`` from the original
    surfaces. The roundoff margin is scalar across XYZ and never decreases
    during restriction. ``max_passes`` and ``resolution`` only stop this
    optional contraction early; a coarse return is still a necessary box.

    ``charge(n)`` must approve setup before residual allocation and each
    axis operation before it executes. Denial returns the box accumulated
    so far. ``None`` denotes an empty residual hull, not a CAD absence
    decision. Callers using this for point/trace coverage should fall back
    on empty, invalid, or denied results.
    """
    stats = dict(passes=0, axes=0, work=0, empty=False, denied=False, valid=True)
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if (first.ndim != 3 or second.ndim != 3
            or first.shape[-1] != 4 or second.shape[-1] != 4
            or not all(first.shape) or not all(second.shape)
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        stats['valid'] = False
        return _UNIT_BOX, stats
    scale = residual_coordinate_scale(first, second)
    if source_scale is not None:
        if not np.isfinite(source_scale) or source_scale < 0.0:
            stats['valid'] = False
            return _UNIT_BOX, stats
        scale = max(scale, float(source_scale))
    if not np.isfinite(scale):
        stats['valid'] = False
        return _UNIT_BOX, stats
    if max_passes <= 0:
        return _UNIT_BOX, stats
    tolerance = (None if resolution is None else
                 np.broadcast_to(np.asarray(resolution, dtype=float), (4,)))
    if tolerance is not None and (not np.isfinite(tolerance).all()
                                  or np.any(tolerance < 0.0)):
        stats['valid'] = False
        return _UNIT_BOX, stats

    coefficient_count = (int(np.prod(first.shape[:2]))
                         * int(np.prod(second.shape[:2])) * 3)
    work_unit = max(1, (coefficient_count + 127) // 128)
    if charge is not None and not charge(work_unit):
        stats['denied'] = True
        return _UNIT_BOX, stats
    stats['work'] += work_unit
    net = psi_vector_net(first, second)
    if not np.isfinite(net).all():
        stats['valid'] = False
        return _UNIT_BOX, stats
    margin = _HULL_MARGIN_K_EPS * scale
    box = np.array(_UNIT_BOX)
    for _ in range(max_passes):
        previous = box.copy()
        stats['passes'] += 1
        for axis in range(4):
            if charge is not None and not charge(work_unit):
                stats['denied'] = True
                return tuple(map(tuple, box)), stats
            stats['axes'] += 1
            stats['work'] += work_unit
            coefficients = np.moveaxis(net, axis, 0).reshape(net.shape[axis], -1, 3)
            lower, upper = coefficients.min(axis=1), coefficients.max(axis=1)
            interval = [0.0, 1.0]
            for component in range(3):
                part = _zero_hull_interval(lower[:, component], upper[:, component], margin)
                if part is None:
                    stats['empty'] = True
                    return None, stats
                interval[0] = max(interval[0], part[0])
                interval[1] = min(interval[1], part[1])
            if interval[0] > interval[1]:
                stats['empty'] = True
                return None, stats

            low, high = box[axis]
            width = high - low
            if width <= 64.0 * _EPS * max(1.0, abs(low), abs(high)):
                continue
            new_low = max(low, float(np.nextafter(low + interval[0] * width, -np.inf)))
            new_high = min(high, float(np.nextafter(low + interval[1] * width, np.inf)))
            if new_low <= low and new_high >= high:
                continue
            # Preserve original cancellation error and accumulate the
            # additional error from restriction of the current coefficients.
            margin += (4.0 * _EPS * (net.shape[axis] - 1)
                       * float(np.max(np.abs(net))))
            net = restrict_net_axis_v(net, axis, new_low, new_high, low, high)
            box[axis] = new_low, new_high
        if tolerance is not None and np.all(box[:, 1] - box[:, 0] <= tolerance):
            break
        if np.max(np.abs(box - previous)) <= 64.0 * _EPS:
            break
    return tuple(map(tuple, box)), stats


def restrict_source_pair(first, second, local_box):
    """Restrict both homogeneous sources using a shared local STUV box."""
    if local_box is None:
        return None
    result = []
    for surface, offset in ((first, 0), (second, 2)):
        current = np.asarray(surface, dtype=float)
        for axis in range(2):
            low, high = local_box[offset + axis]
            current = restrict_net_axis_v(current, axis, low, high, 0.0, 1.0)
        result.append(current)
    return tuple(result)
