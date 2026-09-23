"""Paired geometric coverage by tangential paths already found by SSX.

These optional shortcuts avoid rediscovering an existing path. A point
requires the same segment location in parameters and XYZ. A cell requires
both restricted source control graphs to fit one existing paired segment;
an empty residual may also resolve a remainder without registrations.
Clipping only narrows a copy, and never changes queue ownership.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mmcore.numeric.intersection.ssx._ssx_bernstein_clip import (
    clip_residual_box,
    restrict_source_pair,
)
from mmcore.numeric.intersection.ssx._ssx_polyline import point_matches_polyline

_EPS = np.finfo(float).eps


def _segment_graph_bounds(parameters, xyz, q0, q1, x0, x1,
                          parameter_roundoff=0.):
    direction = x1 - x0
    denominator = float(direction @ direction)
    direction_error = np.where(x1 == x0, 0., _EPS * (abs(x0) + abs(x1)))
    den_error = (8 * _EPS * denominator
                 + 2 * np.sum(abs(direction) * direction_error)
                 + float(direction_error @ direction_error))
    if not np.isfinite(denominator) or denominator <= den_error:
        return None
    delta = xyz - x0
    delta_error = np.where(xyz == x0, 0., _EPS * (abs(xyz) + abs(x0)))
    first_terms = delta * direction
    first_error = (8 * _EPS * np.sum(abs(first_terms), axis=1)
                   + np.sum(delta_error * abs(direction)
                            + abs(delta) * direction_error, axis=1))
    first = np.sum(first_terms, axis=1)
    alpha = first / denominator
    alpha_error = (first_error + abs(alpha) * den_error) / (denominator - den_error)
    # Distance to a closed segment is convex. Distances at every source
    # control therefore bound its whole graph, including a CAD-sized
    # extension into an endpoint cap.
    clamped = np.clip(alpha, 0., 1.)
    perpendicular = delta - clamped[:, None] * direction
    perp_error = (delta_error + abs(clamped[:, None]) * direction_error
                  + alpha_error[:, None] * (abs(direction) + direction_error)
                  + 8 * _EPS * (abs(delta) + abs(clamped[:, None] * direction)))
    xyz_bound = float(np.max(np.linalg.norm(perpendicular, axis=1)
                             + np.linalg.norm(perp_error, axis=1)))
    displacement = q1 - q0
    param_residual = parameters - (q0 + alpha[:, None] * displacement)
    param_error = (parameter_roundoff + alpha_error[:, None] * abs(displacement)
                   + 16 * _EPS * (abs(parameters) + abs(q0)
                                  + abs(alpha[:, None] * displacement)))
    # Parameter residuals before clamping are affine Bernstein graphs.
    # Add an upper bound for the endpoint clamping correction rather than
    # treating clamped control parameters as a new affine mapping.
    excess = max(0., -float(np.min(alpha - alpha_error)),
                 float(np.max(alpha + alpha_error)) - 1.)
    return (xyz_bound, np.max(abs(param_residual) + param_error, axis=0)
            + excess * abs(displacement))


def _polynomial_surface(surface):
    net = np.asarray(surface)
    return bool(net.ndim == 3 and net.shape[-1] == 4
                and min(net.shape[:2]) >= 2 and np.isfinite(net).all()
                and net[0, 0, 3] != 0.
                and np.all(net[..., 3] == net[0, 0, 3]))


def _control_graph(surface, box):
    xyz = (surface[..., :3] / surface[..., 3:]).reshape(-1, 3)
    u = np.linspace(*box[0], surface.shape[0])
    v = np.linspace(*box[1], surface.shape[1])
    parameters = np.stack(np.meshgrid(u, v, indexing='ij'), axis=-1).reshape(-1, 2)
    error = 16 * _EPS * np.array([abs(a) + abs(b) for a, b in box])
    return parameters, xyz, error


@dataclass
class _Trace:
    parameters: np.ndarray
    xyz: np.ndarray
    low: np.ndarray
    high: np.ndarray
    candidates: list


class TangentialTraceCoverage:
    """Cache only published tangent paths, sharing the caller's work cap."""

    def __init__(self, atol, parameter_tolerance, source_scale,
                 charge: Callable[[int], bool]):
        self.atol = float(atol)
        self.ptol = np.asarray(parameter_tolerance, dtype=float)
        self.source_scale = float(source_scale)
        self.charge = charge
        self.traces: list[_Trace] = []
        self._seen_fragments = 0

    def update(self, fragments):
        """Account for new immutable paths without rescanning old samples."""
        fresh = fragments[self._seen_fragments:]
        self._seen_fragments = len(fragments)
        for fragment in fresh:
            if not fragment.tangential:
                continue
            parameters = np.asarray(fragment.stuv_path, dtype=float)
            xyz = np.asarray(fragment.xyz_path, dtype=float)
            if (parameters.shape != (len(xyz), 4) or len(xyz) < 2
                    or not np.isfinite(parameters).all() or not np.isfinite(xyz).all()):
                continue
            if not self.charge(max(1, (len(xyz) * 7 + 127) // 128)):
                return
            candidates = []
            bounds = _segment_graph_bounds(parameters, xyz, parameters[0],
                                            parameters[-1], xyz[0], xyz[-1])
            if (bounds is not None and bounds[0] < self.atol
                    and np.all(bounds[1] < self.ptol)):
                # Charge the approximation of a full straight span once;
                # it cannot widen the accepted tube by another tolerance.
                candidates.append((0, len(xyz) - 1, self.atol - bounds[0],
                                   self.ptol - bounds[1]))
            candidates.extend((i, i + 1, self.atol, self.ptol)
                              for i in range(len(xyz) - 1))
            self.traces.append(_Trace(
                parameters, xyz,
                parameters.min(axis=0), parameters.max(axis=0), candidates))

    def covers_points(self, parameters, xyz):
        parameters, xyz = np.asarray(parameters), np.asarray(xyz)
        if (len(parameters) == 0 or parameters.shape != (len(xyz), 4)
                or not np.isfinite(parameters).all() or not np.isfinite(xyz).all()):
            return False
        for point4, point3 in zip(parameters, xyz):
            covered = False
            for trace in self.traces:
                if np.any(point4 < trace.low - self.ptol) or np.any(point4 > trace.high + self.ptol):
                    continue
                if not self.charge(max(1, (len(trace.xyz) * 7 + 127) // 128)):
                    return False
                # A parameter least-squares projection alone can miss a
                # different segment fraction satisfying both bounds.
                if point_matches_polyline(point4, point3, trace.parameters,
                                          trace.xyz, self.ptol, self.atol):
                    covered = True
                    break
            if not covered:
                return False
        return True

    def covers_cell(self, first, second, box, *, allow_empty=False):
        """Resolve an existing trace's remainder without altering its cell.

        ``allow_empty`` is restricted to remainder cells without registered
        contacts. A source-scaled empty residual hull then excludes another
        equation branch; it never changes public CAD contact membership or
        contradicts a registration the caller already validated.
        """
        if not self.traces or not (_polynomial_surface(first) and _polynomial_surface(second)):
            return False
        bounds = np.asarray(box)
        possible = [trace for trace in self.traces
                    if np.all(bounds[:, 1] >= trace.low - self.ptol)
                    and np.all(bounds[:, 0] <= trace.high + self.ptol)]
        if not possible:
            return False
        necessary, stats = clip_residual_box(first, second, source_scale=self.source_scale,
                                             charge=self.charge)
        if (allow_empty and necessary is None and stats['empty']
                and stats['valid'] and not stats['denied']):
            return True
        if necessary is None or not stats['valid'] or stats['denied']:
            return False
        if not self.charge(max(1, (2 * (first.size + second.size) + 127) // 128)):
            return False
        restricted = restrict_source_pair(first, second, necessary)
        global_box = tuple((lo + a * (hi - lo), lo + b * (hi - lo))
                           for (a, b), (lo, hi) in zip(necessary, box))
        graphs = [_control_graph(restricted[0], global_box[:2]),
                  _control_graph(restricted[1], global_box[2:])]
        cost = max(1, (7 * sum(len(g[0]) for g in graphs) + 127) // 128)
        for trace in possible:
            for left, right, allowed_xyz, allowed_parameters in trace.candidates:
                valid = True
                for index, (parameters, xyz, error) in enumerate(graphs):
                    axes = slice(2 * index, 2 * index + 2)
                    q0, q1 = trace.parameters[left, axes], trace.parameters[right, axes]
                    if (np.any(parameters < np.minimum(q0, q1) - allowed_parameters[axes])
                            or np.any(parameters > np.maximum(q0, q1) + allowed_parameters[axes])):
                        valid = False
                        break
                    if not self.charge(cost):
                        return False
                    measured = _segment_graph_bounds(parameters, xyz, q0, q1,
                                                       trace.xyz[left], trace.xyz[right], error)
                    if (measured is None or measured[0] > allowed_xyz
                            or np.any(measured[1] > allowed_parameters[axes])):
                        valid = False
                        break
                if valid:
                    return True
        return False
