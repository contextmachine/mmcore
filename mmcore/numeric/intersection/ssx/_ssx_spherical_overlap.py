"""A bounded CAD reduction for two standard rational spherical octants.

The control template proposes an ideal chart only. A homogeneous Bernstein
bound compares every point of each actual patch to that chart. An analytic
hemisphere census supplies the complete footprint; an existing numerical
assembler supplies the actual paired rim geometry. Unsupported inputs stay
with ordinary SSX. No empty-intersection result is inferred here.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np

from mmcore.numeric._bezier_common import eval_surface, geometry_collapsed
from mmcore.numeric.bern import bernstein_product_conv, de_casteljau_split_nd
from mmcore.numeric.intersection.ssx._ssx5_overlap import (
    assemble_overlap_regions, _surface_chord_error,
)
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch

_PATTERN = np.array([
    [[1., 0., 0.], [1., 0., 1.], [0., 0., 1.]],
    [[1., 1., 0.], [1., 1., 1.], [0., 0., 1.]],
    [[0., 1., 0.], [0., 1., 1.], [0., 0., 1.]],
])
_WEIGHTS = np.outer([1., np.sqrt(.5), 1.], [1., np.sqrt(.5), 1.])
_DESIGN = np.column_stack((np.ones(9), _PATTERN.reshape(-1, 3)))
_PRODUCT = bernstein_product_conv(2)
_EPS = np.finfo(float).eps


@dataclass
class _Chart:
    center: np.ndarray
    radius: float
    frame: np.ndarray
    transpose: bool
    reverse_u: bool
    reverse_v: bool
    control_error: float

    def canonical_net(self, net):
        result = net.swapaxes(0, 1) if self.transpose else net
        if self.reverse_u:
            result = result[::-1]
        if self.reverse_v:
            result = result[:, ::-1]
        return result

    def original_net(self, net):
        result = net[::-1] if self.reverse_u else net
        if self.reverse_v:
            result = result[:, ::-1]
        return result.swapaxes(0, 1) if self.transpose else result

    def canonical_parameters(self, parameters):
        result = np.asarray(parameters, dtype=float).copy()
        if self.transpose:
            result = result[:, ::-1].copy()
        if self.reverse_u:
            result[:, 0] = 1.-result[:, 0]
        if self.reverse_v:
            result[:, 1] = 1.-result[:, 1]
        return result


def _propose_chart(surface, budget):
    best = None
    for transpose, reverse_u, reverse_v in product((False, True), repeat=3):
        if not budget.charge_cells(1, 'spherical_fit'):
            return None
        candidate = _Chart(None, 0., None, transpose, reverse_u, reverse_v, np.inf)
        net = candidate.canonical_net(surface)
        controls = (net[..., :3]/net[..., 3, None]).reshape(-1, 3)
        origin = controls.mean(axis=0)
        try:
            fitted = np.linalg.lstsq(_DESIGN, controls-origin, rcond=None)[0]
            left, values, right = np.linalg.svd(fitted[1:].T)
        except np.linalg.LinAlgError:
            continue
        radius = float(values.mean())
        if not np.isfinite(radius) or radius <= 0.:
            continue
        frame, center = left@right, origin+fitted[0]
        expected = center+radius*(_PATTERN@frame.T)
        error = float(np.linalg.norm(expected.reshape(-1, 3)-controls, axis=1).max())
        if best is None or error < best.control_error:
            candidate.center, candidate.radius, candidate.frame = center, radius, frame
            candidate.control_error = error
            best = candidate
    return best


def _ideal_net(center, radius, frame):
    points = center+radius*(_PATTERN@frame.T)
    return np.concatenate((points*_WEIGHTS[..., None], _WEIGHTS[..., None]), axis=2)


def _chart_error(actual, chart, center, radius, frame, budget):
    """Whole same-parameter rational difference, not a control-point fit."""
    if not budget.charge_cells(112, 'spherical_bound'):
        return np.inf, None
    first = chart.canonical_net(actual).copy()
    first /= first[..., 3].max()
    # Translation and radius scaling keep product cancellation independent
    # of the model's coordinate placement and of homogeneous gauge.
    original_size = max(float(np.max(abs(first[..., :3]))), float(np.max(abs(center))))
    first[..., :3] = (first[..., :3]-center*first[..., 3, None])/radius
    second = np.concatenate(((_PATTERN@frame.T)*_WEIGHTS[..., None],
                             _WEIGHTS[..., None]), axis=2)
    numerator = (
        np.einsum('aij,bkl,ikd,jl->abd', _PRODUCT, _PRODUCT, first[..., :3], second[..., 3])
        - np.einsum('aij,bkl,ikd,jl->abd', _PRODUCT, _PRODUCT, second[..., :3], first[..., 3]))
    denominator = np.einsum('aij,bkl,ik,jl->ab', _PRODUCT, _PRODUCT,
                            first[..., 3], second[..., 3])
    if (not np.isfinite(numerator).all() or not np.isfinite(denominator).all()
            or np.min(denominator) <= 0.):
        return np.inf, None
    operands = max(1., float(np.max(abs(first[..., :3]))),
                   float(np.max(abs(second[..., :3]))), original_size/radius)
    roundoff = 256.*_EPS*operands
    error = radius*float(np.max((np.linalg.norm(numerator, axis=2)+roundoff)
                                / denominator))
    ideal = chart.original_net(_ideal_net(center, radius, frame))
    return error, ideal


def _footprint(frames, radius, atol, budget):
    """All vertices and edges of a convex intersection of six hemispheres."""
    normals = np.concatenate([frame.T for frame in frames])
    vertices = []
    positional_error = 0.
    for i in range(6):
        for j in range(i+1, 6):
            if not budget.charge_cells(1, 'spherical_footprint'):
                return None
            axis = np.cross(normals[i], normals[j])
            length = float(np.linalg.norm(axis))
            if length <= 256.*_EPS:
                continue
            error = 512.*_EPS/length
            if radius*error > atol/16.:
                return None  # ill-conditioned footprint: ordinary SSX owns it
            positional_error = max(positional_error, error)
            for direction in (axis/length, -axis/length):
                if np.min(normals@direction) < -error:
                    continue
                if not any(np.linalg.norm(direction-old) <= 4.*error for old in vertices):
                    vertices.append(direction)
    if len(vertices) < 3:
        return None  # never claim absence or curve-only contact here
    vertices = np.asarray(vertices)
    inside = vertices.sum(axis=0)
    if np.linalg.norm(inside) <= 256.*_EPS:
        return None
    inside /= np.linalg.norm(inside)
    # Positive area must have an interior in all six inward halfspaces.
    if np.min(normals@inside) <= positional_error:
        return None
    east = vertices[0]-inside*np.dot(vertices[0], inside)
    if np.linalg.norm(east) <= 256.*_EPS:
        return None
    east /= np.linalg.norm(east)
    north = np.cross(inside, east)
    vertices = vertices[np.argsort(np.arctan2(vertices@north, vertices@east))]
    edges = []
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0)):
        cosine = float(np.clip(first@second, -1., 1.))
        angle = float(np.arctan2(np.linalg.norm(np.cross(first, second)), cosine))
        if angle <= 0. or angle >= np.pi or radius*angle <= atol/4.:
            return None
        owners = np.flatnonzero((abs(normals@first) <= 8.*positional_error)
                                & (abs(normals@second) <= 8.*positional_error))
        if not len(owners):
            return None
        tangent = second-cosine*first
        tangent /= np.linalg.norm(tangent)
        edges.append((first, tangent, angle, set(map(int, owners))))
    return edges


def _matches_edge(branch, reverse, edge, charts, ideals, center, radius, atol, budget):
    parameters, xyz = map(np.asarray, branch.curve)
    if reverse:
        parameters, xyz = parameters[::-1], xyz[::-1]
    if len(parameters) < 2 or parameters.shape != (len(xyz), 4):
        return set()
    first, tangent, angle, owners = edge
    matches = set()
    for owner, (chart, ideal) in enumerate(zip(charts, ideals)):
        uv = parameters[:, 2*owner:2*owner+2]
        canonical = chart.canonical_parameters(uv)
        for fixed, value, normal_index in ((0, 0., 1), (0, 1., 0), (1, 0., 2)):
            if 3*owner+normal_index not in owners:
                continue
            if np.any(abs(canonical[:, fixed]-value) > 128.*_EPS):
                continue
            difference = np.diff(canonical[:, 1-fixed])
            if not (np.all(difference >= -128.*_EPS)
                    or np.all(difference <= 128.*_EPS)):
                continue
            if not budget.charge_cells(len(parameters), 'spherical_footprint'):
                return set()
            directions = np.array([eval_surface(ideal, *q, rational=True)-center for q in uv])/radius
            angles = np.arctan2(directions@tangent, directions@first)
            sense = 1 if angles[-1] >= angles[0] else -1
            oriented = angles if sense == 1 else angles[::-1]
            if (np.any(np.diff(oriented) < -512.*_EPS)
                    or radius*max(abs(oriented[0]), abs(oriented[-1]-angle)) > atol/4.):
                continue
            valid = True
            for q0, q1, x0, x1 in zip(uv[:-1], uv[1:], xyz[:-1], xyz[1:]):
                if not budget.charge_cells(1, 'spherical_footprint'):
                    return set()
                # The ideal edge is a monotone circular arc. This bound
                # covers its ENTIRE parameter interval in both directions
                # against the corresponding XYZ chord, without grid sampling.
                if _surface_chord_error(ideal, q0, q1, np.array([x0, x1])) > .75*atol:
                    valid = False
                    break
            if valid:
                matches.add(sense)
    return matches


def _completed_rims_accounted_for(completed, assembled, sources, charts, ideals,
                                  atol, budget):
    """An unreferenced tolerance stub may only contract to a shared pole.

    Ordinary extra rims always decline the reduction. At a shared native
    collapsed edge, free longitude parameters denote the same pole; retain
    that identity rather than imposing an arbitrary pole parameter value.
    """
    branches = assembled['rim_branches']
    extra = [rim for rim in completed if not any(
        np.array_equal(rim['stuv'], branch.curve[0])
        and np.array_equal(rim['xyz'], branch.curve[1]) for branch in branches)]
    if not extra:
        return True
    poles = []
    for source, chart, ideal in zip(sources, charts, ideals):
        actual_edge = chart.canonical_net(source)[:, -1]
        if not geometry_collapsed(actual_edge[:, :3]/actual_edge[:, 3, None]):
            return False
        ideal_edge = chart.canonical_net(ideal)[:, -1]
        poles.append(ideal_edge[0, :3]/ideal_edge[0, 3])
    roundoff = 512.*_EPS*max(1., float(np.max(np.abs(poles))))
    if np.linalg.norm(poles[0]-poles[1]) > roundoff:
        return False
    endpoints = [np.asarray(branch.curve[1])[end] for branch in branches for end in (0, -1)]
    represented = [point for point in endpoints if np.linalg.norm(point-poles[0]) <= atol/4.]
    if not represented:
        return False
    target = min(represented, key=lambda point: np.linalg.norm(point-poles[0]))
    for rim in extra:
        if np.max(np.linalg.norm(rim['xyz']-target, axis=1))+roundoff > atol:
            return False
        for a, b in zip(rim['stuv'][:-1], rim['stuv'][1:]):
            if not budget.charge_cells(2, 'spherical_footprint'):
                return False
            for owner, source in enumerate(sources):
                if _surface_chord_error(source, a[2*owner:2*owner+2],
                        b[2*owner:2*owner+2], np.array([target, target])) > atol:
                    return False
    return True


def _validate_footprint(assembled, edges, charts, ideals, center, radius, atol, budget,
                        completed, sources):
    regions, branches = assembled['regions'], assembled['rim_branches']
    if not _completed_rims_accounted_for(completed, assembled, sources, charts,
                                         ideals, atol, budget):
        return False
    if (len(regions) != 1 or len(regions[0].boundary) != 1
            or not regions[0].certification.get('orientation_consistent', False)):
        return False
    if not budget.charge_cells(2, 'spherical_footprint'):
        return False
    witness = np.asarray(regions[0].interior_stuv)
    if witness.shape != (4,) or np.any(witness < 0.) or np.any(witness > 1.):
        return False
    for owner, ideal in enumerate(ideals):
        direction = (eval_surface(ideal, *witness[2*owner:2*owner+2], rational=True)-center)/radius
        if any(float(np.cross(first, tangent)@direction) <= 256.*_EPS
               for first, tangent, _, _ in edges):
            return False
    loop = regions[0].boundary[0]
    if len(loop) != len(edges) or len(branches) != len(edges):
        return False
    if len({index for index, _ in loop}) != len(branches):
        return False
    matches = {}
    for i, (index, reverse) in enumerate(loop):
        for j, edge in enumerate(edges):
            matches[i, j] = _matches_edge(branches[index], reverse, edge, charts,
                                          ideals, center, radius, atol, budget)
            if budget.exhausted:
                return False
    for sense in (-1, 1):
        for start in range(len(edges)):
            if all(sense in matches[i, (start+sense*i) % len(edges)] for i in range(len(loop))):
                return True
    return False


class _SphericalResult(dict):
    """Private postprocessing facts; these are not public result fields."""

    _c1_resolved = False
    _c3_resolved = False


def _resolve_spherical_c1(sources, charts, result, atol, budget):
    """Exclude rank defects away from represented or disjoint pole caps.

    The whole-map position bound alone does not imply derivative regularity.
    A signed Bernstein normal hull proves it on each truncated chart; the
    omitted cap must fit inside a CAD point already on the returned boundary,
    or be separated from the other actual surface by more than ``atol``.
    """
    from mmcore.numeric.intersection.ssx._ssx5_singular import sigma_normal_net

    if not result.get('overlap_regions'):
        return False
    corners = np.array([np.asarray(branch.curve[1])[end]
                        for branch in result['branches'] for end in (0, -1)])
    if not len(corners):
        return False
    cartesian_sources = [net[..., :3]/net[..., 3, None] for net in sources]
    for owner, (source, chart) in enumerate(zip(sources, charts)):
        # Charge before constructing the rational normal numerator. The
        # input is biquadratic, so this setup always produces an 8x8 net.
        if not budget.charge_cells(256, 'spherical_c1'):
            return False
        net = chart.canonical_net(source).copy()
        net /= net[..., 3].max()
        coordinate_scale = max(1., float(np.max(abs(net[..., :3]))),
                               float(np.max(abs(chart.center))))
        spatial_roundoff = 512.*_EPS*coordinate_scale/float(net[..., 3].min())
        pole = net[0, -1, :3]/net[0, -1, 3]
        net[..., :3] = (net[..., :3]-chart.center*net[..., 3, None])/chart.radius
        normal = sigma_normal_net(net, rational=True)
        direction = np.linalg.det(chart.frame)*chart.frame.sum(axis=1)
        scalar = normal@direction
        normal_roundoff = (4096.*_EPS*max(1., float(np.max(abs(normal))))
                           * max(1., coordinate_scale/chart.radius))
        normalized_pole = (pole-chart.center)/chart.radius
        # Dyadic restriction ends before a rounded cut could equal one.
        # Failure to confine a cap simply leaves ordinary C1 enabled.
        cap_radius = None
        for depth in range(1, 53):
            if not budget.charge_cells(1, 'spherical_c1'):
                return False
            cut = 1.-2.**-depth
            _, cap = de_casteljau_split_nd(net, 1, cut)
            cap_points = cap[..., :3]/cap[..., 3, None]
            radius = (chart.radius*float(np.max(np.linalg.norm(
                cap_points-normalized_pole, axis=2)))+spatial_roundoff)
            if radius <= atol/4.:
                cap_radius = radius
                regular, _ = de_casteljau_split_nd(scalar[..., None], 1, cut)
                if (not np.isfinite(regular).all()
                        or float(np.min(regular)) <= normal_roundoff):
                    return False
                break
        if cap_radius is None:
            return False
        if float(np.min(np.linalg.norm(corners-pole, axis=1)))+cap_radius <= atol:
            continue
        other_controls = cartesian_sources[1-owner]
        separation_roundoff = 512.*_EPS*max(
            coordinate_scale, float(np.max(abs(other_controls))))
        separated = False
        for direction in charts[1-owner].frame.T:
            # The fitted frame is orthogonal; normalizing keeps the
            # support distance in model units despite SVD roundoff.
            direction = direction/np.linalg.norm(direction)
            distance = float(np.min((other_controls-pole)@direction))
            if distance-separation_roundoff > cap_radius+atol:
                separated = True
                break
        if not separated:
            return False
    return True


def _publish(branches, regions, budget):
    result = _SphericalResult(branches=[], points=[], singularities=[], overlap_regions=[])
    budget.extend_output(result['branches'], branches, 'spherical_overlap')
    if len(result['branches']) == len(branches):
        budget.extend_output(result['overlap_regions'], regions, 'spherical_overlap')
    return result


def _partial(completed, budget):
    branches = [SSXBranch(curve=(rim['stuv'], rim['xyz']), kind='overlap', overlap=True)
                for rim in completed]
    return _publish(branches, [], budget)


def try_spherical_overlap(first, second, atol, budget):
    """Return an actual paired overlap region, or decline this reduction."""
    sources = tuple(np.asarray(net, dtype=float) for net in (first, second))
    if any(net.shape != (3, 3, 4) or not np.isfinite(net).all()
           or np.any(net[..., 3] <= 0.) for net in sources):
        return None
    charts = tuple(_propose_chart(net, budget) for net in sources)
    if any(chart is None for chart in charts) or budget.exhausted:
        return None
    # Rejection-only proposal screens keep ordinary biquadratic planes and
    # graphs out of the more expensive rational product checks. Passing a
    # screen never accepts a carrier: the whole-map bound is still required.
    if (any(chart.control_error > atol for chart in charts)
            or np.linalg.norm(charts[0].center-charts[1].center) > atol
            or abs(charts[0].radius-charts[1].radius) > atol):
        return None
    center = .5*(charts[0].center+charts[1].center)
    radius = .5*(charts[0].radius+charts[1].radius)
    if not np.isfinite(radius) or radius <= atol:
        return None
    # Coalescing an almost identical octant is only another proposal. Its
    # whole-patch bound must pass too; actual controls are never snapped.
    frame_pairs = [(charts[0].frame, charts[0].frame[:, permutation])
                   for permutation in permutations(range(3))]
    frame_pairs.append((charts[0].frame, charts[1].frame))
    first_validated = _chart_error(sources[0], charts[0], center, radius, charts[0].frame, budget)
    if budget.exhausted or first_validated[0] > atol/8.:
        return None
    candidates = []
    second_cartesian = charts[1].canonical_net(sources[1])
    second_cartesian = second_cartesian[..., :3]/second_cartesian[..., 3, None]
    for frames in frame_pairs:
        proposal = center+radius*(_PATTERN@frames[1].T)
        if np.max(np.linalg.norm(proposal-second_cartesian, axis=2)) > atol:
            continue
        validated = [first_validated,
                     _chart_error(sources[1], charts[1], center, radius, frames[1], budget)]
        if budget.exhausted:
            return None
        if validated[1][0] > atol/8.:
            continue
        edges = _footprint(frames, radius, atol, budget)
        if budget.exhausted:
            return None
        if edges is not None:
            # Whole-footprint stability, not just matching sampled corners.
            # An actual common point is at most e from each ideal octant,
            # hence every inward normal has n.d >= -e/(R-e). Moving its
            # direction toward an interior direction a by lambda restores
            # all six inequalities. Normalization adds at most lambda again.
            # Reject poorly conditioned narrow footprints rather than let a
            # small chart error conceal a long, separate boundary fringe.
            interior = np.sum([edge[0] for edge in edges], axis=0)
            interior /= np.linalg.norm(interior)
            normals = np.concatenate([frame.T for frame in frames])
            margin = float(np.min(normals@interior))-1024.*_EPS
            error = max(value for value, _ in validated)
            if margin <= 0. or error >= radius:
                continue
            displacement = error/((radius-error)*margin)
            if displacement >= 1. or error+2.*radius*displacement > atol/4.:
                continue
            candidates.append((edges, [ideal for _, ideal in validated]))
    if not candidates:
        return None
    from mmcore.nurbs._nurbs_param_tol import bez_surface_param_tolerance
    ptol = np.array([*bez_surface_param_tolerance(first, .5*atol, rational=True),
                     *bez_surface_param_tolerance(second, .5*atol, rational=True)])
    completed = []
    assembled = assemble_overlap_regions(
        *sources, atol=.5*atol, ptol4=ptol,
        charge=lambda amount: budget.charge_cells(amount, 'spherical_region'),
        _completed_rims=completed)
    if budget.exhausted:
        return _partial(completed, budget)
    if (not assembled['regions']
            or assembled['unmatched_branches']
            or assembled['unmatched_intersection_branches']):
        return None
    for edges, ideals in candidates:
        if _validate_footprint(assembled, edges, charts, ideals, center, radius, atol, budget,
                               completed, sources):
            result = _publish(assembled['rim_branches'], assembled['regions'], budget)
            if result['overlap_regions']:
                # A fully represented convex footprint has no 1D rim
                # self-crossings. Shared corners and pole UV aliases are
                # already part of its boundary, not C3 events.
                result._c3_resolved = True
                result._c1_resolved = _resolve_spherical_c1(
                    sources, charts, result, atol, budget)
            return result
        if budget.exhausted:
            return _partial(completed, budget)
    return None
