"""Spherical CAD overlap checked against independent hemisphere geometry."""
from itertools import combinations

import numpy as np
import pytest

from examples.ssx.ssx5_analytic_audit import distances_to_polylines
from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple, evaluate_nurbs_surface
from mmcore.numeric.intersection.ssx._ssx5_overlap import assemble_overlap_regions


ATOL = 1e-3
_OCTANT_CONTROLS = np.array([
    [[1., 0., 0.], [1., 0., 1.], [0., 0., 1.]],
    [[1., 1., 0.], [1., 1., 1.], [0., 0., 1.]],
    [[0., 1., 0.], [0., 1., 1.], [0., 0., 1.]],
])


def _octant(frame, radius=1., center=(0., 0., 0.)):
    """Independent rational tensor product of two quarter-circle arcs."""
    control = np.asarray(center) + radius * (_OCTANT_CONTROLS @ np.asarray(frame).T)
    weights = np.outer([1., np.sqrt(.5), 1.], [1., np.sqrt(.5), 1.])
    knots = np.array([0., 0., 0., 1., 1., 1.])
    return NURBSSurfaceTuple(3, 3, knots.copy(), knots.copy(), control, weights)


def _rotation(axis, angle):
    direction = np.asarray(axis, dtype=float)
    direction /= np.linalg.norm(direction)
    x, y, z = direction
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return (np.cos(angle) * np.eye(3)
            + (1.-np.cos(angle)) * np.outer(direction, direction)
            + np.sin(angle) * cross)


def _reverse_u(surface):
    return NURBSSurfaceTuple(
        surface.order_u, surface.order_v, surface.knot_u.copy(), surface.knot_v.copy(),
        surface.control_points[::-1].copy(), surface.weights[::-1].copy())


def _homogeneous(surface):
    return np.concatenate((surface.control_points * surface.weights[..., None],
                           surface.weights[..., None]), axis=-1)


def _point(surface, uv):
    return np.asarray(evaluate_nurbs_surface(surface, *uv, d_order=0)['S'])


def _fit_octant_frame(surface):
    # Used only to recover the known construction from rounded user data.
    # No SSI algorithm, sampled intersection, or solver result enters this fit.
    design = np.column_stack((np.ones(9), _OCTANT_CONTROLS.reshape(-1, 3)))
    solution = np.linalg.lstsq(design, surface.control_points.reshape(-1, 3),
                              rcond=None)[0]
    center, scaled_frame = solution[0], solution[1:].T
    radius = np.linalg.norm(scaled_frame, axis=0).mean()
    left, _, right = np.linalg.svd(scaled_frame / radius)
    frame = left @ right
    fitted = center + radius * (_OCTANT_CONTROLS @ frame.T)
    assert np.linalg.norm(fitted-surface.control_points, axis=-1).max() < .01*ATOL
    # Confirm that the supplied rational weights describe this sphere too;
    # fitting only Euclidean controls would not establish the fixture oracle.
    for u in np.linspace(0., 1., 9):
        for v in np.linspace(0., 1., 9):
            point = _point(surface, (u, v))-center
            assert abs(np.linalg.norm(point)-radius) < .01*ATOL
            assert np.min(point @ frame) >= -.01*ATOL
    return center, radius, frame


def _polygon(frames):
    # Each octant is the intersection of three inward hemispheres. Polygon
    # vertices are admissible intersections of pairs of their great circles.
    normals = np.concatenate([frame.T for frame in frames])
    vertices = []
    for first, second in combinations(normals, 2):
        axis = np.cross(first, second)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            continue
        for point in (axis/norm, -axis/norm):
            if (np.min(normals @ point) >= -1e-10
                    and not any(np.linalg.norm(point-old) < 1e-9 for old in vertices)):
                vertices.append(point)
    vertices = np.asarray(vertices)
    assert len(vertices) >= 3, 'This reference requires positive overlap area'
    interior = vertices.sum(axis=0)
    interior /= np.linalg.norm(interior)
    east = vertices[0] - interior * np.dot(vertices[0], interior)
    east /= np.linalg.norm(east)
    north = np.cross(interior, east)
    vertices = vertices[np.argsort(np.arctan2(vertices @ north, vertices @ east))]
    return vertices, normals, interior


def _reference(center, radius, frames):
    vertices, normals, interior = _polygon(frames)
    arcs, angles, owners = [], [], []
    max_angle = 2*np.arccos(1.-ATOL/(8*radius))
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0)):
        angle = np.arctan2(np.linalg.norm(np.cross(first, second)), np.dot(first, second))
        # The independent reference's chord sagitta is at most atol/8.
        count = max(257, int(np.ceil(angle/max_angle))+1)
        fraction = np.linspace(0., 1., count)
        directions = (np.sin((1.-fraction)*angle)[:, None]*first
                      + np.sin(fraction*angle)[:, None]*second) / np.sin(angle)
        arcs.append(np.asarray(center) + radius*directions)
        angles.append(angle)
        active = np.nonzero((abs(normals @ first) < 1e-9)
                            & (abs(normals @ second) < 1e-9))[0]
        owners.append({int(index)//3 for index in active})
    return dict(center=np.asarray(center), radius=radius, vertices=vertices,
                normals=normals, interior=interior, arcs=arcs,
                owners=owners, length=radius*sum(angles))


def _distance_to_rim(points, reference):
    """Distance to finite great-circle arcs, not their sampled polylines."""
    relative = np.asarray(points)-reference['center']
    radius, vertices = reference['radius'], reference['vertices']
    distances = []
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0)):
        normal = np.cross(first, second)
        normal /= np.linalg.norm(normal)
        projected = relative - (relative @ normal)[:, None]*normal
        norms = np.linalg.norm(projected, axis=1)
        directions = projected / np.maximum(norms[:, None], np.finfo(float).tiny)
        within = ((np.cross(first, directions) @ normal >= -1e-12)
                  & (np.cross(directions, second) @ normal >= -1e-12)
                  & (norms > 0.))
        endpoint_distance = np.minimum(np.linalg.norm(relative-radius*first, axis=1),
                                       np.linalg.norm(relative-radius*second, axis=1))
        distances.append(np.where(within,
            np.linalg.norm(relative-radius*directions, axis=1), endpoint_distance))
    return np.min(distances, axis=0)


def _spherical_area(directions, interior):
    # Triangulation about an independent interior point, using solid angles.
    area = 0.
    for first, second in zip(directions, np.roll(directions, -1, axis=0)):
        numerator = np.dot(interior, np.cross(first, second))
        denominator = 1.+np.dot(interior, first)+np.dot(first, second)+np.dot(second, interior)
        area += 2*np.arctan2(numerator, denominator)
    return abs(area)


def _assert_region(result, surfaces, reference, normal_agreement=1):
    assert len(result['overlap_regions']) == 1
    region = result['overlap_regions'][0]
    assert len(region.boundary) == 1
    assert region.normal_agreement == normal_agreement
    assert {index for index, _ in region.boundary[0]} == set(range(len(result['branches'])))
    paths, owner_edges = [], set()
    for index, reverse in region.boundary[0]:
        branch = result['branches'][index]
        assert branch.kind == 'overlap'
        parameters, xyz = map(np.asarray, branch.curve)
        if reverse:
            parameters, xyz = parameters[::-1], xyz[::-1]
        assert len(xyz) >= 2 and parameters.shape == (len(xyz), 4)
        assert np.isfinite(parameters).all() and np.isfinite(xyz).all()
        assert np.all((-1e-12 <= parameters) & (parameters <= 1.+1e-12))
        # Corners and both source parameter curves must describe the same
        # model-space geometry, including between published vertices.
        for fraction in (0., .25, .5, .75, 1.):
            for q, point in zip((1.-fraction)*parameters[:-1]+fraction*parameters[1:],
                                (1.-fraction)*xyz[:-1]+fraction*xyz[1:]):
                a, b = _point(surfaces[0], q[:2]), _point(surfaces[1], q[2:])
                assert np.linalg.norm(a-point) <= ATOL
                assert np.linalg.norm(b-point) <= ATOL
                assert np.linalg.norm(a-b) <= ATOL
        assert _distance_to_rim(xyz, reference).max() <= ATOL
        assert _distance_to_rim(.5*(xyz[:-1]+xyz[1:]), reference).max() <= ATOL
        for owner, surface in enumerate(surfaces):
            uv = parameters[:, 2*owner:2*owner+2]
            for axis in (0, 1):
                for side in (0., 1.):
                    on_edge = uv.copy()
                    on_edge[:, axis] = side
                    if all(np.linalg.norm(_point(surface, edge)-point) <= ATOL
                           for edge, point in zip(on_edge, xyz)):
                        owner_edges.add(owner)
        paths.append(xyz)
    assert owner_edges == {0, 1}, 'The overlap rim must include both domain boundaries'
    for first, second in zip(paths, paths[1:]+paths[:1]):
        assert np.linalg.norm(first[-1]-second[0]) <= ATOL
    for arc in reference['arcs']:
        assert distances_to_polylines(arc, paths).max() <= ATOL
    length = sum(np.linalg.norm(np.diff(path, axis=0), axis=1).sum() for path in paths)
    assert abs(length-reference['length']) <= 8*np.pi*ATOL
    loop = np.concatenate(paths)
    directions = loop-reference['center']
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    expected_area = _spherical_area(reference['vertices'], reference['interior'])
    actual_area = _spherical_area(directions, reference['interior'])
    assert abs(actual_area-expected_area)*reference['radius']**2 <= 2*reference['length']*ATOL
    witness = np.asarray(region.interior_stuv)
    a, b = _point(surfaces[0], witness[:2]), _point(surfaces[1], witness[2:])
    assert np.linalg.norm(a-b) <= ATOL
    assert np.min(reference['normals'] @ (a-reference['center'])) > ATOL
    assert len(region.uv1_loops) == len(region.uv2_loops) == 1
    first_loop, second_loop = region.uv1_loops[0], region.uv2_loops[0]
    assert np.shape(first_loop) == np.shape(second_loop)
    for uv1, uv2 in zip(first_loop, second_loop):
        assert np.linalg.norm(_point(surfaces[0], uv1)-_point(surfaces[1], uv2)) <= ATOL
    for surface, uv in zip(surfaces, (first_loop, second_loop)):
        # At a collapsed pole, different u values denote the same point.
        assert np.linalg.norm(_point(surface, uv[0])-_point(surface, uv[-1])) <= ATOL


def _assemble(surfaces):
    from mmcore.nurbs._nurbs_param_tol import bez_surface_param_tolerance
    homogeneous = tuple(map(_homogeneous, surfaces))
    parameter_tolerance = np.concatenate([
        np.asarray(bez_surface_param_tolerance(net, ATOL, rational=True))
        for net in homogeneous])
    assembled = assemble_overlap_regions(*homogeneous, atol=ATOL,
                                         ptol4=parameter_tolerance)
    return {'overlap_regions': assembled['regions'], 'branches': assembled['rim_branches']}


@pytest.mark.parametrize('variant', ['identity', 'swap', 'reverse_u'])
def test_case17_public_overlap_has_the_whole_spherical_quadrilateral(variant):
    from examples.ssx.case_17 import s1, s2
    from mmcore.numeric.intersection.ssx import nurbs_ssx
    fits = [_fit_octant_frame(surface) for surface in (s1, s2)]
    assert np.linalg.norm(fits[0][0]-fits[1][0]) < .01*ATOL
    assert abs(fits[0][1]-fits[1][1]) < .01*ATOL
    center, radius = np.mean([f[0] for f in fits], axis=0), np.mean([f[1] for f in fits])
    reference = _reference(center, radius, [f[2] for f in fits])
    assert len(reference['vertices']) == 4  # property of the independent fixture
    assert [sum(owners == {owner} for owners in reference['owners'])
            for owner in (0, 1)] == [2, 2]
    # Neither collapsed pole is part of this particular common region.
    assert all(np.min(reference['normals'] @ fit[2][:, 2])*radius < -ATOL
               for fit in fits)
    surfaces = [s1, s2]
    if variant == 'swap':
        surfaces.reverse()
    elif variant == 'reverse_u':
        surfaces[1] = _reverse_u(surfaces[1])
    result = nurbs_ssx(*surfaces, atol=ATOL)
    assert result['complete'] is True, result['status']
    assert result['status']['reasons'] == []
    _assert_region(result, surfaces, reference,
                   normal_agreement=-1 if variant == 'reverse_u' else 1)
    assert result['points'] == [] and result['singularities'] == []


@pytest.mark.parametrize('variant', ['identity', 'swap', 'reverse_u'])
def test_rotated_octant_overlap_keeps_all_six_independent_rims(variant):
    frames = [np.eye(3), _rotation([1., 2., 3.], .35)]
    surfaces = [_octant(frame, center=(2., -3., .5)) for frame in frames]
    reference = _reference([2., -3., .5], 1., frames)
    assert len(reference['vertices']) == 6
    if variant == 'swap':
        surfaces.reverse()
    elif variant == 'reverse_u':
        surfaces[1] = _reverse_u(surfaces[1])
    _assert_region(_assemble(surfaces), surfaces, reference,
                   normal_agreement=-1 if variant == 'reverse_u' else 1)


def test_narrow_spherical_overlap_keeps_distinct_nearby_corners_and_shared_pole():
    frames = [np.eye(3), _rotation([0., 0., 1.], np.pi/2-.02)]
    surfaces = [_octant(frame) for frame in frames]
    reference = _reference([0., 0., 0.], 1., frames)
    # The equatorial corners are 20*atol apart, even though both arcs meet
    # at a common collapsed pole. An enlarged join radius must not merge them.
    assert min(np.linalg.norm(a-b) for a, b in combinations(reference['vertices'], 2)) > 19*ATOL
    _assert_region(_assemble(surfaces), surfaces, reference)


@pytest.mark.parametrize('kind', ['radius_gap', 'shared_meridian', 'shared_pole'])
def test_lower_dimensional_or_separated_spherical_contacts_do_not_make_a_region(kind):
    first = _octant(np.eye(3))
    if kind == 'radius_gap':
        second = _octant(np.eye(3), radius=1.+4*ATOL)
    else:
        angle = np.pi/2 + (.02 if kind == 'shared_pole' else 0.)
        second = _octant(_rotation([0., 0., 1.], angle))
    assert _assemble((first, second))['overlap_regions'] == []
