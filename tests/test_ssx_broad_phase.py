"""Analytic exclusion/retention contracts for the NURBS patch broad phase.

The oracle is known geometry, not agreement with another numerical SSX run.
All geometric distances use the requested model-space CAD tolerance.  These
checks deliberately retain uncertain convex-hull overlap: a broad phase may
prove separation, but cannot decide whether the surfaces intersect.
"""

import numpy as np
import pytest

from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
from mmcore.numeric.intersection.ssx import _ssx_broad_phase as broad_phase


ATOL = 1e-3


def _patch(points, weights=None):
    points = np.asarray(points, dtype=float)
    nu, nv = points.shape[:2]
    if weights is None:
        weights = np.ones((nu, nv))
    return NURBSSurfaceTuple(
        nu, nv,
        np.r_[np.zeros(nu), np.ones(nu)],
        np.r_[np.zeros(nv), np.ones(nv)],
        points, np.asarray(weights, dtype=float),
    )


def _plane(center=(0., 0., 0.), u=(1., 0., 0.), v=(0., 1., 0.)):
    center, u, v = map(lambda value: np.asarray(value, dtype=float), (center, u, v))
    return _patch([[center - u - v, center - u + v],
                   [center + u - v, center + u + v]])


def _slanted(gap=0.):
    tangent = np.array([1., 0., 1.]) / np.sqrt(2.)
    normal = np.array([-1., 0., 1.]) / np.sqrt(2.)
    return _plane(center=gap * normal, u=tangent)


def _kept(first, second, *, atol=ATOL, use_gjk=False):
    return broad_phase._filter_patch_pairs(
        [first], [second], [(0, 0)], atol, use_gjk=use_gjk,
    )


def _aabbs_overlap(first, second, atol=ATOL):
    a = first.control_points.reshape(-1, 3)
    b = second.control_points.reshape(-1, 3)
    return bool(np.all(a.max(0) + 2 * atol >= b.min(0))
                and np.all(b.max(0) + 2 * atol >= a.min(0)))


def _tangent_paraboloid():
    # z = x**2 + y**2 for x,y in [-1,1], in tensor Bernstein form.
    coordinates = [-1., 0., 1.]
    squares = [1., -1., 1.]
    return _patch([[[x, y, squares[i] + squares[j]]
                    for j, y in enumerate(coordinates)]
                   for i, x in enumerate(coordinates)])


def _contact_pair(kind):
    plane = _plane()
    if kind == 'crossing':
        return plane, _plane(u=(1., 0., 0.), v=(0., 0., 1.))
    if kind == 'tangent_point':
        return plane, _tangent_paraboloid()
    if kind == 'area_overlap':
        return plane, _plane(center=(.75, .25, 0.))
    if kind == 'shared_edge':
        return plane, _plane(center=(2., 0., 0.))
    if kind == 'shared_corner':
        return plane, _plane(center=(2., 2., 0.))
    if kind == 'collapsed_line':
        return plane, _plane(u=(1., 0., 0.), v=(0., 0., 0.))
    if kind == 'collapsed_point':
        return plane, _plane(u=(0., 0., 0.), v=(0., 0., 0.))
    if kind == 'coincident_points':
        point = _plane(u=(0., 0., 0.), v=(0., 0., 0.))
        return point, point
    raise AssertionError(kind)


def test_slanted_disjoint_patches_are_rejected_despite_overlapping_aabbs():
    first, second = _slanted(), _slanted(.25)
    assert _aabbs_overlap(first, second)
    assert _kept(first, second) == []


@pytest.mark.parametrize('kind', [
    'crossing', 'tangent_point', 'area_overlap', 'shared_edge',
    'shared_corner', 'collapsed_line', 'collapsed_point', 'coincident_points',
])
def test_actual_contacts_and_degenerate_patches_are_retained(kind):
    first, second = _contact_pair(kind)
    assert _aabbs_overlap(first, second)
    assert _kept(first, second) == [(0, 0)]


@pytest.mark.parametrize('gap_in_atol, retain', [
    (.5, True), (1.9, True), (2., True), (2.1, False), (4., False),
])
def test_normal_gap_uses_model_space_two_sided_margin(gap_in_atol, retain):
    first, second = _slanted(), _slanted(gap_in_atol * ATOL)
    assert _aabbs_overlap(first, second)
    assert _kept(first, second) == ([(0, 0)] if retain else [])


def test_nonuniform_rational_weights_preserve_a_known_tangent_contact():
    # The true rational curve has midpoint x=z=7/12. Its homogeneous XYZ
    # control hull has x+z <= 1, so treating weighted coordinates as Cartesian
    # would incorrectly discard its tangent plane x+z=7/6.
    points = np.array([[[1., -1., 0.], [1., 1., 0.]],
                       [[1., -1., 1.], [1., 1., 1.]],
                       [[0., -1., 1.], [0., 1., 1.]]])
    weights = np.array([[1., 1.], [.2, .2], [1., 1.]])
    rational = _patch(points, weights)
    midpoint = np.einsum('i,ij,ijd->d', [.25, .5, .25], weights, points)
    midpoint /= np.einsum('i,ij->', [.25, .5, .25], weights)
    witness = np.array([7. / 12., 0., 7. / 12.])
    assert np.linalg.norm(midpoint - witness) <= ATOL
    unweighted_midpoint = np.einsum('i,ijd->d', [.25, .5, .25], points) / 2.
    assert np.linalg.norm(unweighted_midpoint - witness) > 10 * ATOL
    tangent = _plane(center=witness, u=np.array([-1., 0., 1.]) / np.sqrt(2.))
    assert _kept(rational, tangent) == [(0, 0)]


def _map_patch(patch, *, rotation=None, translation=None, scale=1., chart=None):
    points = patch.control_points.copy()
    weights = patch.weights.copy()
    if chart == 'transpose':
        points, weights = points.swapaxes(0, 1), weights.T
    elif chart == 'reverse_u':
        points, weights = points[::-1], weights[::-1]
    elif chart == 'reverse_v':
        points, weights = points[:, ::-1], weights[:, ::-1]
    points *= scale
    if rotation is not None:
        points = points @ rotation.T
    if translation is not None:
        points += translation
    return _patch(points, weights)


def _rotation():
    # Rodrigues rotation about an oblique axis avoids axis-aligned fixtures.
    axis = np.array([1., 2., 3.]) / np.sqrt(14.)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = .83
    return np.eye(3) * np.cos(angle) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * skew


@pytest.mark.parametrize('variant', [
    'swap', 'transpose_first', 'transpose_second', 'reverse_u', 'reverse_v',
    'rotation', 'translation', 'small_units', 'large_units', 'combined',
])
def test_exclusion_and_contact_are_invariant_to_representation(variant):
    for gap, retain in ((.25, False), (1.9 * ATOL, True)):
        first, second = _slanted(), _slanted(gap)
        atol = ATOL
        if variant == 'swap':
            first, second = second, first
        elif variant.startswith('transpose_'):
            if variant.endswith('first'):
                first = _map_patch(first, chart='transpose')
            else:
                second = _map_patch(second, chart='transpose')
        elif variant in ('reverse_u', 'reverse_v'):
            first = _map_patch(first, chart=variant)
        else:
            options = {}
            if variant in ('rotation', 'combined'):
                options['rotation'] = _rotation()
            if variant in ('translation', 'combined'):
                options['translation'] = np.array([1.e6, -2.e6, 3.e6])
            if variant in ('small_units', 'large_units', 'combined'):
                scale = .01 if variant == 'small_units' else 1000.
                options['scale'] = scale
                atol *= scale
            first, second = (_map_patch(patch, **options) for patch in (first, second))
        assert _kept(first, second, atol=atol) == ([(0, 0)] if retain else [])


@pytest.mark.parametrize('proposal', [
    np.array([1., 0., 0.]), np.zeros(3), np.full(3, np.nan),
    np.array([np.inf, 0., 0.]), None,
], ids=['wrong_axis', 'zero_axis', 'nan_axis', 'infinite_axis', 'no_axis'])
def test_native_axis_is_only_a_proposal_and_cannot_discard_contact(monkeypatch, proposal):
    calls = []

    def propose(*args, **kwargs):
        calls.append(True)
        return proposal

    monkeypatch.setattr(broad_phase, '_gjk_axis', propose)
    for first, second in [_contact_pair('crossing'), (_slanted(), _slanted(1.9 * ATOL))]:
        assert _kept(first, second, use_gjk=True) == [(0, 0)]
    assert calls, 'the proposal checks must exercise the optional native path'


def test_candidate_order_is_preserved_when_middle_pair_is_excluded():
    patches = [_slanted(), _slanted(.25), _slanted(ATOL)]
    pairs = [(2, 0), (1, 0), (0, 0)]
    assert broad_phase._filter_patch_pairs(
        patches, [_slanted()], pairs, ATOL, use_gjk=False,
    ) == [(2, 0), (0, 0)]


def test_empty_candidate_list_remains_empty():
    assert broad_phase._filter_patch_pairs([], [], [], ATOL, use_gjk=False) == []


def test_projection_chunks_preserve_extrema_across_control_blocks(monkeypatch):
    monkeypatch.setattr(broad_phase, '_PROJECTION_ENTRIES', 15)
    for first, second, expected in (
            (_slanted(), _slanted(.25), []),
            (_slanted(), _slanted(1.9*ATOL), [(0, 0)]),
            (_plane(), _tangent_paraboloid(), [(0, 0)])):
        assert _kept(first, second) == expected


def test_large_control_product_skips_quadratic_native_search(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('large control products must not enter native GJK')

    monkeypatch.setattr(broad_phase, '_gjk_axis', forbidden)
    # Coincident constant patches must survive. Three hundred controls each
    # would require a 90,000-entry native cache merely to confirm contact.
    patch = _patch(np.zeros((300, 1, 3)))
    assert broad_phase._filter_patch_pairs(
        [patch], [patch], [(0, 0)], ATOL, refine=False) == [(0, 0)]


def test_nurbs_adapter_excludes_curved_empty_pair_before_bezier_solver(monkeypatch):
    from mmcore.numeric.intersection.ssx import _nssx5 as adapter

    def forbidden(*args, **kwargs):
        raise AssertionError('a separated patch pair reached the full SSX solver')

    monkeypatch.setattr(adapter, 'bez_ssx', forbidden)
    curved = _tangent_paraboloid()
    curved.control_points[..., 2] += 4*ATOL
    plane = _plane()
    # The graph z=x²+y²+4*atol is everywhere separated from z=0,
    # although their AABBs and unsplit control hulls overlap.
    assert _aabbs_overlap(curved, plane)
    result = adapter.nurbs_ssx(curved, plane, atol=ATOL)
    assert result['branches'] == result['points'] == result['singularities'] == []
    assert result['overlap_regions'] == []
    assert result['complete']


@pytest.mark.parametrize('failure', ['failed_fit', 'nonfinite_axes'])
def test_unusable_axis_fit_retains_uncertain_candidates(monkeypatch, failure):
    def unusable_fit(covariance):
        if failure == 'failed_fit':
            raise np.linalg.LinAlgError('synthetic axis-fit failure')
        return np.zeros(covariance.shape[:-1]), np.full(covariance.shape, np.nan)

    monkeypatch.setattr(broad_phase.np.linalg, 'eigh', unusable_fit)
    first, second = _slanted(), _slanted(.25)
    assert _aabbs_overlap(first, second)
    # Separation is real, but unusable directions do not certify it. Retain
    # the pair so the narrow phase remains responsible for resolving it.
    assert _kept(first, second, use_gjk=False) == [(0, 0)]
