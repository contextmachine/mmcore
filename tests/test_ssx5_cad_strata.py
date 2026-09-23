"""Geometry regressions retained from the withdrawn exact-helper tests.

These call the public CAD solver at a modeling tolerance. Full line, arm,
region, and singular-curve coverage remains required; exact arithmetic
metadata and mandatory exact-mode completeness are not part of this API.
"""
import numpy as np
import pytest

from mmcore.numeric._bezier_common import eval_surface
from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx


def _plane(x=(0., 1.), y=(0., 1.)):
    return np.array([[[s, t, 0.] for t in y] for s in x])


def _homogeneous(net, weight):
    return np.concatenate((net*weight, np.full(net.shape[:2]+(1,), weight)), axis=2)


def _distance(point, path):
    a, d = path[:-1], np.diff(path, axis=0)
    square = np.einsum('ij,ij->i', d, d)
    fraction = np.clip(np.einsum('ij,ij->i', point-a, d)/np.maximum(square, 1e-300), 0., 1.)
    return np.linalg.norm(a+fraction[:, None]*d-point, axis=1).min()


def _cover(result, reference, tolerance):
    paths = [np.asarray(branch.curve[1]) for branch in result['branches']]
    assert paths, result['status']
    for point in reference:
        assert min(_distance(point, path) for path in paths) <= tolerance, point


def _line(first, last):
    return np.linspace(first, last, 101)


def _valid_paths(result, pair, atol, rational=False):
    for branch in result['branches']:
        stuv, xyz = map(np.asarray, branch.curve)
        assert len(stuv) == len(xyz) and np.isfinite(stuv).all() and np.isfinite(xyz).all()
        for index in range(len(stuv)-1):
            for fraction in np.linspace(0., 1., 9):
                q = (1-fraction)*stuv[index]+fraction*stuv[index+1]
                x = (1-fraction)*xyz[index]+fraction*xyz[index+1]
                for surface, uv in zip(pair, (q[:2], q[2:])):
                    assert np.linalg.norm(eval_surface(surface, *uv, rational=rational)-x) <= atol


@pytest.mark.parametrize('degree', [2, 4, 8, 10, 12])
@pytest.mark.parametrize('swap', [False, True])
def test_high_order_ruling_covers_the_full_tangent_line(degree, swap):
    a = np.array([[[s, j/degree, (-1.)**(degree-j)*2.**-degree]
                   for j in range(degree+1)] for s in (0., 1.)])
    pair = [_plane(), a] if swap else [a, _plane()]
    # Keep the whole-patch coincidence regime in the dedicated original C1
    # tests. Here the height excursion exceeds the requested tolerance.
    atol = min(1e-3, .1*2.**-degree)
    result = bez_ssx(*pair, atol=atol, rational=False)
    assert len(result['branches']) == 1
    assert result['branches'][0].kind == 'tangential'
    xyz = np.asarray(result['branches'][0].curve[1])
    assert np.max(np.abs(xyz[:, 1]-.5)) <= 2*atol
    _cover(result, _line([0., .5, 0.], [1., .5, 0.]), 2*atol)


def test_nonbinary_tangent_ruling_is_clipped_to_the_full_target_interval():
    a = np.array([[[s, y, z] for y, z in zip((0., .5, 1.), (1., -2., 4.))]
                  for s in (-1., 2.)])
    result = bez_ssx(a, _plane(), atol=1e-3, rational=False)
    assert len(result['branches']) == 1
    _cover(result, _line([0., 1/3, 0.], [1., 1/3, 0.]), 2e-3)
    xyz = np.asarray(result['branches'][0].curve[1])
    assert np.max(np.abs(xyz[:, 1]-1/3)) <= 2e-3


@pytest.mark.parametrize('same_image', [False, True])
def test_two_rulings_preserve_their_separated_parameter_preimages(same_image):
    a = np.array([[[s, .5 if same_image else y, z]
                   for y, z in zip((0., .5, 1.), (3/16, -5/16, 3/16))]
                  for s in (0., 1.)])
    result = bez_ssx(a, _plane(), atol=1e-3, rational=False)
    assert len(result['branches']) == 2
    roots = sorted(float(np.asarray(b.curve[0])[:, 1].mean()) for b in result['branches'])
    np.testing.assert_allclose(roots, [.25, .75], atol=1e-3, rtol=0.)
    for y in ((.5,) if same_image else (.25, .75)):
        _cover(result, _line([0., y, 0.], [1., y, 0.]), 2e-3)


@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('rational', [False, True])
@pytest.mark.parametrize('transpose', [False, True])
def test_cusp_ruling_keeps_surface_owner_and_full_curve(swap, rational, transpose):
    a = np.array([[[x, y, float(t)] for t in (0, 1)]
                  for x, y in zip((3., -1., -1., 3.), (-1., 1., -1., 1.))])
    b = np.array([[[0., y, z] for z in (-.5, 1.5)] for y in (-1.5, 1.5)])
    if transpose:
        a = a.transpose(1, 0, 2)
    pair = [a, b]
    if rational:
        pair = [_homogeneous(net, w) for net, w in zip(pair, (2., .5))]
    if swap:
        pair.reverse()
    result = bez_ssx(*pair, atol=1e-3, rational=rational)
    curves = [g for g in result['singularities'] if g.kind == 'cusp_curve']
    assert curves, result['status']
    assert all(g.surface == (2 if swap else 1) for g in curves)
    samples = np.concatenate([np.asarray(g.samples) for g in curves])
    owner = 2 if swap else 0
    fixed, free = (owner+1, owner) if transpose else (owner, owner+1)
    assert np.max(np.abs(samples[:, fixed]-.5)) <= 1e-3
    assert samples[:, free].min() <= 1e-3 and samples[:, free].max() >= 1.-1e-3


def _corner_graph(degree=1):
    return np.array([[[s, t, (1-s)*(1-t)]
                      for t in np.linspace(0., 1., degree+1)]
                     for s in np.linspace(0., 1., degree+1)])


@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('degree', [1, 4])
def test_two_boundary_edges_have_full_coverage(degree, swap):
    a, b = _corner_graph(degree), _plane()
    result = bez_ssx(*((b, a) if swap else (a, b)), atol=1e-3, rational=False)
    assert len(result['branches']) == 2
    assert all(branch.kind == 'overlap' for branch in result['branches'])
    _cover(result, _line([1., 0., 0.], [1., 1., 0.]), 1e-3)
    _cover(result, _line([0., 1., 0.], [1., 1., 0.]), 1e-3)


@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('rational', [False, True])
def test_boundary_edges_keep_their_tangent_junction_on_a_convex_target(swap, rational):
    source = _corner_graph()
    source[..., :2] = .25+.25*source[..., :2]
    target = np.array([[[u+.25*u*v, v+.125*u*v, 0.]
                        for v in (0., 1.)] for u in (0., 1.)])
    pair = [source, target]
    if rational:
        pair = [_homogeneous(net, w) for net, w in zip(pair, (2., .5))]
    if swap:
        pair.reverse()
    result = bez_ssx(*pair, atol=1e-3, rational=rational)
    assert len(result['branches']) == 2
    touches = [g for g in result['singularities'] if g.kind == 'tangent_point']
    assert len(touches) == 1
    np.testing.assert_allclose(touches[0].xyz, [.5, .5, 0.], atol=1e-3, rtol=0.)
    _cover(result, _line([.5, .25, 0.], [.5, .5, 0.]), 1e-3)
    _cover(result, _line([.25, .5, 0.], [.5, .5, 0.]), 1e-3)


def test_clipped_boundary_edge_spans_the_full_target():
    a = _corner_graph()
    a[..., :2] = 2*a[..., :2]-.5
    b = _plane(y=(0., 2.))
    result = bez_ssx(a, b, atol=1e-3, rational=False)
    assert len(result['branches']) == 1
    _cover(result, _line([0., 1.5, 0.], [1., 1.5, 0.]), 1e-3)


def test_curved_inverse_on_a_convex_target_has_valid_whole_lifted_chords():
    a = np.array([[[.25+.5*s, .5+.5*t, t] for t in (0., 1.)] for s in (0., 1.)])
    b = np.array([[[u, (1+u)*v, 0.] for v in (0., 1.)] for u in (0., 1.)])
    result = bez_ssx(a, b, atol=1e-3, rational=False)
    assert len(result['branches']) == 1
    _cover(result, _line([.25, .5, 0.], [.75, .5, 0.]), 1e-3)
    _valid_paths(result, (a, b), 1e-3)


@pytest.mark.parametrize('flip,transpose', [(False, False), (True, False), (False, True), (True, True)])
def test_planar_overlap_preserves_paired_chart_orientation(flip, transpose):
    a = _plane()
    b = a[::-1].copy() if flip else a.copy()
    if transpose:
        b = b.swapaxes(0, 1).copy()
    result = bez_ssx(a, b, atol=1e-3, rational=False)
    assert len(result['overlap_regions']) == 1 and len(result['branches']) == 4
    region = result['overlap_regions'][0]
    assert region.normal_agreement == (-1 if flip != transpose else 1)
    assert len(region.boundary[0]) == 4
    _valid_paths(result, (a, b), 1e-3)
    for first, last in (([0., 0., 0.], [1., 0., 0.]), ([0., 1., 0.], [1., 1., 0.]),
                        ([0., 0., 0.], [0., 1., 0.]), ([1., 0., 0.], [1., 1., 0.])):
        _cover(result, _line(first, last), 1e-3)


def test_rotated_planar_overlap_preserves_all_eight_rims():
    a = _plane((-1., 1.), (-1., 1.))
    b = np.array([[[-1.5, 0., 0.], [0., 1.5, 0.]], [[0., -1.5, 0.], [1.5, 0., 0.]]])
    result = bez_ssx(a, b, atol=1e-3, rational=False)
    assert len(result['overlap_regions']) == 1 and len(result['branches']) == 8
    corners = np.array([[-1., -.5, 0.], [-.5, -1., 0.], [.5, -1., 0.], [1., -.5, 0.],
                        [1., .5, 0.], [.5, 1., 0.], [-.5, 1., 0.], [-1., .5, 0.]])
    for first, last in zip(corners, np.roll(corners, -1, axis=0)):
        _cover(result, _line(first, last), 1e-3)
    _valid_paths(result, (a, b), 1e-3)


@pytest.mark.parametrize('root,swap', [(.5, False), (0., False), (1., True), (.5, True)])
def test_quadratic_touch_is_published_at_modeling_tolerance(root, swap):
    height = [root*root, root*root-root, (1-root)**2]
    a = np.array([[[s, t, height[i]+z] for t, z in zip((0., .5, 1.), (.25, -.25, .25))]
                  for i, s in enumerate((0., .5, 1.))])
    result = bez_ssx(*((_plane(), a) if swap else (a, _plane())), atol=1e-3, rational=False)
    touches = [g for g in result['singularities'] if g.kind == 'tangent_point']
    assert len(touches) == 1
    np.testing.assert_allclose(touches[0].xyz, [root, .5, 0.], atol=1e-3, rtol=0.)
    assert result['branches'] == []


@pytest.mark.parametrize('swap,flip,transpose', [(False, False, False), (True, False, False),
                                                (False, True, False), (False, False, True)])
def test_curved_tangential_line_keeps_its_complete_extent(swap, flip, transpose):
    a = np.zeros((3, 4, 3))
    a[..., 0] = np.array([0., 1., 1., 0.])[None, :]+np.array([0., .5, 1.])[:, None]
    a[..., 1] = np.array([0., 1., 2., 3.])[None, :]
    a[..., 2] = np.array([.25, -.25, .25])[:, None]
    b = a.copy()
    b[..., 2] *= -1
    if flip:
        b = b[::-1].copy()
    if transpose:
        b = b.swapaxes(0, 1).copy()
    pair = [b, a] if swap else [a, b]
    result = bez_ssx(*pair, atol=1e-3, rational=False)
    assert result['branches']
    assert all(branch.kind == 'tangential' for branch in result['branches'])
    t = np.linspace(0., 1., 301)
    _cover(result, np.column_stack((.5+3*t*(1-t), 3*t, np.zeros(len(t)))), 2e-3)
    for branch in result['branches']:
        xyz = np.asarray(branch.curve[1])
        t = xyz[:, 1]/3
        np.testing.assert_allclose(xyz[:, 0], .5+3*t*(1-t), atol=2e-3, rtol=0.)
        assert np.max(.75*np.diff(t)**2) <= 2e-3


@pytest.mark.parametrize('swap,flip,transpose', [(False, False, False), (True, False, False),
                                                (False, True, False), (False, False, True)])
def test_saddle_cross_keeps_all_four_arms_and_the_junction(swap, flip, transpose):
    a = np.array([[[s, t, (s-.5)*(t-.5)] for t in (0., 1.)] for s in (0., 1.)])
    b = _plane()
    if flip:
        a = a[::-1].copy()
    if transpose:
        b = b.swapaxes(0, 1).copy()
    result = bez_ssx(*((b, a) if swap else (a, b)), atol=1e-3, rational=False)
    touches = [g for g in result['singularities'] if g.kind == 'tangent_point']
    assert len(touches) == 1
    np.testing.assert_allclose(touches[0].xyz, [.5, .5, 0.], atol=1e-3, rtol=0.)
    _cover(result, _line([0., .5, 0.], [1., .5, 0.]), 2e-3)
    _cover(result, _line([.5, 0., 0.], [.5, 1., 0.]), 2e-3)
    travel = sum(np.linalg.norm(np.diff(branch.curve[1], axis=0), axis=1).sum()
                 for branch in result['branches'])
    assert abs(travel-2.) <= .01
