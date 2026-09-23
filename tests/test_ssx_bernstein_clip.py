"""Necessary residual boxes preserve known geometry at CAD resolution."""
import numpy as np
import pytest

from examples.ssx.ssx5_analytic_audit import case_components, graph_pair
from mmcore.numeric.intersection.ssx import _ssx_bernstein_clip as clip


def _homogeneous(surface):
    return np.concatenate((surface, np.ones(surface.shape[:2] + (1,))), axis=-1)


@pytest.mark.parametrize('swap', [False, True])
def test_closed_circle_samples_remain_in_necessary_box(swap):
    first, second = map(_homogeneous, graph_pair(case_components('one_circle')))
    angle = np.linspace(0., 2*np.pi, 2049)
    st = np.column_stack((.5 + .25*np.cos(angle), .5 + .25*np.sin(angle)))
    uv = (st + .25)/1.5
    roots = np.column_stack((st, uv))
    if swap:
        first, second = second, first
        roots = roots[:, [2, 3, 0, 1]]
    box, stats = clip.clip_residual_box(first, second)
    assert box is not None and stats['valid'] and not stats['denied']
    bounds = np.asarray(box)
    # This is a containment check, not a demand for precise returned digits.
    assert np.all(roots >= bounds[:, 0])
    assert np.all(roots <= bounds[:, 1])


def test_tangent_corner_contracts_both_source_hulls_to_cad_point():
    graph, plane = map(_homogeneous, graph_pair(case_components('two_circles')))
    scale = clip.residual_coordinate_scale(plane, graph)
    # The first circle's rightmost point is (.4375,.5). This child lies
    # outside that circle and away from the other circle; only the corner
    # contact remains. Plane and graph use different parameter charts.
    owner = ((11/24, .48), (.48, .5), (.4375, .46), (.47, .5))
    first, second = clip.restrict_source_pair(plane, graph, owner)
    box, stats = clip.clip_residual_box(first, second, source_scale=scale)
    assert box is not None and stats['valid'] and not stats['denied']
    restricted = clip.restrict_source_pair(first, second, box)
    point = np.array([.4375, .5, 0.])
    for source in restricted:
        xyz = source[..., :3]/source[..., 3:]
        assert np.linalg.norm(xyz-point, axis=-1).max() <= 1e-3


def test_margin_uses_xyz_and_preserves_original_operand_scale():
    plane = np.array([[[s, t, 0., 1.] for t in (0., 1.)] for s in (0., 1.)])
    nearby = plane.copy()
    nearby[..., 2] = 1e-15
    # A z-only error scale would incorrectly exclude a cancellation-sized
    # residual despite order-one coordinates in the supplied sources.
    box, stats = clip.clip_residual_box(plane, nearby)
    assert box is not None and not stats['empty']
    tiny_plane, tiny_nearby = plane.copy(), nearby.copy()
    tiny_plane[..., :3] *= 1e-6
    tiny_nearby[..., :3] *= 1e-6
    tiny_nearby[..., 2] = 1e-15
    box, stats = clip.clip_residual_box(tiny_plane, tiny_nearby, source_scale=2.)
    assert box is not None and not stats['empty']


def test_disjoint_residual_hull_returns_empty_without_claiming_a_root():
    first, second = map(_homogeneous, graph_pair(case_components('one_circle')))
    second[..., 2] = 2.
    box, stats = clip.clip_residual_box(first, second)
    assert box is None and stats['empty']


def test_setup_denial_precedes_residual_allocation(monkeypatch):
    first, second = map(_homogeneous, graph_pair(case_components('one_circle')))
    def forbidden(*args):
        raise AssertionError('residual allocated after setup was denied')
    monkeypatch.setattr(clip, 'psi_vector_net', forbidden)
    box, stats = clip.clip_residual_box(first, second, charge=lambda n: False)
    assert box == ((0., 1.),)*4
    assert stats['denied'] and stats['work'] == 0


def test_early_work_stop_returns_a_coarse_box_containing_the_contact():
    plane = np.array([[[s, t, 0., 1.] for t in (0., 1.)] for s in (0., 1.)])
    crossing = plane.copy()
    crossing[..., 2] = [[-.5, .5], [-.5, .5]]
    remaining = [2]
    def charge(n):
        if remaining[0] < n:
            return False
        remaining[0] -= n
        return True
    box, stats = clip.clip_residual_box(plane, crossing, charge=charge)
    assert stats['denied']
    roots = np.array([[s, .5, s, .5] for s in np.linspace(0., 1., 101)])
    bounds = np.asarray(box)
    assert np.all(roots >= bounds[:, 0]) and np.all(roots <= bounds[:, 1])


@pytest.mark.parametrize('swap', [False, True])
def test_nonuniform_weights_keep_paired_roots_in_different_charts(swap):
    # S1(s,t)=(2*s/(1+s),t,0), represented homogeneously with weights
    # 1 and 2. S2(a,b)=(.4+.2*a,.5,b-.5). Restricting the second chart
    # makes ignoring the first chart's weights exclude genuine roots.
    first = np.array([
        [[0., 0., 0., 1.], [0., 1., 0., 1.]],
        [[2., 0., 0., 2.], [2., 2., 0., 2.]],
    ])
    second = np.array([[[u, .5, v-.5, 1.] for v in (0., 1.)]
                       for u in (.4, .6)])
    a = np.linspace(0., 1., 257)
    x = .4+.2*a
    s = x/(2.-x)
    roots = np.column_stack((s, np.full(len(s), .5), a,
                             np.full(len(s), .5)))
    if swap:
        first, second = second, first
        roots = roots[:, [2, 3, 0, 1]]
    box, stats = clip.clip_residual_box(first, second)
    assert box is not None and stats['valid'] and not stats['denied']
    bounds = np.asarray(box)
    assert np.all(roots >= bounds[:, 0])
    assert np.all(roots <= bounds[:, 1])
