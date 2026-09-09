"""A residual tolerance cannot establish a two-dimensional intersection."""
import numpy as np
import pytest

from mmcore.numeric.intersection.ssx._ssx5_overlap import (
    _exact_common_surface, _exact_surface_injective, assemble_overlap_regions,
)


def _homogeneous(points, weights=None):
    points = np.asarray(points, dtype=float)
    w = np.ones(points.shape[:2]) if weights is None else np.asarray(weights)
    return np.concatenate((points * w[..., None], w[..., None]), axis=-1)


def _plane():
    return _homogeneous([[[0., 0., 0.], [0., 1., 0.]],
                         [[1., 0., 0.], [1., 1., 0.]]])


@pytest.mark.parametrize('gap', [1e-6, 1e-18, np.nextafter(0., 1.)])
def test_arbitrarily_small_nonzero_gap_is_not_a_surface_identity(gap):
    a = _plane()
    b = a.copy()
    b[..., 2] = gap
    assert _exact_common_surface(a, b) is None
    result = assemble_overlap_regions(a, b, atol=.001, ptol4=np.full(4, .001))
    assert result['regions'] == [] and not result['covered']


def test_high_order_contact_is_not_a_common_surface():
    # All heights are below atol, but z=(t-.5)^2 / 1024 vanishes only
    # on one line. Its nonzero coefficient plane prevents promotion.
    a = _homogeneous([[[s, t, z/1024.] for t, z in zip((0., .5, 1.), (.25, -.25, .25))]
                       for s in (0., 1.)])
    assert _exact_common_surface(a, _plane()) is None


def test_rational_planar_patches_have_an_exact_common_image():
    a = _plane()
    b = _homogeneous([[[.5, .5, 0.], [.5, 2., 0.]],
                       [[2., .5, 0.], [2., 2., 0.]]], [[1., 2.], [3., 4.]])
    assert _exact_common_surface(a, b) == 'common_control_plane'


def test_nonplanar_identical_charts_support_reversal_and_weight_scaling():
    a = _homogeneous([[[0., 0., 0.], [0., 1., 0.]],
                       [[1., 0., 0.], [1., 1., 1.]]])
    for b in (a.copy(), a[::-1], a[:, ::-1], a.transpose(1, 0, 2) * 2.):
        assert _exact_common_surface(a, b) == 'homogeneous_chart_identity'
        assert _exact_surface_injective(b)


def test_identical_self_intersecting_chart_does_not_exhaust_lifted_zero_set():
    # Exact dyadic Bernstein representation of a cubic extruded loop.
    # s=.25 and s=.75 share the entire world-space line (.1875,t,0).
    points = np.array([[[x, t, z] for t in (0., 1.)]
                       for x, z in zip(np.array([3., -1., -1., 3.])/4.,
                                       np.array([-9., 13., -13., 9.])/32.)])
    a = _homogeneous(points)
    assert _exact_common_surface(a, a) == 'homogeneous_chart_identity'
    assert not _exact_surface_injective(a)
    result = assemble_overlap_regions(a, a, atol=.001, ptol4=np.full(4, .001),
                                      overlap_boxes=[((.24, .26), (.4, .6),
                                                      (.74, .76), (.4, .6))])
    assert len(result['regions']) == 1
    assert not result['covered']
    assert not result['regions'][0].certification['injective_charts']


def test_exact_projection_proves_graph_injectivity_with_arbitrary_height():
    s, t = np.meshgrid(np.linspace(0., 1., 4), np.linspace(0., 1., 5), indexing='ij')
    height = np.array([[100., -5., 7., 9., -4.], [-9., 80., -6., 2., 9.],
                       [2., -8., -99., 7., 3.], [9., 3., -7., 4., 8.]])
    assert _exact_surface_injective(_homogeneous(np.stack((s, t, height), axis=-1)))


def test_sampled_rim_cannot_absorb_existing_source_branch_from_displayed_chords():
    from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch
    a = _plane()
    stuv = np.array([[0., 0., 0., 0.], [0., 1., 0., 1.]])
    branch = SSXBranch(curve=(stuv, stuv[:, [0, 1, 0]]), overlap=True, kind='overlap')
    result = assemble_overlap_regions(a, a, atol=.001, ptol4=np.full(4, .001),
                                     existing_overlap_branches=[branch])
    assert result['regions']
    assert any(kept is branch for kept in result['unmatched_branches'])


def test_sampled_common_plane_regions_do_not_claim_exhaustive_trim_topology():
    first = _plane()
    second = _homogeneous([[[.25, .25, 0.], [.125, .875, 0.]],
                           [[.75, .25, 0.], [.875, .875, 0.]]])
    assert _exact_common_surface(first, second) == 'common_control_plane'
    assert _exact_surface_injective(first) and _exact_surface_injective(second)
    result = assemble_overlap_regions(first, second, atol=.001, ptol4=np.full(4, .001))
    assert result['regions']
    assert not result['covered']
