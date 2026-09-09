"""Source proofs for straight tangent arcs, independent of residual size."""
import numpy as np
import pytest

from mmcore.numeric.intersection.ssx._ssx_affine_path import certified_straight_isocurve_path
from mmcore.numeric.intersection.ssx._ssx_affine_path import affine_path_representation_bounded
from mmcore.numeric.intersection.ssx._ssx_affine_path import source_box_image_diameter_bounded


def _pair(degree=12, gap=0.):
    surface = np.array([[[s, j/degree, (-1.)**(degree-j)*2.**-degree+gap]
                         for j in range(degree+1)] for s in (0., 1.)])
    plane = np.array([[[u, v, 0.] for v in (-.5, 1.5)] for u in (-.5, 1.5)])
    return surface, plane


@pytest.mark.parametrize('degree', [4, 8, 10, 12])
@pytest.mark.parametrize('swapped', [False, True])
def test_high_order_zero_line_uses_exact_source_inclusion(degree, swapped):
    a, b = _pair(degree)
    start, end = np.array([0., .5, .25, .5]), np.array([1., .5, .75, .5])
    if swapped:
        a, b = b, a
        start, end = start[[2, 3, 0, 1]], end[[2, 3, 0, 1]]
    path = certified_straight_isocurve_path(a, b, start, end, 1e-9, False)
    assert path is not None
    assert np.all(path[1][:, 2] == 0.)
    np.testing.assert_allclose(path[1], [[0., .5, 0.], [1., .5, 0.]], atol=1e-16, rtol=0)


def test_small_positive_gap_cannot_become_an_exact_line():
    a, b = _pair(gap=2.**-30)
    assert certified_straight_isocurve_path(
        a, b, [0., .5, .25, .5], [1., .5, .75, .5], 1e-3, False) is None


def test_off_locus_small_residual_endpoints_are_rejected():
    a, b = _pair()
    assert certified_straight_isocurve_path(
        a, b, [0., .49, .25, .495], [1., .49, .75, .495], 1e-3, False) is None


def test_backtracking_collinear_isocurve_does_not_get_single_arc_proof():
    a, b = _pair(4)
    a = np.repeat(a[:1], 4, axis=0)
    a[:, :, 0] = np.array([0., 2., -1., 1.])[:, None]
    assert certified_straight_isocurve_path(
        a, b, [0., .5, .25, .5], [1., .5, .75, .5], 1e-3, False) is None


def test_identity_certificate_obeys_work_denial():
    a, b = _pair()
    assert certified_straight_isocurve_path(
        a, b, [0., .5, .25, .5], [1., .5, .75, .5], 1e-3, False,
        charge=lambda amount: False) is None


def test_small_first_surface_image_does_not_hide_large_second_image_error():
    e = 2.**-14
    # Exact intersection t=v=.5*u+.5*u^2, s=u. Its physical image is tiny,
    # but the proposed linear lifted chord t=v=u leaves B_z=u*(1-u)/4.
    a = np.array([[[e*(.25+.5*s), e*(1/16+.5*t), 0.]
                   for t in (0., 1.)] for s in (0., 1.)])
    squares = np.array([1/16, 3/16, 9/16])
    b = np.array([[[e*(.25+.5*u), e*(1/16+.5*v), 1/16+.5*v-squares[i]]
                   for v in (0., 1.)] for i, u in enumerate((0., .5, 1.))])
    xyz = np.array([[e*.25, e/16, 0.], [e*.75, e*9/16, 0.]])
    assert not affine_path_representation_bounded(a, b, [0.]*4, [1.]*4, xyz, 1e-3, False)


def test_complete_affine_lifted_path_including_reversed_axes():
    a = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    b = a[::-1, ::-1].copy()
    assert affine_path_representation_bounded(
        a, b, [0., .25, 1., .75], [1., .75, 0., .25],
        [[0., .25, 0.], [1., .75, 0.]], 1e-9, False)


def test_source_box_bound_covers_interior_excursion_not_only_endpoints():
    a = np.array([[[1e-4*s, 1e-4*t, z] for t in (0., 1.)]
                  for s, z in ((0., 0.), (.5, 1.), (1., 0.))])
    xyz = [[0., 0., 0.], [1e-4, 1e-4, 0.]]
    assert not source_box_image_diameter_bounded(a, a, [(0., 1.)]*4, xyz, .01, False)
    # Exact restriction of the same curved surface has a small image.
    assert source_box_image_diameter_bounded(
        a, a, [(0., 1e-4)]*4, [[0., 0., 0.], [1e-8, 1e-8, 2e-4]], .01, False)


def test_source_box_bound_includes_actual_reported_xyz():
    a = np.zeros((2, 2, 3))
    assert not source_box_image_diameter_bounded(
        a, a, [(0., 1.)]*4, [[0., 0., 0.], [0., 0., 1.]], .01, False)
