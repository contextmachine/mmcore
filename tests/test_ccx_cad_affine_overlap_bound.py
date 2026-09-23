"""Whole-span CAD bounds cannot be inferred from endpoints or XYZ alone."""

import numpy as np
import pytest

from mmcore.numeric._bezier_common import eval_curve
from mmcore.numeric.intersection.ccx._bez_ccx4 import (
    _cad_affine_overlap_bound, bez_ccx,
)


def test_close_endpoints_do_not_bound_an_out_of_tolerance_interior():
    first = np.array([[0., 0., 0.], [.5, 0., 0.], [1., 0., 0.]])
    second = first.copy()
    second[1, 1] = .01
    bound = _cad_affine_overlap_bound(first, second, (0., 1.), (0., 1.), False)
    assert bound > 1e-3
    assert np.linalg.norm(eval_curve(first, .5, rational=False)
                          - eval_curve(second, .5, rational=False)) > 1e-3


def test_rational_weighted_parameter_map_is_not_cartesian_control_identity():
    # Same line locus, different parameter maps. Equal dehomogenized
    # controls do not establish the proposed affine parameter pairing.
    first = np.array([[0., 0., 0., 1.], [1., 0., 0., 1.]])
    second = np.array([[0., 0., 0., 1.], [4., 0., 0., 4.]])
    bound = _cad_affine_overlap_bound(first, second, (0., 1.), (0., 1.), True)
    assert bound > 1e-3


@pytest.mark.parametrize('reverse', [False, True])
def test_positive_rational_weights_preserve_the_whole_distance_bound(reverse):
    points = np.array([[0., 0., 0.], [.5, 1., 0.], [1., 0., 0.]])
    weights = np.array([1., 2., 3.])
    first = np.column_stack([points * weights[:, None], weights])
    second = np.column_stack([(points + [0., 0., 1e-5]) * weights[:, None], weights]) * 7.
    v_range = (0., 1.)
    if reverse:
        second = second[::-1].copy()
        v_range = (1., 0.)
    bound = _cad_affine_overlap_bound(first, second, (0., 1.), v_range, True)
    assert bound is not None and bound <= 1e-3
    for t in np.linspace(0., 1., 101):
        v = 1. - t if reverse else t
        gap = np.linalg.norm(eval_curve(first, t, rational=True)
                             - eval_curve(second, v, rational=True))
        assert gap <= bound


def test_sign_changing_weights_cannot_use_a_positive_denominator_bound():
    first = np.array([[0., 0., 0., 1.], [1., 0., 0., 1.]])
    second = np.array([[0., 0., 0., 1.], [-1., 0., 0., -1.]])
    assert _cad_affine_overlap_bound(first, second, (0., 1.), (0., 1.), True) is None


@pytest.mark.parametrize('limits', [dict(max_cells=0), dict(max_results=0)])
def test_near_coincident_overlap_does_not_bypass_denied_allowances(limits):
    first = np.array([[-19.77608536, 23.10065701, 0.],
                      [-14.86834768, 28.69713066, 0.],
                      [-5.8568525, 25.12677787, 0.],
                      [-12.62581769, 15.26478654, 0.]])
    result = bez_ccx(first, first + [0., 1e-9, 0.], atol=1e-3,
                     rational=False, **limits)
    assert result['overlaps'] == [] and result['isolated'] == []
    assert result['budget_exhausted']
    if limits.get('max_cells') == 0:
        assert result['cells_processed'] == 0
