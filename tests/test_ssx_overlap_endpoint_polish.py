"""Overlap endpoint correction cannot choose a remote boundary contact."""
from math import comb

import numpy as np
import pytest

from mmcore.numeric._bezier_common import eval_surface
from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx


def _homogeneous(points):
    return np.concatenate((points, np.ones(points.shape[:2]+(1,))), axis=-1)


@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('rational', [False, True])
def test_straight_boundary_fringe_returns_to_its_local_contact(swap, rational):
    first = np.array([[[t, s, .5*(t-.002)] for t in (0., 1.)]
                      for s in (0., 1.)])
    second = np.array([[[u, 0., v] for v in (0., 1.)] for u in (0., 1.)])
    if rational:
        first, second = _homogeneous(first), _homogeneous(second)
        # Positive weights change the parameterization of the straight edges.
        first *= np.array([[1., 2.], [1.5, 1.]])[..., None]
        second *= np.array([[1., 1.5], [2., 1.]])[..., None]
    if swap:
        first, second = second, first
    result = ssx._polish_overlap_domain_endpoint(
        first, second, np.zeros(4), 2 if swap else 0, (0., 1.), .001, rational)
    points = [eval_surface(surface, *uv, rational=rational)
              for surface, uv in ((first, result[:2]), (second, result[2:]))]
    for point in points:
        np.testing.assert_allclose(point, [.002, 0., 0.], atol=1e-12, rtol=0.)


def test_curved_edge_with_two_contacts_cannot_jump_to_the_far_root():
    # z=.01*(t*t-.01)*(1-t): contacts at .1 and 1. An unrestricted
    # Newton step from the tolerance fringe t=0 jumps directly to t=1.
    power = np.array([-.0001, .0001, .01, -.01])
    height = np.array([sum(power[j]*comb(i, j)/comb(3, j)
                           for j in range(i+1)) for i in range(4)])
    first = np.array([[[t, s, z] for t, z in zip(np.linspace(0., 1., 4), height)]
                      for s in (0., 1.)])
    second = np.array([[[u, 0., v] for v in (0., 1.)] for u in (0., 1.)])
    original = np.zeros(4)
    result = ssx._polish_overlap_domain_endpoint(
        _homogeneous(first), _homogeneous(second), original, 0, (0., 1.), .001, True)
    np.testing.assert_array_equal(result, original)


def test_endpoint_polish_does_not_consume_most_of_a_tolerance_overlap():
    first = np.array([[[t, s, .0001*(t-.9)] for t in (0., 1.)]
                      for s in (0., 1.)])
    second = np.array([[[u, 0., v] for v in (0., 1.)] for u in (0., 1.)])
    original = np.zeros(4)
    result = ssx._polish_overlap_domain_endpoint(
        first, second, original, 0, (0., 1.), .001, False)
    # A far contact requires re-searching the overlap. Correction alone
    # must not replace the already recovered interval with its far end.
    np.testing.assert_array_equal(result, original)
