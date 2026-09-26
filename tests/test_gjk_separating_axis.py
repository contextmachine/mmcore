"""GJK directions are proposals; a caller still verifies its model-space gap."""
import numpy as np
import pytest

import mmcore.numeric.algorithms.cygjk as cygjk


ATOL = 1e-3
BOX = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
TET = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])


def _proposal(first, second, **kwargs):
    propose = getattr(cygjk, "gjk_separating_axis", None)
    assert callable(propose), "the native GJK axis-proposal API is missing"
    return propose(first, second, **kwargs)


def _support_gap(first, second, axis):
    """Independently check all vertices along the proposed direction."""
    assert isinstance(axis, tuple) and len(axis) == 3
    direction = np.asarray(axis, dtype=float)
    assert np.all(np.isfinite(direction))
    direction /= np.max(np.abs(direction))
    direction /= np.linalg.norm(direction)
    origin = first[0]
    a = (first - origin) @ direction
    b = (second - origin) @ direction
    return max(float(b.min() - a.max()), float(a.min() - b.max()))


@pytest.mark.parametrize("translation", [0., 1e6])
@pytest.mark.parametrize("swap", [False, True])
def test_direction_separates_rotated_hulls_with_overlapping_aabbs(translation, swap):
    c = np.sqrt(.5)
    rotation = np.array([[c, -c, 0.], [c, c, 0.], [0., 0., 1.]])
    first = BOX @ rotation.T + translation
    second = first + 1.1 * rotation[:, 0]
    assert np.all(first.max(0) >= second.min(0))
    assert np.all(second.max(0) >= first.min(0))
    if swap:
        first, second = second, first
    assert _support_gap(first, second, _proposal(first, second)) > 2 * ATOL


@pytest.mark.parametrize("shift", [0., .5, 1.])
def test_overlap_and_exact_face_contact_have_no_proposal(shift):
    assert _proposal(BOX, BOX + [shift, 0., 0.]) is None


def test_exhausted_search_does_not_invent_an_axis():
    # The first tetrahedron has x+y+z <= 1, the other >= 1.25.
    # One iteration is insufficient to find their oblique separator.
    second = -TET + .75
    assert _proposal(TET, second, max_iter=1) is None
    assert _support_gap(TET, second, _proposal(TET, second, max_iter=25)) > 2 * ATOL


@pytest.mark.parametrize("points", [
    np.zeros((1, 3)),
    np.array([[0., 0., 0.], [1., 0., 0.]]),
    np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
])
def test_degenerate_hulls_return_only_useful_directions(points):
    assert _proposal(points, points) is None
    separated = points + [0., 0., 1.]
    assert _support_gap(points, separated, _proposal(points, separated)) > 2 * ATOL


def test_sub_tolerance_direction_is_not_a_padded_separation_verdict():
    second = BOX + [1. + .25 * ATOL, 0., 0.]
    axis = _proposal(BOX, second)
    gap = _support_gap(BOX, second, axis)
    assert 0. < gap < ATOL
    assert not gap > 2 * ATOL


def test_large_finite_input_cannot_overflow_the_support_search():
    first = BOX * 1e300
    second = first + [2e300, 0., 0.]
    assert _support_gap(first, second, _proposal(first, second)) > 2 * ATOL


def test_readonly_and_noncontiguous_inputs_are_supported():
    first = BOX[::-1]
    second = (BOX + [2., 0., 0.])[::-1]
    first.setflags(write=False)
    second.setflags(write=False)
    assert _support_gap(first, second, _proposal(first, second)) > 2 * ATOL


@pytest.mark.parametrize("bad", [
    None, np.empty((0, 3)), np.zeros((2, 2)), np.zeros((2, 4)), np.zeros(3),
    np.array([[np.nan, 0., 0.]]), np.array([[0., np.inf, 0.]]),
])
@pytest.mark.parametrize("side", [0, 1])
def test_invalid_vertices_are_rejected_before_native_access(bad, side):
    inputs = [BOX, BOX]
    inputs[side] = bad
    with pytest.raises(ValueError):
        _proposal(*inputs)


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_iteration_limits_are_rejected(limit):
    with pytest.raises(ValueError):
        _proposal(BOX, BOX, max_iter=limit)


def test_fractional_iteration_limit_is_rejected():
    with pytest.raises(TypeError):
        _proposal(BOX, BOX, max_iter=1.5)


@pytest.mark.parametrize("tolerance", [-1., np.nan, np.inf])
def test_invalid_numerical_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError):
        _proposal(BOX, BOX, tol=tolerance)
