"""Exact algebraic certificates for curved-UV planar CSX overlaps."""
from fractions import Fraction

import numpy as np
import pytest

from mmcore.numeric.intersection.csx._planar_overlap import exact_planar_bilinear_overlap


def _quad():
    # S(u,v)=(u*(2+v),2*v,0); a regular non-parallelogram patch.
    return np.array([[[0., 0., 0.], [0., 2., 0.]],
                     [[2., 0., 0.], [3., 2., 0.]]])


def _homogeneous(cp, weights):
    w = np.asarray(weights)
    return np.concatenate((cp*w[..., None], w[..., None]), axis=-1)


def _exact_ranges(result):
    return tuple(Fraction(int(n), int(d)) for n, d in result[0]["exact_t_range"])


def test_line_is_clipped_by_exact_quad_halfspaces():
    curve = np.array([[-1., 1., 0.], [4., 1., 0.]])
    result = exact_planar_bilinear_overlap(curve, _quad())
    assert _exact_ranges(result) == (Fraction(1, 5), Fraction(7, 10))
    assert result[0]["certification"] == "exact"
    assert result[0]["uv_range_is_enclosure"] is True


def test_rational_line_clipping_maps_back_to_its_actual_parameter():
    curve = np.array([[-1., 1., 0.], [4., 1., 0.]])
    result = exact_planar_bilinear_overlap(
        _homogeneous(curve, [2., 3.]), _homogeneous(_quad(), np.ones((2, 2))),
        rational=True)
    assert _exact_ranges(result) == (Fraction(1, 7), Fraction(14, 23))


def test_curved_inverse_uv_is_certified_by_whole_control_hull():
    curve = np.array([[.25, .25, 0.], [1., 1.75, 0.], [1.75, .25, 0.]])
    result = exact_planar_bilinear_overlap(curve, _quad())
    assert _exact_ranges(result) == (Fraction(0), Fraction(1))
    assert result[0]["parameterization"] == "unique_bilinear_inverse"


@pytest.mark.parametrize("surface_reverse", [False, True])
def test_exact_affine_plane_change_and_axis_reversal_preserve_span(surface_reverse):
    curve = np.array([[-1., 1., 0.], [4., 1., 0.]])
    surface = _quad()
    matrix = np.array([[1., 2., 0.], [-1., 1., 0.], [2., 0., 1.]])
    offset = np.array([1024., -2048., 512.])
    curve = curve @ matrix.T + offset
    surface = surface @ matrix.T + offset
    if surface_reverse:
        surface = surface[::-1]
    result = exact_planar_bilinear_overlap(curve, surface)
    assert _exact_ranges(result) == (Fraction(1, 5), Fraction(7, 10))


def test_nonzero_plane_gap_is_rejected_at_any_residual_scale():
    curve = np.array([[.25, 1., 2.**-40], [1.75, 1., 2.**-40]])
    assert exact_planar_bilinear_overlap(curve, _quad()) is None


def test_nonplanar_bilinear_patch_is_not_replaced_by_a_plane():
    surface = _quad()
    surface[1, 1, 2] = 2.**-40
    curve = np.array([[.25, 1., 0.], [1.75, 1., 0.]])
    assert exact_planar_bilinear_overlap(curve, surface) is None


def test_concave_or_folded_quad_has_no_injective_map_certificate():
    surface = _quad()
    surface[1, 1, :2] = [-1., .5]
    curve = np.array([[.25, .25, 0.], [.5, .5, 0.]])
    assert exact_planar_bilinear_overlap(curve, surface) is None


def test_boundary_samples_cannot_certify_uncontained_higher_degree_hull():
    curve = np.array([[.25, .25, 0.], [8., 1., 0.], [1.75, .25, 0.]])
    samples = [(0., .1, .1), (1., .9, .1)]
    assert exact_planar_bilinear_overlap(curve, _quad(), samples) is None


def test_outside_line_or_point_contact_is_not_a_positive_span():
    outside = np.array([[-2., 1., 0.], [-1., 1., 0.]])
    contact = np.array([[-1., 0., 0.], [0., 0., 0.]])
    assert exact_planar_bilinear_overlap(outside, _quad()) is None
    assert exact_planar_bilinear_overlap(contact, _quad()) is None


def test_nonuniform_surface_weights_are_honestly_unsupported():
    curve = np.array([[.25, 1., 0.], [1.75, 1., 0.]])
    weights = np.array([[1., 1.], [1., 1. + 2.**-20]])
    assert exact_planar_bilinear_overlap(
        _homogeneous(curve, np.ones(2)), _homogeneous(_quad(), weights),
        rational=True) is None


def test_original_nonparallelogram_boundary_fixture_has_exact_spans():
    s1 = np.array([[[-2., -3.375, 0.], [-1., -3.125, .625]],
                   [[-4.5, .25, 0.], [-2.5, -.375, 0.]]])
    s2 = np.array([[[-4.25, -.875, 0.], [-3.25, .25, 0.]],
                   [[-2.375, -2.5, 0.], [-2.125, -1.25, 0.]]])
    for curve in (s1[1], s1[:, 0]):
        result = exact_planar_bilinear_overlap(curve, s2)
        lo, hi = _exact_ranges(result)
        assert 0 < lo < hi < 1
        # Both independently determined span ends lie exactly on at
        # least one projected quadrilateral boundary halfspace.
        polygon = s2[[0, 1, 1, 0], [0, 0, 1, 1], :2]
        for t in (lo, hi):
            point = [Fraction(float(curve[0, k])) + t*(Fraction(float(curve[1, k]))
                      - Fraction(float(curve[0, k]))) for k in range(2)]
            boundary_values = []
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                ax, ay, bx, by = map(lambda v: Fraction(float(v)), (*a, *b))
                boundary_values.append((bx-ax)*(point[1]-ay)-(by-ay)*(point[0]-ax))
            assert any(value == 0 for value in boundary_values)
