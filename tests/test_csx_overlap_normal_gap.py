"""Overlap sidedness uses the normal gap, independently of projection lag."""

import numpy as np
import pytest

import mmcore.numeric.intersection.csx._bez_csx4 as csx


@pytest.mark.parametrize("normal_gap, overlap", [
    (4.0 * np.finfo(float).eps, True),
    (1e-7, True),
    (2e-3, False),
])
def test_tangential_projection_error_does_not_amplify_normal_signs(
        monkeypatch, normal_gap, overlap):
    plane = np.array([[[0., 0., 0.], [0., 1., 0.]],
                      [[1., 0., 0.], [1., 1., 0.]]])
    curve = np.array([[0.1, 0.5, -normal_gap],
                      [0.9, 0.5, normal_gap]])

    def lagging_projection(point, surface, u, v, atol, rational):
        # The numerical projector may stop with an in-plane residual.
        # This must not promote sub-roundoff normal signs to a crossing.
        projected = np.array([point[0] + 1e-8, point[1], 0.])
        return (*projected[:2], float(np.linalg.norm(projected - point)))

    monkeypatch.setattr(csx, "_project_point_on_surface", lagging_projection)
    result = csx._tolerance_csx_overlap_certificate(
        curve, plane, 1e-3, False, 1e-5, None)

    if overlap:
        assert len(result) == 1
        assert result[0]["t_range"] == (0., 1.)
    else:
        # The in-tolerance middle interval has neither a curve-domain end
        # nor a surface-domain end. It remains an isolated crossing.
        assert result is None


@pytest.mark.parametrize('curve_scale,surface_scale', [(1., 1.), (1e8, 1e8), (1e-8, 1e8)])
def test_normal_crossing_contact_is_independent_of_homogeneous_gauge(curve_scale, surface_scale):
    plane = np.array([[[0., 0., 0., 1.], [0., 1., 0., 1.]],
                      [[1., 0., 0., 1.], [1., 1., 0., 1.]]]) * surface_scale
    curve = np.array([[.1, .5, -1e-7, 1.], [.9, .5, 1e-7, 1.]]) * curve_scale
    result = csx.bez_csx(curve, plane, atol=1e-3, rational=True)
    assert not result['budget_exhausted']
    assert len(result['overlaps']) == 1
    contacts = result['overlaps'][0]['boundary_contacts']
    crossing = [point for point in contacts if abs(point['t']-.5) <= 1e-3]
    assert crossing
    for point in crossing:
        a = csx.eval_curve(curve, point['t'], rational=True)
        b = csx.eval_surface(plane, point['u'], point['v'], rational=True)
        assert np.linalg.norm(a-b) <= 1e-3
    exact = csx.bez_csx(curve, plane, atol=1e-3, rational=True, tolerance_tier=False)
    assert exact['overlaps'] == []
    assert len(exact['isolated']) == 1
    assert abs(exact['isolated'][0]['t']-.5) <= 1e-3
