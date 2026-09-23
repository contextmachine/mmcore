"""Overlap sidedness uses the normal gap, independently of projection lag."""

import numpy as np
import pytest

import mmcore.numeric.intersection.csx._bez_csx4 as csx


@pytest.mark.parametrize("normal_gap, overlap", [
    (4.0 * np.finfo(float).eps, True),
    (1e-7, False),
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
        # A resolved normal sign change remains a crossing even though
        # the whole curve lies inside the modeling tolerance band.
        assert result is None
