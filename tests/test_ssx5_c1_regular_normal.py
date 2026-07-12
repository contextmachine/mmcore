"""C1 typing controls for small, nonzero surface normals."""

from math import comb

import numpy as np
import pytest

from mmcore.geom._nurbs_param_tol import bez_surface_param_tolerance
from mmcore.numeric.intersection._bezier_common import eval_surface_d1
from mmcore.numeric.intersection.ssx._ssx5_singular import c1_pass


def _homogeneous(surface):
    surface = np.asarray(surface, dtype=np.float64)
    return np.concatenate(
        [surface, np.ones(surface.shape[:-1] + (1,), dtype=np.float64)],
        axis=-1,
    )


def _monomial_to_bernstein_1d(coefficients, degree):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    return np.array([
        sum(
            coefficients[j] * comb(i, j) / comb(degree, j)
            for j in range(min(i, len(coefficients) - 1) + 1)
        )
        for i in range(degree + 1)
    ])


def _regular_sheared_touch(eps=5e-7):
    """Everywhere-regular injective patch touching its z=0 image plane.

    x = s + t, y = (t - 1/2)^3 + eps*t is injective because dy/dt is
    strictly positive.  Its normal has the certified lower bound eps, so
    the touch at (s,t)=(1/2,1/2) is C2 and never C1.
    """
    s_nodes = np.arange(3, dtype=np.float64) / 2.0
    t_nodes = np.arange(4, dtype=np.float64) / 3.0
    y_nodes = _monomial_to_bernstein_1d(
        [-0.125, 0.75 + eps, -1.5, 1.0], 3,
    )
    fs = _monomial_to_bernstein_1d([0.25, -1.0, 1.0], 2)
    ft = _monomial_to_bernstein_1d([0.25, -1.0, 1.0], 3)
    surface = np.array([
        [
            [s_nodes[i] + t_nodes[j], y_nodes[j], fs[i] + ft[j]]
            for j in range(4)
        ]
        for i in range(3)
    ])
    plane = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    ])
    return surface, plane


def _ptol4(surface1_h, surface2_h, atol):
    ps, pt = bez_surface_param_tolerance(
        surface1_h, atol, rational=True,
    )
    pu, pv = bez_surface_param_tolerance(
        surface2_h, atol, rational=True,
    )
    return np.maximum(
        np.array([float(ps), float(pt), float(pu), float(pv)]), 1e-9,
    )


@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0])
def test_regular_small_normal_is_never_typed_as_cusp(scale):
    surface, plane = _regular_sheared_touch()
    surface_h = _homogeneous(surface * scale)
    plane_h = _homogeneous(plane * scale)
    atol = 1e-8 * scale

    _, ds, dt = eval_surface_d1(
        surface_h, 0.5, 0.5, rational=True,
    )
    # Uniform coordinate scaling multiplies the normal by scale^2.
    assert np.linalg.norm(np.cross(ds, dt)) == pytest.approx(
        5e-7 * scale * scale,
    )

    stats = {}
    hits, curve_flag = c1_pass(
        surface_h,
        plane_h,
        atol,
        _ptol4(surface_h, plane_h, atol),
        max_cells=20_000,
        stats=stats,
    )
    assert hits == []
    assert curve_flag is False
    assert stats["incomplete"] is False


def test_resolution_floor_without_zero_certificate_is_partial(monkeypatch):
    """A surviving unresolved normal box cannot prove regularity or a cusp."""
    import mmcore.numeric.intersection.ssx._ssx5_singular as singular

    surface, plane = _regular_sheared_touch()
    surface_h = _homogeneous(surface)
    plane_h = _homogeneous(plane)

    normal = np.empty((2, 2, 3), dtype=np.float64)
    normal[..., 0] = [[-1.0, 1.0], [-1.0, 1.0]]
    normal[..., 1] = [[1.0, -1.0], [1.0, -1.0]]
    normal[..., 2] = [[-1.0, -1.0], [1.0, 1.0]]
    monkeypatch.setattr(
        singular, "sigma_normal_net", lambda *_args, **_kwargs: normal,
    )

    def unresolved(*_args, stats=None, **_kwargs):
        stats.update(
            floor_boxes=1,
            unresolved_floor_boxes=1,
            cells_processed=1,
            boxes_processed=1,
            budget_exhausted=False,
            external_budget_exhausted=False,
        )
        return [], False

    monkeypatch.setattr(singular, "solve_zero_dim", unresolved)
    stats = {}
    hits, curve_flag = c1_pass(
        surface_h,
        plane_h,
        1e-8,
        np.full(4, 1e-4),
        max_cells=10,
        stats=stats,
    )
    assert hits == []
    assert curve_flag is False
    assert stats["incomplete"] is True


def test_exact_cone_cusp_control_is_preserved():
    cone = np.array([
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 1.0, -1.0], [1.0, 1.0, 1.0]],
    ])
    plane = np.array([
        [[-1.5, -1.5, 0.0], [-1.5, 1.5, 0.0]],
        [[1.5, -1.5, 0.0], [1.5, 1.5, 0.0]],
    ])
    cone_h, plane_h = _homogeneous(cone), _homogeneous(plane)
    stats = {}
    hits, curve_flag = c1_pass(
        cone_h,
        plane_h,
        1e-3,
        _ptol4(cone_h, plane_h, 1e-3),
        stats=stats,
    )
    assert hits
    assert any(hit["surface"] == 1 for hit in hits)
    assert curve_flag or any("stuv" in hit for hit in hits)


def test_interior_regular_touch_does_not_publish_tolerance_valley_branches():
    """An isolated touch may be partial, but never a complete 1D SSI."""
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx

    surface, _ = _regular_sheared_touch()
    plane = np.array([
        [[-0.5, -1.0, 0.0], [-0.5, 1.0, 0.0]],
        [[1.5, -1.0, 0.0], [1.5, 1.0, 0.0]],
    ])
    result = bez_ssx(
        _homogeneous(surface), _homogeneous(plane),
        atol=1e-3, rational=True,
        max_postprocess_work=2_000_000,
    )

    assert result["branches"] == []
    assert any(g.kind == "tangent_point"
               for g in result["singularities"])
    # A second-order isolation certificate is not yet part of the regular
    # loop-free tracer. Refuse a complete topology claim in that ambiguity.
    assert result["complete"] is False


def _positive_gap_between_two_endpoint_touches(h):
    s_coeff = np.array([0.0, 0.5, 1.0])
    t_coeff = np.array([0.0, 0.5, 1.0])
    z_s = np.array([0.25, -0.25, 0.25])
    z_t = np.array([0.0, 2.0 * h, 0.0])
    surface = np.array([
        [[s_coeff[i], t_coeff[j], z_s[i] + z_t[j]]
         for j in range(3)]
        for i in range(3)
    ])
    plane = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    ])
    return surface, plane


def _regular_high_order_tangent_line(degree):
    """Everywhere-regular cylinder tangent to z=0 with multiplicity d.

    ``S(s,t) = (s, t, (t - 1/2)**degree)`` is injective and its parameter
    normal never vanishes because the x/y coordinates are the identity.
    For even ``degree`` its exact SSI with the plane is the single tangent
    line ``t=1/2``.  This separates root LOCATION from residual size:
    ``|t-1/2|**degree`` reaches roundoff many geometric tolerances away.
    """
    mono = np.array([
        comb(degree, k) * (-0.5) ** (degree - k)
        for k in range(degree + 1)
    ], dtype=np.float64)
    z_nodes = _monomial_to_bernstein_1d(mono, degree)
    t_nodes = np.arange(degree + 1, dtype=np.float64) / degree
    surface = np.array([
        [[float(i), t_nodes[j], z_nodes[j]]
         for j in range(degree + 1)]
        for i in range(2)
    ])
    plane = np.array([
        [[-0.5, -0.5, 0.0], [-0.5, 1.5, 0.0]],
        [[1.5, -0.5, 0.0], [1.5, 1.5, 0.0]],
    ])
    return surface, plane


@pytest.mark.parametrize("h", [1e-7, 1e-2])
def test_positive_gap_between_endpoint_touches_is_not_a_branch(h):
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx

    surface, plane = _positive_gap_between_two_endpoint_touches(h)
    result = bez_ssx(surface, plane, atol=1e-3, rational=False)

    assert result["branches"] == []
    assert len(result["points"]) == 2
    assert result["complete"] is False


def test_zero_gap_control_remains_a_tangent_line():
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx

    surface, plane = _positive_gap_between_two_endpoint_touches(0.0)
    result = bez_ssx(surface, plane, atol=1e-3, rational=False)

    assert result["complete"] is True
    assert len(result["branches"]) == 1
    assert result["branches"][0].kind == "tangential"


def test_ssx_correctors_compare_squared_residual_to_squared_tolerance():
    from mmcore.numeric.intersection.ssx._bez_ssx5 import (
        _ssx_correct, _ssx_correct_fixed)

    t_coeff = np.array([0.0, 0.5, 1.0])
    z_coeff = np.array([0.25, -0.25, 0.25])
    surface = np.array([
        [[float(i), t_coeff[j], z_coeff[j]] for j in range(3)]
        for i in range(2)
    ])
    plane = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    ])
    start = np.array([0.5, 0.5001, 0.5, 0.5001])

    corrected = _ssx_correct(
        surface, plane, *start, rational=False)
    assert corrected[4] < 1e-12

    fixed, residual, _ = _ssx_correct_fixed(
        surface, plane, start, fixed_axis=0, fixed_value=0.5,
        rational=False)
    assert residual < 1e-12
    assert abs(fixed[1] - 0.5) < 1e-5


def test_quartic_regular_tangent_line_is_complete():
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx

    surface, plane = _regular_high_order_tangent_line(4)
    result = bez_ssx(surface, plane, atol=1e-3, rational=False)

    assert result["complete"] is True
    assert len(result["branches"]) == 1
    branch = result["branches"][0]
    assert branch.kind == "tangential"
    xyz = np.asarray(branch.curve[1])
    assert xyz[:, 0].min() < 0.01 and xyz[:, 0].max() > 0.99
    assert np.allclose(xyz[:, 1], 0.5, atol=1e-3)


@pytest.mark.parametrize("degree", [8, 10, 12])
def test_high_order_tangent_never_publishes_off_locus_branches(degree):
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx

    surface, plane = _regular_high_order_tangent_line(degree)
    result = bez_ssx(surface, plane, atol=1e-3, rational=False)

    # A partial result may omit topology it cannot certify, but every
    # published branch is certified geometry.  Residual-only verification
    # used to emit 4--8 false copies up to 48*atol from the exact line.
    for branch in result["branches"]:
        xyz = np.asarray(branch.curve[1])
        assert np.max(np.abs(xyz[:, 1] - 0.5)) <= 2e-3
