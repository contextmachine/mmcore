"""Ledger L48: regression pins for the 2026-07-12 `_nurbs_param_tol` semantic shift.

The budget-review commit changed the `bez_curve_param_tolerance` /
`bez_surface_param_tolerance` dispatch: ALL non-uniform-weight rational inputs
now take the conservative (OCC-style, control-leg) bound — previously only
negative-weight inputs did, and the optimistic bound's homogeneous-derivative
scaling made ptol spuriously TINY at large world coordinates (review finding
10 measured ~7.6e-7 for a radius-1 rational quarter-circle translated to
(1000, 2000) versus 2.7e-4 at the origin — a ~350x acceptance/dedup radius
jump for consumers). The conservative bound is translation-INVARIANT: that is
the new semantics these tests pin, at the function level and at the two
previously zero-coverage legacy consumers (the
`nurbs_ssx` path, and the Bez-tree closest point), at the origin AND far from
it — so the next tolerance-ladder change trips loudly here instead of
silently shifting legacy acceptance/dedup radii.
"""
import numpy as np
import pytest

from mmcore.geom._nurbs_param_tol import (
    bez_curve_param_tolerance,
    bez_surface_param_tolerance,
)

W = np.sqrt(0.5)
FAR = np.array([1000.0, 2000.0, 0.0])


def _quarter_circle_homog(offset=(0.0, 0.0, 0.0)):
    """Radius-1 rational quarter circle (weights 1, sqrt(1/2), 1), homogeneous."""
    off = np.asarray(offset, dtype=float)
    pc = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]) + off
    ws = np.array([1.0, W, 1.0])
    return np.concatenate([pc * ws[:, None], ws[:, None]], axis=-1)


def _quarter_cylinder_homog(offset=(0.0, 0.0, 0.0)):
    """Quarter-cylinder patch: the arc extruded one unit in z, homogeneous."""
    arc = _quarter_circle_homog(offset)
    top = arc.copy()
    top[:, 2] += arc[:, 3] * 1.0          # z*w column shifts by w
    return np.stack([arc, top], axis=1)   # (3, 2, 4)


def test_nonuniform_rational_curve_bound_is_conservative_and_translation_invariant():
    t_origin = bez_curve_param_tolerance(
        _quarter_circle_homog(), 1e-3, rational=True)
    t_far = bez_curve_param_tolerance(
        _quarter_circle_homog(FAR), 1e-3, rational=True)
    # conservative control-leg bound: 2.7346e-4 for the radius-1 arc
    assert t_origin == pytest.approx(2.7345908e-4, rel=1e-5)
    # translation must not change the bound (the old optimistic dispatch
    # shrank it ~350x at (1000, 2000) via homogeneous-derivative magnitudes)
    assert t_far == pytest.approx(t_origin, rel=1e-9)


def test_uniform_weight_curve_bound_is_optimistic_and_translation_invariant():
    pc = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    for off in (np.zeros(3), FAR):
        homog = np.concatenate([pc + off, np.ones((3, 1))], axis=-1)
        t = bez_curve_param_tolerance(homog, 1e-3, rational=True)
        assert t == pytest.approx(1e-3, rel=1e-6), off


def test_nonuniform_rational_surface_bound_is_translation_invariant():
    p_origin = bez_surface_param_tolerance(
        _quarter_cylinder_homog(), 1e-3, rational=True)
    p_far = bez_surface_param_tolerance(
        _quarter_cylinder_homog(FAR), 1e-3, rational=True)
    assert p_origin[0] == pytest.approx(2.2295145e-4, rel=1e-5)
    assert p_origin[1] == pytest.approx(7.0710678e-4, rel=1e-5)
    assert p_far[0] == pytest.approx(p_origin[0], rel=1e-9)
    assert p_far[1] == pytest.approx(p_origin[1], rel=1e-9)



def test_legacy_beztree_closest_point_is_translation_stable():
    """`closest_point.bez_curve_closest_point`'s tree nodes size ptol from the
    shifted bound; the returned parameter must match at the origin and far."""
    from mmcore.numeric.closest_point import bez_curve_closest_point

    ts = []
    for off in (np.zeros(3), FAR):
        arc = _quarter_circle_homog(off)
        probe = np.array([1.2, 1.2, 0.0]) + off
        t_best, _sq_dist = bez_curve_closest_point(
            arc, probe, atol=1e-3, rational=True)
        ts.append(float(t_best))
    assert ts[0] == pytest.approx(0.5, abs=1e-6)
    assert ts[1] == pytest.approx(ts[0], abs=1e-6)


@pytest.mark.parametrize("scale", [0.0, 5e-324, 1e-200, 2.5e-162, 1e-150])
def test_collapsed_speed_tolerances_stay_finite(scale):
    """L52 (review §10): the optimistic dt/du/dv guards use `< _TINY`.

    NOTE (fixture-first honesty): with numpy's squared-sum norm this cannot
    go RED today — components below ~1.5e-162 flush the norm to exactly
    0.0, so the old `== 0.0` guard was rescued by an underflow accident
    (measured worst reachable tol is ~4.5e158, finite). This pin exists so
    a future norm implementation (e.g. scaled/hypot-style) cannot expose
    the quotient to denormal denominators and ship an inf/NaN ptol — an
    inf ptol would make every destructive dedup box-test true downstream.
    """
    C = np.array([[0.0, 0.0, 0.0], [scale, 0.0, 0.0]])
    tol_u = bez_curve_param_tolerance(C, 1e-3, rational=False)
    assert np.isfinite(tol_u) and tol_u > 0.0

    S = np.array([[[0.0, 0.0, 0.0], [scale, 0.0, 0.0]],
                  [[0.0, scale, 0.0], [scale, scale, 0.0]]])
    tol_su, tol_sv = bez_surface_param_tolerance(S, 1e-3, rational=False)
    assert np.isfinite(tol_su) and tol_su > 0.0
    assert np.isfinite(tol_sv) and tol_sv > 0.0
