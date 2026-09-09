"""Analytic parameter-space regressions for the CSX and CCX substrates."""
import numpy as np
import pytest

from mmcore.nurbs._nurbs_eval import NURBSCurveTuple, NURBSSurfaceTuple
from mmcore.numeric.intersection.ccx._bez_ccx4 import bez_ccx
from mmcore.numeric.intersection.ccx._nccx4 import nurbs_ccx
from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx
from mmcore.numeric.intersection.csx._ncsx4 import nurbs_csx


def _folded_plane(center=0.5):
    # S(u,v)=((u-center)**2,v,0), degree (2,1). Both preimages
    # in the tests are regular; only the intervening fold is singular.
    surface = np.zeros((3, 2, 3))
    surface[:, :, 0] = np.array([
        center**2, center**2 - center, (1-center)**2,
    ])[:, None]
    surface[:, 1, 1] = 1.0
    return surface


@pytest.mark.parametrize('t_root', [0.0, 0.5, 1.0])
def test_csx_retains_roots_on_original_and_subdivision_faces(t_root):
    curve = np.array([[1/16, 1/2, -t_root], [1/16, 1/2, 1-t_root]])
    result = bez_csx(curve, _folded_plane(), atol=1e-6, rational=False)
    actual = sorted((p['t'], p['u'], p['v']) for p in result['isolated'])
    np.testing.assert_allclose(actual, [[t_root, .25, .5], [t_root, .75, .5]], atol=1e-9)
    assert not result['budget_exhausted']
    assert result['boundary_topology_complete']


def test_csx_boundary_root_does_not_remove_other_surface_preimage():
    curve = np.array([[1/16, 1/2, -.5], [1/16, 1/2, .5]])
    result = bez_csx(curve, _folded_plane(.25), atol=1e-6, rational=False)
    actual = sorted((p['t'], p['u'], p['v']) for p in result['isolated'])
    np.testing.assert_allclose(actual, [[.5, 0, .5], [.5, .5, .5]], atol=1e-9)
    assert not result['budget_exhausted']
    assert result['boundary_topology_complete']


def test_csx_small_curve_span_keeps_unresolved_surface_parameters():
    curve = np.array([[1/16, 1/2, -1e-6], [1/16, 1/2, 1e-6]])
    result = bez_csx(curve, _folded_plane(), atol=1e-6, rational=False)
    actual = sorted((p['t'], p['u'], p['v']) for p in result['isolated'])
    np.testing.assert_allclose(actual, [[.5, .25, .5], [.5, .75, .5]], atol=1e-9)
    assert not result['budget_exhausted']
    assert result['boundary_topology_complete']


def test_ccx_exact_roots_keep_distinct_parameter_preimages():
    # A(u)=((u-.25)*(u-.75),0,0), B(v)=(0,v-.5,0).
    a = np.array([[3/16, 0, 0], [-5/16, 0, 0], [3/16, 0, 0]])
    b = np.array([[0, -.5, 0], [0, .5, 0]])
    result = bez_ccx(a, b, atol=1e-6, tolerance_tier=False)
    actual = sorted((p['u'], p['v']) for p in result['isolated'])
    np.testing.assert_allclose(actual, [[.25, .5], [.75, .5]], atol=1e-9)
    assert not result['budget_exhausted']
    assert result['boundary_topology_complete']


def _line(points, weights=None):
    return NURBSCurveTuple(
        order=2, knot=np.array([0., 0., 1., 1.]),
        control_points=np.asarray(points, dtype=float),
        weights=np.ones(2) if weights is None else np.asarray(weights),
    )


def _plane(weights=None):
    return NURBSSurfaceTuple(
        order_u=2, order_v=2,
        knot_u=np.array([0., 0., 1., 1.]),
        knot_v=np.array([0., 0., 1., 1.]),
        control_points=np.array([[[0., 0., 0.], [0., 1., 0.]],
                                 [[1., 0., 0.], [1., 1., 0.]]]),
        weights=np.ones((2, 2)) if weights is None else np.asarray(weights),
    )


def test_nurbs_ccx_preserves_near_unit_weights():
    a = _line([[0., 0., 0.], [1., 0., 0.]], [1., 1.000001])
    b = _line([[.5, -1., 0.], [.5, 1., 0.]])
    isolated, _, status = nurbs_ccx(a, b, tol=1e-9)
    assert status['complete']
    assert len(isolated) == 1
    assert isolated[0]['u'] == pytest.approx(1/2.000001, abs=1e-12)


def test_nurbs_csx_preserves_near_unit_curve_weights():
    curve = _line([[.5, .5, -1.], [.5, .5, 1.]], [1., 1.000001])
    isolated, _, status = nurbs_csx(curve, _plane(), tol=1e-9)
    assert status['complete']
    assert len(isolated) == 1
    assert isolated[0]['t'] == pytest.approx(1/2.000001, abs=1e-12)


def test_nurbs_csx_preserves_near_unit_surface_weights():
    curve = _line([[.5, .5, -1.], [.5, .5, 1.]])
    surface = _plane([[1., 1.], [1.000001, 1.000001]])
    isolated, _, status = nurbs_csx(curve, surface, tol=1e-9)
    assert status['complete']
    assert len(isolated) == 1
    assert isolated[0]['u'] == pytest.approx(1/2.000001, abs=1e-12)


def test_csx_exact_topology_omits_nonzero_tolerance_overlap():
    plane = _plane().control_points
    curve = np.array([[.25, 0., -1/65536], [.25, 1., -1/65536]])
    tolerant = bez_csx(curve, plane, atol=1e-3, rational=False)
    assert tolerant['overlaps']
    exact = bez_csx(curve, plane, atol=1e-3, rational=False, tolerance_tier=False)
    assert exact['isolated'] == []
    assert exact['overlaps'] == []
    assert not exact['budget_exhausted']
    assert exact['boundary_topology_complete']


def test_csx_exact_topology_keeps_affine_overlap():
    curve = np.array([[.25, 0., 0.], [.25, 1., 0.]])
    result = bez_csx(curve, _plane().control_points, atol=1e-3,
                     rational=False, tolerance_tier=False)
    assert len(result['overlaps']) == 1
    assert result['overlaps'][0]['certification'] == 'exact'
    assert not result['budget_exhausted']


def test_csx_exact_topology_does_not_cut_out_a_second_close_root():
    h = 1/16384
    curve = np.array([[0., .5, .25-h*h], [.5, .5, -.25-h*h],
                      [1., .5, .25-h*h]])
    result = bez_csx(curve, _plane().control_points, atol=1e-3,
                     rational=False, tolerance_tier=False)
    assert not result['budget_exhausted']
    assert result['boundary_topology_complete']
    roots = sorted(p['t'] for p in result['isolated'])
    np.testing.assert_allclose(roots, [.5-h, .5+h], atol=1e-9)


def test_csx_clipped_root_cutout_removes_the_certified_box():
    from mmcore.numeric.intersection.csx._bez_csx4 import _cutout_3d
    children = _cutout_3d(
        np.zeros((3, 3, 3)), np.zeros((2, 2, 2, 3)),
        np.array([[0., 0., 0.], [1., 0., 0.]]), np.ones(2), np.ones((2, 2)),
        0., 1., 0., 1., 0., 1., 0,
        0., .5, .5, .1, .1, .1, False)
    volume = sum((c[6]-c[5])*(c[8]-c[7])*(c[10]-c[9]) for c in children)
    assert volume == pytest.approx(1.0 - .1*.2*.2, abs=1e-14)


def test_csx_cutout_keeps_representable_small_complement_intervals():
    from mmcore.numeric.intersection.csx._bez_csx4 import _split_intervals
    intervals = _split_intervals(1e-15, 0., 1e-14, 5e-16)
    assert intervals[0][0] == 0.
    assert intervals[-1][-1] == 1e-14
    assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))


def test_csx_curve_restriction_preserves_small_parameter_trims():
    from mmcore.numeric.intersection.csx._bez_csx4 import _restrict_curve
    restricted = _restrict_curve(np.array([[0., 0., 0.], [1., 1., 1.]]),
                                 1e-13, 2e-13)
    np.testing.assert_allclose(restricted, [[1e-13]*3, [2e-13]*3], atol=1e-28)


def test_csx_residual_exclusion_carries_restriction_roundoff():
    from fractions import Fraction
    from mmcore.numeric._bezier_common import restrict_net_axis_v
    from mmcore.numeric.intersection.csx._bez_csx4 import (
        _residual_excludes_zero, _residual_aligned_excludes_zero,
        _csx_residual_roundoff_bound,
    )
    coefficients = np.array([-.1919081550763977, .7911794353014514,
                             -.685355886898228, -.12556249013327248,
                             1.64157569579305])
    # The binary coefficients have an exact double root at t=1/2.
    assert sum(Fraction(float(c))*n for c, n in zip(
        coefficients, (1, 4, 6, 4, 1))) == 0
    net = np.zeros((5, 2, 2, 3))
    net[..., 0] = coefficients[:, None, None]
    net[:, :, :, 1] = np.array([-.5, .5])[None, :, None]
    net[:, :, :, 2] = np.array([-.5, .5])[None, None, :]
    cell = net
    for axis in range(3):
        cell = restrict_net_axis_v(cell, axis, .5-2**-32, .5+2**-32, 0., 1.)
    error = _csx_residual_roundoff_bound(net, depth=1)
    assert not _residual_excludes_zero(cell, roundoff=error)
    assert not _residual_aligned_excludes_zero(cell, roundoff=error)


def test_csx_residual_separation_uses_geometry_normal_with_tangential_drift():
    from mmcore.numeric.intersection.csx._bez_csx4 import (
        _residual_vec_net, _residual_aligned_excludes_zero,
        _csx_residual_roundoff_bound,
    )
    surface = np.array([[[0., 0., 0.], [0., 1., 1.]],
                        [[1., 0., 1.], [1., 1., 2.]]])
    curve = np.array([[.25, .25, .5+1/65536], [.5, .5, 1.+1/65536]])
    net = _residual_vec_net(curve, surface, rational=False)
    assert _residual_aligned_excludes_zero(
        net, _csx_residual_roundoff_bound(net))


def test_csx_exact_topology_does_not_prune_roots_from_rounded_squared_net():
    curve = np.array([[1/16, 1/2, -1e-6], [1/16, 1/2, 1e-6]])
    result = bez_csx(curve, _folded_plane(), atol=1e-12, rational=False,
                     tolerance_tier=False, max_cells=10000)
    roots = sorted((p['t'], p['u'], p['v']) for p in result['isolated'])
    np.testing.assert_allclose(roots, [[.5, .25, .5], [.5, .75, .5]], atol=1e-9)
    assert not result['budget_exhausted']


def test_csx_exact_topology_does_not_infer_overlap_from_multiple_root_samples():
    curve = np.column_stack((np.zeros(11), np.linspace(0., 1., 11),
                             np.array([(-1.)**i/1024 for i in range(11)])))
    plane = np.array([[[-1., 0., 0.], [-1., 1., 0.]],
                      [[1., 0., 0.], [1., 1., 0.]]])
    result = bez_csx(curve, plane, atol=1e-3, rational=False,
                     tolerance_tier=False, max_depth=20, max_cells=2000)
    assert 'uncertified_overlap_span' not in result
    assert result['overlaps'] == []
    assert not result['budget_exhausted']
    assert len(result['isolated']) == 1
    assert result['isolated'][0]['t'] == .5


def test_csx_root_cut_intervals_stay_in_the_cell_for_outside_newton_root():
    from mmcore.numeric.intersection.csx._bez_csx4 import _split_intervals
    intervals = _split_intervals(-1e-12, 0., 1e-13, 1e-14)
    assert all(0. <= lo <= hi <= 1e-13 for lo, hi in intervals)
    assert sum(hi-lo for lo, hi in intervals) == pytest.approx(1e-13, abs=1e-28)


def test_csx_uniqueness_does_not_publish_nonexistent_endpoint_root():
    surface = np.array([[[u, v, u*v-.25] for v in (0., 1.)] for u in (0., 1.)])
    curve = np.array([[.5, .5, 2.**-60], [.5, .5, 1.]])
    result = bez_csx(curve, surface, rational=False, tolerance_tier=False,
                     max_cells=2000, max_depth=30)
    assert result['isolated'] == [], result
    assert not result['boundary_topology_complete'] or not result['budget_exhausted']


def test_ccx_uniqueness_does_not_publish_nonexistent_endpoint_root():
    a = np.array([[0., 0., 0.], [1., 0., 0.]])
    b = np.array([[0., 2.**-60, 0.], [.5, .5, 0.], [1., 1., 0.]])
    result = bez_ccx(a, b, rational=False, tolerance_tier=False,
                     max_cells=2000, max_depth=30)
    assert result['isolated'] == [], result
    assert not result['boundary_topology_complete'] or not result['budget_exhausted']


def test_ccx_affine_overlap_does_not_discharge_a_second_curve_preimage():
    curve = np.array([[.25, 0., 0.], [-.25, 0., 0.], [.25, 0., 0.]])
    result = bez_ccx(curve, curve, rational=False, tolerance_tier=False,
                     max_cells=2000)
    assert result['overlaps']
    assert not result['boundary_topology_complete'], result
    assert any(box['reason'] == 'curve_preimage_uniqueness'
               for box in result['unresolved_parameter_boxes'])


def test_csx_generic_affine_near_identity_is_not_an_exact_overlap():
    surface = np.array([[[u, v, u*v] for v in (0., 1.)] for u in (0., 1.)])
    curve = np.array([[0., .5, 2.**-60], [1., .5, .5]])
    result = bez_csx(curve, surface, rational=False, tolerance_tier=False,
                     max_cells=2000, max_depth=30)
    assert result['overlaps'] == [], result


def test_ccx_exact_mode_rejects_a_tolerance_only_overlap():
    curve = np.array([[0., 0., 0.], [.5, .5, 0.], [1., 1., 0.]])
    result = bez_ccx(curve, curve+[0., 0., 2.**-60], rational=False,
                     tolerance_tier=False, max_cells=2000, max_depth=30)
    assert result['overlaps'] == [], result


def test_csx_preserves_exact_boundary_curve_line_existence_proof():
    surface = np.array([[[u, v, u*v] for v in (0., 1.)] for u in (0., 1.)])
    curve = np.array([[0., 0., -.5], [.5, 0., -.5], [1., 0., .5]])
    result = bez_csx(curve, surface, rational=False, tolerance_tier=False,
                     max_cells=2000, max_depth=30)
    assert result['boundary_topology_complete'], result
    assert not result['budget_exhausted'], result
    root, = result['isolated']
    assert root['t'] == pytest.approx(np.sqrt(.5))
    assert root['u'] == pytest.approx(np.sqrt(.5))
    assert root['v'] == 0.


def test_csx_exact_affine_overlap_does_not_discharge_other_surface_sheets():
    curve = np.array([[1/16, 0., 0.], [1/16, 1., 0.]])
    result = bez_csx(curve, _folded_plane(), atol=1e-3, rational=False,
                     tolerance_tier=False, max_cells=10000)
    # Two exact correspondences exist: u=.25,v=t and u=.75,v=t.
    # A supported enumeration may return both; otherwise it must preserve
    # the known overlap as a witness and leave the complement unresolved.
    assert result['overlaps']
    complete = not result['budget_exhausted'] and result['boundary_topology_complete']
    assert len(result['overlaps']) >= 2 or not complete


def test_ccx_exact_close_root_cutout_requires_uniqueness_or_partial_status():
    h = 1/16384
    a = np.array([[0., .25-h*h, 0.], [.5, -.25-h*h, 0.],
                  [1., .25-h*h, 0.]])
    b = np.array([[0., 0., 0.], [1., 0., 0.]])
    result = bez_ccx(a, b, atol=1e-3, tolerance_tier=False)
    complete = not result['budget_exhausted'] and result['boundary_topology_complete']
    assert len(result['isolated']) >= 2 or not complete


@pytest.mark.parametrize('tolerance_tier', [False, True])
def test_ccx_boundary_root_preserves_other_partner_parameter(tolerance_tier):
    # Both curves are quadratic: no exact curve-line shortcut applies.
    # A=(u,u²,0), B=(.5,.25+v(v-.75),0).
    a = np.array([[0., 0., 0.], [.5, 0., 0.], [1., 1., 0.]])
    b = np.array([[.5, .25, 0.], [.5, -.125, 0.], [.5, .5, 0.]])
    result = bez_ccx(a, b, atol=1e-3, max_cells=5000,
                     tolerance_tier=tolerance_tier)
    actual = sorted((p['u'], p['v']) for p in result['isolated'])
    np.testing.assert_allclose(actual, [[.5, 0.], [.5, .75]], atol=1e-11)
    assert all(p['certification'] == 'exact' for p in result['isolated'])
    assert result['boundary_topology_complete'], result


def test_generic_ccx_cancellation_does_not_use_positive_squared_coefficients():
    from mmcore.numeric.bern_sq_dist import curve_curve_squared_net_homog
    from mmcore.numeric._bezier_common import restrict_net_axis
    # Integer source coefficients guarantee A(.5)=B(.5)=0 exactly.
    # Squared Bernstein convolution/restriction nevertheless drifts positive
    # on a closed box containing this ordinary transverse intersection.
    a = np.array([[-5., -19., 0.], [21., 18., 0.], [-37., -17., 0.]])*2**20
    b = np.array([[-29., 12., 0.], [-25., 14., 0.], [79., -40., 0.]])*2**20
    f = curve_curve_squared_net_homog(a, b, rational=False)
    for axis in (0, 1):
        f = restrict_net_axis(f, axis, .5, .5+2**-25, 0., 1.)
    assert f.min() > 1e-6
    result = bez_ccx(a, b, atol=1e-3, tolerance_tier=False, max_cells=5000)
    assert result['boundary_topology_complete'] and not result['budget_exhausted'], result
    assert len(result['isolated']) == 2
    assert any(abs(p['u']-.5) < 1e-12 and abs(p['v']-.5) < 1e-12
               for p in result['isolated'])
