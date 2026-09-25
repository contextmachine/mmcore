"""Whole-chart spherical CAD reduction, including ineligible lookalikes."""
import numpy as np
import pytest

from mmcore.numeric._bezier_common import eval_surface
from mmcore.numeric._work_budget import SoftWorkBudget
from mmcore.numeric.intersection.ssx import _ssx_spherical_overlap as spherical
from mmcore.numeric.intersection.ssx import _ssx5_overlap as overlap
from test_ssx_spherical_overlap import (
    ATOL, _octant, _rotation, _reference, _homogeneous, _assert_region,
)


def _budget(**kwargs):
    return SoftWorkBudget(max_cells=kwargs.pop('max_cells', 60000),
                         max_csx_calls=1000, max_output_items=kwargs.pop('max_output_items', 1024),
                         **kwargs)


def _pair(radius=1., center=(0., 0., 0.)):
    frames = (np.eye(3), _rotation([1., 2., 3.], .35))
    return tuple(_octant(frame, radius, center) for frame in frames), frames


def test_case17_direct_reduction_returns_all_four_actual_rims():
    from examples.ssx.case_17 import s1, s2
    from test_ssx_spherical_overlap import _fit_octant_frame
    surfaces = (s1, s2)
    fits = [_fit_octant_frame(surface) for surface in surfaces]
    center = .5*(fits[0][0]+fits[1][0])
    radius = .5*(fits[0][1]+fits[1][1])
    result = spherical.try_spherical_overlap(*map(_homogeneous, surfaces), ATOL, _budget())
    assert result is not None
    assert result._c1_resolved and result._c3_resolved
    assert '_c1_resolved' not in result and '_c3_resolved' not in result
    _assert_region(result, surfaces, _reference(center, radius, [fit[2] for fit in fits]))


@pytest.mark.parametrize('transform', ['identity', 'reverse', 'transpose', 'gauge'])
def test_rotated_scaled_charts_preserve_complete_footprint(transform):
    surfaces, frames = _pair(radius=2.5, center=(20., -10., 5.))
    first, second = map(_homogeneous, surfaces)
    if transform == 'reverse':
        first = first[::-1].copy()
    elif transform == 'transpose':
        first = first.swapaxes(0, 1).copy()
    elif transform == 'gauge':
        first = first*1e-7
        second = second*13.
    budget = _budget()
    result = spherical.try_spherical_overlap(first, second, ATOL, budget)
    assert result is not None and len(result['overlap_regions']) == 1
    assert result._c1_resolved and result._c3_resolved
    assert budget.result_fields()['complete']
    # Coverage oracle uses geometry, independent of the altered parameter chart.
    from examples.ssx.ssx5_analytic_audit import distances_to_polylines
    reference = _reference(np.array([20., -10., 5.]), 2.5, frames)
    paths = [np.asarray(branch.curve[1]) for branch in result['branches']]
    for arc in reference['arcs']:
        assert distances_to_polylines(arc, paths).max() <= ATOL
    for branch in result['branches']:
        for q, xyz in zip(*branch.curve):
            assert np.linalg.norm(eval_surface(first, *q[:2], rational=True)-xyz) <= ATOL
            assert np.linalg.norm(eval_surface(second, *q[2:], rational=True)-xyz) <= ATOL
    assert result['overlap_regions'][0].normal_agreement == (-1 if transform in ('reverse', 'transpose') else 1)


def test_small_control_and_weight_perturbations_are_judged_at_cad_accuracy():
    surfaces, frames = _pair()
    first, second = map(_homogeneous, surfaces)
    first[1, 1, :3] += first[1, 1, 3]*np.array([1e-5, -1e-5, .5e-5])
    second[1, 1] *= 1.+1e-5
    budget = _budget()
    result = spherical.try_spherical_overlap(first, second, ATOL, budget)
    assert result is not None and len(result['overlap_regions']) == 1
    assert budget.result_fields()['complete']


def test_hidden_interior_bulge_cannot_pass_matching_boundary_and_witness():
    first = _homogeneous(_octant(np.eye(3)))
    second = first.copy()
    point = eval_surface(first, .5, .5, rational=True)
    second[1, 1] += 2.*np.r_[point, 1.]
    # Boundaries and the central point coincide; the interior does not.
    assert np.linalg.norm(eval_surface(first, .5, .5, rational=True)
                          - eval_surface(second, .5, .5, rational=True)) <= ATOL
    assert abs(np.linalg.norm(eval_surface(second, .25, .25, rational=True))-1.) > ATOL
    budget = _budget()
    assert spherical.try_spherical_overlap(first, second, ATOL, budget) is None
    assert not budget.exhausted


def test_same_cartesian_controls_with_different_weights_need_whole_map_check():
    first = _homogeneous(_octant(np.eye(3)))
    second = first.copy()
    second[1, 1] *= 1.2
    np.testing.assert_allclose(first[..., :3]/first[..., 3, None],
                               second[..., :3]/second[..., 3, None])
    assert spherical.try_spherical_overlap(first, second, ATOL, _budget()) is None


def test_radius_gap_and_unsupported_shape_decline_without_absence_claim():
    first = _homogeneous(_octant(np.eye(3)))
    second = _homogeneous(_octant(np.eye(3), radius=1.+4*ATOL))
    assert spherical.try_spherical_overlap(first, second, ATOL, _budget()) is None
    higher_degree_shape = np.concatenate((first, first[-1:]), axis=0)
    assert spherical.try_spherical_overlap(first, higher_degree_shape, ATOL, _budget()) is None


def test_curve_only_and_disjoint_footprints_stay_with_general_ssx():
    first = _homogeneous(_octant(np.eye(3)))
    for angle in (np.pi/2, np.pi/2+.02):
        second = _homogeneous(_octant(_rotation([0., 0., 1.], angle)))
        assert spherical.try_spherical_overlap(first, second, ATOL, _budget()) is None


def test_output_denial_keeps_rims_without_dangling_region_references():
    surfaces, _ = _pair()
    budget = _budget(max_output_items=2)
    result = spherical.try_spherical_overlap(*map(_homogeneous, surfaces), ATOL, budget)
    assert result is not None and len(result['branches']) == 2
    assert result['overlap_regions'] == []
    assert 'output_cap' in budget.result_fields()['status']['reasons']


def test_work_denial_during_assembly_retains_completed_validated_rim(monkeypatch):
    surfaces, _ = _pair()
    budget = _budget()
    original = overlap._adaptive_curved_rim
    completed = []
    def stop_after_first(*args, **kwargs):
        result = original(*args, **kwargs)
        if result is not None and not completed:
            completed.append(result)
            budget.max_cells = budget.cells_processed
        return result
    monkeypatch.setattr(overlap, '_adaptive_curved_rim', stop_after_first)
    result = spherical.try_spherical_overlap(*map(_homogeneous, surfaces), ATOL, budget)
    assert budget.exhausted and result is not None
    assert len(result['branches']) == 1 and result['overlap_regions'] == []
    np.testing.assert_array_equal(result['branches'][0].curve[1], completed[0]['xyz'])


def test_setup_denial_leaves_the_general_search_stopped_by_its_budget():
    surfaces, _ = _pair()
    budget = _budget(max_cells=0)
    assert spherical.try_spherical_overlap(*map(_homogeneous, surfaces), ATOL, budget) is None
    assert budget.exhausted and budget.cells_processed == 0


@pytest.mark.parametrize('angle', [0., np.pi/2-.02])
def test_identical_and_thin_octants_have_the_complete_boundary(angle):
    frames = (np.eye(3), _rotation([0., 0., 1.], angle))
    surfaces = tuple(_octant(frame) for frame in frames)
    budget = _budget()
    result = spherical.try_spherical_overlap(*map(_homogeneous, surfaces), ATOL, budget)
    assert result is not None and budget.result_fields()['complete']
    assert result._c1_resolved and result._c3_resolved
    _assert_region(result, surfaces, _reference(np.zeros(3), 1., frames))


def test_singularity_guard_work_denial_preserves_the_validated_region(monkeypatch):
    surfaces, _ = _pair()
    budget = _budget()
    original = spherical._resolve_spherical_c1
    def deny_guard(sources, charts, result, atol, work_budget):
        work_budget.max_cells = work_budget.cells_processed
        return original(sources, charts, result, atol, work_budget)
    monkeypatch.setattr(spherical, '_resolve_spherical_c1', deny_guard)
    result = spherical.try_spherical_overlap(*map(_homogeneous, surfaces), ATOL, budget)
    assert result is not None and len(result['overlap_regions']) == 1
    assert len(result['branches']) == 6
    assert result._c3_resolved and not result._c1_resolved
    assert budget.exhausted
    assert 'work_budget' in budget.result_fields()['status']['reasons']


def test_rank_defect_outside_the_pole_cap_keeps_c1_enabled():
    surface = _homogeneous(_octant(np.eye(3)))
    budget = _budget()
    chart = spherical._propose_chart(surface, budget)
    result = spherical.try_spherical_overlap(surface, surface.copy(), ATOL, budget)
    assert result is not None and result._c1_resolved
    folded = surface.copy()
    folded[-1] = folded[0]
    # This actual biquadratic map has S_u=0 on u=.5, including the
    # equatorial edge far from the pole. It still has a collapsed pole.
    from mmcore.numeric._bezier_common import eval_surface_d1
    _, derivative_u, _ = eval_surface_d1(folded, .5, 0., rational=True)
    assert np.linalg.norm(derivative_u) <= np.finfo(float).eps
    assert not spherical._resolve_spherical_c1(
        (folded, surface), (chart, chart), result, ATOL, _budget())


def test_unrepresented_pole_cap_on_the_other_surface_keeps_c1_enabled():
    surface = _homogeneous(_octant(np.eye(3)))
    chart = spherical._propose_chart(surface, _budget())
    # An unrelated corner cannot represent this pole, and the identical
    # other patch provides no separating support direction.
    from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch
    result = dict(overlap_regions=[object()], branches=[SSXBranch(curve=(
        np.zeros((2, 4)), np.array([[1., 0., 0.], [0., 1., 0.]])))])
    assert not spherical._resolve_spherical_c1(
        (surface, surface), (chart, chart), result, ATOL, _budget())


def test_nonspherical_proposal_does_not_spend_the_general_solver_allowance():
    first = np.array([[[i/2, j/2, 0., 1.] for j in range(3)] for i in range(3)])
    second = np.array([[[i/2, .5, j/2-.5, 1.] for j in range(3)] for i in range(3)])
    budget = _budget(max_cells=1000)
    assert spherical.try_spherical_overlap(first, second, ATOL, budget) is None
    assert not budget.exhausted
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
    result = bez_ssx(first, second, ATOL, rational=True, max_cells=1000)
    assert result['complete'] and len(result['branches']) == 1
    xyz = np.asarray(result['branches'][0].curve[1])
    assert abs(xyz[:, 0].min()) <= ATOL and abs(xyz[:, 0].max()-1.) <= ATOL


def test_thin_perturbed_footprint_declines_when_chart_error_is_amplified(monkeypatch):
    # Twenty-atol corner separation is resolved for the unperturbed charts
    # above. A small interior perturbation can nevertheless make the common
    # carrier's angular footprint uncertain at a nearly opposed pair of faces.
    first = _homogeneous(_octant(np.eye(3)))
    second = _homogeneous(_octant(_rotation([0., 0., 1.], np.pi/2-.02)))
    first[1, 1, 2] += first[1, 1, 3]*1e-5
    def forbidden(*args, **kwargs):
        raise AssertionError('uncertain footprint entered region construction')
    monkeypatch.setattr(spherical, 'assemble_overlap_regions', forbidden)
    budget = _budget()
    assert spherical.try_spherical_overlap(first, second, ATOL, budget) is None
    assert not budget.exhausted
    assert budget.cell_counts.get('spherical_bound', 0) > 0


@pytest.mark.parametrize('angle', [0., np.pi/2-.02])
def test_public_region_does_not_retype_shared_pole_parameters_as_singularities(angle):
    from mmcore.numeric.intersection.ssx import nurbs_ssx
    from test_ssx_spherical_overlap import _assert_region, _reference

    frames = [np.eye(3), _rotation([0., 0., 1.], angle)]
    surfaces = tuple(_octant(frame) for frame in frames)
    result = nurbs_ssx(*surfaces, atol=ATOL)
    _assert_region(result, surfaces, _reference([0., 0., 0.], 1., frames))
    assert result['complete'] and result['status']['reasons'] == []
    assert result['points'] == [] and result['singularities'] == []
