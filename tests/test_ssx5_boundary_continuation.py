"""Spatial continuation at a collapsed isoline, across parameter charts."""
import numpy as np
import pytest

from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
from examples.ssx.bez_ssx5_coverage_check import point_to_polyline_dist


def _pinched_pair():
    anchor = np.array([.5, .5, 0.])
    heights = np.array([.15, -.45, .15])
    first = np.column_stack((np.zeros(3), [0., .5, 1.], heights))
    last = np.column_stack((np.ones(3), [0., .5, 1.], -heights))
    middle = 2.*anchor-.5*(first+last)
    surface = np.stack((first, middle, last))
    plane = np.array([[[-1., -1., 0.], [-1., 2., 0.]],
                      [[2., -1., 0.], [2., 2., 0.]]])
    return surface, plane


@pytest.mark.parametrize('variant', ['swap', 'transpose', 'reverse', 'rotate'])
def test_pinched_incident_branches_keep_both_sides_in_changed_charts(variant):
    first, second = _pinched_pair()
    rotation = np.eye(3)
    if variant == 'swap':
        first, second = second, first
    elif variant == 'transpose':
        first, second = first.swapaxes(0, 1).copy(), second.swapaxes(0, 1).copy()
    elif variant == 'reverse':
        first, second = first[::-1, ::-1].copy(), second[::-1].copy()
    elif variant == 'rotate':
        rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        first, second = first@rotation.T, second@rotation.T

    result = bez_ssx(first, second, atol=1e-3, rational=False)
    assert 'cusp_curve' in [item.kind for item in result['singularities']]
    assert not {'work_budget', 'output_cap', 'depth_limit'} & set(result['status']['reasons'])
    polylines = [branch.curve[1]@rotation for branch in result['branches']]
    assert len(polylines) == 2
    parameter = np.linspace(0., 1., 101)
    for t in (.5-np.sqrt(.5)/2., .5+np.sqrt(.5)/2.):
        truth = np.column_stack((parameter,
                                 .5+(t-.5)*(1.-2.*parameter)**2,
                                 np.zeros_like(parameter)))
        distance = [min(point_to_polyline_dist(point, polyline)
                        for polyline in polylines) for point in truth]
        assert max(distance) <= 5e-3


def _general_boundary_pair():
    # y=0 is a whole intersection curve for x in [.1,1]. Raw face CSX
    # also observes many interior points of that same curved boundary.
    from math import comb
    power = np.array([-.0001, .0001, .01, -.01])
    height = np.array([sum(power[j]*comb(i, j)/comb(3, j)
                           for j in range(i+1)) for i in range(4)])
    first = np.array([[[t, s, z] for t, z in zip(np.linspace(0., 1., 4), height)]
                      for s in (0., 1.)])
    second = np.array([[[u, 0., v] for v in (0., 1.)] for u in (0., 1.)])
    return first, second


def test_general_boundary_curve_consumes_interior_registrations_once():
    first, second = _general_boundary_pair()
    result = bez_ssx(first, second, .001, rational=False,
                     max_cells=30000, max_csx_calls=500, max_xyz_step=.1)
    assert len(result['branches']) == 1
    branch = result['branches'][0]
    xyz = np.asarray(branch.curve[1])
    assert abs(xyz[:, 0].min()-.1) <= .001
    assert abs(xyz[:, 0].max()-1.) <= .001
    for x in np.linspace(.1, 1., 101):
        true_point = np.array([x, 0., .01*(1.-x)*(x*x-.01)])
        assert point_to_polyline_dist(true_point, xyz) <= .001
    assert not {'postprocess_cap', 'output_cap'} & set(result['status']['reasons'])


@pytest.mark.parametrize('boundary_cap, resource_stop', [(100, True), (20000, False)])
def test_boundary_overlap_fallback_retains_actual_stop_reason(boundary_cap, resource_stop):
    first, second = _general_boundary_pair()
    result = bez_ssx(first, second, .001, rational=False,
                     max_cells=30000, max_csx_calls=500, max_xyz_step=.1,
                     boundary_csx_max_cells=boundary_cap)
    assert ('work_budget' in result['status']['reasons']) is resource_stop
    assert result['complete'] is False
    if not resource_stop:
        assert 'overlap_region_unsupported' in result['status']['reasons']
        assert len(result['branches']) == 1
    assert result['status']['work']['cells_processed'] <= 30000
