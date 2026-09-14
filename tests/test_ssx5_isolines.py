"""Numerical full-isoline coverage and clipping before the 4D search."""
import numpy as np
import pytest

from mmcore.numeric._work_budget import SoftWorkBudget
from mmcore.numeric._bezier_common import eval_surface
from mmcore.numeric.intersection.ssx._ssx_isolines import try_isoline_ssx


def _h(points, weight=1.):
    return np.concatenate((np.asarray(points)*weight,
                           np.full(np.shape(points)[:2]+(1,),weight)),axis=2)


def _plane():
    return _h(np.array([[[s,t,0.] for t in (0.,1.)] for s in (0.,1.)]))


def _solve(first, second, atol=1e-3, cells=10000, outputs=100):
    budget=SoftWorkBudget(cells,100,max_output_items=outputs)
    result=try_isoline_ssx(first,second,atol,budget)
    return result,budget


@pytest.mark.parametrize('degree',[2,4,8,10,12])
def test_multiple_root_yields_one_complete_ruling(degree):
    graph=_h(np.array([[[s,j/degree,(-1.)**(degree-j)*2.**-degree]
                         for j in range(degree+1)] for s in (0.,1.)]))
    result,budget=_solve(graph,_plane(),atol=min(1e-3,.1*2.**-degree))
    assert result is not None and not budget.incomplete
    assert len(result['branches'])==1
    branch=result['branches'][0]
    assert branch.kind=='tangential'
    np.testing.assert_allclose(branch.curve[1],[[0.,.5,0.],[1.,.5,0.]],atol=1e-10)


@pytest.mark.parametrize('same_image',[False,True])
def test_distinct_rulings_remain_separate_preimages(same_image):
    graph=_h(np.array([[[s,.5 if same_image else y,z]
                       for y,z in zip((0.,.5,1.),(3/16,-5/16,3/16))] for s in (0.,1.)]))
    result,budget=_solve(graph,_plane())
    assert result is not None and not budget.incomplete
    assert len(result['branches'])==2
    assert [branch.curve[0][0,1] for branch in result['branches']]==pytest.approx([.25,.75])


def test_nonbinary_ruling_is_clipped_at_both_plane_sides():
    graph=_h(np.array([[[s,y,z] for y,z in zip((0.,.5,1.),(1.,-2.,4.))]
                       for s in (-1.,2.)]))
    result,budget=_solve(graph,_plane())
    assert result is not None and not budget.incomplete
    assert len(result['branches'])==1
    np.testing.assert_allclose(result['branches'][0].curve[1],
                               [[0.,1/3,0.],[1.,1/3,0.]],atol=1e-11)


@pytest.mark.parametrize('swap',[False,True])
@pytest.mark.parametrize('transpose',[False,True])
def test_cusp_ruling_keeps_full_parameter_fiber_and_owner(swap,transpose):
    graph=_h(np.array([[[x,y,float(t)] for t in (0,1)]
                        for x,y in zip((3.,-1.,-1.,3.),(-1.,1.,-1.,1.))]),2.)
    plane=_h(np.array([[[0.,y,z] for z in (-.5,1.5)] for y in (-1.5,1.5)]),.5)
    if transpose:graph=graph.swapaxes(0,1)
    result,budget=_solve(*((plane,graph) if swap else (graph,plane)))
    assert result is not None and not budget.incomplete
    singular,=result['singularities']
    assert singular.kind=='cusp_curve' and singular.surface==(2 if swap else 1)
    offset=2 if swap else 0
    assert singular.samples[:,offset+(0 if transpose else 1)].tolist()==[0.,1.]


def test_curved_isoline_chords_meet_both_source_accuracy():
    graph=np.zeros((3,4,3))
    graph[...,0]=np.array([0.,1.,1.,0.])[None,:]+np.array([0.,.5,1.])[:,None]
    graph[...,1]=np.array([0.,1/3,2/3,1.])[None,:]
    graph[...,2]=np.array([.25,-.25,.25])[:,None]
    plane=_h(np.array([[[s,t,0.] for t in (0.,1.)] for s in (0.,2.)]))
    graph=_h(graph)
    result,budget=_solve(graph,plane)
    assert result is not None and not budget.incomplete
    branch,=result['branches']
    q,xyz=branch.curve
    assert len(q)>2
    for i in range(len(q)-1):
        for t in np.linspace(0.,1.,9):
            parameters=(1-t)*q[i]+t*q[i+1]
            point=(1-t)*xyz[i]+t*xyz[i+1]
            for surface,uv in ((graph,parameters[:2]),(plane,parameters[2:])):
                assert np.linalg.norm(eval_surface(surface,*uv,rational=True)-point)<=1e-3


def test_unsupported_nonuniform_weights_fall_back():
    first=_plane()
    first[1,1]*=2.
    assert _solve(first,_plane())[0] is None


def test_denied_root_search_does_not_claim_an_empty_complete_set():
    graph=_h(np.array([[[s,y,z] for y,z in zip((0.,.5,1.),(3/16,-5/16,3/16))]
                       for s in (0.,1.)]))
    result,budget=_solve(graph,_plane(),cells=10)
    assert result is None and budget.exhausted


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('scale', [.25, 8.])
def test_stretched_chart_root_location_uses_geometric_tolerance(scale, transpose):
    c = .06427065884628964
    stretch = 1e12
    graph = np.array([[[stretch*s, t, z] for t in (0., 1.)]
                      for s, z in zip((0., .5, 1.), (-c, -c, 1.-c))])
    plane = np.array([[[stretch*s, t, 0.] for t in (0., 1.)]
                      for s in (0., 1.)])
    if transpose:
        graph = graph.swapaxes(0, 1)
    result, budget = _solve(_h(scale*graph, 2.), _h(scale*plane, .5),
                            atol=scale*1e-3)
    assert result is not None and not budget.incomplete
    branch, = result['branches']
    # A fixed 1e-13 root tolerance put this line 0.025 units away,
    # despite passing a tiny surface-height residual check.
    expected_x = scale*stretch*np.sqrt(c)
    assert np.max(np.abs(branch.curve[1][:, 0]-expected_x)) <= scale*1e-3


def test_stretched_free_curve_clipping_uses_geometric_tolerance():
    c, stretch = .06427065884628964, 1e12
    graph = np.array([[[stretch*t, y, s-.5]
                       for t, y in zip((0., .5, 1.), (0., 0., 1.))]
                      for s in (0., 1.)])
    plane = np.array([[[stretch*s, c*t, 0.] for t in (0., 1.)]
                      for s in (0., 1.)])
    result, budget = _solve(_h(graph), _h(plane))
    assert result is not None and not budget.incomplete
    branch, = result['branches']
    assert abs(branch.curve[1][-1, 0]-stretch*np.sqrt(c)) <= 1e-3
    assert branch.curve[0][-1, 3] == 1.


@pytest.mark.parametrize('offset,stretch', [(1e12, 1.), (0., 1e12)])
def test_coordinate_roundoff_allowance_cannot_hide_free_axis_height(offset, stretch):
    graph = np.array([[[offset+stretch*s, t, s+.01-.02*t]
                       for t in (0., 1.)] for s in (0., 1.)])
    plane = np.array([[[offset+stretch*s, t, 0.]
                       for t in (0., 1.)] for s in (0., 1.)])
    # The actual intersection is s=.02*t-.01 for .5 <= t <= 1.
    # Coordinate-scaled noise used to erase t from the height equation
    # and return an empty completed result from the positive s profile.
    assert _solve(_h(graph), _h(plane))[0] is None


def test_subprecision_parameter_request_uses_general_search():
    graph = np.array([[[1e13*s, t, z] for t in (0., 1.)]
                      for s, z in zip((0., .5, 1.), (-.2, -.2, .8))])
    plane = np.array([[[1e13*s, t, 0.] for t in (0., 1.)]
                      for s in (0., 1.)])
    assert _solve(_h(graph), _h(plane))[0] is None


@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('width', [0., .01])
def test_public_collapsed_isoline_is_a_point_and_nearby_ruling_stays_a_curve(swap, width):
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
    # S1=(s-.5, (s-.5+width)*t, (s-.5)^2), intersected with x=0.
    # At width=0 every t maps to the origin. A nonzero width is a real
    # line segment, even though the construction is otherwise identical.
    graph = np.array([[[x, (x+width)*t, z] for t in (0., 1.)]
                      for x, z in zip((-.5, 0., .5), (.25, -.25, .25))])
    plane = np.array([[[0., u, v] for v in (-1., 1.)] for u in (-1., 1.)])
    pair = (plane, graph) if swap else (graph, plane)
    result = bez_ssx(*pair, atol=1e-3, rational=False)
    if width:
        assert result['points'] == []
        branch, = result['branches']
        np.testing.assert_allclose(branch.curve[1][[0, -1]],
                                   [[0., 0., 0.], [0., width, 0.]],
                                   atol=1e-10, rtol=0.)
    else:
        assert result['branches'] == []
        point, = result['points']
        np.testing.assert_allclose(point.xyz, np.zeros(3), atol=1e-10, rtol=0.)
        assert 'parameter_fiber' in result['status']['reasons']
        owner = 2 if swap else 0
        curves = [g for g in result['singularities']
                  if g.kind == 'cusp_curve' and g.surface == (2 if swap else 1)]
        assert curves
        samples = np.concatenate([g.samples for g in curves])
        assert samples[:, owner+1].min() <= 1e-3
        assert samples[:, owner+1].max() >= 1.-1e-3
        assert all(not g.branch_links for g in curves)
