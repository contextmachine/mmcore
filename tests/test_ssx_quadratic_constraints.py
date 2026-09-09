"""Exact quadratic residual reduction preserves complete tangential curves."""
import numpy as np
import pytest

from mmcore.numeric._work_budget import SoftWorkBudget
from mmcore.numeric.intersection.ssx._ssx_quadratic_constraints import exact_quadratic_constraint_ssx


def _pair(root=.5):
    # Common injective (x,y) chart; opposed squared heights touch on s=u=root.
    heights = np.array([root*root, root*root-root, (1-root)**2])
    a = np.array([[[s+3*t*(1-t), t, heights[i]] for t in (0.,1/3,2/3,1.)]
                  for i,s in enumerate((0.,.5,1.))])
    # Exact degree-3 Bernstein coefficients of 3*t*(1-t), not samples.
    a[...,0] = np.array([0.,1.,1.,0.])[None,:]+np.array([0.,.5,1.])[:,None]
    a[...,1] = np.array([0.,1.,2.,3.])[None,:]
    b = a.copy()
    b[...,2] *= -1
    return a,b


def _solve(a,b,atol=1e-3,rational=False,cells=100000,outputs=100):
    budget = SoftWorkBudget(cells,100,max_output_items=outputs)
    result = exact_quadratic_constraint_ssx(a,b,atol,rational,budget)
    if result is not None:
        result.update(budget.result_fields())
    return result,budget


def _isolated_quadratic_pair(root=.5):
    q = [root*root,root*root-root,(1-root)**2]
    other = [.25,-.25,.25]
    a = np.array([[[s,t,q[i]+other[j]] for j,t in enumerate((0.,.5,1.))]
                  for i,s in enumerate((0.,.5,1.))])
    b = np.array([[[s,t,0.] for t in (0.,1.)] for s in (0.,1.)])
    return a,b


@pytest.mark.parametrize('root,swap,rational',[(.5,False,False),(0.,False,False),
                                              (1.,True,False),(.5,True,True)])
def test_zero_dimensional_quadratic_reduction_publishes_exact_tangent_point(root,swap,rational):
    a,b = _isolated_quadratic_pair(root)
    if rational:
        a = np.concatenate((a,np.ones(a.shape[:2]+(1,))),axis=-1)*2.**-300
        b = np.concatenate((b,np.ones(b.shape[:2]+(1,))),axis=-1)*2.**300
    if swap:
        a,b = b,a
    result,_ = _solve(a,b,rational=rational)
    assert result is not None and result['complete']
    point, = result['singularities']
    assert point.kind == 'tangent_point'
    np.testing.assert_array_equal(point.stuv,[root,.5,root,.5])
    np.testing.assert_array_equal(point.xyz,[root,.5,0.])
    assert not result['branches'] and not result['points']


def test_zero_dimensional_quadratic_root_outside_domain_is_complete_empty():
    result,_ = _solve(*_isolated_quadratic_pair(-1.))
    assert result is not None and result['complete']
    assert not any(result[key] for key in ('branches','points','singularities'))


def test_zero_dimensional_quadratic_output_denial_remains_partial():
    result,budget = _solve(*_isolated_quadratic_pair(),outputs=0)
    assert result is not None and not result['complete']
    assert 'output_cap' in result['status']['reasons']
    assert not result['singularities'] and budget.output_items == 0


@pytest.mark.parametrize('swap',[False,True])
def test_zero_dimensional_degenerate_source_is_c1_not_tangent(swap):
    a = np.array([[[x,t,x+y] for t,y in ((0.,0.),(.5,0.),(1.,1.))]
                  for x in (0.,0.,1.)])
    b = np.array([[[s,t,0.] for t in (0.,1.)] for s in (0.,1.)])
    if swap:
        a,b = b,a
    result,_ = _solve(a,b)
    assert result is not None and result['complete']
    feature, = result['singularities']
    assert feature.kind == 'cusp' and feature.surface == (2 if swap else 1)
    np.testing.assert_array_equal(feature.stuv,np.zeros(4))
    assert not result['branches'] and not result['points']


def test_zero_dimensional_world_representation_failure_is_explicit():
    a,b = _isolated_quadratic_pair()
    a[...,0] = np.array([1e16,1e16,1e16+2])[:,None]
    b[...,0] = np.array([1e16,1e16+2])[:,None]
    result,_ = _solve(a,b)
    assert result is not None and not result['complete']
    assert result['status']['reasons'] == ['parameter_representation']
    assert not result['branches'] and not result['points'] and not result['singularities']


@pytest.mark.parametrize('swap,reverse,transpose,rational',[
    (False,False,False,False),(True,False,False,False),
    (False,True,False,False),(False,False,True,False),
    (True,True,True,True)])
def test_factor_then_linear_reduction_preserves_generator_and_collapsed_fiber(
        swap,reverse,transpose,rational):
    a = np.array([[[0.,0.,0.],[0.,0.,0.]],[[0.,1.,-1.],[1.,1.,1.]]])
    b = np.array([[[-1.5,-1.5,0.],[-1.5,1.5,0.]],
                  [[1.5,-1.5,0.],[1.5,1.5,0.]]])
    if reverse:
        a = a[::-1].copy()
    if transpose:
        a = a.swapaxes(0,1).copy()
    if rational:
        a = np.concatenate((a,np.ones(a.shape[:2]+(1,))),axis=-1)*2.**-300
        b = np.concatenate((b,np.ones(b.shape[:2]+(1,))),axis=-1)*2.**300
    if swap:
        a,b = b,a
    result,_ = _solve(a,b,rational=rational)
    assert result is not None and result['complete']
    branch, = result['branches']
    assert branch.kind == 'transversal'
    np.testing.assert_allclose(np.sort(branch.curve[1][:,1]),[0.,1.],atol=1e-14)
    np.testing.assert_allclose(branch.curve[1][:,0],branch.curve[1][:,1]/2,atol=1e-14)
    np.testing.assert_array_equal(branch.curve[1][:,2],0.)
    fiber, = result['singularities']
    assert fiber.kind == 'cusp_curve' and fiber.surface == (2 if swap else 1)
    np.testing.assert_array_equal(fiber.xyz,np.zeros(3))
    assert len(fiber.samples) >= 2 and len(fiber.branch_links) == 1
    owner = 1 if swap else 0
    free_axis = 2*owner+(0 if transpose else 1)
    assert set(fiber.samples[:,free_axis]) == {0.,1.}
    assert not result['points'] and not result['unresolved_regions']


def test_factor_union_output_cap_keeps_its_first_represented_stratum():
    a = np.array([[[0.,0.,0.],[0.,0.,0.]],[[0.,1.,-1.],[1.,1.,1.]]])
    b = np.array([[[-1.5,-1.5,0.],[-1.5,1.5,0.]],
                  [[1.5,-1.5,0.],[1.5,1.5,0.]]])
    result,budget = _solve(a,b,outputs=1)
    assert result is not None and not result['complete']
    assert len(result['branches'])+len(result['singularities']) == 1
    assert 'output_cap' in result['status']['reasons']
    assert budget.output_items == 1


@pytest.mark.parametrize('root',[.25,.75])
def test_factor_fiber_junction_is_not_fixed_to_the_midpoint(root):
    a = np.array([[[s*t,s,s*(t-root)] for t in (0.,1.)] for s in (0.,1.)])
    b = np.array([[[-1.5,-1.5,0.],[-1.5,1.5,0.]],
                  [[1.5,-1.5,0.],[1.5,1.5,0.]]])
    result,_ = _solve(a,b)
    assert result is not None and result['complete']
    branch, = result['branches']
    np.testing.assert_array_equal(branch.curve[0][:,1],root)
    np.testing.assert_allclose(branch.curve[1],[[0.,0.,0.],[root,1.,0.]],atol=1e-14)
    fiber, = result['singularities']
    assert fiber.kind == 'cusp_curve' and fiber.branch_links == [(0,0)]


def test_factor_with_uneliminated_nonlinear_relation_keeps_general_search():
    # Factoring z alone gives two planes, but x equality still restricts
    # each to a parabola. Neither factor is an affine component.
    a = np.array([[[s*t,s,(s-.25)*(t-.5)] for t in (0.,1.)] for s in (0.,1.)])
    b = np.array([[[x,v,0.] for v in (0.,1.)] for x in (0.,0.,1.)])
    assert _solve(a,b)[0] is None


@pytest.mark.parametrize('swap,flip,transpose',[(False,False,False),(True,False,False),
                                               (False,True,False),(False,False,True)])
def test_complete_curved_tangential_line_under_parameter_changes(swap,flip,transpose):
    a,b = _pair()
    if flip:
        b = b[::-1].copy()
    if transpose:
        b = b.swapaxes(0,1).copy()
    if swap:
        a,b = b,a
    result,_ = _solve(a,b)
    assert result is not None and result['complete']
    branch, = result['branches']
    assert branch.kind == 'tangential' and not branch.closed
    stuv,xyz = branch.curve
    assert len(xyz) > 2
    t = xyz[:,1]/3
    np.testing.assert_allclose(xyz[:,0],.5+3*t*(1-t),atol=1e-12)
    np.testing.assert_allclose(xyz[:,2],0.,atol=1e-12)
    assert {float(t[0]),float(t[-1])} == {0.,1.}
    # Independent quadratic sagitta for every represented chord.
    assert np.max(.75*np.diff(t)**2) <= 1e-3
    assert stuv.shape == (len(xyz),4)


def test_uniform_rational_scaling_preserves_exact_reduction():
    a,b = _pair()
    ah = np.concatenate((a,np.ones(a.shape[:2]+(1,))),axis=-1)*2.**-800
    bh = np.concatenate((b,np.ones(b.shape[:2]+(1,))),axis=-1)*2.**800
    result,_ = _solve(ah,bh,rational=True)
    assert result is not None and result['complete']
    assert len(result['branches']) == 1


def test_positive_minimum_is_complete_empty_even_below_modeling_tolerance():
    a,b = _pair()
    a[...,2] += 2.**-40
    result,_ = _solve(a,b)
    assert result is not None and result['complete']
    assert not result['branches'] and not result['points']


def test_negative_minimum_and_two_dimensional_identity_fall_back():
    a,b = _pair()
    a[...,2] -= 2.**-20
    assert _solve(a,b)[0] is None
    assert _solve(b,b.copy())[0] is None


def test_factored_indefinite_quadratic_keeps_both_complete_source_lines():
    a = np.array([[[s,t,z-w] for t,w in ((0.,.25),(.5,-.25),(1.,.25))]
                  for s,z in ((0.,.25),(.5,-.25),(1.,.25))])
    b = np.array([[[u,v,0.] for v in (0.,1.)] for u in (0.,1.)])
    result,_ = _solve(a,b)
    assert result is not None and result['complete']
    assert len(result['branches']) == 4
    junction, = result['singularities']
    assert junction.kind == 'tangent_point'
    np.testing.assert_array_equal(junction.stuv,[.5]*4)
    assert len(junction.branch_links) == 4
    total = sum(np.linalg.norm(np.diff(branch.curve[1],axis=0),axis=1).sum()
                for branch in result['branches'])
    assert total == pytest.approx(2*np.sqrt(2.))
    ends = [tuple(p) for branch in result['branches'] for p in branch.curve[1][[0,-1]]]
    assert all(p in ends for p in ((0.,0.,0.),(0.,1.,0.),(1.,0.,0.),(1.,1.,0.)))


def test_nonfactorable_indefinite_quadratic_keeps_general_search():
    a = np.array([[[s,t,z-w-.125] for t,w in ((0.,.25),(.5,-.25),(1.,.25))]
                  for s,z in ((0.,.25),(.5,-.25),(1.,.25))])
    b = np.array([[[u,v,0.] for v in (0.,1.)] for u in (0.,1.)])
    assert _solve(a,b)[0] is None


def _factor_graph(a,b):
    from fractions import Fraction as F
    a,b = tuple(map(F,a)),tuple(map(F,b))
    constant,linear_s,linear_t = a[0]*b[0],a[0]*b[1]+a[1]*b[0],a[0]*b[2]+a[2]*b[0]
    square_s,square_t,mixed = a[1]*b[1],a[2]*b[2],a[1]*b[2]+a[2]*b[1]
    graph = np.empty((3,3,3))
    for i in range(3):
        for j in range(3):
            z = constant+linear_s*F(i,2)+linear_t*F(j,2)
            z += square_s*(i == 2)+square_t*(j == 2)+mixed*F(i*j,4)
            assert F(float(z)) == z
            graph[i,j] = i/2,j/2,float(z)
    plane = np.array([[[u,v,0.] for v in (0.,1.)] for u in (0.,1.)])
    return graph,plane


@pytest.mark.parametrize('swap,flip,transpose,rational',[
    (False,False,False,False),(True,False,False,False),
    (False,True,False,False),(False,False,True,False),(True,True,True,True)])
def test_mixed_only_quadratic_factor_union_is_parameter_invariant(swap,flip,transpose,rational):
    a,b = _factor_graph((-.5,1,0),(-.5,0,1))
    if flip:
        a = a[::-1].copy()
    if transpose:
        b = b.swapaxes(0,1).copy()
    if rational:
        a = np.concatenate((a,np.ones(a.shape[:2]+(1,))),axis=-1)*2.**-300
        b = np.concatenate((b,np.ones(b.shape[:2]+(1,))),axis=-1)*2.**300
    if swap:
        a,b = b,a
    result,_ = _solve(a,b,rational=rational)
    assert result is not None and result['complete']
    assert len(result['branches']) == 4
    junction, = result['singularities']
    np.testing.assert_array_equal(junction.xyz,[.5,.5,0.])
    assert sum(np.linalg.norm(np.diff(branch.curve[1],axis=0),axis=1).sum()
               for branch in result['branches']) == pytest.approx(2.)
    assert all(branch.kind == 'transversal' for branch in result['branches'])


def test_nonbinary_factor_junction_retains_exact_source_identity():
    from fractions import Fraction as F
    result,_ = _solve(*_factor_graph((-1,2,1),(-.125,1,-2)),atol=1e-6)
    assert result is not None and result['complete']
    junction, = result['singularities']
    np.testing.assert_allclose(junction.xyz,[17/40,3/20,0.],atol=1e-12)
    exact = (F(17,40),F(3,20),F(17,40),F(3,20))
    for index,end in junction.branch_links:
        branch = result['branches'][index]
        assert branch._exact_endpoint_keys[0 if end == 0 else 1] == exact
        assert branch._source_parameter_path == branch._exact_endpoint_keys
        np.testing.assert_array_equal(branch.curve[0][end],junction.stuv)


def test_factor_union_output_cap_keeps_already_published_arm():
    result,_ = _solve(*_factor_graph((-.5,1,0),(-.5,0,1)),outputs=1)
    assert result is not None and not result['complete']
    assert len(result['branches']) == 1
    assert 'output_cap' in result['status']['reasons']


def test_factor_proof_work_stop_reaches_public_whole_owner_partial():
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
    pair = _factor_graph((-.5,1,0),(-.5,0,1))
    result,budget = _solve(*pair,cells=700)
    assert result is None and budget.exhausted
    public = bez_ssx(*pair,atol=1e-3,rational=False,max_cells=700)
    assert not public['complete'] and 'work_budget' in public['status']['reasons']
    assert any(tuple(box['stuv_min']) == (0.,)*4 and tuple(box['stuv_max']) == (1.,)*4
               for box in public['unresolved_regions'])


def test_factor_representation_work_stop_preserves_the_first_known_arm():
    result,budget = _solve(*_factor_graph((-.5,1,0),(-.5,0,1)),cells=1000)
    assert result is not None and not result['complete']
    assert len(result['branches']) == 1
    assert 'work_budget' in result['status']['reasons']
    assert any(box['source_certificate'] == 'exact_quadratic_affine_union'
               for box in result['unresolved_regions'])
    assert budget.cells_processed <= 1000


def test_nonbinary_root_has_typed_exact_parameter_payload():
    a,b = _pair()
    a[...,2] = np.array([1.,-2.,4.])[:,None]
    b[...,2] = -a[...,2]
    result,_ = _solve(a,b,atol=1e-30)
    assert result is not None and not result['complete']
    assert 'parameter_representation' in result['status']['reasons']
    issue, = result['unresolved_regions']
    assert issue['exact_stuv'][0][0] == '1/3'
    assert not result['branches']


def test_zero_work_does_not_construct_exact_polynomials(monkeypatch):
    from mmcore.numeric.intersection.ssx import _ssx_quadratic_constraints as module
    monkeypatch.setattr(module,'_power',lambda *_: (_ for _ in ()).throw(AssertionError('unpaid')))
    result,budget = _solve(*_pair(),cells=0)
    assert result is None and budget.exhausted and budget.cells_processed == 0


def test_output_cap_retains_the_exact_component_obligation():
    result,_ = _solve(*_pair(),outputs=0)
    assert result is not None and not result['complete']
    assert 'output_cap' in result['status']['reasons']
    assert not result['branches']


def test_work_stop_preserves_certified_prefix_and_exact_remaining_curve():
    result,budget = _solve(*_pair(),cells=4000)
    assert result is not None and not result['complete']
    branch, = result['branches']
    issue, = result['unresolved_regions']
    assert 0. < branch.curve[1][-1,1] < 3.
    assert issue['source_certificate'] == 'exact_quadratic_affine_zero_set'
    from fractions import Fraction
    remaining = Fraction(issue['remaining_path_interval'][0])
    assert float(remaining) == branch.curve[1][-1,1]/3
    assert budget.cells_processed <= budget.max_cells


def test_float_alias_on_one_chart_preserves_the_complete_preimage_obligation():
    a = np.array([[[s,t,z] for t in (0.,1.)]
                  for s,z in ((0.,.25),(.5,-.25),(1.,.25))])
    b = a.copy()
    b[...,1] = np.array([-2.**59,2.**59])[None,:]
    b[...,2] *= -1
    result,_ = _solve(a,b,atol=2.)
    assert result is not None and not result['complete']
    assert not result['branches']
    assert result['unresolved_regions'][0]['reason'] == 'parameter_representation'


def test_boundary_tangency_keeps_overlap_classification():
    result,_ = _solve(*_pair(root=0.))
    assert result is not None and result['complete']
    branch, = result['branches']
    assert branch.kind == 'overlap' and branch.overlap
