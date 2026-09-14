import numpy as np
import pytest

from mmcore.numeric._work_budget import SoftWorkBudget
from mmcore.numeric._bezier_common import eval_surface
from mmcore.numeric.intersection.ssx._ssx_matched_isolines import try_matched_isoline_ssx


def _h(net):
    return np.concatenate((net,np.ones(net.shape[:2]+(1,))),axis=2)


def _pair():
    a=np.zeros((3,4,3))
    a[...,0]=np.array([0.,1.,1.,0.])[None,:]+np.array([0.,.5,1.])[:,None]
    a[...,1]=np.array([0.,1.,2.,3.])[None,:]
    a[...,2]=np.array([.25,-.25,.25])[:,None]
    b=a.copy()
    b[...,2]*=-1
    return a,b


@pytest.mark.parametrize('swap,flip,transpose',[(False,False,False),(True,False,False),
                                               (False,True,False),(False,False,True)])
def test_curved_tangent_ruling_is_complete_in_both_parameter_charts(swap,flip,transpose):
    a,b=_pair()
    if flip:b=b[::-1].copy()
    if transpose:b=b.swapaxes(0,1).copy()
    pair=[_h(b),_h(a)] if swap else [_h(a),_h(b)]
    budget=SoftWorkBudget(10000,100)
    result=try_matched_isoline_ssx(*pair,1e-3,budget)
    assert result is not None and not budget.incomplete
    branch,=result['branches']
    assert branch.kind=='tangential'
    q,xyz=branch.curve
    assert xyz[0,1]==0. and xyz[-1,1]==3.
    t=xyz[:,1]/3
    np.testing.assert_allclose(xyz[:,0],.5+3*t*(1-t),atol=1e-12)
    for i in range(len(q)-1):
        for fraction in np.linspace(0.,1.,9):
            parameters=(1-fraction)*q[i]+fraction*q[i+1]
            point=(1-fraction)*xyz[i]+fraction*xyz[i+1]
            for net,uv in zip(pair,(parameters[:2],parameters[2:])):
                assert np.linalg.norm(eval_surface(net,*uv,rational=True)-point)<=1e-3


def test_common_chart_survives_world_axis_mixing():
    a,b=_pair()
    matrix=np.array([[1.,2.,0.],[0.,1.,1.],[1.,0.,1.]])
    pair=[_h(a@matrix.T),_h(b@matrix.T)]
    budget=SoftWorkBudget(10000,100)
    result=try_matched_isoline_ssx(*pair,1e-3,budget)
    assert result is not None and not budget.incomplete
    assert len(result['branches'])==1
    xyz=result['branches'][0].curve[1]@np.linalg.inv(matrix).T
    np.testing.assert_allclose(xyz[[0,-1]],[[.5,0.,0.],[.5,3.,0.]],atol=1e-11)


def test_noninjective_common_projection_uses_general_search():
    a,b=_pair()
    a[...,1]=b[...,1]=np.array([0.,1.,1.,0.])[None,:]
    budget=SoftWorkBudget(10000,100)
    assert try_matched_isoline_ssx(_h(a),_h(b),1e-3,budget) is None


@pytest.mark.parametrize('transpose,flip', [(False, False), (True, True)])
@pytest.mark.parametrize('scale', [.25, 8.])
def test_stretched_common_chart_retains_geometric_root_accuracy(transpose, flip, scale):
    c, stretch = .06427065884628964, 1e12
    a = scale*np.array([[[stretch*s, t, z] for t in (0., 1.)]
                        for s, z in zip((0., .5, 1.), (-c, -c, 1.-c))])
    b = a.copy()
    b[..., 2] *= -1.
    if flip:
        b = b[::-1].copy()
    if transpose:
        b = b.swapaxes(0, 1)
    budget = SoftWorkBudget(10000, 100)
    result = try_matched_isoline_ssx(_h(a), _h(b), scale*1e-3, budget)
    assert result is not None and not budget.incomplete
    branch, = result['branches']
    assert np.max(np.abs(branch.curve[1][:, 0]-scale*stretch*np.sqrt(c))) <= scale*1e-3
