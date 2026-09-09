"""A known source root owns only a proved complete product neighborhood."""
import numpy as np
import pytest

from mmcore.numeric.intersection.csx._bez_csx4 import (
    _certify_csx_owned_cell, _residual_vec_net, bez_csx,
)


def _two_roots():
    curve = np.array([[0.,0.,.1875],[.5,0.,-.3125],[1.,0.,.1875]])
    surface = np.array([[[u,v,0.] for v in (0.,1.)] for u in (0.,1.)])
    return _residual_vec_net(curve,surface,False)


def test_distinct_neighbor_root_is_not_discarded_by_union_ownership():
    known = dict(t=.25,u=.25,v=0.,parameter_root_box=((.25,.25),(.25,.25),(0.,0.)),
                 root_existence_certification='exact_parameter_identity')
    cell = ((.74,.76),(.74,.76),(0.,.01))
    assert _certify_csx_owned_cell(_two_roots(),cell,[known]) is False
    assert '_unique_box' not in known


def test_union_ownership_requires_established_source_existence():
    known = dict(t=.25,u=.25,v=0.,parameter_root_box=((.25,.25),(.25,.25),(0.,0.)))
    assert _certify_csx_owned_cell(_two_roots(),((.24,.26),(.24,.26),(0.,.01)),[known]) is False


def test_denied_union_proof_retains_cell_and_known_root(monkeypatch):
    from mmcore.numeric.intersection.csx import _bez_csx4 as module
    monkeypatch.setattr(module,'_root_boxes_have_same_root',
                        lambda *_: (_ for _ in ()).throw(AssertionError('unpaid proof')))
    known = dict(t=.25,u=.25,v=0.,parameter_root_box=((.25,.25),(.25,.25),(0.,0.)),
                 root_existence_certification='exact_parameter_identity')
    before = dict(known)
    assert _certify_csx_owned_cell(_two_roots(),((.24,.26),(.24,.26),(0.,.01)),
                                   [known],charge=lambda _:False) is None
    assert known == before


def _case11_cut():
    curve = np.array([[-0.02423657929919547,0.909336245446449,-0.22033485052740379,1.],
                      [-0.02807441846036341,0.9109168376759033,-0.2204205660047735,1.]])
    surface = np.array([
        [[0.7066800534271351,1.7497196903155028,-0.35304130813685486,1.],
         [0.3332343781719969,1.3258402822005553,-0.22299278080365836,1.],
         [-0.031705460044590766,0.9083992084824125,-0.22045277050418186,1.]],
        [[0.7076391747984684,1.748970961660593,-0.352999710566514,1.],
         [0.3341813065433147,1.3250833001133149,-0.22294846422546452,1.],
         [-0.03076414517006998,0.907631921647656,-0.2204084008203659,1.]],
        [[0.7085982010503724,1.7482220952393543,-0.3529583015226634,1.],
         [0.33512816752871855,1.3243261890935822,-0.2229043484967163,1.],
         [-0.02982288101319391,0.906864507067206,-0.22036423222667828,1.]],
        [[0.709557132479323,1.7474730911969996,-0.35291708106117853,1.],
         [0.3360749614636516,1.3235689493593186,-0.22286043367694167,1.],
         [-0.028881667186648523,0.9060969650506249,-0.2203202647827183,1.]],
    ])
    return curve,surface


def test_case11_cut_finishes_when_terminal_cells_extend_past_known_root_box():
    result = bez_csx(*_case11_cut(),atol=.0005,rational=True,
                      max_cells=35984,max_results=128,tolerance_tier=False)
    assert result['boundary_topology_complete'], result
    assert not result['budget_exhausted']
    root, = result['isolated']
    np.testing.assert_allclose([root[k] for k in ('t','u','v')],
                               [.719657194127666,.5982416395775998,.9958659260308748],atol=1e-10)
    assert root['root_existence_certification']
    assert result['cells_processed'] <= 35984


@pytest.mark.parametrize('proof_outcome,cause', [(False,'depth'),(None,'cells')])
def test_unproved_or_unpaid_terminal_ownership_stays_partial(monkeypatch,proof_outcome,cause):
    from mmcore.numeric.intersection.csx import _bez_csx4 as module
    monkeypatch.setattr(module,'_certify_csx_owned_cell',lambda *args,**kwargs:proof_outcome)
    result = bez_csx(*_case11_cut(),atol=.0005,rational=True,
                      max_cells=35984,max_results=128,tolerance_tier=False)
    assert result['budget_exhausted']
    assert result['truncation_cause'] == cause
    assert not result['unresolved_obligations_complete']


def test_case11_retry_reservations_share_an_established_source_root():
    curve = np.array([[0.8265508355388008, 0.7082798326572598, -0.20337564289247778, 1.0],
     [0.06138528049480582, 1.024990270467438, -0.22368648104388122, 1.0]])
    surface = np.array([[[0.6803822393524407, 3.1523277892371873, -0.8054189644450171, 1.0],
      [-0.05470389817498239, 2.2887778527906035, -0.2930158393107198, 1.0],
      [-0.7853761938279971, 1.4510007844044361, -0.2930158393107198, 1.0]],
     [[0.9592934648941902, 2.9861543529989443, -0.7778812969377016, 1.0],
      [0.204786678946383, 2.1167714987262882, -0.2562989493009657, 1.0],
      [-0.531362165225939, 1.2764630795363403, -0.2562989493009657, 1.0]],
     [[1.2236434351280612, 2.808974395686443, -0.7594778968848674, 1.0],
      [0.45404901897652367, 1.9341505625229811, -0.23176108256385358, 1.0],
      [-0.2871615436177955, 1.0883013831572563, -0.23176108256385358, 1.0]],
     [[1.4772978079793928, 2.62106352896405, -0.7509587144790657, 1.0],
      [0.6978319631403389, 1.7430222937181634, -0.22040217268945125, 1.0],
      [-0.046241520551529885, 0.8917706448616347, -0.22040217268945125, 1.0]]])
    result = bez_csx(curve,surface,atol=.0005,rational=True,
                      max_cells=54627,max_results=128,tolerance_tier=False)
    assert result["boundary_topology_complete"], result
    assert not result["budget_exhausted"]
    assert len(result["isolated"]) == 1
    assert not result.get("unresolved_parameter_boxes")
