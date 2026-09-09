"""Derived controls cannot certify an original source residual census."""
import numpy as np
import pytest

from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx, _residual_vec_net
from mmcore.numeric.intersection._root_box_certificate import (
    root_box_contains_zero, unique_root_box, root_boxes_have_same_root,
)


def pair(boundary=False):
    offset = 0. if boundary else .5
    curve = np.array([[0., offset, -offset], [1., offset, 1.-offset]])
    surface = np.array([[[u,v,0.] for v in (0.,1.)] for u in (0.,1.)])
    return curve, surface


def test_original_residual_overrides_rounded_empty_aabb():
    curve, surface = pair()
    source = _residual_vec_net(curve,surface,False)
    surface[...,2] = 10.
    result = bez_csx(curve,surface,rational=False,tolerance_tier=False,
                     source_residual=(source,0.),max_cells=100,max_depth=6)
    # These deliberately poor proposals cannot find the original root,
    # but their disjoint AABBs cannot prove the original source empty.
    assert result['budget_exhausted']
    assert not result['boundary_topology_complete']


def test_source_interior_krawczyk_existence_has_source_enclosure():
    curve, surface = pair()
    source = _residual_vec_net(curve,surface,False)
    source.setflags(write=False)
    result = bez_csx(curve,surface,rational=False,
                     source_residual=(source,1e-14),
                     max_cells=1000)
    assert result['boundary_topology_complete'], result
    root, = result['isolated']
    assert root['root_existence_certification'] == 'source_krawczyk_inclusion'
    assert root['point_certification'] == 'proposal_geometry'
    assert all(lo <= .5 <= hi for lo,hi in root['parameter_root_box'])
    assert max(hi-lo for lo,hi in root['parameter_root_box']) < 1e-9
    assert all(a <= lo <= hi <= b for (lo,hi),(a,b) in zip(
        root['parameter_root_box'],root['parameter_uniqueness_box']))


def test_closed_boundary_requires_exact_source_identity():
    curve, surface = pair(boundary=True)
    calls = []
    def exact_source(parameters):
        calls.append(parameters)
        return all(x == 0. for x in parameters)
    result = bez_csx(curve,surface,rational=False,
                     source_residual=(_residual_vec_net(curve,surface,False),0.),
                     source_exact_root=exact_source,max_cells=1000)
    assert calls
    assert result['boundary_topology_complete'], result
    root, = result['isolated']
    assert root['parameter_root_box'] == ((0.,0.),)*3
    assert root['root_existence_certification'] == 'exact_parameter_identity'


def test_proposal_identity_does_not_establish_source_root():
    curve, surface = pair(boundary=True)
    source = _residual_vec_net(curve,surface,False)
    source[...,2] += 2.**-60
    result = bez_csx(curve,surface,rational=False,
                     source_residual=(source,0.),max_cells=300,max_depth=9)
    assert not result['isolated']
    assert not result['boundary_topology_complete']


def test_coefficient_uncertainty_enters_existence_and_uniqueness():
    curve,surface = pair()
    net = _residual_vec_net(curve,surface,False)
    box = ((.4,.6),)*3
    assert root_box_contains_zero(net,box,coefficient_error=0.)
    assert not root_box_contains_zero(net,box,coefficient_error=1.)
    assert unique_root_box(net,(.5,)*3,(.1,)*3,coefficient_error=0.)
    assert unique_root_box(net,(.5,)*3,(.1,)*3,coefficient_error=1.) is None
    assert root_boxes_have_same_root(net,box,box,coefficient_error=0.)
    assert not root_boxes_have_same_root(net,box,box,coefficient_error=1.)


@pytest.mark.parametrize('error', [-1., np.inf, [0.,0.], [0.,np.nan,0.]])
def test_invalid_source_error_is_rejected(error):
    curve,surface = pair()
    with pytest.raises(ValueError):
        bez_csx(curve,surface,rational=False,
                source_residual=(_residual_vec_net(curve,surface,False),error))


def test_denied_source_search_does_not_use_proposal_fast_tier():
    curve,surface = pair()
    result = bez_csx(curve,surface,rational=False,
                     source_residual=(_residual_vec_net(curve,surface,False),0.),
                     max_cells=0)
    assert not result['boundary_topology_complete']
    assert result['cells_processed'] == 0


def test_source_callback_work_is_prepaid_and_denial_retains_geometry():
    curve,surface = pair()
    calls = []
    result = bez_csx(curve,surface,rational=False,
                     source_residual=(_residual_vec_net(curve,surface,False),0.),
                     source_exact_root=lambda p: calls.append(p) or True,
                     source_exact_root_work=4,max_cells=4)
    assert calls == []
    assert result['budget_exhausted']
    assert not result['boundary_topology_complete']
    assert result['cells_processed'] <= 4
