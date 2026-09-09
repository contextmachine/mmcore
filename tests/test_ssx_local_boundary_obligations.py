"""An unresolved face affects only closed children that intersect it."""
from types import SimpleNamespace

import numpy as np

from mmcore.numeric._work_budget import SoftWorkBudget
from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx


def box(axis=0, interval=(0.,1.)):
    bounds = [(0.,1.)]*4
    bounds[axis] = interval
    return tuple(bounds)


def owner(complete, obligations, budget=None):
    return SimpleNamespace(boundary_complete=complete,
                           boundary_obligations=obligations,work_budget=budget)


def test_incomplete_face_does_not_poison_disjoint_descendants():
    face = box(interval=(0.,0.))
    children = [box(interval=(0.,.5)),box(interval=(.5,1.))]
    result = ssx._source_boundary_child_obligations(owner(False,(face,)),children)
    assert result == [(face,),()]


def test_new_incomplete_cut_is_inherited_on_both_closed_sides_only():
    face = box(interval=(.5,.5))
    children = [box(interval=(0.,.25)),box(interval=(.25,.5)),box(interval=(.5,1.))]
    result = ssx._source_boundary_child_obligations(owner(True,None),children,(face,))
    assert result == [(),(face,),(face,)]


def test_missing_localization_cannot_recover_an_incomplete_census():
    children = [box(interval=(0.,.5)),box(interval=(.5,1.))]
    assert ssx._source_boundary_child_obligations(owner(False,None),children) == [None,None]


def test_all_four_parameter_bounds_participate_in_face_ownership():
    face = list(box(interval=(.5,.5)))
    face[3] = (0.,.25)
    face = tuple(face)
    children = [box(axis=3,interval=(0.,.25)),box(axis=3,interval=(.5,1.))]
    assert ssx._source_boundary_child_obligations(owner(False,(face,)),children) == [(face,),()]


def test_denied_localization_keeps_unknown_census():
    budget = SoftWorkBudget(0,10)
    children = [box(interval=(.5,1.))]
    assert ssx._source_boundary_child_obligations(
        owner(False,(box(interval=(0.,0.)),),budget),children) == [None]
    assert budget.exhausted


def test_top_face_failure_retains_its_exact_closed_face_box():
    net = np.array([[[u,v,0.] for v in (0.,1.)] for u in (0.,1.)])
    def census(curve,surface,axis,value,**kwargs):
        complete = not (axis == 2 and value == 1.)
        return dict(isolated=[],overlaps=[],parameter_fibers=[],
                    boundary_topology_complete=complete,budget_exhausted=False)
    status = {}
    ssx._find_ssx_boundary_zeros(net,net,1e-3,rational=False,
                                face_csx_fn=census,census_sink=status)
    assert not status['complete']
    assert status['boundary_obligations'] == [box(axis=2,interval=(1.,1.))]


def test_progress_filter_cannot_recreate_an_adjacent_float_child():
    lo = .5
    hi = np.nextafter(lo,1.)
    bounds = box(interval=(lo,hi))
    assert ssx._strict_interior_cuts(bounds,0,[.5*(lo+hi)]) == []
    assert ssx._strict_interior_cuts(bounds,1,[.5]) == [.5]
