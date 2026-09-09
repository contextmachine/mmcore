"""A rounded boundary sample cannot decide which child owns its source root."""
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from mmcore.numeric._work_budget import SoftWorkBudget
from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx


def _children(axis=0):
    result = []
    for interval in ((0.,.5),(.5,1.)):
        box = [(0.,1.)]*4
        box[axis] = interval
        result.append(tuple(box))
    return result


def _point():
    q = np.array([np.nextafter(.5,1.),.25,.25,.25])
    box = np.repeat(q[:,None],2,axis=1)
    box[0] = .49,.51
    point = ssx.BoundaryPoint(q,np.zeros(3),(1,0),root_box=box)
    point._source_root_box = True
    return point


def test_source_box_straddling_cut_is_inherited_by_both_children():
    point = _point()
    cell = SimpleNamespace(work_budget=None,root_matcher=None)
    groups = ssx._source_boundary_child_groups(cell,[point],_children())
    assert groups == [[point],[point]]


def test_exact_source_interval_outranks_wrong_side_float_sample():
    point = _point()
    exact = Fraction(1,2)-Fraction(1,2**60)
    exact_box = ((exact,exact),)+((Fraction(1,4),Fraction(1,4)),)*3
    point._source_root_box = False
    matcher = SimpleNamespace(source_certificates={id(point):{'owner':point,'exact_box':exact_box}})
    cell = SimpleNamespace(work_budget=None,root_matcher=matcher)
    assert point.stuv[0] > .5 and float(exact) == .5
    assert ssx._source_boundary_child_groups(cell,[point],_children()) == [[point],[]]


def test_derived_net_box_without_source_proof_cannot_exclude_a_child():
    point = _point()
    point._source_root_box = False
    point.root_box[0] = .6,.7
    cell = SimpleNamespace(work_budget=None,root_matcher=None)
    assert ssx._source_boundary_child_groups(cell,[point],_children()) == [[point],[point]]


def test_closed_child_face_root_is_shared_even_with_wrong_representative():
    point = _point()
    point.root_box[0] = .5,.5
    cell = SimpleNamespace(work_budget=None,root_matcher=None)
    assert ssx._source_boundary_child_groups(cell,[point],_children()) == [[point],[point]]


def test_denied_partition_work_keeps_all_unexamined_events():
    point = _point()
    point.root_box[0] = .1,.2
    budget = SoftWorkBudget(0,10)
    cell = SimpleNamespace(work_budget=budget,root_matcher=None)
    assert ssx._source_boundary_child_groups(cell,[point],_children()) == [[point],[point]]
    assert budget.exhausted


def test_lazy_source_enclosure_is_requested_once_for_all_children():
    point = _point()
    point._source_root_box = False
    calls = []
    def enclose(event,radii):
        calls.append(event)
        box = point.root_box.copy()
        box[0] = .1,.2
        return box
    matcher = SimpleNamespace(source_certificates={},enclose=enclose)
    cell = SimpleNamespace(work_budget=None,root_matcher=matcher)
    groups = ssx._source_boundary_child_groups(cell,[point],_children(),np.full(4,.01))
    assert calls == [point] and point._source_root_box
    assert groups == [[point],[]]
