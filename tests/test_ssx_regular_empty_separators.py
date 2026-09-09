from types import SimpleNamespace

import numpy as np
import pytest

from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx
from mmcore.numeric.intersection.ssx._ssx_cofactor_bounds import SourceCofactorBounds


def _two_arcs():
    # s=(t-.5)^2, clipped to 1/64 <= s <= 9/64, has two arcs
    # on opposite sides of t=.5.  A strict t minor fixes orientation.
    first = np.array([[[s,t,s-q,1.] for t,q in
                       zip((0.,.5,1.),(.25,-.25,.25))] for s in (0.,1.)])
    second = np.array([[[s,t,0.,1.] for t in (0.,1.)] for s in (0.,1.)])
    roots = []
    for t in (.125,.375,.625,.875):
        s = (t-.5)**2
        point = np.array([s,t,s,t])
        root = ssx.BoundaryPoint(point, np.array([s,t,0.]), (0,-1),
                                 root_box=np.column_stack((point,point)))
        root._source_root_box = True
        roots.append(root)
    return SimpleNamespace(crossings=roots, boundary_complete=True,
        box=((1/64,9/64),(0.,1.),(1/64,9/64),(0.,1.)),
        source_cofactors=SourceCofactorBounds(first,second),
        root_matcher=None, work_budget=None)


def test_four_ports_prove_an_empty_monotone_parameter_separator():
    cell = _two_arcs()
    assert ssx._regular_boundary_empty_separators(cell, None) == (1,[.5])
    certificate = cell._regular_empty_cut_certificate
    assert certificate['axis'] == 1
    assert certificate['owner_box'] == cell.box
    assert certificate['cuts'] == (.5,)
    assert certificate['prefix_counts'] == (1,0,1,0)


@pytest.mark.parametrize('failure', ['incomplete','unproved','unknown_germ','wrong_germ'])
def test_missing_census_existence_or_consistent_germs_cannot_prove_empty(failure):
    cell = _two_arcs()
    if failure == 'incomplete':
        cell.boundary_complete = False
    elif failure == 'unproved':
        cell.crossings[0]._source_root_box = False
    else:
        original = cell.source_cofactors
        def bounds(box):
            lo,hi = (a.copy() for a in original.bounds(box))
            if box[1][0] == box[1][1] == .125:
                if failure == 'unknown_germ':
                    lo[0],hi[0] = -1.,1.
                else:
                    lo[0],hi[0] = -hi[0],-lo[0]
            return lo,hi
        cell.source_cofactors = SimpleNamespace(bounds=bounds)
    assert ssx._regular_boundary_empty_separators(cell,None) is None
    assert cell._regular_empty_cut_certificate is None


def test_repeated_object_is_one_event_but_unproved_aliases_are_not_counted():
    cell = _two_arcs()
    cell.crossings.append(cell.crossings[0])
    assert ssx._regular_boundary_empty_separators(cell,None) == (1,[.5])
    first = cell.crossings[0]
    other = ssx.BoundaryPoint(first.stuv.copy(),first.xyz.copy(),first.face,
                              root_box=first.root_box.copy())
    other._source_root_box = True
    cell.crossings.append(other)
    assert ssx._regular_boundary_empty_separators(cell,None) is None


def test_work_denial_never_becomes_an_empty_census():
    cell = _two_arcs()
    cell.work_budget = SimpleNamespace(charge_cells=lambda *args:False)
    assert ssx._regular_boundary_empty_separators(cell,None) is None
