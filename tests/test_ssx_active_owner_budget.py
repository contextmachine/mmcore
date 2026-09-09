"""A work stop after popping an owner must retain its unresolved domain."""
import numpy as np
import pytest

from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx
from mmcore.numeric.intersection.ssx._ssx_source_cut_cache import SourceFaceCensusCache


@pytest.mark.parametrize('denial_site', ['before_children', 'first_cut'])
def test_budget_denial_after_pop_keeps_active_owner(monkeypatch, denial_site):
    first = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    second = np.array([[[u, v, v-.5] for v in (0., 1.)] for u in (0., 1.)])
    actual_budget = ssx._SSXSoftBudget
    budgets, popped_owners, denied = [], [], []

    def create_budget(*args, **kwargs):
        work = actual_budget(*args, **kwargs)
        budgets.append(work)
        return work

    def deny():
        work = budgets[0]
        assert not work.charge_cells(work.remaining_cells+1, 'controlled_denial')
        denied.append(True)

    def split_plan(points, box, **kwargs):
        popped_owners.append(box)
        if denial_site == 'before_children':
            deny()
        return None, None, None, None

    original_census = SourceFaceCensusCache.__call__
    def cut_census(self, axis, value, box):
        if popped_owners and denial_site == 'first_cut':
            deny()
            return dict(isolated=[], boundary_topology_complete=False,
                        budget_exhausted=True, truncation_cause='max_cells')
        return original_census(self, axis, value, box)

    monkeypatch.setattr(ssx, '_SSXSoftBudget', create_budget)
    monkeypatch.setattr(ssx, '_check_loop_free', lambda *a, **k: False)
    monkeypatch.setattr(ssx, '_check_tangency', lambda *a, **k: None)
    monkeypatch.setattr(ssx, '_compute_split_plan', split_plan)
    monkeypatch.setattr(SourceFaceCensusCache, '__call__', cut_census)
    result = ssx.bez_ssx(first, second, rational=False, max_xyz_step=.1)
    assert len(popped_owners) == len(denied) == 1
    assert not result['complete']
    assert 'work_budget' in result['status']['reasons']
    low = tuple(lo for lo, hi in popped_owners[0])
    high = tuple(hi for lo, hi in popped_owners[0])
    assert any(region['stuv_min'] == low and region['stuv_max'] == high
               and region['reason'] == 'work_budget'
               for region in result['unresolved_regions'])
