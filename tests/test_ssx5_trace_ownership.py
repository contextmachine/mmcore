"""A failed trace is provisional while its whole domain is searched again."""
import numpy as np
import pytest

from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx


@pytest.mark.parametrize('max_depth, complete, length', [(8, True, 1.), (0, False, .5)])
def test_failed_parent_prefix_has_one_search_owner(monkeypatch, max_depth, complete, length):
    first = np.array([[[u, v, 0.] for v in (0., 1.)] for u in (0., 1.)])
    second = np.array([[[u, v, v-.375] for v in (0., 1.)] for u in (0., 1.)])
    trace = ssx._trace_cell_by_registrations

    def interrupted_parent(cell, *args, **kwargs):
        fragments, points = trace(cell, *args, **kwargs)
        if cell.depth == 0 and fragments:
            original = fragments[0]
            stuv = np.array([original.stuv_path[0],
                             .5*(original.stuv_path[0]+original.stuv_path[-1])])
            xyz = np.array([original.xyz_path[0],
                            .5*(original.xyz_path[0]+original.xyz_path[-1])])
            cell.trace_incomplete = True
            return [ssx._Fragment(original.start_point, None, stuv, xyz)], points
        return fragments, points

    monkeypatch.setattr(ssx, '_trace_cell_by_registrations', interrupted_parent)
    result = ssx.bez_ssx(first, second, rational=False, atol=.001,
                          max_xyz_step=.1, max_cells=30000, max_depth=max_depth)
    assert result['complete'] is complete
    assert len(result['branches']) == 1
    branch = result['branches'][0]
    assert not branch.closed
    np.testing.assert_allclose(branch.curve[1][:, 1:], [[.375, 0.]]*len(branch.curve[1]),
                               atol=1e-12, rtol=0.)
    assert np.linalg.norm(np.diff(branch.curve[1], axis=0), axis=1).sum() == pytest.approx(length)
    if complete:
        assert not result['points'] and not result['unresolved_regions']
    else:
        assert 'depth_limit' in result['status']['reasons']
        assert result['unresolved_regions']


def test_endpoint_broadphase_leaves_work_for_a_long_registered_chain():
    from mmcore.numeric._work_budget import SoftWorkBudget
    fragments = []
    for i in range(128):
        q = np.array([[i/128, .5, i/128, .5], [(i+1)/128, .5, (i+1)/128, .5]])
        xyz = np.column_stack((q[:, 0], np.full(2, .5), np.zeros(2)))
        ends = [ssx.BoundaryPoint(row, point, (0, -1)) for row, point in zip(q, xyz)]
        fragments.append(ssx._Fragment(*ends, q, xyz))
    # The old endpoint all-pairs scan alone uses 32640 units, leaving
    # nothing for subsequent fragment checks and chain assembly.
    budget = SoftWorkBudget(32768, 1000, max_postprocess_work=32768)
    branches = ssx._assemble_fragments(fragments, unify_tol=np.full(4, 1e-6), work_budget=budget)
    assert not budget.exhausted
    assert len(branches) == 1
    assert np.linalg.norm(np.diff(branches[0].curve[1], axis=0), axis=1).sum() == pytest.approx(1.)
