"""Useful boundary geometry survives a local CSX result cap."""
import numpy as np

from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx


def test_partial_boundary_roots_remain_available_for_validated_tracing(monkeypatch):
    first = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    second = np.array([[[u, v, v-.5] for v in (0., 1.)] for u in (0., 1.)])
    actual_csx = ssx.bez_csx
    partial_calls = []

    def partial_csx(*args, **kwargs):
        result = actual_csx(*args, **kwargs)
        if result['isolated']:
            partial_calls.append(result)
            result = dict(result, budget_exhausted=True,
                          boundary_topology_complete=False,
                          truncation_cause='results')
        return result

    monkeypatch.setattr(ssx, 'bez_csx', partial_csx)
    # Exercise general continuation: this pair otherwise uses the isoline
    # reduction and never calls the CSX solver being fault-injected here.
    result = ssx.bez_ssx(first, second, atol=1e-3, rational=False,
                         max_depth=0, max_xyz_step=.1)

    assert partial_calls, "the test must exercise partial CSX boundary output"
    assert not result['complete']
    assert result['branches'], result['status']
    points = np.vstack([branch.curve[1] for branch in result['branches']])
    assert points[:, 0].min() <= 1e-3
    assert points[:, 0].max() >= 1.-1e-3
    assert np.max(np.abs(points[:, 1]-.5)) <= 1e-3
    assert np.max(np.abs(points[:, 2])) <= 1e-3
