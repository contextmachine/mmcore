import numpy as np
import pytest

import mmcore.numeric.intersection.ssx._bez_ssx6 as ssx6


def _plane_h():
    points = np.array([
        [[0., 0., 0.], [0., 1., 0.]],
        [[1., 0., 0.], [1., 1., 0.]],
    ])
    return np.concatenate([points, np.ones((2, 2, 1))], axis=-1)


@pytest.mark.parametrize('result, reason', [
    ({
        'isolated': [], 'overlaps': [], 'parameter_fibers': [],
        'budget_exhausted': True,
        'boundary_topology_complete': False,
    }, 'incomplete Bezier CSX'),
    ({
        'isolated': [], 'overlaps': [],
        'parameter_fibers': [{'t_range': (0., 1.), 'u': 0.5, 'v': 0.5}],
        'budget_exhausted': False,
        'boundary_topology_complete': True,
    }, 'positive-dimensional parameter fiber'),
])
def test_ssx6_boundary_analysis_aborts_on_unsafe_csx(monkeypatch, result, reason):
    monkeypatch.setattr(ssx6, 'bez_csx', lambda *args, **kwargs: result)

    with pytest.raises(RuntimeError, match=reason):
        ssx6._find_ssx_boundary_zeros(_plane_h(), _plane_h(), 1e-3)


def test_ssx6_csx_contract_accepts_complete_result():
    result = {
        'isolated': [], 'overlaps': [], 'parameter_fibers': [],
        'budget_exhausted': False,
        'boundary_topology_complete': True,
    }
    assert ssx6._require_complete_csx_result(result, 'test') is result
