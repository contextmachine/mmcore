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


def _tilted_plane_h():
    points = np.array([
        [[0., 0., -0.5], [0., 1., -0.5]],
        [[1., 0., 0.5], [1., 1., 0.5]],
    ])
    return np.concatenate([points, np.ones((2, 2, 1))], axis=-1)


def test_ssx6_interior_cut_face_filter_is_a_filter(monkeypatch):
    """Ledger L53 (user decision 2026-07-12: repair + document): the interior
    cut-face path wrote ``list((lambda x: ..., seq))`` — a 2-element list
    ``[<lambda>, seq]``, not a filter call — so the FIRST interior cut of any
    run raised TypeError inside ``_isoline_csx_to_global``. The repaired
    filter must drop t-endpoint roots and convert the rest."""
    from types import SimpleNamespace

    result = {
        'isolated': [
            {'t': 0.5, 'u': 0.5, 'v': 0.5,
             'point': np.array([0.5, 0.5, 0.0])},
            # endpoint root re-found on the cut line: must be filtered out
            {'t': 1.0 - 1e-9, 'u': 0.9, 'v': 0.9,
             'point': np.array([0.9, 0.9, 0.0])},
        ],
        'overlaps': [], 'parameter_fibers': [],
        'budget_exhausted': False,
        'boundary_topology_complete': True,
    }
    monkeypatch.setattr(ssx6, 'bez_csx', lambda *a, **k: result)
    cell = SimpleNamespace(
        g1=SimpleNamespace(surface=_plane_h()),
        g2=SimpleNamespace(surface=_tilted_plane_h()),
        box=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))

    crossings = ssx6._csx_on_cut_face(cell, 0, 0.5, 1e-3)

    assert len(crossings) == 1
    assert float(crossings[0].stuv[0]) == pytest.approx(0.5)
