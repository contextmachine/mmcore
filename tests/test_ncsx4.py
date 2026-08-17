import numpy as np
import pytest

from mmcore.nurbs._nurbs_eval import NURBSCurveTuple, NURBSSurfaceTuple
from mmcore.numeric.intersection.csx._ncsx4 import nurbs_csx
import mmcore.numeric.intersection.csx._ncsx4 as ncsx4


def _curve_and_surface():
    curve = NURBSCurveTuple(
        order=2,
        knot=np.array([2., 2., 4., 4.]),
        control_points=np.array([[0., 0., 0.], [1., 0., 0.]]),
        weights=np.ones(2),
    )
    surface = NURBSSurfaceTuple(
        order_u=2,
        order_v=2,
        knot_u=np.array([10., 10., 20., 20.]),
        knot_v=np.array([30., 30., 50., 50.]),
        control_points=np.array([
            [[0., -1., 0.], [0., 1., 0.]],
            [[1., -1., 0.], [1., 1., 0.]],
        ]),
        weights=np.ones((2, 2)),
    )
    return curve, surface


def test_nurbs_csx_default_returns_status_on_partial_result(monkeypatch):
    # Ledger L41 (review finding 3): the raise-on-incomplete default turned
    # collapsed-edge geometry (cone apex / sphere pole on-surface) into a
    # RuntimeError for callers that never opted into status. The default is
    # now always-return-status; fail-fast stays available as an explicit
    # return_status=False opt-in.
    curve, surface = _curve_and_surface()

    def partial(*args, **kwargs):
        return {
            'isolated': [],
            'overlaps': [],
            'parameter_fibers': [],
            'budget_exhausted': True,
            'cells_processed': 11,
            'boundary_topology_complete': False,
        }

    monkeypatch.setattr(ncsx4, 'bez_csx_v4', partial)

    isolated, overlaps, status = nurbs_csx(curve, surface)
    assert isolated is None
    assert overlaps is None
    assert status['complete'] is False
    assert status['budget_exhausted'] is True
    assert status['boundary_topology_complete'] is False
    assert status['cells_processed'] == 11
    assert status['partial_results'] == 1

    with pytest.raises(RuntimeError, match='incomplete Bezier CSX'):
        nurbs_csx(curve, surface, return_status=False)


def test_nurbs_csx_maps_positive_dimensional_fibers_when_requested(monkeypatch):
    curve, surface = _curve_and_surface()

    def fiber(*args, **kwargs):
        return {
            'isolated': [],
            'overlaps': [],
            'parameter_fibers': [{
                't_range': (0.25, 0.75),
                'u': 0.5,
                'v': 0.25,
                'point': np.array([0.5, 0., 0.]),
                'surface_kind': 'min',
            }],
            'budget_exhausted': False,
            'cells_processed': 3,
            'boundary_topology_complete': True,
        }

    monkeypatch.setattr(ncsx4, 'bez_csx_v4', fiber)

    with pytest.raises(RuntimeError, match='positive-dimensional parameter fiber'):
        nurbs_csx(curve, surface, return_status=False)

    isolated, overlaps, status = nurbs_csx(curve, surface)
    assert isolated is None
    assert overlaps is None
    assert status['budget_exhausted'] is False
    assert status['boundary_topology_complete'] is True
    assert status['cells_processed'] == 3
    assert len(status['parameter_fibers']) == 1
    mapped = status['parameter_fibers'][0]
    assert mapped['t_range'] == pytest.approx((2.5, 3.5))
    assert mapped['u'] == pytest.approx(15.)
    assert mapped['v'] == pytest.approx(35.)


def test_nurbs_csx_complete_result_return_shapes(monkeypatch):
    curve, surface = _curve_and_surface()

    monkeypatch.setattr(ncsx4, 'bez_csx_v4', lambda *args, **kwargs: {
        'isolated': [],
        'overlaps': [],
        'parameter_fibers': [],
        'budget_exhausted': False,
        'cells_processed': 1,
        'boundary_topology_complete': True,
    })

    # Default: always-return-status three-value shape, complete.
    isolated, overlaps, status = nurbs_csx(curve, surface)
    assert (isolated, overlaps) == (None, None)
    assert status['complete'] is True
    # Explicit fail-fast opt-out keeps the legacy two-value shape.
    result = nurbs_csx(curve, surface, return_status=False)
    assert result == (None, None)


def _multi_span_curve_and_surface():
    knots = np.array([0., 0., 0.25, 0.5, 0.75, 1., 1.])
    axis = np.linspace(0.0, 1.0, 5)
    curve = NURBSCurveTuple(
        order=2,
        knot=knots,
        control_points=np.column_stack([
            axis, np.full(5, 0.5), np.zeros(5)]),
        weights=np.ones(5),
    )
    points = np.empty((5, 5, 3), dtype=np.float64)
    for i, x in enumerate(axis):
        for j, y in enumerate(axis):
            points[i, j] = [x, y, 0.0]
    surface = NURBSSurfaceTuple(
        order_u=2,
        order_v=2,
        knot_u=knots,
        knot_v=knots,
        control_points=points,
        weights=np.ones((5, 5)),
    )
    return curve, surface


def test_nurbs_csx_shares_max_cells_across_all_span_pairs(monkeypatch):
    curve, surface = _multi_span_curve_and_surface()
    calls = []

    def complete_but_spends_allowance(*_args, **kwargs):
        allowance = int(kwargs['max_cells'])
        calls.append(allowance)
        return {
            'isolated': [],
            'overlaps': [],
            'parameter_fibers': [],
            'budget_exhausted': False,
            'cells_processed': allowance,
            'boundary_topology_complete': True,
        }

    monkeypatch.setattr(ncsx4, 'bez_csx_v4', complete_but_spends_allowance)
    isolated, overlaps, status = nurbs_csx(
        curve, surface, max_cells=2, return_status=True,
    )

    assert isolated is None and overlaps is None
    assert calls == [2]
    assert status['cells_processed'] == 2
    assert status['max_cells'] == 2
    assert status['complete'] is False
    assert status['budget_exhausted'] is True


def test_nurbs_csx_shares_max_results_across_span_pairs(monkeypatch):
    curve, surface = _multi_span_curve_and_surface()
    calls = []

    def one_overlap(*_args, **kwargs):
        calls.append(int(kwargs['max_results']))
        return {
            'isolated': [],
            'overlaps': [{
                't_range': (0.0, 1.0),
                'u_range': (0.0, 1.0),
                'v_range': (0.0, 1.0),
            }],
            'parameter_fibers': [],
            'budget_exhausted': False,
            'cells_processed': 0,
            'boundary_topology_complete': True,
        }

    monkeypatch.setattr(ncsx4, 'bez_csx_v4', one_overlap)
    _isolated, overlaps, status = nurbs_csx(
        curve, surface, max_cells=100, max_results=2,
        return_status=True,
    )

    assert calls == [2, 1]
    assert overlaps is not None
    assert status['results_processed'] == 2
    assert status['max_results'] == 2
    assert status['complete'] is False
    assert status['budget_exhausted'] is True


def test_nurbs_csx_sanitizes_fibers_to_remaining_result_budget(monkeypatch):
    curve, surface = _curve_and_surface()

    def overproduces(*_args, **_kwargs):
        return {
            'isolated': [],
            'overlaps': [],
            'parameter_fibers': [
                {'t': 0.1, 'u': 0.2, 'v': 0.3},
                {'t': 0.4, 'u': 0.5, 'v': 0.6},
                {'t': 0.7, 'u': 0.8, 'v': 0.9},
            ],
            'budget_exhausted': False,
            'cells_processed': 1,
            'boundary_topology_complete': True,
        }

    monkeypatch.setattr(ncsx4, 'bez_csx_v4', overproduces)
    isolated, overlaps, status = nurbs_csx(
        curve, surface, max_cells=10, max_results=2,
        return_status=True,
    )

    assert isolated is None and overlaps is None
    assert len(status['parameter_fibers']) == 2
    assert status['results_processed'] == 2
    assert status['complete'] is False
    assert status['budget_exhausted'] is True
