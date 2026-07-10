"""Tests for nurbs_ccx and nurbs_ccx_multiple from _nccx4.py.

Verifies correctness (no false positives, no missed intersections),
deduplication (no span-boundary duplicates), and ground-truth match.
"""
import json
import numpy as np
import pytest

from mmcore.geom._nurbs_eval import NURBSCurveTuple, evaluate_nurbs_curve
from mmcore.numeric.intersection.ccx._nccx4 import nurbs_ccx, nurbs_ccx_multiple
import mmcore.numeric.intersection.ccx._nccx4 as nccx4


# ---------------------------------------------------------------------------
# Fixtures: 2D rational ellipses + polynomial spline
# ---------------------------------------------------------------------------

CURVES_2D = [
    NURBSCurveTuple(
        order=3,
        knot=np.array([0., 0., 0., 60.684, 60.684, 121.368, 121.368,
                       182.051, 182.051, 242.735, 242.735, 242.735]),
        control_points=np.array([
            [161.01, 95.097], [175.987, 130.709], [140.376, 145.685],
            [104.764, 160.662], [89.788, 125.051], [74.811, 89.439],
            [110.422, 74.463], [146.034, 59.486], [161.01, 95.097]]),
        weights=np.array([1., 0.707, 1., 0.707, 1., 0.707, 1., 0.707, 1.])
    ),
    NURBSCurveTuple(
        order=3,
        knot=np.array([0., 0., 0., 1.571, 1.571, 3.142, 3.142, 4.712, 4.712,
                       6.283, 6.283, 6.283]),
        control_points=np.array([
            [141.707, 107.412], [114.416, 107.412], [114.416, 183.626],
            [114.416, 259.841], [141.707, 259.841], [168.998, 259.841],
            [168.998, 183.626], [168.998, 107.412], [141.707, 107.412]]),
        weights=np.array([1., 0.707, 1., 0.707, 1., 0.707, 1., 0.707, 1.])
    ),
    NURBSCurveTuple(
        order=3,
        knot=np.array([0., 0., 0., 76.691, 76.691, 153.382, 153.382,
                       230.074, 230.074, 306.765, 306.765, 306.765]),
        control_points=np.array([
            [132.055, 206.591], [141.374, 254.516], [93.449, 263.835],
            [45.523, 273.154], [36.204, 225.229], [26.885, 177.303],
            [74.811, 167.984], [122.736, 158.665], [132.055, 206.591]]),
        weights=np.array([1., 0.707, 1., 0.707, 1., 0.707, 1., 0.707, 1.])
    ),
    NURBSCurveTuple(
        order=4,
        knot=np.array([0., 0., 0., 0., 123.163, 246.326, 369.489,
                       492.652, 615.815, 615.815, 615.815, 615.815]),
        control_points=np.array([
            [3.588, 128.046], [20.895, 198.27], [67.489, 121.723],
            [88.124, 203.595], [133.054, 164.656], [209.934, 231.552],
            [87.791, 295.453], [71.15, 227.225]]),
        weights=np.array([1., 1., 1., 1., 1., 1., 1., 1.])
    ),
]


# ---------------------------------------------------------------------------
# Fixtures: 3D polynomial grid + high-degree rational (from multiple_int_3d)
# ---------------------------------------------------------------------------

def _load_3d_curves():
    """Load the 11-curve 3D test data from the example file."""
    import importlib.util
    # Parse the val list from the example file without executing imports
    with open("examples/ccx/multiple_int_3d.py") as f:
        src = f.read()
    code = src.split("from mmcore.numeric.intersection.ccx import")[0]
    ns = {"np": np, "NURBSCurveTuple": NURBSCurveTuple}
    exec(code, ns)
    return ns["val"]


# ---------------------------------------------------------------------------
# Tests: 2D example — no false positives
# ---------------------------------------------------------------------------

class TestNurbsCCXMultiple2D:
    """2D rational ellipses + polynomial spline."""

    def test_no_false_positives(self):
        """Every reported intersection must have dist < atol between the curves."""
        iso, ovl = nurbs_ccx_multiple(CURVES_2D, tol=0.001, rational=True)
        assert iso is not None
        for entry in iso:
            c1i, c2i = int(entry['curve1_i']), int(entry['curve2_i'])
            pt1 = evaluate_nurbs_curve(CURVES_2D[c1i], float(entry['u']), 0)['C']
            pt2 = evaluate_nurbs_curve(CURVES_2D[c2i], float(entry['v']), 0)['C']
            dist = float(np.linalg.norm(pt1 - pt2))
            assert dist < 0.001, (
                f"False positive: c{c1i}xc{c2i} u={entry['u']:.6f} "
                f"v={entry['v']:.6f} dist={dist:.4f}"
            )

    def test_count(self):
        """Should find exactly 11 intersections (matching the old algorithm)."""
        iso, ovl = nurbs_ccx_multiple(CURVES_2D, tol=0.001, rational=True)
        assert iso is not None
        assert len(iso) == 11


# ---------------------------------------------------------------------------
# Tests: 3D example — ground truth + dedup
# ---------------------------------------------------------------------------

class TestNurbsCCXMultiple3D:
    """3D polynomial grid from multiple_int_3d.py."""

    @pytest.fixture(scope="class")
    def curves_3d(self):
        return _load_3d_curves()

    @pytest.fixture(scope="class")
    def expected(self):
        with open("tests/expected_nurbs_ccx_01.json") as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def result(self, curves_3d):
        return nurbs_ccx_multiple(curves_3d, tol=0.001, rational=True)

    def test_ground_truth(self, result, expected):
        """All 25 known intersections (excl curve 0) must be found."""
        iso, ovl = result
        assert iso is not None
        iso_filt = [i for i in iso if i['curve1_i'] != 0 and i['curve2_i'] != 0]
        for exp in expected:
            pt_exp = np.array(exp['point'])
            found = any(
                np.linalg.norm(i['point'] - pt_exp) < 0.05 for i in iso_filt
            )
            assert found, (
                f"Missing: c{exp['curve_a']}xc{exp['curve_b']} "
                f"u={exp['u']:.4f} v={exp['v']:.4f} pt={exp['point']}"
            )

    def test_no_span_boundary_duplicates(self, result, expected):
        """Excluding curve 0, raw count should equal unique count (no duplicates)."""
        iso, ovl = result
        iso_filt = [i for i in iso if i['curve1_i'] != 0 and i['curve2_i'] != 0]
        assert len(iso_filt) == 25

    def test_no_false_positives(self, result, curves_3d):
        """Every reported intersection must have dist < atol."""
        iso, ovl = result
        for entry in iso:
            c1i, c2i = int(entry['curve1_i']), int(entry['curve2_i'])
            pt1 = evaluate_nurbs_curve(curves_3d[c1i], float(entry['u']), 0)['C']
            pt2 = evaluate_nurbs_curve(curves_3d[c2i], float(entry['v']), 0)['C']
            dist = float(np.linalg.norm(pt1 - pt2))
            assert dist < 0.001, (
                f"False positive: c{c1i}xc{c2i} u={entry['u']:.6f} "
                f"v={entry['v']:.6f} dist={dist:.4f}"
            )


# ---------------------------------------------------------------------------
# Tests: nurbs_ccx (two-curve)
# ---------------------------------------------------------------------------

class TestNurbsCCXPair:
    """Test the two-curve nurbs_ccx function."""

    def test_two_ellipses(self):
        """Two rational ellipses that intersect."""
        iso, ovl = nurbs_ccx(CURVES_2D[0], CURVES_2D[1], tol=0.001)
        assert iso is not None
        assert len(iso) >= 1
        # Verify all results
        for entry in iso:
            pt1 = evaluate_nurbs_curve(CURVES_2D[0], float(entry['u']), 0)['C']
            pt2 = evaluate_nurbs_curve(CURVES_2D[1], float(entry['v']), 0)['C']
            assert np.linalg.norm(pt1 - pt2) < 0.001

    def test_no_intersection(self):
        """Two curves that don't intersect."""
        c1 = NURBSCurveTuple(
            order=2,
            knot=np.array([0., 0., 1., 1.]),
            control_points=np.array([[0., 0., 0.], [1., 0., 0.]]),
            weights=np.array([1., 1.]),
        )
        c2 = NURBSCurveTuple(
            order=2,
            knot=np.array([0., 0., 1., 1.]),
            control_points=np.array([[0., 10., 0.], [1., 10., 0.]]),
            weights=np.array([1., 1.]),
        )
        iso, ovl = nurbs_ccx(c1, c2, tol=0.001)
        assert iso is None


# ---------------------------------------------------------------------------
# Tests: bounded Bezier solver status must not be lost at the NURBS boundary
# ---------------------------------------------------------------------------

def _overlapping_lines():
    c1 = NURBSCurveTuple(
        order=2,
        knot=np.array([0., 0., 1., 1.]),
        control_points=np.array([[0., 0., 0.], [1., 0., 0.]]),
        weights=np.ones(2),
    )
    c2 = NURBSCurveTuple(
        order=2,
        knot=np.array([0., 0., 1., 1.]),
        control_points=np.array([[0., 0., 0.], [1., 0., 0.]]),
        weights=np.ones(2),
    )
    return c1, c2


def _partial_ccx_result(*args, **kwargs):
    return {
        'isolated': [],
        'overlaps': [],
        'budget_exhausted': True,
        'cells_processed': 7,
        'boundary_topology_complete': False,
    }


def test_nurbs_ccx_rejects_silent_partial_result(monkeypatch):
    c1, c2 = _overlapping_lines()
    monkeypatch.setattr(nccx4, 'bez_ccx_v4', _partial_ccx_result)

    with pytest.raises(RuntimeError, match='incomplete Bezier CCX'):
        nurbs_ccx(c1, c2)

    isolated, overlaps, status = nurbs_ccx(c1, c2, return_status=True)
    assert isolated is None
    assert overlaps is None
    assert status['budget_exhausted'] is True
    assert status['boundary_topology_complete'] is False
    assert status['cells_processed'] == 7
    assert status['partial_results'] == 1


def test_nurbs_ccx_multiple_rejects_silent_partial_result(monkeypatch):
    c1, c2 = _overlapping_lines()
    monkeypatch.setattr(nccx4, 'bez_ccx_v4', _partial_ccx_result)

    with pytest.raises(RuntimeError, match='incomplete Bezier CCX'):
        nurbs_ccx_multiple([c1, c2])

    isolated, overlaps, status = nurbs_ccx_multiple(
        [c1, c2], return_status=True,
    )
    assert isolated is None
    assert overlaps is None
    assert status['budget_exhausted'] is True
    assert status['partial_results'] >= 1


def test_nurbs_ccx_shares_max_cells_across_all_span_pairs(monkeypatch):
    calls = []

    def complete_but_spends_allowance(*_args, **kwargs):
        allowance = int(kwargs['max_cells'])
        calls.append(allowance)
        return {
            'isolated': [],
            'overlaps': [],
            'budget_exhausted': False,
            'cells_processed': allowance,
            'boundary_topology_complete': True,
        }

    monkeypatch.setattr(nccx4, 'bez_ccx_v4', complete_but_spends_allowance)
    isolated, overlaps, status = nurbs_ccx(
        CURVES_2D[0], CURVES_2D[0], max_cells=3,
        return_status=True,
    )

    assert isolated is None and overlaps is None
    assert calls == [3]
    assert status['cells_processed'] == 3
    assert status['max_cells'] == 3
    assert status['complete'] is False
    assert status['budget_exhausted'] is True


def test_nurbs_ccx_multiple_shares_max_cells_across_candidates(monkeypatch):
    calls = []

    def complete_but_spends_allowance(*_args, **kwargs):
        allowance = int(kwargs['max_cells'])
        calls.append(allowance)
        return {
            'isolated': [],
            'overlaps': [],
            'budget_exhausted': False,
            'cells_processed': allowance,
            'boundary_topology_complete': True,
        }

    monkeypatch.setattr(nccx4, 'bez_ccx_v4', complete_but_spends_allowance)
    isolated, overlaps, status = nurbs_ccx_multiple(
        [CURVES_2D[0], CURVES_2D[0]], max_cells=2,
        return_status=True,
    )

    assert isolated is None and overlaps is None
    assert calls == [2]
    assert status['cells_processed'] == 2
    assert status['complete'] is False
    assert status['budget_exhausted'] is True


def test_nurbs_ccx_shares_max_results_across_span_pairs(monkeypatch):
    calls = []

    def one_overlap(*_args, **kwargs):
        calls.append(int(kwargs['max_results']))
        return {
            'isolated': [],
            'overlaps': [{
                'u_range': (0.0, 1.0),
                'v_range': (0.0, 1.0),
            }],
            'budget_exhausted': False,
            'cells_processed': 0,
            'boundary_topology_complete': True,
        }

    monkeypatch.setattr(nccx4, 'bez_ccx_v4', one_overlap)
    _isolated, overlaps, status = nurbs_ccx(
        CURVES_2D[0], CURVES_2D[0], max_cells=100,
        max_results=2, return_status=True,
    )

    assert calls == [2, 1]
    assert overlaps is not None and len(overlaps) == 2
    assert status['results_processed'] == 2
    assert status['max_results'] == 2
    assert status['complete'] is False
    assert status['budget_exhausted'] is True


def test_nurbs_ccx_sanitizes_one_span_to_remaining_result_budget(monkeypatch):
    c1, c2 = _overlapping_lines()

    def overproduces(*_args, **_kwargs):
        return {
            'isolated': [],
            'overlaps': [
                {'u_range': (0.0, 0.2), 'v_range': (0.0, 0.2)},
                {'u_range': (0.3, 0.5), 'v_range': (0.3, 0.5)},
                {'u_range': (0.6, 0.8), 'v_range': (0.6, 0.8)},
            ],
            'budget_exhausted': False,
            'cells_processed': 1,
            'boundary_topology_complete': True,
        }

    monkeypatch.setattr(nccx4, 'bez_ccx_v4', overproduces)
    _isolated, overlaps, status = nurbs_ccx(
        c1, c2, max_cells=10, max_results=2, return_status=True,
    )

    assert overlaps is not None and len(overlaps) == 2
    assert status['results_processed'] == 2
    assert status['complete'] is False
    assert status['budget_exhausted'] is True
