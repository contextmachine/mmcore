"""L63: CSX overlap records are paired (t, u, v) ends, one per surface preimage piece."""
import numpy as np
import pytest

from examples.ssx.bilinear_l_junction_user_cases import CASES
from mmcore.nurbs._nurbs_eval import (
    NURBSCurveTuple, NURBSSurfaceTuple, evaluate_nurbs_curve, evaluate_nurbs_surface,
)
from mmcore.nurbs._nurbs_param_tol import nurbs_surface_param_tolerance
from mmcore.nurbs.nurbs_iso import extract_isocurve
from mmcore.numeric.intersection.csx import nurbs_csx

ATOL = 1e-3
_RING = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)]
_RING_KNOTS = np.array([0, 0, 0, .25, .25, .5, .5, .75, .75, 1, 1, 1.])
_RING_WEIGHTS = np.array([1. if i % 2 == 0 else np.sqrt(.5) for i in range(9)])


def _ruled(bottom, top):
    cp = np.array([[bottom(x, y), top(x, y)] for x, y in _RING], dtype=float)
    return NURBSSurfaceTuple(3, 2, _RING_KNOTS, np.array([0, 0, 1, 1.]), cp,
                             np.column_stack((_RING_WEIGHTS, _RING_WEIGHTS)))


def _cylinder():
    return _ruled(lambda x, y: (x, y, -3.), lambda x, y: (x, y, 3.))


def _cone():
    return _ruled(lambda x, y: (0., 0., 0.), lambda x, y: (x, y, 1.))


def _line(p, q):
    return NURBSCurveTuple(2, np.array([0, 0, 1, 1.]), np.array([p, q], dtype=float), np.ones(2))


def _arc(a0, a1, z):
    h, m = (a1 - a0) / 2, (a0 + a1) / 2
    cp = [(np.cos(a0), np.sin(a0), z), (np.cos(m) / np.cos(h), np.sin(m) / np.cos(h), z),
          (np.cos(a1), np.sin(a1), z)]
    return NURBSCurveTuple(3, np.array([0, 0, 0, 1, 1, 1.]), np.array(cp), np.array([1, np.cos(h), 1.]))


def _circle(a0, z):
    cp, weights = [], []
    for k in range(9):
        a, w = a0 + k * np.pi / 4, (1. if k % 2 == 0 else np.sqrt(.5))
        cp.append((np.cos(a) / (1. if k % 2 == 0 else np.sqrt(.5)),
                   np.sin(a) / (1. if k % 2 == 0 else np.sqrt(.5)), z))
        weights.append(w)
    return NURBSCurveTuple(3, _RING_KNOTS, np.array(cp), np.array(weights))


def _near(value, expected, tolerance):
    return abs(float(value) - expected) <= tolerance


def _surface_point(surface, u, v):
    return np.asarray(evaluate_nurbs_surface(surface, u, v, d_order=0)['S'])


def _ends(overlap):
    return [tuple(float(overlap[k][i]) for k in ('t_range', 'u_range', 'v_range')) for i in (0, 1)]


def _assert_ends_on_both_operands(curve, surface, overlaps):
    for overlap in overlaps:
        for t, u, v in _ends(overlap):
            point = np.asarray(evaluate_nurbs_curve(curve, t, d_order=0)['C'])
            assert np.linalg.norm(point - _surface_point(surface, u, v)) <= ATOL


@pytest.mark.parametrize('case, source, direction, side', [(1, 1, 'u', 0), (2, 1, 'u', 0), (3, 0, 'v', 1)])
@pytest.mark.parametrize('tolerance_tier', [True, False])
def test_shared_edge_on_surface_is_one_overlap_with_paired_ends(case, source, direction, side, tolerance_tier):
    surfaces = CASES[case]
    edge_surface, target = surfaces[source], surfaces[1 - source]
    bounds = edge_surface.interval()[0 if direction == 'u' else 1]
    curve = extract_isocurve(edge_surface, bounds[side], direction)
    roots, overlaps, status = nurbs_csx(curve, target, tol=ATOL, tolerance_tier=tolerance_tier)
    assert status['complete'], status
    assert roots is None
    assert overlaps is not None and len(overlaps) == 1
    assert overlaps[0]['t_range'] == curve.interval()
    _assert_ends_on_both_operands(curve, target, overlaps)


@pytest.mark.parametrize('z0, z1', [(-1., 1.), (2., 5.)])
def test_curve_lying_on_a_seam_gives_one_record_per_side(z0, z1):
    surface = _cylinder()
    curve = _line((1., 0., z0), (1., 0., z1))
    roots, overlaps, status = nurbs_csx(curve, surface, tol=ATOL)
    assert status['complete'], status
    assert roots is None
    assert overlaps is not None and len(overlaps) == 2
    ptol_u, _ = nurbs_surface_param_tolerance(surface, ATOL)
    low, high = sorted(overlaps, key=lambda o: o['u_range'][0])
    assert all(_near(u, 0., ptol_u) for u in low['u_range'])
    assert all(_near(u, 1., ptol_u) for u in high['u_range'])
    assert overlaps[0]['t_range'] == overlaps[1]['t_range']
    _assert_ends_on_both_operands(curve, surface, overlaps)


@pytest.mark.parametrize('curve', [_arc(-np.pi / 4, np.pi / 4, .7), _circle(np.pi / 2, .7)],
                         ids=['arc', 'circle_from_90deg'])
def test_curve_crossing_a_seam_is_split_exactly_at_the_seam(curve):
    surface = _cylinder()
    roots, overlaps, status = nurbs_csx(curve, surface, tol=ATOL)
    assert status['complete'], status
    assert roots is None
    assert overlaps is not None and len(overlaps) == 2
    first, second = sorted(overlaps, key=lambda o: o['t_range'][0])
    assert first['t_range'][0] == curve.interval()[0]
    assert second['t_range'][1] == curve.interval()[1]
    assert first['t_range'][1] == second['t_range'][0]
    ptol_u, _ = nurbs_surface_param_tolerance(surface, ATOL)
    assert _near(first['u_range'][1], 1., ptol_u) and _near(second['u_range'][0], 0., ptol_u)
    split = np.asarray(evaluate_nurbs_curve(curve, first['t_range'][1], d_order=0)['C'])
    assert np.linalg.norm(split[:2] - (1., 0.)) <= ATOL
    _assert_ends_on_both_operands(curve, surface, overlaps)


def test_closed_curve_starting_on_the_seam_is_one_record():
    surface = _cylinder()
    curve = _circle(0., .7)
    roots, overlaps, status = nurbs_csx(curve, surface, tol=ATOL)
    assert status['complete'], status
    assert roots is None
    assert overlaps is not None and len(overlaps) == 1
    assert overlaps[0]['t_range'] == (0., 1.)
    ptol_u, _ = nurbs_surface_param_tolerance(surface, ATOL)
    assert _near(overlaps[0]['u_range'][0], 0., ptol_u)
    assert _near(overlaps[0]['u_range'][1], 1., ptol_u)
    _assert_ends_on_both_operands(curve, surface, overlaps)


def test_curve_on_a_surface_knot_line_is_reported_once():
    surface = _cylinder()
    curve = _line((0., 1., -1.), (0., 1., 1.))
    roots, overlaps, status = nurbs_csx(curve, surface, tol=ATOL)
    assert status['complete'], status
    assert roots is None
    assert overlaps is not None and len(overlaps) == 1
    ptol_u, _ = nurbs_surface_param_tolerance(surface, ATOL)
    assert all(_near(u, .25, ptol_u) for u in overlaps[0]['u_range'])
    _assert_ends_on_both_operands(curve, surface, overlaps)


@pytest.mark.parametrize('start, end', [((0., 0., 0.), (np.sqrt(.5), np.sqrt(.5), 1.)),
                                        ((np.sqrt(.5), np.sqrt(.5), 1.), (0., 0., 0.))],
                         ids=['from_apex', 'to_apex'])
def test_overlap_ending_at_a_pole_keeps_the_continuous_surface_parameter(start, end):
    surface = _cone()
    curve = _line(start, end)
    roots, overlaps, status = nurbs_csx(curve, surface, tol=ATOL)
    assert status['complete'], status
    assert roots is None
    assert overlaps is not None and len(overlaps) == 1
    u_start, u_end = overlaps[0]['u_range']
    assert np.linalg.norm(_surface_point(surface, u_start, 1.) - _surface_point(surface, u_end, 1.)) <= ATOL
    v_start, v_end = overlaps[0]['v_range']
    assert (v_start < v_end) == (start == (0., 0., 0.))
    _assert_ends_on_both_operands(curve, surface, overlaps)


def test_generator_on_the_seam_of_a_cone_gives_one_record_per_side():
    surface = _cone()
    curve = _line((0., 0., 0.), (1., 0., 1.))
    roots, overlaps, status = nurbs_csx(curve, surface, tol=ATOL)
    assert status['complete'], status
    assert roots is None
    assert overlaps is not None and len(overlaps) == 2
    ptol_u, _ = nurbs_surface_param_tolerance(surface, ATOL)
    low, high = sorted(overlaps, key=lambda o: o['u_range'][1])
    assert all(_near(u, 0., ptol_u) for u in low['u_range'])
    assert all(_near(u, 1., ptol_u) for u in high['u_range'])
    _assert_ends_on_both_operands(curve, surface, overlaps)
