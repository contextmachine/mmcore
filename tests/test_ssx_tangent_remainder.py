"""Tracing one tangency must retain the search for other curve components."""
from math import comb

import numpy as np

from examples.ssx.ssx5_analytic_audit import distances_to_polylines
from mmcore.numeric._bezier_common import eval_surface
from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx


def _line_and_circle():
    # Independent polynomial construction: z=3*(t-.5)^2*circle(s,t).
    # The circle lies strictly below, and is disjoint from, the tangent line.
    cx, cy, radius = .3125, .1875, .125
    circle = {(0, 0): cx*cx + cy*cy - radius*radius,
              (1, 0): -2*cx, (2, 0): 1., (0, 1): -2*cy, (0, 2): 1.}
    line = {(0, 0): .25, (0, 1): -1., (0, 2): 1.}
    powers = np.zeros((3, 5))
    for (i, j), a in circle.items():
        for (k, l), b in line.items():
            powers[i+k, j+l] += 3*a*b
    graph = np.array([[[i/2, j/4,
        sum(powers[k, l]*comb(i, k)/comb(2, k)*comb(j, l)/comb(4, l)
            for k in range(i+1) for l in range(j+1))]
        for j in range(5)] for i in range(3)])
    plane = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    angle = np.linspace(0., 2*np.pi, 2049)
    ring = np.column_stack((cx+radius*np.cos(angle), cy+radius*np.sin(angle),
                            np.zeros_like(angle)))
    tangent = np.column_stack((np.linspace(0., 1., 1001), np.full(1001, .5), np.zeros(1001)))
    return graph, plane, ring, tangent


def test_tangent_line_does_not_consume_a_disjoint_regular_circle():
    atol = 1e-3
    graph, plane, ring, tangent = _line_and_circle()
    result = bez_ssx(graph, plane, atol=atol, rational=False)
    branches = result['branches']
    assert len(branches) == 2
    assert sum(branch.closed and branch.kind == 'transversal' for branch in branches) == 1
    assert sum(not branch.closed and branch.kind == 'tangential' for branch in branches) == 1
    ring_path = next(np.asarray(b.curve[1]) for b in branches if b.closed)
    line_path = next(np.asarray(b.curve[1]) for b in branches if not b.closed)
    # General continuation has always allowed 2*atol chord sagitta.
    # This checks the polyline representation, not the source accuracy of
    # its vertices. A straight tangent line needs no sagitta allowance.
    assert distances_to_polylines(ring, [ring_path]).max() <= 2*atol
    assert distances_to_polylines(tangent, [line_path]).max() <= atol
    for branch in branches:
        points = np.asarray(branch.curve[1])
        circle_distance = abs(np.linalg.norm(points[:, :2] - [.3125, .1875], axis=1) - .125)
        line_distance = abs(points[:, 1] - .5)
        assert np.maximum(np.minimum(circle_distance, line_distance), abs(points[:, 2])).max() <= atol
        for q, point in zip(branch.curve[0], points):
            assert np.linalg.norm(eval_surface(graph, *q[:2], rational=False) - point) <= atol
            assert np.linalg.norm(eval_surface(plane, *q[2:], rational=False) - point) <= atol


def test_whole_known_tangent_line_can_resolve_before_depth_limit(monkeypatch):
    import mmcore.numeric.intersection.ssx._bez_ssx5 as ssx
    # Exercise the general engine, independently of its optional isoline
    # reduction. The whole contact is already represented at this depth.
    monkeypatch.setattr(ssx, '_try_isoline_intersection', lambda *args, **kwargs: None)
    graph = np.array([[[s, t, z] for t, z in zip(
        (0., .5, 1.), (.25, -.25, .25))] for s in (0., 1.)])
    plane = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    result = ssx.bez_ssx(graph, plane, atol=1e-3, rational=False, max_depth=0)
    assert len(result['branches']) == 1
    assert result['branches'][0].kind == 'tangential'
    assert 'depth_limit' not in result['status']['reasons']
    reference = np.column_stack((np.linspace(0., 1., 101), np.full(101, .5), np.zeros(101)))
    assert distances_to_polylines(reference, [result['branches'][0].curve[1]]).max() <= 1e-3
