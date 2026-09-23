"""Finding a tangent component must preserve the regular remainder of a cell."""
from math import comb

import numpy as np
import pytest

from examples.ssx.ssx5_analytic_audit import distances_to_polylines
from mmcore.numeric._bezier_common import eval_surface
from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx


def _line_and_circles(circles):
    # Independent polynomial construction: y=.5 is a repeated component;
    # each disjoint circle is a regular zero of its own quadratic factor.
    power = {(0, 0): .75, (0, 1): -3., (0, 2): 3.}
    for x, y, radius in circles:
        factor = {(0, 0): x*x+y*y-radius*radius, (1, 0): -2*x,
                  (2, 0): 1., (0, 1): -2*y, (0, 2): 1.}
        product = {}
        for (i, j), a in power.items():
            for (k, l), b in factor.items():
                product[i+k, j+l] = product.get((i+k, j+l), 0.)+a*b
        power = product
    nu = max(i for i, _ in power)
    nv = max(j for _, j in power)
    surface = np.empty((nu+1, nv+1, 3))
    for i in range(nu+1):
        for j in range(nv+1):
            z = sum(value*comb(i, k)/comb(nu, k)*comb(j, l)/comb(nv, l)
                    for (k, l), value in power.items() if k <= i and l <= j)
            surface[i, j] = [i/nu, j/nv, z]
    plane = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    return surface, plane


@pytest.mark.parametrize('variant', ['identity', 'swap', 'reverse_v', 'two_loops'])
def test_tangent_line_through_cell_center_preserves_every_regular_loop(variant):
    atol = 1e-3
    circles = [(.3125, .1875, .125)]
    if variant == 'two_loops':
        circles.append((.75, .8125, .09375))
    first, second = _line_and_circles(circles)
    if variant == 'swap':
        first, second = second, first
    elif variant == 'reverse_v':
        first, second = first[:, ::-1].copy(), second[:, ::-1].copy()
    result = bez_ssx(first, second, atol=atol, rational=False)
    branches = result['branches']
    assert len(branches) == len(circles)+1, result['status']
    lines = [b for b in branches if not b.closed]
    loops = [b for b in branches if b.closed]
    assert len(lines) == 1 and len(loops) == len(circles)
    line = np.column_stack((np.linspace(0., 1., 1025),
                            np.full(1025, .5), np.zeros(1025)))
    assert distances_to_polylines(line, [lines[0].curve[1]]).max() <= 4*atol
    for x, y, radius in circles:
        theta = np.linspace(0., 2*np.pi, 2049)
        reference = np.column_stack((x+radius*np.cos(theta),
                                     y+radius*np.sin(theta), np.zeros_like(theta)))
        candidates = [np.asarray(b.curve[1]) for b in loops
                      if np.max(np.abs(np.linalg.norm(
                          np.asarray(b.curve[1])[:, :2]-[x, y], axis=1)-radius)) <= 4*atol]
        assert len(candidates) == 1
        assert distances_to_polylines(reference, candidates).max() <= 4*atol
        length = np.linalg.norm(np.diff(candidates[0], axis=0), axis=1).sum()
        assert abs(length-2*np.pi*radius) <= 8*np.pi*atol
    for branch in branches:
        for q, point in zip(*branch.curve):
            a = eval_surface(first, *q[:2], rational=False)
            b = eval_surface(second, *q[2:], rational=False)
            assert np.linalg.norm(a-point) <= atol
            assert np.linalg.norm(b-point) <= atol
