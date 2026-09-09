"""A tangential component cannot consume a disjoint regular component."""
from fractions import Fraction as F
from math import comb

import numpy as np

from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx


def mixed_pair():
    # z=3*(t-3/4)^2*((s-3/8)^2+(t-1/4)^2-(1/8)^2).
    # Multiplication by3 makes ALL degree(2,4) Bernstein coefficients
    # dyadic. Thus the supplied binary net, not just an intended rounded
    # polynomial, has exactly the repeated line and regular circle.
    line = {(0, 0): F(9, 16), (0, 1): F(-3, 2), (0, 2): F(1)}
    circle = {(0, 0): F(3, 16), (1, 0): F(-3, 4), (2, 0): F(1),
              (0, 1): F(-1, 2), (0, 2): F(1)}
    power = {}
    for (i, j), a in line.items():
        for (k, l), b in circle.items():
            power[i+k, j+l] = power.get((i+k, j+l), F(0)) + 3*a*b
    surface = np.empty((3, 5, 3))
    for i in range(3):
        for j in range(5):
            z = sum(a*F(comb(i, k), comb(2, k))*F(comb(j, l), comb(4, l))
                    for (k, l), a in power.items() if k <= i and l <= j)
            assert F.from_float(float(z)) == z
            surface[i, j] = [i/2., j/4., float(z)]
    plane = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    return surface, plane


def test_repeated_line_preserves_disjoint_regular_circle():
    from examples.ssx.ssx5_analytic_audit import distances_to_polylines
    a, b = mixed_pair()
    atol = 1e-3
    result = bez_ssx(a, b, atol=atol, rational=False)
    loops, lines = [], []
    for branch in result['branches']:
        xyz = np.asarray(branch.curve[1])
        if np.max(np.abs(xyz[:, 1]-.75)) < 4*atol:
            lines.append(xyz)
        elif np.max(np.abs(np.linalg.norm(xyz[:, :2]-[.375, .25], axis=1)-.125)) < 4*atol:
            assert branch.closed
            loops.append(xyz)
    assert len(loops) == 1, result['status']
    theta = np.linspace(0., 2*np.pi, 2049)
    reference = np.column_stack((.375+.125*np.cos(theta),
                                  .25+.125*np.sin(theta), np.zeros_like(theta)))
    assert distances_to_polylines(reference, loops).max() < 4*atol
    length = sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in loops)
    assert abs(length-2*np.pi*.125) < 8*np.pi*atol
    line_reference = np.column_stack((np.linspace(0., 1., 1001),
                                       np.full(1001, .75), np.zeros(1001)))
    assert distances_to_polylines(line_reference, lines).max() < 4*atol
