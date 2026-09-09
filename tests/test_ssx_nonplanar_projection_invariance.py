"""Entire known curved intersections survive an exact change of world basis."""
import numpy as np
import pytest

from examples.ssx.ssx5_analytic_audit import distances_to_polylines
from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx


MATRICES = (
    np.eye(3),
    np.array([[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]]),
    np.array([[1., 2., 1.], [0., 1., 1.], [1., 1., 1.]]),
)


def curved_graph_pair():
    # Both charts are curved. Their exact difference is the circle
    # 4*(s-1/2)^2+4*(t-1/2)^2-1/2, with s=u,t=v. Every coefficient
    # and every matrix entry is a small dyadic, so the world changes
    # preserve these source identities exactly, including the z=s*t lift.
    square = (1., -1., 1.)
    first = np.array([[[i/2, j/2, i*j/4] for j in range(3)] for i in range(3)])
    second = first.copy()
    second[..., 2] += [[square[i]+square[j]-.5 for j in range(3)] for i in range(3)]
    return first, second


@pytest.mark.parametrize('matrix', MATRICES, ids=['identity', 'mixed', 'sheared'])
@pytest.mark.parametrize('swap', [False, True], ids=['ab', 'ba'])
def test_curved_graph_intersection_covers_the_whole_source_circle(matrix, swap):
    first, second = (source@matrix.T for source in curved_graph_pair())
    if swap:
        first, second = second, first
    atol = 1e-3
    result = bez_ssx(first, second, atol, rational=False, max_cells=60_000)
    assert result['complete'], result['status']
    assert len(result['branches']) == 1
    branch = result['branches'][0]
    assert branch.closed
    xyz = np.asarray(branch.curve[1])
    angle = np.linspace(0., 2*np.pi, 4097)
    radius = np.sqrt(1/8)
    s, t = .5+radius*np.cos(angle), .5+radius*np.sin(angle)
    reference = np.column_stack((s, t, s*t))@matrix.T
    assert distances_to_polylines(reference, [xyz]).max() < 4*atol
    # Count full angular travel, not vertex density. A partial loop,
    # duplicate traversal, or backtracking cannot satisfy this invariant.
    source_xyz = np.linalg.solve(matrix, xyz.T).T
    theta = np.arctan2(source_xyz[:, 1]-.5, source_xyz[:, 0]-.5)
    increments = np.arctan2(np.sin(np.diff(theta)), np.cos(np.diff(theta)))
    assert abs(abs(increments.sum())-2*np.pi) < 1e-6
    assert abs(np.abs(increments).sum()-2*np.pi) < 1e-6
    expected_length = np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()
    output_length = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
    assert abs(output_length-expected_length) < 8*np.pi*atol
    assert not result.get('points') and not result.get('singularities')
