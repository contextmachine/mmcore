"""Standalone endpoint events coalesce only through source root identity."""
import numpy as np

from mmcore.numeric._work_budget import SoftWorkBudget
from mmcore.numeric.intersection.ssx._bez_ssx5 import (
    BoundaryPoint, _registered_point, _remove_registered_endpoint_points,
    _remove_represented_vertex_points, _remove_lifted_polyline_points)
from mmcore.numeric.intersection.ssx._ssx_root_identity import BoundaryRootIdentity
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXPoint


def _h(net):
    return np.concatenate((net, np.ones(net.shape[:2]+(1,))), axis=2)


def _branch(owner, end):
    branch = SSXBranch((np.array([owner.stuv, end]),
                        np.array([owner.xyz, [end[0], end[1], 0.]])))
    branch._registered_root_points = {id(owner): owner}
    return branch


def test_same_source_endpoint_with_independent_representatives_is_coalesced():
    a = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    b = np.array([[[u, v, v-.5] for v in (0., 1.)] for u in (0., 1.)])
    identity = BoundaryRootIdentity(_h(a), _h(b))
    first = BoundaryPoint(np.array([.25, .5, .25+1e-15, .5+1e-15]),
                          np.array([.25, .5, 0.]), (0, -1))
    second = BoundaryPoint(np.array([.25+1e-15, .5-1e-15, .25, .5]),
                           np.array([.25, .5, 0.]), (2, -1))
    point = _registered_point(first)
    assert point._registered_root is first
    result = _remove_registered_endpoint_points(
        [point], [_branch(second, [.75, .5, .75, .5])], np.full(4, .001), .001, identity)
    assert result == []


def test_close_distinct_boundary_roots_are_not_absorbed_by_endpoint():
    left, right = .5-2.**-20, .5+2.**-20
    z = [left*right, left*right-.5, (1-left)*(1-right)]
    a = np.array([[[s, t, z[i]] for t in (0., 1.)] for i, s in enumerate((0., .5, 1.))])
    b = np.array([[[u, v, 0.] for v in (0., 1.)] for u in (0., 1.)])
    identity = BoundaryRootIdentity(_h(a), _h(b))
    roots = [BoundaryPoint(np.array([x, 0., x, 0.]), np.array([x, 0., 0.]), (1, 0))
             for x in (left, right)]
    point = _registered_point(roots[1])
    kept = _remove_registered_endpoint_points(
        [point], [_branch(roots[0], [left, 1., left, 1.])], np.full(4, .001), .001, identity)
    assert len(kept) == 1 and kept[0] is point


def test_denied_endpoint_identity_work_preserves_every_standalone_point():
    owner = BoundaryPoint(np.full(4, .5), np.zeros(3), (0, -1))
    points = [_registered_point(BoundaryPoint(np.full(4, .5), np.zeros(3), (0, -1))) for _ in range(3)]
    budget = SoftWorkBudget(100, 100, max_postprocess_work=0)
    def forbidden(*_):
        raise AssertionError('Certificate ran after denied postprocess work')
    kept = _remove_registered_endpoint_points(
        points, [_branch(owner, np.ones(4))], np.full(4, .001), .001, forbidden, budget)
    assert len(kept) == len(points) and all(a is b for a, b in zip(kept, points))


def test_exact_vertex_index_avoids_quadratic_point_segment_price():
    values = np.linspace(0., 1., 500)
    q = np.column_stack((values, values, values, values))
    xyz = np.column_stack((values, values, values*0.))
    branch = SSXBranch((q, xyz))
    points = [SSXPoint(stuv=a.copy(), xyz=b.copy()) for a, b in zip(q, xyz)]
    budget = SoftWorkBudget(100, 100, max_postprocess_work=2000)
    assert _remove_represented_vertex_points(points, [branch], .001, budget) == []
    assert budget.result_fields()['complete']


def test_vertex_index_requires_exact_lifted_parameters_and_xyz_guard():
    q = np.array([[.5, 0., .5, 0.], [.5, 1., .5, 1.]])
    xyz = np.array([[.5, 0., 0.], [.5, 1., 0.]])
    branch = SSXBranch((q, xyz))
    nearby = q[0].copy()
    nearby[0] = np.nextafter(nearby[0], 1.)
    points = [SSXPoint(nearby, xyz[0].copy()), SSXPoint(q[0].copy(), np.array([.5, .5, 0.]))]
    kept = _remove_represented_vertex_points(points, [branch], .001)
    assert len(kept) == 2 and all(a is b for a, b in zip(kept, points))


def test_parameter_aabb_index_preserves_same_containment_with_sparse_work():
    values = np.arange(1025)/1024.
    q = np.column_stack((values, values, values, values))
    xyz = np.column_stack((values, values, values*0.))
    branch = SSXBranch((q, xyz))
    middle_q, middle_xyz = .5*(q[:-1]+q[1:]), .5*(xyz[:-1]+xyz[1:])
    points = [SSXPoint(a.copy(), b.copy()) for a, b in zip(middle_q, middle_xyz)]
    # The old Cartesian precharge was1,048,576 units; this tree prices its
    # construction plus actually visited nodes/candidate segments.
    budget = SoftWorkBudget(100, 100, max_postprocess_work=40000)
    assert _remove_lifted_polyline_points(points, [branch], .001, budget) == []
    assert budget.result_fields()['complete']


def test_parameter_aabb_index_keeps_a_different_preimage_inside_model_tolerance():
    q = np.array([[.5, 0., .5, 0.], [.5, 1., .5, 1.]])
    xyz = np.array([[.5, 0., 0.], [.5, 1., 0.]])
    branch = SSXBranch((q, xyz))
    point = SSXPoint(np.array([np.nextafter(.5, 1.), .5, .5, .5]), np.array([.5, .5, 0.]))
    kept = _remove_lifted_polyline_points([point], [branch], .001)
    assert len(kept) == 1 and kept[0] is point
