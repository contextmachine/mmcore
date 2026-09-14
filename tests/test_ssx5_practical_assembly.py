"""Geometric controls for numerical paired-parameter assembly."""

import numpy as np
import pytest

from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx


def _fragment(x, y=None):
    x = np.asarray(x, dtype=float)
    y = np.full(len(x), 0.25) if y is None else np.asarray(y, dtype=float)
    parameters = np.column_stack((x, y, x, y))
    xyz = np.column_stack((x, y, np.zeros(len(x))))
    return ssx._Fragment(None, None, parameters, xyz)


@pytest.mark.parametrize("reverse", [False, True])
def test_same_path_with_different_sampling_coalesces(reverse):
    first = _fragment([0.0, 0.5, 1.0])
    second = _fragment(np.linspace(0.0, 1.0, 9))
    if reverse:
        second.stuv_path = second.stuv_path[::-1]
        second.xyz_path = second.xyz_path[::-1]
    kept = ssx._drop_duplicate_fragments([first, second], 1e-3,
                                         param_tol=np.full(4, 1e-4))
    assert len(kept) == 1
    assert np.linalg.norm(np.diff(kept[0].xyz_path, axis=0), axis=1).sum() == pytest.approx(1.0)


def test_shared_endpoints_do_not_hide_resolved_lens_interior():
    straight = _fragment([0.0, 1.0], [0.0, 0.0])
    arch = _fragment([0.0, 0.5, 1.0], [0.0, 0.05, 0.0])
    assert len(ssx._drop_duplicate_fragments([straight, arch], 1e-3)) == 2
    # The final small-branch cleanup must apply the same full-path test.
    result = ssx._assemble_fragments([straight, arch], atol_full=1e-3)
    assert len(result) == 2


def test_endpoint_sweep_keeps_metric_distinct_crossings():
    first = _fragment([0.0, 0.5])
    second = _fragment([0.5 + 1e-6, 1.0])
    first.end_point = ssx.BoundaryPoint(first.stuv_path[-1], first.xyz_path[-1], (0, -1))
    # A large local derivative makes this close parameter a different
    # geometric crossing. The sorted broadphase must retain the XYZ guard.
    second.xyz_path[0, 2] = 0.01
    second.start_point = ssx.BoundaryPoint(second.stuv_path[0], second.xyz_path[0], (0, -1))
    ssx._unify_fragment_endpoints([first, second], np.full(4, 1e-4), unify_atol=1e-3)
    assert first.end_point is not second.start_point


def test_endpoint_sweep_stitches_numerically_matching_crossings():
    first = _fragment([0.0, 0.5])
    second = _fragment([0.5 + 1e-6, 1.0])
    first.end_point = ssx.BoundaryPoint(first.stuv_path[-1], first.xyz_path[-1], (0, -1))
    second.start_point = ssx.BoundaryPoint(second.stuv_path[0], second.xyz_path[0], (0, -1))
    ssx._unify_fragment_endpoints([first, second], np.full(4, 1e-4), unify_atol=1e-3)
    assert first.end_point is second.start_point


def test_small_world_junction_does_not_join_different_parameter_sheets():
    first = _fragment([0.1, 0.5])
    second = _fragment([0.5, 0.9])
    second.stuv_path[:, 0] += 0.1
    result = ssx._assemble_fragments([first, second], atol_full=1e-3)
    assert len(result) == 2


def test_tolerance_closed_chain_includes_final_chord():
    fragment = _fragment([0.2, 0.8, 0.5, 0.2 + 1e-7],
                         [0.2, 0.2, 0.8, 0.2])
    result = ssx._assemble_fragments([fragment], unify_tol=np.full(4, 1e-5))
    assert len(result) == 1 and result[0].closed
    np.testing.assert_array_equal(result[0].curve[0][0], result[0].curve[0][-1])
    np.testing.assert_array_equal(result[0].curve[1][0], result[0].curve[1][-1])


def test_endpoint_clusters_cannot_drift_through_near_neighbors():
    fragments = []
    original_xyz = []
    for i in range(11):
        q = np.array([i * 1e-4, 0.0, 0.0, 0.0])
        xyz = np.array([i * 1.5e-3, 0.0, 0.0])
        point = ssx.BoundaryPoint(q, xyz, (0, -1))
        fragments.append(ssx._Fragment(point, None, np.array([q]), np.array([xyz])))
        original_xyz.append(xyz)
    ssx._unify_fragment_endpoints(fragments, np.full(4, 1e-3), unify_atol=1e-3)
    clusters = {}
    for fragment, xyz in zip(fragments, original_xyz):
        clusters.setdefault(id(fragment.start_point), []).append(xyz)
    assert len(clusters) > 1
    for xyz in clusters.values():
        assert np.linalg.norm(np.ptp(xyz, axis=0)) <= 2e-3


def test_parameter_curvature_credit_cannot_move_a_constant_axis():
    first = _fragment([0.0, 0.5, 1.0], [0.2, 0.8, 0.2])
    first.stuv_path[:, 0] = 0.25
    second = ssx._Fragment(None, None, first.stuv_path.copy(), first.xyz_path.copy())
    second.stuv_path[:, 0] += 0.1
    assert len(ssx._drop_duplicate_fragments([first, second], 1e-3)) == 2
