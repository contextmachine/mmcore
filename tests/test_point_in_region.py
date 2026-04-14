"""Tests for point_in_region (2D point-in-region via nurbs_ccx segment cast).

Uses 3D curves with z=0 for the "2D plane" convention.
"""
from __future__ import annotations

import numpy as np
import pytest

from mmcore.construction import circle
from mmcore.geom._nurbs_eval import NURBSCurveTuple
from mmcore.topo.brep.boolean2d import point_in_region


def line(p0, p1):
    """Build a degree-1 NURBS curve between two 3D points."""
    return NURBSCurveTuple(
        order=2,
        knot=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([p0, p1], dtype=float),
        weights=np.array([1.0, 1.0], dtype=float),
    )


def unit_square_ccw():
    """Unit square boundary (CCW in xy plane) as 4 line segments."""
    return [
        line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        line([1.0, 0.0, 0.0], [1.0, 1.0, 0.0]),
        line([1.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        line([0.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
    ]


def test_point_in_center_of_unit_square_is_inside():
    region = unit_square_ccw()
    assert point_in_region(np.array([0.5, 0.5, 0.0]), region, tol=1e-6) is True


def test_point_far_from_unit_square_is_outside():
    region = unit_square_ccw()
    assert point_in_region(np.array([2.0, 0.5, 0.0]), region, tol=1e-6) is False


def test_point_below_unit_square_is_outside():
    region = unit_square_ccw()
    assert point_in_region(np.array([0.5, -1.0, 0.0]), region, tol=1e-6) is False


def test_point_at_circle_center_is_inside():
    region = [circle(center=(0.0, 0.0, 0.0), radius=1.0)]
    assert point_in_region(np.array([0.0, 0.0, 0.0]), region, tol=1e-6) is True


def test_point_just_inside_circle_is_inside():
    region = [circle(center=(0.0, 0.0, 0.0), radius=1.0)]
    assert point_in_region(np.array([0.8, 0.0, 0.0]), region, tol=1e-6) is True


def test_point_just_outside_circle_is_outside():
    region = [circle(center=(0.0, 0.0, 0.0), radius=1.0)]
    assert point_in_region(np.array([1.2, 0.0, 0.0]), region, tol=1e-6) is False


def test_point_far_from_circle_is_outside():
    region = [circle(center=(0.0, 0.0, 0.0), radius=1.0)]
    assert point_in_region(np.array([5.0, 5.0, 0.0]), region, tol=1e-6) is False


def test_point_in_annulus_ring_is_inside():
    outer = circle(center=(0.0, 0.0, 0.0), radius=1.0)
    inner = circle(center=(0.0, 0.0, 0.0), radius=0.3)
    region = [outer, inner]
    # (0.6, 0) is between inner radius 0.3 and outer radius 1.0
    assert point_in_region(np.array([0.6, 0.0, 0.0]), region, tol=1e-6) is True


def test_point_at_annulus_center_is_outside():
    outer = circle(center=(0.0, 0.0, 0.0), radius=1.0)
    inner = circle(center=(0.0, 0.0, 0.0), radius=0.3)
    region = [outer, inner]
    # (0, 0) is inside the hole — outside the material
    assert point_in_region(np.array([0.0, 0.0, 0.0]), region, tol=1e-6) is False


def test_point_outside_annulus_is_outside():
    outer = circle(center=(0.0, 0.0, 0.0), radius=1.0)
    inner = circle(center=(0.0, 0.0, 0.0), radius=0.3)
    region = [outer, inner]
    assert point_in_region(np.array([2.0, 0.0, 0.0]), region, tol=1e-6) is False


def test_two_disjoint_circles_point_in_only_one_is_inside():
    c1 = circle(center=(-5.0, 0.0, 0.0), radius=1.0)
    c2 = circle(center=(+5.0, 0.0, 0.0), radius=1.0)
    region = [c1, c2]
    assert point_in_region(np.array([-5.0, 0.0, 0.0]), region, tol=1e-6) is True
    assert point_in_region(np.array([+5.0, 0.0, 0.0]), region, tol=1e-6) is True
    assert point_in_region(np.array([0.0, 0.0, 0.0]), region, tol=1e-6) is False


def test_point_on_square_edge_raises():
    """A point lying exactly on a region boundary has an undefined PIP result.

    The helper must raise RuntimeError rather than silently return True or False.
    """
    region = unit_square_ccw()
    # (0, 0.5) is the midpoint of the left edge — exactly on the boundary
    with pytest.raises(RuntimeError, match="lies on a region boundary"):
        point_in_region(np.array([0.0, 0.5, 0.0]), region, tol=1e-6)


def test_point_whose_segment_is_tangent_to_circle_is_outside():
    """Construct P so the default segment direction is exactly tangent to the unit circle.

    Default direction d = (cos θ, sin θ), θ = 0.31415. The tangent point on the
    unit circle where the tangent line is parallel to d is
    Q = (sin θ, -cos θ). Pick P at Q - d (one unit behind Q along d) so the
    segment from P in direction d grazes the circle exactly at Q.
    """
    theta = 0.31415
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    # tangent point on unit circle
    qx, qy = sin_t, -cos_t
    # place P one unit behind Q along d
    px, py = qx - cos_t, qy - sin_t
    # P is outside the unit circle (|P| ≈ sqrt(2))
    assert np.hypot(px, py) > 1.0

    region = [circle(center=(0.0, 0.0, 0.0), radius=1.0)]
    result = point_in_region(np.array([px, py, 0.0]), region, tol=1e-6)
    assert result is False
