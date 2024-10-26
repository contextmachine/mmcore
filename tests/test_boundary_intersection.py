import numpy as np
import pytest

from mmcore.geom.nurbs import NURBSSurface


from mmcore.numeric.intersection.ssx.boundary_intersection import (
    extract_surface_boundaries,
    find_boundary_intersections,
    sort_boundary_intersections,
    IntersectionPoint
)




def create_test_surface1() -> NURBSSurface:
    """Create a simple test NURBS surface"""
    control_points = np.array([
        [[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]],
        [[1., 0., 0.], [1., 1., 1.], [1., 2., 0.]],
        [[2., 0., 0.], [2., 1., 0.], [2., 2., 0.]]
    ], dtype=np.float64)
    return NURBSSurface(control_points, (2, 2))
    #return test_surfaces[2][0]
def create_test_surface2() -> NURBSSurface:
    """Create another test NURBS surface that intersects with surface1"""
    control_points = np.array([
        [[1., -1., -1.], [1., 0., 1.], [1., 1., -1.]],
        [[1., -1., 0.], [1., 0., 2.], [1., 1., 0.]],
        [[1., -1., 1.], [1., 0., 1.], [1., 1., 1.]]
    ], dtype=np.float64)
    return NURBSSurface(control_points, (2, 2))
    #return test_surfaces[2][1]
def test_extract_surface_boundaries():
    """Test boundary curve extraction"""
    surface = create_test_surface1()
    boundaries = extract_surface_boundaries(surface)
    
    assert len(boundaries) == 4, "Should extract exactly 4 boundary curves"
    
    # Test that boundaries form a closed loop
    tol = 1e-6
    for b in boundaries:
        print(b.control_points)
        print(np.array(b.start()),np.array(b.end()))

    # Check if the curves connect at corners
    assert np.allclose(boundaries[0].evaluate(0), boundaries[2].evaluate(0), atol=tol)
    assert np.allclose(boundaries[0].evaluate(1), boundaries[3].evaluate(0), atol=tol)
    assert np.allclose(boundaries[1].evaluate(0), boundaries[2].evaluate(1), atol=tol)
    assert np.allclose(boundaries[1].evaluate(1), boundaries[3].evaluate(1), atol=tol)

def test_find_boundary_intersections():
    """Test finding intersections between surface boundaries"""
    surf1 = create_test_surface1()
    surf2 = create_test_surface2()
    
    intersections = find_boundary_intersections(surf1, surf2)
    
    assert len(intersections) > 0, "Should find at least one intersection"
    
    # Check that all intersection points lie on both surfaces
    tol = 1e-6
    for intersection in intersections:
        # Check that all points lie on both surfaces using surface1_params and surface2_params
        pt1 = surf1.evaluate(np.array(intersection.surface1_params))
        pt2 = surf2.evaluate(np.array(intersection.surface2_params))
        
        print(f"Point: {intersection.point}")
        print(f"Surface1 evaluation at {intersection.surface1_params}: {pt1}")
        print(f"Surface2 evaluation at {intersection.surface2_params}: {pt2}")
        
        assert np.allclose(pt1, intersection.point, atol=tol), "Point should lie on first surface"
        assert np.allclose(pt2, intersection.point, atol=tol), "Point should lie on second surface"
        assert np.allclose(pt1, pt2, atol=tol), "Evaluations should match each other"
        
        # Verify that get_start_params still works correctly for backward compatibility
        start_params = intersection.get_start_params()
        assert np.allclose(surf1.evaluate(start_params[0]), intersection.point, atol=tol), "get_start_params should give correct parameters for first surface"
        assert np.allclose(surf2.evaluate(start_params[1]), intersection.point, atol=tol), "get_start_params should give correct parameters for second surface"

def test_sort_boundary_intersections():
    """Test sorting boundary intersections into sequences"""
    # Create some test intersection points
    points = [
        IntersectionPoint(
            point=np.array([1.0, 0.0, 0.0]),
            curve_param=0.5,
            surface_params=(0.5, 0.0),
            boundary_index=0,
            is_from_first_surface=True
        ),
        IntersectionPoint(
            point=np.array([1.0, 1.0, 0.0]),
            curve_param=0.5,
            surface_params=(0.5, 1.0),
            boundary_index=1,
            is_from_first_surface=True
        ),
        IntersectionPoint(
            point=np.array([0.0, 0.5, 0.0]),
            curve_param=0.5,
            surface_params=(0.0, 0.5),
            boundary_index=2,
            is_from_first_surface=False
        )
    ]
    
    sequences = sort_boundary_intersections(points)
    
    # Should group points into sequences of 1 or 2 points
    assert all(len(seq) <= 2 for seq in sequences), "Sequences should have at most 2 points"
    
    # Total number of points should be preserved
    total_points = sum(len(seq) for seq in sequences)
    assert total_points == len(points), "All points should be included in sequences"