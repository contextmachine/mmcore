import numpy as np
import pytest

from mmcore.geom.nurbs import NURBSSurface


from mmcore.numeric.intersection.ssx.boundary_intersection import (
    extract_surface_boundaries,
    find_boundary_intersections,
    sort_boundary_intersections,
    IntersectionPoint

)
from test_nurbs_algo import degree


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
            is_from_first_surface=True,interval=((0.,1.),(0.,1.))
        ),
        IntersectionPoint(
            point=np.array([1.0, 1.0, 0.0]),
            curve_param=0.5,
            surface_params=(0.5, 1.0),
            boundary_index=1,
            is_from_first_surface=True,interval=((0.,1.),(0.,1.))
        ),
        IntersectionPoint(
            point=np.array([0.0, 0.5, 0.0]),
            curve_param=0.5,
            surface_params=(0.0, 0.5),
            boundary_index=2,
            is_from_first_surface=False,interval=((0.,1.),(0.,1.))
        )
    ]
    
    sequences = sort_boundary_intersections(points)
    
    # Should group points into sequences of 1 or 2 points
    assert all(len(seq) <= 2 for seq in sequences), "Sequences should have at most 2 points"
    
    # Total number of points should be preserved
    total_points = sum(len(seq) for seq in sequences)
    assert total_points == len(points), "All points should be included in sequences"

from mmcore.geom._nurbs_eval import nurbs_surface,_tuple_to_nurbs
def test_boundary_intersections_special_case_1():
    """Test sorting boundary intersections into sequences"""
    # Create some test intersection points
    cpts1=[[[18.372395833333336, -5.833333333333334, -2.4127604166666665], [18.372395833333336, -4.791666666666668, -2.2587890625], [18.372395833333336, -3.8020833333333344, -2.143310546875], [18.372395833333336, -2.838541666666668, -2.0663248697916665]], [[20.3125, -5.833333333333334, -1.6875], [20.3125, -4.791666666666668, -1.453125], [20.3125, -3.8020833333333344, -1.27734375], [20.3125, -2.838541666666668, -1.16015625]], [[22.5, -5.833333333333334, -0.8125], [22.5, -4.791666666666668, -0.484375], [22.5, -3.8020833333333344, -0.23828125], [22.5, -2.838541666666668, -0.07421875]], [[25.0, -5.833333333333334, 0.25], [25.0, -4.791666666666668, 0.6875], [25.0, -3.8020833333333344, 1.015625], [25.0, -2.838541666666668, 1.234375]]]
    cpts2=[[[25.0, 2.313235147924826, 7.311849985118766], [25.0, 1.0938732067072034, 7.891620574641527], [25.0, -0.5978849606469093, 8.037015245332984], [25.0, -2.605881119743166, 7.928986294763618]], [[22.5, 0.27659302761028615, 3.2091334573340564], [22.5, -0.8983450949695972, 3.1855745902516], [22.5, -2.440601220485182, 2.7240266504301345], [22.5, -4.249332714807375, 2.185381019731033]], [[20.30626505634478, -1.2508885626256185, 0.5726903154336518], [20.30782379225859, -2.392508821227198, 0.153932707050348], [20.30899284419394, -3.822638415363887, -0.6914979478008024], [20.309869633145453, -5.481921411105531, -1.4959030988756958]], [[18.35654868487632, -2.396499755302547, -1.0683324994919836], [18.360510471990573, -3.5131316159203982, -1.7315955912099832], [18.363481812326263, -4.859166311522914, -2.7985751602257505], [18.36571031757803, -6.406362933329148, -3.7473043235642725]]]

    surf1=nurbs_surface(cpts1, [2.75, 2.75, 2.75, 2.75, 3.0, 3.0, 3.0, 3.0],[1.0, 1.0, 1.0, 1.0, 1.25, 1.25, 1.25, 1.25], degree=(3,3))
    surf2 = nurbs_surface(cpts2, [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25],
                          [2.0, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25, 2.25], degree=(3, 3))
    pts=sorted([[18.372396, -4.084958, -2.185538],[19.845663, -5.666885, -1.817405]],key=lambda x: x[0])
    res=find_boundary_intersections(_tuple_to_nurbs(surf1),_tuple_to_nurbs(surf2),tol=1e-5)


    # Should group points into sequences of 1 or 2 points

    assert len(res) == len(pts), f"Unexpected intersection count {len(res)}. Excpected {len(pts)}."
    sorted_res=sorted((p.point for p in res), key=lambda x:x[0])
    print(np.array(sorted_res).tolist(),np.array(pts).tolist())
    assert np.allclose(sorted_res,pts), f"The coordinates do not match the expected value\n\tresult: {np.array(sorted_res).tolist()}\n\texpected:{np.array(pts).tolist()}"
