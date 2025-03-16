import numpy as np



from mmcore.numeric.intersection.ssx.boundary_intersection import (

    find_boundary_intersections, IntersectionPoint

)

from mmcore.geom._nurbs_eval import nurbs_surface,_tuple_to_nurbs,_surface_interval

def test_boundary_intersections_special_case_1():
    """Test sorting boundary intersections into sequences"""
    # Create some test intersection points
    cpts1=[[[18.372395833333336, -5.833333333333334, -2.4127604166666665], [18.372395833333336, -4.791666666666668, -2.2587890625], [18.372395833333336, -3.8020833333333344, -2.143310546875], [18.372395833333336, -2.838541666666668, -2.0663248697916665]], [[20.3125, -5.833333333333334, -1.6875], [20.3125, -4.791666666666668, -1.453125], [20.3125, -3.8020833333333344, -1.27734375], [20.3125, -2.838541666666668, -1.16015625]], [[22.5, -5.833333333333334, -0.8125], [22.5, -4.791666666666668, -0.484375], [22.5, -3.8020833333333344, -0.23828125], [22.5, -2.838541666666668, -0.07421875]], [[25.0, -5.833333333333334, 0.25], [25.0, -4.791666666666668, 0.6875], [25.0, -3.8020833333333344, 1.015625], [25.0, -2.838541666666668, 1.234375]]]
    cpts2=[[[25.0, 2.313235147924826, 7.311849985118766], [25.0, 1.0938732067072034, 7.891620574641527], [25.0, -0.5978849606469093, 8.037015245332984], [25.0, -2.605881119743166, 7.928986294763618]], [[22.5, 0.27659302761028615, 3.2091334573340564], [22.5, -0.8983450949695972, 3.1855745902516], [22.5, -2.440601220485182, 2.7240266504301345], [22.5, -4.249332714807375, 2.185381019731033]], [[20.30626505634478, -1.2508885626256185, 0.5726903154336518], [20.30782379225859, -2.392508821227198, 0.153932707050348], [20.30899284419394, -3.822638415363887, -0.6914979478008024], [20.309869633145453, -5.481921411105531, -1.4959030988756958]], [[18.35654868487632, -2.396499755302547, -1.0683324994919836], [18.360510471990573, -3.5131316159203982, -1.7315955912099832], [18.363481812326263, -4.859166311522914, -2.7985751602257505], [18.36571031757803, -6.406362933329148, -3.7473043235642725]]]

    surf1=nurbs_surface(cpts1,
                        [2.75, 2.75, 2.75, 2.75, 3.0, 3.0, 3.0, 3.0],
                        [1.0, 1.0, 1.0, 1.0, 1.25, 1.25, 1.25, 1.25],
                        degree=(3,3))
    surf2 = nurbs_surface(cpts2,
                          [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25],
                          [2.0, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25, 2.25],
                          degree=(3, 3))

    print("First Surface interval:", _surface_interval(surf1)) # Correct: ((2.75, 3.0), (1.0, 1.25))
    print("Second Surface interval:", _surface_interval(surf2)) # Correct: ((0.0, 0.25), (2.0, 2.25))

    pts=sorted([[18.372396, -4.084958, -2.185538],[19.845663, -5.666885, -1.817405]],key=lambda x: x[0])
    
    # Convert tuples to NURBS surfaces
    nurbs_surf1 = _tuple_to_nurbs(surf1)
    nurbs_surf2 = _tuple_to_nurbs(surf2)
    
    # Print debug information
    print("Surface 1 control points shape:", nurbs_surf1.control_points.shape)
    print("Surface 2 control points shape:", nurbs_surf2.control_points.shape)
    
    # Use higher tolerance for this specific hard case
    res=find_boundary_intersections(nurbs_surf1, nurbs_surf2, tol=1e-4, ptol=1e-6)

    sorted_res:list[IntersectionPoint] = sorted(res, key=lambda x: x.point[0])  # Now we do not test the order, so in order to avoid false negatives, we perform an obvious sorting, which guarantees that the order of points of the expected and actual results will coincide
    sorted_res_xyz=[]
    sorted_res_params=[]
    for pt in sorted_res:
        sorted_res_xyz.append(pt.point)
        sorted_res_params.append((pt.surface1_params,pt.surface2_params))
    sorted_res_params=np.array(sorted_res_params)
    sorted_res_xyz=np.array(sorted_res_xyz)
    print("Actual params:")
    print(sorted_res_params)
    print("Actual xyz:")
    print(sorted_res_xyz)
    # Should group points into sequences of 1 or 2 points
    print("\nExpected xyz:\n", np.array(pts))




    assert len(res) == len(pts), f"Unexpected intersection count {len(res)}. Excpected {len(pts)}."


    assert np.allclose(sorted_res_xyz, pts, atol=1e-5), f"The coordinates do not match the expected value\n\tresult: {sorted_res_xyz.tolist()}\n\texpected:{np.array(pts).tolist()}"
if __name__ == "__main__":
    import yappi
    yappi.set_clock_type('wall')
    yappi.start()
    test_boundary_intersections_special_case_1()
    yappi.stop()
    stats=yappi.get_func_stats()
    stats.print_all()
