import math
from typing import NamedTuple

from mmcore.geom._nurbs_eval import (
    NURBSSurfaceTuple,
    NURBSCurveTuple,
    evaluate_nurbs_surface,
    evaluate_nurbs_curve,
    _surface_interval,
    _nurbs_to_tuple,
    _curve_interval,
to_homogeneous_1d
)
import numpy as np
from numpy.typing import NDArray
from mmcore.geom._nurbs_knots import decompose_surface, decompose_curve, split_curve, make_curves_compatible
from mmcore.numeric import compute_parametric_tolerance_curve, evaluate_curvature,compare_curvature
from mmcore.numeric.aabb import aabb,point_in_aabb
from mmcore.numeric.closest_point import nurbs_curve_closest_point

class CCXInt(NamedTuple):
    s:float
    t:float
    c1_eval:dict
    c2_eval:dict
    ds:float
    dt:float

class Overlap(NamedTuple):
    start:CCXInt
    end:CCXInt


def _reverse_param(t:float, interval:tuple[float,float]):
    """
    
    :param t:
    :param interval:
    :return:
    """
    return (interval[0] + interval[1]   )-t
_EPS=np.finfo(np.float64).eps
def _aabb(pts):
    return np.array([np.min(pts, axis=0), np.max(pts, axis=0)])

_min_aeps=math.sin(math.radians(1))

def _bez_find_overlap(c1:NURBSCurveTuple,c2:NURBSCurveTuple,  spt:float=1e-3,angle_tol:float=0.052,**kwargs):
    """Finds the overlap between two Bezier curves.
     """

    s0,s1=_curve_interval(c1)
    t0,t1=_curve_interval(c2)
    start1 = evaluate_nurbs_curve(c1,s0,d_order=2)
    start1['T'],  start1['K'],_=     evaluate_curvature(start1['C1'],start1['C2'])
    end1 =evaluate_nurbs_curve(c1,s1,d_order=2)
    end1['T'], end1['K'],_= evaluate_curvature(end1['C1'],end1['C2'])
    start2 = evaluate_nurbs_curve(c2,t0,d_order=2)
    start2['T'],start2['K'],_=evaluate_curvature(start2['C1'],start2['C2'])
    end2 = evaluate_nurbs_curve(c2,t1,d_order=2)
    end2['T'],end2['K'],_=evaluate_curvature(end2['C1'],end2['C2'])
    bb1,bb2=_aabb(to_homogeneous_1d(c1.control_points,c1.weights)), _aabb(to_homogeneous_1d(c2.control_points,c2.weights))

    c1_ends,c2_ends=[(start1,s0),(end1,s1)],  [(start2,t0),(end2,t1)]

    c1_ends=list(filter(lambda x:point_in_aabb(bb2,x[0]['C']),c1_ends))
    c2_ends=list(filter(lambda x: point_in_aabb(bb1, x[0]['C']), c2_ends))
    if (len(c1_ends)+len(c2_ends))<2:

        return False,
    ints=[]
    for pt,prm in c1_ends:
        curve1_eval =pt
        ds=compute_parametric_tolerance_curve(pt['C1'],pt["C2"],spt=spt,angle_tol=angle_tol)
        t,  (fx, curve2_eval, dt)=nurbs_curve_closest_point(c2, pt['C'], spt=spt, angle_tol=angle_tol)
        curve2_eval["T"], curve2_eval["K"], _ = evaluate_curvature(curve2_eval["C1"], curve2_eval["C2"])

        if fx<spt:
            if( 1-np.abs(np.dot(curve2_eval['T'],pt["T"])))<angle_tol:
                #print("K",curve2_eval["K"],pt["K"])
                if compare_curvature(curve1_eval["C1"], curve1_eval["C2"], curve1_eval["K"], curve2_eval["C1"], curve2_eval["C2"], curve2_eval["K"]):
                   
                  
                        ints.append(((prm,t), (ds,dt),(pt,curve2_eval)))

    for pt, prm in c2_ends:
        dt = compute_parametric_tolerance_curve(pt['C1'],pt["C2"] , spt=spt, angle_tol=angle_tol)
        curve2_eval=pt
        s,(fx, curve1_eval, ds) = nurbs_curve_closest_point(c1, pt["C"], spt=spt, angle_tol=angle_tol)
        curve1_eval["T"], curve1_eval["K"], _ = evaluate_curvature(curve1_eval["C1"], curve1_eval["C2"])
     
        if fx < spt:
            if (1 - np.abs(np.dot(curve1_eval["T"], pt["T"]))) < angle_tol:
                if compare_curvature(curve1_eval["C1"], curve1_eval["C2"], curve1_eval["K"], curve2_eval["C1"], curve2_eval["C2"], curve2_eval["K"]):
                        
                        
                        ints.append(((s, prm), (ds, dt), (curve1_eval, pt)))

    if len(ints)<2:
        return False, ints
    min_s, max_s = min(ints, key=lambda x: x[0][0]), max(
        ints,
        key=lambda x: x[0][0],
    )

    ss,se=min_s[0][0],max_s[0][0]
    ts,te=min_s[0][1],max_s[0][1]

    s_mid = ss + (se - ss) / 2
    curve1_eval=evaluate_nurbs_curve(c1,s_mid,d_order=2)
    curve1_eval["T"], curve1_eval["K"], _ = evaluate_curvature(curve1_eval["C1"], curve1_eval["C2"])

    t_mid, (fx, curve2_eval, dt) = nurbs_curve_closest_point(c2, curve1_eval["C"], spt=spt, angle_tol=angle_tol)
    curve2_eval["T"], curve2_eval["K"], _ = evaluate_curvature(curve2_eval["C1"], curve2_eval["C2"])

    if fx < spt:
        if (1 - np.abs(np.dot(curve2_eval["T"], curve1_eval["T"]))) < angle_tol:
            if compare_curvature(curve1_eval["C1"], curve1_eval["C2"], curve1_eval["K"], curve2_eval["C1"], curve2_eval["C2"], curve2_eval["K"]):
                
                    pass
            else:
                    return False, ints
        else:
            return False, ints

    # if ts>te:
    #    c2 = c2._replace(control_points=np.flip(c2.control_points, axis=0), weights=np.flip(c2.weights, axis=0))
    #    ts=_reverse_param(ts,(t0,t1))
    #    te = _reverse_param(te, (t0, t1))

    # print((ss,se),(ts, te))
    # if (abs(ss-s0)<1e-12) and (abs(s1-se)<1e-12):
    #    c11=c1
    # elif abs(ss-s0)<1e-12:
    #    c11=split_curve(c1,se)[0]
    # elif abs(s1-se)<1e-12:
    #    c11=split_curve(c1,ss)[1]
    # else:
    #    c11=split_curve(split_curve(c1,ss)[1],se)[0]
    #
    # if (abs(ts - t0) < 1e-12) and (abs(t1 - te) < 1e-12):
    #    c21=c2
    # elif abs(ts - t0) < 1e-12:
    #
    #    c21 = split_curve(c2, te)[0]
    # elif abs(t1 - te) < 1e-12:
    #    c21 = split_curve(c2, ts)[1]
    # else:
    #    c21 = split_curve(split_curve(c2, ts)[1], te)[0]
    #
    # cc1,cc2=make_curves_compatible(c11,c21)
    #
    # ll=np.linalg.norm(cc1.control_points-cc2.control_points,axis=1)
    #
    # if np.all(ll<spt):
    #
    #    return True,Overlap(    CCXInt(*min_s[0], *min_s[2],*min_s[1]),    CCXInt(*max_s[0], *max_s[2],*max_s[1]))
    # return False,ints
    return True, Overlap(    CCXInt(*min_s[0], *min_s[2],*min_s[1]),    CCXInt(*max_s[0], *max_s[2],*max_s[1]))


if __name__=="__main__":
    import numpy as np
    from mmcore.geom._nurbs_eval import NURBSCurveTuple

    import numpy as np
    from mmcore.geom._nurbs_eval import NURBSCurveTuple

    crv1 = NURBSCurveTuple(
        order=4,
        knot=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        control_points=np.array(
            [
                [-19.99999996, -15.99999996, 0.0],
                [-14.11168586, -10.11168586, 0.0],
                [-8.22337172, -9.17654935, 0.0],
                [-5.31116635, -12.59936871, 0.0],
            ]
        ),
        weights=np.array([1.0, 1.0, 1.0, 1.0]),
    )

    from mmcore.geom._nurbs_eval import NURBSCurveTuple

    crv2 = NURBSCurveTuple(
    order=4,
    knot=np.array([0., 0., 0., 0., 1., 1., 1., 1.]),
    control_points=np.array([ [ -4.        , -15.        ,   0.        ],
          
           [ -5.56948184, -10.29155448,   0.        ], [-10.21805524,  -9.27801883,   0.        ],[-15.52943104, -12.44265086,   0.        ]]
          ),
    weights=np.array([1., 1., 1., 1.])
    )
    res=_bez_find_overlap(crv1,crv2) # True
    print(res)
    assert res[0] == True
    crv3 = NURBSCurveTuple(
        order=4,
        knot=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        control_points=np.array([[-15.31739026, -12.31832453, 0.0], [-8.0, -8.0, 0.0], [-3.0, -13.0, 0.0], [-4.0, -17.0, 0.0]]),
        weights=np.array([1.0, 1.0, 1.0, 1.0]),
    )

    res2 = _bez_find_overlap(crv2, crv3) # False
    print(res2)
    assert res2[0] == False
    import numpy as np
    from mmcore.geom._nurbs_eval import NURBSCurveTuple

    crv4 =  NURBSCurveTuple(
    order=4,
    knot=np.array([0., 0., 0., 0., 1., 1., 1., 1.]),
    control_points=np.array([[-15.31739026, -12.31832453,   0.        ],
           [-10.22795811,  -9.31481933,   0.        ],
           [ -8.16419053, -10.24855722,   0.        ],
           [ -6.17138081, -11.77864314,   0.        ]]),
    weights=np.array([1., 1., 1., 1.])
    )

    res3 = _bez_find_overlap(crv2, crv4)  # False
    print(res3)
    assert res3[0] == False
    line1 = NURBSCurveTuple(
        order=2,
        knot=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([[-26.0, -24.0, 0.0], [-5.0, -23.0, 0.0]]),
        weights=np.array([1.0, 1.0]),
    )
    line2 = NURBSCurveTuple(
        order=2,
        knot=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([[-26.0, -24.0, 0.0], [-5.0, -23.0, 0.0]]),
        weights=np.array([1.0, 1.0]),
    )

    res4=_bez_find_overlap(line1,line2) # True
    print(res4)
    assert res4[0] == True
    crv5 = NURBSCurveTuple(
        order=9,
        knot=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        control_points=np.array(
            [
                [-14.0, -14.0, 0.0],
                [-10.02802769, 2.68228368, 0.0],
                [-7.61515077, 5.76636677, 0.0],
                [-5.7068885, 7.875, 0.0],
                [-1.98515124, 9.00431836, 0.0],
                [1.52232143, 7.5, 0.0],
                [5.44520052, 4.79218832, 0.0],
                [7.80820182, 1.89765913, 0.0],
                [15.0, -14.0, 0.0],
            ]
        ),
        weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    import numpy as np
    from mmcore.geom._nurbs_eval import NURBSCurveTuple

    import numpy as np
    from mmcore.geom._nurbs_eval import NURBSCurveTuple

    crv6 = NURBSCurveTuple(
        order=3,
        knot=np.array([0.0, 0.0, 0.0, 89.27178858, 89.27178858, 89.27178858]),
        control_points=np.array([[-14.0, -14.0, 0.0], [-4.0, 28.0, 0.0], [15.0, -14.0, 0.0]]),
        weights=np.array([1.0, 1.0, 1.0]),
    )
    res5=_bez_find_overlap(crv5,crv6) # False
    print(res5)
    assert res5[0] == False
    crv7= NURBSCurveTuple(
    order=8,
    knot=np.array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        , 89.27178858, 89.27178858,
           89.27178858, 89.27178858, 89.27178858, 89.27178858, 89.27178858,
           89.27178858]),
    control_points=np.array([[-14.        , -14.        ,   0.        ],
           [-11.14285714,  -2.        ,   0.        ],
           [ -7.85714286,   6.        ,   0.        ],
           [ -4.14285714,  10.        ,   0.        ],
           [  0.        ,  10.        ,   0.        ],
           [  4.57142857,   6.        ,   0.        ],
           [  9.57142857,  -2.        ,   0.        ],
           [ 15.        , -14.        ,   0.        ]]),
    weights=np.array([1., 1., 1., 1., 1., 1., 1., 1.])
    )
    res6 = _bez_find_overlap(crv6, crv7)  # True
    print(res6)
    assert res6[0] == True
    crv8 = NURBSCurveTuple(
        order=8,
        knot=np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
            ]
        ),
        control_points=np.array(
            [
                [-14.0, -14.0, 0.0],
                [-11.14285714, -2.0, 0.0],
                [-8.85714286, 6.0, 0.0],
                [-4.14285714, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [4.67142857, 6.0, 0.0],
                [9.57142857, -2.0, 0.0],
                [15.0, -14.0, 0.0],
            ]
        ),
        weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    res7 = _bez_find_overlap(crv6, crv8)  # False
    print(res7)
    assert res7[0]==False

    crv9 = NURBSCurveTuple(
        order=8,
        knot=np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
            ]
        ),
        control_points=np.array(
            [
                [-14.0, -14.0, 0.0],
                [-11.14285714, -2.0, 0.0],
                [-7.95714286, 6.0, 0.0],
                [-4.14285714, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [4.67142857, 6.0, 0.0],
                [9.57142857, -2.0, 0.0],
                [15.0, -14.0, 0.0],
            ]
        ),
        weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    res8 = _bez_find_overlap(crv6, crv9)  # False
    print(res8)
    assert res8[0]==False