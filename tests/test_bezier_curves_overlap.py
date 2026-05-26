from mmcore.numeric.intersection.ccx._bez_overlap import _bez_curve_overlap

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
    knot=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    control_points=np.array(
        [[-4.0, -15.0, 0.0], [-5.56948184, -10.29155448, 0.0], [-10.21805524, -9.27801883, 0.0], [-15.52943104, -12.44265086, 0.0]]
    ),
    weights=np.array([1.0, 1.0, 1.0, 1.0]),
)
crv3 = NURBSCurveTuple(
    order=4,
    knot=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    control_points=np.array([[-15.31739026, -12.31832453, 0.0], [-8.0, -8.0, 0.0], [-3.0, -13.0, 0.0], [-4.0, -17.0, 0.0]]),
    weights=np.array([1.0, 1.0, 1.0, 1.0]),
)

crv4 = NURBSCurveTuple(
    order=4,
    knot=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    control_points=np.array(
        [
            [-15.31739026, -12.31832453, 0.0],
            [-10.22795811, -9.31481933, 0.0],
            [-8.16419053, -10.24855722, 0.0],
            [-6.17138081, -11.77864314, 0.0],
        ]
    ),
    weights=np.array([1.0, 1.0, 1.0, 1.0]),
)


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

crv6 = NURBSCurveTuple(
    order=3,
    knot=np.array([0.0, 0.0, 0.0, 89.27178858, 89.27178858, 89.27178858]),
    control_points=np.array([[-14.0, -14.0, 0.0], [-4.0, 28.0, 0.0], [15.0, -14.0, 0.0]]),
    weights=np.array([1.0, 1.0, 1.0]),
)
crv7 = NURBSCurveTuple(
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
            [-7.85714286, 6.0, 0.0],
            [-4.14285714, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [4.57142857, 6.0, 0.0],
            [9.57142857, -2.0, 0.0],
            [15.0, -14.0, 0.0],
        ]
    ),
    weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
)

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


def test_case_1():
    res = _bez_curve_overlap(crv1, crv2)  # True
    print(res)
    assert res[0] == True

def test_case_2():
    res2 = _bez_curve_overlap(crv2, crv3)  # False
    print(res2)
    assert res2[0] == False
def test_case_3():
    res3 = _bez_curve_overlap(crv2, crv4)  # False
    print(res3)
    assert res3[0] == False
def test_case_4():
    res4 = _bez_curve_overlap(line1, line2)  # True
    print(res4)
    assert res4[0] == True

def test_case_5():
    res5 = _bez_curve_overlap(crv5, crv6)  # False
    print(res5)
    assert res5[0] == False
def test_case_6():
    res6 = _bez_curve_overlap(crv6, crv7)  # True
    print(res6)
    assert res6[0] == True
def test_case_7():
    res7 = _bez_curve_overlap(crv6, crv8)  # False
    print(res7)
    assert res7[0] == False

def test_case_8():
    res8 = _bez_curve_overlap(crv6, crv9)  # False
    print(res8)
    assert res8[0] == False
