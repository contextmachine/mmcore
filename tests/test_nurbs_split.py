from mmcore.geom._nurbs_knots import split_curve
from mmcore.geom._nurbs_eval import nurbs_curve, _nurbs_to_tuple, nurbs_interval, nurbs_curve, _nurbs_to_tuple
import numpy as np
def prepare_curve_case1(degree:int):
    cpts = np.array(
        [
            [0.56443362168932998, 0.025341918037446676, 0.0],
            [0.4293095543614065, 0.43702220452673946, 0.0],
            [0.83025827194944257, 0.56361988886490066, 0.0],
            [0.89152549258149683, -0.12608982322196316, 0.0],
            [1.2348245134075415, 0.2786410925553513, 0.0],
            [0.75986018478199646, 0.9797269029477047, 0.0],
            [0.44260844486480133, 0.8498002979769389, 0.0],
            [0.62481057823860575, 0.65269428425014153, 0.0],
            [0.14594645765488079, 0.6925639607223979, 0.0],
            [0.059531212949752388, 0.35905121725291644, 0.0],
        ]
    )

    from mmcore.geom.nurbs import NURBSCurve

    crv = NURBSCurve(cpts, 2)



    return _nurbs_to_tuple(crv)

def test_case1():
    crvt=prepare_curve_case1(degree=2)

    a, b = split_curve(crvt, 3.5)
    first=[[0.56443362168933, 0.025341918037446676, 0.0], [0.4293095543614065, 0.43702220452673946, 0.0], [0.8302582719494426, 0.5636198888649007, 0.0], [0.8915254925814968, -0.12608982322196316, 0.0], [1.1489997582010303, 0.17745836361102268, 0.0], [1.1325415947260926, 0.31568545438223117, 0.0]],[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.5, 3.5, 3.5]
    second=[[1.1325415947260926, 0.31568545438223117, 0.0], [1.1160834312511552, 0.45391254515343965, 0.0], [0.7598601847819965, 0.9797269029477047, 0.0], [0.44260844486480133, 0.8498002979769389, 0.0], [0.6248105782386058, 0.6526942842501415, 0.0], [0.1459464576548808, 0.6925639607223979, 0.0], [0.05953121294975239, 0.35905121725291644, 0.0]],[3.5, 3.5, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 8.0]
    assert np.allclose(a.control_points,first[0]), f"Unexpected result: {a.control_points.tolist()}\n\t expected: {first[0]}"
    assert np.allclose(a.knots,
                       first[1]), f"Unexpected result: {np.array(a.knots).tolist()}\n\t expected: {first[1]}"
    assert np.allclose(b.control_points,
                       second[0]), f"Unexpected result: {a.control_points.tolist()}\n\t expected: {second[0]}"
    assert np.allclose(b.knots,
                       second[1]), f"Unexpected result: {np.array(a.knots).tolist()}\n\t expected: {second[1]}"
def test_case2():
    crvt=prepare_curve_case1(degree=3)

    a, b = split_curve(crvt, 3.5)
    first=[[0.56443362168933, 0.025341918037446676, 0.0], [0.4293095543614065, 0.43702220452673946, 0.0], [0.8302582719494426, 0.5636198888649007, 0.0], [0.8915254925814968, -0.12608982322196316, 0.0], [1.177608009936534, 0.2111859399257989, 0.0], [1.0424087643052102, 0.5246844832950958, 0.0], [0.9835808749126183, 0.6180452994026098, 0.0]],[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.5, 3.5, 3.5, 3.5]
    second=[[0.9835808749126183, 0.6180452994026098, 0.0], [0.9247529855200262, 0.7114061155101237, 0.0], [0.7069848947957973, 0.9580724687859105, 0.0], [0.44260844486480133, 0.8498002979769389, 0.0], [0.6248105782386058, 0.6526942842501415, 0.0], [0.1459464576548808, 0.6925639607223979, 0.0], [0.05953121294975239, 0.35905121725291644, 0.0]],[3.5, 3.5, 3.5, 3.5, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0]

    assert np.allclose(a.control_points,first[0]), f"Unexpected result: {a.control_points.tolist()}\n\t expected: {first[0]}"
    assert np.allclose(a.knots,
                       first[1]), f"Unexpected result: {np.array(a.knots).tolist()}\n\t expected: {first[1]}"
    assert np.allclose(b.control_points,
                       second[0]), f"Unexpected result: {a.control_points.tolist()}\n\t expected: {second[0]}"
    assert np.allclose(b.knots,
                       second[1]), f"Unexpected result: {np.array(a.knots).tolist()}\n\t expected: {second[1]}"
