import numpy as np

from mmcore.geom.curves.knot import degree_elevate_curve, refine_curve
from mmcore.geom.nurbs import NURBSSurface,NURBSCurve
__all__=['ruled','Ruled']

def merge_knot_vectors(knots1, knots2):
    """
    Merge two knot vectors, maintaining multiplicities
    """
    # Combine unique knots from both vectors
    all_knots = np.unique(np.concatenate([knots1, knots2]))

    # For each knot, take maximum multiplicity from either vector
    result = []
    for knot in all_knots:
        mult1 = np.sum(np.isclose(knots1, knot))
        mult2 = np.sum(np.isclose(knots2, knot))
        result.extend([knot] * max(mult1, mult2))

    return np.array(result)

def make_curves_compatible(curve1, curve2):
    """
    Make two NURBS curves compatible for ruled surface construction

    Parameters:
    curve1, curve2: dict with keys:
        - control_points: nx4 array (x,y,z,w)
        - degree: int
        - knots: array of knot values

    Returns:
    tuple of two modified curves with same degree, knots and number of control points
    """
    # 1. Degree elevation to match highest degree

    p1, p2 = curve1.degree, curve2.degree
    if p1 < p2:
        curve1 = degree_elevate_curve(curve1, p2 - p1)



    elif p2 < p1:
        curve2 = degree_elevate_curve(curve2, p1 - p2)

    # 2. Merge knot vectors


    # 3. Knot refinement to match knot vectors
    unified_knots = np.array(merge_knot_vectors(curve1.knots, curve2.knots))

    curve1 = refine_curve(curve1, unified_knots)
    curve2 = refine_curve(curve2, unified_knots)

    return curve1, curve2


def ruled(curve1:NURBSCurve, curve2:NURBSCurve):
    """
    Generates a ruled surface between two given NURBS curves. A ruled surface is a
    surface created by linear interpolation between corresponding points on two
    curves. This function assumes that the input curves are NURBS curves and processes
    them to make them compatible before producing the NURBS surface. If the input
    curves have different knot vectors or control points, they will be modified to
    produce a valid ruled surface.

    :param curve1: The first input NURBS curve.
    :type curve1: NURBSCurve
    :param curve2: The second input NURBS curve.
    :type curve2: NURBSCurve

    :return: A NURBS ruled surface created between the two input curves.
    :rtype: NURBSSurface
    """
    # Make curves compatible
    curve1=curve1.copy()
    curve2=curve2.copy()
    curve1.normalize_knots()
    curve2.normalize_knots()
    c1, c2 = make_curves_compatible(curve1, curve2)

    # Create surface control points
    n = len(c1.control_points)
    control_points = np.zeros((n, 2, 4))  # nx2x4 array


    # Fill control points

    for i in range(n):
        control_points[i, 0] = np.array(c1._control_points)[i]
        control_points[i, 1] = np.array(c2._control_points)[i]

    # Create surface knot vectors
    u_knots = c1.knots  # Same for both curves now
    v_knots = np.array([0., 0., 1., 1.])  # Linear interpolation in v direction

    return NURBSSurface( control_points,(c1.degree, 1),u_knots,v_knots)


Ruled=ruled

if __name__ =="__main__":
    s, b = [[[0.41003319883988076, -5.9558709997242776, -0.45524326627631317],
             [0.41003319883988076, -5.5445084274881866, 0.31289808372671224],
             [0.41003319883988076, -4.2689095570901747, 0.27335792945560905],
             [0.41003319883988054, -2.8275390970241014, 0.38227620969285792],
             [0.41003319883988071, -1.5384497736905611, -0.55192398184841063],
             [0.4100331988398806, -1.0649511609106423, 0.18024597033519471],
             [0.41003319883988076, -0.45939612773632504, 0.16055590176917547],
             [0.41003319883988065, -0.039150933363156781, 0.19313507758724083]],
            [[-1.2431856590487269, -5.5356810246947985, 0.0],
             [-1.1678165416732957, -3.8992861465815305, 1.3621802986275990],
             [-1.9780801959755556, -3.9866785366038910, 0.0], [-1.0059685065682198, -0.38528326448094535, 0.0]]]

    c1 = NURBSCurve(np.array(s), 2
                    )
    c3 = NURBSCurve(np.array(b), 3)
    s = ruled(c1, c3)
    from mmcore.compat.step.step_writer import StepWriter
    writer=StepWriter()
    writer.add_nurbs_surface(s)
    with open('my-ruled.stp','w') as f:
        writer.write(f)


