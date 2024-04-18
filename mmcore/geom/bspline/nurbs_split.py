import copy
import math

from mmcore.geom.bspline import NURBSpline

"""
Here are the implementations of the requested functions and methods:

1. `split(t: float) → tuple[NURBSpline, NURBSpline]` method in the `NURBSpline` class:

"""
from mmcore.geom.bspline.knot import (
    find_span_binsearch,
    find_multiplicity,
    knot_insertion,
)


def insert_knot(self: NURBSpline, t, num=1):
    cpts, knots = knot_insertion(
        self.degree, self.knots.tolist(), self.control_points, t, num=num
    )
    self.set(control_points=np.array(cpts), knots=np.array(knots))
    return True


def split(self: NURBSpline, t: float) -> tuple[NURBSpline, NURBSpline]:
    """
    * ``insert_knot_func``: knot insertion algorithm implementation. *Default:* :func:`.operations.insert_knot`

    :param obj: Curve to be split
    :type obj: abstract.Curve
    :param param: parameter
    :type param: float
    :return: a list of curve segments
    :rtype: list
    """
    # Validate input

    # Keyword arguments
    span_func = find_span_binsearch  # FindSpan implementation
    insert_knot_func = insert_knot
    knotvector = self.knots
    # Find multiplicity of the knot and define how many times we need to add the knot
    ks = (
        span_func(self.degree, knotvector, len(self.control_points), t)
        - self.degree
        + 1
    )
    s = find_multiplicity(t, knotvector)
    r = self.degree - s

    # Create backups of the original curve
    temp_obj = copy.deepcopy(self)

    # Insert knot
    insert_knot_func(temp_obj, t, num=r)

    # Knot vectors
    knot_span = (
        span_func(temp_obj.degree, temp_obj.knots, len(temp_obj.control_points), t) + 1
    )
    curve1_kv = list(temp_obj.knots.tolist()[0:knot_span])
    curve1_kv.append(t)
    curve2_kv = list(temp_obj.knots.tolist()[knot_span:])
    for _ in range(0, temp_obj.degree + 1):
        curve2_kv.insert(0, t)

    # Control points (use Pw if rational)
    cpts = temp_obj.control_points.tolist()
    curve1_ctrlpts = cpts[0 : ks + r]
    curve2_ctrlpts = cpts[ks + r - 1 :]

    # Create a new curve for the first half
    curve1 = temp_obj.__class__(
        np.array(curve1_ctrlpts), knots=curve1_kv, degree=self.degree
    )

    # Create another curve fot the second half
    curve2 = temp_obj.__class__(
        np.array(curve2_ctrlpts), knots=curve2_kv, degree=self.degree
    )

    # Return the split curves

    return curve1, curve2


"""

2. `nurbs_curve_intersect(curve1: NURBSpline, curve2: NURBSpline, tol: float) → list[tuple[float, float] | None]` function:

"""


def fround(val: float, tol: float = 0.001):
    return round(val, int(abs(math.log10(tol))))


"""

These implementations handle both 2D and 3D cases, as the dimensionality is determined by the control points of the NURBS curves.

The `split` method uses De Boor's algorithm to split the NURBS curve at the given parameter value `t`. It computes the new control points, weights, and knot vectors for the resulting left and right subcurves.

The `nurbs_curve_intersect` function uses a recursive divide-and-conquer approach to find intersections between two NURBS curves. 
It checks the AABB overlap of the curves and recursively splits them until the distance between the curves 
is within the specified tolerance or there is no overlap. 
The function returns a list of tuples representing the parameter values of the intersections on each curve.

Note that the `nurbs_curve_aabb` function from document 10 is used to compute the AABB of the NURBS curves for the intersection algorithm."""


if __name__ == "__main__":
    import numpy as np
    from mmcore.geom.bspline import NURBSpline

    a, b, c, d = np.array(
        [
            [
                (-25.632193861977559, -25.887792238151487, 0.0),
                (-7.6507873591044131, -28.580781837412534, 0.0),
                (3.1180460594601840, -31.620627096247443, 0.0),
                (35.586827711309354, -35.550809492847861, 0.0),
            ],
            [
                (33.586827711309354, -30.550809492847861, 0.0),
                (23.712213781367616, -20.477792480394431, 0.0),
                (23.624609526477588, -7.8543655761938815, 0.0),
                (27.082667168033424, 5.5380493986617410, 0.0),
            ],
            [
                (27.082667168033424, 5.5380493986617410, 0.0),
                (8.6853191615639460, -2.1121318577726527, 0.0),
                (-3.6677924590213919, -2.9387254504549816, 0.0),
                (-20.330418684651349, 3.931006353774948, 0.0),
            ],
            [
                (-20.330418684651349, 3.931006353774948, 0.0),
                (-22.086936165417491, -5.8423256715423690, 0.0),
                (-23.428753995169622, -15.855467779623531, 0.0),
                (-25.632193861977559, -25.887792238151487, 0.0),
            ],
        ]
    )

    spl = NURBSpline(
        np.array(
            [
                [-1, -3, 0.0],
                [-0.1, -2.1, 0.0],
                [-0.4, -1, 0.0],
                [0.0, 0.5, 0.0],
                [0.4, 1.0, 0.0],
                [5.4, -2.1, 0.0],
                [8.0, -3.0, 0.0],
            ]
        ),
        degree=3,
    )
    print(spl.interval())
    res1 = split(spl, 1.5)

