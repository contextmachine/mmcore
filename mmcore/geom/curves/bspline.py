from __future__ import annotations
import copy
from functools import lru_cache

import numpy as np

from mmcore.func import vectorize
from mmcore.geom.curves.curve import Curve
from mmcore.geom.curves.deboor import deboor,evaluate_nurbs_multi,evaluate_nurbs

from mmcore.geom.curves.knot import find_span_binsearch, find_multiplicity
from mmcore.geom.curves.bspline_utils import calc_bspline_derivatives, insert_knot


def nurbs_split(self, t: float) -> tuple:
    """
    :param self:
    :param t:
    :return: a tuple of curve segments

    """


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
    curve1_ctrlpts = cpts[0: ks + r]
    curve2_ctrlpts = cpts[ks + r - 1:]

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


class BSpline(Curve):
    """ """

    _control_points = None
    _cached_basis_func: callable = None
    degree = 3
    knots = None

    def interval(self):
        return (float(min(self.knots)), float(max(self.knots)))

    def __init__(self, control_points, degree=3, knots=None):
        super().__init__()
        self._control_points_count = None
        self.set(control_points, degree=degree, knots=knots)
        self._wcontrol_points = np.ones((len(control_points), 4), dtype=float)
        self._wcontrol_points[:, :-1] = self.control_points


    def invalidate_cache(self):
        super().invalidate_cache()


        self._cached_basis_func = lru_cache(maxsize=None)(self.basis_function)


    @vectorize(excluded=[0], signature="()->(i)")
    def derivative(self, t):
        return calc_bspline_derivatives(
            self.degree, self.knots, self._wcontrol_points, t, 1
        )[1][:-1]

    @vectorize(excluded=[0], signature="()->(i)")
    def second_derivative(self, t):
        return calc_bspline_derivatives(
            self.degree, self.knots, self._wcontrol_points, t, 2
        )[2][:-1]

    @vectorize(excluded=[0, "n", "return_projective"], signature="()->(j,i)")
    def n_derivative(self, t, n=3, return_projective=False):
        """
        :param t: Parameter on the curve
        :param n: The number of derivatives that need to be evaluated.
        The zero derivative ( n=0 ) is a point on the curve, the last derivative is always==[0,0,0,0,1]

        :param return_projective: If True will return vectors as [x,y,z,w] instead of [x,y,z] default False
        :return:
        """
        res = np.array(
            calc_bspline_derivatives(
                self.degree, self.knots, self._wcontrol_points, t, n
            )
        )
        if return_projective:
            return res
        return res[..., :-1]

    def set(self, control_points=None, degree=None, knots=None):
        if control_points is not None:
            self._control_points = control_points if isinstance(control_points, np.ndarray) else np.array(
                control_points, dtype=float)

        if degree is not None:
            self.degree = degree
        self._control_points_count = len(self.control_points)
        self.knots = self.generate_knots() if knots is None else np.array(knots)
        self.invalidate_cache()

    def generate_knots(self):
        """
        In this code, the `generate_knots` method generates default knots based on the number of control points.
        The `__call__` method computes the parametric B-spline equation at the given parameter `t`.
        It first normalizes the parameter and finds the appropriate knot interval.
        Then, it computes the blending functions within that interval
        and uses them to compute the point on the B-spline curve using the control points.
        https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node4.html

        This function generates default knots based on the number of control points
        :return: A list of knots


        Difference with OpenNURBS.
        ------------------
        Why is there always one more node here than in OpenNURBS?
        In fact, it is OpenNURBS that deviates from the standard here.
        The original explanation can be found in the file `opennurbs/opennurbs_evaluate_nurbs.h`.
        But I will give a fragment:

            `Most literature, including DeBoor and The NURBS Book,
        duplicate the Opennurbs start and end knot values and have knot vectors
        of length d+n+1. The extra two knot values are completely superfluous
        when degree >= 1.`  [source](https://github.com/mcneel/opennurbs/blob/19df20038249fc40771dbd80201253a76100842c/opennurbs_evaluate_nurbs.h#L116-L120)




        """
        n = len(self.control_points)
        knots = (
                [0] * (self.degree + 1)
                + list(range(1, n - self.degree))
                + [n - self.degree] * (self.degree + 1)
        )

        return np.array(knots, float)

    def basis_function(self, t, i, k):
        """
        Calculating basis function with de Boor algorithm
        """
        # print(t,i,k)

        return deboor(self.knots, t, i, k)

    def evaluate(self, t: float):
        result = np.zeros(3, dtype=float)
        if t == 0.0:
            t += 1e-8
        elif t == 1.0:
            t -= 1e-8

        for i in range(self._control_points_count):
            b = self._cached_basis_func(t, i, self.degree)

            result[0] += b * self.control_points[i][0]
            result[1] += b * self.control_points[i][1]
            result[2] += b * self.control_points[i][2]
        return result

    @property
    def control_points(self):
        return self._control_points

    @control_points.setter
    def control_points(self, value):
        self._control_points = np.array(value, dtype=float)
        self.invalidate_cache()

    def __call__(self, t: float) -> tuple[float, float, float]:
        """
        Here write a solution to the parametric equation curves at the point corresponding to the parameter t.
        The function should return three numbers (x,y,z)
        """

        self._control_points_count = n = len(self.control_points)
        assert (
                n > self.degree
        ), "Expected the number of control points to be greater than the degree of the spline"
        return super().__call__(t)


class NURBSpline(BSpline):
    """
    Non-Uniform Rational BSpline (NURBS)
    Example:
        >>> spl = NURBSpline(np.array([(-26030.187675027133, 5601.3871095975337, 31638.841094491760),
        ...                   (14918.717302595671, -25257.061306278192, 14455.443462719517),
        ...                   (19188.604482326708, 17583.891501540096, 6065.9078795798523),
        ...                   (-18663.729281923122, 5703.1869371495322, 0.0),
        ...                   (20028.126297559378, -20024.715164607202, 2591.0893519960955),
        ...                   (4735.5467668945130, 25720.651181520021, -6587.2644037490491),
        ...                   (-20484.795362315021, -11668.741154421798, -14201.431195298581),
        ...                   (18434.653814767291, -4810.2095985021788, -14052.951382291201),
        ...                   (612.94310080525793, 24446.695569574043, -24080.735343204549),
        ...                   (-7503.6320665111089, 2896.2190847052334, -31178.971042788111)]
        ...                  ))

    """

    weights: np.ndarray

    def __init__(self, control_points, weights=None, degree=3, knots=None):
        self.weights = np.ones((len(control_points),), dtype=float) if weights is None else weights
        super().__init__(control_points, degree=degree, knots=knots)

        self.evaluate_multi = self._evaluate_multi
    def set_weights(self, weights=None):
        if weights is not None:
            if len(weights) == len(self.control_points):
                self.weights[:] = weights
                self.invalidate_cache()
            else:
                raise ValueError(
                    f"Weights must have the same length as the control points! Passed weights: {weights}, control_points size: {self._control_points_count}, control_points :{self.control_points}, weights : {weights}"
                )
    def invalidate_cache(self):
        super().invalidate_cache()

        self._cached_eval_func = lru_cache(maxsize=None)(self._evaluate)
    def split(self, t):
        return nurbs_split(self, t)
    def _evaluate(self, t:float):
        return evaluate_nurbs(t, self.control_points,self.knots,self.weights,self.degree)

    def _evaluate_multi(self, t: np.ndarray[float]):
        return evaluate_nurbs_multi(t, self.control_points, self.knots, self.weights, self.degree)

    def evaluate(self, t: float):
        """
        x, y, z = 0.0, 0.0, 0.0
        sum_of_weights = 0.0  # sum of weight * basis function

        if abs(t - 0.0) <= 1e-8:
            t = 0.0
        elif abs(t - 1.0) <= 1e-8:
            t = 1.0

        for i in range(self._control_points_count):
            b = self._cached_basis_func(t, i, self.degree)
            x += b * self.weights[i] * self.control_points[i][0]
            y += b * self.weights[i] * self.control_points[i][1]
            z += b * self.weights[i] * self.control_points[i][2]
            sum_of_weights += b * self.weights[i]
        # normalizing with the sum of weights to get rational B-spline
        x /= sum_of_weights
        y /= sum_of_weights
        z /= sum_of_weights"""

        return self._cached_eval_func(t)


    def __call__(self, t):
        """
        Here write a solution to the parametric equation Rational BSpline at the point corresponding
        to the parameter t. The function should return three numbers (x,y,z)
        """
        #self._control_points_count = len(self.control_points)
        #assert (
        #        self._control_points_count > self.degree
        #), "Expected the number of control points to be greater than the degree of the spline"
        #assert (
        #        len(self.weights) == self._control_points_count
        #), "Expected to have a weight for every control point"
        if isinstance(t, (float,int)):
            return self.evaluate(t)
        else:
            t=np.array(t)
            if t.ndim == 1:
                return np.array(self.evaluate_multi(t))
            else:
                return np.array(self.evaluate_multi(t.flatten())).reshape((*t.shape, 3))


