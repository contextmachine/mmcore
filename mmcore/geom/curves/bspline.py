from __future__ import annotations
import copy
import math

from functools import lru_cache

import numpy as np

from mmcore.geom.curves.curve import Curve
from mmcore.numeric.plane import WORLD_XY, evaluate_plane, inverse_evaluate_plane, inverse_evaluate_plane_arr, \
    evaluate_plane_arr
from mmcore.geom.curves.knot import find_span_binsearch, find_multiplicity
from mmcore.geom.curves.bspline_utils import insert_knot
from mmcore.geom.curves._nurbs import NURBSpline as CNURBSpline

__all__ = ['nurbs_split', 'NURBSpline']


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


class NURBSpline(CNURBSpline, Curve):
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
        Compared to scipy.interpolate.BSpline has weights (because it is a NURBS-spline not a B-spline),
        does not allow to have dimensionality of control points strictly (x,y,z,w), also has more CAD interface.
        For 3D points, for all weights equal to 1, the behaviour will be completely identical:
        >>> pts=np.array([(-26030.187675027133, 5601.3871095975337, 31638.841094491760),
        ...                   (14918.717302595671, -25257.061306278192, 14455.443462719517),
        ...                   (19188.604482326708, 17583.891501540096, 6065.9078795798523),
        ...                   (-18663.729281923122, 5703.1869371495322, 0.0),
        ...                   (20028.126297559378, -20024.715164607202, 2591.0893519960955),
        ...                   (4735.5467668945130, 25720.651181520021, -6587.2644037490491),
        ...                   (-20484.795362315021, -11668.741154421798, -14201.431195298581),
        ...                   (18434.653814767291, -4810.2095985021788, -14052.951382291201),
        ...                   (612.94310080525793, 24446.695569574043, -24080.735343204549),
        ...                   (-7503.6320665111089, 2896.2190847052334, -31178.971042788111)]
        ...                  )
        >>> degree=3
        >>> nspl=NURBSpline(pts, degree=degree)
        >>> from scipy.interpolate import BSpline
        >>> bspl=BSpline(nspl.knots, pts, degree)
        >>> print(tuple(nspl.interval()))
        (0.0, 7.0)
        >>> nspl.evaluate(3.5)
        array([11050.3324133 ,  2605.02029524, -2210.69702887])

        >>> bspl(3.5)
        array([11050.3324133 ,  2605.02029524, -2210.69702887])

        Just like scipy.interpolate.BSpline can also be extrapolated:
        >>> nspl.evaluate(-0.5)
        array([-120493.73709463,   99219.41340439,   68643.18874618])

        >>> bspl(-0.5)
        array([-120493.73709463,   99219.41340439,   68643.18874618])

        Unlike geomdl.NURBS.Curve, it is not limited to the range 0.,1. (does not default knots vector to the range 0 to 1),
        supports extrapolation, and is less nerdy )).

        Create mmcore NURBSpline object:
        >>> nspl=NURBSpline(pts, degree=3)

        Create geomdl.NURBS.Curve object:
        >>> from geomdl import NURBS
        >>> geomdl_nspl=NURBS.Curve()
        >>> geomdl_nspl.degree=3
        >>> geomdl_nspl.ctrlpts=nspl.control_points.tolist()
        >>> from geomdl.knotvector import generate
        >>> geomdl_nspl.knotvector=generate()

        Evaluate objects:
        >>> pt = nspl.evaluate(3.5)
        >>> pt
        array([11050.3324133 ,  2605.02029524, -2210.69702887]))

        >>> geomdl_pt = geomdl_nspl.evaluate_single(0.5) # geomdl will require the parameter to be 0 to 1
        >>> geomdl_pt
        [11050.33241329586, 2605.0202952442223, -2210.6970288670113]

        >>> np.allclose(geomdl_pt, pt)
        True

    """

    def __init__(self, control_points, degree=3, knots=None, weights=None):
        cpts = control_points if isinstance(control_points, np.ndarray) else np.array(control_points, dtype=float)
        if weights is not None:
            cpts = np.ones((len(control_points), 4), dtype=float)
            cpts[:, :control_points.shape[1]] = control_points
            cpts[:, 3] = weights
        if knots is not None:
            knots = knots if isinstance(knots, np.ndarray) else np.array(knots, dtype=float)

        super().__init__(cpts, degree=degree, knots=knots)

        self._evaluate_length_cached = lru_cache(maxsize=None)(self._evaluate_length)

    #def __init__(self, control_points, weights=None, degree=3, knots=None):
    #    self._cached_eval_func = lru_cache(maxsize=None)(self._evaluate)
    #
    #    self._control_points_count = len(control_points)
    #
    #    self._spline = CNURBSpline(np.asarray(control_points, dtype=float), degree=degree, knots=knots)
    #    #self.weights = np.array(self._spline.control_points)
    #    self._control_points = self._spline.control_points
    #    if weights is not None:
    #        pts = self._spline.get_control_points_4d()
    #        pts[:, -1] = weights
    #        self._spline.set_control_points_4d(pts)
    #
    #    super().__init__(init_derivatives=False)
    #@property
    #def weights(self):
    #    return self._spline.weights
    #
    #@weights.setter
    #def weights(self,v):
    #    self._spline.weights=v if isinstance(v, np.ndarray) else np.array(v,dtype=float)
    #
    #def __getstate__(self):
    #    return dict(control_points=np.asarray(self.control_points),
    #                weights=np.asarray(self.weights),
    #                degree=self.degree,
    #                knots=np.array(self.knots))
    #
    #def __setstate__(self, state):
    #    print(state)
    #    self._spline = CNURBSpline(np.asarray(state.get('control_points', state.get('_control_points')), dtype=float),
    #                               degree=state.get('degree'), knots=state.get('knots'))
    #    self._control_points = self._spline.control_points.base[:, :-1]
    #    self.weights = np.array(self._spline.control_points.base[:, -1])
    #    if state.get('weights') is not None:
    #        pts = self._spline.get_control_points_4d()
    #        pts[:, -1] = state.get('weights')
    #        self._spline.set_control_points_4d(pts)
    #    self._cached_eval_func = lru_cache(maxsize=None)(self._evaluate)
    #    self._control_points_count = len(self.control_points)
    #    self._evaluate_cached = lru_cache(maxsize=None)(self.evaluate)
    #    self._evaluate_length_cached = lru_cache(maxsize=None)(self.evaluate_length)
    #    self._cached_basis_func = lru_cache(maxsize=None)(self.basis_function)
    #
    #    self.invalidate_cache()

    #def generate_knots(self):
    #    self._spline.generate_knots()
    #    return self._spline.knots
    #
    #@property
    #def knots(self):
    #    return self._spline.knots.base
    #
    #@knots.setter
    #def knots(self, v):
    #    self._spline.knots = np.asarray(v)
    #    self.invalidate_cache()
    #
    #def basis_function(self, t, i, k):
    #    """
    #    Calculating basis function with de Boor algorithm
    #    """
    #    # print(t,i,k)
    #
    #    return deboor(self.knots, t, i, k)

    #@property
    #def weights(self):
    #    return np.array(self._spline.control_points.base[:,-1])
    #
    #@weights.setter
    #def weights(self, val):
    #    if len(val)== self._control_points_count :
    #        self._spline.control_points.base[:, -1]=val
    #    else:
    #        raise ValueError("Weights must have length equal to the number of control points")
    #def derivative(self, t):
    #    print('ddd')
    #    return self._spline.derivative(t)
    #
    #def second_derivative(self, t):
    #    return self._spline.second_derivative(t)
    #
    #def plane_at(self, t):
    #    return self._spline.plane_at(t)
    #
    #def tangent(self, t):
    #
    #    return self._spline.tangent(t)
    #
    #def curvature(self, t):
    #    return self._spline.curvature(t)
    #
    #def normal(self, t):
    #
    #    return self._spline.normal(t)

    #@property
    #def _control_points(self):
    #    return self._spline.control_points.base[:,:-1]
    #
    #@_control_points.setter
    #def _control_points(self, val):
    #    if len(val)== self._control_points_count :
    #        self._spline.control_points[:, :-1]=val
    #    else:
    #        zz = np.ones((len(val), 4))
    #        zz[:,:-1]=val
    #        self._spline.set_control_points_4d(zz)

    #@property
    #def degree(self):
    #    return self._spline.degree
    #
    #@degree.setter
    #def degree(self, d):
    #    self._spline.degree = d
    #    self.invalidate_cache()
    #
    #@property
    #def control_points(self):
    #    return self._control_points
    #
    #@control_points.setter
    #def control_points(self, value):
    #    self._control_points = np.asarray(value)
    #    self.invalidate_cache()
    #
    #def set_weights(self, weights=None):
    #    if weights is not None:
    #        if len(weights) == len(self.control_points):
    #            self.weights[:] = weights
    #            self.invalidate_cache()
    #        else:
    #            raise ValueError(
    #                f"Weights must have the same length as the control points! Passed weights: {weights}, control_points size: {self._control_points_count}, control_points :{self.control_points}, weights : {weights}"
    #            )
    #
    #def invalidate_cache(self):
    #
    #    super().invalidate_cache()
    #    self._cached_eval_func.cache_clear()
    def points(self, *args, **kwargs):
        if self.degree==1:
            return self.control_points[:-1]
        else:
            return super().points(*args, **kwargs)
    def split(self, t):
        return nurbs_split(self, t)

    def evaluate(self, t):
        return CNURBSpline.evaluate(self, t)

    #def _evaluate(self, t: float):
    #    return evaluate_nurbs(t, self._control_points, self._spline.knots, self.weights, self._spline.degree).base
    #
    #def _evaluate_multi(self, t: np.ndarray[float]):
    #    return evaluate_nurbs_multi(t, self._control_points, self._spline.knots, self.weights, self._spline.degree).base

    #def evaluate(self, t: float):
    #    """
    #    x, y, z = 0.0, 0.0, 0.0
    #    sum_of_weights = 0.0  # sum of weight * basis function
    #
    #    if abs(t - 0.0) <= 1e-8:
    #        t = 0.0
    #    elif abs(t - 1.0) <= 1e-8:
    #        t = 1.0
    #
    #    for i in range(self._control_points_count):
    #        b = self._cached_basis_func(t, i, self.degree)
    #        x += b * self.weights[i] * self.control_points[i][0]
    #        y += b * self.weights[i] * self.control_points[i][1]
    #        z += b * self.weights[i] * self.control_points[i][2]
    #        sum_of_weights += b * self.weights[i]
    #    # normalizing with the sum of weights to get rational B-spline
    #    x /= sum_of_weights
    #    y /= sum_of_weights
    #    z /= sum_of_weights"""
    #
    #    return self._cached_eval_func(t)
    #
    #def evaluate_multi(self, t):
    #    return self._spline.evaluate_multi(np.asarray(t))

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
        if isinstance(t, (float, int)):
            return self.evaluate(t)
        else:
            t = np.array(t)
            if t.ndim == 1:
                return np.array(self.evaluate_multi(t))
            else:
                return np.array(self.evaluate_multi(t.flatten())).reshape((*t.shape, 3))
    @classmethod
    def create_circle(cls, origin=(0.,0.,0.), radius=1, plane=WORLD_XY) -> "OCCNurbsCurve":
        """Construct a NURBS curve from a circle.

        Parameters
        ----------
        circle : :class:`~compas.geometry.Circle`
            The circle geometry.

        Returns
        -------
        :class:`OCCNurbsCurve`

        """

        w =  math.sqrt(2)/2

        points=np.array([[radius, 0., 0.,  1.],
         [radius, radius, 0.,  w],
         [0., radius, 0. ,  1. ],
         [-radius, radius, 0.,   w ],
         [-radius, 0., 0. ,1.  ],
         [-radius, -radius, 0., w  ],
         [0., -radius, 0. ,1.  ],
         [radius, -radius, 0. , w  ],
         [radius, 0., 0. , 1.  ]],dtype=float)
        points[:,:-1]=np.array(evaluate_plane_arr(plane,points[:,:-1] ))

        knots = np.array([0, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi], dtype=float)
        mults = np.array([3, 2, 2, 2, 3],dtype=int)
        knots=np.repeat(knots,mults)
        return cls(control_points=points, degree=2, knots=knots)

from .knot import interpolate_curve


def interpolate_nurbs_curve(points, degree=3, use_centripetal=False):
    degree=min(len(points)-1, degree)

    cpts, knots, degree = interpolate_curve(points, degree=degree, use_centripetal=use_centripetal)
    if cpts.shape[1] < 3:
        z = np.zeros((cpts.shape[0], 3), dtype=float)

        z[:, :cpts.shape[1]] = cpts
        cpts = z
    spl = NURBSpline(cpts, degree=degree, knots=knots)
    #spl.knots=np.array(knots,dtype=float)
    return spl


