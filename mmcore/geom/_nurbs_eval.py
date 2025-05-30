from __future__ import annotations


import math
import warnings

import numpy as np


from typing import TypedDict, NamedTuple
from numpy.typing import NDArray
from mmcore.geom import nurbs

from mmcore.numeric.calgorithms import evaluate_curvature
np.set_printoptions(suppress=True)

# ======================================================================
# Namedtuple definitions for NURBS surface and curve representations.
# ======================================================================
class BSplineCurveTuple(NamedTuple):
    order:int
    knot:NDArray[float]
    control_points:NDArray[float]


class NURBSCurveTuple(NamedTuple):
    order:int
    knot:NDArray[float]
    control_points:NDArray[float]
    weights:NDArray[float]

    def start(self):

        return evaluate_nurbs_curve(self,self.knot[self.order-1],0)['C']
    def end(self):

        return evaluate_nurbs_curve(self,self.knot[self.control_points.shape[0]],0)['C']

    @property
    def degree(self):
        return self.order-1
class BSplineSurfaceTuple(NamedTuple):
    order_u:int
    order_v: int
    knot_u:NDArray[float]
    knot_v:NDArray[float]
    control_points:NDArray[float]

class NURBSSurfaceTuple(NamedTuple):
    order_u:int
    order_v: int
    knot_u:NDArray[float]
    knot_v:NDArray[float]
    control_points:NDArray[float]
    weights:NDArray[float]
    @property
    def knots_u(self):
        return self.knot_u
    @property
    def knots_v(self):
        return self.knot_v
    @property
    def degree(self):
        return (self.order_u-1,self.order_v-1)
class EvaluateCurveData(TypedDict):
    """
    :ivar C: Point at.
    :type C: numpy.ndarray[float]
    :ivar C1: First derivative.
    :type C1: numpy.ndarray[float]
    :ivar C2: Second derivative
    :type C2: numpy.ndarray[float]
    """
    C:NDArray[float]
    C1: NDArray[float]
    C2: NDArray[float]

class EvaluateCurveDifferentialData(TypedDict):
    """
    :ivar C: Point at.
    :type C: numpy.ndarray[float]
    :ivar C1: First derivative.
    :type C1: numpy.ndarray[float]
    :ivar C2: Second derivative
    :type C2: numpy.ndarray[float]
    :ivar K:  RCurvature vector
    :type K: numpy.ndarray[float]
    :ivar Ut: Unit tangent vector
    :type Ut: numpy.ndarray[float]
    """
    C:NDArray[float]
    C1: NDArray[float]
    C2: NDArray[float]
    K: NDArray[float]
    Ut:NDArray[float]


class EvaluateSurfaceData(TypedDict):

    S:NDArray[float]
    Su: NDArray[float]
    Sv: NDArray[float]
    Suu: NDArray[float]
    Suv: NDArray[float]
    Svv: NDArray[float]


def nurbs_interval(knots, degree:int)->tuple[float,float] :
    """
    Calculate the effective parameter interval for a NURBS curve (or a surface in one direction)
    given its knot vector and degree.

    Parameters:
        knots (list or tuple of float): The knot vector.
        degree (int): The degree of the NURBS curve (or surface in the specific direction).

    Returns:
        tuple: A tuple (u_start, u_end) representing the active interval [u_p, u_{n+1}].

    Raises:
        ValueError: If the knot vector length is not consistent with the degree.
    """
    num_knots = len(knots)
    # The number of control points (n+1) can be calculated from the knot vector length:
    num_control_points = num_knots - degree - 1

    if num_control_points < 1:
        raise ValueError("Invalid knot vector or degree: not enough knots for the given degree.")

    # The effective interval is [knots[degree], knots[control_points]]
    return (float(knots[degree]),float( knots[num_control_points]))


def to_homogeneous_1d(control_points, weights):
    """Convert curve control points to homogeneous coordinates by multiplying by weights.

    Args:
        control_points: Array of control points (Nx3 or NxD)
        weights: Array of weights (N)

    Returns:
        Array of homogeneous control points (Nx(D+1)) where last column is weight
    """
    dim = control_points.shape[1]
    result = np.zeros((len(control_points), dim + 1))
    for i in range(len(control_points)):
        # Multiply xyz by weight
        result[i, :-1] = control_points[i] * weights[i]
        # Store weight as last coordinate
        result[i, -1] = weights[i]
    return result


def from_homogeneous_1d(homogeneous_points):
    """Convert homogeneous control points back to Cartesian coordinates.

    Args:
        homogeneous_points: Array of homogeneous control points (Nx(D+1))

    Returns:
        Tuple of (control_points, weights) where:
        - control_points: Array of control points (NxD)
        - weights: Array of weights (N)
    """
    # print(homogeneous_points)
    _cpt = np.asarray(homogeneous_points)
    weights = np.ascontiguousarray(_cpt[..., -1])
    dim = _cpt.shape[1] - 1

    control_points = np.zeros((_cpt.shape[0], dim))
    for i in range(_cpt.shape[0]):
        # Divide homogeneous coordinates by weight
        control_points[i] = _cpt[i, :-1] / _cpt[i, -1]

    return np.ascontiguousarray(control_points), weights


def to_homogeneous_2d(control_points, weights):
    """Convert surface control points to homogeneous coordinates by multiplying by weights.

    Args:
        control_points: Array of control points (MxNx3 or MxNxD)
        weights: Array of weights (MxN)

    Returns:
        Array of homogeneous control points (MxNx(D+1)) where last column is weight
    """
    dim = control_points.shape[2]
    result = np.zeros((control_points.shape[0], control_points.shape[1], dim + 1))
    for i in range(control_points.shape[0]):
        for j in range(control_points.shape[1]):
            # Multiply xyz by weight
            result[i, j, :-1] = control_points[i, j] * weights[i, j]
            # Store weight as last coordinate
            result[i, j, -1] = weights[i, j]
    return result


def from_homogeneous_2d(homogeneous_points):
    """Convert homogeneous surface control points back to Cartesian coordinates.

    Args:
        homogeneous_points: Array of homogeneous control points (MxNx(D+1))

    Returns:
        Tuple of (control_points, weights) where:
        - control_points: Array of control points (MxNxD)
        - weights: Array of weights (MxN)
    """
    _cpt = np.asarray(homogeneous_points)
    weights = np.ascontiguousarray(_cpt[..., -1])
    dim = _cpt.shape[2] - 1

    control_points = np.zeros((_cpt.shape[0], _cpt.shape[1], dim))
    for i in range(_cpt.shape[0]):
        for j in range(_cpt.shape[1]):
            # Divide homogeneous coordinates by weight
            control_points[i, j] = _cpt[i, j, :-1] / _cpt[i, j, -1]

    return np.ascontiguousarray(control_points), weights

# Operations

def _find_span_linear(degree, knot_vector, num_ctrlpts, knot, **kwargs):
    span = degree + 1  # knot span index starts from zero
    while span < num_ctrlpts and knot_vector[span] <= knot:
        span += 1
    return span - 1


def compute_basis_function_derivatives_np(degree, knot_vector, span, knot, order):
    """
    Compute the derivatives of B-spline (or NURBS) basis functions using numpy for efficiency.
    Args:
        degree (int): The degree p of the basis functions.
        knot_vector (array-like): The knot vector U.
        span (int): The knot span index.
        knot (float): The parameter value u at which to evaluate.
        order (int): The maximum derivative order to compute.
    Returns:
        np.ndarray: A 2D array 'ders' of shape (order+1, degree+1) where ders[k, j]
                    is the k-th derivative of the j-th basis function.
    """
    knot_vector = np.asarray(knot_vector, dtype=float)
    # Precompute left/right arrays using vectorized slicing.
    left = np.empty(degree + 1, dtype=float)
    right = np.empty(degree + 1, dtype=float)
    left[0] = 0.0
    right[0] = 0.0
    j_arr = np.arange(1, degree + 1)
    left[1:] = knot - knot_vector[span + 1 - j_arr]
    right[1:] = knot_vector[span + j_arr] - knot
    # Build the 'ndu' table.
    ndu = np.zeros((degree + 1, degree + 1), dtype=float)
    ndu[0, 0] = 1.0
    for j in range(1, degree + 1):
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved
    # Compute unscaled derivative coefficients.
    ders = np.zeros((order + 1, degree + 1), dtype=float)
    for r in range(degree + 1):
        d_coeffs = np.zeros(order + 1, dtype=float)
        d_coeffs[0] = ndu[r, degree]
        a = np.zeros((2, order + 1), dtype=float)
        a[0, 0] = 1.0
        s1 = 0  # current row in temporary array 'a'
        s2 = 1  # next row in 'a'
        for k in range(1, order + 1):
            d = 0.0
            rk = r - k
            pk = degree - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else degree - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            d_coeffs[k] = d
            s1, s2 = s2, s1  # swap rows
        ders[:, r] = d_coeffs
    # Factorial scaling: scale k-th derivative by degree*(degree-1)*...*(degree-k+1)
    scales = np.empty(order + 1, dtype=float)
    scales[0] = 1.0
    for k in range(1, order + 1):
        scales[k] = scales[k - 1] * (degree - k + 1)
    ders[1:, :] *= scales[1:, np.newaxis]
    return ders
from mmcore.numeric.aabb import aabb
def nurbs_bbox(obj:NURBSSurfaceTuple|NURBSCurveTuple|BSplineSurfaceTuple|BSplineCurveTuple):
    return aabb(obj.control_points    )
def evaluate_nurbs_curve(curve:NURBSCurveTuple, u, d_order=2)->EvaluateCurveData:
    """
    Evaluate a rational NURBS curve at parameter u.
    Returns a dictionary with keys:
      'C'  : the evaluated point,
      'C1' : the first derivative,
      'C2' : the second derivative.
    """
    p = curve.order - 1
    n = len(curve.control_points)
    U = curve.knot[:]  # assume knot vector is a list or numpy array
    span = _find_span_linear(p, U, n, u)
    d = min(d_order, p)

    # Compute basis functions and their derivatives.
    # Assumes existence of a function 'compute_basis_function_derivatives_np'
    ders = np.array(compute_basis_function_derivatives_np(p, U, span, u, d),dtype=float)
    # ders has shape (d+1, p+1)

    dim = len(curve.control_points[0])
    # Allocate homogeneous derivatives d_hom[k] for k = 0, 1, ..., d.
    d_hom = np.zeros((d + 1,dim+1))




    for k in range(d + 1):
        for j in range(p + 1):
            i = span - p + j
            P = np.array(curve.control_points[i])
            w = curve.weights[i]
            # Form the homogeneous coordinate [w*P, w]
            H = np.zeros(dim + 1)
            H[:dim] = P * w
            H[dim] = w
            d_hom[k] += ders[k, j] * H

    result:EvaluateCurveData = {}
    # Dehomogenize to get the point on the curve.
    C = d_hom[0,:dim] / d_hom[0,dim]
    result["C"] = C

    # First derivative.
    if d >= 1:
        C1 = (d_hom[1][:dim] - d_hom[1][dim] * C) / d_hom[0][dim]
        result["C1"] = C1
    else:
        result["C1"] = np.zeros(dim,dtype=float)

    # Second derivative.
    if d >= 2:
        C2 = (d_hom[2][:dim] - d_hom[2][dim] * C) / d_hom[0][dim] - 2 * (d_hom[1][dim] / d_hom[0][dim]) * result["C1"]
        result["C2"] = C2
    else:
        result["C2"] = np.zeros(dim,dtype=float)

    return result

def evaluate_nurbs_surface(surface:NURBSSurfaceTuple, u, v, d_order=2)->EvaluateSurfaceData:
    """
    Evaluate a rational NURBS surface at (u,v). Returns a dictionary SKL with keys:
      'S'   : the 3D (or n–dimensional) point,
      'Su'  : first derivative in u,
      'Sv'  : first derivative in v,
      'Suu' : second derivative in u,
      'Suv' : mixed second derivative,
      'Svv' : second derivative in v.
    """

    # print(surface, u, v)

    surface1=surface

    p = surface1.order_u - 1
    q = surface1.order_v - 1
    nu = len(surface1.control_points)
    nv = len(surface1.control_points[0])
    U = surface1.knot_u[:]  # assume these are already lists/numpy arrays
    V = surface1.knot_v[:]
    span_u = _find_span_linear(p, U, nu, u)
    span_v = _find_span_linear(q, V, nv, v)
    # print(p, U, span_u, u, d_order)
    du = min(d_order, p)
    dv = min(d_order, q)
    ders_u = np.array(compute_basis_function_derivatives_np(p, U, span_u, u, du),dtype=float)
    # print(q, V, span_v, v, d_order)
    ders_v = np.array(compute_basis_function_derivatives_np(q, V, span_v, v, dv),dtype=float)
    # print("DU", ders_u)
    # print("DV", ders_v)

    SKL:EvaluateSurfaceData = {}
    dim = len(surface1.control_points[0][0])
    # print(surface)
    # Allocate and initialize homogeneous derivatives.
    d = [[np.zeros(dim + 1,dtype=float) for l in range(dv + 1)] for k in range(du + 1)]
    for k in range(du + 1):
        for l in range(dv + 1):
            d[k][l] = np.zeros(dim + 1)
    # Compute homogeneous surface derivatives d[k][l]
    for l in range(q + 1):
        temp = [np.zeros(dim + 1) for i in range(du + 1)]
        for k in range(p + 1):
            i_index = span_u - p + k
            j_index = span_v - q + l
            w = surface1.weights[i_index, j_index]
            cp = np.array(surface1.control_points[i_index][j_index])*w

            tmp = np.zeros(dim + 1)
            tmp[:dim] = cp
            tmp[dim] = w
            for i in range(du + 1):
                temp[i] += ders_u[i, k] * tmp
        for j in range(dv + 1):
            for i in range(du + 1):
                d[i][j] += ders_v[j, l] * temp[i]
    # Dehomogenize
    SKL["S"] = d[0][0][:dim] / d[0][0][dim]
    SKL["Su"] = np.zeros(dim,dtype=float)
    SKL["Sv"] = np.zeros(dim,dtype=float)
    SKL["Suu"] = np.zeros(dim,dtype=float)
    SKL["Suv"] = np.zeros(dim,dtype=float)
    SKL["Svv"] = np.zeros(dim,dtype=float)
    if du >= 1:
        Su = (d[1][0][:dim] - d[1][0][dim] * SKL["S"]) / d[0][0][dim]

        SKL["Su"] = Su

    if dv >= 1:
        Sv = (d[0][1][:dim] - d[0][1][dim] * SKL["S"]) / d[0][0][dim]
        SKL["Sv"] = Sv
    if du >= 2:
        Suu = (d[2][0][:dim] - d[2][0][dim] * SKL["S"]) / d[0][0][dim] - 2 * (d[1][0][dim] / d[0][0][dim]) * SKL["Su"]
        SKL["Suu"] = Suu

    if dv >= 2:

        Svv = (d[0][2][:dim] - d[0][2][dim] * SKL["S"]) / d[0][0][dim] - 2 * (d[0][1][dim] / d[0][0][dim]) * SKL["Sv"]

        SKL["Svv"] = Svv

    if du >= 2 or dv >= 2:
        Suv = (
            (d[1][1][:dim] - d[1][1][dim] * SKL["S"]) / d[0][0][dim]
            - (d[1][0][dim] / d[0][0][dim]) * SKL["Sv"]
            - (d[0][1][dim] / d[0][0][dim]) * SKL["Su"]
        )
        SKL["Suv"] = Suv
    # print(SKL)
    return SKL


def evaluate_bspline_curve(curve: BSplineCurveTuple, u: float) -> NDArray[float]:
    p = curve.order - 1
    U = curve.knot
    pts = curve.control_points
    n = len(pts)
    span = _find_span_linear(p, U, n, u)
    d = [pts[span - p + i].copy() for i in range(p + 1)]
    for r in range(1, p + 1):
        for i in range(p, r - 1, -1):
            alpha = (u - U[span - p + i]) / (U[i + span - r + 1] - U[span - p + i])
            d[i] = (1 - alpha) * d[i - 1] + alpha * d[i]
    return d[p]


def bspline_basis(j, degree, knot_vector, u):
    """
    Recursively compute the B-spline basis function N_{j,degree}(u) using the Cox–de Boor formula.

    Parameters:
        j           : index of the basis function.
        degree      : degree of the basis function.
        knot_vector : array of knot values.
        u           : parameter at which to evaluate.

    Returns:
        Value of the basis function.
    """
    if degree == 0:
        # Special care at the right endpoint.
        if knot_vector[j] <= u < knot_vector[j + 1] or (u == knot_vector[-1] and u == knot_vector[j + 1]):
            return 1.0
        else:
            return 0.0
    denom1 = knot_vector[j + degree] - knot_vector[j]
    denom2 = knot_vector[j + degree + 1] - knot_vector[j + 1]
    term1 = 0.0
    term2 = 0.0
    if denom1 != 0:
        term1 = (u - knot_vector[j]) / denom1 * bspline_basis(j, degree - 1, knot_vector, u)
    if denom2 != 0:
        term2 = (knot_vector[j + degree + 1] - u) / denom2 * bspline_basis(j + 1, degree - 1, knot_vector, u)
    return term1 + term2

def evaluate_nurbs_curve_array(curve: NURBSCurveTuple, t, d_order=0):
    """
    Evaluate a NURBS curve (which may be rational) at parameter value t.
    d_order = 0 returns only the point; d_order = 1 returns [point, derivative].
    Works in any dimension.
    """
    return np.array(list(evaluate_nurbs_curve(curve, t, d_order).values()))


def evaluate_nurbs_curve_curvature(curve, u, data:EvaluateCurveData|None=None)->EvaluateCurveDifferentialData:
    if data is None:
        data=EvaluateCurveDifferentialData(**evaluate_nurbs_curve(curve, u, d_order=2))

    dim=data['C'].shape[0]
    data['K']=np.zeros(dim,dtype=float)
    data['Ut'] = np.zeros(dim, dtype=float)

    recalculate=evaluate_curvature(data['C1'], data['C2'],data['K'],data['Ut'])
    return data


def _curve_degree(self:BSplineCurveTuple|NURBSCurveTuple)->int:
    return self.order-1

def _curve_interval(self:BSplineCurveTuple|NURBSCurveTuple)->tuple[float,float]:
    _=_curve_degree(self)
    return nurbs_interval(self.knot,_)

def _surface_degree(self: BSplineSurfaceTuple|NURBSSurfaceTuple)->tuple[int,int]:
    return self.order_u - 1,self.order_v - 1

def _surface_interval(self:BSplineSurfaceTuple|NURBSSurfaceTuple)->tuple[tuple[float,float],tuple[float,float]]:
    _u,_v=_surface_degree(self)
    return  (nurbs_interval(self.knot_u,_u), nurbs_interval(self.knot_v,_v))

def _copy_curve(curve:BSplineCurveTuple|NURBSCurveTuple)->BSplineCurveTuple|NURBSCurveTuple:
    cpts=np.copy(curve.control_points)
    knots=np.copy(curve.knot)
    if isinstance(curve,NURBSCurveTuple):
        return NURBSCurveTuple(curve.order,knots,cpts, np.copy(curve.weights))

    return BSplineCurveTuple(curve.order,knots,cpts)

def _copy_surface(surface:BSplineSurfaceTuple|NURBSSurfaceTuple)->BSplineSurfaceTuple|NURBSSurfaceTuple:
    cpts=np.copy(surface.control_points)
    knots_u=np.copy(surface.knot_u)
    knots_v = np.copy(surface.knot_v)
    if isinstance(surface,NURBSSurfaceTuple):
        return NURBSSurfaceTuple(surface.order_u,surface.order_v,knots_u,knots_v,cpts,np.copy(surface.weights))

    return BSplineSurfaceTuple(surface.order_u,surface.order_v,knots_u,knots_v,cpts)


# Construction
def _process_knots(knots):
    #if isinstance(knots,list):
    #    return list(knots)
    #else:
    #    return np.asarray(knots).tolist()
    return np.array(knots,dtype=float)

def nurbs_surface(control_points, knots_u, knots_v, degree: tuple[int, int] | None = None, *, weights=None,
                  order: tuple[int, int] | None = None, **kwargs)->NURBSSurfaceTuple:
    """
    Generates a NURBS (Non-Uniform Rational B-Splines) surface representation by processing
    control points, knot vectors, degree or order, and optional weights. It ensures proper
    initialization and normalization of the input data and returns a tuple containing the
    required components to represent the NURBS surface.

    :param control_points: A 2D array-like structure representing the coordinates of the
        control points for the surface.
    :param knots_u: A 1D array-like structure or sequence of numbers specifying the knot
        vector along the u-direction.
    :param knots_v: A 1D array-like structure or sequence of numbers specifying the knot
        vector along the v-direction.
    :param degree: An optional tuple of two integers representing the degree of the NURBS
        surface along the u and v directions respectively. Defaults to None.
    :param weights: An optional 2D array-like structure of weights corresponding to the
        control points. Defaults to an array of ones if not provided.
    :param order: An optional tuple of integers indicating the orders in the u and v
        directions. If not provided, it defaults to computed orders based on the control
        points or degree.
    :param kwargs: Additional keyword arguments that might be passed but are not used in
        this function.
    :return: A NURBSSurfaceTuple with the following elements:
        - Order in the u-direction.
        - Order in the v-direction.
        - Processed knot vector in the u-direction.
        - Processed knot vector in the v-direction.
        - Control points as a 2D array without weights (if they were provided).
        - Computed or provided weights for the control points.
    :rtype: NURBSSurfaceTuple
    """
    control_points=np.array(control_points,dtype=float)

    u_size,v_size=control_points.shape[0],control_points.shape[1]
    if degree is not None:
        order_u,order_v=degree[0]+1,degree[1]+1
    elif order is not None:
        order_u, order_v=order
    else:
        order_u, order_v =min(u_size, 4),min(v_size, 4)
    if weights is None:
        if control_points.shape[-1]==4:
            weights=np.ascontiguousarray(control_points[...,-1])
            control_points=np.ascontiguousarray(control_points[...,:-1])

        else:
            weights=np.ones((u_size,v_size),dtype=float)
    else:
        weights=np.array(weights,dtype=float)

    knots_u=_process_knots(knots_u)
    knots_v=_process_knots(knots_v)
    return NURBSSurfaceTuple(order_u, order_v, knots_u , knots_v, control_points, weights)


def nurbs_curve(control_points, knots, degree: int| None = None, *, weights=None,
                  order: int| None = None, **kwargs)->NURBSCurveTuple:
    """
    Constructs a Non-Uniform Rational B-Spline (NURBS) curve using the provided
    control points, knot vector, degree, and optional weights.

    This method initializes a NURBS representation based on the given information
    about control points, knot sequence, and order. The degree can be specified
    directly, or derived from the order parameter. If weights are not defined, all
    weights are set to 1 by default unless the control points include them as a
    fourth dimension. The function also processes the knot vector to ensure
    compatibility.

    :param control_points: Array-like collection of control points that define
        the shape of the curve. If weights are embedded as the fourth dimension,
        they will be extracted automatically.
    :type control_points: numpy.ndarray
    :param knots: The knot vector specifying parameter dividers.
    :type knots: array-like
    :param degree: Degree of the NURBS curve. If provided, it is used to calculate
        the order by adding 1.
    :type degree: int, optional
    :param weights: Optional weights for the control points. Defaults to 1 if
        omitted or extracted from the control points if provided in 4D.
    :type weights: numpy.ndarray, optional
    :param order: The order of the NURBS curve, equal to degree + 1. Either degree
        or order must be provided.
    :type order: int, optional
    :param kwargs: Additional arguments for further customization of the NURBS
        curve initialization process.
    :type kwargs: dict
    :return: An instance of `NURBSCurveTuple` containing the order, processed
        knots, control points, and weights.
    :rtype: NURBSCurveTuple
    """
    control_points = np.array(control_points, dtype=float)

    size = control_points.shape[0]
    if degree is not None:
        order = degree + 1
    elif order is not None:
        pass
    else:
        order= min(size, 4)
    if weights is None:
        if control_points.shape[-1] == 4:
            weights = np.ascontiguousarray(control_points[..., -1])
            control_points = np.ascontiguousarray(control_points[..., :-1])

        else:
            weights = np.ones((size,), dtype=float)
    knots = _process_knots(knots)

    return NURBSCurveTuple(order, knots, control_points,
                             weights)

def _join_weights_1d(pts,weights):
    """Join control points and weights, but does not apply homogeneous transformation."""
    cpts=np.zeros((pts.shape[0],pts.shape[1]+1) )
    for i in range(pts.shape[0]):
        cpts[i,:-1]=pts[i]
        cpts[i,-1]=weights[i]
    return cpts


# Conversion for surfaces
def _join_weights(pts,weights):
    """Join control points and weights for surfaces, but does not apply homogeneous transformation."""
    cpts=np.zeros((*pts.shape[:-1],pts.shape[-1]+1) )
    for i in range(pts.shape[0]):
        for j in range(pts.shape[1]):
            cpts[i,j,:-1]=pts[i,j]
            cpts[i,j,-1]=weights[i,j]
    return cpts


def _nurbs_to_tuple(s1:nurbs.NURBSCurve | nurbs.NURBSSurface)->NURBSCurveTuple | NURBSSurfaceTuple:
    if isinstance(s1,nurbs.NURBSSurface):
        
        surf1 = NURBSSurfaceTuple(order_u=s1.degree[0] + 1, order_v=s1.degree[1] + 1, knot_u=s1.knots_u.tolist(),
                              knot_v=s1.knots_v.tolist(), control_points=np.array(s1.control_points),
                              weights=np.ascontiguousarray(s1.control_points_w[..., -1]))
        return surf1
    elif isinstance(s1,nurbs.NURBSCurve):
        cpts,weights=from_homogeneous_1d(np.array(s1.control_pointsw))
        curve1 = NURBSCurveTuple(order=s1.degree + 1,knot=s1.knots.tolist(),
                                  control_points=cpts,
                                  weights=weights)
        return curve1
    else:

        raise TypeError(f"Arguments must be {nurbs.NURBSCurve.__module__}.{nurbs.NURBSCurve.__name__} or {nurbs.NURBSSurface.__module__}.{nurbs.NURBSSurface.__name__}, not {type(s1).__name__}")


def _tuple_to_nurbs(obj:BSplineCurveTuple|NURBSCurveTuple|BSplineSurfaceTuple|NURBSSurfaceTuple):
    if isinstance(obj,(NURBSSurfaceTuple,BSplineSurfaceTuple)):
        degree=obj.order_u-1,obj.order_v-1

        pts= to_homogeneous_2d(obj.control_points,obj.weights)

        return nurbs.NURBSSurface(pts, degree,np.array(obj.knot_u),np.array(obj.knot_v))
    elif isinstance(obj,(NURBSCurveTuple,BSplineCurveTuple)):
        degree = obj.order - 1

        pts = to_homogeneous_1d(obj.control_points, obj.weights) if isinstance(obj,NURBSCurveTuple) else obj.control_points

        return nurbs.NURBSCurve(pts, degree,knots=np.array(obj.knot))
    else:
        raise TypeError(
            f"Arguments must be {BSplineCurveTuple.__name__}|{NURBSCurveTuple.__name__}|{BSplineSurfaceTuple.__name__}|{NURBSSurfaceTuple.__name__}, not {type(obj).__name__}")


'''
def join_weights(surf:NURBSSurfaceTuple):
    ptsw = np.zeros((*surf.control_points.shape[:-1], 4))

    ptsw[..., :-1] = surf.control_points
    ptsw[..., -1] = surf.weights
    return ptsw
def bspline_basis(j, L, t, knot):
    """
    Evaluate the j-th B-spline basis function of order L at parameter value t using Cox-de Boor.
    """
    if L == 1:
        if (knot[j] <= t < knot[j + 1]) or (t == knot[-1] and t == knot[j + 1]):
            return 1.0
        else:
            return 0.0
    else:
        denom1 = knot[j + L - 1] - knot[j]
        term1 = 0.0
        if denom1 != 0:
            term1 = (t - knot[j]) / denom1 * bspline_basis(j, L - 1, t, knot)
        denom2 = knot[j + L] - knot[j + 1]
        term2 = 0.0
        if denom2 != 0:
            term2 = (knot[j + L] - t) / denom2 * bspline_basis(j + 1, L - 1, t, knot)
        return term1 + term2


def evaluate_bspline_curve(curve:BSplineCurveTuple, t):
    """
    Evaluate a (non-rational) B-spline curve at parameter value t.
    """
    #'order', 'knot', 'control_points', 'weights'
    nc=NURBSCurveTuple(curve.order, curve.knot,curve.control_points, np.ones(curve.control_points.shape[:-1],dtype=float))
    return evaluate_nurbs_curve_array(nc, t, d_order=0)


def evaluate_bspline_curve_derivative(curve, t):
    """
    Evaluate the derivative of the B-spline curve at t using formula (48):
      S'(t) = (L-1) * sum_{j=1}^{n-1} (Q_j - Q_{j-1})/(s_{j+L-1} - s_j) * N_{j,L-1}(t)
    """
    nc = NURBSCurveTuple(curve.order, curve.knot, curve.control_points,
                         np.ones(curve.control_points.shape[:-1], dtype=float))
    return evaluate_nurbs_curve_array(nc, t, d_order=1)[1]





def compute_left_right_arrays(degree, knot, knot_vector, span):
    """
    Compute the arrays of distances from the parameter value to neighboring knots.

    In the literature these are called the left and right differences. For each index j,
    the arrays are defined as:

       left[j]  = knot - knot_vector[span + 1 - j]
       right[j] = knot_vector[span + j] - knot

    Args:
        degree (int): The degree p of the basis functions.
        knot (float): The parameter value u at which to evaluate.
        knot_vector (list or tuple): The knot vector U.
        span (int): The knot span index.

    Returns:
        tuple: (left, right), each a list of length (degree+1).
    """
    left = [None] * (degree + 1)

    right = [None] * (degree + 1)
    #print(degree, knot, knot_vector, span)
    ixs1 = []
    ixs2 = []
    for j in range(1, degree + 1):
        ixs1.append(span + 1 - j)
        ixs2.append(span + j)

        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot
    #print(ixs1)
    #print(ixs2)
    return left, right


def compute_ndu_row(j, left, right, ndu):
    """
    Compute one row of the 'ndu' table.

    For a given level j (1 <= j <= degree), the ndu table is filled according to
    a convolution–like recurrence:

       ndu[j][r] = right[r+1] + left[j - r]     for r in 0 <= r < j
       temp = ndu[r][j-1] / ndu[j][r]
       ndu[r][j] = saved + right[r+1] * temp
       saved = left[j - r] * temp

    Finally, ndu[j][j] is set to the last saved value.

    Args:
        j (int): The current row (or level) in the recurrence.
        left (list of float): Precomputed left differences.
        right (list of float): Precomputed right differences.
        ndu (list of list of float): The working table, modified in place.
    """
    saved = 0.0
    for r in range(j):
        ndu[j][r] = right[r + 1] + left[j - r]
        temp = ndu[r][j - 1] / ndu[j][r]
        ndu[r][j] = saved + right[r + 1] * temp
        saved = left[j - r] * temp
    ndu[j][j] = saved


def compute_ndu(degree, knot, knot_vector, span):
    """
    Build the complete 'ndu' table for evaluating the B-spline basis functions.

    This function first computes the left/right arrays and then fills
    the ndu table level by level.

    Args:
        degree (int): The degree p of the basis functions.
        knot (float): The parameter value u at which to evaluate.
        knot_vector (list or tuple): The knot vector U.
        span (int): The knot span index.

    Returns:
        tuple: (ndu, left, right)
            - ndu: A 2D list of size (degree+1) x (degree+1) that holds intermediate values.
            - left: The left differences.
            - right: The right differences.
    """
    # Allocate the table with zeros and set the initial condition
    ndu = [[0.0] * (degree + 1) for _ in range(degree + 1)]
    ndu[0][0] = 1.0

    left, right = compute_left_right_arrays(degree, knot, knot_vector, span)
    for j in range(1, degree + 1):
        compute_ndu_row(j, left, right, ndu)
    return ndu, left, right


def compute_inner_loop_bounds(r, k, degree):
    """
    Compute the inner summation bounds for the derivative accumulation.

    In the derivative algorithm the inner loop runs from j1 to j2.
    These bounds are computed as follows:

         rk = r - k
         pk = degree - k
         j1 = 1 if rk >= -1 else -rk
         j2 = k - 1 if (r - 1) <= pk else degree - r

    Args:
        r (int): The basis function index.
        k (int): The current derivative order (>= 1).
        degree (int): The degree p of the basis functions.

    Returns:
        tuple: (j1, j2) the lower and upper bounds for the inner loop index.
    """
    rk = r - k
    pk = degree - k
    j1 = 1 if rk >= -1 else -rk
    j2 = k - 1 if (r - 1) <= pk else degree - r
    return j1, j2


def compute_derivative_coefficients_for_r(r, ndu, degree, order):
    """
    Compute the derivative coefficients for one basis function (index r).

    The 0th derivative is simply the basis function value, and higher derivatives
    are computed by recursively “convolving” the lower–order contributions.

    Args:
        r (int): The basis function index (0 <= r <= degree).
        ndu (list of list of float): The table computed in compute_ndu.
        degree (int): The degree p of the basis functions.
        order (int): The maximum derivative order to compute.

    Returns:
        list: A list of length (order+1) where the k-th element is the k-th derivative
              coefficient for basis function r.
    """
    # Initialize the list of derivative coefficients; the 0th derivative is directly from ndu.
    d_coeffs = [0.0] * (order + 1)
    d_coeffs[0] = ndu[r][degree]

    # Temporary storage in two alternating rows.
    a = [[0.0] * (order + 1) for _ in range(2)]
    a[0][0] = 1.0
    s1 = 0  # current row index in a
    s2 = 1  # next row index in a

    # Loop over derivative orders k = 1, 2, ..., order
    for k in range(1, order + 1):
        d = 0.0
        rk = r - k
        pk = degree - k

        # First term in the recurrence, if available.
        if r >= k:
            a[s2][0] = a[s1][0] / ndu[pk + 1][rk]
            d += a[s2][0] * ndu[rk][pk]

        # Compute inner summation contributions.
        j1, j2 = compute_inner_loop_bounds(r, k, degree)
        for j in range(j1, j2 + 1):
            a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]
            d += a[s2][j] * ndu[rk + j][pk]

        # Final term in the recurrence, if available.
        if r <= pk:
            a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]
            d += a[s2][k] * ndu[r][pk]

        d_coeffs[k] = d
        # Swap the temporary rows for the next iteration.
        s1, s2 = s2, s1

    return d_coeffs


def compute_basis_function_derivatives_from_ndu(ndu, degree, order):
    """
    Build the full table of (unscaled) derivative coefficients from the ndu table.

    For each basis function (index r from 0 to degree) the derivative coefficients
    (orders 0 through order) are computed.

    Args:
        ndu (list of list of float): The table computed in compute_ndu.
        degree (int): The degree p of the basis functions.
        order (int): The maximum derivative order to compute.

    Returns:
        list of list of float: A 2D list 'ders' where ders[k][r] is the k-th derivative
                               for basis function index r.
    """
    ders = [[0.0] * (degree + 1) for _ in range(order + 1)]
    for r in range(degree + 1):
        d_coeffs = compute_derivative_coefficients_for_r(r, ndu, degree, order)
        for k in range(order + 1):
            ders[k][r] = d_coeffs[k]
    return ders


def apply_factorial_scaling(ders, degree, order):
    """
    Scale the derivative coefficients by the appropriate factorial-like factors.

    The scaling multiplies the k-th derivative by
         p * (p-1) * ... * (p - k + 1)
    so that the results agree with the standard mathematical definition.

    Args:
        ders (list of list of float): The (unscaled) derivatives.
        degree (int): The degree p of the basis functions.
        order (int): The maximum derivative order computed.

    Returns:
        list of list of float: The scaled derivatives.
    """
    factor = float(degree)
    for k in range(1, order + 1):
        for j in range(degree + 1):
            ders[k][j] *= factor
        factor *= degree - k
    return ders


def basis_function_ders(degree, knot_vector, span, knot, order):
    """
    Compute the derivatives of B-spline (or NURBS) basis functions.

    This high-level function organizes the work into several intuitive steps:

      1. Build the ndu table (and compute left/right differences).
      2. Extract the (unscaled) derivative coefficients via a convolution–like recurrence.
      3. Apply the necessary factorial scaling to obtain the true derivative values.

    Args:
        degree (int): The degree p of the basis functions.
        knot_vector (list or tuple): The knot vector U.
        span (int): The knot span index.
        knot (float): The parameter value u at which to evaluate.
        order (int): The maximum derivative order to compute.

    Returns:
        list of list of float: A 2D list 'ders' where ders[k][j] is the k-th derivative
                               of the j-th basis function.
    """
    ndu, left, right = compute_ndu(degree, knot, knot_vector, span)
    ders = compute_basis_function_derivatives_from_ndu(ndu, degree, order)
    ders = apply_factorial_scaling(ders, degree, order)
    return ders
'''
