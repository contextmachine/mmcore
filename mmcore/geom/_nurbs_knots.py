from __future__ import annotations

import functools
from collections import Counter
from copy import deepcopy

import numpy as np

from mmcore.geom._nurbs_eval import (
    nurbs_interval,
    _find_span_linear,
    _copy_curve,
    _copy_surface,
    to_homogeneous_1d,
    from_homogeneous_1d,
    NURBSCurveTuple,
    BSplineCurveTuple,
    NURBSSurfaceTuple,
    BSplineSurfaceTuple,
    _curve_interval,
    to_homogeneous_2d,
    from_homogeneous_2d,
)


from numpy.typing import NDArray

@functools.lru_cache(maxsize=None)
def generate_knots(control_points_count, degree, interval=None):
    """Generate default knot vector for NURBS/B-spline curves.

    Creates a uniform knot vector with proper multiplicities at the boundaries.
    The resulting knot vector has the form: [0,...,0, 1, 2,..., m, m,...,m]
    where 0 and m are repeated (degree+1) times.

    :param control_points_count: Number of control points
    :type control_points_count: int
    :param degree: Degree of the curve
    :type degree: int
    :param interval: Optional interval (start, end) to map knots to. If None, uses default [0, n-degree]
    :type interval: tuple[float, float] or None
    :return: Generated knot vector
    :rtype: numpy.ndarray
    """
    n = control_points_count

    knots = np.array([0] * (degree + 1) + list(range(1, n - degree)) + [n - degree] * (degree + 1),
                     dtype=float)
    if interval is not None:
        new_start,new_end=interval
        new_d=new_end-new_start
        start, end = nurbs_interval(knots, degree)
        d = end - start
        return  new_start+((np.asarray(knots) - start) / d)*new_d

    return knots

def normalize_knots(knots, degree):
    """Normalize knot vector to the interval [0, 1].

    :param knots: Knot vector to normalize
    :type knots: array-like
    :param degree: Degree of the NURBS/B-spline
    :type degree: int
    :return: Normalized knot vector in [0, 1]
    :rtype: numpy.ndarray
    """
    start, end = nurbs_interval(knots, degree)
    d = abs(end - start)
    return (np.asarray(knots) - start) / d
def normalize_knots_curve(curve: NURBSCurveTuple):
    """Normalize knot vector of a NURBS curve to [0, 1] interval.

    Creates a new curve with normalized knot vector while preserving the geometry.

    :param curve: NURBS curve to normalize
    :type curve: NURBSCurveTuple
    :return: New NURBS curve with normalized knot vector
    :rtype: NURBSCurveTuple
    """
    knots = normalize_knots(curve.knot, curve.order - 1)
    return curve._replace(knot=knots)
def normalize_knots_curve_inplace(curve: NURBSCurveTuple):
    """Normalize knot vector of a NURBS curve to [0, 1] interval in-place.

    Modifies the curve's knot vector directly while preserving the geometry.

    :param curve: NURBS curve to normalize in-place
    :type curve: NURBSCurveTuple
    """
    curve.knot[:] = normalize_knots(curve.knot, curve.order - 1)


def normalize_knots_surface_inplace(surf: NURBSSurfaceTuple):
    """Normalize knot vectors of a NURBS surface to [0, 1] interval in-place.
    Modifies both u and v knot vectors directly while preserving the geometry.


    :param surf: NURBS surface to normalize in-place
    :type surf: NURBSSurfaceTuple
    """
    surf.knot_u[:] = normalize_knots(surf.knot_u, surf.order_u - 1)
    surf.knot_v[:] = normalize_knots(surf.knot_v, surf.order_v - 1)

def knot_insertion_alpha(u, knotvector, span, idx, leg):
    """Compute the alpha coefficient for knot insertion.

    This function computes the blending coefficient used in knot insertion algorithms.
    Part of Algorithm A5.1 from The NURBS Book by Piegl & Tiller, 2nd Edition.

    :param u: Knot value to be inserted
    :type u: float
    :param knotvector: Knot vector
    :type knotvector: tuple or list
    :param span: Knot span index
    :type span: int
    :param idx: Control point index
    :type idx: int
    :param leg: Leg index for computation
    :type leg: int
    :return: Alpha blending coefficient
    :rtype: float
    """
    return (u - knotvector[leg + idx]) / (knotvector[idx + span + 1] - knotvector[leg + idx])


def find_multiplicity(knot, knot_vector, **kwargs):
    """ Finds knot multiplicity over the knot vector.

    Keyword Arguments:
        * ``spt``: tolerance (delta) value for equality checking

    :param knot: knot or parameter, :math:`u`
    :type knot: float
    :param knot_vector: knot vector, :math:`U`
    :type knot_vector: list, tuple
    :return: knot multiplicity, :math:`s`
    :rtype: int
    """
    # Get tolerance value
    tol = kwargs.get('spt', 10e-15)

    mult = 0  # initial multiplicity

    for kv in knot_vector:

        if abs(knot - kv) <= tol:
            mult += 1

    return mult


def knot_insertion_kv(knotvector, u, span, r):
    """ Computes the knot vector of the rational/non-rational spline after knot insertion.

    Part of Algorithm A5.1 of The NURBS Book by Piegl & Tiller, 2nd Edition.

    :param knotvector: knot vector
    :type knotvector: list, tuple
    :param u: knot
    :type u: float
    :param span: knot span
    :type span: int
    :param r: number of knot insertions
    :type r: int
    :return: updated knot vector
    :rtype: list
    """
    # Initialize variables
    kv_size = len(knotvector)
    kv_updated = [0.0 for _ in range(kv_size + r)]

    # Compute new knot vector
    for i in range(0, span + 1):
        kv_updated[i] = knotvector[i]
    for i in range(1, r + 1):
        kv_updated[span + i] = u
    for i in range(span + 1, kv_size):
        kv_updated[i + r] = knotvector[i]

    # Return the new knot vector
    return kv_updated


def knot_insertion(degree, knotvector, ctrlpts, u, num: int = 1, span=None, s=None, **kwargs):
    """Compute the control points after knot insertion.

    Part of Algorithm A5.1 of The NURBS Book by Piegl & Tiller, 2nd Edition.

    :param degree: Degree of the curve
    :type degree: int
    :param knotvector: Knot vector
    :type knotvector: list or tuple
    :param ctrlpts: Control points
    :type ctrlpts: list
    :param u: Knot value to be inserted
    :type u: float
    :param num: Number of knot insertions (default: 1)
    :type num: int
    :param span: Knot span (default: computed automatically)
    :type span: int or None
    :param s: Multiplicity of the knot (default: computed automatically)
    :type s: int or None
    :return: Updated control points after knot insertion
    :rtype: list
    """
    # Get keyword arguments

    s =  find_multiplicity(u, knotvector) if s is None else s  # multiplicity
    k = _find_span_linear(degree, knotvector, len(ctrlpts), u) if span is None else span  # knot span

    # Initialize variables
    np = len(ctrlpts)
    nq = np + num

    # Initialize new control points array (control points may be weighted or not)
    ctrlpts_new = [[] for _ in range(nq)]

    # Initialize a local array of length p + 1
    temp = [[] for _ in range(degree + 1)]

    # Save unaltered control points
    for i in range(0, k - degree + 1):
        ctrlpts_new[i] = ctrlpts[i]
    for i in range(k - s, np):
        ctrlpts_new[i + num] = ctrlpts[i]

    # Start filling the temporary local array which will be used to update control points during knot insertion
    for i in range(0, degree - s + 1):
        temp[i] = deepcopy(ctrlpts[k - degree + i])

    # Insert knot "num" times
    for j in range(1, num + 1):
        L = k - degree + j
        for i in range(0, degree - j - s + 1):

            alpha = knot_insertion_alpha(u, tuple(knotvector), k, i, L)
            if isinstance(temp[i][0], float):
                temp[i][:] = [alpha * elem2 + (1.0 - alpha) * elem1 for elem1, elem2 in zip(temp[i], temp[i + 1])]
            else:
                for idx in range(len(temp[i])):
                    temp[i][idx][:] = [alpha * elem2 + (1.0 - alpha) * elem1 for elem1, elem2 in
                                       zip(temp[i][idx], temp[i + 1][idx])]
        ctrlpts_new[L] = deepcopy(temp[0])
        ctrlpts_new[k + num - j - s] = deepcopy(temp[degree - j - s])

    # Load remaining control points
    L = k - degree + num
    for i in range(L + 1, k - s):
        ctrlpts_new[i] = deepcopy(temp[i - L])

    # Return control points after knot insertion
    return ctrlpts_new


def knot_refinement(degree, knotvector, ctrlpts, density: int = 1, knot_list=None, add_knot_list=None, tol=1e-12, **kwargs):
    """Compute knot vector and control points after knot refinement.

    Implementation of Algorithm A5.4 of The NURBS Book by Piegl & Tiller, 2nd Edition.

    The algorithm automatically finds the knots to be refined and their multiplicities.
    This can be overridden by providing a list of knots via ``knot_list`` argument.
    Additional knots can be provided via ``add_knot_list`` argument.

    The ``density`` parameter automates extra knot insertions by finding middle knots
    in each internal knot span.

    **Example**: For degree 2 and knot vector ``[0, 2, 4]``:

    * ``density = 1``: ``[0, 1, 1, 2, 2, 3, 3, 4]``
    * ``density = 2``: ``[0, 0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4]``

    :param degree: Degree of the curve
    :type degree: int
    :param knotvector: Knot vector
    :type knotvector: list or tuple
    :param ctrlpts: Control points
    :type ctrlpts: list
    :param density: Knot density multiplier (default: 1)
    :type density: int
    :param knot_list: Specific knots to refine (default: internal knots)
    :type knot_list: list or None
    :param add_knot_list: Additional knots to refine (default: [])
    :type add_knot_list: list or None
    :param tol: Tolerance for numerical comparisons (default: 1e-12)
    :type tol: float
    :return: Updated control points and knot vector
    :rtype: tuple[list, list]
    """
    # Get keyword arguments

    knot_list =knotvector[degree:-degree] if knot_list is None else knot_list
    add_knot_list = list() if add_knot_list is None else add_knot_list





    # Add additional knots to be refined
    if add_knot_list:
        knot_list += list(add_knot_list)

    # Sort the list and convert to a set to make sure that the values are unique
    knot_list = sorted(set(knot_list))

    # Increase knot density
    for d in range(0, density):
        rknots = []
        for i in range(len(knot_list) - 1):
            knot_tmp = knot_list[i] + ((knot_list[i + 1] - knot_list[i]) / 2.0)
            rknots.append(knot_list[i])
            rknots.append(knot_tmp)
        rknots.append(knot_list[i + 1])
        knot_list = rknots

    # Find how many knot insertions are necessary
    X = []
    for mk in knot_list:
        s = find_multiplicity(mk, knotvector)
        r = degree - s
        X += [mk for _ in range(r)]

    # Check if the knot refinement is possible
    if not X:

        return list(ctrlpts),list(knotvector)

    # Initialize common variables
    r = len(X) - 1
    n = len(ctrlpts) - 1
    m = n + degree + 1
    a = _find_span_linear(degree, knotvector, n+1, X[0])
    b = _find_span_linear(degree, knotvector, n+1, X[r]) + 1

    # Initialize new control points array
    if isinstance(ctrlpts[0][0], float):
        new_ctrlpts = [[] for _ in range(n + r + 2)]
    else:
        new_ctrlpts = [[[] for _ in range(len(ctrlpts[0]))] for _ in range(n + r + 2)]

    # Fill unchanged control points
    for j in range(0, a - degree + 1):
        new_ctrlpts[j] = ctrlpts[j]
    for j in range(b - 1, n + 1):
        new_ctrlpts[j + r + 1] = ctrlpts[j]

    # Initialize new knot vector array
    new_kv = [0.0 for _ in range(m + r + 2)]

    # Fill unchanged knots
    for j in range(0, a + 1):
        new_kv[j] = knotvector[j]
    for j in range(b + degree, m + 1):
        new_kv[j + r + 1] = knotvector[j]

    # Initialize variables for knot refinement
    i = b + degree - 1
    k = b + degree + r
    j = r

    # Apply knot refinement
    while j >= 0:
        while X[j] <= knotvector[i] and i > a:
            new_ctrlpts[k - degree - 1] = ctrlpts[i - degree - 1]
            new_kv[k] = knotvector[i]
            k -= 1
            i -= 1
        new_ctrlpts[k - degree - 1] = deepcopy(new_ctrlpts[k - degree])
        for l in range(1, degree + 1):
            idx = k - degree + l
            alpha = new_kv[k + l] - X[j]
            if abs(alpha) < tol:
                new_ctrlpts[idx - 1] = deepcopy(new_ctrlpts[idx])
            else:
                alpha = alpha / (new_kv[k + l] - knotvector[i - degree + l])
                if isinstance(ctrlpts[0][0], float):
                    new_ctrlpts[idx - 1] = [alpha * p1 + (1.0 - alpha) * p2 for p1, p2 in
                                            zip(new_ctrlpts[idx - 1], new_ctrlpts[idx])]
                else:
                    for idx2 in range(len(ctrlpts[0])):
                        new_ctrlpts[idx - 1][idx2] = [alpha * p1 + (1.0 - alpha) * p2 for p1, p2 in
                                                      zip(new_ctrlpts[idx - 1][idx2], new_ctrlpts[idx][idx2])]
        new_kv[k] = X[j]
        k = k - 1
        j -= 1

    # Return control points and knot vector after refinement
    return new_ctrlpts, new_kv

def knot_removal_alpha_i(u, degree, knotvector, num, idx):
    """Compute the alpha_i coefficient for knot removal.

    This function computes the blending coefficient used in knot removal algorithms.
    Part of Algorithm A5.8 from The NURBS Book by Piegl & Tiller, 2nd Edition.

    :param u: Knot value to be removed
    :type u: float
    :param degree: Degree of the curve
    :type degree: int
    :param knotvector: Knot vector
    :type knotvector: list or tuple
    :param num: Number of knot removals
    :type num: int
    :param idx: Index in the knot vector
    :type idx: int
    :return: Alpha_i blending coefficient
    :rtype: float
    """
    return (u - knotvector[idx]) / (knotvector[idx + degree + 1 + num] - knotvector[idx])


def knot_removal_alpha_j(u, degree, knotvector, num, idx):
    """Compute the alpha_j coefficient for knot removal.

    This function computes the blending coefficient used in knot removal algorithms.
    Part of Algorithm A5.8 from The NURBS Book by Piegl & Tiller, 2nd Edition.

    :param u: Knot value to be removed
    :type u: float
    :param degree: Degree of the curve
    :type degree: int
    :param knotvector: Knot vector
    :type knotvector: list or tuple
    :param num: Number of knot removals
    :type num: int
    :param idx: Index in the knot vector
    :type idx: int
    :return: Alpha_j blending coefficient
    :rtype: float
    """
    return (u - knotvector[idx - num]) / (knotvector[idx + degree + 1] - knotvector[idx - num])


def knot_removal_kv(knotvector, span, r):
    """ Computes the knot vector of the rational/non-rational spline after knot removal.

    Part of Algorithm A5.8 of The NURBS Book by Piegl & Tiller, 2nd Edition.

    :param knotvector: knot vector
    :type knotvector: list, tuple
    :param span: knot span
    :type span: int
    :param r: number of knot removals
    :type r: int
    :return: updated knot vector
    :rtype: list
    """
    # Edge case
    if r < 1:
        return knotvector

    # Create a deep copy of the input knot  vector
    kv_updated = np.copy(knotvector)

    # Shift knots
    for k in range(span + 1, len(knotvector)):
        kv_updated[k - r] = knotvector[k]

    # Slice to get the new knot vector
    kv_updated = kv_updated.tolist()[0:-r]

    # Return the new knot vector
    return kv_updated

def knot_removal(
        degree, knotvector, ctrlpts, u, num=1,
        tol=1e-12, s=None, span=None):
    """
    Remove `num` copies of knot value *u* (at most `degree`) from a
    B‑spline/NURBS curve.  Returns (new_knots, new_ctrlpts).

    Implementation follows Algorithm A5.8 (Piegl & Tiller, 2nd ed.).
    """

    # ---- set‑up ----------------------------------------------------------
    p = degree
    U = list(knotvector)                  # work on mutable copies
    P = [np.array(Pi, dtype=float) for Pi in ctrlpts]

    # multiplicity and span
    s = s if s is not None else U.count(u)
    if s == 0:
        raise ValueError("knot value is not present in the vector")
    r = span if span is not None else next(i for i in range(len(U)-1) if
                                           U[i] <= u < U[i+1] or
                                           (u == U[-1] and i == len(U)-2))

    num = min(num, s)                     # cannot remove more than s copies

    # ---- main loop -------------------------------------------------------
    for t in range(1, num+1):
        first = r - p
        last  = r - s
        # temp array
        temp = [None] * (2*p+1)

        temp[0]               = P[first-1].copy()
        temp[last-first+2]    = P[last+1].copy()
        i, j = first, last
        ii, jj = 1, last-first+1
        removable = False

        while j - i >= t:
            alpha_i = (u - U[i])   / (U[i+p+1-t]   - U[i])
            alpha_j = (u - U[j-t]) / (U[j+p+1] - U[j-t])

            temp[ii] = (P[i]   - (1-alpha_i)*temp[ii-1]) / alpha_i
            temp[jj] = (P[j]   -  alpha_j   *temp[jj+1]) / (1-alpha_j)

            i  += 1;  j  -= 1
            ii += 1;  jj -= 1

        # ‑‑ error test ---------------------------------------------------
        if j - i < t:                      # Case 1 (Eq. 5.30)
            removable = np.linalg.norm(temp[ii-1] - temp[jj+1]) <= tol
        else:                              # Case 2 (Eq. 5.31)
            alpha = (u - U[i]) / (U[i+p+1-t] - U[i])
            testp = alpha*temp[ii+t+1] + (1-alpha)*temp[ii-1]
            removable = np.linalg.norm(P[i] - testp) <= tol

        if not removable:
            break                          # cannot remove further

        # ‑‑ update polygon ----------------------------------------------
        i, j = first, last
        while j - i > t:
            P[i] = temp[i-first+1]
            P[j] = temp[j-first+1]
            i += 1; j -= 1

        # ‑‑ delete one knot ---------------------------------------------
        del U[r]                # remove ONE copy of u
        del P[last]             # remove the matching control point
        r -= 1; s -= 1          # array shrank by one

    return np.asarray(U), np.asarray(P)

def insert_knot_curve(curve:NURBSCurveTuple,u:float, num:int=1):
    """Insert a knot into a curve multiple times.
    
    Args:
        curve: The curve to modify
        u: The knot value to insert
        num: Number of times to insert the knot (default: 1)
        
    Returns:
        A new curve with the knot inserted
    """
    rational = isinstance(curve, NURBSCurveTuple) and not np.allclose(curve.weights,1)
    knots = np.array(curve.knot).tolist()
    degree = curve.order-1
    span = _find_span_linear(degree, knots, len(curve.control_points), u)

    kv_new = knot_insertion_kv(knots, u, span, num)
    
    if rational:
        # For rational curves, convert to homogeneous coordinates first
        control_points_h = to_homogeneous_1d(curve.control_points, curve.weights)
        control_points = control_points_h.tolist()
    else:
        control_points = curve.control_points.tolist()
        
    new_control_points = knot_insertion(curve.order-1, np.array(curve.knot).tolist(), control_points, u, num=num)
    
    if rational:

        # Convert back from homogeneous coordinates
        new_control_points_xyz, weights = from_homogeneous_1d(np.array(new_control_points))
        return NURBSCurveTuple(curve.order, knot=np.array(kv_new), 
                              control_points=new_control_points_xyz, 
                              weights=weights)
    #print(new_control_points)
    return NURBSCurveTuple(curve.order, knot=np.array(kv_new), control_points=np.array(new_control_points),weights=np.ones(len(new_control_points)))


def split_curve(curve: BSplineCurveTuple | NURBSCurveTuple, t: float, **kwargs):
    """Split the curve at the given parametric coordinate.

    This method splits the curve into two pieces at the given parametric coordinate,
    generates two different curve objects and returns them. It does not modify the input curve.

    :param curve: Curve to be split
    :type curve: BSplineCurveTuple or NURBSCurveTuple
    :param t: Parameter value where to split the curve
    :type t: float
    :return: Tuple of two curve segments (left, right)
    :rtype: tuple[BSplineCurveTuple | NURBSCurveTuple, BSplineCurveTuple | NURBSCurveTuple]
    """
    # Validate input

    interval = nurbs_interval(curve.knot, curve.order-1)
    if t == interval[0] or t == interval[1]:
        raise ValueError(f"Parameter t: {t} Cannot split from the domain edge: {interval}")
    if not (interval[0]<t< interval[1]):
        raise ValueError(f"Parameter t: {t} is outside the domain: {interval}")

    # Find multiplicity of the knot and define how many times we need to add the knot
    degree=curve.order-1
    ks = _find_span_linear(degree, curve.knot, len(curve.control_points), t) - degree+ 1
    s = find_multiplicity(t,  curve.knot)
    r = degree - s

    # Create backups of the original curve

    temp_obj = _copy_curve(curve)

    # Insert knot

    temp_obj=insert_knot_curve(temp_obj, t, num=r)


    # Knot vectors
    knot_span = _find_span_linear(temp_obj.order-1, temp_obj.knot, len(temp_obj.control_points), t) + 1
    temp_knot=np.array(temp_obj.knot).tolist()
    curve1_kv = list(temp_knot[0:knot_span])
    curve1_kv.append(t)
    curve2_kv = list(temp_knot[knot_span:])

    for _ in range(0, temp_obj.order ):
        curve2_kv.insert(0, t)

    # Control points (use homogeneous coordinates if rational)
    rational=isinstance(curve, NURBSCurveTuple) and not np.allclose(curve.weights,1)

    if rational:
        # Convert to homogeneous coordinates first
        cpts_homo = to_homogeneous_1d(temp_obj.control_points, temp_obj.weights).tolist()
        curve1_ctrlpts_homo = cpts_homo[0:ks + r]
        curve2_ctrlpts_homo = cpts_homo[ks + r - 1:]
        
        # Convert back from homogeneous for each curve
        curve1_ctrlpts, curve1_weights = from_homogeneous_1d(curve1_ctrlpts_homo)
        curve2_ctrlpts, curve2_weights = from_homogeneous_1d(curve2_ctrlpts_homo)
        
        curve1 = NURBSCurveTuple(temp_obj.order, curve1_kv, curve1_ctrlpts, curve1_weights)
        curve2 = NURBSCurveTuple(temp_obj.order, curve2_kv, curve2_ctrlpts, curve2_weights)
    else:
        cpts = temp_obj.control_points.tolist()
        curve1_ctrlpts =  np.asarray(cpts[0:ks + r])
        curve2_ctrlpts =  np.asarray(cpts[ks + r - 1:])
        curve1 = NURBSCurveTuple(temp_obj.order, curve1_kv, curve1_ctrlpts, np.ones(len(curve1_ctrlpts)))
        curve2 = NURBSCurveTuple(temp_obj.order, curve2_kv, curve2_ctrlpts, np.ones(len(curve2_ctrlpts)))
    return curve1,curve2

def split_curve_multiple(crv:NURBSCurveTuple, params:list[float]|NDArray[float])->list[NURBSCurveTuple]:
    crvs = []
    #temp = _copy_curve(crv)

    for i in range(len(params)):
        tpl = split_curve(crv, params[i])
        crv = tpl[1]
        crvs.append(tpl[0])
    crvs.append(crv)
    return crvs

def decompose_curve(crv:NURBSCurveTuple)->list[NURBSCurveTuple]:
    params=np.unique(crv.knot)
    params=params[1:][:params.shape[0]-2]

    return split_curve_multiple(crv,params)


def trim_curve(curve:NURBSCurveTuple, t0:float,t1:float):
    print(t0,t1)
    t0,t1=min(t0,t1),max(t0,t1)
    t_min,t_max=_curve_interval(curve)
    if t0==t_min and t1==t_max:
        return curve.__class__(*curve)

    elif t0==t_min:

        return split_curve(curve,t1)[0]
    elif t1==t_max:
        return split_curve(curve, t0)[1]
    else:
        return split_curve(split_curve(curve, t0)[1],t1)[0]


def insert_knot_surface_u(self: BSplineSurfaceTuple | NURBSSurfaceTuple, t, num=1):
    cpts = np.copy(self.control_points)
    count=num
    cpts_size_u, cpts_size_v, dim = cpts.shape
    new_count_u = cpts_size_u+ count
    new_count_v = cpts_size_v
    degree_u = self.order_u - 1

    span = _find_span_linear(
        degree_u, self.knot_u, cpts_size_u, t
    )

    # Compute new knot vector
    k_v = knot_insertion_kv(self.knot_u, t, span, count)
    s_u = find_multiplicity(t, self.knot_u)

    if isinstance(self, BSplineSurfaceTuple):
        self = NURBSSurfaceTuple(*self, weights=np.ones(self.control_points.shape[:-1], dtype=float))
    new_pts = np.zeros((new_count_u, new_count_v, dim))
    new_weights = np.zeros((new_count_u, new_count_v))
    knot_u_list=np.array(self.knot_u).tolist()
    for v in range(cpts_size_v):
        row_control_points = cpts[:, v, :]
        row_weights = self.weights[:, v]
        # Convert to homogeneous coordinates
        row_homo = to_homogeneous_1d(row_control_points, row_weights)
        row_homo_list = row_homo.tolist()

        # Apply knot insertion

        new_row_homo_list = knot_insertion(degree_u,
                                           knot_u_list,
                                           row_homo_list,
                                           t,
                                           num=count,

                                           span=span, s=s_u)
        # Convert back from homogeneous
        new_row_cp, new_row_w = from_homogeneous_1d(new_row_homo_list)
        new_weights[:, v] = new_row_w
        new_pts[:, v, :] = new_row_cp

    return self._replace(knot_u=k_v, control_points=new_pts, weights=new_weights)


def insert_knot_surface_v(self:BSplineSurfaceTuple|NURBSSurfaceTuple,t, num=1):


        count=num

        cpts=np.copy(self.control_points)

        cpts_size_u  ,      cpts_size_v, dim=cpts.shape
        new_count_u = cpts_size_u
        new_count_v = cpts_size_v + count
        degree_v=self.order_v-1

        span = _find_span_linear(
            degree_v, self.knot_v, cpts_size_v, t
        )

        # Compute new knot vector
        k_v = knot_insertion_kv(self.knot_v, t, span, count)
        s_v = find_multiplicity(t, self.knot_v)

        if isinstance(self,BSplineSurfaceTuple):
            self=NURBSSurfaceTuple(*self, weights=np.ones(self.control_points.shape[:-1],dtype=float))
        new_pts=np.zeros((new_count_u,new_count_v,dim))
        new_weights=np.zeros((new_count_u,new_count_v))
        knot_v_list=np.asarray(self.knot_v).tolist()
        for u in range(cpts_size_u):
            col_control_points=cpts[u, :, :]
            col_weights=self.weights[u,:]
            # Convert to homogeneous coordinates
            col_homo = to_homogeneous_1d(col_control_points, col_weights)
            col_homo_list = col_homo.tolist()

            # Apply knot insertion

            new_col_homo_list=knot_insertion(degree_v,
                           knot_v_list,
                           col_homo_list,
                           t,
                           num=count,

                           span=span, s=s_v)
            # Convert back from homogeneous
            new_col_cp, new_col_w = from_homogeneous_1d(new_col_homo_list)
            new_weights[u,:]=new_col_w
            new_pts[u, :, :] = new_col_cp




        return self._replace(knot_v=k_v, control_points=new_pts,weights=new_weights)


def split_surface_u(surface:BSplineSurfaceTuple|NURBSSurfaceTuple, u:float, **kwargs):
    """Splits the surface at the given parametric coordinate in the u-direction.
    
    This method splits the surface into two pieces at the given u parameter, generating 
    two different surface objects and returning them. It does not modify the input surface.
    
    Args:
        surface: Surface to be split
        u: Parameter value in the u-direction
        
    Returns:
        Tuple of (left_surface, right_surface)
    """
    # Validate input
    interval_u = nurbs_interval(surface.knot_u, surface.order_u-1)
    if u == interval_u[0] or u == interval_u[1]:
        raise ValueError(f"Parameter u: {u} Cannot split from the domain edge: {interval_u}")
    if not (interval_u[0] < u < interval_u[1]):
        raise ValueError(f"Parameter u: {u} is outside the domain: {interval_u}")
    
    # Find multiplicity of the knot and define how many times we need to add the knot
    degree_u = surface.order_u - 1
    ks_u = _find_span_linear(degree_u, surface.knot_u, surface.control_points.shape[0], u) - degree_u + 1
    s_u = find_multiplicity(u, surface.knot_u)
    r_u = degree_u - s_u
    
    # Create backup of the original surface
    temp_surface = surface
    
    # Insert knot
    if r_u > 0:
        temp_surface = insert_knot_surface_u(temp_surface, u, num=r_u)
    
    # Knot vectors
    knot_span_u = _find_span_linear(temp_surface.order_u-1, temp_surface.knot_u, temp_surface.control_points.shape[0], u) + 1
    temp_knot_u = np.array(temp_surface.knot_u).tolist()
    
    # Left and right knot vectors
    left_kv_u = list(temp_knot_u[0:knot_span_u])
    left_kv_u.append(u)
    right_kv_u = list(temp_knot_u[knot_span_u:])
    
    for _ in range(0, temp_surface.order_u):
        right_kv_u.insert(0, u)
    
    # Control points for left and right surfaces
    rational = isinstance(surface, NURBSSurfaceTuple)
    
    if rational:
        # Create left and right surfaces with proper control points
        left_cps = temp_surface.control_points[0:ks_u + r_u, :, :]
        right_cps = temp_surface.control_points[ks_u + r_u - 1:, :, :]
        
        left_weights = temp_surface.weights[0:ks_u + r_u, :]
        right_weights = temp_surface.weights[ks_u + r_u - 1:, :]
        
        left_surface = NURBSSurfaceTuple(
            order_u=temp_surface.order_u,
            order_v=temp_surface.order_v,
            knot_u=np.array(left_kv_u),
            knot_v=temp_surface.knot_v,
            control_points=left_cps,
            weights=left_weights
        )
        
        right_surface = NURBSSurfaceTuple(
            order_u=temp_surface.order_u,
            order_v=temp_surface.order_v,
            knot_u=np.array(right_kv_u),
            knot_v=temp_surface.knot_v,
            control_points=right_cps,
            weights=right_weights
        )
    else:
        left_cps = temp_surface.control_points[0:ks_u + r_u, :, :]
        right_cps = temp_surface.control_points[ks_u + r_u - 1:, :, :]
        
        left_surface = BSplineSurfaceTuple(
            order_u=temp_surface.order_u,
            order_v=temp_surface.order_v,
            knot_u=np.array(left_kv_u),
            knot_v=temp_surface.knot_v,
            control_points=left_cps
        )
        
        right_surface = BSplineSurfaceTuple(
            order_u=temp_surface.order_u,
            order_v=temp_surface.order_v,
            knot_u=np.array(right_kv_u),
            knot_v=temp_surface.knot_v,
            control_points=right_cps
        )
    
    return left_surface, right_surface

def split_surface_u_multiple(surf: BSplineSurfaceTuple | NURBSSurfaceTuple, params: list[float] | NDArray[float]) -> list[BSplineCurveTuple] | list[NURBSCurveTuple]:
    crvs = []


    for i in range(len(params)):
        tpl = split_surface_u(surf, params[i])
        surf = tpl[1]
        crvs.append(tpl[0])
    crvs.append(surf)
    return crvs


def split_surface_v(surface:BSplineSurfaceTuple|NURBSSurfaceTuple, v:float, **kwargs):
    """Splits the surface at the given parametric coordinate in the v-direction.
    
    This method splits the surface into two pieces at the given v parameter, generating 
    two different surface objects and returning them. It does not modify the input surface.
    
    Args:
        surface: Surface to be split
        v: Parameter value in the v-direction
        
    Returns:
        Tuple of (bottom_surface, top_surface)
    """
    # Validate input
    interval_v = nurbs_interval(surface.knot_v, surface.order_v-1)
    if v == interval_v[0] or v == interval_v[1]:
        raise ValueError(f"Parameter v: {v} Cannot split from the domain edge: {interval_v}")
    if not (interval_v[0] < v < interval_v[1]):
        raise ValueError(f"Parameter v: {v} is outside the domain: {interval_v}")
    
    # Find multiplicity of the knot and define how many times we need to add the knot
    degree_v = surface.order_v - 1
    ks_v = _find_span_linear(degree_v, surface.knot_v, surface.control_points.shape[1], v) - degree_v + 1
    s_v = find_multiplicity(v, surface.knot_v)
    r_v = degree_v - s_v
    
    # Create backup of the original surface
    temp_surface = surface
    
    # Insert knot
    if r_v > 0:
        temp_surface = insert_knot_surface_v(temp_surface, v, num=r_v)
    
    # Knot vectors
    knot_span_v = _find_span_linear(temp_surface.order_v-1, temp_surface.knot_v, temp_surface.control_points.shape[1], v) + 1
    temp_knot_v = np.array(temp_surface.knot_v).tolist()
    
    # Bottom and top knot vectors
    bottom_kv_v = list(temp_knot_v[0:knot_span_v])
    bottom_kv_v.append(v)
    top_kv_v = list(temp_knot_v[knot_span_v:])
    
    for _ in range(0, temp_surface.order_v):
        top_kv_v.insert(0, v)
    
    # Control points for bottom and top surfaces
    rational = isinstance(surface, NURBSSurfaceTuple)
    
    if rational:
        # Create bottom and top surfaces with proper control points
        bottom_cps = temp_surface.control_points[:, 0:ks_v + r_v, :]
        top_cps = temp_surface.control_points[:, ks_v + r_v - 1:, :]
        
        bottom_weights = temp_surface.weights[:, 0:ks_v + r_v]
        top_weights = temp_surface.weights[:, ks_v + r_v - 1:]
        
        bottom_surface = NURBSSurfaceTuple(
            order_u=temp_surface.order_u,
            order_v=temp_surface.order_v,
            knot_u=temp_surface.knot_u,
            knot_v=np.array(bottom_kv_v),
            control_points=bottom_cps,
            weights=bottom_weights
        )
        
        top_surface = NURBSSurfaceTuple(
            order_u=temp_surface.order_u,
            order_v=temp_surface.order_v,
            knot_u=temp_surface.knot_u,
            knot_v=np.array(top_kv_v),
            control_points=top_cps,
            weights=top_weights
        )
    else:
        bottom_cps = temp_surface.control_points[:, 0:ks_v + r_v, :]
        top_cps = temp_surface.control_points[:, ks_v + r_v - 1:, :]
        
        bottom_surface = BSplineSurfaceTuple(
            order_u=temp_surface.order_u,
            order_v=temp_surface.order_v,
            knot_u=temp_surface.knot_u,
            knot_v=np.array(bottom_kv_v),
            control_points=bottom_cps
        )
        
        top_surface = BSplineSurfaceTuple(
            order_u=temp_surface.order_u,
            order_v=temp_surface.order_v,
            knot_u=temp_surface.knot_u,
            knot_v=np.array(top_kv_v),
            control_points=top_cps
        )
    
    return bottom_surface, top_surface
def split_surface_v_multiple(surf: BSplineSurfaceTuple|NURBSSurfaceTuple, params: list[float] | NDArray[float]) -> list[BSplineCurveTuple] | list[NURBSCurveTuple]:
    crvs = []


    for i in range(len(params)):
        tpl = split_surface_v(surf, params[i])
        surf = tpl[1]
        crvs.append(tpl[0])
    crvs.append(surf)
    return crvs


def subdivide_surface(surface:BSplineSurfaceTuple|NURBSSurfaceTuple, u:float, v:float):
    """Subdivides a surface into four parts at the given (u,v) parameter values.
    
    Args:
        surface: The surface to subdivide
        u: Parameter value in the u-direction
        v: Parameter value in the v-direction
        
    Returns:
        Tuple of (bottom_left, bottom_right, top_left, top_right) surfaces
    """
    # First, split the surface in the u-direction
    left_surface, right_surface = split_surface_u(surface, u)
    
    # Then split each half in the v-direction
    bottom_left, top_left = split_surface_v(left_surface, v)
    bottom_right, top_right = split_surface_v(right_surface, v)
    
    return bottom_left, bottom_right, top_left, top_right

def decompose_surface(surface:NURBSSurfaceTuple, decompose_dir="uv"):
    degrees=surface.order_u-1,surface.order_v-1
    def decompose_direction(srf:NURBSSurfaceTuple, idx):
        nonlocal degrees
        srf_list = []
        knots = srf.knot_u if idx == 0 else srf.knot_v
        degree = degrees[idx]
        unique_knots = sorted(set(knots[degree + 1 : -(degree + 1)]))

        while unique_knots:
            knot = unique_knots[0]
            if idx == 0:
                srfs = split_surface_u(srf, knot)
            else:
                srfs = split_surface_v(srf, knot)
            srf_list.append(srfs[0])
            srf = srfs[1]
            unique_knots = unique_knots[1:]

        srf_list.append(srf)
        return srf_list

    surf = _copy_surface(surface)

    if decompose_dir == "u":
        surfs_u=decompose_direction(surf, 0)

        return surfs_u
    elif decompose_dir == "v":
        surfs_v=decompose_direction(surf, 1)

        return surfs_v


    elif decompose_dir == "uv":
        multi_surf = []
        surfs_u = decompose_direction(surf, 0)


        for sfu in surfs_u:
            dsf=decompose_direction(sfu, 1)

            multi_surf+=dsf
        return multi_surf
    else:
        raise ValueError(
            f"Cannot decompose in {decompose_dir} direction. Acceptable values: u, v, uv"
        )


def trim_surface(surface:BSplineSurfaceTuple|NURBSSurfaceTuple, u0:float, u1:float, v0:float, v1:float)->BSplineSurfaceTuple|NURBSSurfaceTuple:
    (ustart,uend)    , (vstart,vend)  =surface.interval()
    u0,u1=min(u0,u1),max(u0,u1)
    v0,v1=min(v0,v1),max(v0,v1)
    splits_u,splits_v=[u0, u1],[v0,v1]
    ix_u=1
    ix_v=1
    if u0==ustart:
        del splits_u[0]
        ix_u-=1
    if u1==uend:
        del splits_u[1]
       
    if v0==vstart:
        del splits_v[0]
        ix_v -= 1
    if v1==vend:
        del splits_v[1]
        
    
    
    surf= split_surface_u_multiple(surface, splits_u)[ix_u]
    
    return split_surface_v_multiple(surf, splits_v)[ix_v]
    

from mmcore.numeric.binom import binomial_coefficient_py

def degree_elevation(degree, ctrlpts, num=1, **kwargs):
    """Compute control points after degree elevation for Bezier shapes.

    Implementation of Eq. 5.36 of The NURBS Book by Piegl & Tiller, 2nd Edition, p.205.

    Note: This algorithm only operates on Bezier shapes (curves, surfaces, volumes).

    :param degree: Current degree of the shape
    :type degree: int
    :param ctrlpts: Control points of the Bezier shape
    :type ctrlpts: list or tuple
    :param num: Number of degree elevations (default: 1)
    :type num: int
    :return: Control points of the degree-elevated shape
    :rtype: list
    """
    # Get keyword arguments



    # Initialize variables
    num_pts_elev = degree + 1 + num
    pts_elev = [[0.0 for _ in range(len(ctrlpts[0]))] for _ in range(num_pts_elev)]

    # Compute control points of degree-elevated 1-dimensional shape
    for i in range(0, num_pts_elev):
        start = max(0, (i - num))
        end = min(degree, i)
        for j in range(start, end + 1):
            coeff = binomial_coefficient_py(degree, j) * binomial_coefficient_py(num, (i - j))
            coeff /= binomial_coefficient_py((degree + num), i)
            pts_elev[i] = [p1 + (coeff * p2) for p1, p2 in zip(pts_elev[i], ctrlpts[j])]

    # Return computed control points after degree elevation
    return pts_elev

def _bezier_knots(order:int, interval:tuple[float,float]):
    start,end=interval
    return [start]*order+[end]*order

def link_curves(curves):
    """
    Concatenate a list of *cubic* NURBS curves that meet G0‑continuously.
    All curves must have the same order ``p+1`` and share the end/start point.
    Returns: (new_curve, interior_knots)
    """

    if not curves:
        raise ValueError("Empty input list")

    order = curves[0].order                     # all pieces have the same order
    p      = order - 1

    kv, cpts, wgts = [], [], []
    interior_knots = []                        # the knot to which each join collapses

    # running offset of the global parameter domain
    offset = 0.0

    for i, crv in enumerate(curves):
        k   = np.asarray(crv.knot,   dtype=float)
        cp  = np.asarray(crv.control_points, dtype=float)
        w   = np.asarray(crv.weights, dtype=float)

        # Shift this piece so that its *first* knot equals the current offset
        d   = offset - k[0]                    # <─── Δ computed once
        k   = k + d


        if i == 0:
            # keep everything *except* the trailing clamping knots
            kv.extend(k[:-order])
            cpts.extend(cp)
            wgts.extend(w)
        else:
            # skip the duplicate first knot and first control point
            kv.extend(k[1:-order])
            cpts.extend(cp[1:])
            wgts.extend(w[1:])

        # new offset = (last interior knot of this piece)
        offset = k[-order]                     # first of the trailing (p+1) equal knots
        interior_knots.append(offset)

    # add clamping knots at the very end
    kv.extend([offset] * order)
    interior_knots.pop()                       # last one is the global end knot

    return (
        NURBSCurveTuple(
            order=order,
            knot=np.asarray(kv),
            control_points=np.asarray(cpts),
            weights=np.asarray(wgts),
        ),
        interior_knots,                        # you will see only 0.3 here
    )

from typing import List, NamedTuple, Optional
import numpy as np


def stitch_surface_grid(grid: list[list[NURBSSurfaceTuple]]
                        ) -> tuple[NURBSSurfaceTuple,
                                   np.ndarray, np.ndarray]:
    """
    Merge a rectangular grid of *compatible* NURBS patches into a single
    NURBS surface.

    Parameters
    ----------
    grid : List[List[NURBSSurfaceTuple]]
        grid[r][c] is the patch located at  (u‑index=c , v‑index=r).

    Returns
    -------
    (big_surface , interior_u , interior_v)

    * `big_surface`        the merged NURBS surface (same orders p,q)
    * `interior_u`         knot values that separate columns (length C‑1)
    * `interior_v`         knot values that separate rows    (length R‑1)

    Preconditions
    -------------
    * Every patch has the same `(order_u, order_v)`.
    * Adjacent patches share *exactly* the same boundary curve
      (i.e. their knot vectors are *clamped* and identical on the common
      edge, up to a uniform offset applied by the stitching code).
    * The grid is topologically rectangular: all rows have the same length.

    The routine **never re‑evaluates geometry**   it concatenates control
    meshes and pastes knot vectors, shifting and de‑duplicating knot values
    only where mathematically required.
    """
    # ------------------------------------------------------------------
    # 0.  Basic shape checks
    # ------------------------------------------------------------------
    if not grid or not grid[0]:
        raise ValueError("Empty grid")

    R, C = len(grid), len(grid[0])
    if any(len(row) != C for row in grid):
        raise ValueError("The grid must be rectangular")

    order_u = grid[0][0].order_u
    order_v = grid[0][0].order_v
    p, q    = order_u - 1, order_v - 1

    # ------------------------------------------------------------------
    # 1.  Stitch every ROW horizontally  (u‑direction)
    # ------------------------------------------------------------------
    stitched_rows     = []      # list of NURBSSurfaceTuple
    interior_knots_u  = None    # will be fixed by the first row

    for r, row in enumerate(grid):
        row_surface, row_ku_split = _stitch_row(row, order_u, order_v)

        if interior_knots_u is None:
            interior_knots_u = row_ku_split
            #print(interior_knots_u, row_ku_split,)
        elif not np.allclose(interior_knots_u, row_ku_split, atol=1e-9):
            raise ValueError(f"Row {r} has interior‑u knots "
                             "incompatible with previous rows")

        stitched_rows.append(row_surface)

    # ------------------------------------------------------------------
    # 2.  Stitch those ROWS vertically  (v‑direction)
    # ------------------------------------------------------------------
    merged_surface, interior_knots_v = _stitch_column(stitched_rows,
                                                      order_u, order_v)

    return merged_surface, interior_knots_u, interior_knots_v


# ======================================================================
#  Helpers
# ======================================================================
def _stitch_row(row: list[NURBSSurfaceTuple],
                order_u: int, order_v: int
                ) -> tuple[NURBSSurfaceTuple, np.ndarray]:
    """
    Concatenate patches [P₀, P₁, …, P_{C-1}] along *u*.
    All patches in the row must share *exactly* the same v‑knot vector
    and (order_u, order_v).

    Returns  (new_row_surface , interior_knots_u_of_this_row)
    """
    p = order_u - 1

    # Storage for the growing row
    ku_out   = []         # global u‑knot vector under construction
    cp_rows  = []         # list of control meshes (to np.concatenate later)
    w_rows   = []         # list of weight meshes
    split_ku = []         # the knots that separate patches inside the row
    u_offset = 0.0        # right‑most knot value of what we have stitched

    common_kv = None      # will hold the v‑knot vector shared by the row

    for c, surf in enumerate(row):
        if surf.order_u != order_u or surf.order_v != order_v:
            raise ValueError(f"Orders differ inside the row: {surf.order_u},{order_u}, {surf.order_v},{order_v}")

        ku = surf.knot_u.astype(float).copy()
        kv = surf.knot_v.astype(float).copy()

        if common_kv is None:
            common_kv = kv
        elif not np.allclose(common_kv, kv, atol=1e-9):
            raise ValueError("Patches in the same row "
                             "have different v‑knot vectors")

        # Shift this patch so that its left clamp coincides with u_offset
        shift = u_offset - ku[0]
        ku += shift

        # --- append to the growing knot vector / control mesh -----------
        if c == 0:
            # First patch   keep its whole (clamped) knot vector except
            # the last (p+1) knots: they will be provided by the *last*
            # patch in the row so we do not duplicate them here.
            ku_out.extend(ku[:-order_u])
            cp_rows.append(surf.control_points)
            w_rows.append(surf.weights)
        else:
            # Middle or last   discard first knot (duplicate) and the last
            # p+1 knots.  Control mesh: drop the first ctrl‑row (dup)
            ku_out.extend(ku[1:-order_u])
            cp_rows.append(surf.control_points[1:, ...])
            w_rows.append( surf.weights       [1:, ...])

        u_offset = ku[-order_u]        # first of the trailing (p+1) clamp
        split_ku.append(u_offset)      # this is the end of patch c

    # Add the final right‑side clamp (p+1 identical knots)
    ku_out.extend([u_offset] * order_u)
    split_ku.pop()                     # last entry is the global u‑max

    # Glue control meshes and weights
    cp_row = np.concatenate(cp_rows, axis=0)          # concat in u
    w_row  = np.concatenate(w_rows , axis=0)

    stitched = NURBSSurfaceTuple(
        order_u = order_u,
        order_v = order_v,
        knot_u  = np.asarray(ku_out),
        knot_v  = common_kv,
        control_points = cp_row,
        weights        = w_row
    )
    return stitched, np.asarray(split_ku)


def _stitch_column(rows: list[NURBSSurfaceTuple],
                   order_u: int, order_v: int
                   ) -> tuple[NURBSSurfaceTuple, np.ndarray]:
    """
    Vertically concatenate the *rows* produced by _stitch_row.

    Returns  (big_surface , interior_knots_v)
    """
    q = order_v - 1

    kv_out   = []
    cp_cols  = []
    w_cols   = []
    split_kv = []
    v_offset = 0.0

    common_ku = None       # must be identical across rows

    for r, surf in enumerate(rows):
        if common_ku is None:
            common_ku = surf.knot_u
        elif not np.allclose(common_ku, surf.knot_u, atol=1e-9):
            raise ValueError("Rows disagree on their u‑knot vector")

        kv = surf.knot_v.astype(float).copy()
        shift = v_offset - kv[0]
        kv += shift

        if r == 0:
            kv_out.extend(kv[:-order_v])
            cp_cols.append(surf.control_points)
            w_cols.append(surf.weights)
        else:
            # Discard first v‑knot and first control‑column (dup)
            kv_out.extend(kv[1:-order_v])
            cp_cols.append(surf.control_points[:, 1:, ...])
            w_cols .append(surf.weights       [:, 1:, ...])

        v_offset = kv[-order_v]
        split_kv.append(v_offset)

    kv_out.extend([v_offset] * order_v)
    split_kv.pop()

    cp_big = np.concatenate(cp_cols, axis=1)          # concat in v
    w_big  = np.concatenate(w_cols , axis=1)

    big = NURBSSurfaceTuple(
        order_u        = order_u,
        order_v        = order_v,
        knot_u         = common_ku,
        knot_v         = np.asarray(kv_out),
        control_points = cp_big,
        weights        = w_big
    )
    return big, np.asarray(split_kv)


def degree_elevate_curve(curve: NURBSCurveTuple, num: int = 1):
    """Elevate the degree of a NURBS curve.

    Applies degree elevation algorithm to spline geometries by decomposing the curve
    into Bezier segments, elevating each segment, and then re-linking them.

    :param curve: NURBS curve to elevate
    :type curve: NURBSCurveTuple
    :param num: Number of degree elevations (default: 1)
    :type num: int
    :return: Curve with elevated degree
    :rtype: NURBSCurveTuple
    """

    # Start curve degree manipulation operations

    # Find multiplicity of the internal knots
    degree = curve.order - 1
    num_knots = len(curve.knot)
    num_control_points = num_knots - degree - 1
    #int_knots = num_knots[degree:][:num_control_points]
    crv_list = decompose_curve(curve)

    # Decompose the input by knot insertion

    # If parameter is positive, apply degree elevation. Otherwise, apply degree reduction
    crv_list_new = []
    # Loop through to apply degree elevation

    for crv in crv_list:
        interv=nurbs_interval(crv.knot,crv.order-1)
        new_cptsw = degree_elevation(crv.order - 1,
                                              to_homogeneous_1d(crv.control_points, crv.weights).tolist(), num=num)
        new_deg=crv.order - 1+num

        crv_list_new.append(NURBSCurveTuple(new_deg + 1, np.array(_bezier_knots(new_deg+1 ,interv)),
                              *from_homogeneous_1d(np.array(new_cptsw))))
        #print('d',crv_list_new[-1].knot)

    crv, joints = link_curves(crv_list_new)
    #print('d2', crv.knot)
    for k in joints:
        crv=remove_knot_curve(crv, k, crv.order - 1)
    return crv


def remove_knot_curve(curve: NURBSCurveTuple, knot: float, num: int = 1, **kwargs):
    """ Removes a knot from a spline curve."""

    mult=find_multiplicity(knot,curve.knot)

    if mult<num:

        raise ValueError(f"Cannot remove knot {knot} from knots: {curve.knot} with multiplicity {mult}")

    span=_find_span_linear(curve.order - 1, curve.knot, curve.control_points.shape[0], knot)
    hpts=to_homogeneous_1d(curve.control_points, curve.weights)
    new_kv,new_pt=knot_removal(curve.order-1, curve.knot.tolist(),ctrlpts=hpts,u=knot,num=num,span=span,**kwargs)
    # new_kv=knot_removal_kv(curve.knot.tolist(),span=span, r=num)
    return NURBSCurveTuple(curve.order , np.array(new_kv), *from_homogeneous_1d(np.array(new_pt)))


def remove_knot_curve_max(curve: NURBSCurveTuple, knot: float, num: int = 1, **kwargs):
    """Removes a knot from a spline curve."""
    stack = [(knot, num)]
    crv = curve
    result_n = 0

    while stack:

        k, n = stack.pop(0)
        #print(k, n)
        mult = find_multiplicity(k, crv.knot)

        if mult < n:
            if n > 1:

                stack.append((k, n - 1))
                continue
            else:
                result_n = 0
                break
        result_n = n
        span = _find_span_linear(crv.order - 1, crv.knot, crv.control_points.shape[0], k)
        hpts = to_homogeneous_1d(crv.control_points, crv.weights)
        new_kv, new_pt = knot_removal(crv.order - 1, crv.knot.tolist(), ctrlpts=hpts, u=k, num=n, span=span, **kwargs)
        crv = NURBSCurveTuple(crv.order, np.array(new_kv), *from_homogeneous_1d(np.array(new_pt)))
    #print(result_n, crv.knot.shape, curve.knot.shape)
    # new_kv=knot_removal_kv(curve.knot.tolist(),span=span, r=num)
    return crv, result_n


def remove_knot_surface_u(self: NURBSSurfaceTuple, t: float, num: int = 1, **kwargs):
    """Removes a knot from a spline curve."""
    cpts = np.copy(self.control_points)
    count = num
    cpts_size_u, cpts_size_v, dim = cpts.shape
    new_count_u = cpts_size_u - count
    new_count_v = cpts_size_v
    degree_u = self.order_u - 1

    span = _find_span_linear(degree_u, self.knot_u, cpts_size_u, t)

    # Compute new knot vector
    # k_v = knot_removal_kv(self.knot_u,  span, count)
    s_u = find_multiplicity(t, self.knot_u)

    if isinstance(self, BSplineSurfaceTuple):
        self = NURBSSurfaceTuple(*self, weights=np.ones(self.control_points.shape[:-1], dtype=float))
    new_pts = np.zeros((new_count_u, new_count_v, dim))
    new_weights = np.zeros((new_count_u, new_count_v))
    knot_u_list = np.array(self.knot_u).tolist()
    new_pts=[]
    for v in range(cpts_size_v):
        row_control_points = cpts[:, v, :]
        row_weights = self.weights[:, v]
        # Convert to homogeneous coordinates
        row_homo = to_homogeneous_1d(row_control_points, row_weights)
        row_homo_list = row_homo.tolist()

        # Apply knot insertion

        k_v,        new_row_homo_list = knot_removal(degree_u, knot_u_list, row_homo_list, t, num=count, span=span, s=s_u)
        new_pts.append(new_row_homo_list)

    return NURBSSurfaceTuple(self.order_u,self.order_v,k_v, self.knot_v,*from_homogeneous_2d(np.asarray(new_pts).swapaxes(0, 1)))


def remove_knot_surface_v(self: NURBSSurfaceTuple, t: float,num: int = 1, **kwargs):

    count = num

    cpts = np.copy(self.control_points)

    cpts_size_u, cpts_size_v, dim = cpts.shape
    new_count_u = cpts_size_u
    new_count_v = cpts_size_v- count
    degree_v = self.order_v - 1

    span = _find_span_linear(degree_v, self.knot_v, cpts_size_v, t)

    # Compute new knot vector
    # k_v = knot_removal_kv(self.knot_v, span, count)
    s_v = find_multiplicity(t, self.knot_v)

    if isinstance(self, BSplineSurfaceTuple):
        self = NURBSSurfaceTuple(*self, weights=np.ones(self.control_points.shape[:-1], dtype=float))
    new_pts = []
    new_weights = np.zeros((new_count_u, new_count_v))
    knot_v_list = np.asarray(self.knot_v).tolist()
    for u in range(cpts_size_u):
        col_control_points = cpts[u, :, :]
        col_weights = self.weights[u, :]
        # Convert to homogeneous coordinates
        col_homo = to_homogeneous_1d(col_control_points, col_weights)
        col_homo_list = col_homo.tolist()

        # Apply knot insertion

        k_v, new_row_homo_list = knot_removal(degree_v, knot_v_list, col_homo_list, t, num=count, span=span, s=s_v)
        new_pts.append(new_row_homo_list)

    return NURBSSurfaceTuple(self.order_u, self.order_v,  self.knot_u, k_v,*from_homogeneous_2d(np.asarray(new_pts)))

def remove_knot_surface(surface: NURBSSurfaceTuple, u: float, v: float, num_u: int = 1, num_v: int = 1,**kwargs):
    return remove_knot_surface_u(remove_knot_surface_v(surface, v, num=num_v, **kwargs), u, num=num_u, **kwargs)

def degree_reduction(degree, ctrlpts, **kwargs):
    """ Computes the control points of the rational/non-rational spline after degree reduction.

    Implementation of Eqs. 5.41 and 5.42 of The NURBS Book by Piegl & Tiller, 2nd Edition, p.220

    Please note that degree reduction algorithm can only operate on Bezier shapes, i.e. curves, surfaces, volumes and
    this implementation does NOT compute the maximum error tolerance as described via Eqs. 5.45 and 5.46 of The NURBS
    Book by Piegl & Tiller, 2nd Edition, p.221 to determine whether the shape is degree reducible or not.

    :param degree: degree
    :type degree: int
    :param ctrlpts: control points
    :type ctrlpts: list, tuple
    :return: control points of the degree-reduced shape
    :rtype: list
    """


    # Initialize variables
    pts_red = [[0.0 for _ in range(len(ctrlpts[0]))] for _ in range(degree)]

    # Fix start and end control points
    pts_red[0] = ctrlpts[0]
    pts_red[-1] = ctrlpts[-1]

    # Find if the degree is an even or an odd number
    p_is_odd = True if degree % 2 != 0 else False

    # Compute control points of degree-reduced 1-dimensional shape
    r = int((degree - 1) / 2)
    # Handle a special case when degree = 2
    if degree == 2:
        r1 = r - 2
    else:
        # Determine r1 w.r.t. degree evenness
        r1 = r - 1 if p_is_odd else r
    for i in range(1, r1 + 1):
        alpha = float(i) / float(degree)
        pts_red[i] = [(c1 - (alpha * c2)) / (1 - alpha) for c1, c2 in zip(ctrlpts[i], pts_red[i - 1])]
    for i in range(degree - 2, r1 + 2):
        alpha = float(i + 1) / float(degree)
        pts_red[i] = [(c1 - ((1 - alpha) * c2)) / alpha for c1, c2 in zip(ctrlpts[i + 1], pts_red[i + 1])]

    if p_is_odd:
        alpha = float(r) / float(degree)
        left = [(c1 - (alpha * c2)) / (1 - alpha) for c1, c2 in zip(ctrlpts[r], pts_red[r - 1])]
        alpha = float(r + 1) / float(degree)
        right = [(c1 - ((1 - alpha) * c2)) / alpha for c1, c2 in zip(ctrlpts[r + 1], pts_red[r + 1])]
        pts_red[r] = [0.5 * (pl + pr) for pl, pr in zip(left, right)]

    # Return computed control points after degree reduction
    return pts_red


def refine_curve(curve: NURBSCurveTuple, new_knots, density: int = 0, **kwargs):
    """Refine a NURBS curve by inserting new knots.

    Refines the curve by inserting additional knots without changing the shape.
    Uses knot refinement algorithm to maintain geometric continuity.

    :param curve: NURBS curve to refine
    :type curve: NURBSCurveTuple
    :param new_knots: New knots to insert into the knot vector
    :type new_knots: array-like
    :param density: Knot density multiplier for automatic refinement (default: 0)
    :type density: int
    :return: Refined NURBS curve with additional knots
    :rtype: NURBSCurveTuple
    """

    cptsw=to_homogeneous_1d( np.asarray(curve.control_points,dtype=float),np.asarray(curve.weights,dtype=float))

    new_cptsw,new_knots=knot_refinement(curve.order-1, np.asarray(curve.knot,dtype=float).tolist(),cptsw.tolist(),density=density, add_knot_list=np.asarray(list(set(new_knots)-set(curve.knot)),dtype=float).tolist(),**kwargs)
    new_cpts,new_weights=from_homogeneous_1d(np.asarray(new_cptsw))
    return curve._replace(knot=np.asarray(new_knots,dtype=float),control_points=new_cpts,weights=new_weights)


def reverse_curve(curve: NURBSCurveTuple) -> NURBSCurveTuple:
    """
    Reverse the direction of a NURBS curve by flipping the control points, weights,
    and recalculating the knot vector.
    
    Parameters:
    -----------
    curve: NURBSCurveTuple
        The NURBS curve to reverse
        
    Returns:
    --------
    NURBSCurveTuple
        The reversed curve
    """
    # Create copies of data to avoid modifying the original curve
    control_points = np.copy(curve.control_points)
    weights = np.copy(curve.weights)
    knots = np.copy(curve.knot)

    # Reverse control points and weights
    control_points = np.flip(control_points, axis=0)
    weights = np.flip(weights, axis=0)

    # Calculate the reversed knot vector
    # The idea is to maintain the correct parametrization
    # First, normalize the original knot vector to [0, 1]
    a, b = nurbs_interval(knots, curve.order - 1)
    knot_span = b - a

    # Compute the reversed knot vector
    # We use the formula k'_i = a + b - k_{n-i}
    # where n is the last index of the knot vector
    n = len(knots) - 1
    reversed_knots = np.zeros_like(knots)
    for i in range(len(knots)):
        reversed_knots[i] = a + b - knots[n - i]

    # Return the new NURBSCurveTuple with reversed components
    return NURBSCurveTuple(
        order=curve.order,
        knot=reversed_knots,
        control_points=control_points,
        weights=weights
    )



# --- tuning knobs ---
SNAP_TOL_ABS = 1e-2  # absolute tolerance for treating interior knots as "the same"
MIDPOINT = 0.5  # tie-break reference inside [0,1] after normalization


def _get_order(curve):
    # Your curves use .order throughout
    return int(curve.order)


def _get_knots(curve):
    return np.asarray(curve.knot, dtype=float)


def _set_knots(curve, new_knot):
    """Return a curve with updated knot vector while preserving everything else."""
    new_knot = np.asarray(new_knot, dtype=float)
    if hasattr(curve, "_replace"):
        return curve._replace(knot=new_knot)
    # fallback: assume mutable
    curve.knot = new_knot
    return curve


def _interior(knots, order):
    """Interior knots excluding the clamped ends of multiplicity = order."""
    if order <= 0 or len(knots) < 2 * order:
        return np.array([], dtype=float)
    return np.asarray(knots[order:-order], dtype=float)


def _rebuild_knots_from_interior(knots_like, order, interior):
    """Build a full open-clamped knot vector from ends of knots_like and given interior."""
    k = np.asarray(knots_like, dtype=float)
    a, b = k[0], k[-1]
    head = np.full(order, a, dtype=float)
    tail = np.full(order, b, dtype=float)
    return np.concatenate([head, np.asarray(interior, dtype=float), tail])


def _choose_rep_value(values):
    """
    Choose a representative value from a cluster of near-equal values.
    Rule:
      1) prefer the most frequent exact value,
      2) tie-break by closeness to MIDPOINT,
      3) then by smaller numeric value.
    """
    values = [float(v) for v in values]
    counts = Counter(values)
    maxc = max(counts.values())
    candidates = [v for v, c in counts.items() if c == maxc]
    rep = min(candidates, key=lambda v: (abs(v - MIDPOINT), v))
    return rep


def _cluster_all_interiors(curves, order, tol_abs=SNAP_TOL_ABS):
    """
    Cluster all interior knots across curves using a simple single-link threshold.
    Returns:
      clusters: list of dicts with keys {rep, lo, hi, target_mult, members}
        - rep: chosen representative value
        - lo, hi: min/max observed in the cluster
        - target_mult: max per-curve multiplicity needed in the final target
        - members: list of (curve_index, value)
      target_interior: sorted list with each cluster's rep repeated target_mult times
    """
    entries = []
    for ci, curve in enumerate(curves):
        k = _get_knots(curve)
        inter = _interior(k, order)
        for v in inter:
            entries.append((float(v), ci))
    if not entries:
        return [], []

    entries.sort(key=lambda x: x[0])

    clusters = []
    cur_cluster = [entries[0]]
    for v, ci in entries[1:]:
        v_prev = cur_cluster[-1][0]
        if abs(v - v_prev) <= tol_abs:
            cur_cluster.append((v, ci))
        else:
            clusters.append(cur_cluster)
            cur_cluster = [(v, ci)]
    clusters.append(cur_cluster)

    result = []
    for cluster in clusters:
        vals = [v for v, _ in cluster]
        rep = _choose_rep_value(vals)
        lo, hi = min(vals), max(vals)

        per_curve = Counter(ci for _, ci in cluster)
        target_mult = max(per_curve.values())  # union multiplicity rule

        result.append({"rep": rep, "lo": lo, "hi": hi, "target_mult": int(target_mult), "members": cluster})

    # Build the target interior multiset
    target_interior = []
    for c in result:
        target_interior.extend([c["rep"]] * c["target_mult"])
    target_interior = np.array(sorted(target_interior), dtype=float)
    return result, target_interior


def _snap_curve_to_clusters(curve, order, clusters):
    """
    Replace interior knots that fall into a cluster's [lo, hi] band with the cluster's rep.
    Does NOT insert/remove; only changes values. Returns a curve.
    """
    knots = _get_knots(curve)
    inter = _interior(knots, order)
    if inter.size == 0:
        return curve  # nothing to snap

    new_inter = []
    j = 0
    for u in inter:
        # advance cluster pointer to catch up
        while j < len(clusters) and u > clusters[j]["hi"] + SNAP_TOL_ABS:
            j += 1
        if j < len(clusters) and clusters[j]["lo"] - SNAP_TOL_ABS <= u <= clusters[j]["hi"] + SNAP_TOL_ABS:
            new_inter.append(clusters[j]["rep"])
        else:
            # not in any cluster; keep as is
            new_inter.append(u)

    new_knots = _rebuild_knots_from_interior(knots, order, new_inter)
    return _set_knots(curve, new_knots)





def _insert_missing_to_reach_target(curve, order, target_interior):
    """
    Insert missing knots so that the curve's interior matches the target multiplicities.
    We assume values are already snapped to cluster reps, so the multiset difference is well-defined.
    """
    k = _get_knots(curve)
    inter = _interior(k, order)

    need = Counter(map(float, target_interior))
    have = Counter(map(float, inter))
    to_insert = []
    for val, cnt_needed in need.items():
        missing = cnt_needed - have.get(val, 0)
        if missing > 0:
            to_insert.extend([val] * missing)

    if to_insert:
        # refine_curve is assumed to insert the provided knot values (density=0 to avoid extra sampling)

        for i in sorted(to_insert):
            curve = insert_knot_curve(curve, i, 1)
            curve, rem_count = remove_knot_curve_max(curve, i, curve.order - 1)
            #print(rem_count)
        return curve
    # Final re-snap to ensure numeric equality with the chosen reps (guard against tiny numerical drift)
    # Build the full target knot vector explicitly, then set it.
    new_knots = _rebuild_knots_from_interior(k, order, target_interior)
    curve = _set_knots(curve, new_knots)
    return curve


def _all_same_length(arrs):
    if not arrs:
        return True
    L = len(arrs[0])
    return all(len(a) == L for a in arrs)


def _pick_reference_interior(interiors):
    """
    Pick one existing interior vector to use as the shared reference when all lengths match.
    We choose a medoid (minimizes sum of L1 distances to others). On ties, take the first.
    """
    if not interiors:
        return np.array([], dtype=float)
    if len(interiors) == 1:
        return interiors[0].copy()

    dists = []
    for i, a in enumerate(interiors):
        total = 0.0
        for j, b in enumerate(interiors):
            if i == j:
                continue
            total += np.sum(np.abs(a - b))
        dists.append((total, i))
    _, idx = min(dists, key=lambda t: (t[0], t[1]))
    return interiors[idx].copy()


# ----------------------------------------------------------------------
# Pairwise version
# ----------------------------------------------------------------------
def make_curves_compatible(curve1, curve2):
    """
    Make two NURBS curves compatible for ruled surface construction.

    Strategy:
      1) Elevate degrees (orders) to the max.
      2) Normalize knots to [0, 1].
      3) If lengths match, snap both to a single existing knot vector (no insertion).
      4) Else, cluster near-duplicates, snap to cluster reps, then insert only missing knots.
    """
    # Normalize and degree-elevate
    c1 = normalize_knots_curve(curve1)
    c2 = normalize_knots_curve(curve2)
    p1, p2 = c1.order, c2.order
    if p1 < p2:
        c1 = degree_elevate_curve(c1, p2 - p1)
    elif p2 < p1:
        c2 = degree_elevate_curve(c2, p1 - p2)

    # Re-normalize just in case degree elevation perturbed knots
    c1 = normalize_knots_curve(c1)
    c2 = normalize_knots_curve(c2)
    order = _get_order(c1)

    k1 = _get_knots(c1)
    k2 = _get_knots(c2)

    # If same total knot-vector length, simply align values to a single existing set.
    if len(k1) == len(k2):
        inter1 = _interior(k1, order)
        inter2 = _interior(k2, order)
        ref_inter = _pick_reference_interior([inter1, inter2])  # chooses an existing interior vector
        new_knots = _rebuild_knots_from_interior(k1, order, ref_inter)
        c1 = _set_knots(c1, new_knots)
        c2 = _set_knots(c2, new_knots)
        return c1, c2

    # Otherwise: smart union with snapping
    clusters, target_interior = _cluster_all_interiors([c1, c2], order, tol_abs=SNAP_TOL_ABS)

    # Snap each curve to cluster reps
    c1 = _snap_curve_to_clusters(c1, order, clusters)
    c2 = _snap_curve_to_clusters(c2, order, clusters)

    # Insert only what is missing, then finalize to exact target_interior for both
    c1 = _insert_missing_to_reach_target(c1, order, target_interior)
    c2 = _insert_missing_to_reach_target(c2, order, target_interior)

    return c1, c2


# ----------------------------------------------------------------------
# Multiple-curves version
# ----------------------------------------------------------------------
def make_curves_compatible_multiple(curves):
    """
    Make multiple NURBS curves compatible for surface construction.

    Steps:
      1) Elevate all to the highest order.
      2) Normalize knots to [0,1].
      3) If all knot-vector lengths match, snap everyone to a single existing knot vector (no insertion).
      4) Else, cluster near-duplicates across ALL interiors, snap, then insert only what is missing.
    """
    curves = list(curves)
    # Normalize and find max order
    max_order = 0
    for i in range(len(curves)):
        curves[i] = normalize_knots_curve(curves[i])
        if curves[i].order > max_order:
            max_order = curves[i].order

    # Elevate to max order, re-normalize
    for i in range(len(curves)):
        c = curves[i]
        delta = max_order - c.order
        if delta > 0:
            c = degree_elevate_curve(c, delta)
        curves[i] = normalize_knots_curve(c)

    order = max_order
    knot_lists = [_get_knots(c) for c in curves]

    # Case A: all same length -> choose a single existing interior and snap to it
    if _all_same_length(knot_lists):
        interiors = [_interior(k, order) for k in knot_lists]
        ref_inter = _pick_reference_interior(interiors)
        for i, c in enumerate(curves):
            new_knot = _rebuild_knots_from_interior(_get_knots(c), order, ref_inter)
            curves[i] = _set_knots(c, new_knot)
        return curves

    # Case B: smart union with snapping across all
    clusters, target_interior = _cluster_all_interiors(curves, order, tol_abs=SNAP_TOL_ABS)

    # Snap and insert missing for each curve
    for i, c in enumerate(curves):
        c = _snap_curve_to_clusters(c, order, clusters)
        c = _insert_missing_to_reach_target(c, order, target_interior)
        curves[i] = c

    return curves


class KnotRemovalResult(NamedTuple):
    curve: Optional[NURBSCurveTuple]
    success: bool
    error: float
    removed_knot: float


# --- Helper: Basis Functions (Standard Cox-de Boor) ---

def find_span(n: int, p: int, u: float, U: NDArray[np.float64]) -> int:
    """Finds the knot span index for a given parameter u."""
    if u >= U[n + 1]:
        return n
    low = p
    high = n + 1
    mid = (low + high) // 2
    while (u < U[mid]) or (u >= U[mid + 1]):
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def ders_basis_funs(span_i: int, u: float, p: int, U: NDArray[np.float64], n_ders: int = 1) -> NDArray[np.float64]:
    """
    Computes basis functions and their derivatives.
    Returns shape (n_ders + 1, p + 1).
    """
    ders = np.zeros((n_ders + 1, p + 1))
    ndu = np.zeros((p + 1, p + 1))
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)

    ndu[0, 0] = 1.0

    for j in range(1, p + 1):
        left[j] = u - U[span_i + 1 - j]
        right[j] = U[span_i + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    for j in range(p + 1):
        ders[0, j] = ndu[j, p]

    # Compute derivatives
    a = np.zeros((2, p + 1))
    for r in range(0, p + 1):
        s1 = 0;
        s2 = 1
        a[0, 0] = 1.0
        for k in range(1, n_ders + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            j = s1;
            s1 = s2;
            s2 = j

    r = p
    for k in range(1, n_ders + 1):
        for j in range(p + 1):
            ders[k, j] *= r
        r *= (p - k)

    return ders


# --- Helper: Homogeneous Coordinate Math ---

def to_homogeneous(cps: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Lifts Euclidean points (NxD) and weights (N) to Homogeneous points (Nx(D+1))."""
    n, dim = cps.shape
    # Shape: (N, Dim + 1)
    # [x*w, y*w, z*w, w]
    hom_cps = np.zeros((n, dim + 1))
    hom_cps[:, :dim] = cps * weights[:, np.newaxis]
    hom_cps[:, dim] = weights
    return hom_cps


def to_euclidean(hom_cps: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Projects Homogeneous points (Nx(D+1)) back to Euclidean (NxD) and weights (N)."""
    dim = hom_cps.shape[1] - 1
    weights = hom_cps[:, dim]

    # Avoid division by zero in degenerate cases
    safe_weights = np.where(np.abs(weights) < 1e-12, 1.0, weights)

    cps = hom_cps[:, :dim] / safe_weights[:, np.newaxis]
    return cps, weights


def evaluate_b_spline_derivs(
        knot: NDArray[np.float64],
        cps: NDArray[np.float64],
        p: int,
        u: float,
        n_ders: int = 1
) -> NDArray[np.float64]:
    """
    Evaluates a B-Spline curve and its derivatives at u.
    This works generically for N-dimensional control points.
    If 'cps' are Homogeneous, this returns Homogeneous derivatives.
    """
    n = len(cps) - 1
    span = find_span(n, p, u, knot)
    basis = ders_basis_funs(span, u, p, knot, n_ders)

    dim = cps.shape[1]
    result = np.zeros((n_ders + 1, dim))

    active_cps = cps[span - p: span + 1]

    for k in range(n_ders + 1):
        result[k] = np.dot(basis[k], active_cps)

    return result


# --- Main Algorithm ---

def remove_knot(
        curve: NURBSCurveTuple,
        u_remove: float,
        tolerance: float = 1e-4
) -> KnotRemovalResult:
    """
    Removes a knot from a Rational NURBS curve using Homogeneous Hermite-Birkhoff interpolation.

    Steps:
    1. Convert Control Points to Homogeneous Space (Cw).
    2. Identify the local knot span and affected Control Points.
    3. Sample the original curve's Homogeneous values (Pos + Deriv) at boundaries.
    4. Solve the linear system (Least Squares) to find new Homogeneous Control Points.
    5. Project back to Euclidean space.
    6. Measure Euclidean error.
    """
    p = curve.order - 1
    U_old = curve.knot

    # 1. Convert to Homogeneous
    P_hom_old = to_homogeneous(curve.control_points, curve.weights)
    hom_dim = P_hom_old.shape[1]  # D + 1

    # 2. Identify Knot to Remove
    matches = np.where(np.abs(U_old - u_remove) < 1e-9)[0]
    if len(matches) == 0:
        return KnotRemovalResult(None, False, float('inf'), u_remove)

    r = matches[-1]

    # Safety check for boundary knots
    if r >= len(U_old) - p - 1:
        if len(matches) > 1:
            r = matches[-2]
        else:
            return KnotRemovalResult(None, False, float('inf'), u_remove)

    U_new = np.delete(U_old, r)

    # Range of CPs to recalculate: Q_{r-p} ... Q_{r-1}
    start_idx = r - p
    end_idx = r - 1
    num_unknowns = p

    if start_idx < 0:
        return KnotRemovalResult(None, False, float('inf'), u_remove)

    # 3. Define Constraints (Hermite + Internal)
    u_a = U_new[r - 1]
    u_b = U_new[r]

    samples = []
    # Hermite Constraints (Preserve C1 continuity of the weighted curve)
    samples.append((u_a, 0))  # Pos
    samples.append((u_a, 1))  # Deriv
    samples.append((u_b, 0))  # Pos
    samples.append((u_b, 1))  # Deriv

    # Internal Constraints for p > 3
    if p > 3:
        num_internal = max(0, p - 4)
        if num_internal > 0:
            t_internal = np.linspace(u_a, u_b, num_internal + 2)[1:-1]
            for t in t_internal:
                samples.append((t, 0))

    # 4. Build Linear System in Homogeneous Space
    num_constraints = len(samples)
    A = np.zeros((num_constraints, num_unknowns))
    B = np.zeros((num_constraints, hom_dim))

    n_new = len(P_hom_old) - 2

    for row_i, (u, mode) in enumerate(samples):
        # Compute Basis on NEW Knot Vector
        span_new = find_span(n_new, p, u, U_new)
        ders_new = ders_basis_funs(span_new, u, p, U_new, n_ders=1)
        basis_vals = ders_new[mode, :]

        # Target: Evaluate OLD curve in Homogeneous Space
        # returns shape (n_ders+1, hom_dim), we take [mode]
        target_val_hom = evaluate_b_spline_derivs(U_old, P_hom_old, p, u, n_ders=1)[mode]

        rhs_contrib = target_val_hom.copy()

        for j in range(p + 1):
            cp_idx = span_new - p + j
            weight_basis = basis_vals[j]

            if start_idx <= cp_idx <= end_idx:
                # Unknown Variable
                col_idx = cp_idx - start_idx
                if 0 <= col_idx < num_unknowns:
                    A[row_i, col_idx] += weight_basis
            else:
                # Fixed / Known Control Point
                if cp_idx < start_idx:
                    fixed_cp = P_hom_old[cp_idx]
                else:
                    fixed_cp = P_hom_old[cp_idx + 1]

                rhs_contrib -= weight_basis * fixed_cp

        B[row_i] = rhs_contrib

    # 5. Solve (Least Squares)
    try:
        Q_hom_local, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    except np.linalg.LinAlgError:
        return KnotRemovalResult(None, False, float('inf'), u_remove)

    # Reconstruct Full Homogeneous Array
    P_hom_new = np.zeros((len(P_hom_old) - 1, hom_dim))
    P_hom_new[:start_idx] = P_hom_old[:start_idx]
    P_hom_new[start_idx: end_idx + 1] = Q_hom_local
    P_hom_new[end_idx + 1:] = P_hom_old[end_idx + 2:]

    # 6. Project back to Euclidean
    P_new, W_new = to_euclidean(P_hom_new)

    new_curve = NURBSCurveTuple(p + 1, U_new, P_new, W_new)

    # 7. Evaluate Euclidean Error
    # We compare the Euclidean 3D positions at u_remove
    # Original
    pt_old_hom = evaluate_b_spline_derivs(U_old, P_hom_old, p, u_remove, 0)[0]
    pt_old_euc = pt_old_hom[:-1] / pt_old_hom[-1]

    # New
    pt_new_hom = evaluate_b_spline_derivs(U_new, P_hom_new, p, u_remove, 0)[0]
    pt_new_euc = pt_new_hom[:-1] / pt_new_hom[-1]

    dist = np.linalg.norm(pt_old_euc - pt_new_euc)
    success = dist <= tolerance

    return KnotRemovalResult(new_curve, success, dist, u_remove)