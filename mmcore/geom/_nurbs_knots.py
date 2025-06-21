from __future__ import annotations

from copy import deepcopy

import numpy as np

from mmcore.geom._nurbs_eval import (
    nurbs_interval, _find_span_linear, _copy_curve, _copy_surface, to_homogeneous_1d, from_homogeneous_1d,
    NURBSCurveTuple, BSplineCurveTuple, NURBSSurfaceTuple, BSplineSurfaceTuple, _curve_interval
)




from numpy.typing import NDArray


def generate_knots(control_points_count,degree, interval=None):
    """

    This function generates default knots based on the number of control points
    :return: A list of knots
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

def normalize_knots(knots,degree):
    start,end=nurbs_interval(knots,degree)

    d=abs(end-start)

    return (np.asarray(knots)-start)/d
def normalize_knots_curve(curve:NURBSCurveTuple):
    knots=normalize_knots(curve.knot,curve.order-1)


    return curve._replace(knot=knots)
def normalize_knots_curve_inplace(curve:NURBSCurveTuple):
    curve.knot[:]=normalize_knots(curve.knot,curve.order-1)


def normalize_knots_surface_inplace(surf:NURBSSurfaceTuple):
    surf.knot_u[:]=normalize_knots(surf.knot_u,surf.order_u-1)
    surf.knot_v[:]=normalize_knots(surf.knot_v,surf.order_v-1)

def knot_insertion_alpha( u,  knotvector,  span,  idx, leg):
    return (u - knotvector[leg + idx]) / (knotvector[idx + span + 1] - knotvector[leg + idx])


def find_multiplicity(knot, knot_vector, **kwargs):
    """ Finds knot multiplicity over the knot vector.

    Keyword Arguments:
        * ``tol``: tolerance (delta) value for equality checking

    :param knot: knot or parameter, :math:`u`
    :type knot: float
    :param knot_vector: knot vector, :math:`U`
    :type knot_vector: list, tuple
    :return: knot multiplicity, :math:`s`
    :rtype: int
    """
    # Get tolerance value
    tol = kwargs.get('tol', 10e-15)

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


def knot_insertion(degree, knotvector, ctrlpts, u, num:int=1, span=None,s=None,**kwargs):
    """ Computes the control points of the rational/non-rational spline after knot insertion.

    Part of Algorithm A5.1 of The NURBS Book by Piegl & Tiller, 2nd Edition.

    Keyword Arguments:
        * ``num``: number of knot insertions. *Default: 1*
        * ``s``: multiplicity of the knot. *Default: computed via :func:`.find_multiplicity`*
        * ``span``: knot span. *Default: computed via :func:`.find_span_linear`*

    :param degree: degree
    :type degree: int
    :param knotvector: knot vector
    :type knotvector: list, tuple
    :param ctrlpts: control points
    :type ctrlpts: list
    :param u: knot to be inserted
    :type u: float
    :return: updated control points
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


def knot_removal_alpha_i( u, degree,  knotvector,  num, idx) :
    return (u - knotvector[idx]) / (knotvector[idx + degree + 1 + num] - knotvector[idx])


def knot_removal_alpha_j(u,  degree, knotvector, num, idx) :
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


def knot_removal(degree, knotvector, ctrlpts, u, num=1,s:int=None,span=None,tol=1e-12,**kwargs):
    """ Computes the control points of the rational/non-rational spline after knot removal.

    Implementation based on Algorithm A5.8 and Equation 5.28 of The NURBS Book by Piegl & Tiller

    Keyword Arguments:
        * ``num``: number of knot removals

    :param degree: degree
    :type degree: int
    :param knotvector: knot vector
    :type knotvector: list, tuple
    :param ctrlpts: control points
    :type ctrlpts: list
    :param u: knot to be removed
    :type u: float
    :return: updated control points
    :rtype: list
    """



    s=find_multiplicity(u, knotvector) if s is None else s
    r =_find_span_linear(degree, knotvector, len(ctrlpts), u) if span is None else span


    # Edge case
    if num < 1:
        return ctrlpts

    # Initialize variables
    first = r - degree
    last = r - s

    # Don't change input variables, prepare new ones for updating
    ctrlpts_new = deepcopy(ctrlpts)

    is_volume = True
    if isinstance(ctrlpts_new[0][0], float):
        is_volume = False

    # Initialize temp array for storing new control points
    if is_volume:
        temp = [[[] for _ in range(len(ctrlpts_new[0]))] for _ in range((2 * degree) + 1)]
    else:
        temp = [[] for _ in range((2 * degree) + 1)]

    # Loop for Eqs 5.28 & 5.29
    for t in range(0, num):
        temp[0] = ctrlpts[first - 1]
        temp[last - first + 2] = ctrlpts[last + 1]
        i = first
        j = last
        ii = 1
        jj = last - first + 1
        remflag = False

        # Compute control points for one removal step
        while j - i >= t:
            alpha_i = knot_removal_alpha_i(u, degree, tuple(knotvector), t, i)
            alpha_j = knot_removal_alpha_j(u, degree, tuple(knotvector), t, j)
            if is_volume:
                for idx in range(len(ctrlpts[0])):
                    temp[ii][idx] = [(cpt - (1.0 - alpha_i) * ti) / alpha_i for cpt, ti
                                     in zip(ctrlpts[i][idx], temp[ii - 1][idx])]
                    temp[jj][idx] = [(cpt - alpha_j * tj) / (1.0 - alpha_j) for cpt, tj
                                     in zip(ctrlpts[j][idx], temp[jj + 1][idx])]
            else:
                temp[ii] = [(cpt - (1.0 - alpha_i) * ti) / alpha_i for cpt, ti in zip(ctrlpts[i], temp[ii - 1])]
                temp[jj] = [(cpt - alpha_j * tj) / (1.0 - alpha_j) for cpt, tj in zip(ctrlpts[j], temp[jj + 1])]
            i += 1
            j -= 1
            ii += 1
            jj -= 1

        # Check if the knot is removable
        if j - i < t:
            if is_volume:
                if np.linalg.norm(np.array(temp[ii - 1][0])- np.array(temp[jj + 1][0])) <= tol:
                    remflag = True
            else:
                if np.linalg.norm(np.array(temp[ii - 1]) - np.array(temp[jj + 1])) <= tol:
                    remflag = True
        else:
            alpha_i = knot_removal_alpha_i(u, degree, tuple(knotvector), t, i)
            if is_volume:
                ptn = [(alpha_i * t1) + ((1.0 - alpha_i) * t2) for t1, t2 in zip(temp[ii + t + 1][0], temp[ii - 1][0])]
            else:
                ptn = [(alpha_i * t1) + ((1.0 - alpha_i) * t2) for t1, t2 in zip(temp[ii + t + 1], temp[ii - 1])]
            if np.linalg.norm(np.asarray(ctrlpts[i]) - np.asarray( ptn))<= tol:
                remflag = True

        # Check if we can remove the knot and update new control points array
        if remflag:
            i = first
            j = last
            while j - i > t:
                ctrlpts_new[i] = temp[i - first + 1]
                ctrlpts_new[j] = temp[j - first + 1]
                i += 1
                j -= 1

        # Update indices
        first -= 1
        last += 1

    # Fix indexing
    t += 1

    # Shift control points (refer to p.183 of The NURBS Book, 2nd Edition)
    j = int((2*r - s - degree) / 2)  # first control point out
    i = j
    for k in range(1, t):
        if k % 2 == 1:
            i += 1
        else:
            j -= 1
    for k in range(i+1, len(ctrlpts)):
        ctrlpts_new[j] = ctrlpts[k]
        j += 1

    # Slice to get the new control points
    ctrlpts_new = ctrlpts_new[0:-t]

    return ctrlpts_new

def insert_knot_curve(curve:BSplineCurveTuple|NURBSCurveTuple,u:float, num:int=1):
    """Insert a knot into a curve multiple times.
    
    Args:
        curve: The curve to modify
        u: The knot value to insert
        num: Number of times to insert the knot (default: 1)
        
    Returns:
        A new curve with the knot inserted
    """
    rational = isinstance(curve, NURBSCurveTuple)
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
    
    return BSplineCurveTuple(curve.order, knot=np.array(kv_new), control_points=np.array(new_control_points))


def split_curve(curve:BSplineCurveTuple|NURBSCurveTuple, t:float, **kwargs):
    """ Splits the curve at the input parametric coordinate.

    This method splits the curve into two pieces at the given parametric coordinate, generates two different
    curve objects and returns them. It does not modify the input curve.

    
    :param obj: Curve to be split
    :type obj: abstract.Curve
    :param param: parameter
    :type param: float
    :return: a list of curve segments
    :rtype: list
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
    rational=isinstance(curve, NURBSCurveTuple)

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
        curve1_ctrlpts = cpts[0:ks + r]
        curve2_ctrlpts = cpts[ks + r - 1:]
        curve1 = BSplineCurveTuple(temp_obj.order, curve1_kv, np.asarray(curve1_ctrlpts))
        curve2 = BSplineCurveTuple(temp_obj.order, curve2_kv, np.asarray(curve2_ctrlpts))
    return curve1,curve2

def split_curve_multiple(crv:BSplineCurveTuple|NURBSCurveTuple, params:list[float]|NDArray[float])->list[BSplineCurveTuple]|list[NURBSCurveTuple]:
    crvs = []
    #temp = _copy_curve(crv)

    for i in range(len(params)):
        tpl = split_curve(crv, params[i])
        crv = tpl[1]
        crvs.append(tpl[0])
    crvs.append(crv)
    return crvs

def decompose_curve(crv:BSplineCurveTuple|NURBSCurveTuple)->list[BSplineCurveTuple]|list[NURBSCurveTuple]:
    params=np.unique(crv.knot)
    params=params[1:][:params.shape[0]-2]

    return split_curve_multiple(crv,params)


def trim_curve(curve:BSplineCurveTuple|NURBSCurveTuple, t0:float,t1:float):

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



def knot_refinement(degree, knotvector, ctrlpts, density:int=1,knot_list=None,add_knot_list=None,tol=1e-12,**kwargs):
    """ Computes the knot vector and the control points of the rational/non-rational spline after knot refinement.

    Implementation of Algorithm A5.4 of The NURBS Book by Piegl & Tiller, 2nd Edition.

    The algorithm automatically find the knots to be refined, i.e. the middle knots in the knot vector, and their
    multiplicities, i.e. number of same knots in the knot vector. This is the basis of knot refinement algorithm.
    This operation can be overridden by providing a list of knots via ``knot_list`` argument. In addition, users can
    provide a list of additional knots to be inserted in the knot vector via ``add_knot_list`` argument.

    Moreover, a numerical ``density`` argument can be used to automate extra knot insertions. If ``density`` is bigger
    than 1, then the algorithm finds the middle knots in each internal knot span to increase the number of knots to be
    refined.

    **Example**: Let the degree is 2 and the knot vector to be refined is ``[0, 2, 4]`` with the superfluous knots
    from the start and end are removed. Knot vectors with the changing ``density (d)`` value will be:

    * ``d = 1``, knot vector ``[0, 1, 1, 2, 2, 3, 3, 4]``
    * ``d = 2``, knot vector ``[0, 0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4]``

    Keyword Arguments:
        * ``knot_list``: knot list to be refined. *Default: list of internal knots*
        * ``add_knot_list``: additional list of knots to be refined. *Default: []*
        * ``density``: Density of the knots. *Default: 1*

    :param degree: degree
    :type degree: int
    :param knotvector: knot vector
    :type knotvector: list, tuple
    :param ctrlpts: control points
    :return: updated control points and knot vector
    :rtype: tuple
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
    a = _find_span_linear(degree, knotvector, n, X[0])
    b = _find_span_linear(degree, knotvector, n, X[r]) + 1

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




from mmcore.numeric.binom import binomial_coefficient_py

def degree_elevation(degree, ctrlpts, num=1,**kwargs):
    """ Computes the control points of the rational/non-rational spline after degree elevation.

    Implementation of Eq. 5.36 of The NURBS Book by Piegl & Tiller, 2nd Edition, p.205

    Keyword Arguments:
        * ``num``: number of degree elevations

    Please note that degree elevation algorithm can only operate on Bezier shapes, i.e. curves, surfaces, volumes.

    :param degree: degree
    :type degree: int
    :param ctrlpts: control points
    :type ctrlpts: list, tuple
    :return: control points of the degree-elevated shape
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

def link_curves(curves, **kwargs):
    """ Links the input curves together.

    The end control point of the curve k has to be the same with the start control point of the curve k + 1.


    :return: a tuple containing knot vector, control points, weights vector and knots
    """



    kv = []  # new knot vector
    cpts = []  # new control points array
    wgts = []  # new weights array
    kv_connected = []  # superfluous knots to be removed
    pdomain_end = 0

    # Loop though the curves
    for arg in curves:
        control_points_size=arg.control_points.shape[0]
        knot_l=arg.knot.tolist() if isinstance(arg.knot,np.ndarray) else list(arg.knot)
        control_points_l=arg.control_points.tolist()
        weights_l = arg.weights.tolist()
        # Process knot vectors
        if not kv:
            kv += list(knot_l[:-arg.order])  # get rid of the last superfluous knot to maintain split curve notation
            cpts +=list(control_points_l)
            # Process control points
            wgts += list(weights_l)


        else:
            tmp_kv = [pdomain_end + k for k in knot_l[1:-arg.order]]
            kv += tmp_kv
            cpts += list(control_points_l[1:])
            # Process control points

            wgts += list(weights_l[1:])


        pdomain_end += knot_l[-1]
        kv_connected.append(pdomain_end)

    # Fix curve by appending the last knot to the end
    kv += [pdomain_end for _ in range(arg.order)]
    # Remove the last knot from knot insertion list
    kv_connected.pop()

    return kv, cpts, wgts, kv_connected


def degree_elevate_curve(curve: NURBSCurveTuple, num: int = 1):
    """ Applies degree elevation and degree reduction algorithms to spline geometries.

    :param obj: spline geometry
    :type obj: abstract.SplineGeometry
    :param param: operation definition
    :type param: list, tuple
    :return: updated spline geometry
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

    knots, cpts, weights, joints = link_curves(crv_list_new)

    return NURBSCurveTuple(curve.order + num, np.asarray(knots, dtype=float), np.asarray(cpts), np.asarray(weights))


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


def refine_curve(curve:NURBSCurveTuple, new_knots, density:int=0,**kwargs):
    """
    Refine a NURBS curve by inserting new knots.

    Parameters:
    -----------
    curve : NURBSCurve
        The curve to refine
    new_knots : array-like
        New knots to insert (must be sorted)

    Returns:
    --------
    NURBSCurve : Refined curve
    """

    cptsw=to_homogeneous_1d( np.asarray(curve.control_points,dtype=float),np.asarray(curve.weights,dtype=float))
    new_cptsw,new_knots=knot_refinement(curve.order-1, np.asarray(curve.knot,dtype=float).tolist(),cptsw.tolist(),density=density, add_knot_list=np.asarray(new_knots,dtype=float).tolist(),**kwargs)
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
