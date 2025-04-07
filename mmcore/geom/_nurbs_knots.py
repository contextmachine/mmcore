from __future__ import annotations

from copy import deepcopy

import numpy as np

from mmcore.geom._nurbs_eval import (
    nurbs_interval, _find_span_linear, _join_weights, _copy_curve, _join_weights_1d, 
    to_homogeneous_1d, from_homogeneous_1d, to_homogeneous_2d, from_homogeneous_2d,
    NURBSCurveTuple, BSplineCurveTuple, NURBSSurfaceTuple, BSplineSurfaceTuple
)


def knot_insertion_alpha( u,  knotvector,  span,  idx, leg):
    return (u - knotvector[leg + idx]) / (knotvector[idx + span + 1] - knotvector[leg + idx])


def knot_removal_alpha_i( u, degree,  knotvector,  num, idx) :
    return (u - knotvector[idx]) / (knotvector[idx + degree + 1 + num] - knotvector[idx])


def knot_removal_alpha_j(u,  degree, knotvector, num, idx) :
    return (u - knotvector[idx - num]) / (knotvector[idx + degree + 1] - knotvector[idx - num])

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


def knot_insertion_alpknot_removalha(u, knotvector, span, idx, leg):
    """ Computes :math:`\\alpha` coefficient for knot insertion algorithm.

    :param u: knot
    :type u: float
    :param knotvector: knot vector
    :type knotvector: tuple
    :param span: knot span
    :type span: int
    :param idx: index value (degree-dependent)
    :type idx: int
    :param leg: i-th leg of the control points polygon
    :type leg: int
    :return: coefficient value
    :rtype: float
    """
    return (u - knotvector[leg + idx]) / (knotvector[idx + span + 1] - knotvector[leg + idx])




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
        new_control_points_xyz, weights = from_homogeneous_1d(new_control_points)
        return NURBSCurveTuple(curve.order, knot=np.array(kv_new), 
                              control_points=new_control_points_xyz, 
                              weights=weights)
    
    return BSplineCurveTuple(curve.order, knot=np.array(kv_new), control_points=np.array(new_control_points))



def split_curve(curve:BSplineCurveTuple|NURBSCurveTuple, t:float, **kwargs):
    """ Splits the curve at the input parametric coordinate.

    This method splits the curve into two pieces at the given parametric coordinate, generates two different
    curve objects and returns them. It does not modify the input curve.

    Keyword Arguments:
        * ``find_span_func``: FindSpan implementation. *Default:* :func:`.helpers.find_span_linear`
        * ``insert_knot_func``: knot insertion algorithm implementation. *Default:* :func:`.operations.insert_knot`

    :param obj: Curve to be split
    :type obj: abstract.Curve
    :param param: parameter
    :type param: float
    :return: a list of curve segments
    :rtype: list
    """
    # Validate input

    interval=nurbs_interval(curve.knot,curve.order-1)
    if t == interval[0] or t == interval[1]:
        raise ValueError(f"Parameter t: {t} Cannot split from the domain edge: {interval}")
    if not (interval[0]<t< interval[1]):
        raise ValueError(f"Parameter t: {t} is outside the domain: {interval}")

    # Keyword arguments
    span_func = _find_span_linear

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
    knot_span = span_func(temp_obj.order-1, temp_obj.knot, len(temp_obj.control_points), t) + 1
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


def insert_knot_surface_u(surface:BSplineSurfaceTuple|NURBSSurfaceTuple, u:float, num:int=1):
    """Insert a knot into a surface in the u-direction multiple times.
    
    Args:
        surface: The surface to modify
        u: The knot value to insert
        num: Number of times to insert the knot (default: 1)
        
    Returns:
        A new surface with the knot inserted in the u-direction
    """
    rational = isinstance(surface, NURBSSurfaceTuple)
    knots_u = np.array(surface.knot_u).tolist()
    degree_u = surface.order_u - 1
    span_u = _find_span_linear(degree_u, knots_u, surface.control_points.shape[0], u)
    
    kv_new_u = knot_insertion_kv(knots_u, u, span_u, num)
    
    # For each row of control points in the v-direction, perform knot insertion in u-direction
    num_rows = surface.control_points.shape[0]
    num_cols = surface.control_points.shape[1]
    new_num_cols = num_cols + num  # New number of control points in u-direction
    
    if rational:
        # Initialize arrays for new control points and weights
        new_control_points = np.zeros((num_rows, new_num_cols, surface.control_points.shape[2]))
        new_weights = np.zeros((num_rows, new_num_cols))
        
        for i in range(num_rows):
            # Extract the i-th row of control points
            row_control_points = surface.control_points[i, :, :]
            row_weights = surface.weights[i, :]
            
            # Convert to homogeneous coordinates
            row_homo = to_homogeneous_1d(row_control_points, row_weights)
            row_homo_list = row_homo.tolist()
            
            # Apply knot insertion
            new_row_homo_list = knot_insertion(degree_u, knots_u, row_homo_list, u, num=num)
            
            # Convert back from homogeneous
            new_row_cp, new_row_w = from_homogeneous_1d(new_row_homo_list)
            
            # Store in the new arrays
            new_control_points[i, :, :] = new_row_cp
            new_weights[i, :] = new_row_w
            
        return NURBSSurfaceTuple(
            order_u=surface.order_u, 
            order_v=surface.order_v, 
            knot_u=np.array(kv_new_u), 
            knot_v=surface.knot_v, 
            control_points=new_control_points, 
            weights=new_weights
        )
    else:
        # For non-rational case, we can work directly with control points
        new_control_points = np.zeros((num_rows, new_num_cols, surface.control_points.shape[2]))
        
        for i in range(num_rows):
            # Extract the i-th row of control points
            row_control_points = surface.control_points[i, :, :]
            row_list = row_control_points.tolist()
            
            # Apply knot insertion
            new_row_list = knot_insertion(degree_u, knots_u, row_list, u, num=num)
            
            # Store in the new array
            new_control_points[i, :, :] = np.array(new_row_list)
            
        return BSplineSurfaceTuple(
            order_u=surface.order_u, 
            order_v=surface.order_v, 
            knot_u=np.array(kv_new_u), 
            knot_v=surface.knot_v, 
            control_points=new_control_points
        )


def insert_knot_surface_v(surface:BSplineSurfaceTuple|NURBSSurfaceTuple, v:float, num:int=1):
    """Insert a knot into a surface in the v-direction multiple times.
    
    Args:
        surface: The surface to modify
        v: The knot value to insert
        num: Number of times to insert the knot (default: 1)
        
    Returns:
        A new surface with the knot inserted in the v-direction
    """
    rational = isinstance(surface, NURBSSurfaceTuple)
    knots_v = np.array(surface.knot_v).tolist()
    degree_v = surface.order_v - 1
    span_v = _find_span_linear(degree_v, knots_v, surface.control_points.shape[1], v)
    
    kv_new_v = knot_insertion_kv(knots_v, v, span_v, num)
    
    # For each column of control points in the u-direction, perform knot insertion in v-direction
    num_rows = surface.control_points.shape[0]
    num_cols = surface.control_points.shape[1]
    new_num_rows = num_rows + num  # New number of control points in v-direction
    
    if rational:
        # Initialize arrays for new control points and weights
        new_control_points = np.zeros((new_num_rows, num_cols, surface.control_points.shape[2]))
        new_weights = np.zeros((new_num_rows, num_cols))
        
        for j in range(num_cols):
            # Extract the j-th column of control points
            col_control_points = surface.control_points[:, j, :]
            col_weights = surface.weights[:, j]
            
            # Convert to homogeneous coordinates
            col_homo = to_homogeneous_1d(col_control_points, col_weights)
            col_homo_list = col_homo.tolist()
            
            # Apply knot insertion
            new_col_homo_list = knot_insertion(degree_v, knots_v, col_homo_list, v, num=num)
            
            # Convert back from homogeneous
            new_col_cp, new_col_w = from_homogeneous_1d(new_col_homo_list)
            
            # Store in the new arrays
            new_control_points[:, j, :] = new_col_cp
            new_weights[:, j] = new_col_w
            
        return NURBSSurfaceTuple(
            order_u=surface.order_u, 
            order_v=surface.order_v, 
            knot_u=surface.knot_u, 
            knot_v=np.array(kv_new_v), 
            control_points=new_control_points, 
            weights=new_weights
        )
    else:
        # For non-rational case, we can work directly with control points
        new_control_points = np.zeros((new_num_rows, num_cols, surface.control_points.shape[2]))
        
        for j in range(num_cols):
            # Extract the j-th column of control points
            col_control_points = surface.control_points[:, j, :]
            col_list = col_control_points.tolist()
            
            # Apply knot insertion
            new_col_list = knot_insertion(degree_v, knots_v, col_list, v, num=num)
            
            # Store in the new array
            new_control_points[:, j, :] = np.array(new_col_list)
            
        return BSplineSurfaceTuple(
            order_u=surface.order_u, 
            order_v=surface.order_v, 
            knot_u=surface.knot_u, 
            knot_v=np.array(kv_new_v), 
            control_points=new_control_points
        )