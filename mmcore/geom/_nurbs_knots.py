from __future__ import annotations

from copy import deepcopy

import numpy as np

from mmcore.geom._nurbs_eval import (
    nurbs_interval, _find_span_linear, _join_weights, _copy_curve, _copy_surface,_join_weights_1d,
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
from numpy.typing import NDArray

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
