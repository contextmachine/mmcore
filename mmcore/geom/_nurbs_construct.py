from __future__ import annotations

from collections import namedtuple

import numpy as np
import math


from mmcore.geom._nurbs_eval import NURBSCurveTuple, NURBSSurfaceTuple, from_homogeneous_2d
from mmcore.geom._nurbs_knots import degree_elevate_curve, generate_knots,normalize_knots_curve, refine_curve,_copy_curve,normalize_knots_curve,from_homogeneous_1d,to_homogeneous_1d,knot_refinement,degree_elevation,_bezier_knots,nurbs_interval


def circle(radius=1.0, start_angle=0.0, end_angle=2 * math.pi, center=None, normal=None, xaxis=None, yaxis=None):
    """
    Create a NURBS representation of a circle (or circular arc).

    Parameters:
      radius      : The radius of the circle.
      start_angle : The starting angle in radians.
      end_angle   : The ending angle in radians.
      center      : A 3D point (iterable of 3 numbers) representing the center.
                    Defaults to [0,0,0] if not provided.
      normal      : A 3D vector (iterable of 3 numbers) normal to the plane.
                    Used if xaxis and yaxis are not provided.
      xaxis, yaxis: Two 3D vectors (iterable of 3 numbers) defining the in‐plane axes.
                    If provided, these are used to orient the circle.

    Returns:
      A NURBSCurveTuple representing the circle (or arc).
    """
    # Set default center if not provided
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.array(center, dtype=float)

    # Determine the coordinate system.
    # Option 1: Use provided xaxis and yaxis.
    if (xaxis is not None) and (yaxis is not None):
        xaxis = np.array(xaxis, dtype=float)
        yaxis = np.array(yaxis, dtype=float)
        xaxis = xaxis / np.linalg.norm(xaxis)
        yaxis = yaxis / np.linalg.norm(yaxis)
    # Option 2: Use the provided normal vector.
    elif normal is not None:
        normal = np.array(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        # Choose an arbitrary vector that is not parallel to the normal.
        if abs(normal[0]) < 1e-6 and abs(normal[1]) < 1e-6:
            # If normal is (nearly) vertical, choose xaxis = (1,0,0)
            xaxis = np.array([1.0, 0.0, 0.0])
        else:
            # Otherwise, use the cross product with a reference vector (here z-axis).
            xaxis = np.cross([0, 0, 1], normal)
            if np.linalg.norm(xaxis) < 1e-6:
                xaxis = np.array([1.0, 0.0, 0.0])
            else:
                xaxis = xaxis / np.linalg.norm(xaxis)
        # yaxis is then given by normal x xaxis
        yaxis = np.cross(normal, xaxis)
        yaxis = yaxis / np.linalg.norm(yaxis)
    # Option 3: Default to the xy-plane.
    else:
        xaxis = np.array([1.0, 0.0, 0.0])
        yaxis = np.array([0.0, 1.0, 0.0])

    # Ensure that end_angle > start_angle.
    delta = end_angle - start_angle
    if delta <= 0:
        raise ValueError("end_angle must be greater than start_angle")

    # Divide the arc into segments of at most 90 degrees (pi/2 radians)
    n_seg = int(math.ceil(delta / (math.pi / 2)))
    seg_angle = delta / n_seg

    # Build lists to hold the control points (in local 2D coordinates) and weights.
    # For a quadratic (degree-2) NURBS arc segment, three control points are used.
    # When joining segments, the endpoint of one is the start point of the next.
    control_points_local = []
    weights = []

    for i in range(n_seg):
        theta0 = start_angle + i * seg_angle
        theta2 = start_angle + (i + 1) * seg_angle
        theta_mid = (theta0 + theta2) / 2.0
        # Weight for the middle control point.
        w_mid = math.cos(seg_angle / 2)

        # Compute the control points in 2D (local coordinates).
        P0 = np.array([radius * math.cos(theta0), radius * math.sin(theta0)])
        # For the middle control point, scale so that it is off the circle
        # by a factor of 1/cos(seg_angle/2)
        P1 = np.array([radius / w_mid * math.cos(theta_mid), radius / w_mid * math.sin(theta_mid)])
        P2 = np.array([radius * math.cos(theta2), radius * math.sin(theta2)])

        if i == 0:
            control_points_local.append(P0)
            control_points_local.append(P1)
            control_points_local.append(P2)
            weights.append(1.0)
            weights.append(w_mid)
            weights.append(1.0)
        else:
            # Avoid duplicating the shared control point between segments.
            control_points_local.append(P1)
            control_points_local.append(P2)
            weights.append(w_mid)
            weights.append(1.0)

    # Number of control points for a piecewise quadratic curve.
    n_cp = len(control_points_local)
    # For a degree-2 curve, the knot vector has length: n_cp + 3.
    # Construct the knot vector in clamped form:
    # [0, 0, 0, u1, u1, u2, u2, ..., 1, 1, 1] with internal knots equally spaced.
    knot_vector = [0, 0, 0]
    for i in range(1, n_seg):
        u = i / n_seg
        knot_vector.extend([u, u])
    knot_vector.extend([1, 1, 1])

    # Transform the local 2D control points to 3D using the coordinate system.
    control_points_global = []
    for pt in control_points_local:
        # Map a local point (a, b) to 3D: center + a*xaxis + b*yaxis.
        global_pt = center + pt[0] * xaxis + pt[1] * yaxis
        control_points_global.append(global_pt)

    # Convert lists to numpy arrays.
    control_points_global = np.array(control_points_global)
    weights = np.array(weights)
    knot_vector = np.array(knot_vector, dtype=float)

    # Return the NURBS curve tuple.
    # Here, we assume NURBSCurveTuple takes (dimension, knot_vector, control_points, weights).
    return NURBSCurveTuple(3, knot_vector, control_points_global, weights)







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

    p1, p2 = curve1.order, curve2.order
    if p1 < p2:
        curve1 = degree_elevate_curve(curve1, p2 - p1)



    elif p2 < p1:
        curve2 = degree_elevate_curve(curve2, p1 - p2)

    curve1=normalize_knots_curve(curve1)
    curve2=normalize_knots_curve(curve2)



    curve1_r=refine_curve(curve1, curve2.knot, 0)
    curve2_r=refine_curve(curve2, curve1_r.knot, 0)


    return curve1_r, curve2_r


def make_curves_compatible_multiple(curves):
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
    max_order=0

    curves=list(curves)

    for i in range(len(curves)):
        curve=curves[i]
        curves[i]=curve=normalize_knots_curve(curve)



        if curve.order>max_order:
            max_order=curve.order

    for i in range(len(curves)):

        curve=curves[i]
        num=max_order-curve.order
        if num>0:
            curve=degree_elevate_curve(curve, num)
            #print('nn',curve.knot.tolist())
        curve = normalize_knots_curve(curve)
        curves[i]=curve


            #new_cpts, new_weights = from_homogeneous_1d(np.asarray(new_cptsw))

    for i in  range(len(curves)):
        curve=curves[i]

        for j in range(len(curves)):
            if i==j:
                continue
            curve_j=curves[j]
            if len(curve_j.knot)==len(curve.knot) and np.allclose(curve_j.knot,curve.knot):
                continue
            try:
                curves[i]=curve=refine_curve(curve,curve_j.knot,density=0)
            except Exception as err:
                print(curve.knot.tolist(),curve_j.knot.tolist())
                raise err

    return curves

def ruled(curve1:NURBSCurveTuple, curve2:NURBSCurveTuple)->NURBSSurfaceTuple:
    """
    Generates a ruled surface between two given NURBS curves. A ruled surface is a
    surface created by linear interpolation between corresponding points on two
    curves. This function assumes that the input curves are NURBS curves and processes
    them to make them compatible before producing the NURBS surface. If the input
    curves have different knot vectors or control points, they will be modified to
    produce a valid ruled surface.

    :param curve1: The first input NURBS curve.
    :type curve1: NURBSCurveTuple
    :param curve2: The second input NURBS curve.
    :type curve2: NURBSCurveTuple

    :return: A NURBS ruled surface created between the two input curves.
    :rtype: NURBSSurfaceTuple
    """
    # Make curves compatible
    curve1=normalize_knots_curve(curve1)
    curve2=normalize_knots_curve(curve2)

    c1, c2 = make_curves_compatible(curve1, curve2)

    # Create surface control points
    n = len(c1.control_points)
    control_points = np.zeros((n, 2, 4))  # nx2x4 array


    # Fill control points

    for i in range(n):
        control_points[i, 0, :] = np.array(c1._control_points)[i]
        control_points[i, 1, :] = np.array(c2._control_points)[i]

    # Create surface knot vectors
    u_knots = c1.knots  # Same for both curves now
    v_knots = np.array([0., 0., 1., 1.])  # Linear interpolation in v direction

    return NURBSSurfaceTuple( order_u=c1.degree+1,
                              order_v=2,
                              knot_u=u_knots,
                              knot_v=v_knots,
                              control_points=np.ascontiguousarray(control_points[...,:-1]),
                              weights=np.ascontiguousarray(control_points[...,-1]))
from mmcore.geom._nurbs_interp import interpolate_curve,fair_interpolate_curve

from typing import NamedTuple,Literal
from enum import Enum,auto
LoftType=Literal['normal', 'loose','straight']


class LoftOptions(NamedTuple):
    loft_type:LoftType
def default_loft_options()->LoftOptions:
    return LoftOptions(LoftType.NORMAL)


def loft(curves:list[NURBSCurveTuple], loft_type:LoftType='normal', **kwargs)->NURBSSurfaceTuple:
    """
    Generates a ruled surface between two given NURBS curves. A ruled surface is a
    surface created by linear interpolation between corresponding points on two
    curves. This function assumes that the input curves are NURBS curves and processes
    them to make them compatible before producing the NURBS surface. If the input
    curves have different knot vectors or control points, they will be modified to
    produce a valid ruled surface.

    :param curves: list of curves
    :param loft_type: Loft type



    :return: A NURBS ruled surface created between the two input curves.
    :rtype: NURBSSurfaceTuple
    """
    # Make curves compatible

    compat_curves=make_curves_compatible_multiple(curves)
    u_count=compat_curves[0].control_points.shape[0]
    degree_u=compat_curves[0].order-1
    knots_u=compat_curves[0].knot
    grid_cptsw=np.zeros((u_count,len(curves),4))
    surf_cptsw=[None for i in range(u_count)]
    print(loft_type)
    if loft_type == 'straight':
        kn=generate_knots(len(curves), 1)
        knots_v = [kn for i in range(u_count)]
        degree_v = 1
    else:
        knots_v = [None for i in range(u_count)]
        degree_v=3

    for i in  range(len(curves)):
        crv = compat_curves[i]
        grid_cptsw[:, i, :]=to_homogeneous_1d(crv.control_points,crv.weights)

    for i in range(u_count):


        if loft_type =="normal":


            surf_cptsw[i] ,            knots_v[i] =fair_interpolate_curve(grid_cptsw[i, :, :], degree_v, lambda_reg=0.0001)
        elif loft_type == "loose":

            knots_v[i] = generate_knots(grid_cptsw[i, :, :].shape[0], degree_v)
            surf_cptsw[i] = grid_cptsw[i, :, :]
        elif loft_type ==  'straight':
            knots_v[i] =np.array([0.0,0.0, 1.,1.])
            surf_cptsw[i]=grid_cptsw[i,:,:]



        else:
            raise TypeError(f"Unknown loft type: {loft_type}")

    print(knots_v)
    return NURBSSurfaceTuple(degree_u+1,degree_v+1,knots_u,np.asarray(knots_v[0]),*from_homogeneous_2d(surf_cptsw))
