import math

import numpy as np

from mmcore.nurbs._nurbs_construct import circle,ruled
from mmcore.nurbs._nurbs_eval import to_homogeneous_1d, from_homogeneous_2d, NURBSSurfaceTuple


def cylinder_surface(radius=1.0, height=1.0,start_angle=0.0, end_angle=2 * math.pi, origin=None, normal=None, xaxis=None, yaxis=None ):

    if origin is None:
        origin = np.array([0.0, 0.0, 0.0])
    else:
        origin = np.array(origin, dtype=float)

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
        normal=np.array([0.,0.,1.0])
    c1=circle(radius, start_angle, end_angle, origin, normal, xaxis, yaxis)
    c2=circle(radius, start_angle, end_angle, origin+normal*height, normal, xaxis, yaxis)
    control_points=np.zeros((c1.control_points.shape[0],2,c1.control_points.shape[1]+1))
    control_points[:, 0, :] = to_homogeneous_1d(c1.control_points, c1.weights)
    control_points[:, 1, :] = to_homogeneous_1d(c2.control_points, c1.weights)
    cpts,weights=from_homogeneous_2d(control_points)
    # Create surface knot vectors
    u_knots = c1.knot  # Same for both curves now
    v_knots = np.array([0., 0., 1., 1.])  # Linear interpolation in v direction

    return NURBSSurfaceTuple( order_u=c1.degree+1,
                              order_v=2,
                              knot_u=u_knots,
                              knot_v=v_knots,
                              control_points=cpts,
                              weights=weights)


def cylinder_surface_2pt(start, end, radius=1.0,start_angle=0.0, end_angle=2 * math.pi):
    end=np.array(end)
    start=np.array(start)
    N=end-start
    h=np.linalg.norm(N)
    N/=h
    return cylinder_surface(radius, h, start_angle,end_angle,start,N)