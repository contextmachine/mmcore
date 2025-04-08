import numpy as np
import math
from mmcore.geom._nurbs_eval import NURBSCurveTuple

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
