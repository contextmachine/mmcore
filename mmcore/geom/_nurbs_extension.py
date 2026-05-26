
from mmcore.geom._nurbs_eval import NURBSCurveTuple,NURBSSurfaceTuple,evaluate_nurbs_curve,evaluate_nurbs_surface,nurbs_interval,_surface_interval,_curve_interval

import numpy as np


def extend_nurbs_curve(
    curve: NURBSCurveTuple, interval: tuple[float, float], smooth: bool
) -> NURBSCurveTuple:
    """
    Extend a NURBS curve to a new parameter interval.

    Args:
        curve: The NURBS curve to extend
        interval: The new parameter interval (t_min, t_max)
        smooth: If True, maintain smooth continuity; if False, use linear extension

    Returns:
        Extended NURBS curve
    """
    # Get current interval and degree
    degree = curve.order - 1
    current_interval = _curve_interval(curve)
    t_min_current, t_max_current = current_interval
    t_min_new, t_max_new = interval

    # Check if extension is needed
    if t_min_new >= t_min_current and t_max_new <= t_max_current:
        return curve  # No extension needed

    # Work with copies
    knots = list(curve.knot)
    control_points = [np.array(cp) for cp in curve.control_points]
    weights = list(curve.weights)

    # Extension at the start
    if t_min_new < t_min_current:
        extension_length = t_min_current - t_min_new

        if smooth:
            # Evaluate curve and derivatives at start
            eval_data = evaluate_nurbs_curve(curve, t_min_current, d_order=1)
            C0 = eval_data["C"]
            C1 = eval_data["C1"]

            # For smooth extension, we add one new control point
            # This maintains the minimal necessary extension
            new_cp = C0 - C1 * extension_length

            # Insert new control point at the beginning
            control_points.insert(0, new_cp)
            weights.insert(0, weights[0])

            # Update knot vector: remove the first degree+1 knots and add new ones
            knots = [t_min_new] * (degree + 1) + knots[degree:]
            # Add one internal knot to maintain the relationship
            knots.insert(degree + 1, (t_min_new + t_min_current) / 2)

        else:
            # Linear extension
            eval_start = evaluate_nurbs_curve(curve, t_min_current, d_order=1)
            C0 = eval_start["C"]
            C1 = eval_start["C1"]

            # Add one control point for linear extension
            new_cp = C0 - C1 * extension_length

            control_points.insert(0, new_cp)
            weights.insert(0, weights[0])

            # Update knot vector
            knots = [t_min_new] * (degree + 1) + knots[degree:]
            knots.insert(degree + 1, (t_min_new + t_min_current) / 2)

    # Extension at the end
    if t_max_new > t_max_current:
        extension_length = t_max_new - t_max_current

        if smooth:
            # Evaluate curve and derivatives at end
            eval_data = evaluate_nurbs_curve(curve, t_max_current, d_order=1)
            C0 = eval_data["C"]
            C1 = eval_data["C1"]

            # Add one new control point
            new_cp = C0 + C1 * extension_length

            # Append new control point
            control_points.append(new_cp)
            weights.append(weights[-1])

            # Update knot vector: remove the last degree+1 knots and add new ones
            n = len(control_points)
            knots = (
                knots[: -degree - 1]
                + [(t_max_current + t_max_new) / 2]
                + [t_max_new] * (degree + 1)
            )

        else:
            # Linear extension
            eval_end = evaluate_nurbs_curve(curve, t_max_current, d_order=1)
            C0 = eval_end["C"]
            C1 = eval_end["C1"]

            # Add one control point for linear extension
            new_cp = C0 + C1 * extension_length

            control_points.append(new_cp)
            weights.append(weights[-1])

            # Update knot vector
            knots = (
                knots[: -degree - 1]
                + [(t_max_current + t_max_new) / 2]
                + [t_max_new] * (degree + 1)
            )

    # Verify knot vector length
    #expected_knot_length = len(control_points) + curve.order
    #if len(knots) != expected_knot_length:
    #    raise ValueError(
    #        f"Knot vector length mismatch: expected {expected_knot_length}, got {len(knots)}"
    #    )

    # Create extended curve
    return NURBSCurveTuple(
        order=curve.order,
        knot=np.array(knots),
        control_points=np.array(control_points),
        weights=np.array(weights),
    )


def extend_nurbs_surface(
    surface: NURBSSurfaceTuple,
    interval_u: tuple[float, float],
    interval_v: tuple[float, float],
    smooth: bool,
) -> NURBSSurfaceTuple:
    """
    Extend a NURBS surface to new parameter intervals.

    Args:
        surface: The NURBS surface to extend
        interval_u: The new parameter interval in u direction (u_min, u_max)
        interval_v: The new parameter interval in v direction (v_min, v_max)
        smooth: If True, maintain smooth continuity; if False, use linear extension

    Returns:
        Extended NURBS surface
    """
    # Get current intervals and degrees
    degree_u = surface.order_u - 1
    degree_v = surface.order_v - 1
    (u_min_current, u_max_current), (v_min_current, v_max_current) = _surface_interval(
        surface
    )
    u_min_new, u_max_new = interval_u
    v_min_new, v_max_new = interval_v

    # First extend in U direction
    if u_min_new < u_min_current or u_max_new > u_max_current:
        # Create temporary curves for each v parameter
        nu = len(surface.control_points)
        nv = len(surface.control_points[0])

        new_control_points = []
        new_weights = []
        new_knot_u = None

        for j in range(nv):
            # Extract j-th isocurve in u direction
            curve_cps = [surface.control_points[i][j] for i in range(nu)]
            curve_weights = [surface.weights[i, j] for i in range(nu)]

            temp_curve = NURBSCurveTuple(
                order=surface.order_u,
                knot=surface.knot_u,
                control_points=np.array(curve_cps),
                weights=np.array(curve_weights),
            )

            # Extend this curve
            extended_curve = extend_nurbs_curve(
                temp_curve, (u_min_new, u_max_new), smooth
            )

            if new_knot_u is None:
                new_knot_u = extended_curve.knot

            # Store extended control points and weights
            if j == 0:
                for i in range(len(extended_curve.control_points)):
                    new_control_points.append([])
                    new_weights.append([])

            for i in range(len(extended_curve.control_points)):
                new_control_points[i].append(extended_curve.control_points[i])
                new_weights[i].append(extended_curve.weights[i])

        # Update surface with extended u direction
        surface = NURBSSurfaceTuple(
            order_u=surface.order_u,
            order_v=surface.order_v,
            knot_u=new_knot_u,
            knot_v=surface.knot_v,
            control_points=np.array(new_control_points),
            weights=np.array(new_weights),
        )

    # Then extend in V direction
    if v_min_new < v_min_current or v_max_new > v_max_current:
        nu = len(surface.control_points)
        nv = len(surface.control_points[0])

        new_control_points = []
        new_weights = []
        new_knot_v = None

        for i in range(nu):
            # Extract i-th isocurve in v direction
            curve_cps = [surface.control_points[i][j] for j in range(nv)]
            curve_weights = [surface.weights[i, j] for j in range(nv)]

            temp_curve = NURBSCurveTuple(
                order=surface.order_v,
                knot=surface.knot_v,
                control_points=np.array(curve_cps),
                weights=np.array(curve_weights),
            )

            # Extend this curve
            extended_curve = extend_nurbs_curve(
                temp_curve, (v_min_new, v_max_new), smooth
            )

            if new_knot_v is None:
                new_knot_v = extended_curve.knot

            # Store extended control points and weights
            new_control_points.append(extended_curve.control_points)
            new_weights.append(extended_curve.weights)

        # Update surface with extended v direction
        surface = NURBSSurfaceTuple(
            order_u=surface.order_u,
            order_v=surface.order_v,
            knot_u=surface.knot_u,
            knot_v=new_knot_v,
            control_points=np.array(new_control_points),
            weights=np.array(new_weights),
        )

    return surface


if __name__ == '__main__':
    # For example, one can construct a NURBS circle (or any NURBS curve)
    # and then call offset_nurbs_curve. Similarly for surfaces.
    #
    # Here we create a simple planar NURBS curve (for instance a quarter circle)
    # and then compute its offset.
    #
    # NOTE: In a full application these data would be provided by the CAD system.

    # Example NURBS curve: quarter circle in 2D
    order = 3
    knot = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    # Control points and weights for a quarter circle (approximation)
    control_points = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    weights = np.array([1.0, 1 / np.sqrt(2), 1.0], dtype=float)
    nurbs_curve = NURBSCurveTuple(order=order, knot=knot, control_points=control_points, weights=weights)
    # Compute an offset of 0.1 with tolerance 1e-2.
    t0,t1=_curve_interval(nurbs_curve)
    res=extend_nurbs_curve(nurbs_curve,(t0-1.0,t1+1.0),True)
    print(res)

