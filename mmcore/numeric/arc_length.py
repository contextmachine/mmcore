import numpy as np
import math

from mmcore.numeric.vectors import scalar_norm,norm


def curvature_based_step(tolerance, curvature_radius):
    return 2 * np.sqrt(2 * curvature_radius * tolerance - tolerance ** 2)


def arc_height(chord_length, curvature_radius):
    """
    ___
    :param chord_length:
    :param curvature_radius:
    :return:
    """
    return curvature_radius - math.sqrt((curvature_radius - chord_length / 2) * (curvature_radius + chord_length / 2))


def step(crv, t, tol):
    K = crv.curvature(t)
    r = 1 / np.linalg.norm(K)
    return np.sqrt(r ** 2 - (r - tol) ** 2) * 2


def parametric_arc_length(func, t_start, t_end, dt=1e-3):
    # Generate a list of t values from t_start to t_end
    t_values = np.arange(t_start, t_end+dt, dt)
    num_points = len(t_values)-1

    # Calculate the derivatives using finite differences
    print(dt)
    arc_length = 0.
    for i in range(num_points):
        derivative = (np.array(func(t_values[i + 1])) - np.array(func(t_values[i]))) / dt
        # It is similar by each component
        # dx_dt = (x_t(t_values[i + 1]) - x_t(t_values[i])) / dt
        # dy_dt = (y_t(t_values[i + 1]) - y_t(t_values[i])) / dt
        # ...

        # Calculate the integrand sqrt((dx/dt)^2 + (dy/dt)^2)
        integrand = scalar_norm(derivative)
        # Use the trapezoidal rule to approximate the integral
        if i == 0 or i == num_points - 1:
            integrand *= 0.5
        arc_length += integrand

    arc_length *= dt
    return arc_length


