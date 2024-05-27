import numpy as np

from mmcore.geom.curves.curve import Curve
from mmcore.geom.surfaces import Surface
from mmcore.geom.vec.vec_speedups import scalar_dot,scalar_norm
from numpy.typing import NDArray


# Assuming the following functions are already implemented


def point_inversion_curve(curve: Curve, P: np.ndarray, u0: float, tol1: float, tol2: float,
                          max_iter: int = 100) -> float:
    """
    Perform point inversion on a curve to find parameter u such that C(u) is close to P
    :param curve: Curve to be tested
    :param P: The point to project
    :param u0:  Initial guess of parameter on curve
    :param tol1 = Tolerance for Euclidean distance
    :param tol2 = Tolerance for zero cosine measure
    :param max_iter: Maximum number of iterations

    :return: Parameter u such that C(u) is close to P
    :rtype: float

    # Example usage
    >>> P = np.array([x, y, z])  # The point to project
    >>> u0 = 0.5  # Initial guess
    >>> tol1 = 1e-6  # Tolerance for Euclidean distance
    >>> tol2 = 1e-6  # Tolerance for zero cosine measure
    >>> u = point_inversion_curve(P, u0, tol1, tol2)

    """

    def f(C_point, C_prime) -> float:
        return scalar_dot(C_prime, C_point - P)

    def f_prime(C_point, C_prime, C_double_prime) -> float:

        return scalar_dot(C_double_prime, C_point - P) + scalar_dot(C_prime, C_prime)

    C_point = curve(u0)
    C_prime = curve.derivative(u0)
    C_double_prime = curve.second_derivative(u0)
    ui = u0
    for _ in range(max_iter):

        fi = f(C_point, C_prime)
        fpi = f_prime(C_point, C_prime, C_double_prime)

        if np.isclose(fpi, 0):
            break

        ui1 = ui - fi / fpi

        if abs(ui1 - ui) * scalar_norm(C_prime) <= tol1:
            break
        C_point = curve(ui1)

        if scalar_norm(C_point - P) <= tol1:
            break
        C_prime = curve.derivative(ui1)

        if abs(scalar_dot(C_prime, C_point - P)) / (
                scalar_norm(C_prime) * scalar_norm(C_prime - P)) <= tol2:
            break
        C_double_prime = curve.second_derivative(ui1)
        ui = ui1

    return ui


import numpy as np


# Assuming the following functions are already implemented 0.6741571976269591


def point_inversion_surface(surface: Surface, P: np.ndarray, u0: float, v0: float, tol1: float, tol2: float,
                            max_iter: int = 100) -> (
        float, float):
    """
    Perform point inversion on a surface to find parameters (u, v) such that S(u, v) is close to P


    # Example usage
    P = np.array([x, y, z])  # The point to project
    u0, v0 = 0.5, 0.5  # Initial guess
    tol1 = 1e-6  # Tolerance for Euclidean distance
    tol2 = 1e-6  # Tolerance for zero cosine measure
    u, v = point_inversion_surface(P, u0, v0, tol1, tol2)

    """

    def f(uv: NDArray[float]) -> float:
        return np.dot(surface.derivative_u(uv), surface(uv) - P)

    def g(uv: NDArray[float]) -> float:
        return np.dot(surface.derivative_v(uv), surface(uv) - P)

    ui, vi = u0, v0
    uivi = np.array([u0, v0])
    for _ in range(max_iter):
        f_val = f(uivi)
        g_val = g(uivi)

        # Calculate the Jacobian matrix
        J = np.array([
            [np.dot(surface.derivative_u(uivi), surface.derivative_u(uivi)),
             np.dot(surface.derivative_u(uivi), surface.derivative_v(uivi))],
            [np.dot(surface.derivative_v(uivi), surface.derivative_u(uivi)),
             np.dot(surface.derivative_v(uivi), surface.derivative_v(uivi))]
        ])

        # Calculate the k vector
        k = np.array([
            f_val,
            g_val
        ])
        """
        k = np.array(
            [
                [
                    np.dot(surface_second_derivative_uu(ui, vi), surface(ui, vi) - P)
                    + np.dot(
                        surface_first_derivative_u(ui, vi),
                        surface_first_derivative_u(ui, vi),
                    )
                ],
                [
                    np.dot(surface_second_derivative_uv(ui, vi), surface(ui, vi) - P)
                    + np.dot(
                        surface_first_derivative_v(ui, vi),
                        surface_first_derivative_u(ui, vi),
                    )
                ],
            ]
        )

        """
        # Solve for delta
        delta = np.linalg.solve(J, k)

        # Update u and v
        ui1vi1 = uivi - delta

        # Check convergence criteria
        if np.linalg.norm(ui1vi1 - uivi) * (np.linalg.norm(surface.derivative_u(uivi)) + np.linalg.norm(
                surface.derivative_v(uivi))) <= tol1:
            break

        if np.linalg.norm(surface(ui1vi1) - P) <= tol1:
            break

        if abs(np.dot(surface.derivative_u(ui1vi1), surface(ui1vi1) - P)) / (
                np.linalg.norm(surface.derivative_u(ui1vi1)) * np.linalg.norm(surface(ui1vi1) - P)) <= tol2:
            break

        uivi = ui1vi1

    return uivi
