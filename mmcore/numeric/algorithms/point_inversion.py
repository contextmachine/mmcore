import numpy as np


from mmcore.geom.curves.curve import Curve
from mmcore.geom.surfaces import Surface
from mmcore.numeric.vectors import scalar_dot, scalar_norm, dot, norm, cross, solve2x2
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
    Perform point inversion on a surface to find parameters (u, v) such that P - S(u, v) will be perpendicular to the surface.

    :param surface: The surface object to perform point inversion on.
    :param P: The point to be inverted.
    :param u0: The initial u parameter value.
    :param v0: The initial v parameter value.
    :param tol1: The tolerance for convergence criteria 1.
    :param tol2: The tolerance for convergence criteria 2.
    :param max_iter: The maximum number of iterations to perform. Defaults to 100.
    :return: The u and v parameter values of the inverted point.

    This method performs point inversion on a given surface object. It iteratively updates the u and v parameter values based on convergence criteria until a sufficient solution is found. The surface object should have methods to compute the surface's derivatives and evaluate the surface at a given parameter.

    The convergence criteria are defined as follows:

        1. If the change in u and v parameter values is below the tolerance given by tol1 multiplied by the sum of the norms of the surface's u and v derivatives at the current parameter values, the iteration is considered converged.
        2. If the distance between the inverted point and the surface evaluated at the updated parameter values is below tol1, the iteration is considered converged.
        3. If the absolute value of the dot product between the surface's u derivative and the vector connecting the inverted point and the surface evaluated at the updated parameter values, divided by the product of the norms of the surface's u derivative and the vector, is below tol2, the iteration is considered converged.




    Important
    ----
    Since the inversion of a point on the surface may not be unique, the algorithm finds the closest solution to u0,v0.
    This does not ensure that the solution is the closest inversion to point P,
    and even less does it ensure that the solution is the closest point on the surface.
    To find the closest point, see:
        mmcore.numeric.closest_point.closest_point_on_surface

    # Example usage
    P = np.array([x, y, z])  # The point to project
    u0, v0 = 0.5, 0.5  # Initial guess
    tol1 = 1e-6  # Tolerance for Euclidean distance
    tol2 = 1e-6  # Tolerance for zero cosine measure
    u, v = point_inversion_surface(P, u0, v0, tol1, tol2)

    The method returns the final u and v parameter values of the inverted point.
    """

    def f(uv: NDArray[float]) -> float:
        return scalar_dot(surface.derivative_u(uv), surface.evaluate(uv) - P)

    def g(uv: NDArray[float]) -> float:
        return scalar_dot(surface.derivative_v(uv), surface.evaluate(uv) - P)

    ui, vi = u0, v0
    uivi = np.array([u0, v0])
    delta=  np.zeros(2)
    for _ in range(max_iter):
        f_val = f(uivi)
        g_val = g(uivi)
        du=surface.derivative_u(uivi)
        dv=surface.derivative_v(uivi)
        # Calculate the Jacobian matrix
        J = np.array([
            [scalar_dot(du, du),
             scalar_dot(du, dv)],
            [scalar_dot(dv, du),
             scalar_dot(dv, dv)]
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

        success = solve2x2(J, k, delta)
        if success==0:
            raise ValueError(f'Failure to compute a matrix system of equations A={J} b={k}')
        # Update u and v
        ui1vi1 = uivi - delta

        # Check convergence criteria
        if np.linalg.norm(ui1vi1 - uivi) * (np.linalg.norm(du) + np.linalg.norm(
                dv)) <= tol1:
            break
        new_pt=surface.evaluate(ui1vi1)
        if scalar_norm( new_pt- P) <= tol1:
            break
        new_du=surface.derivative_u(ui1vi1)

        pt_diff= new_pt-P
        if abs(scalar_dot(new_du, pt_diff)) / (
                np.linalg.norm(new_du) *  np.linalg.norm(pt_diff)) <= tol2:
            break

        uivi = ui1vi1

    return uivi


_vec_solve = np.vectorize(np.linalg.solve, signature='(i,i),(i)->(i)'
                          )


def points_inversion_surface(surface: Surface, P: np.ndarray, u0: float, v0: float, tol1: float, tol2: float,
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
    sdu = np.vectorize(surface.derivative_u, signature='(i)->(j)', cache=True)
    sdv = np.vectorize(surface.derivative_v, signature='(i)->(j)', cache=True)

    def f(uv: NDArray[float]) -> float:
        return np.array(dot(sdu(uv), surface(uv) - P))

    def g(uv: NDArray[float]) -> float:
        return np.array(dot(sdv(uv), surface(uv) - P))

    uivi = np.empty((P.shape[0], 2), dtype=float)
    uivi[..., 0] = u0
    uivi[..., 1] = v0

    J = np.zeros((P.shape[0], 2, 2), dtype=np.float64)
    k = np.zeros((P.shape[0], 2), dtype=np.float64)
    result_uv = np.zeros((P.shape[0], 2), dtype=np.float64)
    mask_indices = np.arange(P.shape[0], dtype=int)
    for _ in range(max_iter):

        f_val = f(uivi)
        g_val = g(uivi)
        print(_, f_val, g_val)
        # Calculate the Jacobian matrix
        du = sdu(uivi)
        dv = sdv(uivi)
        du_norm = np.array(norm(du))
        dv_norm = np.array(norm(du))

        dot_du_du, dot_du_dv, dot_dv_dv, dot_dv_du = np.array(dot(du, du)), np.array(dot(du, dv)), np.array(
            dot(dv, dv)), np.array(dot(dv, du))
        J[..., 0, 0] = dot_du_du
        J[..., 0, 1] = dot_du_dv
        J[..., 1, 0] = dot_dv_du
        J[..., 1, 1] = dot_dv_dv

        # Calculate the k vector
        k[..., 0] = f_val
        k[..., 1] = g_val

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

        delta = _vec_solve(J, k)
        print(_, delta)

        # Update u and v
        ui1vi1 = uivi - delta

        mask = np.array(norm(ui1vi1) * (du_norm + dv_norm)) <= tol1

        pt_vec = surface(ui1vi1) - P
        pt_norm = np.array(norm(pt_vec))
        #print(pt_norm
        #      )
        mask2 = pt_norm <= tol1
        dui1 = sdu(ui1vi1)
        mask3 = abs(np.array(dot(dui1, pt_vec))) / np.array(norm(dui1) * pt_norm) <= tol2
        mask = np.bitwise_or(mask, np.bitwise_or(mask2, mask3))
        mask_inv = np.bitwise_not(mask)
        if np.any(mask):
            print("M", uivi)
        result_uv[mask_indices[mask]] = uivi[mask]

        mask_indices = mask_indices[mask_inv]
        #uivi[mask_inv]=ui1vi1[mask_inv]
        uivi = ui1vi1[mask_inv]
        P = P[mask_inv]
        J = J[mask_inv]
        k = k[mask_inv]

        if len(uivi) == 0:
            break
        # Check convergence criteria
        #if np.linalg.norm(ui1vi1 - uivi,axis=1) * (np.linalg.norm(surface.derivative_u(uivi)) + np.linalg.norm(
        #        surface.derivative_v(uivi))) <= tol1:
        #
        #    break
        #
        #if np.linalg.norm(surface(ui1vi1) - P) <= tol1:
        #    break
        #
        #if abs(scalar_dot(surface.derivative_u(ui1vi1), surface(ui1vi1) - P)) / (
        #        np.linalg.norm(surface.derivative_u(ui1vi1)) * np.linalg.norm(surface(ui1vi1) - P)) <= tol2:
        #    break

        #uivi = ui1vi1

    return result_uv

