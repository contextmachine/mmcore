import numpy as np

from mmcore.geom.curves.curve import Curve
from mmcore.geom.surfaces import Surface
from mmcore.numeric.vectors import scalar_dot,scalar_norm,dot,norm,cross
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
        return scalar_dot(surface.derivative_u(uv), surface(uv) - P)

    def g(uv: NDArray[float]) -> float:
        return scalar_dot(surface.derivative_v(uv), surface(uv) - P)

    ui, vi = u0, v0
    uivi = np.array([u0, v0])
    for _ in range(max_iter):
        f_val = f(uivi)
        g_val = g(uivi)

        # Calculate the Jacobian matrix
        J = np.array([
            [scalar_dot(surface.derivative_u(uivi), surface.derivative_u(uivi)),
             scalar_dot(surface.derivative_u(uivi), surface.derivative_v(uivi))],
            [scalar_dot(surface.derivative_v(uivi), surface.derivative_u(uivi)),
             scalar_dot(surface.derivative_v(uivi), surface.derivative_v(uivi))]
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

        if abs(scalar_dot(surface.derivative_u(ui1vi1), surface(ui1vi1) - P)) / (
                np.linalg.norm(surface.derivative_u(ui1vi1)) * np.linalg.norm(surface(ui1vi1) - P)) <= tol2:
            break

        uivi = ui1vi1

    return uivi
_vec_solve    =np.vectorize(np.linalg.solve, signature='(i,i),(i)->(i)'
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
    sdu=np.vectorize(surface.derivative_u,signature='(i)->(j)',cache=True)
    sdv=np.vectorize(surface.derivative_v, signature='(i)->(j)',cache=True)

    def f(uv: NDArray[float]) -> float:
        return np.array(dot(sdu(uv), surface(uv) - P))

    def g(uv: NDArray[float]) -> float:
        return np.array(dot(sdv(uv), surface(uv) - P))


    uivi = np.empty((P.shape[0],2), dtype=float)
    uivi[...,0]=u0
    uivi[..., 1] = v0

    J=np.zeros((P.shape[0], 2,2), dtype=np.float64)
    k=np.zeros((P.shape[0], 2), dtype=np.float64)
    result_uv=np.zeros((P.shape[0],2), dtype=np.float64)
    mask_indices = np.arange(P.shape[0],dtype=int)
    for _ in range(max_iter):

        f_val = f(uivi)
        g_val = g(uivi)
        print(_,f_val,g_val)
        # Calculate the Jacobian matrix
        du = sdu(uivi)
        dv = sdv(uivi)
        du_norm=np.array(norm(du))
        dv_norm = np.array(norm(du))

        dot_du_du,dot_du_dv,dot_dv_dv,dot_dv_du=np.array(dot(du, du)), np.array(dot(du, dv)), np.array(dot(dv, dv)), np.array(dot(dv, du))
        J[..., 0, 0]=dot_du_du
        J[..., 0, 1] = dot_du_dv
        J[..., 1, 0] = dot_dv_du
        J[..., 1, 1] = dot_dv_dv


        # Calculate the k vector
        k[...,0] = f_val
        k[...,1] = g_val








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
        print(_,delta)

        # Update u and v
        ui1vi1 = uivi - delta

        mask=np.array(norm(ui1vi1)*(du_norm + dv_norm))<=tol1


        pt_vec=surface(ui1vi1) - P
        pt_norm=np.array(norm( pt_vec))
        #print(pt_norm
        #      )
        mask2=        pt_norm<=tol1
        dui1=sdu(ui1vi1)
        mask3=abs(np.array(dot(dui1, pt_vec)))/np.array(norm( dui1)*pt_norm)<=tol2
        mask=np.bitwise_or(mask,np.bitwise_or(mask2,mask3))
        mask_inv=np.bitwise_not(mask)
        if np.any(mask):
            print("M",uivi)
        result_uv[mask_indices[mask]] = uivi[mask]

        mask_indices=mask_indices[mask_inv]
        #uivi[mask_inv]=ui1vi1[mask_inv]
        uivi=ui1vi1[mask_inv]
        P=P[mask_inv]
        J=J[mask_inv]
        k = k[mask_inv]

        if len(uivi)==0:
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
