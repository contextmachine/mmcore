from enum import Enum

import numpy as np
from mmcore.geom.evaluator import surface_evaluator
from mmcore.numeric.vectors import scalar_cross, scalar_unit, scalar_norm, scalar_normalize


def first_fundamental_form(ru, rv):
    """

    :param ru:
    :param rv:
    :return:

    - If F = 0, the parametric lines are orthogonal on the surface.
    - E and G relate to how much the surface stretches in u and v directions.
    - F relates to the shearing of the parametrization.
    """
    E = np.dot(ru, ru)
    F = np.dot(ru, rv)
    G = np.dot(rv, rv)
    return E, F, G

def second_fundamental_form(ruu, ruv, rvv, normal):
    L = np.dot(normal, ruu)
    M = np.dot(normal, ruv)
    N = np.dot(normal, rvv)
    return L, M, N

def gaussian_curvature(E, F, G,L,M,N):
    K = (L * N - M ** 2) / (E * G - F ** 2)
    return K

def mean_curvature(E, F, G,L,M,N):
    H = (E*N + G*L - 2 * F * M) / (2 * (E * G - F ** 2))
    return H


def principal_curvatures(E, F, G, L, M, N):
    """

     Calculates the two principal curvatures using the characteristic equation.

    :param E:
    :param F:
    :param G:
    :param L:
    :param M:
    :param N:
    :return:



    """
    # Calculate coefficients of the characteristic equation
    a = E * G - F ** 2
    b = 2 * M * F - E * N - G * L
    c = L * N - M ** 2

    # Solve the quadratic equation
    discriminant = b ** 2 - 4 * a * c
    k1 = (-b + np.sqrt(discriminant)) / (2 * a)
    k2 = (-b - np.sqrt(discriminant)) / (2 * a)

    return k1, k2


def normal_curvature(E, F, G, L, M, N, du, dv):
    """
    Computes the normal curvature in a given direction (du, dv).

    :param E:
    :param F:
    :param G:
    :param L:
    :param M:
    :param N:
    :param du:
    :param dv:
    :return:


    """
    # Calculate the normal curvature in the direction (du, dv)
    numerator = L * du ** 2 + 2 * M * du * dv + N * dv ** 2
    denominator = E * du ** 2 + 2 * F * du * dv + G * dv ** 2
    return numerator / denominator


def principal_directions(E, F, G, L, M, N, k1, k2):
    """


3. `principal_directions`: Calculates the directions corresponding to the principal curvatures.
    :param E:
    :param F:
    :param G:
    :param L:
    :param M:
    :param N:
    :param k1:
    :param k2:
    :return:
    """
    # Calculate principal directions
    a1 = L - k1 * E
    b1 = M - k1 * F
    a2 = L - k2 * E
    b2 = M - k2 * F

    dir1 = np.array([-b1, a1])
    dir2 = np.array([-b2, a2])

    # Normalize directions
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = dir2 / np.linalg.norm(dir2)

    return dir1, dir2

def euler_curvature(k1, k2, theta):
    """

    :param k1: first principal curvatures
    :param k2: second principal curvatures
    :param theta:
    :return:
    Euler's theorem relates the normal curvature in any direction to the principal curvatures. The formula is:

    κₙ = κ₁cos²θ + κ₂sin²θ

    Where:
    - κₙ is the normal curvature in a direction
    - κ₁ and κ₂ are the principal curvatures
    - θ is the angle between the direction and the first principal direction

    """
    return k1 * np.cos(theta)**2 + k2 * np.sin(theta)**2

def infinitesimal_arc_length(E,F,G, du,dv):
    return np.sqrt(E * du ** 2 + 2 * F * du * dv + G * dv ** 2)


def classify_point(K,H):
    """
    K - Gaussian
    H - Mean
       - Elliptic Point: K > 0 (surface is bowl-shaped)
   - Hyperbolic Point: K < 0 (surface is saddle-shaped)
   - Parabolic Point: K = 0, H ≠ 0 (surface is cylinder-like)
   - Flat Point: K = 0, H = 0 (surface is locally flat)

    :param K: Gaussian Curvature
    :param H: Mean Curvature

    :return:

    """


    if np.isclose(K,0) and np.isclose(H,0):
        return PointOnSurfaceType.FLAT
    elif  ( np.isclose(K,0)) and (not np.isclose(H,0)):
        return PointOnSurfaceType.PARABOLIC
    if K > 0:
        return PointOnSurfaceType.ELLIPTIC
    elif K < 0:
        return PointOnSurfaceType.HYPERBOLIC
    else:
        raise ValueError('Classification fail: K={K},H={H}')

from scipy.integrate import odeint, ode, solve_ivp, simps, simpson


def line_of_curvature(surface, u0, v0, principal_index, t_range):
    def _ode(state, t):
        u, v = state

        pt,du,dv, duu,dvv,duv=surface_evaluator.second_derivatives(getattr(surface,'evaluate_v2',surface), u, v)
        normal=scalar_unit(scalar_cross(du,dv))
        (E, F, G),( L, M, N )= first_fundamental_form(du,dv),second_fundamental_form(duu,duv,dvv,normal)
        k1, k2 = principal_curvatures(E, F, G, L, M, N)
        dir1, dir2 = principal_directions(E, F, G, L, M, N, k1, k2)
        direction = dir1 if principal_index == 0 else dir2
        return direction

    initial_state = [u0, v0]
    solution = odeint(_ode, initial_state, t_range, full_output=True)
    return solution

def surface_distance( surface, u1, v1, u2, v2, num_steps=100):
    # Create a straight line path in parameter space
    u = np.linspace(u1, u2, num_steps)
    v = np.linspace(v1, v2, num_steps)

    # Calculate the length of each step in 3D space
    du = np.diff(u)
    dv = np.diff(v)

    lengths = []
    for i in range(num_steps - 1):

        E, F, G = first_fundamental_form(*surface.derivatives(np.array((u[i], v[i]))))
        dl = infinitesimal_arc_length(E,F,G,du,dv)
        lengths.append(dl)

    # Integrate the lengths
    return np.sum(lengths)


def predict_point(surface, u0, v0, distance, direction, step_size=0.01):
    (E, F, G,L, M, N), ders = fundamental_forms_from_surface(surface, u0, v0)
    k1, k2 = principal_curvatures(E, F, G, L, M, N)
    dir1, dir2 = principal_directions(E, F, G, L, M, N, k1, k2)

    # Convert direction to parameter space
    du, dv = np.cos(direction) * dir1 + np.sin(direction) * dir2

    u, v = u0, v0
    traveled_distance = 0
    while traveled_distance < distance:
        E, F, G = first_fundamental_form(*surface.derivatives(np.array((u, v))))
        step_length = np.sqrt(E * du ** 2 + 2 * F * du * dv + G * dv ** 2)

        if traveled_distance + step_length > distance:
            # Adjust the last step
            factor = (distance - traveled_distance) / step_length
            u += du * step_size * factor
            v += dv * step_size * factor
            break

        u += du * step_size
        v += dv * step_size
        traveled_distance += step_length

    return u, v



class PointOnSurfaceType(int,Enum):
    FLAT=0
    PARABOLIC=1
    ELLIPTIC=2
    HYPERBOLIC=3

def classify_point_dupin(k1,k2):
    if k1 * k2 > 0:
        return PointOnSurfaceType.ELLIPTIC
    elif k1 * k2 < 0:
        return PointOnSurfaceType.HYPERBOLIC
    else:  # Parabolic or Flat point
        # t = np.linspace(-2, 2, num_points)
        if k1 != 0:
            return PointOnSurfaceType.PARABOLIC
        elif k2 != 0:
            return PointOnSurfaceType.FLAT
        else:
            return PointOnSurfaceType.FLAT


def dupin_indicatrix(k1, k2):
    """
    This function calculates the indicatrix curve for a given pair of curvatures, k1 and k2. The indicatrix curve describes the shape of a point on a surface.

    :param k1: The first curvature value
    :param k2: The second curvature value
    :return: A tuple containing two elements - the indicatrix curve function and the type of point on the surface (elliptic, hyperbolic, parabolic, or flat)

    The function first checks if k1 * k2 > 0. If true, it defines a curve function for an elliptic point. The curve function returns an array of x and y coordinates calculated using the given curvatures and the parameter t. The interval of the curve is set to [0, 2*pi]. The curve type is set to elliptic.

    If k1 * k2 < 0, the function defines a curve function for a hyperbolic point. The curve function returns an array of x and y coordinates calculated using the given curvatures and the parameter t. The interval of the curve is set to [-2, 2]. The curve type is set to hyperbolic.

    If neither of the above conditions is true, the function checks if k1 or k2 is zero. If k1 is non-zero, it defines a curve function for a parabolic point. The curve function returns an array of x and y coordinates calculated using the given curvature and the parameter t. The interval of the curve is set to [-2, 2]. The curve type is set to parabolic.

    If k2 is non-zero and k1 is zero, the function defines a curve function for a parabolic point. The curve function returns an array of x and y coordinates calculated using the given curvature and the parameter t. The interval of the curve is set to [-2, 2]. The curve type is set to parabolic.

    If both k1 and k2 are zero, the curve function is set to None and the curve type is set to flat.

    The function returns a tuple containing the curve function and the curve type.
    """
    if np.isclose(k1,0) and np.isclose(k2,0):
        return PointOnSurfaceType.FLAT,None
    if k1 * k2 > 0:  # Elliptic point
        #t = np.linspace(0, 2 * np.pi, num_points)
        def curve(t):
            return np.array([np.sqrt(np.abs(1 / k1)) * np.cos(t),np.sqrt(np.abs(1 / k2)) * np.sin(t)])

        curve.interval = np.array([0, 2*np.pi])
        curve_type = PointOnSurfaceType.ELLIPTIC

    elif k1 * k2 < 0:  # Hyperbolic point
        def curve(t):
            return np.array([np.sqrt(np.abs(1 / k1)) * np.cosh(t), np.sqrt(np.abs(1 / k2)) * np.sinh(t)])
        curve.interval=np.array([-2, 2])
        curve_type = PointOnSurfaceType.HYPERBOLIC
    else:  # Parabolic

        #t = np.linspace(-2, 2, num_points)
        if  not np.isclose(k1,0):
            def curve(t):
                return np.array([t,  np.sign(k1) * t ** 2 / (2 * np.sqrt(np.abs(k1)))])

            curve.interval = np.array([-2, 2])
            curve_type = PointOnSurfaceType.PARABOLIC
        elif not np.isclose(k2, 0):
            def curve(t):
                return np.array([np.sign(k2) * t ** 2 / (2 * np.sqrt(np.abs(k2))),t])

            curve.interval = np.array([-2, 2])
            curve_type = PointOnSurfaceType.PARABOLIC
        else:
            raise ValueError('Classification fail: k1={k1},k2={k2}')
    return curve_type,curve


# User Friendly Functions
# ======================================================================================================================


def fundamental_forms_from_surface(surface, u,v):
    o,du,dv,duu,dvv,duv=surface_evaluator.second_derivatives(surface.evaluate_v2,u,v)
    normal=np.array(scalar_cross(du,dv))
    normal/=scalar_norm(normal)
    return first_fundamental_form(du,dv)+second_fundamental_form(duu,duv,dvv,normal),( o,du,dv,duu,duv,dvv,normal)
def first_fundamental_form_from_surface(surface,u,v):
    return first_fundamental_form(*surface.derivatives(np.array([ u, v])))

def dupin_indicatrix_on_surface(surface,u,v):
    o, du, dv, duu, dvv, duv = surface_evaluator.second_derivatives(surface.evaluate_v2, u, v)

    E, F, G = first_fundamental_form(du, dv)
    n = scalar_cross(du, dv)
    scalar_normalize(n)

    L, M, N = second_fundamental_form(np.array(n), duu, duv, dvv)
    k1, k2 = principal_curvatures(E, F, G, L, M, N)
    return dupin_indicatrix(k1,k2)


def classify_point_on_surface(surface,u,v, use_dupin=False, return_indicatrix=False):
    """
    Classify Point on Surface

    Function to classify a point on a surface using various geometric properties of the surface, such as the first and second fundamental forms, mean curvature, Gaussian curvature, and principal curvatures.

    :param surface: Surface object representing the surface on which the point lies.
    :param u: Parameter `u` specifying the position of the point on the surface in the u-direction.
    :param v: Parameter `v` specifying the position of the point on the surface in the v-direction.
    :param use_dupin: Flag indicating whether to use the Dupin indicatrix for classification. Defaults to False.
    :param return_indicatrix: Flag indicating whether to return the Dupin indicatrix. Only applicable when `use_dupin` is True. Defaults to False.
    :return: Classification result, which can be either a single classification value or the Dupin indicatrix depending on the arguments.

    """
    o,du,dv,duu,dvv,duv=surface_evaluator.second_derivatives(surface.evaluate_v2,u,v)

    E, F, G=first_fundamental_form( du, dv)
    n=scalar_cross(du,dv)
    scalar_normalize(n)

    L,M,N=second_fundamental_form(np.array(n), duu, duv, dvv)

    if use_dupin:
        k1, k2 = principal_curvatures(E, F, G, L, M, N)
        if return_indicatrix:
            return dupin_indicatrix(k1,k2)
        else:
            return classify_point_dupin(k1,k2)

    else:
        H=mean_curvature(E,F,G,L,M,N)
        K=gaussian_curvature(E, F, G, L, M, N)
        return classify_point(K,H)
