import numpy as np
from scipy.integrate import solve_ivp

from mmcore.geom.nurbs import NURBSSurface


def compute_metric(surface, uv):
    r_u = np.array(surface.derivative_u(np.array(uv)))
    r_v = np.array(surface.derivative_v(np.array(uv)))
    E = np.dot(r_u, r_u)
    F = np.dot(r_u, r_v)
    G = np.dot(r_v, r_v)
    return E, F, G


def compute_metric_derivatives(surface:NURBSSurface, uv):
    uv=np.array(uv)
    # Compute first derivatives
    r_u = np.array(surface.derivative_u(uv))
    r_v = np.array(surface.derivative_v(uv))
    # Compute second derivatives
    r_uu = np.array(surface.second_derivative_uu(uv))
    r_uv = np.array(surface.second_derivative_uv(uv))
    r_vv = np.array(surface.second_derivative_vv(uv))

    # Metric components
    #E = np.dot(r_u, r_u)
    #F = np.dot(r_u, r_v)
    #G = np.dot(r_v, r_v)

    # Partial derivatives of metric components
    E_u = 2 * np.dot(r_uu, r_u)
    E_v = 2 * np.dot(r_uv, r_u)
    F_u = np.dot(r_uu, r_v) + np.dot(r_u, r_uv)
    F_v = np.dot(r_uv, r_v) + np.dot(r_u, r_vv)
    G_u = 2 * np.dot(r_uv, r_v)
    G_v = 2 * np.dot(r_vv, r_v)

    return E_u, E_v, F_u, F_v, G_u, G_v


def compute_christoffel_symbols(surface, uv):
    E, F, G = compute_metric(surface, uv)
    E_u, E_v, F_u, F_v, G_u, G_v = compute_metric_derivatives(surface, uv)
    denom = E * G - F ** 2

    # Compute the inverse of the metric tensor
    g_inv = np.array([[G, -F],
                      [-F, E]]) / denom

    # Christoffel symbols:
    # For u equation:
    Gamma_111 = 0.5 * g_inv[0, 0] * (E_u) + 0.5 * g_inv[0, 1] * (E_v)
    Gamma_112 = 0.5 * g_inv[0, 0] * (F_u) + 0.5 * g_inv[0, 1] * (F_v)
    Gamma_122 = 0.5 * g_inv[0, 0] * (G_u) + 0.5 * g_inv[0, 1] * (G_v)

    # For v equation:
    Gamma_211 = 0.5 * g_inv[1, 0] * (E_u) + 0.5 * g_inv[1, 1] * (E_v)
    Gamma_212 = 0.5 * g_inv[1, 0] * (F_u) + 0.5 * g_inv[1, 1] * (F_v)
    Gamma_222 = 0.5 * g_inv[1, 0] * (G_u) + 0.5 * g_inv[1, 1] * (G_v)

    return Gamma_111, Gamma_112, Gamma_122, Gamma_211, Gamma_212, Gamma_222


def geodesic_equations(s, y, surface):
    # y is the state vector [u, v, p, q]
    u, v, p, q = y
    print(s)
    # Compute Christoffel symbols at (u, v)
    Gamma_111, Gamma_112, Gamma_122, Gamma_211, Gamma_212, Gamma_222 = compute_christoffel_symbols(surface, (u, v))

    # Derivatives
    du_ds = p
    dv_ds = q
    dp_ds = -(Gamma_111 * p ** 2 + 2 * Gamma_112 * p * q + Gamma_122 * q ** 2)
    dq_ds = -(Gamma_211 * p ** 2 + 2 * Gamma_212 * p * q + Gamma_222 * q ** 2)
    return [du_ds, dv_ds, dp_ds, dq_ds]


def geodesic_path(surface, u_start, v_start, u_end, v_end, initial_guess=(0.1, 0.1), max_s=1.0, tol=1e-6):
    # We'll solve the ODE from s=0 to s=max_s
    # We'll adjust initial guesses for p and q to match final boundary (u_end, v_end).
    # This is a simplified approach and might require a more robust root-finding method.

    #TODO: The weakest point, which consumes most of the time of the algorithm.
    # We need to find the best algorithm for finding the initial vector for geodesic
    def objective(direction):
        # direction is an array [p0, q0]
        # Solve the ODE with this initial direction
        y0 = [u_start, v_start, direction[0], direction[1]]

        sol = solve_ivp(lambda s, y: geodesic_equations(s, y, surface),
                        [0, max_s], y0, rtol=tol, atol=tol)
        print('\n\nobj:')
        print('sol: ', sol)
        # Evaluate difference at the end
        u_diff = sol.y[0, -1] - u_end
        v_diff = sol.y[1, -1] - v_end
        print('diff: ', u_diff,v_diff)
        return u_diff ** 2 + v_diff ** 2

    # Use an optimization method to find direction that minimizes objective
    from scipy.optimize import minimize

    result = minimize(objective, np.array(initial_guess), method='Nelder-Mead', tol=tol)
    if not result.success:
        raise RuntimeError("Failed to find a suitable initial direction for the geodesic.")
    print(result)
    # Once we have the best direction, solve ODE again for the final path
    best_direction = result.x
    y0 = [u_start, v_start, best_direction[0], best_direction[1]]
    sol = solve_ivp(lambda s, y: geodesic_equations(s, y, surface),
                    [0, max_s], y0, rtol=tol, atol=tol)

    # The solution 'sol' now approximates the geodesic path in parameter space
    # We can convert param space coordinates to 3D coordinates
    param_curve = sol.y[:2].T  # param curve in (u, v)
    xyz_curve = [np.array(surface.evaluate(np.array((u, v)))) for (u, v) in param_curve]
    return xyz_curve,param_curve
