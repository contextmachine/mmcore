import numpy as np
import warnings

from mmcore.numeric.newthon.cnewthon import hessian
def newtons_method22(F, J_F, initial_point, tol=1e-5, max_iter=100):
    """
    Apply Newton's method to solve F(point) = 0.

    Parameters:
    - F: Function that computes the system of equations.
    - J_F: Function that computes the Jacobian matrix of F.
    - initial_point: Initial guess for the variables (including lambda).
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.
    """
    point = np.asarray(initial_point, dtype=float)
    for _ in range(max_iter):
        F_val = F(point)
        JF_val = J_F(point)
        try:
            delta = np.linalg.solve(JF_val, -F_val)
        except np.linalg.LinAlgError:
            warnings.warn(f"Jacobian is singular at the point {point}")
            break
        new_point = point + delta
        if np.linalg.norm(new_point - point) < tol:
            return new_point
        point = new_point
    warnings.warn(f"Iteration limit {max_iter} reached at {point}")
    return point
from mmcore._test_data import ssx
s1,s2=ssx[2]
def f(x):
    u, v, s, t = x
    p = s1.evaluate_v2(u, v)
    q = s2.evaluate_v2(s, t)
    return np.linalg.norm(p - q)**2

def h(x):
    u, v, s, t = x
    p = s1.evaluate_v2(u, v)
    q = s2.evaluate_v2(s, t)
    m = (p + q) / 2  # Midpoint
    return a * m[0] + b * m[1] + c * m[2] + d


def grad_f(x):
    u, v, s, t = x
    # Compute the gradient of f with respect to u, v, s, t.
    # This requires the derivatives of s1 and s2 with respect to their parameters.
    # Let's assume you've defined functions that compute these derivatives:
    dp_du = s1.derivative_u(np.array([u, v]))
    dp_dv = s1.derivative_v(np.array([u, v]))
    dq_ds = s2.derivative_u(np.array([s, t]))
    dq_dt = s2.derivative_v(np.array([s, t]))
    p = s1.evaluate_v2(u, v)
    q = s2.evaluate_v2(s, t)
    diff = p - q
    grad_u = 2 * np.dot(diff, dp_du)
    grad_v = 2 * np.dot(diff, dp_dv)
    grad_s = -2 * np.dot(diff, dq_ds)
    grad_t = -2 * np.dot(diff, dq_dt)
    return np.array([grad_u, grad_v, grad_s, grad_t])
def grad_h(x):
    u, v, s, t = x
    dp_du = s1.derivative_u(np.array([u, v]))
    dp_dv  = s1.derivative_v(np.array([u, v]))
    dq_ds = s2.derivative_u(np.array([s, t]))
    dq_dt = s2.derivative_v(np.array([s, t]))
    # Gradient of m w.r.t variables
    dm_du = dp_du / 2
    dm_dv = dp_dv / 2
    dm_ds = dq_ds / 2
    dm_dt = dq_dt / 2
    # Gradient of h w.r.t variables
    grad_u = a * dm_du[0] + b * dm_du[1] + c * dm_du[2]
    grad_v = a * dm_dv[0] + b * dm_dv[1] + c * dm_dv[2]
    grad_s = a * dm_ds[0] + b * dm_ds[1] + c * dm_ds[2]
    grad_t = a * dm_dt[0] + b * dm_dt[1] + c * dm_dt[2]
    return np.array([grad_u, grad_v, grad_s, grad_t])
def F(point):
    x = point[:-1]  # variables u, v, s, t
    lambd = point[-1]  # Lagrange multiplier
    grad_f_val = grad_f(x)
    grad_h_val = grad_h(x)
    h_val = h(x)
    F_val = np.concatenate([grad_f_val - lambd * grad_h_val, [-h_val]])
    return F_val

def J_F(point):
    x = point[:-1]
    lambd = point[-1]
    # Compute Hessians
    H_f = hessian(f, x)  # Hessian of f w.r.t x
    H_h = hessian(h, x)   # Hessian of h w.r.t x (often zero if h is linear)
    grad_h_val = grad_h(x)
    # Build the Jacobian matrix
    n = len(x)
    JF = np.zeros((n+1, n+1))
    # Top-left block: Hessian of the Lagrangian w.r.t x
    JF[:n, :n] = H_f - lambd * H_h
    # Top-right block: -grad_h^T
    JF[:n, n] = -grad_h_val
    # Bottom-left block: -grad_h
    JF[n, :n] = -grad_h_val
    # Bottom-right corner is zero
    return JF


a, b, c, d = 1., 2., 3., 4.  # Example plane coefficients

initial_point = np.array([0.5,0.5,0.5,0.5, 0.0])

solution = newtons_method22(F, J_F, initial_point)

u_sol, v_sol, s_sol, t_sol, lambda_sol = solution

import numpy as np
import warnings

import numpy as np
import warnings


def sqp_method(f, x0, tol=1e-5, max_iter=100, grad_f=None, hess_l=None, constr=None, grad_c=None, lambda0=None):
    """
    Apply Sequential Quadratic Programming (SQP) to solve the constrained optimization problem.

    Minimize f(x) subject to constraints c(x) = 0.

    Parameters:
    f : function to compute the objective function f(x)
    x0 : initial guess for the variables x
    tol : tolerance for convergence
    max_iter : maximum number of iterations
    grad_f : function to compute the gradient of f at x
    hess_l : function to compute the Hessian of the Lagrangian L at x (or approximation)
    constr : function to compute the constraints c(x)
    grad_c : function to compute the Jacobian matrix of the constraints at x
    lambda0 : initial guess for Lagrange multipliers lambda
    """
    x = np.asarray(x0, dtype=float)
    if lambda0 is None:
        lambda0 = np.zeros(constr(x).shape)
    lam = lambda0

    for _ in range(max_iter):
        g = grad_f(x)
        H = hess_l(x, lam)
        c_val = constr(x)
        A = grad_c(x)

        # Form the KKT matrix
        KKT_matrix = np.block([
            [H, A.T],
            [A, np.zeros((A.shape[0], A.shape[0]))]
        ])

        # Form the right-hand side
        rhs = -np.hstack([g, c_val])

        # Solve the KKT system
        try:
            sol = np.linalg.solve(KKT_matrix, rhs)
        except np.linalg.LinAlgError:
            warnings.warn(f"KKT matrix is singular at x = {x}")
            break

        p = sol[:x.size]
        nu = sol[x.size:]

        x_new = x + p
        lam_new = nu  # Update Lagrange multipliers

        # Check convergence
        if np.linalg.norm(p) < tol and np.linalg.norm(c_val) < tol:
            return x_new

        x = x_new
        lam = lam_new

    warnings.warn(f"Iteration limit {max_iter} exceeded at x = {x}")
    return x


from mmcore._test_data import ssx

s1, s2 = ssx[2]


def f(x):
    u, v, s, t = x
    r = s1.evaluate_v2(u, v) - s2.evaluate_v2(s, t)
    return np.linalg.norm(r)


def grad_f(x):
    u, v, s, t = x
    r = s1.evaluate_v2(u, v) - s2.evaluate_v2(s, t)
    norm_r = np.linalg.norm(r)
    if norm_r == 0:
        return np.zeros_like(x)
    ds1_du = s1.derivative_u(np.array([u, v]))
    ds1_dv = s1.derivative_v(np.array([u, v]))
    ds2_ds = s2.derivative_u(np.array([s, t]))
    ds2_dt = s2.derivative_v(np.array([s, t]))
    grad_u = (ds1_du @ r) / norm_r
    grad_v = (ds1_dv @ r) / norm_r
    grad_s = - (ds2_ds @ r) / norm_r
    grad_t = - (ds2_dt @ r) / norm_r
    return np.array([grad_u, grad_v, grad_s, grad_t])


def hess_l(x, lam):
    J = compute_Jacobian(x)
    H = J.T @ J
    return H


def compute_Jacobian(x):
    u, v, s, t = x
    ds1_du = s1.derivative_u(np.array([u, v]))
    ds1_dv = s1.derivative_v(np.array([u, v]))
    ds2_ds = s2.derivative_u(np.array([s, t]))
    ds2_dt = s2.derivative_v(np.array([s, t]))
    J = np.zeros((3, 4))
    J[:, 0] = ds1_du
    J[:, 1] = ds1_dv
    J[:, 2] = -ds2_ds
    J[:, 3] = -ds2_dt
    return J


def constr(x):
    u, v, s, t = x
    x1, y1, z1 = s1.evaluate_v2(u, v)
    return np.array([a * x1 + b * y1 + c * z1 + d])


def grad_c(x):
    u, v, s, t = x
    ds1_du = s1.derivative_u(np.array([u, v]))
    ds1_dv = s1.derivative_v(np.array([u, v]))
    dc_du = a * ds1_du[0] + b * ds1_du[1] + c * ds1_du[2]
    dc_dv = a * ds1_dv[0] + b * ds1_dv[1] + c * ds1_dv[2]
    return np.array([[dc_du, dc_dv, 0.0, 0.0]])


# Define the parameters of the plane
a, b, c, d = 0, 0, 1., -4.
# Example values

# Define s1, s1_u, s1_v, s2, s2_s, s2_t based on your specific functions

# Initial guess
x0 = np.array([1.44, 1.2, 1.6, 1.8])

# Call the SQP method
solution = sqp_method(f, x0, max_iter=10000, grad_f=grad_f, hess_l=hess_l, constr=constr, grad_c=grad_c)