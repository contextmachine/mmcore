
import math
import warnings
from enum import IntEnum

import numpy as np

from mmcore.numeric.newton import gradient, hessian
from mmcore.numeric.vectors import scalar_norm

__all__=['bounded_newtons_method','line_search','line_search_strong_wolfe']

_TOO_SMALL=np.finfo(float).resolution
_TOO_SMALL_REGULARIZATION=_TOO_SMALL*1000



class _Matrix:
    def __init__(self, n=0):
        # Represents a matrix. We'll store an n-by-n matrix.
        self.n = n
        self.data = [[0.0] * n for _ in range(n)]

    def form_reduced_hessian(self, active_set, hessian_full):
        # Construct a reduced Hessian from full Hessian based on the active set (constraints)
        active_indices = [i for i, a in enumerate(active_set) if a == 0]
        m = len(active_indices)
        reduced = [[0.0] * m for _ in range(m)]
        for i_i, i in enumerate(active_indices):
            for j_j, j in enumerate(active_indices):
                reduced[i_i][j_j] = hessian_full[i][j]
        self.n = m
        self.data = reduced

    def row_reduce(self):
        # Perform Gaussian elimination with partial pivoting in-place.
        # This modifies self.data to an upper-triangular form.
        # Typically this would be used to prepare for back_solve or factorization.
        A = self.data
        n = self.n
        for i in range(n):
            # Partial pivoting: find pivot row
            pivot_row = i
            pivot_val = abs(A[i][i])
            for r in range(i + 1, n):
                if abs(A[r][i]) > pivot_val:
                    pivot_row = r
                    pivot_val = abs(A[r][i])

            # Swap rows if needed
            if pivot_row != i:
                A[i], A[pivot_row] = A[pivot_row], A[i]

            # If pivot is zero or extremely small, matrix is singular or near-singular
            if abs(A[i][i]) < _TOO_SMALL:
                # Matrix is singular or nearly singular
                # Attempt a small regularization:
                A[i][i] = _TOO_SMALL_REGULARIZATION

                #continue

            # Eliminate below the pivot
            for r in range(i + 1, n):
                factor = A[r][i] / A[i][i]
                for c in range(i, n):
                    A[r][c] -= factor * A[i][c]

        self.data = A

    def back_solve(self, rhs):
        # Solve upper-triangular system after row_reduce.
        # Assumes matrix is in upper-triangular form.
        n = len(rhs)
        A = self.data
        b = rhs[:]

        # Forward elimination already done by row_reduce.
        # Now do back substitution.
        x = [0] * n
        for i in reversed(range(n)):
            pivot = A[i][i]
            if abs(pivot) < _TOO_SMALL:
                # If pivot is too small, treat as zero for now
                return [0] * n
            s = b[i]
            for j in range(i + 1, n):
                s -= A[i][j] * x[j]
            x[i] = s / pivot
        return x


def line_search_backtrack(fobj, x, n, dx, xnew, grad, alpha_init, max_iter):
    alpha = alpha_init
    c = 1e-4
    f0 = fobj(x)
    g = grad(x)
    directional_derivative = sum(g[i] * dx[i] for i in range(n))
    for i in range(max_iter):
        trial = [x[j] + alpha * dx[j] for j in range(n)]
        ftrial = fobj(trial)
        if ftrial <= f0 + c * alpha * directional_derivative:
            return trial, True
        alpha *= 0.5
    return x, False


def line_search_strong_wolfe(f, grad, x, d, f0=None, g0=None, c1=1e-4, c2=0.9, max_iter=20):
    """
    Strong Wolfe line search to find alpha that satisfies:
    f(x+alpha*d) <= f(x) + c1*alpha*g(x)^T*d   (Armijo)
    |g(x+alpha*d)^T*d| <= c2*|g(x)^T*d|

    f: objective function
    grad: gradient function
    x: current point
    d: search direction
    f0: f(x) [optional precomputed]
    g0: grad(x) [optional precomputed]
    c1, c2: Wolfe constants
    max_iter: maximum iterations for the line search

    Returns (alpha, x_new, success)
    """
    if f0 is None:
        f0 = f(x)
    if g0 is None:
        g0 = grad(x)
    d_dot_g0 = sum(di * gi for di, gi in zip(d, g0))

    if d_dot_g0 >= 0:
        # Not a descent direction
        return 0.0, x, False

    alpha_max = 1e20
    alpha0 = 0
    alpha1 = 1.0
    f_prev = f0
    alpha_prev = 0
    for i in range(max_iter):
        x_new = [xi + alpha1 * di for xi, di in zip(x, d)]
        f_new = f(x_new)
        if f_new > f0 + c1 * alpha1 * d_dot_g0 or (i > 0 and f_new >= f_prev):
            # Perform a zoom between alpha_prev and alpha1
            return _line_search_zoom_wolfe(f, grad, x, d, f0, g0, alpha_prev, alpha1, c1, c2)

        g_new = grad(x_new)
        d_dot_g_new = sum(di * gi for di, gi in zip(d, g_new))
        if abs(d_dot_g_new) <= -c2 * d_dot_g0:
            # Satisfies strong Wolfe
            return alpha1, x_new, True

        if d_dot_g_new >= 0:
            # If slope is positive, we have a bracket
            return _line_search_zoom_wolfe(f, grad, x, d, f0, g0, alpha1, alpha_prev, c1, c2)

        # Update for next iteration
        alpha_prev = alpha1
        f_prev = f_new
        alpha1 = min(alpha1 * 2, alpha_max)

    # If we exit loop, just return last
    return alpha1, [xi + alpha1 * di for xi, di in zip(x, d)], False


def _line_search_zoom_wolfe(f, grad, x, d, f0, g0, alpha_lo, alpha_hi, c1, c2):
    d_dot_g0 = sum(di * gi for di, gi in zip(d, g0))
    f_lo = f([xi + alpha_lo * di for xi, di in zip(x, d)])
    for _ in range(20):
        alpha_j = 0.5 * (alpha_lo + alpha_hi)
        x_j = [xi + alpha_j * di for xi, di in zip(x, d)]
        f_j = f(x_j)
        x_lo = [xi + alpha_lo * di for xi, di in zip(x, d)]
        f_lo = f(x_lo)

        if f_j > f0 + c1 * alpha_j * d_dot_g0 or f_j >= f_lo:
            alpha_hi = alpha_j
        else:
            g_j = grad(x_j)
            d_dot_gj = sum(di * gi for di, gi in zip(d, g_j))
            if abs(d_dot_gj) <= -c2 * d_dot_g0:
                return alpha_j, x_j, True
            if d_dot_gj * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j
    # If zoom fails, just return alpha_lo
    x_lo = [xi + alpha_lo * di for xi, di in zip(x, d)]
    return alpha_lo, x_lo, False





def project_to_bounds(x, lower_bounds, upper_bounds):
    """
    Projects x onto the box defined by lower_bounds and upper_bounds.
    """
    x = np.maximum(x, lower_bounds)
    x = np.minimum(x, upper_bounds)
    return x


def line_search(f, grad, x, direction, lower_bounds, upper_bounds, c1=1e-4, c2=0.9, max_iter=20):
    """
    Strong Wolfe line search with bound projection.

    This line search ensures that we stay within the bounds by projecting
    trial points onto the feasible region.

    :param f: objective function
    :param grad: gradient function
    :param x: current point
    :param direction: descent direction
    :param lower_bounds, upper_bounds: box constraints
    :param c1, c2: Wolfe constants
    :return: (alpha, x_new, success)
    """
    f0 = f(x)
    g0 = grad(x)
    d_dot_g0 = np.dot(direction, g0)

    if d_dot_g0 >= 0:
        # Not a descent direction
        return 0.0, x, False

    alpha_max = 1e20
    alpha_prev = 0.0
    alpha = 1.0
    f_prev = f0

    for i in range(max_iter):
        x_new = project_to_bounds(x + alpha * direction, lower_bounds, upper_bounds)
        f_new = f(x_new)
        if f_new > f0 + c1 * alpha * d_dot_g0 or (i > 0 and f_new >= f_prev):
            # Zoom
            return _line_search_zoom(f, grad, x, direction, f0, g0, lower_bounds, upper_bounds, alpha_prev, alpha, c1,
                                    c2)

        g_new = grad(x_new)
        d_dot_g_new = np.dot(direction, g_new)
        if abs(d_dot_g_new) <= -c2 * d_dot_g0:
            # Strong Wolfe satisfied
            return alpha, x_new, True

        if d_dot_g_new >= 0:
            # Zoom
            return _line_search_zoom(f, grad, x, direction, f0, g0, lower_bounds, upper_bounds, alpha, alpha_prev, c1,
                                    c2)

        alpha_prev = alpha
        f_prev = f_new
        alpha = min(alpha * 2, alpha_max)

    # If we reach here, line search failed
    x_new = project_to_bounds(x + alpha * direction, lower_bounds, upper_bounds)
    return alpha, x_new, False


def _line_search_zoom(f, grad, x, d, f0, g0, lower_bounds, upper_bounds, alpha_lo, alpha_hi, c1, c2):
    d_dot_g0 = np.dot(d, g0)
    for _ in range(20):
        alpha_j = 0.5 * (alpha_lo + alpha_hi)
        x_j = project_to_bounds(x + alpha_j * d, lower_bounds, upper_bounds)
        f_j = f(x_j)
        x_lo = project_to_bounds(x + alpha_lo * d, lower_bounds, upper_bounds)
        f_lo = f(x_lo)

        if (f_j > f0 + c1 * alpha_j * d_dot_g0) or (f_j >= f_lo):
            alpha_hi = alpha_j
        else:
            g_j = grad(x_j)
            d_dot_gj = np.dot(d, g_j)
            if abs(d_dot_gj) <= -c2 * d_dot_g0:
                return alpha_j, x_j, True
            if d_dot_gj * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j
    # If zoom fails
    x_lo = project_to_bounds(x + alpha_lo * d, lower_bounds, upper_bounds)
    return alpha_lo, x_lo, False


def bounded_newtons_method(f, initial_point, bounds, tol=1e-5, step_tol=1e-8, gtol=1e-8,max_iter=100, no_warn=False,
                           full_return=False, grad=None, hess=None, h=1e-5, min_value=None):
    """
    A bounded version of Newton's method that:
    - Uses numerical gradients and Hessians if not provided.
    - Ensures the updated point always lies within the given bounds.
    - Employs a line search to ensure sufficient decrease and robust steps.
    - Can handle difficult landscapes like Rosenbrock.

    :param f: Objective function to minimize.
    :param lower_bounds: Array of lower bounds for each variable.
    :param upper_bounds: Array of upper bounds for each variable.
    :param initial_point: Starting point for optimization.
    :param tol: Tolerance for termination.
    :param max_iter: Maximum number of iterations.
    :param no_warn: If True, suppress warnings.
    :param full_return: If True, return intermediate values.
    :param grad: Optional gradient function. If None, finite differences are used.
    :param hess: Optional Hessian function. If None, finite differences are used.
    :param h: Step size for finite differences.
    :return: If full_return, (x, grad, H, H_inv, iter), else x.
    """
    point = np.asarray(initial_point, dtype=float)
    lower_bounds, upper_bounds=zip(*bounds)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)

    # Project initial point to bounds
    point = project_to_bounds(point, lower_bounds, upper_bounds)

    if grad is None:
        _grad = lambda x: gradient(f, x, h=h)
    else:
        _grad = grad
    if hess is None:
        _hess = lambda x: hessian(f, x, h=h)
    else:
        _hess = hess

    for it in range(max_iter):
        g = _grad(point)
        H = _hess(point)

        # Check convergence by gradient norm
        if scalar_norm(g) < gtol:
            if full_return:
                return point, g, H, np.linalg.inv(H) if np.linalg.cond(H) < 1 / 1e-8 else None, it
            return point

        # Try to invert Hessian
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            # Hessian is singular or ill-conditioned, try adding a small regularization:
            H_reg = H + np.eye(len(point)) * _TOO_SMALL_REGULARIZATION
            try:
                H_inv = np.linalg.inv(H_reg)
            except:
                if not no_warn:
                    warnings.warn(f"Hessian is singular at {point}, can't invert even with regularization.")
                # If still fail, fallback to gradient step (like steepest descent)
                H_inv = np.eye(len(point))

        step = H_inv @ g

        # Line search to ensure sufficient decrease and respect bounds
        # We want to move point in the direction -step (Newton direction is -H_inv*g)
        direction = -step
        alpha, new_point, success = line_search(f, _grad, point, direction, lower_bounds, upper_bounds, max_iter=20)
        if not success:
            # If line search fails to find a suitable step, try a smaller step or fallback
            # Here we just project and move a tiny bit along -g if Newton step fails
            direction_fallback = -g
            alpha_fb, new_point_fb, success_fb = line_search(f, _grad, point, direction_fallback, lower_bounds,
                                                             upper_bounds, max_iter=20)
            if not success_fb:
                if not no_warn:
                    warnings.warn("Line search failed. No suitable step found.")
                return point if not full_return else (point, g, H, H_inv, it)
            new_point = new_point_fb
            # After fallback, we continue

        # Check step size
        if min_value is not None:
            if f(point)-min_value < tol:
                if full_return:
                    return new_point, g, H, H_inv, it
                return new_point
        if scalar_norm(new_point - point) < step_tol :
            if full_return:
                return new_point, g, H, H_inv, it
            return new_point

        point = new_point

    if not no_warn:
        warnings.warn(f"Iteration limit {max_iter} reached at {point}.")
    if full_return:
        return None, g, H, None, max_iter
    return None



if __name__ == "__main__":
    import time

    from mmcore.numeric.newton.cnewton import newtons_method as cnewtons_method

    from scipy.optimize import minimize
    import time


    # Rosenbrock function
    def rosenbrock(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2



    bounds = [(0.1, 1.5), (-1.1, 1.5)]
    x0 = [1.4,1.1]



    s = time.perf_counter_ns()
    res2 = bounded_newtons_method(rosenbrock, x0, bounds,tol=1e-5)
    br2 = (time.perf_counter_ns() - s) * 1e-9

    print('bounded_newthon2 time:', br2)
    print("Result:", res2)



    s = time.perf_counter_ns()
    res = cnewtons_method(rosenbrock, np.array(x0), tol=1e-5)
    res=np.array(res)
    nr=(time.perf_counter_ns() - s)*1e-9
    print("cnewthon time: ", nr)
    print("Result: ", res)



    s=time.perf_counter_ns()
    result_scipy = minimize(rosenbrock, np.array(x0), tol=1e-5,bounds=bounds)
    sr=(time.perf_counter_ns() - s)*1e-9
    print('scipy minimize time:',sr)






    print(f"bounded_newtons_method: x={res2}, fun={rosenbrock(res2)}")
    print(f"cnewthon: x={res}, fun={rosenbrock(res)}")
    print(f"scipy minimize: x={result_scipy.x}, fun={result_scipy.fun}")





    if rosenbrock(res)<result_scipy.fun:
        print(f"bounded_newthon (fun={rosenbrock(res)}) found a better solution than scipy minimize (fun={result_scipy.fun})")
    elif rosenbrock(res)>result_scipy.fun:
        print(f"scipy minimize found a better solution (fun={rosenbrock(res)}) than bounded_newthon (fun={result_scipy.fun})")
    else:
        print(f"similar results. bounded_newthon: {rosenbrock(res)}, scipy minimize: {result_scipy.fun}" )
    if (sr / nr)>1:
        print("cnewtons is", sr / nr, "times faster than scipy minimize")
    else:
        print("cnewton is",  nr/sr, "times slower than scipy minimize")
    if (sr / br2)>1:
        print("bounded_newton is", sr / br2, "times faster than scipy minimize")
    else:
        print("bounded_newthon is",  nr/br2, "times slower than scipy minimize")