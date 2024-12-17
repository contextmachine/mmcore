import numpy as np

import warnings
from mmcore.numeric.fdm import newtons_method,hessian,gradient
__all__ = ['newtons_method', 'newtons_method_with_constraints']

from mmcore.numeric.vectors import scalar_norm, scalar_dot


def newtons_method_with_constraints(f, constraint, initial_point, tol=0.00001, max_iter=100, no_warn=False, full_return=False, grad=None, hess=None, lagrange_multiplier=1.0):
    """
    Apply Newton's method to find the root of a function with constraints.
    The same powerful newton converging in a few iterations.

    :param f: The function for which the root is to be found.
    :param constraint: The constraint function.
    :param initial_point: The initial point for the iteration.
    :param tol: Tolerance for the stopping criterion. Default is DEFAULT_H.
    :param max_iter: Maximum number of iterations. Default is 100.
    :param no_warn: If True, suppress warnings. Default is False.
    :param full_return: If True, return all intermediate variables. Default is False.
    :param grad: The gradient of the function. If None, compute the gradient using the gradient function.
    :param hess: The Hessian of the function. If None, compute the Hessian using the hessian function.
    :param lagrange_multiplier: The initial value of the Lagrange multiplier. Default is 1.0.
    :return: The root of the function if found, None otherwise.
    """
    point = np.asarray(initial_point)
    lagrange_multiplier = lagrange_multiplier
    H_inv = None
    H = None
    if grad is None:
        _grad = lambda x: gradient(f, x)
    else:
        _grad = grad
    if hess is None:
        hess = lambda x: hessian(f, x)
    else:
        hess = hess

    for _ in range(max_iter):
        grad_f = _grad(point)
        grad_g = gradient(constraint, point)
        H_f = hess(point)
        H_g = hessian(constraint,point)

        try:
            H_inv_f = np.linalg.inv(H_f)
        except np.linalg.LinAlgError:
            if not no_warn:
                warnings.warn(f"Hessian of f is singular at the point {point}")
            break

        try:
            H_inv_g = np.linalg.inv(H_g)
        except np.linalg.LinAlgError:
            if not no_warn:
                warnings.warn(f"Hessian of g is singular at the point {point}")
            break

        grad_F = grad_f - lagrange_multiplier * grad_g
        H_F = H_f + lagrange_multiplier * H_g

        step = H_inv_f @ grad_F

        new_point = point - step


        if scalar_dot(step,step) < tol:
            if full_return:
                return new_point, grad_F, H_F, H_inv_f, _
            return new_point
        if scalar_norm(point-new_point)<tol:
            if full_return:
                return new_point, grad_F, H_F, H_inv_f, _
            return new_point
        print(max_iter, new_point,point)
        point = new_point

    if not no_warn:

        warnings.warn(f"Iteration limit {max_iter} at {new_point}, {scalar_dot(new_point-point,new_point-point)}")
    if full_return:

        return None, grad_F, H_F, H_inv_f, max_iter

    return point