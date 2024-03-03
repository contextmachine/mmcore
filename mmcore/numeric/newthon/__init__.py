from typing import List, Tuple, Callable

import numpy as np



def _newton(x: np.ndarray, f: Callable, gf: Callable, hf: Callable, lr=0.01, lr_decr=0.999, maxiter=100, tol=0.001) -> \
Tuple[np.ndarray, List[np.ndarray], int]:
    """
    Applies the Newton's method to find the minimum of a multidimensional function, using the update criterion:
    x_k+1 = x_k - lr * inverse(hf(x)) * gf(x), for the k-th iteration.

    Args:
        x (np.ndarray): An array representing the initial point where the algorithm starts.
        f (Callable): Objective function to minimize.
        gf (Callable): Gradient of the objective function.
        hf (Callable): Hessian of the objective function.
        lr (float, optional): Initial learning rate. Default is 0.01.
        lr_decr (float, optional): Decay factor for the learning rate. Default is 0.999.
        maxiter (int, optional): Maximum number of iterations. Default is 100.
        tol (float, optional): Tolerance for the gradient norm that determines convergence. Default is 0.001.

    Returns:
        Tuple[np.ndarray, List[np.ndarray], int]: A tuple with three elements:
            - The approximate minimum point.
            - A list of intermediate points (arrays) calculated during optimization.
            - The number of iterations performed.

    Example:

    # Define a 2-dimensional quadratic function: f(x, y) = x^2 + 2y^2
    def objective_function(x):
        return x[0] ** 2 + 2 * x[1] ** 2

    # Define the gradient of the objective function: f'(x, y) = [2x, 4y]
    def gradient_function(x):
        return np.array([2 * x[0], 4 * x[1]])

    # Define the Hessian of the objective function: f''(x, y) = [[2, 0], [0, 4]]
    def hessian_function(x):
        return np.array([[2, 0], [0, 4]])

    # Initial point for optimization
    initial_point = np.array([3.0, 2.0])

    # Apply the Newton's method for optimization
    result, intermediate_points, iterations = newton(initial_point, objective_function, gradient_function,
                                                                                            hessian_function)
    """
    points = [x]
    nit = 0
    gradient = gf(x)
    hessian = hf(x)

    while nit < maxiter and np.linalg.norm(gradient) >= tol:
        x = x - lr * np.dot(np.linalg.inv(hessian), gradient)  # Matrix multiplication using np.dot(m1, m2)
        lr *= lr_decr  # Learning rate update: tk+1 = tk * ρ, with ρ being the decay factor.
        points.append(x)
        nit += 1
        gradient = gf(x)
        hessian = hf(x)

    return x, points, nit


class NewtonRaphson:

    lr:float=0.01
    lr_decr:float=0.999
    maxiter:float=100
    tol:float=0.001

    def __init__(self, fun, gradient_fun=None, hessian_fun=None, **kwargs):
        super().__init__()

        self.fun=fun
        self.props = {**dict(lr=0.01,
                             lr_decr=0.999,
                             maxiter=100,
                             tol=0.001), **kwargs}
        self.gradient_fun = gradient_fun
        self.hessian_fun = hessian_fun
        if hessian_fun is None:
            self.hessian_fun = jax.hessian(fun)
        if gradient_fun is None:
            self.gradient_fun = jax.grad(fun)

    def __call__(self, x=0.0, full_return=False):


        result, intermediate_points, iterations=_newton(x, self.fun, self.gradient_fun, self.hessian_fun,
                                                        self.props)
        if full_return:
            return result, intermediate_points, iterations
        else:
            return result


def FDM(f, method='central', h=0.001):
    '''Compute the FDM formula for f'(t) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable

    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    lambda t:
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h
    '''
    if method == 'central':
        return lambda t: (f(t + h) - f(t - h)) / (2 * h)
    elif method == 'forward':
        return lambda t: (f(t + h) - f(t)) / h
    elif method == 'backward':
        return lambda t: (f(t) - f(t - h)) / h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
def newthon(x, fun, gradient_fun=None, hessian_fun=None):
    optimizer = NewtonRaphson(fun, gradient_fun, hessian_fun)
    return optimizer(x=x, full_return=False)
