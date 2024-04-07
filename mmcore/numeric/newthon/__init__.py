import numpy as np

import warnings
def newtons_method(x0, f, f_prime, tolerance, epsilon, max_iterations, return_small_d=False, no_warn=False):
    """Newton's method

    Args:
      x0:              The initial guess
      f:               The function whose root we are trying to find
      f_prime:         The derivative of the function
      tolerance:       Stop when iterations change by less than this
      epsilon:         Do not divide by a number smaller than this
      max_iterations:  The maximum number of iterations to compute
    """
    if isinstance(x0,np.ndarray):
        check=lambda t0,t1: np.all(np.abs(t1 - t0)<=tolerance)
    else:
        check = lambda t0, t1: abs(t1 - t0) <= tolerance
    for _ in range(max_iterations):
        y = f(x0)
        yprime = f_prime(x0)

        if abs(yprime) < epsilon:  # Give up if the denominator is too small
            if not no_warn:
                warnings.warn(f'The denominator is too small! fprime(x0)<epsilon, fprime({x0})={yprime}, {yprime}<{epsilon}')
            if return_small_d:
                return x0
            else:
                break

        x1 = x0 - y / yprime  # Do Newton's computation

        if check(x0,x1):  # Stop when the result is within the desired tolerance
            return x1  # x1 is a solution within tolerance and maximum number of iterations

        x0 = x1  # Update x0 to start the process again
    if not no_warn:
        warnings.warn(f"Newton's method did not converge")
    return None  # Newton's method did not converge
