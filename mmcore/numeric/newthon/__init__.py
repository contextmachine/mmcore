def newtons_method(x0, f, f_prime, tolerance, epsilon, max_iterations):
    """Newton's method

    Args:
      x0:              The initial guess
      f:               The function whose root we are trying to find
      f_prime:         The derivative of the function
      tolerance:       Stop when iterations change by less than this
      epsilon:         Do not divide by a number smaller than this
      max_iterations:  The maximum number of iterations to compute
    """
    for _ in range(max_iterations):
        y = f(x0)
        yprime = f_prime(x0)

        if abs(yprime) < epsilon:  # Give up if the denominator is too small
            break

        x1 = x0 - y / yprime  # Do Newton's computation

        if abs(x1 - x0) <= tolerance:  # Stop when the result is within the desired tolerance
            return x1  # x1 is a solution within tolerance and maximum number of iterations

        x0 = x1  # Update x0 to start the process again

    return None  # Newton's method did not converge
