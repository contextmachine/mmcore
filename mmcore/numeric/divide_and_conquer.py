import numpy as np


def recursive_divide_and_conquer_min(fun, bounds, tol):
    """
    Find the minimum value of a function within a given range using a recursive divide and conquer approach.

    :param fun: A callable function that takes a single parameter and returns a numeric value.
    :param bounds: A tuple representing the lower and upper bounds of the range to search for the minimum value.
    :param tol: The tolerance level for stopping the search when the range becomes smaller than this value.
    :return: A tuple containing the x-coordinate and the minimum value found within the given range.

    Algorithm Steps:
    1. Decompose the range into two parts, m1 and m2.
    2. Check if the precision level is achieved (high - low < tol). If yes, return the midpoint (x_min) and the corresponding function value.
    3. If not, compare the function values at m1 and m2.
    4. Recurse on the range with the lower function value.
    5. Repeat steps 2-4 until the minimum value is found within the desired precision level.

    Complexity Analysis:
    - The time complexity of this algorithm is O(log(1/tol)), where tol is the desired precision level.
    - The space complexity is O(log(1/tol)) due to the recursive calls.

    Example Usage:
    >>> def square(x):
    ...     return x ** 2
    >>> recursive_divide_and_conquer_min(square, (0, 10), 0.01)
    (0.0, 0.0)
    """
    low, high = bounds

    # Check if the precision level is achieved.
    if high - low < tol:
        x_min = (low + high) / 2

        return x_min, fun(x_min)
    else:
        # Divide the range into four parts.
        m1 = low + (high - low) / 4
        m2 = high - (high - low) / 4

        # Recurse on the half where the function value is lower.
        if fun(m1) < fun(m2):
            return recursive_divide_and_conquer_min(fun, (low, m2), tol)
        else:
            return recursive_divide_and_conquer_min(fun, (m1, high), tol)


def recursive_divide_and_conquer_max(fun, bounds, tol):
    """
    Find the maximum value of a function within a given range using a recursive divide and conquer approach.

    :param fun: A callable function that takes a single parameter and returns a numeric value.
    :param bounds: A tuple representing the lower and upper bounds of the range to search for the minimum value.
    :param tol: The tolerance level for stopping the search when the range becomes smaller than this value.
    :return: A tuple containing the x-coordinate and the minimum value found within the given range.

    Algorithm Steps:
    1. Decompose the range into two parts, m1 and m2.
    2. Check if the precision level is achieved (high - low < tol). If yes, return the midpoint (x_min) and the corresponding function value.
    3. If not, compare the function values at m1 and m2.
    4. Recurse on the range with the lower function value.
    5. Repeat steps 2-4 until the minimum value is found within the desired precision level.

    Complexity Analysis:
    - The time complexity of this algorithm is O(log(1/tol)), where tol is the desired precision level.
    - The space complexity is O(log(1/tol)) due to the recursive calls.

    Example Usage:
    >>> def square(x):
    ...     return x ** 2
    >>> recursive_divide_and_conquer_min(square, (0, 10), 0.01)
    (0.0, 0.0)
    """
    low, high = bounds

    # Check if the precision level is achieved.
    if high - low < tol:
        x_max = (low + high) / 2

        return x_max, fun(x_max)
    else:
        # Divide the range into four parts.
        m1 = low + (high - low) / 4
        m2 = high - (high - low) / 4

        # Recurse on the half where the function value is lower.
        if fun(m1) > fun(m2):
            return recursive_divide_and_conquer_max(fun, (low, m2), tol)
        else:
            return recursive_divide_and_conquer_max(fun, (m1, high), tol)


def recursive_divide_and_conquer_roots(fun, bounds, tol=1e-5):
    low, high = bounds
    roots = []
    # Check if the precision level is achieved.
    if abs((high - low)) < tol:

        x_root = (low + high) / 2
        roots.append([x_root, fun(x_root)])
        return roots
    else:
        # Divide the range into four parts.
        m1 = low + (high - low) / 2
        m2 = high - (high - low) / 2

        # Recurse on the half where the function value is lower.
        if np.all(abs(fun(m1)) < abs(fun(m2))):
            roots.extend(recursive_divide_and_conquer_roots(fun, (low, m2), tol))
        else:
            roots.extend(recursive_divide_and_conquer_roots(fun, (m1, high), tol))

        return roots
