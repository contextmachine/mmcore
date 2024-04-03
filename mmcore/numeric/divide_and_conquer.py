import math

import numpy as np
from scipy.optimize import bisect

from mmcore.numeric.fdm import Grad


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


def iterative_divide_and_conquer_min(fun, bounds, tol):
    """
    Find the minimum value of a function within a given range using an iterative divide and conquer approach.

    :param fun: A callable function that takes a single parameter and returns a numeric value.
    :param bounds: A tuple representing the lower and upper bounds of the range to search for the minimum value.
    :param tol: The tolerance level for stopping the search when the range becomes smaller than this value.
    :return: A tuple containing the x-coordinate and the minimum value found within the given range.

    Complexity Analysis:
    - The time complexity of this algorithm is O(log(1/tol)), where tol is the desired precision level.
    - The space complexity is O(1) as we're using iteration now.

    Example Usage:
    >>> def square(x):
    ...     return x ** 2
    >>> iterative_divide_and_conquer_min(square, (0, 10), 0.01)
    (0.0, 0.0)
    """

    low, high = bounds

    while abs(high - low) >= tol:
        m1 = low + (high - low) / 4
        m2 = high - (high - low) / 4

        if fun(m1) < fun(m2):
            high = m2
        else:
            low = m1

    x_min = (low + high) / 2

    return x_min, fun(x_min)


from mmcore.numeric.bisection import closest_local_minimum


def find_best(fun, bounds, jac=None, tol=1e-6):
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mmcore.numeric.fdm import Grad

    from mmcore.numeric.divide_and_conquer import find_best

    def f(x):
        return np.cos(x -1) **3*x

    t0, t1 = -8, 8
    x = np.linspace(t0, t1, 100)
    y = f(x)

    global_min= find_best(f, (t0,t1), jac=Grad(f),tol=1e-5)


    plt.plot(x, y, "-k")
    plt.plot(*global_min, "rx")
    plt.show()

        :param fun:
        :param bounds:
        :param jac:
        :param tol:
        :return:
    """
    low, high = bounds
    if high - low < tol:
        return iterative_divide_and_conquer_min(fun, (low, high), tol)
    if not jac:
        jac = Grad(fun)
    branches = []
    minx, miny = iterative_divide_and_conquer_min(fun, (low, high), tol)
    branches.append((minx, miny))

    ((x1,), y1), ((x2,), y2) = (
        closest_local_minimum(jac, minx),
        closest_local_minimum(lambda x: -jac(x), minx),
    )

    if low < x1 < high:
        branches.append(find_best(fun, (low, x1), jac, tol))
    if low < x2 < high:
        branches.append(find_best(fun, (x2, high), jac, tol))

    return sorted(branches, key=lambda x: x[1])[0]


def recursive_divide_and_conquer_roots(fun, bounds, tol=0.01):
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

def sign(val):
    return math.copysign(1, val)


def test_all_roots(fun, bounds, tol):
    t0, t1 = bounds
    if t1 - t0 <= tol:
        return []


    t_max, y_max = iterative_divide_and_conquer_min(lambda t: -fun(t), (t0, t1), tol)

    t_min, y_min = iterative_divide_and_conquer_min(fun, (t0, t1), tol)
    (t11, y11, typ11), (t21, y21, typ21) = sorted([(t_min, y_min, 0), (t_max, -y_max, 1)], key=lambda x: x[0])

    if sign(y11) != sign(y21):

        bs = bisect(fun, t11, t21)


        if t21 - t11 <= tol:
            return [bs]

        return [*test_all_roots(fun, (t0, t11-tol), tol), bs, *test_all_roots(fun, (t21+tol, t1), tol)]

    else:

        if t21 - t11 <= tol:
            return []
        return [*test_all_roots(fun, (t21 + tol, t1), tol)]




