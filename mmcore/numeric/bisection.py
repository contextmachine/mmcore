
from scipy.optimize import minimize

from .fdm import Grad


def closest_local_minimum(func, x0, full_output=False):
    res = minimize(func, x0, method="BFGS", jac=Grad(func))
    if full_output:
        return res
    else:
        return res.x, res.fun




def bisection1d(f, step=0.00001, start=-1, stop=3):
    # Smaller step values produce more accurate and precise results

    sign = f(start) > 0
    x = start
    roots = []
    minimum = np.inf
    while x <= stop:
        value = f(x)
        if value < minimum:
            minimum = value
        if value == 0:
            # We hit a root
            roots.append((x, value))

        elif (value > 0) != sign:
            # We passed a root
            roots.append((x, value))

        # Update our sign
        sign = value > 0
        x += step
    return roots, minimum


import numpy as np


def bisection_2d(f, x_range, y_range, tol=1e-6):
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Check if the initial range is valid
    if f(x_min, y_min) * f(x_max, y_max) > 0:
        print("Bisection method may not converge: f(x_min, y_min) and f(x_max, y_max) have the same sign.")
        return None

    while (x_max - x_min) / 2 > tol or (y_max - y_min) / 2 > tol:
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2

        # Evaluate the function at the corners and midpoints
        f00 = f(x_min, y_min)
        f10 = f(x_max, y_min)
        f01 = f(x_min, y_max)
        f11 = f(x_max, y_max)
        fmid = f(x_mid, y_mid)

        # Determine the subregion to search next
        if f00 * fmid < 0:
            x_max, y_max = x_mid, y_mid
        elif f10 * fmid < 0:
            x_min, y_max = x_mid, y_mid
        elif f01 * fmid < 0:
            x_max, y_min = x_mid, y_mid
        elif f11 * fmid < 0:
            x_min, y_min = x_mid, y_mid
        else:
            # Shrink the search region
            if abs(f(x_min, y_mid)) < abs(f(x_max, y_mid)):
                x_max = x_mid
            else:
                x_min = x_mid

            if abs(f(x_mid, y_min)) < abs(f(x_mid, y_max)):
                y_max = y_mid
            else:
                y_min = y_mid

    return (x_min + x_max) / 2, (y_min + y_max) / 2


def scalar_min_1d(f, start, end, step=0.001, divs=32):
    def wrap(start, end):
        t = np.linspace(start, end, divs)

        res = f(t)
        m = res.min()
        i = np.where(np.isclose(res, m))[0][0]

        if abs(t[0] - t[1]) <= step:
            return t[i]
        else:
            ixs = np.array([i - 1, i + 1], int)

            return wrap(t[ixs[0]], t[ixs[1]])

    return wrap(start, end)

