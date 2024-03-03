import numpy as np

from mmcore.geom.vec import norm_sq


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


def bisection2d(f, step=0.00001, start=(0., 0.), stop=(1., 1.)):
    signx, signy = np.array(f(start[0], start[1])) > 0

    x, y = start

    rootsx = []
    rootsy = []
    minimum = []

    while x <= stop[0]:
        while y <= stop[1]:

            valuex, valuey = f(x, y)

            if valuex == 0 and valuey == 0:
                # We hit a root

                rootsx.append((x, valuex))
                rootsy.append((y, valuey))


            elif ((valuex > 0) != signx) and ((valuey > 0) != signy):
                # We passed a root
                rootsx.append((x, valuex))
                rootsy.append((y, valuey))
            # Update our sign

            signy = valuey > 0
            signx = valuex > 0
            y += step

        x += step
    return rootsx, rootsy


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


def closest_point_on_curve(curve, point, step=0.001, divs=32):
    def objective(t):
        return norm_sq(curve(t) - point)

    return scalar_min_1d(objective, *curve.interval(), step=step, divs=divs)
