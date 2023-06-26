from enum import Enum

import numpy as np


class PDMethods(str, Enum):
    central = "central"
    forward = 'forward'
    backward = 'backward'


def derivative(f, method: PDMethods = PDMethods.central, h=0.01):
    '''Compute the difference formula for f'(t) with step size h.

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


def _ns(dx, dy):
    return np.sqrt((dx ** 2) + (dy ** 2))


def offset_curve_2d(c, d):
    df = derivative(c)

    def wrap(t):
        x, y = c(t)
        dx, dy = df(t)
        ox = x + (d * dy / _ns(dx, dy))
        oy = y - (d * dx / _ns(dx, dy))
        return [ox, oy]

    wrap.__name__ = c.__name__ + f"_normal_{d}"
    return wrap


"""    {\displaystyle x_{d}(t)=x(t)+{\frac {d\;y'(t)}{\sqrt {x'(t)^{2}+y'(t)^{2}}}}}
    y d ( t ) = y ( t ) − d x ′ ( t ) x ′ ( t ) 2 + y ′ ( t ) 2
    . {\displaystyle y_{d}(t)=y(t)-{\frac {d\;x'(t)}{\sqrt {x'(t)^{2}+y'(t)^{2}}}}\ .}"""
