import numpy as np

DEFAULT_H = 1e-6


def fdm(f, method='central', h=DEFAULT_H):
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

class FDM:

    def __init__(self, fun):
        self._fun = fun

    def dydx(self, t):
        """
        Consider the plane curve defined by the parametric equations x=x(t) and y=y(t).
        Suppose that x′(t) and y′(t)
        exist, and assume that x′(t)≠0.



        Then the derivative dydx is given by dy/dx == (dy/dt)/(dx/dt) == y′(t)/x′(t)
        :param t:
        :return:
        """
        res = np.atleast_2d(self(t))
        return res[..., 1] / res[..., 0]

    def dzdx(self, t):
        """
        Consider the plane curve defined by the parametric equations x=x(t) and y=y(t).
        Suppose that x′(t) and y′(t)
        exist, and assume that x′(t)≠0.



        Then the derivative dydx is given by dz/dx == (dz/dt)/(dx/dt) == z′(t)/x′(t)
        :param t:
        :return:
        """
        res = np.atleast_2d(self(t))
        return res[..., 2] / res[..., 1]

    def dzdy(self, t):
        """
        Consider the plane curve defined by the parametric equations x=x(t) and y=y(t).
        Suppose that x′(t) and y′(t)
        exist, and assume that x′(t)≠0.



        Then the derivative dydx is given by dz/dy == (dz/dt)/(dy/dt) == z′(t)/y′(t)
        :param t:
        :return:
        """
        res = np.atleast_2d(self(t))
        return res[..., 2] / res[..., 1]

    def dzdydx(self, t):
        """
        Consider the plane curve defined by the parametric equations x=x(t) and y=y(t).
        Suppose that x′(t) and y′(t)
        exist, and assume that x′(t)≠0.



        Then the derivative dydx is given by dz/dy == (dz/dt)/(dy/dt) == z′(t)/y′(t)
        :param t:
        :return:
        """
        res = np.atleast_2d(self(t))
        return res[..., 2] / (res[..., 1] / res[..., 0])

    @property
    def prime(self):
        return self._fun

    def interval(self):
        return getattr(self._fun, 'interval', lambda: (0., 1.))()

    def central(self, t, h=DEFAULT_H):
        return (self._fun(t + h) - self._fun(t - h)) / (2 * h)

    def forward(self, t, h=DEFAULT_H):
        return (self._fun(t + h) - self._fun(t)) / h

    def backward(self, t, h=DEFAULT_H):
        return (self._fun(t) - self._fun(t - h)) / h

    def __call__(self, t, h=DEFAULT_H, method="central"):
        return getattr(self, method)(t, h=h)
