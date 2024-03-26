import numpy as np

DEFAULT_H = 1e-20


def fdm(f, method="central", h=DEFAULT_H):
    """Compute the FDM formula for f'(t) with step size h.

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
    """
    if method == "central":
        return lambda t: (f(t + h) - f(t - h)) / (2 * h)
    elif method == "forward":
        return lambda t: (f(t + h) - f(t)) / h
    elif method == "backward":
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
        return getattr(self._fun, "interval", lambda: (0.0, 1.0))()

    def central(self, t, h=DEFAULT_H):
        return (self._fun(t + h) - self._fun(t - h)) / (2 * h)

    def forward(self, t, h=DEFAULT_H):
        return (self._fun(t + h) - self._fun(t)) / h

    def backward(self, t, h=DEFAULT_H):
        return (self._fun(t) - self._fun(t - h)) / h

    def __call__(self, t, h=DEFAULT_H, method="central"):
        return getattr(self, method)(t, h=h)


class PartialFDM(FDM):
    def central(self, t, h=DEFAULT_H):
        t = np.atleast_1d(t)
        H = np.eye(len(t)) * h

        return np.array(
            [(self._fun(t + vh) - self._fun(t - vh)) / (2 * vh) for vh in H]
        )

    def forward(self, t, h=DEFAULT_H):
        t = np.atleast_1d(t)
        H = np.eye(len(t)) * h

        return np.array([(self._fun(t + h) - self._fun(t)) / h for h in H])

    def backward(self, t, h=DEFAULT_H):
        t = np.atleast_1d(t)
        H = np.eye(len(t)) * h

        return np.array([(self._fun(t) - self._fun(t - h)) / h for h in H])


class Grad(FDM):
    def central(self, t, h=DEFAULT_H):
        t = np.atleast_1d(t)
        z = np.zeros(len(t), dtype=float)
        for i in range(len(t)):
            o = np.zeros(len(t), dtype=float)
            o[i] = h
            z[i] = (self._fun(t + o) - self._fun(t - o)) / (2 * h)

        return z

    def __call__(self, t, h=DEFAULT_H, method="auto"):
        return super().__call__(t, h, method)

    def forward(self, t, h=DEFAULT_H):
        t = np.atleast_1d(t)
        H = np.eye(len(t)) * h

        return np.array([(self._fun(t + h) - self._fun(t)) / h for h in H])

    def backward(self, t, h=DEFAULT_H):
        t = np.atleast_1d(t)
        H = np.eye(len(t)) * h
        # print(H, t)
        return np.array([(self._fun(t) - self._fun(t - h)) / h for h in H])


class Hess(Grad):
    def __init__(self, fun):
        super().__init__(fun)
        self._grad = Grad(self._fun)

    def central(self, t, h=DEFAULT_H):
        t = np.atleast_1d(t)
        lx, ly = len(t), len(t)
        z = np.zeros((lx, ly), dtype=float)

        for i in range(lx):
            o = np.zeros(ly, dtype=float)
            o[i] = h
            z[i, ...] = (self._grad(t + o) - self._grad(t - o)) / (2 * h)

        return z
