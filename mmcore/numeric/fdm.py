def fdm(f, method='central', h=1e-6):
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

    @property
    def prime(self):
        return self._fun

    def interval(self):
        return getattr(self._fun, 'interval', lambda: (0., 1.))()

    def central(self, t, h=1e-6):
        return (self._fun(t + h) - self._fun(t - h)) / (2 * h)

    def forward(self, t, h=1e-6):
        return (self._fun(t + h) - self._fun(t)) / h

    def backward(self, t, h=1e-6):
        return (self._fun(t) - self._fun(t - h)) / h

    def __call__(self, t, h=1e-6, method="central"):
        return getattr(self, method)(t, h=h)
