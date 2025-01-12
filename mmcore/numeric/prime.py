import numpy as np
from scipy.integrate import quad_vec

from mmcore.numeric.fdm import FDM


class Prime:
    """
    Находит первообразную параметрическую форму по параметрической форме касательной, такую что
    >>> fun=np.vectorize(lambda t:np.array((t**2*4+t * 2 - 3, -t**2+2 * t + 1)),signature='()->(i)')
    >>> pfun=Prime(fun)
    >>> dfun = FDM(pfun)
    >>> np.allclose(fun(0.1),dfun(0.1))
    True

    Или
    >>>
    >>> dfun = FDM(fun)
    >>> pfun=Prime(dfun, simplify_fdm=False)
    >>> pfun(0.1), fun(0.1)
    (array([2.8, 2.2]), array([-2.76,  1.21]))

    Ожидаемо не совпадают,
    тк производная не дает о том насколько относительно
    сдвижки первообразной относительно x,y

    Чтобы все работало правильно нужно знать хотя бы одну точку с соответствующим параметром на первообразной
    >>> pfun(0.)
    array([0.,  0.])
    >>> fun(0.0)
    array([-3.,  1.]) # Это и есть смещение
    >>> pfun.translate=fun(0.0)
    >>> np.allclose(fun(0.1),pfun(0.1))
    True

    """

    def __new__(cls, fun, dim=2, translate=None, simplify_fdm=True):
        if simplify_fdm:
            if isinstance(fun, FDM):
                return fun.prime
        self = super().__new__(cls)
        self._fun = fun
        self.dim = dim
        self._translate = np.zeros(dim)
        if translate is not None:
            self._translate[:] = translate

        self.evaluate_multi = np.vectorize(self._fun, excluded=[1, 2], signature='()->(i)')
        return self

    @property
    def translate(self):
        return self._translate

    @translate.setter
    def translate(self, v):
        self._translate[:] = v

    def evaluate(self, t, t0=0.0, tol=1e-6):

        res, err = quad_vec(self._fun, t0, t)
        if err > tol:
            raise ValueError(f"Error is not less than tolerance err={err}, {tol}.")
        return res + self._translate

    def __call__(self, t, tol=1e-6, t0=None):
        if t0 is None:
            t0 = getattr(self._fun, 'interval', lambda: (0.,))()[0]
        if isinstance(t, (float, int)):
            return self.evaluate(t, t0=t0, tol=tol)
        else:
            return self.evaluate_multi(t, t0=t0, tol=tol)
