import dataclasses
import functools
import math
import sys
from typing import Callable, Optional, Iterable

import numpy as np

DEFAULT_H = 10 ** (-((sys.float_info.dig) - 1) // 2)
_DECIMALS = abs(int(math.log10(DEFAULT_H)))


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
    _decimals = abs(int(math.log10(h)))

    if method == "central":
        return lambda t: np.round((f(t + h) - f(t - h)) / (2 * h), _decimals)
    elif method == "forward":
        return lambda t: np.round((f(t + h) - f(t)) / h, _decimals)
    elif method == "backward":
        return lambda t: np.round((f(t) - f(t - h)) / h, _decimals)
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


def bounded_fdm(f, h=DEFAULT_H, bounds=(0., 1.)):
    _decimals = abs(int(math.log10(h)))

    def wrp(t):
        if abs(bounds[0] - t) <= h:
            return np.round((f(t + h) - f(t)) / h, _decimals)
        elif abs(bounds[1] - t) <= h:
            return np.round((f(t) - f(t - h)) / h, _decimals)
        else:
            return np.round((f(t + h) - f(t - h)) / (2 * h), _decimals)

    return wrp


class FDM:
    def __new__(cls, fun=None):
        #record = Memo.get_or_create_record(fun)

        #rec = getattr(record, cls.__name__.lower(), None)
        #if rec:
        #    return rec

        obj = super().__new__(cls)

        obj._fun = fun

        #Memo.update_record(fun, {cls.__name__.lower(): obj})
        return obj

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
        _decimals = abs(int(math.log10(h)))
        return np.round(getattr(self, method)(t, h=h), _decimals)


from mmcore.numeric.routines import remove_dim


def _prepare_memo_record_args(*args, **kwargs):
    keys = list(MemoRecord.__dataclass_fields__.keys())
    return {**{keys[i]: val for i, val in enumerate(args)}, **kwargs}


@dataclasses.dataclass
class MemoRecord:
    func: Callable
    fdm: Optional[Callable] = None
    grad: Optional[Callable] = None
    hess: Optional[Callable] = None

    __match_args__ = ("func_hash",)

    @property
    def func_hash(self):
        return getattr(self.func, "__hash__", lambda: id(self.func))()


class Memo:
    _data = dict()

    def __new__(cls, *args, **kwargs):
        raise TypeError("Cannot instantiate")

    @classmethod
    def get_record(cls, fun, default=None) -> MemoRecord:
        return cls._data.get(fun, default)

    @classmethod
    def get_or_create_record(cls, func) -> MemoRecord:
        if func not in cls._data:
            cls.create_or_update_record(func)

        return cls.get_record(func)

    @classmethod
    def create_record(cls, func, *args, **kwargs):
        cls._data[func] = MemoRecord(func, **_prepare_memo_record_args(*args, **kwargs))

    @classmethod
    def update_record(cls, func, *args, **kwargs):
        cls._data[func].__dict__.update(**_prepare_memo_record_args(*args, **kwargs))

    @classmethod
    def create_or_update_record(cls, func, *args, **kwargs) -> None:
        if func not in cls._data:
            cls.create_record(func, *args, **kwargs)
        cls.update_record(
            func, *args, **kwargs
        ) if func not in cls._data else cls.create_record(func, *args, **kwargs)

    @classmethod
    def __contains__(cls, item) -> bool:
        return cls._data.__contains__(item)

    @classmethod
    def __len__(cls) -> int:
        return cls._data.__len__()

    @classmethod
    def records(cls) -> Iterable[MemoRecord]:
        return cls._data.values()


__memo__ = Memo


def pde(fun, x, h=DEFAULT_H):
    b = x.shape[-1]
    z = np.eye(b)
    forw = z * h

    z = np.zeros(x.shape)
    for i in range(b):
        z[..., i] = (fun(x + forw[i]) - fun(x - forw[i])) / 2 / h

    return z


class Grad(FDM):
    def central(self, t, h=DEFAULT_H):
        if np.isscalar(t):
            return pde(self._fun, np.atleast_1d(t), h=h)[0]
        return pde(self._fun, t, h=h)

    def oldcentral(self, t, h=DEFAULT_H):
        _t = np.atleast_2d(t)

        shp = np.broadcast_shapes(_t.shape, tuple(np.ones(len(_t.shape), dtype=int)))

        z = np.zeros(shp, dtype=float)
        T = _t.reshape(shp)

        for i in range(shp[0]):
            o = np.zeros(shp[1:], dtype=float)
            for j in range(shp[1]):
                o[j, ...] = h

            z[i, ...] = (self._fun(T[i] + o) - self._fun(T[i] - o)) / (2 * h)
        return self.finalize(z, t)

    def finalize(self, arr, t):
        shp = arr.shape
        cnt = 0
        for i in shp:
            if i != 1:
                break
            else:
                cnt += 1

        zz = remove_dim(arr, cnt) if cnt > 0 else arr
        if zz.shape == (1,) and np.isscalar(t):
            return zz[0]
        else:
            return zz

    def __call__(self, t, h=DEFAULT_H, method="central"):
        return super().__call__(t, h=h, method=method)

    def forward(self, t, h=DEFAULT_H):
        t = np.atleast_1d(t)
        z = np.zeros(t.shape, dtype=float)
        for i in range(len(t)):
            o = np.zeros(len(t), dtype=float)
            o[i] = h
            z[i] = (self._fun(t + o) - self._fun(t)) / h

        return z

    def backward(self, t, h=DEFAULT_H):
        t = np.atleast_1d(t)
        z = np.zeros(t.shape, dtype=float)
        for i in range(len(t)):
            o = np.zeros(len(t), dtype=float)
            o[i] = h
            z[i] = (self._fun(t) - self._fun(t - o)) / h

        return z


class Hess(Grad):
    def central(self, t, h=DEFAULT_H):
        t = np.atleast_1d(t)
        lx, ly = t.shape[0], t.shape[0]
        z = np.zeros((lx, ly, *t.shape[1:]), dtype=float)

        t = np.atleast_1d(t)
        lx, ly = t.shape[0], t.shape[0]
        z = np.zeros((lx, ly, *t.shape[1:]), dtype=float)

        for i in range(lx):
            o = np.zeros(ly, dtype=float)
            o[i] = h
            z[i] = (self._grad(t + o) - self._grad(t - o)) / (2 * h)

        return z

    def backward(self, t, h=DEFAULT_H):
        t = np.atleast_1d(t)
        lx, ly = t.shape[0], t.shape[0]
        z = np.zeros((lx, ly, *t.shape[1:]), dtype=float)

        for i in range(lx):
            o = np.zeros(ly, dtype=float)
            o[i] = h
            z[i, ...] = (self._grad.backward(t + o) - self._grad.backward(t - o)) / (
                    2 * h
            )

        return z

    def forward(self, t, h=DEFAULT_H):
        t = np.atleast_1d(t)
        lx, ly = t.shape[0], t.shape[0]
        z = np.zeros((lx, ly, *t.shape[1:]), dtype=float)

        for i in range(lx):
            o = np.zeros(ly, dtype=float)
            o[i] = h
            z[i, ...] = (self._grad.forward(t + o) - self._grad.forward(t - o)) / (
                    2 * h
            )

        return z


def jac(fun, h=DEFAULT_H):
    def jac_wrap(t):
        inp = len(t)
        H = np.eye(inp) * h
        z = np.zeros((inp, inp))
        for i in range(inp):
            z[i, :] = (fun(t + H[i]) - fun(t - H[i])) / (2 * h)
        return z

    return jac_wrap
