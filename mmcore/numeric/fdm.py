import dataclasses
import functools
import math
import sys
from .vectors import scalar_dot
from typing import Callable, Optional, Iterable

import numpy as np
from scipy.optimize import minimize

DEFAULT_H = 1e-3
_DECIMALS = 3
from scipy.sparse import eye, csr_matrix

_PDE_H = csr_matrix(eye(128))


def _get_pde_h(dim):
    return _PDE_H[:dim, :dim]


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
        return lambda t: (f(t + h) - f(t - h)) / (2 * h)
    elif method == "forward":
        return lambda t: (f(t + h) - f(t)) / h
    elif method == "backward":
        return lambda t: (f(t) - f(t - h)) / h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


def bounded_fdm(f, h=DEFAULT_H, bounds=(0., 1.)):
    _decimals = _DECIMALS

    def wrp(t):
        if abs(bounds[0] - t) <= h:
            return (f(t + h) - f(t)) / h
        elif abs(bounds[1] - t) <= h:
            return (f(t) - f(t - h)) / h
        else:
            return (f(t + h) - f(t - h)) / (2 * h)

    return wrp


def pde(f, H, h=DEFAULT_H, bounds=(0., 1.)):
    def wrp(t):

        ts, te = t + H, t - H
        if abs(bounds[0] - t) <= h:
            return (f(ts) - f(t)) / h
        elif abs(bounds[1] - t) <= h:
            return (f(t) - f(te)) / h
        else:
            return (f(ts) - f(te)) / (2 * h)

    return wrp


class PDE:
    def __init__(self, f, dim=3, h=DEFAULT_H):
        self.f = f
        self.h = h
        self._dim = dim
        self.H = _get_pde_h(dim) * self.h
        self._full_dim_select = np.arange(self.dim)

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, v):
        self._dim = v
        self.H = _get_pde_h(self._dim) * self.h

    def _wrp(self, t, H, bounds=None):
        if H is None:
            H = self.H
        ts, te = t + H, t - H

        if bounds:
            if abs(bounds[0] - t) <= self.h:
                return (self.f(ts) - self.f(t)) / self.h
            elif abs(bounds[1] - t) <= self.h:
                return (self.f(t) - self.f(te)) / self.h
            else:
                return (self.f(ts) - self.f(te)) / (2 * self.h)
        else:
            return (self.f(ts) - self.f(te)) / (2 * self.h)

    def dx(self, t, bounds=None):

        return self._wrp(t, self.get_h(np.array((0,), dtype=int)), bounds=bounds)

    def dy(self, t, bounds=None):

        return self._wrp(t, self.get_h(np.array((1,), dtype=int)), bounds=bounds)

    def get_h(self, axes):
        _dH = np.zeros(self.dim, dtype=float)
        _dH[axes] = self.h
        return np.diag(_dH)

    def dz(self, t, bounds=None):

        return self._wrp(t, self.get_h(np.array((2,), dtype=int)), bounds=bounds)

    def dn(self, t, i, bounds=None):

        return self._wrp(t, self.get_h(np.array((i,), dtype=int)), bounds=bounds)

    def dxy(self, t, bounds=None):

        return self._wrp(t, self.get_h(np.array((0, 1), dtype=int)), bounds=bounds)

    def dyz(self, t, bounds=None):

        return self._wrp(t, self.get_h(np.array((1, 2), dtype=int)), bounds=bounds)

    def dzx(self, t, bounds=None):

        return self._wrp(t, self.get_h(np.array((0, 2), dtype=int)), bounds=bounds)

    def dnn(self, t, ij, bounds=None):

        return self._wrp(t, self.get_h(ij), bounds=bounds)

    def __call__(self, t, ijk=None, bounds=None):
        if ijk is None:
            ijk = self._full_dim_select
        return self.dnn(t, ijk, bounds=bounds)


class FDM:
    def __new__(cls, fun=None):
        obj = super().__new__(cls)

        obj._fun = fun

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
        return getattr(self, method)(t, h=h)


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


def _construct_fdm_arguments(x, h=DEFAULT_H):
    b = x.shape[-1]
    arg_matrix = np.eye(b) * h + x
    return arg_matrix


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

class Hess:
    def __init__(self, fun, h=DEFAULT_H):
        self.fun = fun
        self.h = h

    def __call__(self, x):
        return hessian(self.fun, x, h=self.h)


# Step 3: Numerical Hessian calculation
def hessian(f, point, h=DEFAULT_H):
    point = np.asarray(point)
    n = point.size
    H = np.zeros((n, n))
    fp = f(point)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                forward_i = point.copy()
                backward_i = point.copy()
                forward_i[i] += h
                backward_i[i] -= h
                H[i, i] = (f(forward_i) - 2 * fp + f(backward_i)) / h ** 2
            else:
                forward_i_j = point.copy()
                forward_i_j[i] += h
                forward_i_j[j] += h

                forward_i_backward_j = point.copy()
                forward_i_backward_j[i] += h
                forward_i_backward_j[j] -= h

                backward_i_forward_j = point.copy()
                backward_i_forward_j[i] -= h
                backward_i_forward_j[j] += h

                backward_i_j = point.copy()
                backward_i_j[i] -= h
                backward_i_j[j] -= h

                H[i, j] = H[j, i] = (f(forward_i_j) - f(forward_i_backward_j) - f(backward_i_forward_j) + f(
                    backward_i_j)) / (4 * h ** 2)
    return H





def jac(fun, h=DEFAULT_H):
    def jac_wrap(t):
        inp = len(t)
        H = np.eye(inp) * h
        z = np.zeros((inp, inp))
        for i in range(inp):
            z[i, :] = (fun(t + H[i]) - fun(t - H[i])) / (2 * h)
        return z

    return jac_wrap

def gradient(f, point, h=DEFAULT_H):
    point = np.asarray(point)
    grad = np.zeros_like(point)
    for i in range(point.size):

        point[i] += h
        f_plus_h = f(point)
        point[i] -= 2 * h
        f_minus_h = f(point)
        point[i] += h
        grad[i] = (f_plus_h - f_minus_h) / (2 * h)
    return grad


def newtons_method(f, initial_point, tol=DEFAULT_H, max_iter=100):
    point = np.asarray(initial_point)
    for _ in range(max_iter):
        grad = gradient(f, point)
        H = hessian(f, point)
        try:

            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("Hessian is singular at the point", point)
            break
        step = H_inv @ grad
        new_point = point - step
        if np.linalg.norm(new_point - point) < tol:
            break
        point = new_point
    return point



def calculate_bnds(surface, centroid=(0., 0., 0.)):
    cons = {'type': 'eq', 'fun': surface}

    axis = np.eye(3)
    bounds = np.zeros((2, 3))

    c = np.array(centroid, dtype=float)
    for j, n in enumerate(axis):
        x1 = minimize(lambda point: -1 * scalar_dot(n, point - c), centroid,
                      constraints=cons)
        x2 = minimize(lambda point: -1 * scalar_dot(-n, point - c), centroid,
                      constraints=cons)
        bounds[0][j] = x1.x[j]

        bounds[1][j] = x2.x[j]

    return bounds
if __name__ == '__main__':
    from mmcore.geom.primitives import Cylinder

    x, y, v, u, z = [[[12.359112840551504, -7.5948049557495425, 0.0], [2.656625109045951, 1.2155741170561933, 0.0]],
                     [[7.14384241216015, -6.934735074711716, -0.1073366304415263],
                      [7.0788761013028365, 10.016931402130641, 0.8727530304189204]],
                     [[8.072688942425103, -2.3061831591019826, 0.2615779273274319],
                      [7.173685617288537, -3.4427234423361512, 0.4324928834164773],
                      [7.683972288682133, -2.74630545102506, 0.07413871667321925],
                      [7.088944240699163, -4.61458155002528, -0.22460509818398067],
                      [7.304629277158477, -3.9462033818505433, 0.8955725109783643],
                      [7.304629277158477, -3.3362864951018985, 0.8955725109783643],
                      [7.304629277158477, -2.477065729786164, 0.7989970582016114],
                      [7.304629277158477, -2.0988672326949933, 0.7989970582016114]], 0.72648, 1.0]

    aa = np.array(x)
    bb = np.array(y)
    t11 = Cylinder(aa[0], aa[1], z)
    t21 = Cylinder(bb[0], bb[1], u)
    from mmcore.numeric.fdm import newtons_method, Grad


    def uu(xyz):
        a, b = t11.implicit(xyz), t21.implicit(xyz)
        print(xyz,a, b)
        return a * b


    res=newtons_method(uu, np.array([3.,5.,10.]),tol=1e-8, max_iter=100)


    print(t11.implicit(res),t21.implicit(res) )