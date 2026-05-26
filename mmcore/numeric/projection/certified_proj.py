# -*- coding: utf-8 -*-
"""
Certified curve tracking (Python port with consistent linear algebra)

Key conventions (Pythonic and consistent):
- Points/vectors are treated as COLUMN vectors.
- The local coordinate transform is:
      x = x0 + R @ u
  where R is (numerically) unitary and its last column is the tangent direction.
- Original -> local:
      u = R^H @ (x - x0)
  (R^H = conjugate transpose)

This removes all confusing transpose/inverse mismatches and makes
tangents / hyperplanes / transforms consistent everywhere.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from blosc2 import NDArray
from mpmath import iv
from scipy.linalg import null_space


# ---------------------------
# Interval / complex-ball layer
# ---------------------------

IVMPC = type(iv.mpc(0, 0))
IVMPF = type(iv.mpf([0, 1]))


def _mpi_mid(x: IVMPF) -> float:
    return (float(x.a) + float(x.b)) / 2.0


def _mpi_rad(x: IVMPF) -> float:
    return (float(x.b) - float(x.a)) / 2.0


def _to_ivmpc(val: Any) -> IVMPC:
    """Convert various numeric types to an mpmath interval complex (iv.mpc)."""
    if isinstance(val, IVMPC):
        return val
    if isinstance(val, IVMPF):
        return iv.mpc(val, iv.mpf(0))
    if isinstance(val, (int, float)):
        return iv.mpc(iv.mpf(val), iv.mpf(0))
    if isinstance(val, complex):
        return iv.mpc(iv.mpf(val.real), iv.mpf(val.imag))
    # Let mpmath attempt the conversion
    return iv.mpc(val, 0)


class ComplexBallField:
    """
    Minimal "complex ball" wrapper compatible with mpmath.iv.

    Supports strings:
      "a +/- r"
      "+/- r"
    with any whitespace.
    Radii are always treated as nonnegative (abs(r)).
    """

    _pm_re = re.compile(
        r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\+/-\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
    )
    _only_rad_re = re.compile(
        r"^\s*\+/-\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
    )

    def _parse_real_ball(self, s: str) -> IVMPF:
        m = self._pm_re.match(s)
        if m:
            mid = float(m.group(1))
            rad = abs(float(m.group(2)))
            a, b = mid - rad, mid + rad
            return iv.mpf([min(a, b), max(a, b)])

        m = self._only_rad_re.match(s)
        if m:
            rad = abs(float(m.group(1)))
            return iv.mpf([-rad, rad])

        # fallback: exact point
        try:
            mid = float(s)
            return iv.mpf([mid, mid])
        except Exception as e:
            raise ValueError(f"Could not parse real ball from string: {s!r}") from e

    def __call__(self, re_part: Any, im_part: Any = None) -> IVMPC:
        if im_part is None:
            if isinstance(re_part, str):
                re_iv = self._parse_real_ball(re_part)
                return iv.mpc(re_iv, iv.mpf(0))
            return _to_ivmpc(re_part)

        if isinstance(re_part, str):
            re_iv = self._parse_real_ball(re_part)
        else:
            re_iv = re_part if isinstance(re_part, IVMPF) else iv.mpf(re_part)

        if isinstance(im_part, str):
            im_iv = self._parse_real_ball(im_part)
        else:
            im_iv = im_part if isinstance(im_part, IVMPF) else iv.mpf(im_part)

        return iv.mpc(re_iv, im_iv)


CC = ComplexBallField()


# ---------------------------
# Polynomial system + PolyEta
# ---------------------------

@dataclass
class PolySystem:
    polys: List[sp.Expr]
    vars: Tuple[sp.Symbol, ...]
    field: ComplexBallField = CC

    def __len__(self) -> int:
        return len(self.polys)


@dataclass
class PolyEta:
    """
    Univariate polynomial in η with interval complex coefficients.
    coeffs[k] is coefficient of η^k.
    """

    coeffs: List[IVMPC]

    @staticmethod
    def const(val: Any) -> "PolyEta":
        if isinstance(val, PolyEta):
            return val
        return PolyEta([_to_ivmpc(val)])

    def _trim(self) -> None:
        z0 = _to_ivmpc(0)
        i = len(self.coeffs) - 1
        while i > 0 and self.coeffs[i] == z0:
            i -= 1
        self.coeffs = self.coeffs[: i + 1]

    def __add__(self, other: Any) -> "PolyEta":
        other = PolyEta.const(other)
        n = max(len(self.coeffs), len(other.coeffs))
        z0 = _to_ivmpc(0)
        out = []
        for i in range(n):
            a = self.coeffs[i] if i < len(self.coeffs) else z0
            b = other.coeffs[i] if i < len(other.coeffs) else z0
            out.append(a + b)
        res = PolyEta(out)
        res._trim()
        return res

    __radd__ = __add__

    def __neg__(self) -> "PolyEta":
        return PolyEta([-c for c in self.coeffs])

    def __sub__(self, other: Any) -> "PolyEta":
        return self + (-PolyEta.const(other))

    def __rsub__(self, other: Any) -> "PolyEta":
        return PolyEta.const(other) - self

    def __mul__(self, other: Any) -> "PolyEta":
        other = PolyEta.const(other)
        a = self.coeffs
        b = other.coeffs
        z0 = _to_ivmpc(0)
        out = [z0 for _ in range(len(a) + len(b) - 1)]
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
        res = PolyEta(out)
        res._trim()
        return res

    def __rmul__(self, other: Any) -> "PolyEta":
        return self * other

    def __pow__(self, exp: Any) -> "PolyEta":
        e = int(exp)
        if e < 0:
            raise ValueError("PolyEta does not support negative exponents.")
        res = PolyEta.const(1)
        base = self
        while e > 0:
            if e & 1:
                res = res * base
            base = base * base
            e >>= 1
        res._trim()
        return res

    def eval(self, eta_val: Any) -> IVMPC:
        eta = _to_ivmpc(eta_val)
        res = _to_ivmpc(0)
        for c in reversed(self.coeffs):
            res = res * eta + c
        return res


# ---------------------------
# Lambdify caches
# ---------------------------

_lambdify_iv_cache: Dict[Tuple[sp.Expr, Tuple[sp.Symbol, ...]], Any] = {}
_lambdify_polyeta_cache: Dict[Tuple[sp.Expr, Tuple[sp.Symbol, ...]], Any] = {}


def _lambdify_iv(expr: sp.Expr, vars_tuple: Tuple[sp.Symbol, ...]):
    key = (expr, vars_tuple)
    fn = _lambdify_iv_cache.get(key)
    if fn is None:
        fn = sp.lambdify(vars_tuple, expr, modules=[iv])
        _lambdify_iv_cache[key] = fn
    return fn


def _lambdify_polyeta(expr: sp.Expr, vars_tuple: Tuple[sp.Symbol, ...]):
    key = (expr, vars_tuple)
    fn = _lambdify_polyeta_cache.get(key)
    if fn is None:
        # Polynomials only => Python operators suffice for PolyEta arithmetic.
        fn = sp.lambdify(vars_tuple, expr, modules=[{}])
        _lambdify_polyeta_cache[key] = fn
    return fn


# ---------------------------
# Utilities: conversion, eval, norms
# ---------------------------

def convert_to_double_int(z: Any) -> complex:
    z = _to_ivmpc(z)
    return complex(_mpi_mid(z.real), _mpi_mid(z.imag))


def convert_to_double_vector(v: Sequence[Any]) -> np.ndarray:
    return np.array([convert_to_double_int(z) for z in v], dtype=complex)


def convert_to_double_matrix(M: Any) -> np.ndarray:
    M = np.asarray(M, dtype=object)
    nr, nc = M.shape
    out = np.empty((nr, nc), dtype=complex)
    for i in range(nr):
        for j in range(nc):
            out[i, j] = convert_to_double_int(M[i, j])
    return out


def midpoint_complex_int(z: Any) -> IVMPC:
    z = _to_ivmpc(z)
    return iv.mpc(iv.mpf(_mpi_mid(z.real)), iv.mpf(_mpi_mid(z.imag)))


def midpoint_complex_box(vec: Sequence[Any]) -> np.ndarray:
    return np.array([midpoint_complex_int(z) for z in vec], dtype=object)


def _pad_vec_to_vars(vec: Sequence[Any], nvars: int, field: ComplexBallField) -> List[IVMPC]:
    v = list(vec)
    if len(v) < nvars:
        v = v + [field(0)] * (nvars - len(v))
    elif len(v) > nvars:
        v = v[:nvars]
    return v


def evaluate_matrix(m: Any, vec: Sequence[Any], vars_tuple: Optional[Tuple[sp.Symbol, ...]] = None) -> np.ndarray:
    """
    Evaluate:
      - PolySystem at vec
      - sympy-expression matrix at vec (requires vars_tuple)
      - PolyEta matrix at eta = vec[-1]
    """
    if isinstance(m, PolySystem):
        vars_tuple = m.vars
        vec_full = _pad_vec_to_vars(vec, len(vars_tuple), m.field)
        out = np.empty((1, len(m.polys)), dtype=object)
        for i, expr in enumerate(m.polys):
            fn = _lambdify_iv(expr, vars_tuple)
            out[0, i] = fn(*vec_full)
        return out

    M = np.asarray(m, dtype=object)

    if M.size > 0 and isinstance(M.flat[0], PolyEta):
        eta_val = vec[-1] if len(vec) > 0 else CC(0)
        eta_val = _to_ivmpc(eta_val)
        out = np.empty(M.shape, dtype=object)
        it = np.nditer(M, flags=["multi_index", "refs_ok"])
        for x in it:
            out[it.multi_index] = x.item().eval(eta_val)
        return out

    if M.size > 0 and isinstance(M.flat[0], sp.Basic):
        if vars_tuple is None:
            raise ValueError("vars_tuple must be provided when evaluating sympy matrices.")
        vec_full = _pad_vec_to_vars(vec, len(vars_tuple), CC)
        out = np.empty(M.shape, dtype=object)
        it = np.nditer(M, flags=["multi_index", "refs_ok"])
        for x in it:
            expr = x.item()
            fn = _lambdify_iv(expr, vars_tuple)
            out[it.multi_index] = fn(*vec_full)
        return out

    return M


def max_int_norm(z: Any) -> float:
    """
    A safe max norm for an interval complex:
      max( |mid(Re)| + rad(Re), |mid(Im)| + rad(Im) )
    """
    z = _to_ivmpc(z)
    r = z.real
    i = z.imag
    rmax = abs(_mpi_mid(r)) + _mpi_rad(r)
    imax = abs(_mpi_mid(i)) + _mpi_rad(i)
    return float(max(rmax, imax))


def max_norm(intmat: Any) -> float:
    M = np.asarray(intmat, dtype=object)
    maxv = 0.0
    it = np.nditer(M, flags=["multi_index", "refs_ok"])
    for x in it:
        v = max_int_norm(x.item())
        if v > maxv:
            maxv = v
    return maxv


def matvec_interval(M: np.ndarray, v: Sequence[Any]) -> np.ndarray:
    """
    Interval matrix-vector product where M is complex float ndarray (n,n),
    and v is length n of iv.mpc.
    """
    M = np.asarray(M, dtype=complex)
    n = M.shape[0]
    out = np.empty(n, dtype=object)
    for i in range(n):
        s = _to_ivmpc(0)
        for j in range(n):
            s += _to_ivmpc(M[i, j]) * _to_ivmpc(v[j])
        out[i] = s
    return out


# ---------------------------
# Jacobians
# ---------------------------

def jac_nsquare(system: PolySystem) -> np.ndarray:
    """Full Jacobian (m x n) for a curve system (m=n-1 equations in n variables)."""
    vars_ = system.vars
    m = len(system.polys)
    n = len(vars_)
    mat = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            d = sp.diff(system.polys[i], vars_[j])
            mat[i, j] = d if d != 0 else 0
    return mat


def jac(system: PolySystem) -> np.ndarray:
    """
    Square Jacobian (m x m) using only the first m variables,
    matching the algorithm's transversal system convention.
    """
    vars_ = system.vars
    m = len(system.polys)
    mat = np.empty((m, m), dtype=object)
    for i in range(m):
        for j in range(m):
            d = sp.diff(system.polys[i], vars_[j])
            mat[i, j] = d if d != 0 else 0
    return mat


def pseudo_inv(eval_jac: np.ndarray) -> Tuple[np.ndarray, IVMPC]:
    """
    Midpoint inverse. Returned as a degenerate interval matrix.
    """
    Mmid = convert_to_double_matrix(eval_jac)
    Minv = np.linalg.inv(Mmid)
    out = np.empty(Minv.shape, dtype=object)
    for i in range(Minv.shape[0]):
        for j in range(Minv.shape[1]):
            out[i, j] = _to_ivmpc(Minv[i, j])
    return out, _to_ivmpc(1)


def jacobian_inverse(system: PolySystem, vec: Sequence[Any]) -> np.ndarray:
    j = jac(system)
    eval_jac = evaluate_matrix(j, vec, vars_tuple=system.vars)
    Jinv, factor = pseudo_inv(eval_jac)
    return Jinv * (1 / factor)


# ---------------------------
# Krawczyk / Moore refinement
# ---------------------------

def krawczyk_operator(system: PolySystem, p: Sequence[Any], r: float, A: np.ndarray) -> np.ndarray:
    m = len(system.polys)
    j = jac(system)
    CCi = system.field
    B = CCi("+/-1", "+/-1")

    mat = np.empty((m, 1), dtype=object)
    for i in range(m):
        mat[i, 0] = B

    idm = np.empty((m, m), dtype=object)
    for i in range(m):
        for j0 in range(m):
            idm[i, j0] = _to_ivmpc(1 if i == j0 else 0)

    eval_sys = evaluate_matrix(system, p)  # 1 x m

    # Box for the first m variables (transversal system ignores the last variable)
    p_box = [_to_ivmpc(p[i]) + r * mat[i, 0] for i in range(m)]
    eval_jac = evaluate_matrix(j, p_box, vars_tuple=system.vars)

    oper = (-1.0 / r) * (A @ eval_sys.T) + (idm - (A @ eval_jac)) @ mat
    K = np.empty((m, 1), dtype=object)
    for i in range(m):
        K[i, 0] = oper[i, 0]
    return K


def krawczyk_test(system: PolySystem, point: Sequence[Any], r: float, A: np.ndarray, rho: float) -> bool:
    K = krawczyk_operator(system, point, r, A)
    return max_norm(K) < rho


def refine_moore_box(
        f: PolySystem,
        x: Sequence[Any],
        r: float,
        A: np.ndarray,
        rho: float,
        max_steps: int = 10_000,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Moore-style refinement loop with a hard cap to avoid infinite loops.
    """
    y = np.array(list(x), dtype=object)

    steps = 0
    while not krawczyk_test(f, y, r, A, rho):
        steps += 1
        if steps > max_steps:
            raise RuntimeError("refine_moore_box exceeded max_steps; Krawczyk test never passed.")

        d = A @ evaluate_matrix(f, y).T
        if max_norm(d) <= (1 / 64) * rho * r:
            r *= 0.5
        else:
            y = midpoint_complex_box(y - d[:, 0])
        A = jacobian_inverse(f, y)

    while 2 * r <= 1 and krawczyk_test(f, x, 2 * r, A, rho):
        r *= 2

    return y, r, A


# ---------------------------
# Predictor / Taylor model
# ---------------------------

def speed_vector(H: PolySystem, x: Sequence[Any], A: np.ndarray) -> np.ndarray:
    """
    v = -A * (∂H/∂t) evaluated at x, where t is the last variable.
    Returned as midpoint complex balls (degenerate intervals).
    """
    vars_ = H.vars
    d_var = vars_[-1]
    m = len(H.polys)
    vals = _pad_vec_to_vars(x, len(vars_), H.field)

    deriv_vals = np.empty((1, m), dtype=object)
    for i in range(m):
        deriv_expr = sp.diff(H.polys[i], d_var)
        fn = _lambdify_iv(deriv_expr, vars_)
        deriv_vals[0, i] = fn(*vals)

    v_col = (-A) @ deriv_vals.T
    return midpoint_complex_box(v_col[:, 0])


def linear_predictor(H: PolySystem, v: Sequence[Any], x: Sequence[Any]) -> List[PolyEta]:
    eta = PolyEta([_to_ivmpc(0), _to_ivmpc(1)])  # η
    out: List[PolyEta] = []
    for i in range(len(v)):
        out.append(PolyEta.const(x[i]) + PolyEta.const(v[i]) * eta)
    return out


def hermite_predictor(
        H: PolySystem,
        x: Sequence[Any],
        xprev: Sequence[Any],
        v: Sequence[Any],
        vprev: Sequence[Any],
        hprev: float,
) -> List[PolyEta]:
    """
    Cubic Hermite predictor where:
      current point is at η=0, previous point at η=-hprev.
    """
    eta = PolyEta([_to_ivmpc(0), _to_ivmpc(1)])
    eta2 = eta ** 2
    eta3 = eta ** 3

    out: List[PolyEta] = []
    for i in range(len(v)):
        vi = _to_ivmpc(v[i])
        vip = _to_ivmpc(vprev[i])
        xi = _to_ivmpc(x[i])
        xpi = _to_ivmpc(xprev[i])

        c2 = (3 * vi / hprev - (vi - vip) / hprev - 3 * (xi - xpi) / (hprev ** 2))
        c3 = (2 * vi / (hprev ** 2) - (vi - vip) / (hprev ** 2) - 2 * (xi - xpi) / (hprev ** 3))

        poly = (
                PolyEta.const(xi)
                + PolyEta.const(vi) * eta
                + PolyEta.const(c2) * eta2
                + PolyEta.const(c3) * eta3
        )
        out.append(poly)

    return out


def taylor_model(H: PolySystem, lp: List[PolyEta], tval: Any, A: np.ndarray, r: float) -> np.ndarray:
    """
    Build the Taylor model vector K(η) used in the proceeding step test.

    IMPORTANT: This function intentionally mutates 'lp' by appending η,
    mirroring Julia's `push!(lp, η)` behavior. After this, lp has length n.
    """
    m = len(H.polys)
    vars_ = H.vars
    CCi = H.field

    eta = PolyEta([_to_ivmpc(0), _to_ivmpc(1)])  # η
    lp.append(eta)  # now len(lp)=m+1 = n

    t_poly = PolyEta.const(tval) + eta

    # Substitute x-vars by lp[i], t-var by tval+η
    args = []
    for i in range(len(vars_)):
        if i < m:
            args.append(lp[i])
        else:
            args.append(t_poly)

    # eH(η)
    eH: List[PolyEta] = []
    for i in range(m):
        fn = _lambdify_polyeta(H.polys[i], vars_)
        val = fn(*args)
        eH.append(val if isinstance(val, PolyEta) else PolyEta.const(val))

    # square Jacobian wrt first m variables
    ejac = np.empty((m, m), dtype=object)
    for i in range(m):
        for j in range(m):
            ejac[i, j] = sp.diff(H.polys[i], vars_[j])

    # evaluate Jacobian on a box of radius r in x-vars (not in η)
    B = CCi("+/-1", "+/-1")
    args_box = []
    for i in range(len(vars_)):
        if i < m:
            args_box.append(lp[i] + PolyEta.const(r * B))
        else:
            args_box.append(t_poly)

    eHjac = np.empty((m, m), dtype=object)
    for i in range(m):
        for j in range(m):
            fn = _lambdify_polyeta(ejac[i, j], vars_)
            val = fn(*args_box)
            eHjac[i, j] = val if isinstance(val, PolyEta) else PolyEta.const(val)

    # A as PolyEta constants
    A_poly = np.empty_like(A, dtype=object)
    for i in range(m):
        for j in range(m):
            A_poly[i, j] = PolyEta.const(A[i, j])

    # identity
    idm = np.empty((m, m), dtype=object)
    for i in range(m):
        for j in range(m):
            idm[i, j] = PolyEta.const(1 if i == j else 0)

    mat_vec = np.empty((m, 1), dtype=object)
    for i in range(m):
        mat_vec[i, 0] = PolyEta.const(B)

    eH_col = np.empty((m, 1), dtype=object)
    for i in range(m):
        eH_col[i, 0] = eH[i]

    tm = (-1.0 / r) * (A_poly @ eH_col) + (idm - (A_poly @ eHjac)) @ mat_vec
    return tm


def proceeding_step(h: float, CCi: ComplexBallField, nvars: int, tm: np.ndarray, K: np.ndarray) -> float:
    """
    Halve h until max_norm(K) <= 7/8.
    Uses an η-interval of radius h/2.
    """
    while max_norm(K) > 7 / 8:
        h *= 0.5
        radii = h / 2
        if abs(h) < 1e-10:
            raise RuntimeError("h is too small!")

        T = CCi(f"{radii} +/- {radii}")
        input_vec = [CCi(0)] * (nvars + 1)
        input_vec[-1] = T
        K = evaluate_matrix(tm, input_vec)

    return h


# ---------------------------
# Core: system transform + tracking
# ---------------------------

def system_transform(F: PolySystem, x: Sequence[Any]) -> Tuple[np.ndarray, PolySystem, PolySystem]:
    """
    Build a local chart at x:

      x = x0 + R @ u

    where R is (numerically) unitary and its last column is the tangent direction.
    The transformed system is:

      H(u) = F(x0 + R u)

    and Ft is H with the last variable fixed to 0 (transversal slice).
    """
    vars_ = F.vars
    n = len(vars_)
    m = len(F.polys)

    if m != n - 1:
        raise ValueError(f"Expected a curve system with m=n-1 equations, got m={m}, n={n}.")

    x0_iv = midpoint_complex_box(x)
    x0_mid = convert_to_double_vector(x0_iv)

    # Jacobian at x0 (midpoint)
    J_expr = jac_nsquare(F)  # m x n sympy expressions
    J_val = evaluate_matrix(J_expr, x0_iv, vars_tuple=vars_)
    J_mid = convert_to_double_matrix(J_val)

    # Tangent direction: smallest right singular vector of J_mid
    _, _, Vh = np.linalg.svd(J_mid, full_matrices=True)
    v = Vh.conj().T[:, -1]
    nv = np.linalg.norm(v)
    if nv == 0:
        raise RuntimeError("Tangent computation failed (zero vector).")
    v = v / nv

    # Orthonormal complement basis (hyperplane orthogonal to v)
    W = null_space(v.conj().reshape(1, n))  # n x (n-1)
    if W.shape[1] != n - 1:
        raise RuntimeError("Failed to compute orthonormal complement to tangent.")

    # R maps local u-coordinates -> original coordinates (new->old)
    R = np.column_stack([W, v.reshape(n, 1)])  # n x n

    # Build substitution: old vars -> x0 + R @ (new vars)
    subs_map: Dict[sp.Symbol, sp.Expr] = {}
    for i in range(n):
        lin = x0_mid[i]
        for j in range(n):
            lin += R[i, j] * vars_[j]
        subs_map[vars_[i]] = sp.expand(lin)

    H_polys = [sp.expand(poly.subs(subs_map, simultaneous=True)) for poly in F.polys]
    H = PolySystem(H_polys, vars_, F.field)

    # transversal system: fix the tangent coordinate to 0
    t_var = vars_[-1]
    Ft_polys = [sp.expand(poly.subs({t_var: 0}, simultaneous=True)) for poly in H.polys]
    Ft = PolySystem(Ft_polys, vars_, F.field)

    return R, H, Ft


def refine_step(
        H: PolySystem,
        Ft: PolySystem,
        x: Sequence[Any],
        r: float,
        A: np.ndarray,
        h: float,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, float, float]:
    """
    Refine transversal coordinates and compute speed vector.
    """
    n = len(x)
    m = n - 1

    trunc_x = np.array(list(x[:m]), dtype=object)
    trunc_x, r, A = refine_moore_box(Ft, trunc_x, r, A, 1 / 8)

    x_new = np.empty(n, dtype=object)
    x_new[:m] = trunc_x
    x_new[m] = x[m]

    v = speed_vector(H, x_new, A)

    h *= 5 / 4
    radii = h / 2
    return x_new, r, A, v, h, radii


def safe_path(file_name: str) -> str:
    if file_name.startswith("~"):
        return os.path.expanduser(file_name)
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(os.getcwd(), file_name)


def pretty_print_status(x: Sequence[Any], iter_: int, total_iter: int) -> None:
    coords = []
    for xi in x:
        xi = _to_ivmpc(xi)
        mid_real = round(_mpi_mid(xi.real), 6)
        err_real = float(f"{_mpi_rad(xi.real):.2g}")
        coords.append(f"{mid_real} ± {err_real}")
    out = f"[Iteration {iter_}/{total_iter}] Tracking point: x = [" + ", ".join(coords) + "]"
    print("\r" + out, end="")
    import sys
    sys.stdout.flush()


def track_curve(
        F: PolySystem,
        x: Sequence[Any],
        r: float,
        max_iter: int,
        file_name: str,
        show_tubular_neighborhood: bool = False,
        figure_scale: float = 1.0,
        box_thickness: float = 0.1,
        line_thickness: float = 0.2,
        bounds:Optional[Sequence[tuple[float,float]]]=((-1.,1.),(-1.,1.),(-1.,1.)),
        out:Optional[NDArray]=None,

        out_h:Optional[NDArray]=None,

) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Main curve tracking loop.

    Writes TikZ output to file_name + ".tex".
    """
    if out is None:
        out=np.empty((max_iter,len(x)),dtype=float)

    if out_h is None:
            out_h = np.empty((max_iter, 1))
    CCi = F.field
    n = len(x)
    m = n - 1
    h = 0.5
    initial_x=np.array(convert_to_double_vector(x))
    x = np.array(list(x), dtype=object)

    #tex_path = safe_path(f"{file_name}.txt")
    #os.makedirs(os.path.dirname(tex_path) or ".", exist_ok=True)

    # --- initial transform + first predictor
    R, H, Ft = system_transform(F, x)
    A = jacobian_inverse(Ft, [CCi(0)] * n)

    u_x, r, A, v, h, radii = refine_step(H, Ft, [CCi(0)] * n, r, A, h)

    X = linear_predictor(H, v, u_x[:m])
    t = u_x[-1]
    tm = taylor_model(H, X, t, A, r)

    T = CCi(f"{radii} +/- {radii}")
    input_vec = [CCi(0)] * (n + 1)
    input_vec[-1] = T
    K = evaluate_matrix(tm, input_vec)

    h = proceeding_step(h, CCi, n, tm, K)
    u_x[-1] = _to_ivmpc(u_x[-1]) + h

    xprev = x.copy()
    delta = matvec_interval(R, u_x)
    x = midpoint_complex_box(delta) + x

    hprev = h
    vprev = v
    rev_count = 0



    iter_ = 0
    while iter_ < max_iter:
            # local chart at current x
            R, H, Ft = system_transform(F, x)
            A = jacobian_inverse(Ft, [CCi(0)] * n)

            u_x, r, A, v, h, radii = refine_step(H, Ft, [CCi(0)] * n, r, A, h)

            # previous point in the current u-coordinates (η = -hprev)
            delta_prev_mid = convert_to_double_vector(xprev - x)  # previous - current
            u_prev_mid = R.conj().T @ delta_prev_mid
            u_prev = np.array([CCi(complex(z)) for z in u_prev_mid], dtype=object)

            X = hermite_predictor(H, u_x[:m], u_prev[:m], v, vprev, hprev)
            t = u_x[-1]
            tm = taylor_model(H, X, t, A, r)

            T = CCi(f"{radii} +/- {radii}")
            input_vec = [CCi(0)] * (n + 1)
            input_vec[-1] = T
            K = evaluate_matrix(tm, input_vec)

            # tube point b (for drawing)
            X_eval = evaluate_matrix(np.array([X], dtype=object), input_vec)  # 1 x n (X has η appended)
            u_pred = X_eval.reshape(-1)

            b = matvec_interval(R, u_pred) + x

            h = proceeding_step(h, CCi, n, tm, K)
            u_x[-1] = _to_ivmpc(u_x[-1]) + h

            # reversal detection: compare previous step direction with new predicted displacement
            step_vec_mid = convert_to_double_vector(x - xprev)  # current - previous
            step_u_mid = R.conj().T @ step_vec_mid
            dot = float(np.real(np.dot(step_u_mid, convert_to_double_vector(u_x))))  # transpose (no conjugate)

            if dot < 0:
                rev_count += 1
                u_x[-1] = _to_ivmpc(u_x[-1]) - 2 * h
                radii = -radii

                T = CCi(f"{radii} +/- {radii}")
                input_vec = [CCi(0)] * (n + 1)
                input_vec[-1] = T
                X_eval = evaluate_matrix(np.array([X], dtype=object), input_vec)
                u_pred = X_eval.reshape(-1)
                b = matvec_interval(R, u_pred) + x

            # update point
            old_x = x.copy()
            delta = matvec_interval(R, u_x)
            x = midpoint_complex_box(delta) + x

            xyz =  tuple(convert_to_double_int(__x).real for __x in x)
            _break=False
            for _i in range(  out.shape[1]):
                if xyz[_i]>bounds[_i][1] or xyz[_i]<bounds[_i][0] :

                    print('bbbbb')
                    _break=True
            if _break:
                break
            out[iter_]=xyz
            xprev = old_x




            #out_v[iter_,:] = convert_to_double_int(v)
            #out[iter_, :] = convert_to_double_int(x[0]).real, convert_to_double_int(x[1]).real
            out_h[iter_,:] = h

            pretty_print_status(x, iter_, max_iter)

            # drawing (only meaningful if variables include x,y as first two coordinates)
            if show_tubular_neighborhood and iter_ > 5 and n >= 2:
                b1 = _to_ivmpc(b[0])
                b2 = _to_ivmpc(b[1])

                x_mid = convert_to_double_int(b1).real
                y_mid = convert_to_double_int(b2).real

                x_min = x_mid - _mpi_rad(b1.real)
                x_max = x_mid + _mpi_rad(b1.real)
                y_min = y_mid - _mpi_rad(b2.real)
                y_max = y_mid + _mpi_rad(b2.real)

                #file.write(f"[[[{x_min},{y_min}],[{x_max},{y_min}]], [[{x_min},{y_min}],[{x_max},{y_max}]],[[{x_min},{y_max}],[{x_max},{y_max}]],[[{x_max},{y_max}],[{x_max},{y_max}]]],\n")


            #if iter_ > 5 and n >= 2:
            #    x1 = convert_to_double_int(x[0]).real
            #    y1 = convert_to_double_int(x[1]).real
            #    x0 = convert_to_double_int(xprev[0]).real
            #    y0 = convert_to_double_int(xprev[1]).real
            #    #file.write(f"[[{x1},{y1}] ,[{x0},{y0}]],\n")

            hprev = h
            vprev = v
            iter_ += 1

        #file.write("\n]")



    #print()  # finish the carriage-return line
    return x, r, A, out, out_h,iter_



# ---------------------------
# Examples (matching the Julia paper examples)
# ---------------------------

def run_figure3_example() -> Tuple[np.ndarray, float, np.ndarray]:
    CCi = CC
    x, y, z = sp.symbols("x y z")

    # Julia: F = [-z-x^3+2.7*x  y^2-2+z]  (1x2 matrix literal => two equations)
    F = PolySystem(
        [
            -z - x**3 + sp.Rational(27, 10) * x,
            y**2 - 2 + z,
            ],
        (x, y, z),
        CCi,
    )

    p = [CCi(3), CCi(3), CCi(-1)]
    return track_curve(F, p, r=0.1, bounds=[(-3,3),(-3,3),(-float('inf'),float('inf'))], max_iter=450, file_name="curve_fig3", figure_scale=1.)



def run_krtz_example() -> Tuple[np.ndarray, float, np.ndarray]:
    CCi = CC
    x, y, t = sp.symbols("x y t")

    # Julia: F = [x-...-2   y-...+7*t]  (two equations)
    F = PolySystem(
        [
            x - t**8 + 8 * t**6 - 20 * t**4 + 16 * t**2 - 2,
            y - t**7 + 7 * t**5 - 14 * t**3 + 7 * t,
            ],
        (x, y, t),
        CCi,
    )

    p = [CCi(-1), CCi(0), CCi(1)]
    return track_curve(F, p, r=0.1, max_iter=345, file_name="curve_krtz", figure_scale=2)


def run_mggj_example() -> Tuple[np.ndarray, float, np.ndarray]:
    CCi = CC
    x, y = sp.symbols("x y")

    # Use an exact rational e for stability (keeps the start point exactly on the curve).
    e = sp.Rational(99, 100)

    poly = (
            x**8
            - (1 - e) * x**6
            + 4 * x**6 * y**2
            - (3 + 15 * e) * x**4 * y**2
            + 6 * x**4 * y**4
            - (3 - 15 * e) * x**2 * y**4
            + 4 * x**2 * y**6
            - (1 + e) * y**6
            + y**8
    )

    F = PolySystem([poly], (x, y), CCi)

    # sqrt(1 - 0.99) = 0.1 exactly with the rational choice above.
    p = [CCi(0.1), CCi(0)]
    return track_curve(F, p, r=0.1, max_iter=1400, bounds=[(-3,3),(-3,3),(-float('inf'),float('inf'))],file_name="curve_mggj", figure_scale=4.5)


def run_figure5_example() -> Tuple[np.ndarray, float, np.ndarray]:
    CCi = CC
    x, y, z = sp.symbols("x y z")

    # Julia: F = [x+z^5-1.3*z^3   y-z^3+z]  (two equations)
    F = PolySystem(
        [
            x + z**5 - sp.Rational(13, 10) * z**3,
            y - z**3 + z,
            ],
        (x, y, z),
        CCi,
    )

    p = [CCi(-1.2), CCi(0), CCi(1)]
    return track_curve(F, p, r=0.1, max_iter=550, file_name="curve_fig5", figure_scale=1.5)


if __name__ == "__main__":
    r1=run_figure3_example()
    r2=run_mggj_example()