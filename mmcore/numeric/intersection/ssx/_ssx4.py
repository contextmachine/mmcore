from __future__ import annotations

import contextlib
import inspect
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import currentframe

from mmcore.construction import nurbs_curve
from mmcore.geom._nurbs_eval import evaluate_nurbs_curve, evaluate_nurbs_surface, NURBSSurfaceTuple
from mmcore.geom._nurbs_interp import interpolate_curve, interpolate_nurbs_curve

import functools
import pickle
from dataclasses import dataclass, field

from mmcore.geom._nurbs_eval import (
    NURBSCurveTuple,
    _nurbs_to_tuple,
    _surface_interval,
)
from mmcore.numeric import fdm
from mmcore.numeric.intersection.ssx.trace_inter_segm import (
    _from_homogeneous,
    _to_homogeneous,
    _eval_tensor_bezier,
    _derivative_net_u,
    _derivative_net_v,
    trace_between,
    remove_knots_after_merge,
)
from mmcore.numeric.bern import bern_eval, bernstein_cutout_box_nd
from mmcore.numeric.intersection._deflate import (
    analyse_deflated_system,
    minors_Tpsi_from_control_nets, DeflatedSystem,
)
from mmcore.numeric.ndinterval import get_iarray, interval as iv_interval
from mmcore.numeric.sbern import bern_to_nurbs_bezier


import numpy as np
from numpy.typing import NDArray

from mmcore.geom._nurbs_knots import decompose_surface, link_curves, reverse_curve
from mmcore.numeric._aabb import aabb, aabb_intersect_fast_3d, aabb_intersection
from mmcore.numeric.algorithms.cygjk import gjk
from mmcore.numeric.vectors import unit
from mmcore.numeric.gauss_map import compute_gauss_map_rational
from mmcore.numeric.intersection.csx._bez_csx3 import bez_csx
from mmcore.numeric.intersection.ssx.refine import refine_intersection_point
from mmcore.geom.bvh.lbvh import AABB, build_bvh, bvh_intersect
TIME_PROF_PRINT=False
def time_prof(func) :
    @functools.wraps(func)
    def wrapper(*args, **kwargs ):
        global TIME_PROF_PRINT
        if TIME_PROF_PRINT:
            p=time.perf_counter_ns()
        res=func(*args, **kwargs)
        if TIME_PROF_PRINT:
            delta=time.perf_counter_ns()-p

            print(f'{func.__name__}({kwargs})->({res})', delta*1e-9)
        return res
    return wrapper

# ======================================================================================
# Homogeneous helpers (FIXED)
# ======================================================================================


# ======================================================================================
# Tensor-product Bezier evaluation + 1D splits
# ======================================================================================


def split_tensor_bezier_axis(net: NDArray[np.float64], t: float, axis: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Split tensor-product Bezier control net along one axis using de Casteljau.
    net shape: (p+1, q+1, dim)  axis=0 splits u, axis=1 splits v.
    Returns (left, right) control nets with same shape.
    """
    net = np.asarray(net, dtype=np.float64)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (u) or 1 (v)")

    A = np.swapaxes(net, 0, axis)  # bring split axis to front
    d = A.shape[0] - 1

    left = np.empty_like(A)
    right = np.empty_like(A)

    tmp = A.copy()
    left[0] = tmp[0]
    right[d] = tmp[d]

    omt = 1.0 - t
    for r in range(1, d + 1):
        tmp = omt * tmp[:-1] + t * tmp[1:]
        left[r] = tmp[0]
        right[d - r] = tmp[-1]

    left = np.swapaxes(left, 0, axis)
    right = np.swapaxes(right, 0, axis)
    return left, right


def _clamp01(x: float, eps: float = 1e-6) -> float:
    if x <= eps:
        return eps
    if x >= 1.0 - eps:
        return 1.0 - eps
    return x


# ======================================================================================
# Conservative bounding-plane (slab) separation test
# ======================================================================================

def _interval_proj(P: NDArray[np.float64], a: NDArray[np.float64]) -> tuple[float, float]:
    s = P @ a
    return float(s.min()), float(s.max())


def _intervals_separated(i1: tuple[float, float], i2: tuple[float, float], tol: float) -> bool:
    # "tol" is the required gap before declaring separation (conservative if tol > 0)
    return (i1[1] < i2[0] - tol) or (i2[1] < i1[0] - tol)

# @time_prof
def bounding_plane_separated(
    P1: NDArray[np.float64],
    P2: NDArray[np.float64],
    axes: list[NDArray[np.float64]],
    *,
    tol: float,
    eps: float = 1e-12,
) -> bool:
    for a in axes:
        a = np.asarray(a, dtype=np.float64)
        na = float(np.linalg.norm(a))
        if na <= eps:
            continue
        a = a / na
        if _intervals_separated(_interval_proj(P1, a), _interval_proj(P2, a), tol):
            return True
    return False


# ======================================================================================
# Hemisphere witness via incremental (expected-linear) minimal spherical cap
# ======================================================================================

def _unit_rows(X: NDArray[np.float64], eps: float = 1e-12) -> NDArray[np.float64]:
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    n = np.linalg.norm(X, axis=1)
    keep = n > eps
    if not np.any(keep):
        return X[:0]
    X = X[keep]
    n = n[keep]
    return X / n[:, None]


def _orthogonal_unit(v: NDArray[np.float64], eps: float = 1e-12) -> NDArray[np.float64]:
    ax = np.array([1.0, 0.0, 0.0])
    ay = np.array([0.0, 1.0, 0.0])
    a = ax if abs(v[0]) < 0.9 else ay
    u = np.cross(v, a)
    nu = np.linalg.norm(u)
    if nu <= eps:
        a = np.array([0.0, 0.0, 1.0])
        u = np.cross(v, a)
        nu = np.linalg.norm(u)
    return u / max(nu, eps)


def _cap_contains(c: NDArray[np.float64], m: float, p: NDArray[np.float64], tol: float) -> bool:
    return float(np.dot(c, p)) >= (m - tol)


def _cap_from_1(p: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    p = p / np.linalg.norm(p)
    return p, 1.0


def _cap_from_2(a: NDArray[np.float64], b: NDArray[np.float64], eps: float = 1e-12) -> tuple[NDArray[np.float64], float]:
    s = a + b
    ns = float(np.linalg.norm(s))
    if ns <= eps:
        c = _orthogonal_unit(a, eps=eps)
        return c, 0.0
    c = s / ns
    m = float(np.dot(c, a))
    return c, m


def _cap_from_3(a: NDArray[np.float64], b: NDArray[np.float64], d: NDArray[np.float64],
                tol: float = 1e-12, eps: float = 1e-12) -> tuple[NDArray[np.float64], float]:
    # if a 2-point cap already contains the third (obtuse case)
    best = None
    for x, y, z in ((a, b, d), (a, d, b), (b, d, a)):
        c2, m2 = _cap_from_2(x, y, eps=eps)
        if _cap_contains(c2, m2, z, tol):
            if (best is None) or (m2 > best[1]):
                best = (c2, m2)
    if best is not None:
        return best

    # acute case: 3-point cap
    n = np.cross(b - a, d - a)
    nn = float(np.linalg.norm(n))
    if nn <= eps:
        # nearly collinear; fall back to best pair cap
        c2ab, m2ab = _cap_from_2(a, b, eps=eps)
        c2ad, m2ad = _cap_from_2(a, d, eps=eps)
        c2bd, m2bd = _cap_from_2(b, d, eps=eps)
        c, m = max([(c2ab, m2ab), (c2ad, m2ad), (c2bd, m2bd)], key=lambda t: t[1])
        m = float(min(np.dot(c, a), np.dot(c, b), np.dot(c, d)))
        return c, m

    c = n / nn
    m = float(np.dot(c, a))
    if m < 0.0:
        c = -c
        m = -m
    m = float(min(np.dot(c, a), np.dot(c, b), np.dot(c, d)))
    return c, m


_RNG = np.random.default_rng(0)

# @time_prof
def hemisphere_witness_incremental(
    normals: NDArray[np.float64],
    *,
    eps: float = 1e-8,
    tol: float = 1e-12,
    shuffle: bool = True,
) -> tuple[NDArray[np.float64], float] | None:
    """
    Returns (center, margin) with margin = min_i center·n_i.
    If margin <= eps -> None.
    """
    N = _unit_rows(normals, eps=tol)
    k = N.shape[0]
    if k == 0:
        return None
    if k == 1:
        return N[0], 1.0

    # cheap mean test
    s = N.sum(axis=0)
    ns = float(np.linalg.norm(s))
    if ns > tol:
        c0 = s / ns
        m0 = float((N @ c0).min())
        if m0 > eps:
            return c0, m0

    # randomized incremental
    P = N[_RNG.permutation(k)] if shuffle else N

    c, m = _cap_from_1(P[0])

    for i in range(1, k):
        if _cap_contains(c, m, P[i], tol):
            continue
        c, m = _cap_from_1(P[i])

        for j in range(i):
            if _cap_contains(c, m, P[j], tol):
                continue
            c, m = _cap_from_2(P[i], P[j], eps=tol)
            if m <= eps:
                return None

            for t in range(j):
                if _cap_contains(c, m, P[t], tol):
                    continue
                c, m = _cap_from_3(P[i], P[j], P[t], tol=tol, eps=tol)
                if m <= eps:
                    return None

    m_final = float((N @ c).min())
    if m_final <= eps:
        return None
    return c, m_final
try:
    from mmcore.numeric._cap_witness import hemisphere_witness_incremental as hemisphere_witness_incremental_fast
except Exception:
    hemisphere_witness_incremental_fast = hemisphere_witness_incremental  # fallback (your python version)
    print('using hemisphere_witness_incremental')
# @time_prof


def separate_gauss_maps(N1, N2, *, eps=1e-8, tol=1e-12):
    # Choose order to reduce average calls (optional but good):

    m1 = N1.mean(axis=0); m2 = N2.mean(axis=0)

    if np.dot(m1, m2) >= 0.0:
        # likely same-side -> P1 (opposite) more likely infeasible => run it first
        w1 = hemisphere_witness_incremental_fast(np.vstack([N1, -N2]), eps=eps, tol=tol, shuffle=True)
        if w1 is None:
            return None, None
        P1, _ = w1
        w2 = hemisphere_witness_incremental_fast(np.vstack([N1,  N2]), eps=eps, tol=tol, shuffle=True)
        if w2 is None:
            return None, None
        P2, _ = w2
        return P1, P2
    else:
        # likely opposite -> same-side P2 more likely infeasible first
        w2 = hemisphere_witness_incremental_fast(np.vstack([N1,  N2]), eps=eps, tol=tol, shuffle=True)
        if w2 is None:
            return None, None
        P2, _ = w2
        w1 = hemisphere_witness_incremental_fast(np.vstack([N1, -N2]), eps=eps, tol=tol, shuffle=True)
        if w1 is None:
            return None, None
        P1, _ = w1
        return P1, P2


# ======================================================================================
# GaussMapBern: cache everything, 1D splits, and optional Newton "magic point"
# ======================================================================================

class GaussMapBern:
    __slots__ = (
        "surface", "_map",
        "children",
        "_surf_pts", "_bbox", "_diag2", "_center",
        "_map_dirs_net", "_map_dirs_flat",
        "_mean_n", "_gauss_radius", "_var_u", "_var_v",
        "_plane_n", "_rank_quality",
        "_Hu_ctrl", "_Hv_ctrl",
    )

    def __init__(self, mp: NDArray[np.float64], surf: NDArray[np.float64]):
        self.surface = np.asarray(surf, dtype=np.float64)  # (p+1,q+1,4)
        self._map = np.asarray(mp, dtype=np.float64)       # (r+1,s+1,4) (Gauss map in homogeneous form)
        self.children: list[GaussMapBern] = []

        # caches
        self._surf_pts = None
        self._bbox = None
        self._diag2 = None
        self._center = None

        self._map_dirs_net = None
        self._map_dirs_flat = None

        self._mean_n = None
        self._gauss_radius = None
        self._var_u = None
        self._var_v = None

        self._plane_n = None
        self._rank_quality = None

        self._Hu_ctrl = None
        self._Hv_ctrl = None

    @classmethod
    def from_surf(cls, surf: NDArray[np.float64], rational: bool = False) -> "GaussMapBern":
        # ensure homogeneous
        if not rational:
            surf = _to_homogeneous(surf, np.ones(surf.shape[:-1], dtype=np.float64))

        # build gauss map control net once
        maph = compute_gauss_map_rational(surf)

        # normalize dehomogenized xyz directions onto unit sphere (matches your original approach)
        mp_xyz, mp_w = _from_homogeneous(maph)
        mp_xyz = np.array(unit(mp_xyz.reshape(-1, 3))).reshape(mp_xyz.shape)
        maph = _to_homogeneous(mp_xyz, mp_w)

        return cls(maph, surf)

    # ---- surface geometry caches ----

    def surf_points(self) -> NDArray[np.float64]:
        if self._surf_pts is None:
            self._surf_pts = _from_homogeneous(self.surface)[0].reshape(-1, 3)
        return self._surf_pts

    def bbox(self) -> NDArray[np.float64]:
        if self._bbox is None:
            self._bbox = np.asarray(aabb(self.surf_points()), dtype=np.float64)
        return self._bbox

    def diag2(self) -> float:
        if self._diag2 is None:
            d = self.bbox()[1] - self.bbox()[0]
            self._diag2 = float(np.dot(d, d))
        return self._diag2

    def center(self) -> NDArray[np.float64]:
        if self._center is None:
            bb = self.bbox()
            self._center = 0.5 * (bb[0] + bb[1])
        return self._center

    def plane_normal(self) -> NDArray[np.float64] | None:
        """
        Best-fit plane normal from PCA (smallest eigenvector).
        Useful for slab separation in near-planar / low-dim cases.
        """
        if self._plane_n is not None:
            return self._plane_n

        P = self.surf_points()
        if P.shape[0] < 3:
            self._plane_n = None
            return None

        c = P.mean(axis=0)
        X = P - c
        cov = X.T @ X
        try:
            w, V = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            self._plane_n = None
            return None

        idx = np.argsort(w)
        n = V[:, idx[0]]
        nn = float(np.linalg.norm(n))
        if nn <= 1e-14:
            self._plane_n = None
            return None
        self._plane_n = n / nn

        # rank quality for GJK trust (smallest/ largest eigenvalue)
        wmax = float(w[idx[-1]])
        wmin = float(w[idx[0]])
        self._rank_quality = (wmin / wmax) if wmax > 1e-14 else 0.0

        return self._plane_n

    def rank_quality(self) -> float:
        if self._rank_quality is None:
            _ = self.plane_normal()
            if self._rank_quality is None:
                self._rank_quality = 0.0
        return float(self._rank_quality)

    # ---- Gauss map caches ----

    def map_dirs_net(self) -> NDArray[np.float64]:
        if self._map_dirs_net is None:
            xyz = _from_homogeneous(self._map)[0]          # (ru,rv,3)
            self._map_dirs_net = np.array(unit(xyz.reshape(-1,3))    ).reshape(xyz.shape)            # keep net shape
        return self._map_dirs_net

    def map_dirs(self) -> NDArray[np.float64]:
        if self._map_dirs_flat is None:
            self._map_dirs_flat = self.map_dirs_net().reshape(-1, 3)
        return self._map_dirs_flat

    def mean_normal(self) -> NDArray[np.float64] | None:
        if self._mean_n is not None:
            return self._mean_n
        N = self.map_dirs()
        s = N.sum(axis=0)
        ns = float(np.linalg.norm(s))
        if ns <= 1e-12:
            self._mean_n = None
        else:
            self._mean_n = s / ns
        return self._mean_n

    def gauss_radius(self) -> float:
        """
        Angular radius (radians) of the Gauss-map control directions about mean normal.
        If mean is undefined -> return pi (max).
        """
        if self._gauss_radius is not None:
            return float(self._gauss_radius)
        mn = self.mean_normal()
        if mn is None:
            self._gauss_radius = float(np.pi)
            return float(self._gauss_radius)
        N = self.map_dirs()
        dots = np.clip(N @ mn, -1.0, 1.0)
        md = float(dots.min())
        self._gauss_radius = float(np.arccos(md))
        return float(self._gauss_radius)

    def gauss_variation_uv(self) -> tuple[float, float]:
        if (self._var_u is not None) and (self._var_v is not None):
            return float(self._var_u), float(self._var_v)

        N = self.map_dirs_net()  # (ru,rv,3)
        # small-angle proxy: 1 - dot
        Du = 1.0 - np.sum(N[:-1, :, :] * N[1:, :, :], axis=-1)
        Dv = 1.0 - np.sum(N[:, :-1, :] * N[:, 1:, :], axis=-1)
        self._var_u = float(Du.max()) if Du.size else 0.0
        self._var_v = float(Dv.max()) if Dv.size else 0.0
        return float(self._var_u), float(self._var_v)

    # ---- 1D & 2D splits (split surface and gauss map together) ----

    def split_u(self, t: float = 0.5) -> list["GaussMapBern"]:
        t = float(t)
        sL, sR = split_tensor_bezier_axis(self.surface, t, axis=0)
        mL, mR = split_tensor_bezier_axis(self._map, t, axis=0)
        return [self.__class__(mL, sL), self.__class__(mR, sR)]

    def split_v(self, t: float = 0.5) -> list["GaussMapBern"]:
        t = float(t)
        sL, sR = split_tensor_bezier_axis(self.surface, t, axis=1)
        mL, mR = split_tensor_bezier_axis(self._map, t, axis=1)
        return [self.__class__(mL, sL), self.__class__(mR, sR)]

    def split_uv(self, u: float, v: float) -> list["GaussMapBern"]:
        u = float(u)
        v = float(v)
        # split in u -> 2
        sL, sR = split_tensor_bezier_axis(self.surface, u, axis=0)
        mL, mR = split_tensor_bezier_axis(self._map, u, axis=0)
        # split each in v -> 4
        out = []
        for s_net, m_net in ((sL, mL), (sR, mR)):
            sA, sB = split_tensor_bezier_axis(s_net, v, axis=1)
            mA, mB = split_tensor_bezier_axis(m_net, v, axis=1)
            out.append(self.__class__(mA, sA))
            out.append(self.__class__(mB, sB))
        return out

    # ---- evaluation + derivatives for Newton magic point ----

    def _derivative_ctrl_nets(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self._Hu_ctrl is None or self._Hv_ctrl is None:
            self._Hu_ctrl = _derivative_net_u(self.surface)
            self._Hv_ctrl = _derivative_net_v(self.surface)
        return self._Hu_ctrl, self._Hv_ctrl

    def eval_point_and_partials(self, u: float, v: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Evaluate Euclidean point F(u,v) and Euclidean partials Fu,Fv for rational Bezier surface.
        """
        u = float(u)
        v = float(v)
        H = _eval_tensor_bezier(self.surface, u, v)          # (4,)
        Hu_ctrl, Hv_ctrl = self._derivative_ctrl_nets()
        Hu = _eval_tensor_bezier(Hu_ctrl, u, v)              # (4,)
        Hv = _eval_tensor_bezier(Hv_ctrl, u, v)              # (4,)

        W = float(H[3])
        if abs(W) <= 1e-14:
            # extremely degenerate
            P = H[:3]
            return P, np.zeros(3), np.zeros(3)

        X = H[:3]
        Xu = Hu[:3]
        Xv = Hv[:3]
        Wu = float(Hu[3])
        Wv = float(Hv[3])

        invW = 1.0 / W
        F = X * invW

        invW2 = invW * invW
        Fu = (Xu * W - X * Wu) * invW2
        Fv = (Xv * W - X * Wv) * invW2
        return F, Fu, Fv


# ======================================================================================
# Hard-case detection + Newton "magic point" (eqs 6.4-6.7)
# ======================================================================================
import warnings
def _angle_between(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(d))

# @time_prof
def near_parallel_hard_case(
    g1: GaussMapBern,
    g2: GaussMapBern,
    *,
    parallel_angle: float = 0.053,   # ~3. degrees
    flat_angle: float = 0.15,       # ~8.6 degrees
) -> bool:
    n1 = g1.mean_normal()
    n2 = g2.mean_normal()
    if n1 is None or n2 is None:
        return False
    if _angle_between(n1, n2) > parallel_angle:
        return False
    if g1.gauss_radius() > flat_angle:
        return False
    if g2.gauss_radius() > flat_angle:
        return False
    return True


def _magic_residual(gA: GaussMapBern, gB: GaussMapBern, x: NDArray[np.float64]) -> NDArray[np.float64]:
    s, t, u, v = map(float, x)
    F, Fs, Ft = gA.eval_point_and_partials(s, t)
    G, Gu, Gv = gB.eval_point_and_partials(u, v)

    NA = np.cross(Fs, Ft)
    if float(np.dot(NA, NA)) <= 1e-24:
        # Degenerate normal => treat as non-solvable here
        return np.array([1e6, 1e6, 1e6, 1e6], dtype=np.float64)

    D = F - G
    H1 = float(np.dot(NA, Gu))
    H2 = float(np.dot(NA, Gv))
    H3 = float(np.dot(D, Fs))
    H4 = float(np.dot(D, Ft))
    return np.array([H1, H2, H3, H4], dtype=np.float64)

# @time_prof
def find_magic_point_newton(
    gA: GaussMapBern,
    gB: GaussMapBern,
    *,
    x0: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5),
    tol: float = 1e-6,
    max_iter: int = 5,
    fd_step: float = 1e-4,
) -> tuple[float, float, float, float] | None:
    """
    Damped Newton on the 4 auxiliary equations (6.4-6.7).
    Returns (s,t,u,v) in (0,1)^4 or None.
    """
    x = np.array([_clamp01(v) for v in x0], dtype=np.float64)
    r = _magic_residual(gA, gB, x)
    nr = float(np.linalg.norm(r))
    if not np.isfinite(nr):
        return None
    if nr < tol:
        return tuple(map(float, x))

    for _ in range(max_iter):
        # finite-difference Jacobian
        J = np.zeros((4, 4), dtype=np.float64)
        for j in range(4):
            h = fd_step
            # choose step direction to stay inside [0,1]
            if x[j] + h > 1.0:
                h = -h
            if x[j] + h < 0.0:
                h = fd_step
            xh = x.copy()
            xh[j] = _clamp01(xh[j] + h)
            rh = _magic_residual(gA, gB, xh)
            J[:, j] = (rh - r) / (xh[j] - x[j])

        # solve J dx = -r (robustly)
        try:
            dx = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(J, -r, rcond=None)

        if not np.all(np.isfinite(dx)):
            return None

        # damped step
        lam = 1.0
        accepted = False
        for _ls in range(10):
            xn = np.clip(x + lam * dx, 0.0, 1.0)
            xn = np.array([_clamp01(v) for v in xn], dtype=np.float64)
            rn = _magic_residual(gA, gB, xn)
            nrn = float(np.linalg.norm(rn))
            if np.isfinite(nrn) and nrn < nr:
                x, r, nr = xn, rn, nrn
                accepted = True
                break
            lam *= 0.5

        if not accepted:
            return None
        if nr < tol:
            return tuple(map(float, x))

    return None


# ======================================================================================
# Optional: guard GJK on low-dimensional sets
# ======================================================================================
# @time_prof
def _trust_gjk(g: GaussMapBern, *, min_rank_quality: float = 1e-10) -> bool:
    # rank_quality ~ smallest_eig / largest_eig. If too small -> very flat -> skip gjk.
    return g.rank_quality() >= min_rank_quality


# ======================================================================================
# Refinement: BB -> slab -> (optional GJK) -> Gauss separability -> 1D split / Newton split
# ======================================================================================

def _refine_pair_to_simple(
    g1: GaussMapBern,
    g2: GaussMapBern,
    *,
    spt: float,
    aabb_tol: float = 0.0,
    slab_tol_scale: float = 1e-14,
    gjk_tol: float = 1e-5,
    gjk_max_iter: int = 64,
    gm_eps: float = 1e-5,
    gm_tol: float = 1e-8,
    max_depth: int = 24,
    magic_start_depth: int = 6,
    parallel_angle: float = 0.05,
    flat_angle: float = 0.15,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Returns potentially intersecting *simple* subpatch pairs (homogeneous control nets).
    """
    out: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    spt2 = float(spt * spt)

    stack: list[tuple[GaussMapBern, GaussMapBern, int]] = [(g1, g2, 0)]

    while stack:
        a, b, depth = stack.pop()

        bb1 = a.bbox()
        bb2 = b.bbox()
        if not aabb_intersect_fast_3d(bb1, bb2):
            continue

        # termination by intersection box
        iib = np.array(aabb_intersection(bb1, bb2))
        d = iib[1] - iib[0]
        if float(np.dot(d, d)) <= spt2:
            out.append((a.surface, b.surface))
            continue

        P1 = a.surf_points()
        P2 = b.surf_points()

        # bounding-plane (slab) cull: conservative, works for flat sets
        axes: list[NDArray[np.float64]] = []
        pn1 = a.plane_normal()
        pn2 = b.plane_normal()
        if pn1 is not None:
            axes.append(pn1)
        if pn2 is not None:
            axes.append(pn2)

        mn1 = a.mean_normal()
        mn2 = b.mean_normal()
        if mn1 is not None:
            axes.append(mn1)
        if mn2 is not None:
            axes.append(mn2)

        cd = b.center() - a.center()
        if float(np.dot(cd, cd)) > 1e-30:
            axes.append(cd)

        # conservative tolerance: scaled by patch size
        slab_tol = slab_tol_scale * float(np.sqrt(max(a.diag2(), b.diag2(), 1e-30)))
        # print('slab_tol',slab_tol)
        r=bounding_plane_separated(P1, P2, axes, tol=slab_tol)
        # print('bounding_plane_separated',r,[P1.tolist(),P2.tolist()])
        if r:
            continue

        # optional GJK cull — only when both sets are well 3D (avoid low-dim failures)
        if _trust_gjk(a) and _trust_gjk(b):
            if not gjk(P1, P2, gjk_tol, gjk_max_iter):
                continue

        # Gauss separability (loop criterion / simplicity)
        p_sep1, p_sep2 = separate_gauss_maps(a.map_dirs(), b.map_dirs(), eps=gm_eps, tol=gm_tol)
        if (p_sep1 is not None) and (p_sep2 is not None):
            out.append((a.surface, b.surface))
            continue
        #is_sep, w3, info = great_circle_separable(a.map_dirs(), a.map_dirs())
        #is_sep2, w32, info2 = great_circle_separable(a.map_dirs(), -b.map_dirs())
        #
        #if is_sep and is_sep2:
        #    continue
        if depth >= max_depth:
            # give up refinement: still a potentially intersecting pair
            from inspect import currentframe ,getframeinfo
            frame=currentframe()
            frinfo=getframeinfo(frame)

            warnings.warn(f'({frinfo.function}) Maximum depth reached: {frinfo.filename}:{frinfo.lineno}')
            out.append((a.surface, b.surface))
            continue

        # Hard case: near-parallel, flat Gauss maps, still failing criterion -> try Newton magic point
        if depth >= magic_start_depth and near_parallel_hard_case(a, b, parallel_angle=parallel_angle, flat_angle=flat_angle):
            mp = find_magic_point_newton(a, b)
            if mp is not None:
                s, t, u, v = mp
                # Subdivide at magic point (2D split in each) -> 4x4 = 16 pairs (paper says "intersect all pairs")
                a_children = a.split_uv(_clamp01(s), _clamp01(t))
                b_children = b.split_uv(_clamp01(u), _clamp01(v))
                for ca in a_children:
                    for cb in b_children:
                        stack.append((ca, cb, depth + 1))
                continue
            # if Newton fails, fall through to Gauss-driven splitting

        # Split strategy: split ONLY ONE patch, and ONLY in ONE direction (2 children)
        # Choose patch with "larger Gauss map" (paper) rather than larger bbox.
        score_a = max(a.gauss_radius(), 10.0 * max(a.gauss_variation_uv()))
        score_b = max(b.gauss_radius(), 10.0 * max(b.gauss_variation_uv()))

        split_a = score_a >= score_b
        target = a if split_a else b
        other = b if split_a else a

        vu, vv = target.gauss_variation_uv()
        if vu >= vv:
            kids = target.split_u(0.5)
        else:
            kids = target.split_v(0.5)

        for k in kids:
            if split_a:
                stack.append((k, other, depth + 1))
            else:
                stack.append((other, k, depth + 1))

    return out


# ======================================================================================
# SSX entities + helpers (branch / point assembly)
# ======================================================================================

@dataclass
class SSXPoint:
    stuv: NDArray[np.float64] =None # (4,)
    xyz:NDArray[np.float64] = None

from functools import cached_property, lru_cache


@dataclass(unsafe_hash=True)
class SSXBranch:
    curve: NURBSCurveTuple
    closed: bool = False
    overlap: bool = False
    curve_xyz: NURBSCurveTuple=field(default=None,init=False)
    curve_st: NURBSCurveTuple=field(default=None,init=False)
    curve_uv: NURBSCurveTuple=field(default=None,init=False)


def _param_tol_from(tol: float, spt: float) -> float:
    # Prefer a slightly looser param tolerance than numeric solver tol.
    return max(1e-6, 10.0 * float(tol), 0.1 * float(spt))


def _map_uv_to_interval(uv: NDArray[np.float64], interval: tuple[float, float, float, float]) -> NDArray[np.float64]:
    u0, u1, v0, v1 = interval
    u = u0 + (u1 - u0) * float(uv[0])
    v = v0 + (v1 - v0) * float(uv[1])
    return np.array([u, v], dtype=np.float64)


def _map_uv_path_to_interval(
    uv_path: NDArray[np.float64],
    interval: tuple[float, float, float, float],
) -> NDArray[np.float64]:
    u0, u1, v0, v1 = interval
    uv = np.asarray(uv_path, dtype=np.float64)
    out = np.empty_like(uv)
    out[:, 0] = u0 + (u1 - u0) * uv[:, 0]
    out[:, 1] = v0 + (v1 - v0) * uv[:, 1]
    return out


def _split_interval(
    interval: tuple[float, float, float, float],
    axis: str,
    t: float,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], float]:
    u0, u1, v0, v1 = interval
    if axis == "u":
        um = u0 + (u1 - u0) * t
        return (u0, um, v0, v1), (um, u1, v0, v1), um
    if axis == "v":
        vm = v0 + (v1 - v0) * t
        return (u0, u1, v0, vm), (u0, u1, vm, v1), vm
    raise ValueError("axis must be 'u' or 'v'")


def _split_interval_uv(
    interval: tuple[float, float, float, float],
    u: float,
    v: float,
) -> tuple[list[tuple[float, float, float, float]], tuple[float, float]]:
    u0, u1, v0, v1 = interval
    um = u0 + (u1 - u0) * u
    vm = v0 + (v1 - v0) * v
    # Order must match GaussMapBern.split_uv: (uL,vL), (uL,vR), (uR,vL), (uR,vR)
    intervals = [
        (u0, um, v0, vm),
        (u0, um, vm, v1),
        (um, u1, v0, vm),
        (um, u1, vm, v1),
    ]
    return intervals, (um, vm)

from mmcore.numeric.bern import bernstein_boundaries_2d, bernstein_trim_nd


def _boundary_curves_from_net(
    H: NDArray[np.float64],
) -> list[tuple[str, float, NDArray[np.float64]]]:
    # Returns (axis, value, curve_net)
    bnd=bernstein_boundaries_2d(H)
    return [
        ("u", 0.0, bnd[0]),
        ("u", 1.0, bnd[1]),
        ("v", 0.0, bnd[2]),
        ("v", 1.0, bnd[3]),
    ]


def _boundary_uv(axis: str, value: float, t: float) -> NDArray[np.float64]:
    if axis == "u":
        return np.array([value, t], dtype=np.float64)
    if axis == "v":
        return np.array([t, value], dtype=np.float64)
    raise ValueError("axis must be 'u' or 'v'")


def _append_unique_point(points: list[SSXPoint], stuv: NDArray[np.float64], tol: float) -> None:
    for p in points:
        if np.max(np.abs(p.stuv - stuv)) <= tol:
            return
    points.append(SSXPoint(stuv=stuv))


def _pair_points_by_nearest(points: list[SSXPoint]) -> tuple[list[tuple[int, int]], list[int]]:
    n = len(points)
    if n < 2:
        return [], list(range(n))
    unused = set(range(n))
    pairs: list[tuple[int, int]] = []
    while len(unused) >= 2:
        i = min(unused)
        unused.remove(i)
        best_j = None
        best_d = float("inf")
        for j in unused:
            d = float(np.linalg.norm(points[i].stuv - points[j].stuv))
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is None:
            break
        unused.remove(best_j)
        pairs.append((i, best_j))
    return pairs, list(unused)


def _curve_endpoints(curve: NURBSCurveTuple) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return curve.start(), curve.end()


def _finite_diff_derivatives(points: NDArray[np.float64]) -> NDArray[np.float64]:
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    ders = np.zeros_like(pts)
    if n <= 1:
        return ders
    ders[0] = pts[1] - pts[0]
    ders[-1] = pts[-1] - pts[-2]
    if n > 2:
        ders[1:-1] = 0.5 * (pts[2:] - pts[:-2])
    return ders


def _curve_from_stuv_path(stuv_path: NDArray[np.float64],ders=None) -> NURBSCurveTuple | None:
    dd=stuv_path[-1]-stuv_path[0]
    if np.linalg.norm(dd) < 1e-6:
        return None

    pts = np.asarray(stuv_path, dtype=np.float64)
    if pts.shape[0] < 2:
        return None
    if ders is None:
        ders = _finite_diff_derivatives(pts)

    params = np.cumsum(np.linalg.norm(np.diff(pts, axis=0, prepend=pts[:1]), axis=1))

    return hermite_interpolate_nurbs(points=pts, derivatives=ders, params=params, degree=3)


def _dedupe_stuv_path(stuv_path: NDArray[np.float64], tol: float) -> NDArray[np.float64]:
    pts = np.asarray(stuv_path, dtype=np.float64).reshape(-1, 4)
    if pts.shape[0] == 0:
        return pts
    out = [pts[0]]
    for p in pts[1:]:
        if float(np.max(np.abs(p - out[-1]))) > float(tol):
            out.append(p)
    return np.asarray(out, dtype=np.float64)


def _try_deflated_hard_case(H1: NDArray[np.float64], H2: NDArray[np.float64],
                            interval1: tuple[float, float, float, float], interval2: tuple[float, float, float, float],
                            *, atol: float, angle_tol: float, param_tol: float,
                            recursive_fn=None, recursive_kwargs=None,
                            ) -> tuple[list[SSXBranch], list[SSXPoint]] | None:
    """
    Trace-first deflated fallback for tangential/degenerate SSX cases.
    Runs in local [0,1]^4 patch coordinates and maps back to global stuv.

    After finding the singular intersection, cuts out per-step boxes around the
    traced curve from one surface's parameter domain, then recurses on the
    remaining sub-domains to find any non-singular intersections.
    """
    from mmcore.numeric.intersection._interval_cutout import subtract_intervals_2d

    try:
        P1 = np.asarray(_from_homogeneous(H1)[0], dtype=np.float64)
        P2 = np.asarray(_from_homogeneous(H2)[0], dtype=np.float64)

        T1, T2, T3, T4 = minors_Tpsi_from_control_nets(P1, P2)
        is1 = get_iarray(P1, P1)
        is2 = get_iarray(P2, P2)
        T1i, T2i, T3i, T4i = (np.asarray(t, dtype=iv_interval) for t in (T1, T2, T3, T4))
        B = (iv_interval(0.0, 1.0), iv_interval(0.0, 1.0), iv_interval(0.0, 1.0), iv_interval(0.0, 1.0))

        res = analyse_deflated_system(
            is1,
            is2,
            T1i,
            T2i,
            T3i,
            T4i,
            B,
            bern_eval=bern_eval,
            interval_ctor=iv_interval,
            isolate_points=False,
            point_min_width=max(1e-8, 0.1 * param_tol),
            curve_slice_count=None,
            curve_mode="trace",
            build_cover=False,
            curve_krawczyk_fallback=False,
        )
    except Exception:
        return None

    branches: list[SSXBranch] = []
    points: list[SSXPoint] = []

    # --- Process traced singular curve ---
    trace_points = res.get("trace_points", [])
    trace_h_steps = res.get("trace_h_steps", [])
    singular_points=res.get("singular_points", [])

    if (trace_points and len(trace_points) >= 2 )or len(singular_points)>=1 :
        # Build stuv_path directly from trace points (no refinement needed —
        # trace_gamma's Newton corrector already achieves ||Δ|| < 1e-7).
        if (trace_points and len(trace_points) >= 2):
            stuv_path = []
            for pt in trace_points:
                s, t, u, v = (float(x) for x in pt)
                uv1 = _map_uv_to_interval(np.array([s, t], dtype=np.float64), interval1)
                uv2 = _map_uv_to_interval(np.array([u, v], dtype=np.float64), interval2)
                stuv_path.append(np.array([uv1[0], uv1[1], uv2[0], uv2[1]], dtype=np.float64))

            stuv_arr = _dedupe_stuv_path(np.asarray(stuv_path, dtype=np.float64), tol=param_tol)
            curve = _curve_from_stuv_path(stuv_arr)
            if curve is not None:
                branches.append(SSXBranch(curve=curve))
        if len(singular_points)>=1:
            for sp in singular_points:
                s, t, u, v = (float(x) for x in sp["param"])

                uv1 = _map_uv_to_interval(np.array([s, t], dtype=np.float64), interval1)
                uv2 = _map_uv_to_interval(np.array([u, v], dtype=np.float64), interval2)
                stuv = np.array([uv1[0], uv1[1], uv2[0], uv2[1]], dtype=np.float64)
                _append_unique_point(points, stuv, tol=param_tol)

        # --- Cutout and recurse on remaining sub-domains ---
        if recursive_fn is not None and len(trace_points) >= 2:
            eps = 1e-15

            # Compute footprint extent on each surface in local [0,1]^2
            if  (len(trace_points) >= 2 ):
                tp_arr = np.array([np.asarray(pt, dtype=np.float64) for pt in trace_points])
            else:
                tp_arr = np.array([np.asarray(pt['param'], dtype=np.float64) for pt in singular_points])

            s1_extent_u = float(tp_arr[:, 0].max() - tp_arr[:, 0].min())
            s1_extent_v = float(tp_arr[:, 1].max() - tp_arr[:, 1].min())
            s2_extent_u = float(tp_arr[:, 2].max() - tp_arr[:, 2].min())
            s2_extent_v = float(tp_arr[:, 3].max() - tp_arr[:, 3].min())
            s1_max_extent = max(s1_extent_u, s1_extent_v)
            s2_max_extent = max(s2_extent_u, s2_extent_v)

            # Pick the surface where the trace's longer axis is shorter
            if s1_max_extent <= s2_max_extent:
                    cut_surface_idx = 0  # cut surface 1
                    H_cut, interval_cut = H1, interval1
                    H_other, interval_other = H2, interval2
                    ax_offset = 0  # columns 0,1 in trace_points are (s,t) for surface 1
            else:
                    cut_surface_idx = 1  # cut surface 2
                    H_cut, interval_cut = H2, interval2
                    H_other, interval_other = H1, interval1
                    ax_offset = 2  # columns 2,3 in trace_points are (u,v) for surface 2

            # Compute local param_tol for the cut surface
            u_span = max(interval_cut[1] - interval_cut[0], eps)
            v_span = max(interval_cut[3] - interval_cut[2], eps)

            local_pad_u = param_tol / u_span
            local_pad_v = param_tol / v_span

            # Build per-step boxes in local [0,1]^2 of the cut surface

            boxes = []
            if len(trace_points)>=2:
                for i in range(len(trace_points) - 1):
                    p0 = np.asarray(trace_points[i], dtype=np.float64)
                    p1 = np.asarray(trace_points[i + 1], dtype=np.float64)
                    h_i = float(trace_h_steps[i + 1]) if i + 1 < len(trace_h_steps) else 0.0
                    pad_u = max(local_pad_u, h_i)
                    pad_v = max(local_pad_v, h_i)
                    u_lo = min(p0[ax_offset], p1[ax_offset]) - pad_u
                    u_hi = max(p0[ax_offset], p1[ax_offset]) + pad_u
                    v_lo = min(p0[ax_offset + 1], p1[ax_offset + 1]) - pad_v
                    v_hi = max(p0[ax_offset + 1], p1[ax_offset + 1]) + pad_v
                    boxes.append(((u_lo, v_lo), (u_hi, v_hi)))
            if len(singular_points)>=1:
                for i in range(len(singular_points) ):
                    p=singular_points[i]['param']
                    p0=p[:2],p[2:]
                    pad_uv = np.array([local_pad_u, local_pad_v], dtype=np.float64)


                    u_lo ,v_lo=  p0[ax_offset] - pad_uv
                    u_hi, v_hi = p0[ax_offset] + pad_uv
                    print(u_lo,v_lo,u_hi,v_hi)

                    boxes.append(((u_lo, v_lo), (u_hi, v_hi)))


            # Subtract the traced boxes from [0,1]^2 to get remaining sub-domains
            remaining = subtract_intervals_2d(boxes, bounds=((0., 0.), (1., 1.)))

            # Recurse on each remaining sub-domain
            rk = recursive_kwargs or {}
            for (lo, hi) in remaining:
                u_range = (lo[0], hi[0])
                v_range = (lo[1], hi[1])
                if (u_range[1] - u_range[0]) < eps or (v_range[1] - v_range[0]) < eps:
                    continue

                sub_H = bernstein_trim_nd(H_cut, [u_range, v_range])
                sub_interval = (
                    interval_cut[0] + lo[0] * (interval_cut[1] - interval_cut[0]),
                    interval_cut[0] + hi[0] * (interval_cut[1] - interval_cut[0]),
                    interval_cut[2] + lo[1] * (interval_cut[3] - interval_cut[2]),
                    interval_cut[2] + hi[1] * (interval_cut[3] - interval_cut[2]),
                )

                g_sub = GaussMapBern.from_surf(sub_H, rational=True)
                g_other = GaussMapBern.from_surf(H_other, rational=True)

                if cut_surface_idx == 0:
                    sub_b, sub_p = recursive_fn(g_sub, g_other, sub_interval, interval_other,
                                                deflate_hard_case=False, **rk)
                else:
                    sub_b, sub_p = recursive_fn(g_other, g_sub, interval_other, sub_interval,
                                                deflate_hard_case=False, **rk)
                branches.extend(sub_b)
                points.extend(sub_p)




    # --- Process singular points (only when a curve was traced) ---


    return branches, points
def _get_pt_t(s1,s2,stuv):

    S_eval= evaluate_nurbs_surface(s1, stuv[0], stuv[1], d_order=1)
    T_eval = evaluate_nurbs_surface(s2, stuv[2], stuv[3], d_order=1)

    Su = S_eval['Su']
    Sv =  S_eval['Sv']
    Tu =  T_eval['Su']
    Tv = T_eval['Sv']
    Sn = np.cross(Su, Sv)
    Tn = np.cross(Tu, Tv)
    Sn = Sn / np.linalg.norm(Sn)
    Tn = Tn / np.linalg.norm(Tn)
    T = np.cross(Sn, Tn)
    T/= np.linalg.norm(T)
    return  (S_eval['S'],Su,Sv),(T_eval['S'],Tu,Tv),T
def duv_from_eval(Su, Sv, T):

    # Form the Jacobian matrices for each surface (3x2).
    J = np.column_stack((Su,Sv))

    # Compute least-squares corrections using the pseudoinverse.

    delta_uv = np.linalg.pinv(J) @ T
    return delta_uv
from mmcore.geom._nurbs_interp import hermite_interpolate_nurbs
from mmcore.geom._nurbs_ders import _greville_abscissae


def _post_process_bez_csx_results(isolated, overlaps, curve_net, atol):
    """Post-process raw bez_csx results for a single boundary curve.

    Each overlap from bez_csx is already a fully traced path — we treat it as a
    ready-made sub-branch.  No range-merging is performed; we only:
      (a) estimate a parameter tolerance,
      (b) demote individually short overlaps to isolated points,
      (c) filter isolated points that fall inside any overlap range,
      (d) absorb isolated points near overlap endpoints,
      (e) deduplicate remaining isolated points.
    """
    if not isolated and not overlaps:
        return isolated, overlaps

    eps = 1e-15

    # (a) Estimate parameter tolerance from control polygon arc length.
    pts = np.asarray(curve_net, dtype=np.float64)
    if pts.ndim == 2:
        if pts.shape[1] == 4:
            w = pts[:, 3:4]
            w = np.where(np.abs(w) < eps, eps, w)
            xyz = pts[:, :3] / w
        else:
            xyz = pts[:, :3]
    else:
        xyz = pts.reshape(-1, 3)
    diffs = np.diff(xyz, axis=0)
    poly_length = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
    t_tol = atol / max(poly_length, eps)

    # (b) Compute per-overlap ranges; demote short ones to isolated points.
    overlap_ranges = []  # [(t_min, t_max)] for surviving overlaps
    new_isolated_from_short = []
    short_indices = set()
    for idx, ovl in enumerate(overlaps):
        t_path = np.asarray(ovl["t_path"], dtype=np.float64).reshape(-1)
        t_lo = float(min(t_path[0], t_path[-1]))
        t_hi = float(max(t_path[0], t_path[-1]))
        if (t_hi - t_lo) < t_tol:
            # Too short to be a real overlap — convert to an isolated point.
            short_indices.add(idx)
            uv_path = np.asarray(ovl["uv_path"], dtype=np.float64).reshape(-1, 2)
            mid_idx = len(t_path) // 2
            pt_dict = {"t": float(t_path[mid_idx]), "u": float(uv_path[mid_idx, 0]), "v": float(uv_path[mid_idx, 1])}
            if "xyz_path" in ovl:
                xyz_path = np.asarray(ovl["xyz_path"], dtype=np.float64).reshape(-1, 3)
                pt_dict["point"] = xyz_path[mid_idx]
            new_isolated_from_short.append(pt_dict)
        else:
            overlap_ranges.append((t_lo, t_hi))

    if short_indices:
        overlaps = [o for i, o in enumerate(overlaps) if i not in short_indices]

    isolated = list(isolated) + new_isolated_from_short

    # (c) Filter isolated points inside any surviving overlap range.
    if overlap_ranges and isolated:
        def _in_any_overlap(t_val):
            for t_lo, t_hi in overlap_ranges:
                if (t_lo - t_tol) <= t_val <= (t_hi + t_tol):
                    return True
            return False

        isolated = [iso for iso in isolated if not _in_any_overlap(float(iso["t"]))]

    # (d) Absorb isolated points near overlap endpoints (spatial check).
    if isolated and overlaps:
        absorbed = set()
        for i, pt in enumerate(isolated):
            t_pt = float(pt["t"])
            p_pt = pt.get("point", None)
            for ovl in overlaps:
                t_path = np.asarray(ovl["t_path"], dtype=np.float64).reshape(-1)
                if p_pt is not None and "xyz_path" in ovl:
                    xyz_path = np.asarray(ovl["xyz_path"], dtype=np.float64).reshape(-1, 3)
                    d0 = float(np.linalg.norm(p_pt - xyz_path[0]))
                    d1 = float(np.linalg.norm(p_pt - xyz_path[-1]))
                    if d0 <= atol or d1 <= atol:
                        absorbed.add(i)
                        break
                else:
                    if abs(t_pt - float(t_path[0])) <= t_tol or abs(t_pt - float(t_path[-1])) <= t_tol:
                        absorbed.add(i)
                        break
        if absorbed:
            isolated = [iso for i, iso in enumerate(isolated) if i not in absorbed]

    # (e) Deduplicate remaining isolated points by t proximity.
    if isolated:
        isolated_sorted = sorted(isolated, key=lambda p: float(p["t"]))
        deduped = [isolated_sorted[0]]
        merge_tol = 2.0 * t_tol
        for p in isolated_sorted[1:]:
            if abs(float(p["t"]) - float(deduped[-1]["t"])) <= merge_tol:
                continue
            deduped.append(p)
        isolated = deduped

    return isolated, overlaps


def _collect_boundary_intersections(H_owner: NDArray[np.float64], H_other: NDArray[np.float64],
                                    interval_owner: tuple[float, float, float, float],
                                    interval_other: tuple[float, float, float, float], *, owner_is_first: bool,
                                    atol: float, angle_tol: float, param_tol: float, points: list[SSXPoint],
                                    branches: list[SSXBranch]) -> None:
    count=0

    for axis, value, curve_net in _boundary_curves_from_net(H_owner):
        # print(curve_net.tolist(),H_other.tolist())

        res = bez_csx(
            curve_net,
            H_other,
            atol=atol,
            rational=True,
            angle_tol=angle_tol,



        )
        filtered_isolated, filtered_overlaps = _post_process_bez_csx_results(
            res.get("isolated", []), res.get("overlaps", []), curve_net, atol
        )
        for iso in filtered_isolated:
            t = float(iso["t"])
            uv_owner_local = _boundary_uv(axis, value, t)
            uv_other_local = np.array([float(iso["u"]), float(iso["v"])], dtype=np.float64)
            if owner_is_first:
                uv1 = _map_uv_to_interval(uv_owner_local, interval_owner)
                uv2 = _map_uv_to_interval(uv_other_local, interval_other)
            else:
                uv1 = _map_uv_to_interval(uv_other_local, interval_other)
                uv2 = _map_uv_to_interval(uv_owner_local, interval_owner)
            stuv = np.array([uv1[0], uv1[1], uv2[0], uv2[1]], dtype=np.float64)
            _append_unique_point(points, stuv, tol=param_tol)
            count+=1
        for ovl in filtered_overlaps:
            t_path = np.asarray(ovl["t_path"], dtype=np.float64).reshape(-1)

            uv_path = np.asarray(ovl["uv_path"], dtype=np.float64).reshape(-1, 2)
            uv_owner_local = np.stack([_boundary_uv(axis, value, t) for t in t_path], axis=0)
            if owner_is_first:
                uv1_path = _map_uv_path_to_interval(uv_owner_local, interval_owner)
                uv2_path = _map_uv_path_to_interval(uv_path, interval_other)
            else:
                uv1_path = _map_uv_path_to_interval(uv_path, interval_other)
                uv2_path = _map_uv_path_to_interval(uv_owner_local, interval_owner)
            stuv_path = np.hstack([uv1_path, uv2_path])
            delta = stuv_path[-1] - stuv_path[0]
            if np.linalg.norm(delta) < param_tol:
                continue
            delta/=np.linalg.norm(delta)

            cnt,kv=interpolate_curve(stuv_path,min(len(stuv_path)-1,3),remove_duplicates=True,use_centripetal=True,tol=param_tol)

            curve = NURBSCurveTuple(order=min(len(stuv_path)-1,3)+1, knot=np.array(kv),control_points=np.array(cnt),weights=np.ones(len(cnt)) )


            branches.append(SSXBranch(curve=curve, overlap=True))


def _dedup_overlap_branches(branches, param_tol):
    """Remove duplicate overlap branches that represent the same geometric intersection.

    When both _collect_boundary_intersections calls detect the same overlap
    (coinciding patch boundaries), this removes the duplicate.
    """
    if len(branches) < 2:
        return branches

    overlap_indices = [i for i, b in enumerate(branches) if b.overlap]
    if len(overlap_indices) < 2:
        return branches

    to_remove = set()
    for ii in range(len(overlap_indices)):
        if overlap_indices[ii] in to_remove:
            continue
        br_a = branches[overlap_indices[ii]]
        start_a = br_a.curve.control_points[0]
        end_a = br_a.curve.control_points[-1]
        for jj in range(ii + 1, len(overlap_indices)):
            if overlap_indices[jj] in to_remove:
                continue
            br_b = branches[overlap_indices[jj]]
            start_b = br_b.curve.control_points[0]
            end_b = br_b.curve.control_points[-1]
            # Same direction match
            same_dir = (np.max(np.abs(start_a - start_b)) <= param_tol and
                        np.max(np.abs(end_a - end_b)) <= param_tol)
            # Reversed direction match
            rev_dir = (np.max(np.abs(start_a - end_b)) <= param_tol and
                       np.max(np.abs(end_a - start_b)) <= param_tol)
            if same_dir or rev_dir:
                to_remove.add(overlap_indices[jj])

    if to_remove:
        branches = [b for i, b in enumerate(branches) if i not in to_remove]
    return branches


def _leaf_boundary_test_and_march(H1: NDArray[np.float64], H2: NDArray[np.float64],
                                  interval1: tuple[float, float, float, float],
                                  interval2: tuple[float, float, float, float], *, atol: float, param_tol: float,
                                  angle_tol: float = 0.01) -> tuple[list[SSXBranch], list[SSXPoint]]:
    points: list[SSXPoint] = []
    branches: list[SSXBranch] = []

    _collect_boundary_intersections(H1, H2, interval1, interval2, owner_is_first=True, atol=atol, angle_tol=angle_tol,
                                    param_tol=param_tol, points=points, branches=branches)
    _collect_boundary_intersections(H2, H1, interval2, interval1, owner_is_first=False, atol=atol, angle_tol=angle_tol,
                                    param_tol=param_tol, points=points, branches=branches)

    branches = _dedup_overlap_branches(branches, param_tol)

    # Remove isolated points that coincide with overlap branch endpoints.
    # These arise because different boundary curves detect the same geometric
    # point — once as an isolated hit, once as an overlap endpoint — and the
    # per-boundary post-processing cannot catch the cross-boundary duplicate.
    # Keeping them would cause trace_between to draw a spurious curve between
    # two overlap endpoints through the interior.
    if points and branches:
        overlap_endpoints = []
        for br in branches:
            if br.overlap:
                overlap_endpoints.append(br.curve.control_points[0])
                overlap_endpoints.append(br.curve.control_points[-1])
        if overlap_endpoints:
            points = [p for p in points
                      if not any(np.max(np.abs(p.stuv - ep)) <= param_tol
                                 for ep in overlap_endpoints)]

    if not points and not branches:
        return [], []
    #if len(branches)>=2:
    #    branches=[]
    # If we only have isolated points, try to march between paired points.
    if len(points) >= 2:
        pairs, leftover = _pair_points_by_nearest(points)
        if pairs:
            #s1 = bern_to_nurbs_bezier(H1, interval=((interval1[0], interval1[1]), (interval1[2], interval1[3])), rational=True)
            #s2 = bern_to_nurbs_bezier(H2, interval=((interval2[0], interval2[1]), (interval2[2], interval2[3])), rational=True)
            for i, j in pairs:
                curve = trace_between(
                    H1,
                    H2,
                    points[i].stuv,
                                        points[j].stuv,
                                        interval1=interval1,
                                        interval2=interval2,
                                        spt=atol,
                                        fit_max_depth=10,angle_tol=angle_tol,

                )
                branches.append(SSXBranch(curve=curve))
        points = [points[k] for k in leftover]

    return branches, points


def _endpoint_match(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    tol: float,
    ignore_idx: int | None = None,
) -> bool:
    d = np.abs(a - b)
    if ignore_idx is not None:
        d[ignore_idx] = 0.0
    return float(np.max(d)) <= tol


def _merge_branches_by_match(
    branches: list[SSXBranch],
    match_fn,
        atol
) -> list[SSXBranch]:
    i = 0
    while i < len(branches):
        if branches[i].closed:
            i += 1
            continue
        merged = False
        for j in range(i + 1, len(branches)):
            if branches[j].closed:
                continue
            a_start, a_end = _curve_endpoints(branches[i].curve)
            b_start, b_end = _curve_endpoints(branches[j].curve)
            a_ends = (a_start, a_end)
            b_ends = (b_start, b_end)
            for end_i in (0, 1):
                for end_j in (0, 1):
                    if not match_fn(a_ends[end_i], b_ends[end_j]):
                        continue
                    # Orient so match point is at end of A and start of B.
                    curve_a = branches[i].curve
                    curve_b = branches[j].curve
                    if end_i == 0:
                        curve_a = reverse_curve(curve_a)
                    if end_j == 1:
                        curve_b = reverse_curve(curve_b)
                    new_curve, _interior_knots = link_curves([curve_a, curve_b])
                    new_curve=remove_knots_after_merge(new_curve, _interior_knots,tol=atol)
                    branches[i] = SSXBranch(curve=new_curve, overlap=branches[i].overlap or branches[j].overlap)
                    branches.pop(j)
                    merged = True
                    break
                if merged:
                    break
            if merged:
                break
        if merged:
            i = 0
        else:
            i += 1
    return branches


def _merge_branches_on_split(
    branches: list[SSXBranch],
    *,
    surf_index: int,
    axis: str,
    split_value: float,
    tol: float,
        atol:float
) -> list[SSXBranch]:
    if surf_index == 1:
        idx = 0 if axis == "u" else 1
    else:
        idx = 2 if axis == "u" else 3

    def match_fn(a, b):
        #print(a,b)
        if abs(a[idx] - split_value) > tol:
            return False
        if abs(b[idx] - split_value) > tol:
            return False
        return _endpoint_match(a, b, tol=tol, ignore_idx=idx)

    return _merge_branches_by_match(branches, match_fn, atol)


def _merge_branches_global(branches: list[SSXBranch], tol: float,atol:float) -> list[SSXBranch]:
    return _merge_branches_by_match(branches, lambda a, b: _endpoint_match(a, b, tol=tol),atol=atol)


def _close_branches(branches: list[SSXBranch], tol: float) -> list[SSXBranch]:
    for br in branches:
        if br.closed:
            continue
        start, end = _curve_endpoints(br.curve)
        if _endpoint_match(start, end, tol=tol):
            br.closed = True
    return branches


def _prune_points_on_branches(points: list[SSXPoint], branches: list[SSXBranch], tol: float) -> list[SSXPoint]:
    if not points or not branches:
        return points
    out = []
    for p in points:
        keep = True
        for br in branches:
            start, end = _curve_endpoints(br.curve)
            if _endpoint_match(p.stuv, start, tol=tol) or _endpoint_match(p.stuv, end, tol=tol):
                keep = False
                break
        if keep:
            out.append(p)
    return out


def _bez_ssx_recursive(g1: GaussMapBern, g2: GaussMapBern, interval1: tuple[float, float, float, float],
                       interval2: tuple[float, float, float, float], *, atol: float, tol: float, param_tol: float,
                       aabb_tol: float = 0.0, slab_tol_scale: float = 1e-14, gjk_tol: float = 1e-5,
                       gjk_max_iter: int = 64, gm_eps: float = 1e-5, gm_tol: float = 1e-8, max_depth: int = 24,
                       magic_start_depth: int = 2, parallel_angle: float = 0.053, flat_angle: float = 0.01,
                       march_samples: int = 8, deflate_hard_case: bool = True, depth: int = 0) -> tuple[list[SSXBranch], list[SSXPoint]]:
    bb1 = g1.bbox()
    bb2 = g2.bbox()

    if not aabb_intersect_fast_3d(bb1, bb2):
        return [], []

    # termination by intersection box
    iib = np.array(aabb_intersection(bb1, bb2))
    d = iib[1] - iib[0]
    if float(np.dot(d, d)) <= float(atol * atol):
        return _leaf_boundary_test_and_march(g1.surface, g2.surface, interval1, interval2, atol=atol,
                                             param_tol=param_tol, angle_tol=parallel_angle)


    P1 = g1.surf_points()
    P2 = g2.surf_points()

    # bounding-plane (slab) cull: conservative, works for flat sets
    axes: list[NDArray[np.float64]] = []
    pn1 = g1.plane_normal()
    pn2 = g2.plane_normal()
    if pn1 is not None:
        axes.append(pn1)
    if pn2 is not None:
        axes.append(pn2)

    mn1 = g1.mean_normal()
    mn2 = g2.mean_normal()
    if mn1 is not None:
        axes.append(mn1)
    if mn2 is not None:
        axes.append(mn2)

    cd = g2.center() - g1.center()
    if float(np.dot(cd, cd)) > 1e-30:
        axes.append(cd)

    slab_tol = slab_tol_scale * float(np.sqrt(max(g1.diag2(), g2.diag2(), 1e-30)))
    if bounding_plane_separated(P1, P2, axes, tol=slab_tol):
        return [], []

    if _trust_gjk(g1) and _trust_gjk(g2):
        if not gjk(P1, P2, gjk_tol, gjk_max_iter):
            return [], []

    p_sep1, p_sep2 = separate_gauss_maps(g1.map_dirs(), g2.map_dirs(), eps=gm_eps, tol=gm_tol)
    if (p_sep1 is not None) and (p_sep2 is not None):
        return _leaf_boundary_test_and_march(g1.surface, g2.surface, interval1, interval2, atol=atol,
                                             param_tol=param_tol, angle_tol=parallel_angle)
    _deflate_recursive_kwargs = dict(
        atol=atol, tol=tol, param_tol=param_tol, aabb_tol=aabb_tol, slab_tol_scale=slab_tol_scale,
        gjk_tol=gjk_tol, gjk_max_iter=gjk_max_iter, gm_eps=gm_eps, gm_tol=gm_tol,
        max_depth=max_depth, magic_start_depth=magic_start_depth, parallel_angle=parallel_angle,
        flat_angle=flat_angle, march_samples=march_samples, depth=depth + 1,
    )

    if deflate_hard_case:
        # We call deflation before reaching hard_case detection,
        # as this can significantly reduce the number of subproblems and avoid splitting the patch at the problem location.
        hard = _try_deflated_hard_case(g1.surface, g2.surface, interval1, interval2, atol=atol,
                                       angle_tol=parallel_angle, param_tol=param_tol,
                                       recursive_fn=_bez_ssx_recursive,
                                       recursive_kwargs=_deflate_recursive_kwargs)

        #print('try deflated (return): ', f'{frame_info.filename}:{frame_info.lineno + 2}:0')
        #print(_branches)
        _b,_p=hard
        if len(_b)>0 or len(_p)>0:
            if len(hard[1])>0 and len(hard[0])==0:
                _branches, _points=_leaf_boundary_test_and_march(
                    g1.surface,
                    g2.surface,
                    interval1,
                    interval2,
                    atol=atol,
                    param_tol=param_tol,
                    angle_tol=parallel_angle,

                )
                #print('try deflated (return): ', f'{frame_info.filename}:{frame_info.lineno + 2}:0')
                #print(_branches)
                print('fff')
                return hard[0]+_branches,hard[1]+_points

            print(hard)
            return hard
        deflate_hard_case=False # ATTENTION!!! If the first deflation ended in failure, there is no point in looking for deflation in subsidiary subproblems.
        #print('try deflated (no): ', f'{frame_info.filename}:{frame_info.lineno + 2}:0')
    is_hard_case = near_parallel_hard_case(g1, g2, parallel_angle=parallel_angle, flat_angle=flat_angle)

    if deflate_hard_case and is_hard_case:
            hard = _try_deflated_hard_case(g1.surface, g2.surface, interval1, interval2, atol=atol,
                                           angle_tol=parallel_angle, param_tol=param_tol,
                                           recursive_fn=_bez_ssx_recursive,
                                           recursive_kwargs=_deflate_recursive_kwargs)
            if hard is not None:
                return hard
    if depth>=max_depth:
        return _leaf_boundary_test_and_march(g1.surface, g2.surface, interval1, interval2, atol=atol,
                                             param_tol=param_tol, angle_tol=parallel_angle)

    # Hard case: near-parallel, flat Gauss maps, still failing criterion -> try Newton magic point
    if depth >= magic_start_depth and is_hard_case:

        mp = find_magic_point_newton(g1, g2)

        if mp is not None:
            s, t, u, v = mp
            a_children = g1.split_uv(_clamp01(s), _clamp01(t))
            b_children = g2.split_uv(_clamp01(u), _clamp01(v))

            intervals_a, (ua_split, va_split) = _split_interval_uv(interval1, _clamp01(s), _clamp01(t))
            intervals_b, (ub_split, vb_split) = _split_interval_uv(interval2, _clamp01(u), _clamp01(v))

            branches: list[SSXBranch] = []
            points: list[SSXPoint] = []
            for ia, ca in enumerate(a_children):
                for ib, cb in enumerate(b_children):
                    bch, pts = _bez_ssx_recursive(ca, cb, intervals_a[ia], intervals_b[ib], atol=atol, tol=tol,
                                                  param_tol=param_tol, aabb_tol=aabb_tol, slab_tol_scale=slab_tol_scale,
                                                  gjk_tol=gjk_tol, gjk_max_iter=gjk_max_iter, gm_eps=gm_eps,
                                                  gm_tol=gm_tol, max_depth=max_depth,
                                                  magic_start_depth=magic_start_depth, parallel_angle=parallel_angle,
                                                  flat_angle=flat_angle, march_samples=march_samples,
                                                  deflate_hard_case=deflate_hard_case, depth=depth + 1)
                    branches.extend(bch)
                    points.extend(pts)

            # Merge across both split lines on both surfaces.
            branches = _merge_branches_on_split(branches, surf_index=1, axis="u", split_value=ua_split, tol=param_tol, atol=atol)
            branches = _merge_branches_on_split(branches, surf_index=1, axis="v", split_value=va_split, tol=param_tol, atol=atol)
            branches = _merge_branches_on_split(branches, surf_index=2, axis="u", split_value=ub_split, tol=param_tol, atol=atol)
            branches = _merge_branches_on_split(branches, surf_index=2, axis="v", split_value=vb_split, tol=param_tol, atol=atol)
            branches = _close_branches(branches, param_tol)
            points = _prune_points_on_branches(points, branches, param_tol)
            return branches, points

    # Split strategy: split ONLY ONE patch, and ONLY in ONE direction (2 children)
    score_a = max(g1.gauss_radius(),  max(g1.gauss_variation_uv()))
    score_b = max(g2.gauss_radius(),  max(g2.gauss_variation_uv()))

    split_a = score_a >= score_b
    target = g1 if split_a else g2
    other = g2 if split_a else g1

    vu, vv = target.gauss_variation_uv()
    axis = "u" if vu >= vv else "v"
    kids = target.split_u(0.5) if axis == "u" else target.split_v(0.5)

    if split_a:
        left_int, right_int, split_val = _split_interval(interval1, axis, 0.5)
        b0, p0 = _bez_ssx_recursive(kids[0], other, left_int, interval2, atol=atol, tol=tol, param_tol=param_tol,
                                    aabb_tol=aabb_tol, slab_tol_scale=slab_tol_scale, gjk_tol=gjk_tol,
                                    gjk_max_iter=gjk_max_iter, gm_eps=gm_eps, gm_tol=gm_tol, max_depth=max_depth,
                                    magic_start_depth=magic_start_depth, parallel_angle=parallel_angle,
                                    flat_angle=flat_angle, march_samples=march_samples,
                                    deflate_hard_case=deflate_hard_case, depth=depth + 1)
        b1, p1 = _bez_ssx_recursive(kids[1], other, right_int, interval2, atol=atol, tol=tol, param_tol=param_tol,
                                    aabb_tol=aabb_tol, slab_tol_scale=slab_tol_scale, gjk_tol=gjk_tol,
                                    gjk_max_iter=gjk_max_iter, gm_eps=gm_eps, gm_tol=gm_tol, max_depth=max_depth,
                                    magic_start_depth=magic_start_depth, parallel_angle=parallel_angle,
                                    flat_angle=flat_angle, march_samples=march_samples,
                                    deflate_hard_case=deflate_hard_case, depth=depth + 1)
        branches = _merge_branches_on_split([*b0, *b1], surf_index=1, axis=axis, split_value=split_val, tol=param_tol, atol=atol)
        points = _prune_points_on_branches([*p0, *p1], branches, param_tol)
        branches = _close_branches(branches, param_tol)
        return branches, points
    else:
        left_int, right_int, split_val = _split_interval(interval2, axis, 0.5)
        b0, p0 = _bez_ssx_recursive(other, kids[0], interval1, left_int, atol=atol, tol=tol, param_tol=param_tol,
                                    aabb_tol=aabb_tol, slab_tol_scale=slab_tol_scale, gjk_tol=gjk_tol,
                                    gjk_max_iter=gjk_max_iter, gm_eps=gm_eps, gm_tol=gm_tol, max_depth=max_depth,
                                    magic_start_depth=magic_start_depth, parallel_angle=parallel_angle,
                                    flat_angle=flat_angle, march_samples=march_samples,
                                    deflate_hard_case=deflate_hard_case, depth=depth + 1)
        b1, p1 = _bez_ssx_recursive(other, kids[1], interval1, right_int, atol=atol, tol=tol, param_tol=param_tol,
                                    aabb_tol=aabb_tol, slab_tol_scale=slab_tol_scale, gjk_tol=gjk_tol,
                                    gjk_max_iter=gjk_max_iter, gm_eps=gm_eps, gm_tol=gm_tol, max_depth=max_depth,
                                    magic_start_depth=magic_start_depth, parallel_angle=parallel_angle,
                                    flat_angle=flat_angle, march_samples=march_samples,
                                    deflate_hard_case=deflate_hard_case, depth=depth + 1)
        branches = _merge_branches_on_split([*b0, *b1], surf_index=2, axis=axis, split_value=split_val, tol=param_tol, atol=atol)
        points = _prune_points_on_branches([*p0, *p1], branches, param_tol)
        branches = _close_branches(branches, param_tol)
        return branches, points


def compute_branch_curves_hermite(branch: SSXBranch, surface1:NURBSSurfaceTuple, surface2:NURBSSurfaceTuple,atol,**kwargs):
    branch.curve_st = branch.curve._replace(control_points=branch.curve.control_points[..., :2])
    branch.curve_uv = branch.curve._replace(control_points=branch.curve.control_points[..., 2:])
    def _eval_curve(t):
        stuv = evaluate_nurbs_curve(branch.curve, t, d_order=0)["C"]
        se1=evaluate_nurbs_surface(surface1,stuv[0],stuv[1],d_order=0)

        return se1['S']

    eval_curve_ders=fdm(_eval_curve)
    _points = []
    _ders=[]
    _params=[]
    params=_greville_abscissae(branch.curve.knot, branch.curve.degree)
    # params=np.unique(branch.curve.knot)
    #p_cur=0
    params=branch.curve.control_points
    for s in params:
        #d1=eval_curve_ders(s)
        #d1/=np.linalg.norm(d1)

        #_ders.append(d1)


        _points.append(evaluate_nurbs_surface(surface1, s[0], s[1], d_order=0)['S'])


    #from more_itertools import pairwise
    #crvs=[]
    #
    #for (ps,ds,ts),(pe,de,te) in pairwise(zip(_points,_ders,params)):
    #    h=np.linalg.norm(ps-pe)/3
    #
    #
    #    crvs.append(bern_to_nurbs_bezier(np.array([ps,ps+ds*h,pe-ds*h,pe]),interval=(ts,te),rational=False))
    branch.curve_xyz=interpolate_nurbs_curve(np.array(_points),degree=branch.curve.degree,use_centripetal=True,method='lu',remove_duplicates=True,tol=atol)
    #branch.curve_xyz=remove_knots_after_merge(crv,interior,atol)

def compute_branch_curves(branch: SSXBranch, surface1:NURBSSurfaceTuple, surface2:NURBSSurfaceTuple,**kwargs):
    return compute_branch_curves_hermite(branch, surface1, surface2, **kwargs)
    branch.curve_st = branch.curve._replace(control_points=branch.curve.control_points[..., :2])
    branch.curve_uv = branch.curve._replace(control_points=branch.curve.control_points[..., 2:])

    _points = []
    for t in _greville_abscissae(branch.curve.knot, branch.curve.degree):
            stuv=evaluate_nurbs_curve(branch.curve, t, d_order=0)["C"]
            _points.append((evaluate_nurbs_surface(surface1,stuv[0],stuv[1],d_order=0)['S']+evaluate_nurbs_surface(surface2,stuv[2],stuv[3],d_order=0)['S'])/2)

    cpt, kv = interpolate_curve(np.array(_points), degree=branch.curve.degree,use_centripetal=True,remove_duplicates=True,tol=1e-12)
    branch.curve_xyz = NURBSCurveTuple(branch.curve.order, knot=np.array(kv), control_points=np.array(cpt), weights=np.ones(len(cpt)))

def compute_point_xyz(point:SSXPoint, surface1:NURBSSurfaceTuple, surface2:NURBSSurfaceTuple,**kwargs):
    point.xyz=evaluate_nurbs_surface(surface1,point.stuv[0],point.stuv[1],d_order=0)['S']

def bez_ssx(H1: NDArray[np.float64], H2: NDArray[np.float64], *, atol: float = 0.001, angle_tol: float = 0.052,
            tol: float = 1e-8, gjk_max_iter: int = 64, gm_eps: float = 1e-5, gm_tol: float = 1e-8, max_depth: int = 24,
            magic_start_depth: int = 6, flat_angle: float = 0.015, march_samples: int = 8,
            deflate_hard_case: bool = True) -> tuple[list[SSXBranch], list[SSXPoint]]:
    param_tol = _param_tol_from(tol, atol)
    g1 = GaussMapBern.from_surf(H1, rational=True)
    g2 = GaussMapBern.from_surf(H2, rational=True)
    branches, points = _bez_ssx_recursive(g1, g2, (0.0, 1.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), atol=atol, tol=tol,
                                          param_tol=param_tol, gjk_tol=tol, gjk_max_iter=gjk_max_iter, gm_eps=gm_eps,
                                          gm_tol=gm_tol, max_depth=max_depth, magic_start_depth=magic_start_depth,
                                          parallel_angle=angle_tol, flat_angle=flat_angle, march_samples=march_samples,
                                          deflate_hard_case=deflate_hard_case)
    branches = _merge_branches_global(branches, param_tol,atol=atol)
    branches = _close_branches(branches, param_tol)
    points = _prune_points_on_branches(points, branches, param_tol)
    return branches, points


def nurbs_ssx(surf1, surf2, *, atol: float = 0.001, angle_tol: float = 0.052, tol: float = 1e-8, gjk_max_iter: int = 64,
              gm_eps: float = 1e-5, gm_tol: float = 1e-8, max_depth: int = 24, magic_start_depth: int = 2,
              flat_angle: float = 0.15, march_samples: int = 8,
              deflate_hard_case: bool = True) -> tuple[list[SSXBranch], list[SSXPoint]]:
    s1 = surf1
    s2 =surf2
    if s1.control_points.shape ==s2.control_points.shape:
        if np.allclose(s1.control_points,s2.control_points):
            return
    s1d = decompose_surface(s1)
    s2d = decompose_surface(s2)

    s1d_h = [_to_homogeneous(s.control_points, s.weights) for s in s1d]
    s2d_h = [_to_homogeneous(s.control_points, s.weights) for s in s2d]

    s1_intervals = []
    for s in s1d:
        (u0, u1), (v0, v1) = _surface_interval(s)
        s1_intervals.append((u0, u1, v0, v1))
    s2_intervals = []
    for s in s2d:
        (u0, u1), (v0, v1) = _surface_interval(s)
        s2_intervals.append((u0, u1, v0, v1))

    tree1 = build_bvh([AABB.from_points(_from_homogeneous(H)[0].reshape(-1, 3)) for H in s1d_h])
    tree2 = build_bvh([AABB.from_points(_from_homogeneous(H)[0].reshape(-1, 3)) for H in s2d_h])

    gm1: list[GaussMapBern | None] = [None] * len(s1d_h)
    gm2: list[GaussMapBern | None] = [None] * len(s2d_h)

    branches: list[SSXBranch] = []
    points: list[SSXPoint] = []
    param_tol = _param_tol_from(tol, atol)
    candidates=bvh_intersect(tree1, tree2, exact=False)

    for obj1, obj2 in candidates:
        i = obj1.object
        j = obj2.object

        H1 = s1d_h[i]
        H2 = s2d_h[j]

        if gm1[i] is None:
            gm1[i] = GaussMapBern.from_surf(H1, rational=True)
        if gm2[j] is None:
            gm2[j] = GaussMapBern.from_surf(H2, rational=True)

        bch, pts = _bez_ssx_recursive(gm1[i], gm2[j], s1_intervals[i], s2_intervals[j], atol=atol, tol=tol,
                                      param_tol=param_tol, gjk_tol=tol, gjk_max_iter=gjk_max_iter, gm_eps=gm_eps,
                                      gm_tol=gm_tol, max_depth=max_depth, magic_start_depth=magic_start_depth,
                                      parallel_angle=angle_tol, flat_angle=flat_angle, march_samples=march_samples,
                                      deflate_hard_case=deflate_hard_case)
        branches.extend(bch)
        points.extend(pts)

    branches = _merge_branches_global(branches, param_tol,atol=atol)
    branches = _close_branches(branches, param_tol)
    points = _prune_points_on_branches(points, branches, param_tol)

    for b in branches:
        compute_branch_curves(b, surf1,surf2,atol=atol)
    for p in points:
        compute_point_xyz(p, surf1,surf2)
    return branches, points


# ======================================================================================
# Public API: detect_intersections
# ======================================================================================

def detect_intersections(
    surf1,
    surf2,
    spt: float = 0.1,
    tol: float = 1e-8,
    *,
    gjk_max_iter: int = 64,
    gm_eps: float = 1e-5,
    gm_tol: float = 1e-8,
    max_depth: int = 24,
    magic_start_depth: int = 3,
    parallel_angle: float = 0.053,
    flat_angle: float = 0.2,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Returns pairs of potentially intersecting *simple* subpatches as (H1, H2),
    where each H is a homogeneous Bezier control net (..,4).
    """
    # Decompose to Bezier patches (NURBS -> Bezier), keep homogeneous control nets
    s1d = [_to_homogeneous(s.control_points, s.weights) for s in decompose_surface(surf1)]
    s2d = [_to_homogeneous(s.control_points, s.weights) for s in decompose_surface(surf2)]

    # BVH over Euclidean control points
    tree1 = build_bvh([AABB.from_points(_from_homogeneous(H)[0].reshape(-1, 3)) for H in s1d])
    tree2 = build_bvh([AABB.from_points(_from_homogeneous(H)[0].reshape(-1, 3)) for H in s2d])

    # Cache root GaussMapBern per patch index
    gm1: list[GaussMapBern | None] = [None] * len(s1d)
    gm2: list[GaussMapBern | None] = [None] * len(s2d)

    out: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []

    for obj1, obj2 in bvh_intersect(tree1, tree2, exact=False):
        i = obj1.object
        j = obj2.object

        H1 = s1d[i]
        H2 = s2d[j]

        # If you want an extra cheap early cull:
        # bb1 = np.asarray(aabb(_from_homogeneous(H1)[0].reshape(-1,3)))
        # bb2 = np.asarray(aabb(_from_homogeneous(H2)[0].reshape(-1,3)))
        # if not aabb_intersect_fast_3d(bb1, bb2): continue

        if gm1[i] is None:
            gm1[i] = GaussMapBern.from_surf(H1, rational=True)
        if gm2[j] is None:
            gm2[j] = GaussMapBern.from_surf(H2, rational=True)

        out.extend(
            _refine_pair_to_simple(
                gm1[i],
                gm2[j],
                spt=spt,
                gjk_tol=tol,
                gjk_max_iter=gjk_max_iter,
                gm_eps=gm_eps,
                gm_tol=gm_tol,
                max_depth=max_depth,
                magic_start_depth=magic_start_depth,
                parallel_angle=parallel_angle,
                flat_angle=flat_angle,
            )
        )

    return out


if __name__ == "__main__":
    from mmcore._test_data import ssx as td

    S1, S2 =(_nurbs_to_tuple(i) for i in  td[1])

    TOL = 1e-3
    import time

    s = time.perf_counter_ns()
    res = detect_intersections(S1, S2,spt= TOL)
    print((time.perf_counter_ns() - s) * 1e-9)
    fff = []

    for i, j in res:
        ip = np.array(i)
        jp = np.array(j)

        if np.any(np.isnan(ip.flatten())) or np.any(np.isnan(jp.flatten())):
            import warnings

            warnings.warn("NAN")
        else:

            a=bern_to_nurbs_bezier(ip)
            b=bern_to_nurbs_bezier(jp)
            fff.append((a,b))

    with open("/Users/sthv/PycharmProjects/mmcore/tests/norm1.pkl", "wb") as f:
        pickle.dump(fff, f)

    S1, S2 =(_nurbs_to_tuple(i) for i in  td[2])

    import time

    s = time.perf_counter_ns()
    res = detect_intersections(S1, S2, spt=TOL)
    print((time.perf_counter_ns() - s) * 1e-9)
    fff = []
    s = time.perf_counter_ns()
    ptss = []
    for i, j in res:
        ip = np.array(i)
        jp = np.array(j)

        if np.any(np.isnan(ip.flatten())) or np.any(np.isnan(jp.flatten())):
            import warnings

            warnings.warn("NAN")
        else:
            a = bern_to_nurbs_bezier(ip)
            b = bern_to_nurbs_bezier(jp)
            fff.append((a, b))

    with open("/Users/sthv/PycharmProjects/mmcore/tests/norm2.pkl", "wb") as f:
        pickle.dump(fff, f)

    s1 = NURBSSurfaceTuple(
        order_u=2,
        order_v=2,
        knot_u=np.array([0.0, 0.0, 256.50009777, 256.50009777]),
        knot_v=np.array([0.0, 0.0, 259.71657438, 259.71657438]),
        control_points=np.array(
            [
                [[-128.25004889, -129.85828719, 67.43742325], [-128.25004889, 129.85828719, 0.0]],
                [[128.25004889, -46.98266257, 0.0], [128.25004889, 129.85828719, 0.0]],
            ]
        ),
        weights=np.array([[1.0, 1.0], [1.0, 1.0]]),
    )

    s2 = NURBSSurfaceTuple(
        order_u=2,
        order_v=2,
        knot_u=np.array([0.0, 0.0, 256.50009777, 256.50009777]) ,
        knot_v=np.array([0.0, 0.0, 259.71657438, 259.71657438]) ,
        control_points=np.array(
            [
                [[-128.25004889, -129.85828719, 0.0], [-128.25004889, 129.85828719, 0.0]],
                [[128.25004889, -129.85828719, 0.0], [128.25004889, 129.85828719, 0.0]],
            ]
        ),
        weights=np.array([[1.0, 1.0], [1.0, 1.0]]),
    )

    import time

    s = time.perf_counter_ns()
    res = detect_intersections(s1, s2, spt=TOL)
    print((time.perf_counter_ns() - s) * 1e-9)
    fff = []
    s = time.perf_counter_ns()
    ptss = []
    for i, j in res:
        ip = np.array(i)
        jp = np.array(j)

        if np.any(np.isnan(ip.flatten())) or np.any(np.isnan(jp.flatten())):
            import warnings

            warnings.warn("NAN")
        else:
            a = bern_to_nurbs_bezier(ip)
            b = bern_to_nurbs_bezier(jp)
            fff.append((a, b))

    with open("/Users/sthv/PycharmProjects/mmcore/tests/norm4.pkl", "wb") as f:
        pickle.dump(fff, f)





    s1 = NURBSSurfaceTuple(
        order_u=3,
        order_v=3,
        knot_u=np.array([0., 0., 0., 17.27875959, 17.27875959,
                         34.55751919, 34.55751919, 51.83627878, 51.83627878, 69.11503838,
                         69.11503838, 69.11503838]),
        knot_v=np.array([-17.27875959, -17.27875959, -17.27875959, -0.,
                         -0., 17.27875959, 17.27875959, 17.27875959]),
        control_points=np.array([[[6., 0., -11.],
                                  [17., 0., -11.],
                                  [17., 0., -0.],
                                  [17., 0., 11.],
                                  [6., 0., 11.]],

                                 [[6., 0., -11.],
                                  [17., 11., -11.],
                                  [17., 11., -0.],
                                  [17., 11., 11.],
                                  [6., 0., 11.]],

                                 [[6., 0., -11.],
                                  [6., 11., -11.],
                                  [6., 11., -0.],
                                  [6., 11., 11.],
                                  [6., 0., 11.]],

                                 [[6., 0., -11.],
                                  [-5., 11., -11.],
                                  [-5., 11., -0.],
                                  [-5., 11., 11.],
                                  [6., 0., 11.]],

                                 [[6., 0., -11.],
                                  [-5., 0., -11.],
                                  [-5., 0., -0.],
                                  [-5., 0., 11.],
                                  [6., 0., 11.]],

                                 [[6., 0., -11.],
                                  [-5., -11., -11.],
                                  [-5., -11., -0.],
                                  [-5., -11., 11.],
                                  [6., 0., 11.]],

                                 [[6., 0., -11.],
                                  [6., -11., -11.],
                                  [6., -11., -0.],
                                  [6., -11., 11.],
                                  [6., 0., 11.]],

                                 [[6., 0., -11.],
                                  [17., -11., -11.],
                                  [17., -11., -0.],
                                  [17., -11., 11.],
                                  [6., 0., 11.]],

                                 [[6., 0., -11.],
                                  [17., 0., -11.],
                                  [17., 0., -0.],
                                  [17., 0., 11.],
                                  [6., 0., 11.]]]),
        weights=np.array([[1., 0.70710678, 1., 0.70710678, 1.],
                          [0.70710678, 0.5, 0.70710678, 0.5, 0.70710678],
                          [1., 0.70710678, 1., 0.70710678, 1.],
                          [0.70710678, 0.5, 0.70710678, 0.5, 0.70710678],
                          [1., 0.70710678, 1., 0.70710678, 1.],
                          [0.70710678, 0.5, 0.70710678, 0.5, 0.70710678],
                          [1., 0.70710678, 1., 0.70710678, 1.],
                          [0.70710678, 0.5, 0.70710678, 0.5, 0.70710678],
                          [1., 0.70710678, 1., 0.70710678, 1.]])
    )

    s2 = NURBSSurfaceTuple(
        order_u=3,
        order_v=3,
        knot_u=np.array([0., 0., 0., 17.27875959, 17.27875959,
                         34.55751919, 34.55751919, 51.83627878, 51.83627878, 69.11503838,
                         69.11503838, 69.11503838]),
        knot_v=np.array([-17.27875959, -17.27875959, -17.27875959, -0.,
                         -0., 17.27875959, 17.27875959, 17.27875959]),
        control_points=np.array([[[11., 0., -13.],
                                  [22., 0., -13.],
                                  [22., 0., -2.],
                                  [22., 0., 9.],
                                  [11., 0., 9.]],

                                 [[11., 0., -13.],
                                  [22., 11., -13.],
                                  [22., 11., -2.],
                                  [22., 11., 9.],
                                  [11., 0., 9.]],

                                 [[11., 0., -13.],
                                  [11., 11., -13.],
                                  [11., 11., -2.],
                                  [11., 11., 9.],
                                  [11., 0., 9.]],

                                 [[11., 0., -13.],
                                  [0., 11., -13.],
                                  [0., 11., -2.],
                                  [0., 11., 9.],
                                  [11., 0., 9.]],

                                 [[11., 0., -13.],
                                  [0., 0., -13.],
                                  [0., 0., -2.],
                                  [0., 0., 9.],
                                  [11., 0., 9.]],

                                 [[11., 0., -13.],
                                  [0., -11., -13.],
                                  [0., -11., -2.],
                                  [0., -11., 9.],
                                  [11., 0., 9.]],

                                 [[11., 0., -13.],
                                  [11., -11., -13.],
                                  [11., -11., -2.],
                                  [11., -11., 9.],
                                  [11., 0., 9.]],

                                 [[11., 0., -13.],
                                  [22., -11., -13.],
                                  [22., -11., -2.],
                                  [22., -11., 9.],
                                  [11., 0., 9.]],

                                 [[11., 0., -13.],
                                  [22., 0., -13.],
                                  [22., 0., -2.],
                                  [22., 0., 9.],
                                  [11., 0., 9.]]]),
        weights=np.array([[1., 0.70710678, 1., 0.70710678, 1.],
                          [0.70710678, 0.5, 0.70710678, 0.5, 0.70710678],
                          [1., 0.70710678, 1., 0.70710678, 1.],
                          [0.70710678, 0.5, 0.70710678, 0.5, 0.70710678],
                          [1., 0.70710678, 1., 0.70710678, 1.],
                          [0.70710678, 0.5, 0.70710678, 0.5, 0.70710678],
                          [1., 0.70710678, 1., 0.70710678, 1.],
                          [0.70710678, 0.5, 0.70710678, 0.5, 0.70710678],
                          [1., 0.70710678, 1., 0.70710678, 1.]])
    )

    s = time.perf_counter_ns()
    res = detect_intersections(s1, s2, spt=TOL)
    print((time.perf_counter_ns() - s) * 1e-9)
    fff = []
    s = time.perf_counter_ns()
    ptss = []
    for i, j in res:
        ip = np.array(i)
        jp = np.array(j)

        if np.any(np.isnan(ip.flatten())) or np.any(np.isnan(jp.flatten())):
            import warnings

            warnings.warn("NAN")
        else:
            a = bern_to_nurbs_bezier(ip)
            b = bern_to_nurbs_bezier(jp)
            fff.append((a, b))

    with open("/Users/sthv/PycharmProjects/mmcore/tests/norm7.pkl", "wb") as f:
        pickle.dump(fff, f)













    from mmcore.geom._nurbs_knots import normalize_knots_surface_inplace
    normalize_knots_surface_inplace(S1)
    normalize_knots_surface_inplace(S2)
    s=time.perf_counter()
    curves,points= nurbs_ssx(S1, S2, atol=TOL)
    print((time.perf_counter() - s))
    print(S1)
    if curves:
        print(curves[0].curve)

    crvs_all=[]
    from mmcore.geom._nurbs_eval import evaluate_nurbs_curve,evaluate_nurbs_surface
    from mmcore.geom._nurbs_interp import interpolate_curve

    for c in curves:

        crvs_all.append(c.curve_xyz)

    with open("/Users/sthv/PycharmProjects/mmcore/tests/norm3.pkl", "wb") as f:
        pickle.dump(crvs_all, f)
    print(len(crvs_all),'branches')
    pts = []
    for pt in points:
        pts.append(evaluate_nurbs_surface(S1, pt.stuv[0], pt.stuv[1], d_order=0)["S"].tolist())
    print(pts)
