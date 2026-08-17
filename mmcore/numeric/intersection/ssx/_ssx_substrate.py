"""Shared SSX substrate (formerly the head of _ssx4.py).

The v4 solver that lived below these definitions was superseded by
_bez_ssx5/_nssx5 and deleted in the 2026-08 restructure (Step 10).
What remains is the substrate the live engine imports: GaussMapBern,
separate_gauss_maps, hemisphere witnesses, _trust_gjk, and the
SSXPoint/SSXBranch result types with _append_unique_point.
"""
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

            #print(f'{func.__name__}({kwargs})->({res})', delta*1e-9)
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


def reset_witness_rng() -> None:
    """Reset the hemisphere-witness shuffle RNGs to their fixed seeds.

    The randomized-incremental cap search consumes a module-global PRNG
    (numpy `_RNG` in the Python path; a never-reseeded xorshift32 state in
    the Cython fast path), so on MARGINAL near-tangent normal sets the
    witness outcome — and through `separate_gauss_maps` the main loop's
    trace-vs-subdivide decision — depended on how many draws EARLIER calls
    had consumed: bit-identical repeated bez_ssx calls returned different
    branch topologies (measured [3,2,3,2,2,2] by call index; fresh
    processes always agree). A witness miss is completeness-only (found
    caps are margin-verified), so pinning the shuffle costs nothing sound;
    reproducibility is worth far more than the randomized average case.
    Called once at every bez_ssx entry."""
    global _RNG
    _RNG = np.random.default_rng(0)
    try:
        from mmcore.numeric._cap_witness import set_rng_seed
        set_rng_seed(0)
    except Exception:
        pass
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

        # Keep the exact homogeneous normal-numerator net.  Normalizing
        # each control ray is a harmless positive rescaling for the TOP
        # cone, but de Casteljau subdivision does not commute with those
        # independent rescalings: child cones can then exclude a true
        # normal of the restricted surface.  Split the exact net and only
        # normalize rays on read in ``map_dirs_net``.

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
    kind: str = "transversal"   # 'transversal' | 'tangential' | 'overlap'
    curve_xyz: NURBSCurveTuple=field(default=None,init=False)
    curve_st: NURBSCurveTuple=field(default=None,init=False)
    curve_uv: NURBSCurveTuple=field(default=None,init=False)


def _append_unique_point(points: list[SSXPoint], stuv: NDArray[np.float64], tol: float) -> None:
    for p in points:
        if np.max(np.abs(p.stuv - stuv)) <= tol:
            return
    points.append(SSXPoint(stuv=stuv))


