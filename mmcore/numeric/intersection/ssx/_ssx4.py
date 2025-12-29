from __future__ import annotations

import functools
import pickle

import numpy as np
from numpy.typing import NDArray

from mmcore.geom._nurbs_eval import _nurbs_to_tuple
from mmcore.numeric.sbern import bern_to_nurbs_bezier


import numpy as np
from numpy.typing import NDArray

from mmcore.geom._nurbs_knots import decompose_surface
from mmcore.numeric._aabb import aabb, aabb_intersect_fast_3d, aabb_intersection
from mmcore.numeric.algorithms.cygjk import gjk
from mmcore.numeric.vectors import unit
from mmcore.numeric.gauss_map import compute_gauss_map_rational
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

def _from_homogeneous(H: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    H = np.asarray(H, dtype=np.float64)
    w = H[..., -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        P = H[..., :-1] / w[..., None]
    return P, w


def _to_homogeneous(P: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
    P = np.asarray(P, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if P.shape[:-1] != w.shape:
        raise ValueError(f"weights shape {w.shape} must match control_points[...,0] shape {P.shape[:-1]}")
    return np.concatenate((P * w[..., None], w[..., None]), axis=-1)


# ======================================================================================
# Tensor-product Bezier evaluation + 1D splits
# ======================================================================================

def _eval_tensor_bezier(ctrl: NDArray[np.float64], u: float, v: float) -> NDArray[np.float64]:
    """
    Evaluate tensor-product Bezier control net at (u,v) using de Casteljau.
    ctrl: (p+1, q+1, dim)
    """
    B = np.array(ctrl, dtype=np.float64, copy=True)
    p = B.shape[0] - 1
    q = B.shape[1] - 1

    # de Casteljau in u (vectorized over v,dim)
    omt = 1.0 - u
    for r in range(1, p + 1):
        B[: p - r + 1] = omt * B[: p - r + 1] + u * B[1 : p - r + 2]

    # de Casteljau in v on the reduced curve
    C = B[0].copy()  # (q+1, dim)
    omt = 1.0 - v
    for r in range(1, q + 1):
        C[: q - r + 1] = omt * C[: q - r + 1] + v * C[1 : q - r + 2]

    return C[0]


def _derivative_net_u(H: NDArray[np.float64]) -> NDArray[np.float64]:
    p = H.shape[0] - 1
    if p <= 0:
        return np.zeros((1, H.shape[1], H.shape[2]), dtype=np.float64)
    return p * (H[1:, :, :] - H[:-1, :, :])


def _derivative_net_v(H: NDArray[np.float64]) -> NDArray[np.float64]:
    q = H.shape[1] - 1
    if q <= 0:
        return np.zeros((H.shape[0], 1, H.shape[2]), dtype=np.float64)
    return q * (H[:, 1:, :] - H[:, :-1, :])


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
import numpy as np

import numpy as np
import numpy as np

import numpy as np

def great_circle_separable(
    U: np.ndarray,
    V: np.ndarray,
    *,
    gamma: float = 1e-3,          # required angular margin in dot-product units
    max_updates: int = 200_000,
    passes: int = 50,
    seed: int = 0,
    shuffle: bool = True,
    max_step: float = 2.0,
    normalize_w: bool = True,
    min_w_norm: float = 1e-12,    # reject degenerate w
):
    """
    Great-circle (hemisphere) separability test on S^2 with a strict margin gamma.

    Find w != 0 such that:
        w·u >=  gamma  for all u in U
        w·v <= -gamma  for all v in V

    Returns (is_separable, w_unit, info).
    If separable, w_unit is a unit normal defining the great circle w·x = 0.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if U.ndim != 2 or V.ndim != 2 or U.shape[1] != 3 or V.shape[1] != 3:
        raise ValueError("U and V must be arrays of shape (m,3) and (k,3).")

    # Ensure unit vectors (safe even if already unit)
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-30)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-30)

    # Combine into labeled constraints y*(w·x) >= gamma
    X = np.vstack([U, V])
    y = np.concatenate([np.ones(len(U), dtype=np.float64),
                        -np.ones(len(V), dtype=np.float64)])

    rng = np.random.default_rng(seed)
    m = X.shape[0]
    idx = np.arange(m)

    # Start with a small random w to avoid the w=0 degeneracy
    w = rng.normal(size=3).astype(np.float64)
    wn = np.linalg.norm(w)
    w = w / max(wn, 1e-30)

    updates = 0

    def max_violation(w_):
        # violation for y*(w·x) >= gamma is max(gamma - y*(w·x))
        s = y * (X @ w_)
        return float(np.max(gamma - s))

    best_w = w.copy()
    best_v = max_violation(best_w)

    for p in range(passes):
        if shuffle:
            rng.shuffle(idx)

        any_violation = False
        for i in idx:
            s = y[i] * (X[i] @ w)
            if s >= gamma:
                continue

            any_violation = True
            v = gamma - s  # positive violation amount
            eta = min(max_step, v)

            w += eta * y[i] * X[i]
            updates += 1

            if normalize_w:
                wn = np.linalg.norm(w)
                if wn > 0:
                    w /= wn

            if updates >= max_updates:
                break

        v_now = max_violation(w)
        if v_now < best_v:
            best_v = v_now
            best_w = w.copy()

        if not any_violation or updates >= max_updates:
            break

    wn = np.linalg.norm(best_w)
    if wn < min_w_norm:
        # degenerate; treat as not separable
        return False, best_w, {
            "updates": updates,
            "passes_done": p + 1,
            "max_violation": best_v,
            "gamma": gamma,
            "reason": "degenerate_w",
        }

    w_unit = best_w / wn

    # Decide separability: all constraints satisfy margin => max_violation <= 0
    is_sep = best_v <= 0.0

    info = {
        "updates": updates,
        "passes_done": p + 1,
        "max_violation": best_v,
        "gamma": gamma,
    }
    return is_sep, w_unit, info

def spherical_cap_separable(
    U: np.ndarray,
    V: np.ndarray,
    *,
    max_updates: int = 300_000,
    passes: int = 60,
    tol: float = 1e-12,
    shuffle: bool = False,
    seed: int = 0,
    max_step: float = 2.0,
    average: bool = False,
    return_witness: bool = False,
):
    """
    Determine separability on the sphere by a spherical cap boundary:
        w·u + b >= 0 for all u in U
        w·v + b <= 0 for all v in V

    Equivalent to linear separability in R^3 with a bias term.
    Returns (is_separable, w, b, info).
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if U.ndim != 2 or V.ndim != 2 or U.shape[1] != 3 or V.shape[1] != 3:
        raise ValueError("U and V must be arrays of shape (m,3) and (k,3).")

    X = np.vstack([U, V])
    y = np.concatenate([np.ones(len(U), dtype=np.float64),
                        -np.ones(len(V), dtype=np.float64)])

    # Ensure unit vectors (optional but stabilizing)
    Xn = np.linalg.norm(X, axis=1)
    X = X / np.maximum(Xn, 1e-30)[:, None]

    # Augment with bias
    Xa = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])

    rng = np.random.default_rng(seed)
    w = np.zeros(4, dtype=np.float64)
    w_avg = np.zeros_like(w) if average else None
    avg_count = 0

    idx = np.arange(Xa.shape[0])
    updates = 0

    def max_violation(w_):
        s = y * (Xa @ w_)
        return float(np.max(-s))

    best_w = w.copy()
    best_max_viol = np.inf

    for p in range(passes):
        if shuffle:
            rng.shuffle(idx)

        any_violation = False
        for i in idx:
            s = y[i] * (Xa[i] @ w)
            if s >= -tol:
                if average:
                    w_avg += w
                    avg_count += 1
                continue

            any_violation = True
            v = -s
            eta = min(max_step, v)
            w += eta * y[i] * Xa[i]
            updates += 1

            if average:
                w_avg += w
                avg_count += 1

            if updates >= max_updates:
                break

        cand_w = (w_avg / avg_count) if (average and avg_count > 0) else w
        mv = max_violation(cand_w)
        if mv < best_max_viol:
            best_max_viol = mv
            best_w = cand_w.copy()

        if not any_violation or updates >= max_updates:
            break

    # Normalize for interpretability: scale so ||w3||=1
    w3 = best_w[:3].copy()
    b = float(best_w[3])
    nrm = np.linalg.norm(w3)
    if nrm > 0:
        w3 /= nrm
        b /= nrm

    mv_out = best_max_viol
    info = {
        "updates": updates,
        "passes_done": p + 1,
        "max_violation": mv_out,
        "tol": tol,
        "averaged": bool(average),
    }

    is_sep = mv_out <= tol
    if return_witness:
        s = y * (Xa @ best_w)
        worst_i = int(np.argmax(-s))
        info["worst_index"] = worst_i
        info["worst_label"] = float(y[worst_i])
        info["worst_margin"] = float(s[worst_i])

    return is_sep, w3, b, info

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
            from inspect import getlineno,currentframe ,getframeinfo
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
    from mmcore.numeric.intersection.csx import nurbs_csx

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
