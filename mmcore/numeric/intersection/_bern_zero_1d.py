"""Robust univariate Bernstein zero-finder for squared-distance boundary analysis.

Finds parameter values where a univariate Bernstein polynomial touches zero.
Designed for non-negative polynomials (squared distance restrictions), where
zeros correspond to minima that reach zero.

Key design choices:
- Sign-change count on derivative coefficients bounds the number of minima
- Subdivision avoids splitting near roots (picks center of longest positive run)
- Endpoint values are exact (Bernstein coefficients at t=0 and t=1)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _count_sign_changes(coeffs: NDArray) -> int:
    """Count the number of sign changes in a 1D coefficient array.

    Zeros are skipped (not counted as sign changes).
    """
    signs = np.sign(coeffs)
    # Remove zeros — they don't contribute sign changes
    signs = signs[signs != 0]
    if len(signs) <= 1:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def _bernstein_deriv_coeffs_1d(coeffs: NDArray) -> NDArray:
    """Derivative coefficients for a 1D Bernstein polynomial.

    If f(t) = sum_i B_{i,n}(t) * c_i, then
    f'(t) = n * sum_i B_{i,n-1}(t) * (c_{i+1} - c_i).
    """
    n = len(coeffs) - 1
    if n <= 0:
        return np.zeros(1, dtype=coeffs.dtype)
    return n * np.diff(coeffs)


def _de_casteljau_eval_1d(coeffs: NDArray, t: float) -> float:
    """Evaluate a 1D Bernstein polynomial at parameter t."""
    c = coeffs.copy()
    n = len(c) - 1
    omt = 1.0 - t
    for r in range(1, n + 1):
        c[:n + 1 - r] = omt * c[:n + 1 - r] + t * c[1:n + 2 - r]
    return float(c[0])


def _de_casteljau_split_1d(coeffs: NDArray, t: float) -> tuple[NDArray, NDArray]:
    """Split a 1D Bernstein polynomial at parameter t."""
    n = len(coeffs) - 1
    tmp = coeffs.copy()
    left = [tmp[0]]
    right_rev = [tmp[n]]
    for r in range(1, n + 1):
        tmp[:n + 1 - r] = (1.0 - t) * tmp[:n + 1 - r] + t * tmp[1:n + 2 - r]
        left.append(tmp[0])
        right_rev.append(tmp[n - r])
    return np.array(left, dtype=coeffs.dtype), np.array(right_rev[::-1], dtype=coeffs.dtype)


def _newton_bernstein_root_1d(coeffs: NDArray, t0: float, tol: float = 1e-12,
                               max_it: int = 30) -> float:
    """Newton's method on a 1D Bernstein polynomial to find a root.

    Clamped to [0, 1].
    """
    deriv = _bernstein_deriv_coeffs_1d(coeffs)
    t = float(np.clip(t0, 0.0, 1.0))
    for _ in range(max_it):
        f = _de_casteljau_eval_1d(coeffs, t)
        df = _de_casteljau_eval_1d(deriv, t)
        if abs(df) < 1e-30:
            break
        dt = -f / df
        t_new = float(np.clip(t + dt, 0.0, 1.0))
        if abs(t_new - t) < tol:
            t = t_new
            break
        t = t_new
    return t


def _newton_bernstein_min_1d(coeffs: NDArray, t0: float, tol: float = 1e-12,
                              max_it: int = 30) -> float:
    """Find a local minimum of a 1D Bernstein polynomial via Newton on its derivative.

    Clamped to [0, 1].
    """
    deriv = _bernstein_deriv_coeffs_1d(coeffs)
    return _newton_bernstein_root_1d(deriv, t0, tol=tol, max_it=max_it)


def _longest_positive_run_center(coeffs: NDArray) -> float:
    """Find the center of the longest run of strictly positive values.

    Returns parameter t in [0, 1] corresponding to the center index / degree.
    Falls back to 0.5 if no positive run exists.
    """
    n = len(coeffs)
    if n <= 1:
        return 0.5

    best_start = -1
    best_len = 0
    cur_start = -1
    cur_len = 0

    for i in range(n):
        if coeffs[i] > 0:
            if cur_start < 0:
                cur_start = i
                cur_len = 1
            else:
                cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_start = -1
            cur_len = 0

    if best_len == 0:
        return 0.5

    center_idx = best_start + best_len / 2.0
    degree = n - 1
    if degree == 0:
        return 0.5
    return float(np.clip(center_idx / degree, 0.05, 0.95))


def find_bernstein_zeros_1d(coeffs: NDArray, atol: float,
                             t_start: float = 0.0, t_end: float = 1.0,
                             max_depth: int = 30) -> list[float]:
    """Find all parameter values where a 1D Bernstein polynomial touches zero.

    The polynomial is assumed to represent a squared-distance restriction,
    so it is non-negative in exact arithmetic. We look for minima that
    reach zero (value < atol^2).

    Parameters
    ----------
    coeffs : NDArray
        1D array of Bernstein coefficients.
    atol : float
        Geometric tolerance. A zero is accepted if value < atol^2.
    t_start, t_end : float
        Parameter range this polynomial covers (for mapping back to parent domain).
    max_depth : int
        Maximum recursion depth for subdivision.

    Returns
    -------
    list of float
        Parameter values (in [t_start, t_end]) where the polynomial touches zero.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64).ravel()
    atol_sq = atol * atol

    if len(coeffs) == 0:
        return []

    # Single coefficient (degree 0)
    if len(coeffs) == 1:
        if abs(coeffs[0]) < atol_sq:
            return [0.5 * (t_start + t_end)]
        return []

    zeros = []

    # 1. Check endpoints (exact values)
    if coeffs[0] < atol_sq:
        zeros.append(t_start)
    if coeffs[-1] < atol_sq:
        zeros.append(t_end)

    # 2. Quick exit: all coefficients positive and above threshold → no interior zeros
    if np.min(coeffs) >= atol_sq:
        return zeros

    # 3. Analyze derivative for interior minima
    deriv = _bernstein_deriv_coeffs_1d(coeffs)
    n_sign_changes = _count_sign_changes(deriv)

    if n_sign_changes == 0:
        # Derivative doesn't change sign → polynomial is monotone → no interior minimum
        return zeros

    if n_sign_changes <= 2:
        # At most one interior minimum. Use Newton on derivative to find it.
        # Seed from the argmin of coefficients
        degree = len(coeffs) - 1
        seed_idx = int(np.argmin(coeffs))
        seed_t = seed_idx / max(degree, 1)

        t_min = _newton_bernstein_min_1d(coeffs, seed_t)
        val_min = _de_casteljau_eval_1d(coeffs, t_min)

        if val_min < atol_sq and 0.0 < t_min < 1.0:
            # Map local parameter to parent domain
            t_global = t_start + t_min * (t_end - t_start)
            # Avoid duplicates with endpoints
            if (not zeros or
                (abs(t_global - t_start) > atol * 0.01 and
                 abs(t_global - t_end) > atol * 0.01)):
                zeros.append(t_global)

        return zeros

    # 4. 3+ sign changes: subdivide
    if max_depth <= 0:
        # Fallback: try Newton from argmin
        degree = len(coeffs) - 1
        seed_idx = int(np.argmin(coeffs))
        seed_t = seed_idx / max(degree, 1)
        t_min = _newton_bernstein_min_1d(coeffs, seed_t)
        val_min = _de_casteljau_eval_1d(coeffs, t_min)
        if val_min < atol_sq and 0.0 < t_min < 1.0:
            t_global = t_start + t_min * (t_end - t_start)
            if not zeros or (abs(t_global - t_start) > atol * 0.01 and
                             abs(t_global - t_end) > atol * 0.01):
                zeros.append(t_global)
        return zeros

    # Pick safe subdivision point: center of longest positive run in coefficients
    t_split = _longest_positive_run_center(coeffs)

    left, right = _de_casteljau_split_1d(coeffs, t_split)
    t_mid = t_start + t_split * (t_end - t_start)

    # Recurse on both halves (but don't re-report the shared endpoint at t_mid)
    left_zeros = find_bernstein_zeros_1d(left, atol, t_start, t_mid, max_depth - 1)
    right_zeros = find_bernstein_zeros_1d(right, atol, t_mid, t_end, max_depth - 1)

    # Merge, deduplicating near the split point
    all_zeros = []
    for z in left_zeros:
        all_zeros.append(z)
    for z in right_zeros:
        # Skip if too close to an existing zero
        if not any(abs(z - ez) < atol * 0.01 for ez in all_zeros):
            all_zeros.append(z)

    return sorted(all_zeros)
