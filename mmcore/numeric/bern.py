import math
from functools import lru_cache

import numpy as np
from mmcore.numeric.binom import binomial_coefficient_py
from mmcore.numeric.newton import cnewton

from mmcore.numeric.fdm import newtons_method, bounded_newtons_method
from mmcore.numeric.newton.cnewton import newton_method2
from numpy.typing import NDArray
from typing import NamedTuple, Sequence

from scipy import optimize

from mmcore.numeric.sbern import bern_to_nurbs_bezier

CONVERGED = 'converged'
SIGNERR = 'sign error'
CONVERR = 'convergence error'
VALUEERR = 'value error'
INPROGRESS = 'No error'
def bernstein_product_conv(deg: int) -> np.ndarray:
    """
    Exact Bernstein product convolution tensor for equal degrees:
      B_i^deg(t) * B_j^deg(t) = Conv[r,i,j] * B_r^{2deg}(t), with r = i + j
      Conv[r,i,j] = C(deg,i) C(deg,j) / C(2deg, r)
    Returns array with shape (2*deg+1, deg+1, deg+1).
    """
    m = deg
    C2m = np.array([comb(2*m, r) for r in range(2*m+1)], dtype=np.float64)
    Ci  = np.array([comb(m, i)    for i in range(m+1)],  dtype=np.float64)
    Conv = np.zeros((2*m+1, m+1, m+1), dtype=np.float64)
    for i in range(m+1):
        for j in range(m+1):
            r = i + j
            Conv[r, i, j] = (Ci[i] * Ci[j]) / C2m[r]
    return Conv

def _bivariate_squared_norm_net(E: np.ndarray) -> np.ndarray:
    """
    Given a bivariate Bernstein vector control net E of degree (p,q):
        E.shape == (p+1, q+1, d)
    compute the bivariate control net F of f(u,v) = ||E(u,v)||^2 of degree (2p, 2q):
        F.shape == (2p+1, 2q+1)
    """
    if E.ndim != 3:
        raise ValueError("E must have shape (p+1, q+1, d).")
    p = E.shape[0] - 1
    q = E.shape[1] - 1

    ConvU = bernstein_product_conv(p)   # (2p+1, p+1, p+1)
    ConvV = bernstein_product_conv(q)   # (2q+1, q+1, q+1)
    ConvU2 = ConvU.reshape(2*p + 1, (p + 1) * (p + 1))
    ConvV2 = ConvV.reshape(2*q + 1, (q + 1) * (q + 1))

    # Convolve along u: for each (j,j'), A = sum_k E[:,j,k] * E[:,j',k]^T, then apply ConvU
    Tu = np.zeros((2*p + 1, q + 1, q + 1), dtype=np.float64)
    Dj = E.transpose(1, 0, 2)  # (q+1, p+1, d)
    for j in range(q + 1):
        Dj_mat = Dj[j]  # (p+1, d)
        for jp in range(q + 1):
            Djp_mat = Dj[jp]  # (p+1, d)
            A = Dj_mat @ Djp_mat.T  # (p+1, p+1) = sum_k E[i,j,k] E[i',j',k]
            Tu[:, j, jp] = ConvU2 @ A.ravel()

    # Convolve along v
    F = np.zeros((2*p + 1, 2*q + 1), dtype=np.float64)
    for r in range(2 * p + 1):
        B = Tu[r, :, :]  # (q+1, q+1)
        F[r, :] = ConvV2 @ B.ravel()

    return F

def _square_univariate_bernstein_net(w: np.ndarray) -> np.ndarray:
    """
    Given univariate Bernstein control points w (degree p), return control points
    of w(t)^2 in Bernstein form of degree 2p. Shape -> (2p+1,).
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    p = w.shape[0] - 1
    Conv = bernstein_product_conv(p)  # (2p+1, p+1, p+1)
    # Contraction: sum_{i,i'} Conv[r,i,i'] * w[i] * w[i']
    return np.tensordot(Conv, np.outer(w, w), axes=([1, 2], [0, 1]))

def bernstein_rational_distance_squared_nets(P1: np.ndarray, W1: np.ndarray,
                                             P2: np.ndarray, W2: np.ndarray):
    """
    Compute bivariate Bernstein control nets (G, H) for the squared distance between two rational Bézier curves:
        C1(u) = (Σ_i w1[i] P1[i] B_i^p(u)) / (Σ_i w1[i] B_i^p(u))
        C2(v) = (Σ_j w2[j] P2[j] B_j^q(v)) / (Σ_j w2[j] B_j^q(v))

    Returns G, H such that:
        ||C1(u) - C2(v)||^2 = g(u,v)/h(u,v),
      with
        g(u,v) = Σ_{r,s} G[r,s] B_r^{2p}(u) B_s^{2q}(v),
        h(u,v) = Σ_{r,s} H[r,s] B_r^{2p}(u) B_s^{2q}(v).

    Parameters
    ----------
    P1 : (p+1, d)  control points of curve 1 in Euclidean space
    W1 : (p+1,)    positive weights of curve 1
    P2 : (q+1, d)  control points of curve 2 in Euclidean space
    W2 : (q+1,)    positive weights of curve 2

    Returns
    -------
    G : (2p+1, 2q+1)  bivariate Bernstein control net of the numerator g(u,v)
    H : (2p+1, 2q+1)  bivariate Bernstein control net of the denominator h(u,v)

    Notes
    -----
    Let c1(u) = Σ_i (w1[i] P1[i]) B_i^p(u),  w1(u) = Σ_i w1[i] B_i^p(u),
        c2(v) = Σ_j (w2[j] P2[j]) B_j^q(v),  w2(v) = Σ_j w2[j] B_j^q(v).
    Then
        g(u,v) = || c1(u) w2(v) - c2(v) w1(u) ||^2,
        h(u,v) = w1(u)^2 * w2(v)^2.
    This function constructs both control nets exactly using Bernstein product identities.
    """
    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64).reshape(-1)
    W2 = np.asarray(W2, dtype=np.float64).reshape(-1)

    if P1.ndim != 2 or P2.ndim != 2:
        raise ValueError("P1 and P2 must be 2D arrays of shapes (p+1,d) and (q+1,d).")
    if P1.shape[1] != P2.shape[1]:
        raise ValueError("P1 and P2 must have the same ambient dimension d.")
    if P1.shape[0] != W1.shape[0] or P2.shape[0] != W2.shape[0]:
        raise ValueError("Weights must match the number of control points.")

    # Homogeneous numerators (control nets of c1 and c2)
    C1_num = W1[:, None] * P1  # (p+1, d)
    C2_num = W2[:, None] * P2  # (q+1, d)

    # Build bivariate vector net E_{i,j,:} = c1_i * w2_j - c2_j * w1_i   (degree (p,q))
    E = C1_num[:, None, :] * W2[None, :, None] - C2_num[None, :, :] * W1[:, None, None]  # (p+1,q+1,d)

    # Numerator G: squared norm net of E  (degree (2p,2q))
    G = _bivariate_squared_norm_net(E)

    # Denominator H: (w1(u)^2) * (w2(v)^2) -> outer of the squared-univariate nets
    W1_sq_net = _square_univariate_bernstein_net(W1)  # (2p+1,)
    W2_sq_net = _square_univariate_bernstein_net(W2)  # (2q+1,)
    H = np.outer(W1_sq_net, W2_sq_net)                # (2p+1, 2q+1)

    return G, H

def bernstein_rational_distance_squared_nets_homog(H1: np.ndarray, H2: np.ndarray):
    """
    Variant that accepts *homogeneous* control points directly.
    H1 : (p+1, d+1) with last column weights (w), first d columns = w*P
    H2 : (q+1, d+1)
    Returns (G, H) as above.
    """
    H1 = np.asarray(H1, dtype=np.float64)
    H2 = np.asarray(H2, dtype=np.float64)
    if H1.ndim != 2 or H2.ndim != 2:
        raise ValueError("H1 and H2 must be 2D arrays of shapes (p+1,d+1) and (q+1,d+1).")
    if H1.shape[1] != H2.shape[1]:
        raise ValueError("H1 and H2 must have the same ambient homogeneous dimension (d+1).")

    # Extract numerator control points (w*P) and weights
    C1_num = H1[:, :-1]  # (p+1, d)
    W1     = H1[:, -1]   # (p+1,)
    C2_num = H2[:, :-1]  # (q+1, d)
    W2     = H2[:, -1]   # (q+1,)

    # E_{i,j,:} = c1_i * w2_j - c2_j * w1_i
    E = C1_num[:, None, :] * W2[None, :, None] - C2_num[None, :, :] * W1[:, None, None]

    # Build G, H exactly
    G = _bivariate_squared_norm_net(E)
    W1_sq_net = _square_univariate_bernstein_net(W1)
    W2_sq_net = _square_univariate_bernstein_net(W2)
    H = np.outer(W1_sq_net, W2_sq_net)
    return G, H

def bernstein_partial_derivative_coeffs(control_grid: np.ndarray, axis: int) -> np.ndarray:
    """
    Compute the Bernstein control grid for the partial derivative along a given parametric axis.

    Parameters
    ----------
    control_grid : np.ndarray
        Tensor-product Bernstein coefficients arranged with the last axis as the value dimension.
        Examples of valid shapes:
          - (U, N)                     : 1D curve with N-dimensional values
          - (U, V, N)                  : surface
          - (U, V, P, N)               : trivariate
          - (U, V, P, Q, N)            : four-parameter
        Here U, V, P, Q are (degree+1) counts along each parametric axis, and N is the value dimension.

    axis : int
        Parametric axis to differentiate along. This is indexed among the *parametric* axes only
        (i.e., from 0 up to control_grid.ndim - 2). The last axis (value dimension) is not valid.

    Returns
    -------
    np.ndarray
        A new control grid holding the Bernstein coefficients of the partial derivative d/da
        along the specified axis. Its shape equals `control_grid.shape` with the length
        of the chosen axis reduced by 1 (degree reduced by one).

    Notes
    -----
    For a single parametric axis with degree n, the derivative control points are:
        D_i = n * (C_{i+1} - C_i)   for i = 0..(n-1)
    This generalizes by applying the forward difference along the selected axis in the
    multi-dimensional control grid.

    Examples
    --------
    # 1D cubic (U=4) with 3D control points (N=3)
    C = np.array([[0.,0.,0.],
                  [1.,0.,0.],
                  [1.,1.,0.],
                  [0.,1.,0.]])  # shape (4, 3)
    dC_du = bernstein_partial_derivative_coeffs(C, axis=0)  # shape (3, 3)

    # 2D example: bicubic (U=4, V=4) scalar field (N=1)
    S = np.random.rand(4, 4, 1)
    dS_du = bernstein_partial_derivative_coeffs(S, axis=0)  # shape (3, 4, 1)
    dS_dv = bernstein_partial_derivative_coeffs(S, axis=1)  # shape (4, 3, 1)
    """
    if control_grid.ndim < 2:
        raise ValueError(
            "control_grid must have at least one parametric axis and a trailing value axis."
        )

    param_ndim = control_grid.ndim - 1  # exclude trailing value dimension
    # Normalize axis w.r.t. parametric axes
    if axis < 0:
        axis += param_ndim
    if not (0 <= axis < param_ndim):
        raise ValueError(
            f"'axis' must be in [0, {param_ndim-1}] for the parametric axes; "
            "the trailing value axis is not differentiable."
        )

    degree = control_grid.shape[axis] - 1
    # Forward differences along the chosen axis
    diffs = np.diff(control_grid, axis=axis)
    # If degree == 0, diffs already has length 0 along that axis (derivative of constant is zero)
    return diffs * degree


def bernstein_distance_squared_net(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Given two Bézier curves in Bernstein form:
        C1(u) = sum_{i=0..p} P[i] * B_i^p(u),  P.shape == (p+1, d)
        C2(v) = sum_{j=0..q} Q[j] * B_j^q(v),  Q.shape == (q+1, d)
    return the bivariate Bernstein control net F of:
        f(u,v) = || C1(u) - C2(v) ||^2
    so that
        f(u,v) = sum_{r=0..2p} sum_{s=0..2q} F[r, s] * B_r^{2p}(u) * B_s^{2q}(v)

    Parameters
    ----------
    P : (p+1, d) float64
    Q : (q+1, d) float64

    Returns
    -------
    F : (2p+1, 2q+1) float64
        Bivariate control net of f(u,v).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.ndim != 2 or Q.ndim != 2:
        raise ValueError("P and Q must be 2D arrays of shapes (p+1,d) and (q+1,d).")
    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same ambient dimension d.")
    p = P.shape[0] - 1
    q = Q.shape[0] - 1
    d = P.shape[1]
    
    # Difference control net D_{i,j,:} = P_i - Q_j (tensor-product net)
    # Shape: (p+1, q+1, d)
    D = P[:, None, :] - Q[None, :, :]
    
    # Exact Bernstein product convolution tensors in u and v
    ConvU = bernstein_product_conv(p)  # (2p+1, p+1, p+1)
    ConvV = bernstein_product_conv(q)  # (2q+1, q+1, q+1)
    
    # For efficient contractions, flatten the (i,i') and (j,j') pairs:
    ConvU2 = ConvU.reshape(2 * p + 1, (p + 1) * (p + 1))  # (2p+1, (p+1)^2)
    ConvV2 = ConvV.reshape(2 * q + 1, (q + 1) * (q + 1))  # (2q+1, (q+1)^2)
    
    # First convolve along u (degree p -> 2p), but sum over Cartesian dims right away.
    # For each (j, j'), build A = sum_k D[:,j,k] * D[:,j',k]^T  (shape (p+1, p+1)),
    # then Tu[:, j, j'] = ConvU2 @ A.ravel()
    Tu = np.zeros((2 * p + 1, q + 1, q + 1), dtype=np.float64)
    Dj = D.transpose(1, 0, 2)  # (q+1, p+1, d) to help locality
    
    for j in range(q + 1):
        # (p+1, d)
        Dj_mat = Dj[j]  # rows indexed by i, columns by k
        for jp in range(q + 1):
            Djp_mat = Dj[jp]  # (p+1, d)
            # A[i,i'] = sum_k D[i,j,k] * D[i',j',k]
            A = Dj_mat @ Djp_mat.T  # (p+1, p+1)
            Tu[:, j, jp] = ConvU2 @ A.ravel()
    
    # Then convolve along v (degree q -> 2q) for each r, again exact.
    F = np.zeros((2 * p + 1, 2 * q + 1), dtype=np.float64)
    for r in range(2 * p + 1):
        B = Tu[r, :, :]  # (q+1, q+1)
        F[r, :] = ConvV2 @ B.ravel()
    
    return F


import numpy as np

def bernstein_eval_1d(P, u):
    # Horner/de Casteljau eval
    Q = P.copy()
    n = P.shape[0] - 1
    for r in range(1, n + 1):
        Q = (1.0 - u) * Q[:-1] + u * Q[1:]
    return Q[0].item()
def de_casteljau_split_nd(control_grid: np.ndarray, axis: int, t) -> tuple[np.ndarray, np.ndarray]:
    """
    Subdivide an N-D Bernstein control grid along one parametric axis at parameter t using de Casteljau.

    Parameters
    ----------
    control_grid : np.ndarray
        Tensor-product Bernstein coefficients with trailing value dimension.
        Examples of valid shapes:
          - (U, N)                        : 1D curve with N-dimensional values
          - (U, V, N)                     : surface
          - (U, V, P, N)                  : trivariate
          - (U, V, P, Q, N)               : 4D-parameter tensor
        Here each parametric axis length = degree+1 along that axis, and the last axis is value dimension.

    axis : int
        Parametric axis to split along (0-based among parametric axes only).
        The last axis (value dimension) is not valid.

    t : float or np.ndarray
        Split parameter. Typically in [0, 1].
        You may pass a scalar, or an array broadcastable to the product of the *other* parametric axes.
        If an array is provided, it is interpreted as varying t per “fiber” orthogonal to `axis`.

    Returns
    -------
    (left, right) : tuple[np.ndarray, np.ndarray]
        The Bernstein control grids of the two sub-patches:
          - `left`  represents the polynomial restricted to the interval [0, t] along `axis`.
          - `right` represents the polynomial restricted to the interval [t, 1] along `axis`.
        Both have the same shape as `control_grid`.

    Notes
    -----
    - This is a pure-control-net operation; evaluation points are not needed.
    - For each fixed index of the other parametric axes (a “fiber”), the standard 1D de Casteljau
      triangle is built along `axis`. The first and last entries of each level form the left and
      right control nets, respectively.
    - Complexity is O(n) linear interpolations along the chosen axis, vectorized across all fibers.

    Examples
    --------
    # 1D cubic (U=4) → split into two cubics at t=0.3
    C = np.array([[0.,0.],
                  [1.,0.],
                  [1.,1.],
                  [0.,1.]])       # shape (4, 2)
    L, R = de_casteljau_split_nd(C, axis=0, t=0.3)

    # 2D bicubic (U=4, V=4) scalar field (N=1) → split along U at t=0.5
    S = np.random.rand(4, 4, 1)
    SL, SR = de_casteljau_split_nd(S, axis=0, t=0.5)
    """
    if control_grid.ndim < 2:
        raise ValueError("control_grid must have at least one parametric axis and a trailing value axis.")
    
    param_ndim = control_grid.ndim - 1  # exclude trailing value dimension
    # Normalize axis with respect to parametric axes
    if axis < 0:
        axis += param_ndim
    if not (0 <= axis < param_ndim):
        raise ValueError(
            f"'axis' must be in [0, {param_ndim - 1}] among parametric axes; the trailing value axis is not allowed."
        )
    
    # Move the chosen parametric axis to the front to simplify the recurrence
    A = np.moveaxis(control_grid, axis, 0)  # shape: (m, *B, N)
    m = A.shape[0]  # m = degree+1 along chosen axis
    B_shape = A.shape[1:-1]  # all other parametric axes
    N = A.shape[-1]  # value dimension
    
    if m == 0:
        raise ValueError("Invalid control grid: zero length along the chosen axis.")
    
    # Prepare t to broadcast as (1, *B, 1), so it multiplies each fiber and value uniformly
    t_arr = np.asarray(t, dtype=np.result_type(A.dtype, np.float64))
    if t_arr.ndim == 0:
        # Scalar t broadcasts automatically; keep as scalar
        t_b = t_arr
        omt_b = 1.0 - t_arr
    else:
        # Broadcast to the grid of fibers
        try:
            t_b_fibers = np.broadcast_to(t_arr, B_shape)
        except ValueError as e:
            raise ValueError(
                f"t with shape {t_arr.shape} is not broadcastable to fiber shape {B_shape}."
            ) from e
        # Expand to (1, *B, 1)
        t_b = t_b_fibers[(None,) + tuple(slice(None) for _ in B_shape) + (None,)]
        omt_b = 1.0 - t_b
    
    # de Casteljau pyramid along the first axis (originally `axis`)
    left_layers = []
    right_layers = []
    cur = A
    for _ in range(m - 1):
        # Collect boundary elements of this level
        left_layers.append(cur[0])  # shape: (*B, N)
        right_layers.append(cur[-1])  # shape: (*B, N)
        # Next level via linear interpolation
        cur = omt_b * cur[:-1] + t_b * cur[1:]
    # Final point (shared by both halves)
    left_layers.append(cur[0])
    right_layers.append(cur[0])
    
    # Stack and reorder axes back
    left = np.stack(left_layers, axis=0)  # shape: (m, *B, N)
    right = np.stack(right_layers[::-1], axis=0)  # reverse to get [point, ..., last]
    left = np.moveaxis(left, 0, axis)  # back to original axis placement
    right = np.moveaxis(right, 0, axis)
    
    return left, right

import numpy as np
from math import comb


def bernstein_trim_nd(control_grid: np.ndarray, ranges) -> np.ndarray:
    """
    Trim a tensor-product Bernstein control grid to per-axis sub-intervals using de Casteljau
    (matrix form), returning a grid with the SAME SHAPE as the input.

    Parameters
    ----------
    control_grid : np.ndarray
        Shape (U, V, P, Q, ..., N), where the last axis N is the value dimension (e.g. 1, 2, 3, 4).
        Each parametric length equals degree+1 along that axis.
    ranges : array-like of shape (D, 2)
        For D = control_grid.ndim - 1 parametric axes, ranges[k] = (a_k, b_k), 0 <= a_k <= b_k <= 1.

    Returns
    -------
    np.ndarray
        Trimmed control grid on the subdomain Π_k [a_k, b_k], reparameterized back to [0,1]^D,
        with the same shape as `control_grid`.

    Notes
    -----
    • Only one full-size output array is allocated once. Per axis, two small scratch arrays of size
      (degree_k+1, value_dim) are reused across all fibers.
    • Complexity is O( Σ_k ( (n_k+1)^2 * Π_{j≠k} (n_j+1) * N ) ), which is the optimal
      separable cost for de Casteljau-based trimming without building massive Kronecker products.
    """
    if control_grid.ndim < 2:
        raise ValueError("Need at least one parametric axis plus a trailing value axis.")

    D = control_grid.ndim - 1
    ranges = np.asarray(ranges, dtype=float)
    if ranges.shape != (D, 2):
        raise ValueError(f"`ranges` must have shape ({D}, 2).")

    # Output (one full-size allocation); do all transforms in-place on `out`.
    out = control_grid.astype(np.result_type(control_grid.dtype, np.float64), copy=True)
    Nval = out.shape[-1]

    for ax in range(D):
        n = out.shape[ax] - 1
        if n <= 0:
            continue

        a, b = float(ranges[ax, 0]), float(ranges[ax, 1])
        if a == 0.0 and b == 1.0:
            continue  # no-op on this axis

        T = _trim_transform(n, a, b, dtype=out.dtype)  # (n+1, n+1)

        # Small, per-axis scratch (reused for every fiber)
        buf = np.empty((n + 1, Nval), dtype=out.dtype)
        res = np.empty_like(buf)

        # Iterate over all fibers orthogonal to `ax`
        fiber_shape = out.shape[:ax] + out.shape[ax + 1 : -1]
        if fiber_shape:  # multi-fiber case
            for idx in np.ndindex(fiber_shape):
                # Build index: (..., :, ..., :) where ':' at `ax` and trailing ':' for value dim
                sl = []
                it = iter(idx)
                for k in range(D):
                    sl.append(slice(None) if k == ax else next(it))
                sl.append(slice(None))  # value axis
                sl = tuple(sl)

                buf[...] = out[sl]          # (n+1, N)
                res[...] = T @ buf          # (n+1, N)
                out[sl] = res
        else:
            # Only one fiber (D == 1)
            buf[...] = out[(slice(None), slice(None))]
            res[...] = T @ buf
            out[(slice(None), slice(None))] = res

    return out
def de_casteljau_section_nd(control_grid: np.ndarray, axis: int, t, keepdims: bool = True) -> np.ndarray:
    """
    Fix the parameter on `axis` at `t` via de Casteljau, returning an ND control grid
    with that axis collapsed (length 1 if keepdims=True, otherwise removed).
    """
    param_ndim = control_grid.ndim - 1
    if axis < 0:
        axis += param_ndim
    A = np.moveaxis(control_grid, axis, 0)  # (m, *B, N)
    B_shape = A.shape[1:-1]

    t_arr = np.asarray(t, dtype=np.result_type(A.dtype, np.float64))
    if t_arr.ndim == 0:
        t_b, omt_b = t_arr, 1.0 - t_arr
    else:
        t_b_fibers = np.broadcast_to(t_arr, B_shape)
        t_b = t_b_fibers[(None,) + tuple(slice(None) for _ in B_shape) + (None,)]
        omt_b = 1.0 - t_b

    cur = A
    for _ in range(A.shape[0] - 1):
        cur = omt_b * cur[:-1] + t_b * cur[1:]

    # cur has shape (1, *B, N). Put axis back and squeeze if requested.
    out = np.moveaxis(cur, 0, axis)
    return out if keepdims else np.squeeze(out, axis=axis)

def bern_eval(grid,params):
    current_grid=grid
    for i,v in enumerate(params)    :
        current_grid=de_casteljau_section_nd(current_grid,i,v)
    return current_grid
def spread(ctrl):
    """Geometric spread (used to choose split axis)."""
    mu = ctrl.mean(axis=0, keepdims=True)
    return float(np.max(np.linalg.norm(ctrl - mu, axis=1)))


import functools


@functools.lru_cache(maxsize=None)
def _pascal_table(n):
    """Pascal rows 0..n (n+1, n+1) upper-left filled; dtype=int64."""
    P = np.zeros((n + 1, n + 1), dtype=np.int64)
    P[:, 0] = 1
    for i in range(1, n + 1):
        P[i, 1:i] = P[i - 1, 1:i] + P[i - 1, :i - 1]
        P[i, i] = 1
    return P

import numpy as np

# ---------------- Bernstein basis derivatives (DP) ----------------

def _bernstein_derivatives(n, u, m):
    """
    All derivatives up to order m of degree-n Bernstein polynomials at u.
    Returns D of shape (m+1, n+1) with D[k,i] = d^k/du^k B_i^n(u).
    """
    m = min(m, n)
    B = np.zeros((n+1, n+1), dtype=np.float64)
    B[0, 0] = 1.0
    u1 = 1.0 - u
    for j in range(1, n+1):
        saved = 0.0
        for k in range(j):
            temp = B[k, j-1]
            B[k, j] = saved + u1 * temp
            saved = u * temp
        B[j, j] = saved

    D = np.zeros((m+1, n+1), dtype=np.float64)
    D[0, :] = B[:, n]

    if m >= 1:
        for i in range(n):
            D[1, i] = n * (B[i, n-1] - B[i+1, n-1])
        D[1, n] = -n * B[n, n-1]

    # Higher derivatives via reduced-degree Bernstein and falling factorial
    for k in range(2, m+1):
        nf = 1.0
        for t in range(k):
            nf *= (n - t)
        col = B[:, n - k]
        for i in range(n+1):
            s_min = max(0, i - (n - k))
            s_max = min(k, i)
            acc = 0.0
            # acc = sum_{s=s_min..s_max} (-1)^s C(k,s) B_{i-s}^{n-k}(u)
            for s in range(s_min, s_max + 1):
                # tiny k (<= few hundred) so python comb via small pascal isn't needed here
                acc += ((-1.0)**s) * _comb_fast(k, s) * col[i - s]
            D[k, i] = nf * acc
    return D


def __comb_fast(n, k):
    """Small, integer-safe binomial (n choose k) as float."""
    if k < 0 or k > n:
        return 0.0
    k = min(k, n - k)
    num = 1
    den = 1
    for i in range(1, k + 1):
        num *= (n - (k - i))
        den *= i
    return float(num / den)
_comb_fast=binomial_coefficient_py



# ---------------- Curves (compact): R[k, :] ----------------

def bezier_curve_derivatives_compact(ctrl, u, n, rational=False):
    """
    Curve in Bernstein form.

    Parameters
    ----------
    ctrl : (N, dim) or (N, dim_h) with weights in last coordinate if rational=True
    u    : float
    n    : max derivative order

    Returns
    -------
    R : (m+1, dim_e) where m=min(n, degree), R[k,:] = d^k P(u) / du^k
        dim_e = dim (non-rational) or dim_h-1 (rational)
    """
    P = np.asarray(ctrl, dtype=np.float64)
    N = P.shape[0]
    p = N - 1
    m = min(n, p)

    D = _bernstein_derivatives(p, u, m)  # (m+1, N)

    if not rational:
        return D @ P  # (m+1, dim)

    d_e = P.shape[1] - 1
    Pw = P[:, :-1]
    w  = P[:, -1]

    Xj = D @ Pw          # (m+1, d_e)
    wj = D @ w           # (m+1,)

    # Rational jet recurrence (vector-friendly but simple)
    R = np.zeros_like(Xj)
    w0 = wj[0]
    if w0 == 0.0:
        raise ZeroDivisionError("Weight evaluates to zero at u.")
    R[0] = Xj[0] / w0

    # Precompute Pascal rows once
    Pas = _pascal_table(m)  # (m+1, m+1)
    for k in range(1, m + 1):
        coeffs = Pas[k, 1:k+1].astype(np.float64)        # shape (k,)
        # Build weighted sum sum_{j=1..k} C(k,j) * wj * R_{k-j}
        # Align R terms in reverse: R[k-1], R[k-2], ..., R[0]
        R_rev = R[:k][::-1]                              # (k, dim)
        w_slice = wj[1:k+1][:, None]                     # (k, 1)
        acc = (coeffs[:, None] * w_slice * R_rev).sum(axis=0)
        R[k] = (Xj[k] - acc) / w0
    return R


# ---------------- Surfaces (compact): C[j, i, :] ----------------

def bezier_surface_derivatives_compact(ctrl, u, v, n, rational=False):
    """
    Tensor-product Bézier surface in Bernstein form.

    Parameters
    ----------
    ctrl : (Nu, Nv, dim) or (Nu, Nv, dim_h) with weights last if rational
    u,v  : floats
    n    : max total derivative order

    Returns
    -------
    C : (mv+1, mu+1, dim_e) with C[j, i, :] = ∂^{i+j} P / ∂u^i ∂v^j,
        where mu=min(n, pu), mv=min(n, pv). dim_e = dim or dim_h-1.
    """
    S = np.asarray(ctrl, dtype=np.float64)
    Nu, Nv = S.shape[0], S.shape[1]
    pu, pv = Nu - 1, Nv - 1
    mu = min(n, pu)
    mv = min(n, pv)

    Bu = _bernstein_derivatives(pu, u, mu)  # (mu+1, Nu)
    Bv = _bernstein_derivatives(pv, v, mv)  # (mv+1, Nv)

    if not rational:
        d = S.shape[2]
        # Two GEMMs to get all mixed partials:
        U = (Bu @ S.reshape(Nu, Nv * d)).reshape(mu + 1, Nv, d)                 # (mu+1, Nv, d)
        T = (Bv @ U.transpose(1, 0, 2).reshape(Nv, (mu + 1) * d))\
              .reshape(mv + 1, mu + 1, d)                                       # (mv+1, mu+1, d)
        return T  # C[j, i, :]
    else:
        d_e = S.shape[2] - 1
        SX = S[..., :d_e]
        Sw = S[..., -1]

        # X mixed partials
        Ux = (Bu @ SX.reshape(Nu, Nv * d_e)).reshape(mu + 1, Nv, d_e)
        Tx = (Bv @ Ux.transpose(1, 0, 2).reshape(Nv, (mu + 1) * d_e))\
               .reshape(mv + 1, mu + 1, d_e)

        # w mixed partials (scalar)
        Uw = Bu @ Sw.reshape(Nu, Nv)                    # (mu+1, Nv)
        Tw = (Bv @ Uw.T)                                # (mv+1, mu+1)

        w00 = Tw[0, 0]
        if w00 == 0.0:
            raise ZeroDivisionError("Weight evaluates to zero at (u,v).")

        # Vectorized rational recurrence per total order s (compact tensor)
        C = np.zeros((mv + 1, mu + 1, d_e), dtype=np.float64)
        C[0, 0] = Tx[0, 0] / w00

        # Precompute Pascal tables
        Pu_tab = _pascal_table(mu)   # (mu+1, mu+1)
        Pv_tab = _pascal_table(mv)   # (mv+1, mv+1)

        max_s = max(mu, mv)
        for s in range(1, max_s + 1):
            i_max = min(s, mu)
            I = np.arange(i_max + 1)
            J = s - I
            mask = J <= mv
            if not np.any(mask):
                continue
            I = I[mask]; J = J[mask]
            Tx_s = Tx[J, I]                       # (M, d_e)
            acc = np.zeros_like(Tx_s)

            a_max = int(I.max())
            b_max = int(J.max())

            Bi = Pu_tab[I, :a_max + 1].astype(np.float64)   # (M, a_max+1)
            Bj = Pv_tab[J, :b_max + 1].astype(np.float64)   # (M, b_max+1)

            # Accumulate over (a,b) ≠ (0,0); still loops a,b but vectorizes over M, d_e
            for a in range(0, a_max + 1):
                Ii = I - a
                valid_a = Ii >= 0
                if not np.any(valid_a):
                    continue
                Bi_a = Bi[valid_a, a][:, None]              # (m', 1)
                for b in range(0, b_max + 1):
                    if a == 0 and b == 0:
                        continue
                    Jb = J[valid_a] - b
                    valid_b = Jb >= 0
                    if not np.any(valid_b):
                        continue
                    rows = np.where(valid_a)[0][valid_b]    # indices in 0..M-1
                    w_ab = Tw[b, a]                         # scalar
                    Bj_b = Bj[valid_a, b][valid_b][:, None] # (m'', 1)
                    C_sub = C[Jb[valid_b], Ii[valid_b], :]  # (m'', d_e)
                    acc[rows] += (Bi_a[valid_b] * Bj_b) * w_ab * C_sub

            C[J, I] = (Tx_s - acc) / w00
        return C

def get_partial(C, i_u, j_v):
    """C[j, i, :] convention: returns ∂^{i+j}P/∂u^i ∂v^j as (dim,)"""
    return C[j_v, i_u, :]

def total_order_band(C, s):
    """
    Return all mixed partials with total order s in the canonical order
    [ (i=0,j=s), (i=1,j=s-1), ..., (i=s,j=0) ] as an array of shape (s+1, dim).
    """
    j = np.arange(s, -1, -1)           # s, s-1, ..., 0  (v-der orders)
    i = np.arange(0, s+1)              # 0, 1, ..., s    (u-der orders)
    return C[j, i, :]                  # (s+1, dim)

def ders_to_triangular(C, n=None):
    """
    Convert compact C[j,i,:] to triangular packed list [P, Pu, Pv, Puu, Puv, Pvv, ...].
    If n is None, uses n = min(C.shape[0]-1, C.shape[1]-1).
    """
    mv, mu = C.shape[0]-1, C.shape[1]-1
    if n is None:
        n = min(mv, mu)
    out = []
    for s in range(n+1):
        for i in range(s+1):
            j = s - i
            val = C[j, i, :] if (j <= mv and i <= mu) else np.zeros(C.shape[2], dtype=C.dtype)
            out.append(val)
    return np.vstack(out)
import time


def bern_no_sign_change(coeffs):
    return (np.min(coeffs) > 0.0) or (np.max(coeffs) < 0.0)
def bern_gt(b1,b2):
    return spread(b1)  >spread(b2)
def bern_ge(b1,b2):
    return spread(b1)  >spread(b2)
def bern_lt(b1,b2):
    return spread(b1)  <spread(b2)
class BernPartial(NamedTuple):
    axis:int
    coeffs:NDArray

def eval_bezier(P, t, dims=None):
    Q = P.copy()
    n = P.shape[0] - 1
    
    for _ in range(n):
        
        Q = (1.0 - t) * Q[:-1] + t * Q[1:]
    return Q[0]
def map_local_to_global(u_loc,u0, u1):
    return (u0 + (u1 - u0) * u_loc)
def _bernstein_basis_matrix(n: int, t, dtype=np.float64) -> np.ndarray:
    """
    Bernstein basis B^n_i(t) for i=0..n evaluated at t.

    Parameters
    ----------
    n : int
        Degree.
    t : float or 1D array-like
        Parameter(s) in [0,1]. Scalars are allowed.
    dtype : numpy dtype
        Floating dtype for the basis values.

    Returns
    -------
    B : (M, n+1) ndarray
        Basis matrix; M = number of samples in `t`.
        Row m contains [B^n_0(t_m), ..., B^n_n(t_m)].
    """
    tt = np.asarray(t, dtype=dtype)
    if tt.ndim == 0:
        tt = tt.reshape(1)
    M = tt.shape[0]

    i = np.arange(n + 1, dtype=int)                          # (n+1,)
    binom = np.array([binomial_coefficient_py(n, j) for j in range(n + 1)], dtype=dtype)  # (n+1,)

    # B_i^n(t) = C(n,i) * t^i * (1-t)^(n-i)
    t_pow   = tt[:, None] ** i[None, :]                       # (M, n+1)
    omt_pow = (1 - tt)[:, None] ** (n - i)[None, :]           # (M, n+1)
    return (t_pow * omt_pow) * binom[None, :]


def bernstein_eval_nd(control_grid: np.ndarray, params, keepdims: bool = False) -> np.ndarray:
    """
    Evaluate an ND Bernstein tensor-product polynomial at given parameters.

    Parameters
    ----------
    control_grid : np.ndarray
        Shape (n0+1, n1+1, ..., n_{D-1}+1, N), with the last axis the value dimension.
        Works for scalar fields (N=1) and vector-valued (e.g., 2D/3D) outputs.
    params : sequence of length D
        params[k] is the parameter(s) for axis k; each can be a scalar or a 1D array.
        If arrays are provided for multiple axes, the result is evaluated on the full
        tensor product grid.
    keepdims : bool
        If False (default), any param axis with a scalar input is squeezed from the output.

    Returns
    -------
    values : np.ndarray
        If params are scalars → shape (N,).
        In general → shape (len(params[0]), ..., len(params[D-1]), N).
    """
    if control_grid.ndim < 2:
        raise ValueError("control_grid must have at least one parametric axis plus a trailing value axis. current shape: ", control_grid.shape, control_grid)
    D = control_grid.ndim - 1
    if len(params) != D:
        raise ValueError(f"`params` must have length {D} (one per parametric axis).")

    # Choose a working dtype that safely holds products/sums
    dtype = np.result_type(control_grid.dtype, *(np.asarray(p).dtype for p in params), np.float64)
    C = control_grid.astype(dtype, copy=False)

    # Build basis matrices per axis
    B_list = []
    sample_sizes = []
    for ax in range(D):
        n = C.shape[ax] - 1
        B = _bernstein_basis_matrix(n, params[ax], dtype=dtype)  # (M_ax, n_ax+1)
        B_list.append(B)
        sample_sizes.append(B.shape[0])

    # Contract all param axes against their basis matrices in one go via einsum.
    # Indices:
    #   C:  i0 i1 ... i{D-1} v
    #   Bk: s{k} i{k}
    # Output: s0 s1 ... s{D-1} v
    
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if 2 * D + 1 > len(letters):
        raise ValueError("Too many dimensions for the predefined index set.")

    i_idx = letters[:D]
    s_idx = letters[D:2 * D]
    v_idx = letters[2 * D]
    c_sub = "".join(i_idx) + v_idx
    b_subs = [s_idx[k] + i_idx[k] for k in range(D)]
    out_sub = "".join(s_idx) + v_idx
    expr = f"{c_sub}," + ",".join(b_subs) + f"->{out_sub}"

    values = np.einsum(expr, C, *B_list, optimize=True)

    if not keepdims:
        # Squeeze exactly the param axes where input was scalar (M==1); keep value axis.
        axes_to_squeeze = tuple(i for i, m in enumerate(sample_sizes) if m == 1)
        if axes_to_squeeze:
            values = np.squeeze(values, axis=axes_to_squeeze)

    return values
def _sign(x: float, eps: float) -> int:
    # +1 if > eps, -1 if < -eps, 0 otherwise
    if x > eps:  return 1
    if x < -eps: return -1
    return 0

def _count_sign_changes(seq, eps: float) -> int:
    changes = 0
    prev = None
    for x in seq:
        
        s = _sign(x, eps)
        if s == 0:
            # ignore zeros in change counting, but keep prev if prev already set
            continue
        if prev is None:
            prev = s
            continue
        if s != prev:
            changes += 1
            prev = s
    return changes


import numpy as np

from math import comb
from itertools import product


# ---------- Bernstein 1D trimming transforms (matrix-form de Casteljau) ----------

def _lower_transform(n: int, t: float, dtype=float) -> np.ndarray:
    """
    Lower-triangular (n+1)x(n+1) matrix L(t) for the 'left' subdivision boundary.
    Row i: L[i, j] = C(i, j) * (1 - t)^(i - j) * t^j  for j<=i, else 0.
    """
    L = np.zeros((n + 1, n + 1), dtype=dtype)
    om = float(1.0 - t)
    t = float(t)
    for i in range(n + 1):
        for j in range(i + 1):
            L[i, j] = binomial_coefficient_py(i, j) * (om ** (i - j)) * (t ** j)
    return L


def _right_transform(n: int, a: float, dtype=float) -> np.ndarray:
    """
    Upper-triangular (n+1)x(n+1) matrix R(a) producing the 'right' polygon at a.
    Identity when n==0.  R(a) = J * L(1 - a) * J.
    """
    if n == 0:
        return np.array([[1.0]], dtype=dtype)
    J = np.flipud(np.eye(n + 1, dtype=dtype))
    return J @ _lower_transform(n, 1.0 - a, dtype=dtype) @ J


def _trim_transform(n: int, a: float, b: float, dtype=float) -> np.ndarray:
    """
    (n+1)x(n+1) matrix mapping coefficients on [0,1] to coefficients on [a,b],
    reparameterized back to [0,1]. Requires 0<=a<=b<=1.
    """
    if n == 0:
        return np.array([[1.0]], dtype=dtype)
    if a == b:
        # Degenerate interval -> constant limit. Use left at t=0 after right at a.
        # Equivalent to L(0) @ R(a).
        return _lower_transform(n, 0.0, dtype=dtype) @ _right_transform(n, a, dtype=dtype)
    if a == 1.0:
        # Interval can only be [1,1] or empty; handled by a==b above.
        return _lower_transform(n, 0.0, dtype=dtype) @ _right_transform(n, 1.0, dtype=dtype)
    t2 = (b - a) / (1.0 - a) if a < 1.0 else 0.0
    return _lower_transform(n, t2, dtype=dtype) @ _right_transform(n, a, dtype=dtype)


# ---------- Multi-mode application of per-axis transforms in one shot ----------

def _apply_multi_axis_transforms(C: np.ndarray, T_list: list[np.ndarray]) -> np.ndarray:
    """
    Apply square (n_k+1)x(n_k+1) transforms T_list[k] along each parametric axis k
    in a single einsum. Keeps the last axis (value dimension) untouched.
    """
    D = len(T_list)
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if 2 * D + 1 > len(letters):
        raise ValueError("Too many dimensions for the predefined index set.")
    
    # Indices for einsum
    i_idx = letters[:D]  # input param indices
    o_idx = letters[D:2 * D]  # output param indices
    v_idx = letters[2 * D]  # value dimension
    c_sub = "".join(i_idx) + v_idx
    t_subs = [o_idx[k] + i_idx[k] for k in range(D)]
    out_sub = "".join(o_idx) + v_idx
    
    expr = f"{c_sub}," + ",".join(t_subs) + f"->{out_sub}"
    return np.einsum(expr, C, *T_list, optimize=True)


# ---------- Main: cut out a central box and return all remaining sub-patches ----------

def bernstein_cutout_box_nd(control_grid: np.ndarray,
                            param,
                            half,
                            *,
                            clamp_to_unit: bool = True,
                            tol: float = 1e-12,
                            return_ranges: bool = False):
    """
    Cut an axis-aligned box out of the parameter domain and return all Bernstein sub-patches
    of the complement (each reparameterized back to [0,1]^D, shape preserved).

    Parameters
    ----------
    control_grid : np.ndarray
        Shape (n0+1, n1+1, ..., n_{D-1}+1, N). Last axis is the value dimension.
    param : sequence of length D
        Center point (u0, ..., u_{D-1}) in parameter space (typically in [0,1]^D).
    half : sequence of length D
        Half-interval (h0, ..., h_{D-1}), defining the box [(u-h), (u+h)].
    clamp_to_unit : bool, default True
        If True, clamp the box to [0,1] along each axis before splitting.
        If the box does not intersect the domain, the function returns the original patch only.
    tol : float
        Small threshold to treat a segment length as zero.
    return_ranges : bool, default False
        If True, return a list of (subgrid, ranges) where ranges is a tuple of per-axis (lo, hi).

    Returns
    -------
    patches : list[np.ndarray]                      (if return_ranges=False)
              or list[tuple[np.ndarray, tuple]]]    (if return_ranges=True)
        All sub-patches except the central box. Each sub-patch has the same shape as `control_grid`.

    Notes
    -----
    • Per axis k we split into up to three slabs: L=[0,a_k], M=[a_k,b_k], H=[b_k,1], with matrices
      T^L_k, T^M_k, T^H_k computed once from de Casteljau (matrix form).
    • We enumerate the Cartesian product of available slabs across axes and exclude the all‑'M' box.
    • Each sub‑patch is produced by a single `einsum` that applies all T_k at once.
    """
    if control_grid.ndim < 2:
        raise ValueError("control_grid must have at least one parametric axis plus a trailing value axis.")
    
    D = control_grid.ndim - 1
    if len(param) != D or len(half) != D:
        raise ValueError(f"`param` and `half` must have length {D}.")
    
    # Working dtype
    dtype = np.result_type(control_grid.dtype, np.float64)
    C = np.asarray(control_grid, dtype=dtype, order='C')
    
    u = np.asarray(param, dtype=float).reshape(D)
    h = np.asarray(half, dtype=float).reshape(D)
    if np.any(h < 0):
        raise ValueError("All half-interval components must be nonnegative.")
    
    # Build (and optionally clamp) the box per axis
    a = u - h
    b = u + h
    if clamp_to_unit:
        a = np.maximum(0.0, a)
        b = np.minimum(1.0, b)
    
    # Prepare segments per axis: list of [('L'|'M'|'H', T_matrix, (lo,hi)), ...]
    segments_per_axis = []
    mid_exists_all_axes = True
    
    for ax in range(D):
        n = C.shape[ax] - 1
        if n < 0:
            raise ValueError("Invalid control grid axis length.")
        ak = float(a[ax])
        bk = float(b[ax])
        
        # Clamp ordering locally, just in case
        if bk < ak:
            ak, bk = bk, ak
        
        segs = []
        # Low slab [0, ak]
        if ak > tol:
            T_low = _trim_transform(n, 0.0, ak, dtype=dtype)
            segs.append(('L', T_low, (0.0, ak)))
        # Mid slab [ak, bk]
        if bk - ak > tol:
            T_mid = _trim_transform(n, ak, bk, dtype=dtype)
            segs.append(('M', T_mid, (ak, bk)))
        else:
            mid_exists_all_axes = False
        # High slab [bk, 1]
        if 1.0 - bk > tol:
            T_high = _trim_transform(n, bk, 1.0, dtype=dtype)
            segs.append(('H', T_high, (bk, 1.0)))
        
        # If no slabs (box outside domain and clamp_to_unit=False), keep full axis
        if not segs:
            I = np.eye(n + 1, dtype=dtype)
            segs = [('M', I, (0.0, 1.0))]
            # No central mid to exclude if other axes have L/H; mark mid existence as True for this axis
            # but 'all-mid' exclusion will only trigger if every axis is exactly this 'M'.
        segments_per_axis.append(segs)
    
    # Enumerate combinations across axes and build patches, skipping the all-'M' central box.
    patches = []
    for combo in product(*segments_per_axis):
        labels = [lab for (lab, _, _) in combo]
        if all(lab == 'M' for lab in labels) and mid_exists_all_axes:
            # Skip the central sub-patch (the cut-out box)
            continue
        
        T_list = [T for (_, T, _) in combo]
        sub = _apply_multi_axis_transforms(C, T_list)
        
        if return_ranges:
            ranges = tuple(r for (_, _, r) in combo)
            patches.append((sub, ranges))
        else:
            patches.append(sub)
    
    return patches


class BernRootsOutput(NamedTuple):
    roots:NDArray[float]
    errors:NDArray[float]
    iters:int


from itertools import product




def _canonical_neighbor_offsets(ndim, radius=1, include_diagonals=True):
    """
    Generate exactly half of the neighbor offsets (no duplicates).
    Rule: keep offsets whose first non-zero component is positive.
    """
    rng = range(-radius, radius + 1)
    for off in product(rng, repeat=ndim):
        if all(v == 0 for v in off):
            continue  # skip the zero offset
        if not include_diagonals and sum(v != 0 for v in off) != 1:
            continue  # keep only axis-aligned neighbors if requested
        # keep only "half" of the directions to avoid duplicates
        for v in off:
            if v != 0:
                if v > 0:
                    yield np.array(off, dtype=np.int64)
                break


def _overlap_slices(shape, offset):
    """
    For a given offset, build two slice tuples selecting overlapping regions:
      base_slices for indices i that have a neighbor at i+offset
      shift_slices for the neighbors (i+offset)
    Also return base_start, the starting index (per axis) of the base region.
    """
    base_slices, shift_slices, base_start = [], [], []
    for d, o in enumerate(offset):
        if o < 0:
            base_slices.append(slice(-o, shape[d]))
            shift_slices.append(slice(0, shape[d] + o))
            base_start.append(-o)
        elif o > 0:
            base_slices.append(slice(0, shape[d] - o))
            shift_slices.append(slice(o, shape[d]))
            base_start.append(0)
        else:
            base_slices.append(slice(0, shape[d]))
            shift_slices.append(slice(0, shape[d]))
            base_start.append(0)
    return tuple(base_slices), tuple(shift_slices), np.asarray(base_start, dtype=np.int64)


def sign_change_edges_nd(
        x,
        eps=1e-6,
        radius=1,
        include_diagonals=True,
        return_linear=False,
        allowed_pairs=None,
        xp=np,
):
    """
    Compute undirected sign-change edges for an N-D array.

    Parameters
    ----------
    x : ndarray
        Real-valued array.
    eps : float
        Threshold for sign classes: (-inf,-eps)->-1, [-eps,+eps]->0, (eps,inf)->+1.
    radius : int
        Neighborhood radius (1 gives Moore neighbors in 2D, 26-neighborhood in 3D, etc.).
    include_diagonals : bool
        If False, only axis-aligned neighbors (Manhattan) are used.
    return_linear : bool
        If True, return pairs of linear indices; else return pairs of N-D indices.
    allowed_pairs : set[tuple[int,int]] | None
        If provided, only keep edges whose endpoint sign labels are in this set
        (treating it as undirected). Example: {(-1, 1)} to ignore 0.
    xp : module
        Backend (numpy or cupy). Pass `xp=cupy` for GPU.

    Returns
    -------
    src, dst :
        If return_linear=False:
            src, dst are (M, ndim) integer arrays of coordinates.
        If return_linear=True:
            src, dst are (M,) integer arrays of linear indices.
        Each pair (src[k], dst[k]) is an undirected edge listed once.
    """
    # backend arrays
    x = xp.asarray(x)
    sg = xp.zeros_like(x, dtype=xp.int8)
    sg = sg + (x > eps) - (x < -eps)  # values in {-1, 0, 1}
    
    ndim = x.ndim
    shape = x.shape
    
    src_chunks = []
    dst_chunks = []
    
    # Iterate only over half of the offsets to avoid duplicates
    for off in _canonical_neighbor_offsets(ndim, radius, include_diagonals):
        base_sl, nbr_sl, base_start = _overlap_slices(shape, off)
        A = sg[base_sl]
        B = sg[nbr_sl]
        
        if allowed_pairs is None:
            mask = (A != B)
        else:
            # keep only selected undirected sign pairs
            mask = xp.zeros_like(A, dtype=bool)
            for a, b in allowed_pairs:
                mask |= ((A == a) & (B == b)) | ((A == b) & (B == a))
        
        # indices within the aligned (cropped) subarray
        idx_local = xp.argwhere(mask)
        if idx_local.size == 0:
            continue
        
        # map back to original coordinates
        base_idx = idx_local + base_start  # (M, ndim)
        neighbor_idx = base_idx + xp.asarray(off, dtype=base_idx.dtype)
        
        src_chunks.append(base_idx)
        dst_chunks.append(neighbor_idx)
    
    if not src_chunks:
        if return_linear:
            return xp.empty((0,), dtype=np.int64), xp.empty((0,), dtype=np.int64)
        else:
            return xp.empty((0, ndim), dtype=np.int64), xp.empty((0, ndim), dtype=np.int64)
    
    src = xp.concatenate(src_chunks, axis=0)
    dst = xp.concatenate(dst_chunks, axis=0)
    
    if return_linear:
        # Note: cupy.ravel_multi_index exists in recent CuPy versions.
        src_lin = xp.ravel_multi_index(tuple(src.T), shape)
        dst_lin = xp.ravel_multi_index(tuple(dst.T), shape)
        return src_lin, dst_lin
    
    return src, dst





from mmcore.geom._nurbs_knots import generate_knots

@lru_cache(maxsize=None)
def bern_greville_abscissae(control_points_count: int, interval=(0., 1.)) -> np.ndarray:
    """
    Greville points for degree p with knot vector U.
    xi_i = (U[i+1] + ... + U[i+p]) / p, for i=0..n
    """
    p = control_points_count - 1
    U = generate_knots(control_points_count, p, interval=interval)
    n = len(U) - p - 2
    if p <= 0:
        # degree 0 Greville points are undefined; not used in this code path
        raise ValueError("Degree must be >=1 for Greville abscissae.")
    # Sum p interior knots per control index:
    xi = np.empty(n + 1, dtype=float)
    for i in range(n + 1):
        xi[i] = np.sum(U[i + 1:i + p + 1]) / p
    return xi
@lru_cache(maxsize=None)
def bern_greville_abscissae_nd(shape, interval=None):
    if interval is None:
        interval = [(0.,1.)]*len(shape)
    return tuple(bern_greville_abscissae(shape[i],interval[i]) for i in range(len(shape)))

def zero_crossing_nd(p0, p1, d0, d1):
    """
    Compute the nD point where scalar d crosses zero along the segment [p0, p1].

    Parameters
    ----------
    p0, p1 : array_like
        Endpoints of the segment (shape (n,))
    d0, d1 : float
        Scalar values at p0 and p1, must satisfy d0 < 0 < d1

    Returns
    -------
    np.ndarray
        Coordinates of intersection point (shape (n,))
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    t = -d0 / (d1 - d0)
    return p0 + t * (p1 - p0)


def de_casteljau_subdivide_2d(control_points, u, v):
    def gen():
        for part in de_casteljau_split_nd(control_points, 0, u):
            yield from de_casteljau_split_nd(part, 1, v)

    return np.asarray(tuple(gen()))


import numpy as np
from typing import Sequence, Tuple


def de_casteljau_restrict_nd(control_grid: np.ndarray, axis: int, t, keepdims: bool = True) -> np.ndarray:
    """
    Fix the parameter on `axis` at `t` via de Casteljau, returning an ND control grid
    with that axis collapsed (length 1 if keepdims=True, otherwise removed).
    Last axis of `control_grid` is the value dimension.
    """
    if control_grid.ndim < 2:
        raise ValueError("control_grid must have at least one parametric axis and a trailing value axis.")
    
    param_ndim = control_grid.ndim - 1
    # Normalize axis w.r.t. parametric axes
    if axis < 0:
        axis += param_ndim
    if not (0 <= axis < param_ndim):
        raise ValueError(f"'axis' must be in [0, {param_ndim - 1}] among parametric axes.")
    
    # Move target param axis to front to apply 1D de Casteljau
    A = np.moveaxis(control_grid, axis, 0)  # (m, *B, N)
    m = A.shape[0]
    B_shape = A.shape[1:-1]  # other param axes
    dtype = np.result_type(A.dtype, np.float64)
    
    if m == 0:
        raise ValueError("Invalid control grid: zero length along the chosen axis.")
    
    # Prepare t as shape (1, *B, 1) for correct broadcasting
    t_arr = np.asarray(t, dtype=dtype)
    
    if t_arr.ndim > len(B_shape):
        raise ValueError(
            f"t has {t_arr.ndim} dimensions but the fiber has {len(B_shape)}; "
            "provide a scalar or an array broadcastable to the fiber shape."
        )
    
    # Right-align t_arr to B_shape with leading ones
    pad = len(B_shape) - t_arr.ndim
    t_aligned_shape = (1,) * pad + t_arr.shape
    # Validate broadcastability against B_shape (each dim must be 1 or equal)
    for td, bd in zip(t_aligned_shape, B_shape):
        if td != 1 and td != bd:
            raise ValueError(
                f"t with shape {t_arr.shape} is not broadcastable to fiber shape {B_shape}."
            )
    
    # Finally reshape to (1, *B, 1) so it broadcasts over the de Casteljau level and value dims
    t_b = t_arr.reshape((1,) + t_aligned_shape + (1,))
    omt_b = 1.0 - t_b
    
    cur = A
    for _ in range(m - 1):
        cur = omt_b * cur[:-1] + t_b * cur[1:]
    
    out = np.moveaxis(cur, 0, axis)  # shape (..., 1, ..., N)
    return out if keepdims else np.squeeze(out, axis=axis)


def de_casteljau_restrict_multi_nd(
        control_grid: np.ndarray,
        axes: Sequence[int],
        t: Sequence,
        keepdims: bool = False
) -> np.ndarray:
    """
    Restrict (fix) multiple parameters of an ND tensor-product Bernstein control grid
    via de Casteljau, collapsing each axis in `axes` at the corresponding parameter in `t`.

    Parameters
    ----------
    control_grid : np.ndarray
        Tensor-product Bernstein coefficients with the last axis as value dimension.
        Example shapes:
          - (U, N), (U, V, N), (U, V, P, N), (U, V, P, Q, N)

    axes : Sequence[int]
        Parametric axes (0-based among the parametric axes only) to fix.
        Negative indices are supported (relative to the parametric axes).
        Must have no duplicates.

    t : Sequence
        Parameters for each axis in `axes`. Must be the same length as `axes`.
        Each entry can be a scalar or an array broadcastable to the fiber
        orthogonal to that axis (see single-axis function).

    keepdims : bool, default False
        If True, the restricted axes are kept with length 1 (so shape is preserved
        except for degree reductions to 1). If False, the restricted axes are removed
        (squeezed) from the output, so the number of remaining parametric axes equals
        the number of “free” axes.

    Returns
    -------
    np.ndarray
        The restricted control grid. If all parametric axes are restricted and
        keepdims=False, the result is a single point of shape (N,).

    Raises
    ------
    ValueError
        - If len(axes) != len(t)
        - If there are duplicate axes
        - If any axis is out of range
        - If len(axes) > number of parametric axes (more parameters than axes)
        - If any `t[i]` is not broadcastable for its fiber

    Notes
    -----
    - The operation is independent of the order of `axes`; internally we keep each axis
      during the step (keepdims=True) to avoid index shifts, then optionally squeeze once.
    - Typical Bernstein parameters lie in [0, 1], but values outside are mathematically valid.

    Examples
    --------
    # Bicubic scalar field (U=4, V=4, N=1). Fix u=0.25 only → returns curve in v.
    S = np.random.rand(4, 4, 1)
    Su = de_casteljau_restrict_multi_nd(S, axes=(0,), t=(0.25,))     # shape (4, 1)

    # Fix both u and v → evaluation to a point
    p = de_casteljau_restrict_multi_nd(S, axes=(0,1), t=(0.25, 0.5)) # shape (1,)

    # Trivariate vector field (U=5, V=6, P=4, N=3). Fix u=0.3 and w=0.8 → surface in v.
    T = np.random.rand(5, 6, 4, 3)
    Tw = de_casteljau_restrict_multi_nd(T, axes=(0,2), t=(0.3, 0.8)) # shape (6, 3)
    """
    if control_grid.ndim < 2:
        raise ValueError("control_grid must have at least one parametric axis and a trailing value axis.")
    
    param_ndim = control_grid.ndim - 1
    
    axes = tuple(axes)
    t = tuple(t)
    if len(axes) != len(t):
        raise ValueError("`axes` and `t` must have the same length.")
    if len(axes) > param_ndim:
        raise ValueError(
            f"More parameters than parametric axes: got {len(axes)} parameters, "
            f"but only {param_ndim} parametric axes are available."
        )
    
    # Normalize and validate axes; ensure uniqueness
    def _norm(ax: int) -> int:
        return ax + param_ndim if ax < 0 else ax
    
    #print(axes, t)
    axes_norm = tuple(_norm(ax) for ax in axes)
    if any((ax < 0 or ax >= param_ndim) for ax in axes_norm):
        raise ValueError(f"All axes must be in [0, {param_ndim - 1}] among parametric axes.")
    if len(set(axes_norm)) != len(axes_norm):
        raise ValueError("`axes` must not contain duplicates.")
    
    # Apply restrictions one by one, but keep dims during each step to avoid index shifts
    out = control_grid
    for ax, ti in zip(axes_norm, t):
        out = de_casteljau_restrict_nd(out, axis=ax, t=ti, keepdims=True)
    
    # Optionally remove the restricted axes in one go
    if not keepdims and len(axes_norm) > 0:
        # squeeze expects axes in the full array indexing (param axes + value axis at the end),
        # and our axes_norm already refer to those same positions.
        out = np.squeeze(out, axis=tuple(sorted(axes_norm)))
    
    return out


def bern_roots_1d(bern, eps: float = 1e-3, interval=None) -> BernRootsOutput:
    bern = np.squeeze(bern)[..., np.newaxis]
    if interval is None:
        interval = (0., 1.)
    
    stack = [(bern, tuple(interval))]
    roots = []
    iters = 0
    while stack:
        coeffs, interv = stack.pop(0)
        iters += 1
        
        sign_changes = _count_sign_changes(np.squeeze(coeffs), 0.)
        split_axis = 0
        if sign_changes == 0:
            continue
        elif sign_changes == 1:
            d1 = bernstein_partial_derivative_coeffs(coeffs, axis=0)
            
            # d2 = bernstein_partial_derivative_coeffs(d1, axis=0)
            def fun(x):
                
                return bernstein_eval_1d(coeffs[..., 0], x)
            
            def fprime(x):
                
                return bernstein_eval_1d(d1[..., 0], x)
            
            # def fprime2(x):
            #    return bernstein_eval_1d(d2[...,0], x)
            
            # newton(fun,0.5,  fprime=fprime,fprime2=fprime2,full_output=True)
            # res,out=brentq(fun,0.,1.,full_output=True)
            # res,out=newton(fun,0.5,  fprime=fprime,fprime2=fprime2,full_output=True)
            out = cnewton.newton(fun, fprime, 0.5, eps, 25)
            
            # if out.converged :
            #   res = out.root
            if out is not None:
                res = out
                fx = fun(res)
                if (not np.any((res == 0 or res < 0 or res > 1 or res == 1))) and abs(fx) < eps:
                    roots.append((float(map_local_to_global(res, interv[0], interv[1])), fx))
                    continue
        
        low, upp = interv
        mid = low + (upp - low) / 2
        coeffs_a, coeffs_b = de_casteljau_split_nd(coeffs, split_axis, 0.5)
        stack.append((coeffs_a, (low, mid)))
        stack.append((coeffs_b, (mid, upp)))
    
    if len(roots) == 0:
        return BernRootsOutput(np.array([]), np.array([]), iters)
    roots, errs = zip(*sorted(roots, key=lambda x: x[0]))
    
    return BernRootsOutput(np.array(roots), np.array(errs), iters)


def bern_roots_2d(bern,eps:float=1e-3,interval=None)->BernRootsOutput:
    sg = np.zeros_like(bern, dtype=int)
    sg[bern > eps] = 1
    sg[bern < -eps] = -1
    def _gen_cpts_to_display(scalar_net):
        
        Pts = np.zeros((*scalar_net.shape, scalar_net.ndim + 1))
        for i in range(scalar_net.shape[0]):
            for j in range(scalar_net.shape[0]):
                Pts[i, j, 2] = scalar_net[i, j]
                Pts[i, j, 0] = gril[0][ i]
                Pts[i, j, 1] = gril[1][ j]
        
        return Pts
    sign_change_edges_nd(bern, return_linear=True)
    src, dst = sign_change_edges_nd(bern, return_linear=True)
    from collections import defaultdict
    
    uu = defaultdict(dict)
    fg = bern.flat
    for s, d in list(zip(src, dst)):
        uu[s.item()][d.item()] = fg[d]
        uu[d.item()][s.item()] = fg[s]
        uu[d.item()][-1] = fg[d]
        uu[s.item()][-1] = fg[s]
    
    ndims = bern.ndim
    isolines = []
    axes_all = np.arange(ndims, dtype=int)
    
    gril = [bern_greville_abscissae(bern.shape[d]) for d in range(len(bern.shape))]
    nn = bern_to_nurbs_bezier(_gen_cpts_to_display(bern), rational=False)
    mabyroots = []
    for k, v in uu.items():
        k_multiindex = np.unravel_index(k, bern.shape)
        coord_k = np.array(tuple(gril[_d][_j] for _d, _j in enumerate(k_multiindex)))
        
        for j, cf in v.items():
            if j != -1:
                j_multiindex = np.unravel_index(j, bern.shape)
                coord_j = np.array(tuple(gril[_d][_j] for _d, _j in enumerate(j_multiindex)))
                mabyroots.append((zero_crossing_nd(coord_k, coord_j, v[-1], v[j])
                                      , (coord_k, coord_j)))
    
    for candidate, ends in mabyroots:
        candidate = np.array(candidate)
        
        for ax in range(ndims):
            cl = candidate.tolist()
            
            l = axes_all.tolist()
            del l[ax]
            del cl[ax]
            
 

            iso = np.squeeze(de_casteljau_restrict_multi_nd(nn.control_points, l, cl)
                             
                             )
            
            isolines.append(iso
                            )
    
    pts = []
    for i in isolines:
        
        roots = bern_roots_1d(i[..., -1][..., None], 1e-6)
        
        try:
            
            pts.extend([bezier_curve_derivatives_compact(i, rr, 0) for rr in roots.roots])
        except Exception as err:
            print(err)
            print('d', roots.roots)
    return pts,nn

if __name__ =="__main__":
    
    import numpy as np
    
    cf = np.array([[[0.3, 0., 153.35865109],
                    [0.3, 0.16666667, 81.19076014],
                    [0.3, 0.33333333, 46.53978987],
                    [0.3, 0.5, 5.34939517],
                    [0.3, 0.66666667, 6.92238807],
                    [0.3, 0.83333333, -14.8134324],
                    [0.3, 1., 11.52518909]]])
    
    cf2 = np.array([[-50.567631429638432, 0.0, -20.249229555071079], [-45.278590167481184, 0.0, 63.840434704091194],
                    [-7.1806054859095099, 0.0, -78.298949251270301], [-7.8798106316942835, 0.0, 26.120536947668867],
                    [67.639525668587439, 0.0, 33.322697665659618], [67.639525668587439, 0.0, -16.424785571413096]]
                   
                   )
    
   
    
    import time
    
    s = time.perf_counter()
    eps = 1e-8
    res2 = bern_roots_1d(cf2[..., -1],eps=eps)
    print(time.perf_counter() - s)
    assert res2.roots.shape[0]==4 and np.all(res2.errors<eps),res2
   
    
    print(res2)
    print()
    s = time.perf_counter()
    res1 = bern_roots_1d(bernstein_partial_derivative_coeffs(np.squeeze(cf[..., -1])[..., None], 0),eps=eps)
    print(time.perf_counter() - s)
    assert  res1.roots.shape[0]==1 and np.all(res1.errors<eps),res1
   
    print(res1)
    #grid = np.random.uniform(-5., 5., (7, 7))
    #print(bern_roots_2d(grid,eps=eps))


# ---------- Boundary extraction for Bernstein patches ----------

def bernstein_boundary_nd(
        control_grid: np.ndarray,
        axis: int,
        side: int
) -> np.ndarray:
    """
    Extract a single boundary isocline from an ND Bernstein control grid.

    For a tensor-product Bernstein patch, fixing a parameter at 0 or 1 yields
    a lower-dimensional patch whose control points are simply the first or
    last slice along that parametric axis.

    Parameters
    ----------
    control_grid : np.ndarray
        Tensor-product Bernstein coefficients with the last axis as value dimension.
        Shape: (n0+1, n1+1, ..., n_{D-1}+1, N)
        where D is the number of parametric axes and N is the value dimension.
    axis : int
        Parametric axis (0-indexed among parametric axes only) along which to
        extract the boundary. Negative indices are supported.
    side : int
        Which boundary to extract:
        - 0: boundary at parameter = 0 (first slice)
        - 1: boundary at parameter = 1 (last slice)

    Returns
    -------
    np.ndarray
        The boundary control grid with shape equal to `control_grid.shape` but
        with the specified axis removed. For a surface (D=2), this returns a
        curve (D=1). For a curve (D=1), this returns a point (D=0, i.e., shape (N,)).

    Examples
    --------
    # Bicubic surface (U=4, V=4, dim=3)
    S = np.random.rand(4, 4, 3)
    u0_boundary = bernstein_boundary_nd(S, axis=0, side=0)  # shape (4, 3), curve at u=0
    u1_boundary = bernstein_boundary_nd(S, axis=0, side=1)  # shape (4, 3), curve at u=1
    v0_boundary = bernstein_boundary_nd(S, axis=1, side=0)  # shape (4, 3), curve at v=0
    v1_boundary = bernstein_boundary_nd(S, axis=1, side=1)  # shape (4, 3), curve at v=1
    """
    if control_grid.ndim < 2:
        raise ValueError(
            "control_grid must have at least one parametric axis and a trailing value axis."
        )

    param_ndim = control_grid.ndim - 1  # exclude trailing value dimension

    # Normalize axis
    if axis < 0:
        axis += param_ndim
    if not (0 <= axis < param_ndim):
        raise ValueError(
            f"'axis' must be in [0, {param_ndim - 1}] for the parametric axes."
        )

    if side not in (0, 1):
        raise ValueError("'side' must be 0 (parameter=0) or 1 (parameter=1).")

    # Extract boundary: first slice (side=0) or last slice (side=1)
    idx = 0 if side == 0 else -1

    # Build a slicing tuple that selects all along other axes
    slices = [slice(None)] * control_grid.ndim
    slices[axis] = idx

    return control_grid[tuple(slices)]


def bernstein_boundaries_2d(
        control_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract all four boundary isoclines from a 2D Bernstein surface.

    For a tensor-product Bernstein surface with control grid shape (nu+1, nv+1, dim),
    the boundaries are the four edge curves:
    - u=0: first row of control points
    - u=1: last row of control points
    - v=0: first column of control points
    - v=1: last column of control points

    Parameters
    ----------
    control_grid : np.ndarray
        Shape (nu+1, nv+1, dim) where dim is the spatial dimension (e.g., 2 or 3).

    Returns
    -------
    u0, u1, v0, v1 : tuple of np.ndarray
        The four boundary curves, each as Bernstein control points:
        - u0: shape (nv+1, dim), isocurve at u=0
        - u1: shape (nv+1, dim), isocurve at u=1
        - v0: shape (nu+1, dim), isocurve at v=0
        - v1: shape (nu+1, dim), isocurve at v=1

    Examples
    --------
    >>> S = np.array([
    ...     [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
    ...     [[0, 1, 0], [1, 1, 1], [2, 1, 0]],
    ...     [[0, 2, 0], [1, 2, 0], [2, 2, 0]],
    ... ], dtype=float)  # shape (3, 3, 3) - biquadratic surface
    >>> u0, u1, v0, v1 = bernstein_boundaries_2d(S)
    >>> u0.shape  # curve at u=0, parameterized by v
    (3, 3)
    >>> v0.shape  # curve at v=0, parameterized by u
    (3, 3)
    """
    if control_grid.ndim != 3:
        raise ValueError(
            f"Expected 3D array (nu+1, nv+1, dim), got shape {control_grid.shape}."
        )

    u0 = bernstein_boundary_nd(control_grid, axis=0, side=0)  # control_grid[0, :, :]
    u1 = bernstein_boundary_nd(control_grid, axis=0, side=1)  # control_grid[-1, :, :]
    v0 = bernstein_boundary_nd(control_grid, axis=1, side=0)  # control_grid[:, 0, :]
    v1 = bernstein_boundary_nd(control_grid, axis=1, side=1)  # control_grid[:, -1, :]

    return u0, u1, v0, v1


def bernstein_all_boundaries_nd(
        control_grid: np.ndarray
) -> list[Tuple[int, int, np.ndarray]]:
    """
    Extract all boundary faces from an ND Bernstein control grid.

    For a D-dimensional parameter domain, there are 2*D boundary faces
    (two per parametric axis: one at parameter=0, one at parameter=1).
    Each boundary face is a (D-1)-dimensional Bernstein patch.

    Parameters
    ----------
    control_grid : np.ndarray
        Tensor-product Bernstein coefficients with the last axis as value dimension.
        Shape: (n0+1, n1+1, ..., n_{D-1}+1, N)

    Returns
    -------
    boundaries : list of (axis, side, boundary_grid)
        A list of tuples where:
        - axis: int, the parametric axis (0 to D-1)
        - side: int, 0 for parameter=0, 1 for parameter=1
        - boundary_grid: np.ndarray, the (D-1)-dimensional boundary patch

    Examples
    --------
    # Bicubic surface
    >>> S = np.random.rand(4, 4, 3)
    >>> boundaries = bernstein_all_boundaries_nd(S)
    >>> len(boundaries)
    4
    >>> for axis, side, bnd in boundaries:
    ...     print(f"axis={axis}, side={side}, shape={bnd.shape}")
    axis=0, side=0, shape=(4, 3)
    axis=0, side=1, shape=(4, 3)
    axis=1, side=0, shape=(4, 3)
    axis=1, side=1, shape=(4, 3)

    # Trivariate Bernstein (3 parametric axes)
    >>> T = np.random.rand(3, 4, 5, 2)
    >>> boundaries = bernstein_all_boundaries_nd(T)
    >>> len(boundaries)
    6
    """
    if control_grid.ndim < 2:
        raise ValueError(
            "control_grid must have at least one parametric axis and a trailing value axis."
        )

    param_ndim = control_grid.ndim - 1
    boundaries = []

    for axis in range(param_ndim):
        for side in (0, 1):
            bnd = bernstein_boundary_nd(control_grid, axis=axis, side=side)
            boundaries.append((axis, side, bnd))

    return boundaries