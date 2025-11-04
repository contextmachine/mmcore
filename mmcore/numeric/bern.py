import math

import numpy as np
from mmcore.numeric.binom import binomial_coefficient_py
from mmcore.numeric.newton import cnewton

from mmcore.numeric.fdm import newtons_method, bounded_newtons_method
from mmcore.numeric.newton.cnewton import newton_method2
from numpy.typing import NDArray
from typing import NamedTuple

from scipy import optimize

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
    C2m = np.array([binomial_coefficient_py(2*m, r) for r in range(2*m+1)], dtype=np.float64)
    Ci  = np.array([binomial_coefficient_py(m, i)    for i in range(m+1)],  dtype=np.float64)
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

def _lower_transform(n: int, t: float, dtype=float) -> np.ndarray:
    """
    Lower-triangular (n+1)x(n+1) matrix L(t) for the 'left' de Casteljau subdivision boundary.
    Row i (0..n): L[i, j] = C(i, j) * (1 - t)^(i - j) * t^j for j<=i, else 0.
    """
    L = np.zeros((n + 1, n + 1), dtype=dtype)
    one_minus_t = 1.0 - t
    t = t
    for i in range(n + 1):
        # Row i
        pow1 = one_minus_t ** i
        powt = 1.0
        for j in range(i + 1):
            L[i, j] = binomial_coefficient_py(i, j) * pow1 * powt
            # Update factors
            if j < i:
                pow1 = pow1 / (one_minus_t if one_minus_t != 0 else 1.0)
                powt *= t
    return L

def _right_transform(n: int, a: float, dtype=float) -> np.ndarray:
    """
    Upper-triangular (n+1)x(n+1) matrix R(a) producing the 'right' polygon at a,
    with the first control equal to p(a) and the last unchanged (c_n).
    R(a) = J * L(1 - a) * J.
    """
    if n == 0:
        return np.array([[1.0]], dtype=dtype)
    J = np.flipud(np.eye(n + 1, dtype=dtype))
    return J @ _lower_transform(n, 1.0 - a, dtype=dtype) @ J

def _trim_transform(n: int, a: float, b: float, dtype=float) -> np.ndarray:
    """
    (n+1)x(n+1) matrix mapping coefficients on [0,1] to coefficients on [a,b],
    reparameterized back to [0,1].
    """
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("All ranges must be inside [0, 1].")
    if b < a:
        raise ValueError("Each range must satisfy a <= b.")
    if n == 0:
        return np.array([[1.0]], dtype=dtype)
    t2 = 0.0 if (1.0 - a) == 0.0 else (b - a) / (1.0 - a)
    return _lower_transform(n, t2, dtype=dtype) @ _right_transform(n, a, dtype=dtype)

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

import numpy as np

import numpy as no
import numpy as np
import numpy as np

# def _pascal_table(n):
#    """Pascal rows 0..n (n+1, n+1) upper-left filled; dtype=int64."""
#    if (not hasattr(_pascal_table, '_PASCAL_TABLE')):
#        _pascal_table._PASCAL_TABLE = np.zeros((n + 1, n + 1), dtype=np.int64)
#
#    elif ((n + 1) > _pascal_table._PASCAL_TABLE.shape[0]):
#
#        _new_sz = _pascal_table._PASCAL_TABLE.shape[0] + ((n + 1) - _pascal_table._PASCAL_TABLE.shape[0])
#        _pascal_table._PASCAL_TABLE.resize((_new_sz, _new_sz), refcheck=False)
#
#
#    else:
#        return _pascal_table._PASCAL_TABLE[:(n + 1), :(n + 1)]
#    P = _pascal_table._PASCAL_TABLE[:(n + 1), :(n + 1)]
#    P[:, 0] = 1
#    for i in range(1, n + 1):
#        P[i, 1:i] = P[i - 1, 1:i] + P[i - 1, :i - 1]
#        P[i, i] = 1
#    return P
#
#
# _ = _pascal_table(100)

import numpy as np

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
        raise ValueError("control_grid must have at least one parametric axis plus a trailing value axis.")
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
def bern_roots_1d(grid,eps:float=1e-3,interval=None)->BernRootsOutput:
    grid=np.squeeze(grid)[...,np.newaxis]
    if interval is None:
        interval=(0.,1.)
    
    stack=[(grid,  tuple(interval))]
    roots=[]
    iters=0
    while stack:
        coeffs, interv=stack.pop(0)
        iters+=1

        sign_changes = _count_sign_changes(np.squeeze(coeffs), 0.)
        split_axis=0
        if  sign_changes==0:
            continue
        elif sign_changes==1:
                d1=bernstein_partial_derivative_coeffs(coeffs,axis=0)
                #d2 = bernstein_partial_derivative_coeffs(d1, axis=0)
                def fun(x):
                    
                    return  bernstein_eval_1d(coeffs[...,0],x)
                
                def fprime(x):
                    
                    return bernstein_eval_1d(d1[...,0], x)
                
                #def fprime2(x):
                #    return bernstein_eval_1d(d2[...,0], x)
           
                
                #newton(fun,0.5,  fprime=fprime,fprime2=fprime2,full_output=True)
                #res,out=brentq(fun,0.,1.,full_output=True)
                #res,out=newton(fun,0.5,  fprime=fprime,fprime2=fprime2,full_output=True)
                out=cnewton.newton(fun,fprime, 0.5,eps,25)
                
          
           
                #if out.converged :
                #   res = out.root
                if out is not None:
                    res=out
                    fx = fun(res)
                    if (not np.any((res==0 or res<0 or res>1 or res==1))) and abs(   fx)<eps :
                        roots.append((float(map_local_to_global(res, interv[0],interv[1])),fx))
                        continue

        low ,upp=interv
        mid=low+(upp-low)/2
        coeffs_a,coeffs_b=de_casteljau_split_nd(coeffs, split_axis,0.5)
        stack.append((coeffs_a, (low,mid)))
        stack.append((coeffs_b, (mid, upp)))
 
    if len(roots)==0:
        return BernRootsOutput(np.array(0), np.array(0), iters)
    roots,errs=zip(*sorted(roots, key=lambda x: x[0]))
    
    return BernRootsOutput(np.array(roots),np.array(errs),iters)
    
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
    np.moveaxis()