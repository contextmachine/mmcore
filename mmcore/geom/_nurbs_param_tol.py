import numpy as np
from mmcore.geom._nurbs_eval import NURBSCurveTuple, evaluate_nurbs_curve, NURBSSurfaceTuple

import numpy as np

def _bezier_curve_param_tol_conservative(P, w, tol, u0=0.0, u1=1.0):
    """
    Computes a conservative parametric tolerance for a (possibly rational) Bézier curve segment.

    This is the Bézier-specialized analogue of the OCC-style bound used for rational B-spline
    curves, but restricted to a single Bézier span. Conceptually, a Bézier segment is a clamped
    NURBS with one knot interval, so the "knot-block" length is just (u1 - u0) and the effective
    neighborhood is all control points.

    Parameters
    ----------
    P : numpy.ndarray
        Control points array of shape (degree+1, dim).
    w : numpy.ndarray or None
        Weights array of shape (degree+1,). If None, the curve is treated as non-rational (all weights = 1).
        All weights must be strictly positive for this bound.
    tol : float
        Geometric tolerance to be converted into a parametric tolerance.
    u0, u1 : float
        Parameter interval for the Bézier segment. Many implementations use [0, 1] by default.

    Returns
    -------
    float
        Conservative parametric tolerance `tol_u` such that (heuristically/bounded) a parameter
        perturbation of size <= tol_u yields a geometric deviation <= tol.

    Raises
    ------
    ValueError
        If degree is inconsistent with P, if u1 <= u0, or if any weight is <= 0.

    Notes
    -----
    * The bound uses an L1-based variation measure (sum of abs per coordinate). This is conservative
      for Euclidean distance since ||v||_2 <= ||v||_1.
    * For a non-rational Bézier curve (all weights = 1), this reduces to:
        L = (degree / (u1-u0)) * max_i ||P[i] - P[i-1]||_1
        tol_u = tol / max(L, tiny)
    """
    P = np.asarray(P, dtype=float)

    if P.ndim != 2:
        raise ValueError("P must be a 2D array of shape (degree+1, dim).")
    n, dim = P.shape
    p=n-1
    if p < 1:
        raise ValueError("Degree p must be >= 1.")
    du = float(u1 - u0)
    if du <= 0.0:
        raise ValueError("Invalid parameter interval: require u1 > u0.")

    # Handle polynomial vs rational Bézier
    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(f"Expected w.shape[0] == {n}, got {w.shape[0]}.")

    min_w = float(np.min(w))
    if min_w <= 0.0:
        raise ValueError("All weights must be > 0 for this bound.")

    inv_du = 1.0 / du

    # OCC-inspired conservative "local variation" scan over edges (i-1 -> i),
    # specialized to a single Bézier span: neighborhood = all control points.
    Lmax = 0.0
    for i in range(1, n):
        Wi, Wim1 = float(w[i]), float(w[i - 1])
        Pi, Pim1 = P[i], P[i - 1]

        # For Bezier: Pj is the full control net (single span)
        # term[j] = || (Pj-Pi)*Wi - (Pj-Pim1)*Wim1 ||_1
        term = np.abs((P - Pi) * Wi - (P - Pim1) * Wim1).sum(axis=1)
        value = float(term.max()) * inv_du
        if value > Lmax:
            Lmax = value

    # Degree scaling + denominator lower bound, mirroring the OCC structure
    L = (p * Lmax) / min_w

    if L == 0.0:
        return float(tol)
    RealSmall = np.finfo(float).tiny
    tol_u = min(float(tol), float(tol) / max(L, RealSmall))
    return tol_u
import numpy as np

import numpy as np
from math import comb


def _binom_row(n: int) -> np.ndarray:
    """[C(n,0), ..., C(n,n)] as float64"""
    return np.array([comb(n, k) for k in range(n + 1)], dtype=np.float64)

from mmcore.numeric.bern import bernstein_product_conv



import numpy as np

from mmcore.numeric.bern import bernstein_partial_derivative_coeffs
def _bezier_curve_param_tol_optimistic( P,w,tol, interval=(0.,1.))->float:
    #np.abs((P[:,1:,:]-P[:,:-1,:])*w[:,:-1,None]-  (P[:,1:,:]-P[:,:-1,:]-1)*w[:,:-1,None]-1)

    #np.abs((P[:,1:,:]-P[:,:-1,:])*w[:,:-1,None]-  (P[:,1:,:]-P[:,:-1,:]-1)*w[:,:-1,None]-1)/
    P0=P[:-1,:]
    P1=P[1:,:]

    W0 = w[:-1,None]
    W1 = w[1:,None]

    # ``P`` is already Euclidean (``from_homogeneous_1d`` divided the
    # homogeneous numerators by ``w``).  The quotient-rule numerator for
    # an edge is therefore
    #
    #   (P1*w1)*w0 - (P0*w0)*w1 = w0*w1*(P1-P0),
    #
    # not ``P1*w0 - P0*w1``.  The old expression mixed Euclidean controls
    # with homogeneous algebra, was not translation invariant, and gave a
    # non-zero speed to a geometrically constant rational curve whenever
    # adjacent weights differed (SSX case 14's collapsed cone-apex edge).
    s=(P1 - P0) * (W1 / W0)

    dt=np.linalg.norm(s,axis=-1).max()
    ddt=interval[1]-interval[0]


    # `< _TINY`, not `== 0.0` (review 2026-07-12 §10): the same collapsed-
    # speed guard as `_nurbs_curve_param_tol_optimistic`/the conservative
    # variants.  With the current squared-sum norm the two are equivalent
    # (sub-1.5e-162 components flush to a norm of exactly 0.0), but a
    # scaled-norm implementation would expose the quotient to denormal
    # denominators; the guard must not depend on that accident.
    if dt < _TINY:
        tol_u=tol

    else:


        tol_u = tol * (ddt/dt)



    return tol_u

def _bezier_surface_param_tol_optimistic(P,w,tol, interval_u=(0.,1.), interval_v=(0.,1.))->tuple[float,float]:
    #np.abs((P[:,1:,:]-P[:,:-1,:])*w[:,:-1,None]-  (P[:,1:,:]-P[:,:-1,:]-1)*w[:,:-1,None]-1)

    #np.abs((P[:,1:,:]-P[:,:-1,:])*w[:,:-1,None]-  (P[:,1:,:]-P[:,:-1,:]-1)*w[:,:-1,None]-1)/
    P0=P[:,:-1,:]
    P1=P[:,1:,:]

    W0 = w[:,:-1,None]
    W1 = w[:,1:,None]

    # P is Euclidean here; retain the weight ratio but use control-point
    # differences so the derivative estimate is translation invariant.
    s=(P1 - P0) * (W1 / W0)

    dv=np.linalg.norm(s,axis=-1).max()

    P0=P[:-1,:,:]
    P1=P[1:,:,:]

    W0 = w[:-1,:,None]
    W1 = w[1:,:,None]
    s=(P1 - P0) * (W1 / W0)

    du=np.linalg.norm(s,axis=-1).max()
    ddu=interval_u[1]-interval_u[0]
    ddv=interval_v[1]-interval_v[0]

    # `< _TINY`, not `== 0.0` — see the curve variant's comment.
    if du < _TINY:
        tol_u=tol

    else:


        tol_u = tol * (ddu/du)
    if dv < _TINY:
        tol_v=tol
    else:
        tol_v = tol * (ddv/dv)

    return tol_u,tol_v

def _bezier_surface_param_tol_conservative(P, w, tol,
                                          u0=0.0, u1=1.0,
                                          v0=0.0, v1=1.0):
    """
    Computes conservative parametric tolerances for a *rational* tensor-product Bézier surface patch.

    This is the tensor-product surface analogue of the curve routine shown in the prompt:
    we compute conservative Lipschitz-like upper bounds for the surface partial derivatives
    in u and v, then convert a geometric tolerance `tol` into per-parameter tolerances.

    Parameters
    ----------
    P : numpy.ndarray
        Control points array of shape (p+1, q+1, dim).
    w : numpy.ndarray
        Weights array of shape (p+1, q+1). All weights must be strictly positive.
    p, q : int
        Degrees of the Bézier surface in u and v (p >= 1, q >= 1).
    tol : float
        Geometric tolerance to be converted into parametric tolerances.
    u0, u1 : float
        Parameter interval in u.
    v0, v1 : float
        Parameter interval in v.

    Returns
    -------
    (float, float)
        (tol_u, tol_v) conservative parametric tolerances.
        Heuristically/bounded: moving by |du| <= tol_u with dv=0 gives deviation <= tol,
        and moving by |dv| <= tol_v with du=0 gives deviation <= tol.

    Raises
    ------
    ValueError
        If shapes/degrees are inconsistent, if intervals are invalid, or if any weight <= 0.

    Notes
    -----
    * Uses an L1-based variation measure, conservative for Euclidean distance since
      ||v||_2 <= ||v||_1.
    * For a non-rational patch (all weights = 1), the implied bounds reduce to the familiar
      polynomial Bézier bounds:
        L_u = (p/(u1-u0)) * max_{i,j} ||P[i,j] - P[i-1,j]||_1
        L_v = (q/(v1-v0)) * max_{i,j} ||P[i,j] - P[i,j-1]||_1
      and tol_u = tol/L_u, tol_v = tol/L_v.
    * If you need a *single* scalar UV step bound for simultaneous perturbations,
      a very conservative option is:
        tol_uv = tol / max(L_u + L_v, tiny)
      because ||ΔS|| <= L_u|Δu| + L_v|Δv| <= (L_u+L_v)||[Δu,Δv]||_2.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 3:
        raise ValueError("P must be a 3D array of shape (p+1, q+1, dim).")
    nu, nv, dim = P.shape
    p, q=nu-1,nv-1
    if p < 1 or q < 1:
        raise ValueError("Degrees p and q must both be >= 1.")
    if nu != p + 1 or nv != q + 1:
        raise ValueError(
            f"For a Bézier surface, expected P.shape[:2] == (p+1, q+1) == ({p+1}, {q+1}), "
            f"got ({nu}, {nv})."
        )

    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or w.shape != (nu, nv):
        raise ValueError(f"w must have shape ({nu}, {nv}), got {w.shape}.")

    min_w = float(np.min(w))
    if min_w <= 0.0:
        raise ValueError("All weights must be > 0 for this bound.")

    tol = float(tol)
    if tol < 0.0:
        raise ValueError("tol must be >= 0.")

    du = float(u1 - u0)
    dv = float(v1 - v0)
    if du <= 0.0:
        raise ValueError("Invalid u-interval: require u1 > u0.")
    if dv <= 0.0:
        raise ValueError("Invalid v-interval: require v1 > v0.")

    inv_du = 1.0 / du
    inv_dv = 1.0 / dv

    # Flatten the control net once; each scan uses max over the full neighborhood (single patch).
    Pflat = P.reshape(-1, dim)

    # --- u-direction scan (edges between (i-1,j) and (i,j)) ---

    Lmax_u = 0.0
    for i in range(1, nu):
        for j in range(nv):
            Wi = float(w[i, j])
            Wim1 = float(w[i - 1, j])
            Pi = P[i, j]
            Pim1 = P[i - 1, j]

            # term[k,l] = || (P[k,l]-Pi)*Wi - (P[k,l]-Pim1)*Wim1 ||_1
            term = np.abs((Pflat - Pi) * Wi - (Pflat - Pim1) * Wim1).sum(axis=1)
            value = float(term.max()) * inv_du
            if value > Lmax_u:
                Lmax_u = value

    # Degree scaling + denominator lower bound, mirroring the OCC-style structure
    L_u = (p * Lmax_u) / min_w

    # --- v-direction scan (edges between (i,j-1) and (i,j)) ---
    Lmax_v = 0.0
    for i in range(nu):
        for j in range(1, nv):
            Wj = float(w[i, j])
            Wjm1 = float(w[i, j - 1])
            Pj = P[i, j]
            Pjm1 = P[i, j - 1]

            term = np.abs((Pflat - Pj) * Wj - (Pflat - Pjm1) * Wjm1).sum(axis=1)
            value = float(term.max()) * inv_dv
            if value > Lmax_v:
                Lmax_v = value

    L_v = (q * Lmax_v) / min_w

    RealSmall = np.finfo(float).tiny
    tol_u = tol if L_u == 0.0 else min(tol, tol / max(L_u, RealSmall))
    tol_v = tol if L_v == 0.0 else min(tol, tol / max(L_v, RealSmall))
    return float(tol_u), float(tol_v)
def _nurbs_curve_param_tol_conservative(P, w, U, p, tol):
    """
    Computes a conservative parametric tolerance for a NURBS curve.

    This function estimates the maximum parametric tolerance for a NURBS
    (non-uniform rational B-spline) curve based on its control points, weights,
    knot vector, and degree. It takes into account geometric variations and weights
    to calculate a bound on the required parametric tolerance.

    Parameters:
    P : numpy.ndarray
        A 2D array of shape (n, dim) representing the control points of the NURBS
        curve. Here, `n` is the number of control points, and `dim` is the
        dimension of the space.
    w : numpy.ndarray
        A 1D array of shape (n,) representing the weights corresponding to each
        control point.
    U : numpy.ndarray
        A 1D array of shape (n + p + 1,) representing the non-decreasing knot
        vector for the NURBS curve.
    p : int
        The degree of the NURBS curve.
    tol : float
        The desired tolerance to be divided by the computed scaling factor.

    Returns:
    float
        The computed conservative parametric tolerance.

    Raises:
    ValueError
        If any of the weights in `w` are not greater than zero.

    Notes:
    This function assumes valid input shapes and values, including the consistency
    of array dimensions and the degree `p` with respect to the provided knot vector.
    It also assumes that the knot vector is non-decreasing. The algorithm computes
    a local maximum over neighborhoods of control points and involves scaling based
    on the largest local variation observed.
    """
    n, dim = P.shape
    assert U.shape[0] == n + p + 1
    min_w = float(np.min(w))
    if min_w <= 0:
        raise ValueError("All original weights must be > 0 for this bound.")
    Lmax = 0.0
    # edges i-1 -> i, with their knot-block [U_i, U_{i+p}]
    for i in range(1, n):
        du = U[i + p] - U[i]
        if du <= 0:
            continue  # degenerate or repeated knots; skip or handle specially
        inv_du = 1.0 / du
        # OCC's neighborhood: [i-(p+1), i+(2p+1))
        lower = max(0, i - (p + 1))
        upper = min(n, i + (2 * p + 1))
        Wi, Wim1 = w[i], w[i - 1]
        Pi, Pim1 = P[i], P[i - 1]
        # inner max over j in neighborhood
        # vector form; loop is fine too if you prefer clarity
        Pj = P[lower:upper]                       # (m, dim)
        term = np.abs((Pj - Pi) * Wi - (Pj - Pim1) * Wim1).sum(axis=1)  # (m,)
        value = term.max() * inv_du
        if value > Lmax:
            Lmax = value
    # degree scaling and denominator lower bound, exactly like OCC
    L = (p * Lmax) / min_w
    # final parametric tolerance
    RealSmall = np.finfo(float).tiny  # mirror OCC's "RealSmall()" idea
    tol_u = tol / max(L, RealSmall)
    return tol_u
def _nurbs_surface_param_tol_conservative(P, w, U, V, p, q, tol)->tuple[float,float]:
    """
    Computes conservative parametric tolerances for a NURBS surface.

    Extends the curve-based conservative parametric tolerance estimation
    to tensor-product NURBS surfaces, producing independent tolerances
    for the u- and v-parametric directions.

    Parameters
    ----------
    P : numpy.ndarray
        Control points, shape (n_u, n_v, dim).
    w : numpy.ndarray
        Weights, shape (n_u, n_v).  All must be > 0.
    U : numpy.ndarray
        Knot vector in the u-direction, shape (n_u + p + 1,).
    V : numpy.ndarray
        Knot vector in the v-direction, shape (n_v + q + 1,).
    p : int
        Degree in the u-direction.
    q : int
        Degree in the v-direction.
    tol : float
        Desired spatial tolerance.

    Returns
    -------
    tol_u : float
        Conservative parametric tolerance in u.
    tol_v : float
        Conservative parametric tolerance in v.

    Raises
    ------
    ValueError
        If any weight is not strictly positive.
    """
    n_u, n_v, dim = P.shape
    assert U.shape[0] == n_u + p + 1
    assert V.shape[0] == n_v + q + 1

    min_w = float(np.min(w))
    if min_w <= 0:
        raise ValueError("All weights must be > 0 for this bound.")

    RealSmall = np.finfo(float).tiny

    # ------------------------------------------------------------------
    # u-direction: edges (i-1, j) -> (i, j), knot block [U_i, U_{i+p}]
    # ------------------------------------------------------------------
    Lmax_u = 0.0
    for i in range(1, n_u):
        du = U[i + p] - U[i]
        if du <= 0:
            continue
        inv_du = 1.0 / du

        u_lo = max(0, i - (p + 1))
        u_hi = min(n_u, i + (2 * p + 1))

        for j in range(n_v):
            v_lo = max(0, j - (q + 1))
            v_hi = min(n_v, j + (2 * q + 1))

            Wi  = w[i, j]
            Wim = w[i - 1, j]
            Pi  = P[i, j]          # (dim,)
            Pim = P[i - 1, j]      # (dim,)

            # neighbourhood block
            Pkls = P[u_lo:u_hi, v_lo:v_hi]              # (mu, mv, dim)
            Pkls_flat = Pkls.reshape(-1, dim)            # (mu*mv, dim)

            term = np.abs(
                (Pkls_flat - Pi) * Wi - (Pkls_flat - Pim) * Wim
            ).sum(axis=1)

            value = term.max() * inv_du
            if value > Lmax_u:
                Lmax_u = value

    # ------------------------------------------------------------------
    # v-direction: edges (i, j-1) -> (i, j), knot block [V_j, V_{j+q}]
    # ------------------------------------------------------------------
    Lmax_v = 0.0
    for j in range(1, n_v):
        dv = V[j + q] - V[j]
        if dv <= 0:
            continue
        inv_dv = 1.0 / dv

        v_lo = max(0, j - (q + 1))
        v_hi = min(n_v, j + (2 * q + 1))

        for i in range(n_u):
            u_lo = max(0, i - (p + 1))
            u_hi = min(n_u, i + (2 * p + 1))

            Wj  = w[i, j]
            Wjm = w[i, j - 1]
            Pj  = P[i, j]          # (dim,)
            Pjm = P[i, j - 1]      # (dim,)

            Pkls = P[u_lo:u_hi, v_lo:v_hi]
            Pkls_flat = Pkls.reshape(-1, dim)

            term = np.abs(
                (Pkls_flat - Pj) * Wj - (Pkls_flat - Pjm) * Wjm
            ).sum(axis=1)

            value = term.max() * inv_dv
            if value > Lmax_v:
                Lmax_v = value

    # degree scaling + weight denominator (mirrors OCC)
    L_u = (p * Lmax_u) / min_w
    L_v = (q * Lmax_v) / min_w

    tol_u = tol / max(L_u, RealSmall)
    tol_v = tol / max(L_v, RealSmall)

    return tol_u, tol_v
from ._nurbs_ders import derivative_nurbs
_TINY=np.finfo(float).tiny
def _nurbs_curve_param_tol_optimistic(curve: NURBSCurveTuple, tol: float, der:NURBSCurveTuple=None) -> float:
    if np.allclose(curve.control_points,0):
        return tol
    if der is None:
        der=derivative_nurbs(curve)
    u0, u1 = curve.interval()
    du = (u1 - u0)
    cpts=np.abs(der.control_points)
    res=np.linalg.norm(cpts, axis=1).max()
    if res<_TINY:
        return tol
    
    tol_u = tol * (du / res)

    return tol_u


from numpy.typing import NDArray
from mmcore.geom._nurbs_eval import from_homogeneous_1d,from_homogeneous_2d,to_homogeneous_2d
def bez_curve_param_tolerance(bez: NDArray, tol: float, rational:bool=False, interval=(0.,1.)) -> float:

    if interval is None:
        interval=(0.,1.)
    if rational:

        P,w=from_homogeneous_1d(bez)
    else:

        P,w=bez,np.ones_like(bez[...,0])
    if rational and not np.all(w == w.flat[0]):
        return _bezier_curve_param_tol_conservative(P,w,tol=tol,u0=interval[0],u1=interval[1])






    return _bezier_curve_param_tol_optimistic(
        P, np.ones_like(w), tol=tol, interval=interval)

def bez_surface_param_tolerance(bez: NDArray, tol: float, rational:bool=False, interval_u=(0.,1.),interval_v=(0.,1.)) -> tuple[float,float]:




    if interval_u is None:
        interval_u=(0.,1.)
    if interval_v is None:
        interval_v=(0.,1.)
    if rational:
        P,w=from_homogeneous_2d(bez)
    else:
        P,w=bez,np.ones_like(bez[...,0])
    if rational and not np.all(w == w.flat[0]):
        return _bezier_surface_param_tol_conservative(P,w,tol=tol,u0=interval_u[0],u1=interval_u[1],v0=interval_v[0],v1=interval_v[1])


    else:




        return _bezier_surface_param_tol_optimistic(
            P, np.ones_like(w), tol, interval_u, interval_v)
def nurbs_curve_param_tolerance(curve: NURBSCurveTuple, tol: float, der:NURBSCurveTuple=None) -> float:
    if np.any(curve.weights<0):
        return _nurbs_curve_param_tol_conservative(curve.control_points, curve.weights, curve.knot, curve.order - 1, tol)
    else:
        return _nurbs_curve_param_tol_optimistic(curve,tol,der)

def nurbs_surface_param_tolerance(curve: NURBSSurfaceTuple, tol: float,*args,**kwargs) -> tuple[float,float]:
    return _nurbs_surface_param_tol_conservative(curve.control_points, curve.weights, curve.knot_u,curve.knot_v, curve.order_u - 1, curve.order_v - 1, tol)


if __name__=="__main__":
    import tqdm
    from mmcore.geom._nurbs_knots import generate_knots
    from itertools import pairwise
    
    color_interpolation = NURBSCurveTuple(3, np.array([1., 1., 1., 0., 0., 0.
                                                       ]), np.array(
        [[27 / 255, 222 / 255, 95 / 255], [222 / 255, 219 / 255, 27 / 255], [222 / 255, 27 / 255, 79 / 255]]),
                                          np.array([1., 1., 1.]))
    
    
    def rgb_to_hex(color: np.ndarray) -> str:
        """
        Convert an RGB color given as a NumPy array of floats in [0, 1]
        to a hex string in the format '#rrggbb'.

        Parameters
        ----------
        color : np.ndarray
            A 1D array of length 3 with float values in the range [0.0, 1.0].

        Returns
        -------
        str
            A hex color string, e.g. '#326fa8'.
        """
        # Ensure input is the right shape
        if color.shape != (3,):
            raise ValueError(f"Expected color array of shape (3,), got {color.shape}")
        
        # Clip values to [0,1], scale to [0,255], and convert to integers
        rgb_int = np.clip(color, 0.0, 1.0) * 255
        r, g, b = rgb_int.astype(int)
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    
    def test_approach(curve: NURBSCurveTuple, tol=1e-3):
    
        u0, u1 = curve.interval()
        du = (u1 - u0)
        tol_u = nurbs_curve_param_tolerance(curve, tol)
        steps, _ = divmod(du, tol_u)
        
        steps = int(steps)
        steps_sizes = np.ones(steps)
        steps_sizes[:] = tol_u
        u0_vals = np.clip(np.cumsum(steps_sizes), u0, u1)
        
        u1_vals = np.clip(u0_vals + tol_u, u0, u1)
        mask = ~np.isclose(u1_vals - u0_vals, 0)
        u0_vals = u0_vals[mask]
        u1_vals = u1_vals[mask]
        red = tol
        green = tol / 2
        color_interpolation = NURBSCurveTuple(3, np.array([green, green, green, red, red, red
                                                           ]), np.array(
            [[27 / 255, 222 / 255, 95 / 255], [222 / 255, 219 / 255, 27 / 255], [222 / 255, 27 / 255, 79 / 255]]),
                                              np.array([1., 1., 1.]))
        
        prec = int(np.abs(np.log10(tol_u)).item()) + 2
        
        print(u0_vals + tol_u)
        pb = tqdm.tqdm(zip(u0_vals, u1_vals), total=steps, dynamic_ncols=True,
                       colour=rgb_to_hex(evaluate_nurbs_curve(color_interpolation, 0)['C']))
        best = float('inf')
        wrong = -float('inf')
        
        def format_num(n):
            return f"{n:{1}.{prec}f}"
        
        def fun(t0, t1):
            nonlocal best, wrong, pb, prec
            val = float(np.linalg.norm(evaluate_nurbs_curve(curve, t0)['C'] - evaluate_nurbs_curve(curve, t1)['C']))
            
            col = evaluate_nurbs_curve(color_interpolation, val)['C']
            pb.colour = rgb_to_hex(col)
            best = float(min(best, val))
            wrong = float(max(wrong, val))
            
            pb.set_description_str(
                f'({format_num(t0)},{format_num(t1)}; ptol: {tol_u} curr: {format_num(val)}; best: {format_num(best)}; wrong: {format_num(wrong)}')
        
        for interv in pb:
            fun(interv[0], interv[1])
    
    from mmcore.geom._nurbs_construct import circle
    curve3 = circle(10, normal=np.array((1.,1.,0.5))/np.linalg.norm((1.,1.,0.5)))
    curve4 = np.array([[-45.36434109, -7.12015504, 0.],
                       [-25.49612403, 13.94186047, 0.],
                       [-2.13178295, -17.35271318, 0.],
                       [12.02325581, 20.42248062, 0.]])
    tpl = NURBSCurveTuple(4, generate_knots(curve4.shape[0], 3), curve4, np.ones((curve4.shape[0])))

    #test_approach(tpl)
    test_approach(curve3)
