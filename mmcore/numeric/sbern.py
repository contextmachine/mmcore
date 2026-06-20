from __future__ import annotations


from mmcore.geom._nurbs_knots import (
    generate_knots,
    decompose_surface,
    decompose_curve,
    split_curve_multiple,
    remove_knot_curve_max,
    normalize_knots_curve,
    normalize_knots,
)

# ----------------------------------------------------------------------
#  Helpers for the scaled‑Bernstein (SB) representation
# ----------------------------------------------------------------------
import math
from functools import lru_cache

import numpy as np

from mmcore.geom._nurbs_eval import NURBSSurfaceTuple, NURBSCurveTuple, to_homogeneous_2d, from_homogeneous_1d, to_homogeneous_1d, \
    from_homogeneous_2d, evaluate_nurbs_curve

# Relative tolerance for locating which Bézier sub‑patch / segment a parameter
# value falls into.  Scaled by the local knot‑span width so it behaves the same
# for parameter domains of any magnitude — a fixed absolute tolerance raises
# spurious "not inside any sub‑patch" errors for boundary‑hugging curves on
# large domains and is too loose on tiny ones.
_LOCATE_RTOL = 1e-9

# ------------------------------------------------------------------
#  Scaled‑Bernstein helpers
# ------------------------------------------------------------------


@lru_cache(maxsize=None)
def _binom_scale(n: int) -> np.ndarray:
    """Cached row of binomial coefficients ``[C(n,0) … C(n,n)]`` as float.

    Uses ``math.comb`` (exact, unbounded) rather than the fixed-size
    ``binomial_coefficient_py`` table, which is only valid for n <= 30 and
    silently reads out-of-bounds memory above that.  SB composition reaches
    those degrees easily (e.g. a bicubic patch composed with a degree-6 curve
    has degree (3+3)*6 = 36), so the table version produced ``inf`` control
    points with no error.  Returned array is read-only (callers only multiply).
    """
    v = np.fromiter((math.comb(n, i) for i in range(n + 1)), dtype=float, count=n + 1)
    v.setflags(write=False)
    return v


def to_scaled(ctrl: np.ndarray) -> np.ndarray:
    """
    Standard Bernstein control points  ->  scaled‑Bernstein coefficients
    (multiplies by the binomial C(n,i)).
    """
    scale = _binom_scale(len(ctrl) - 1)
    return ctrl * scale[..., None] if ctrl.ndim > 1 else ctrl * scale


def from_scaled(coeff: np.ndarray) -> np.ndarray:
    """
    Scaled‑Bernstein  ->  ordinary Bernstein control points.
    Divides by C(n,i).
    """
    scale = _binom_scale(len(coeff) - 1)
    return coeff / scale[..., None] if coeff.ndim > 1 else coeff / scale


def nurbs_bezier_to_bern(bez:NURBSCurveTuple|NURBSSurfaceTuple, rational:bool=True):
    if bez.control_points.ndim==2:
        if bez.control_points.shape[0]-bez.order!=0:
            raise ValueError(f"input curve is not a Bezier curve. knots={bez.knot}")
        if rational:
            return to_homogeneous_1d(bez.control_points,bez.weights)
        else:
            return bez.control_points

    elif  bez.control_points.ndim==3:
        ku_check=bez.control_points.shape[0]-bez.order_u!=0
        kv_check=bez.control_points.shape[1]-bez.order_v!=0
        if ku_check and kv_check:
            raise ValueError(f"input surface is not a Bezier surface. knots_u={bez.knot_u}, knots_v={bez.knot_v}")
        elif ku_check:
            raise ValueError(f"input surface is not a Bezier surface along u direction. knots_u={bez.knot_u}")
        elif kv_check:
            raise ValueError(f"input surface is not a Bezier surface along v direction. knots_v={bez.knot_v}")
        if rational:
            return to_homogeneous_2d(bez.control_points,bez.weights)
        else:
            return bez.control_points
    raise ValueError(f"input is not a curve or surface. type={type(bez)}")


def bern_to_nurbs_bezier(bern, interval:tuple=None, rational:bool=True):
    
    if bern.ndim==2:
        if rational:

            return NURBSCurveTuple(bern.shape[0], generate_knots(bern.shape[0], bern.shape[0] - 1,interval=interval), *from_homogeneous_1d(bern))
        else:
            return NURBSCurveTuple(bern.shape[0], generate_knots(bern.shape[0], bern.shape[0] - 1,interval=interval), np.copy(bern), np.ones(bern.shape[0],dtype=float) )

    elif bern.ndim==3:
        if  interval is None:
            interval=(None,None)
        if rational:

            return NURBSSurfaceTuple(
                bern.shape[0],  bern.shape[1], generate_knots(bern.shape[0], bern.shape[0] - 1, interval=interval[0]), generate_knots(bern.shape[1], bern.shape[1] - 1, interval=interval[1]), *from_homogeneous_2d(bern)
            )
        else:
            return NURBSSurfaceTuple(
                bern.shape[0],
                bern.shape[1], generate_knots(bern.shape[0], bern.shape[0] - 1, interval=interval[0]), generate_knots(bern.shape[1], bern.shape[1] - 1, interval=interval[1]),
                np.copy(bern),
                np.ones(bern.shape[:-1], dtype=float),
            )
        
    else:
        raise ValueError(f"input is not a curve or surface. {bern}")

def sb_pow(poly_sb: np.ndarray, k: int) -> np.ndarray:
    """
    k‑fold power of a scaled‑Bernstein polynomial via repeated convolution.
    """
    if k == 0:
        return np.array([1.0])
    result = np.array([1.0])
    base   = poly_sb.copy()
    e = k
    while e:
        if e & 1:
            result = np.convolve(result, base)
        if e > 1:
            base = np.convolve(base, base)
        e >>= 1
    return result


def sb_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convolution that keeps us in the SB world (plain np.convolve)."""
    return np.convolve(a, b)


def _all_powers(base: np.ndarray, nmax: int) -> list:
    """Return ``[base**0, base**1, …, base**nmax]`` under 1-D SB convolution.

    When the *whole* list of powers is needed (as every composer does), the
    cumulative recurrence ``p[k] = p[k-1] * base`` costs exactly ``nmax``
    convolutions, versus the per-power binary exponentiation in :func:`sb_pow`
    which recomputes the shared squarings of ``base`` for every exponent.
    """
    powers = [np.array([1.0])]
    for _ in range(nmax):
        powers.append(np.convolve(powers[-1], base))
    return powers


def compose_curve_curve(spatial_ctrl: np.ndarray,
                                 param_ctrl:   np.ndarray,
                                 return_cartesian: bool = False) -> np.ndarray:
    """
    Parameters
    ----------
    spatial_ctrl : (n+1, D+1) ndarray
        Homogeneous control points of the original D‑dimensional rational curve
        ``(x0*w, …, x_{D-1}*w, w)``.  ``D`` may be 2 or 3 (in fact any ``D >= 1``);
        the last column is always the weight.  Degree n.
    param_ctrl   : (p+1, 2) ndarray
        Control points of the 1‑D rational re‑parametrisation curve, columns
        ``(s, w)``: the Cartesian parameter value and its weight.  Degree p.
        Internally this becomes ``sigma = s*w`` (numerator) and ``omega = w``
        (denominator), so the parameter map is ``sigma(t)/omega(t)``.
    return_cartesian : bool, default False
        If True, divides by the weights and returns Cartesian points.
        Otherwise returns homogeneous coordinates.

    Returns
    -------
    ctrl_out : ndarray ((n*p)+1, D+1)  or  (..., D)
        Control points of the composed rational Bézier curve
        C_new(t) = C(sigma(t)/omega(t)).
    """
    # ---------------------------------------------------------------
    # 0.  Validate + degrees / spatial dimension
    # ---------------------------------------------------------------
    if spatial_ctrl.ndim != 2 or spatial_ctrl.shape[1] < 2:
        raise ValueError(f"spatial_ctrl must be (n+1, D+1) with D >= 1; got shape {spatial_ctrl.shape}")
    if param_ctrl.ndim != 2 or param_ctrl.shape[1] != 2:
        raise ValueError(f"param_ctrl must be (p+1, 2) with columns (s, w); got shape {param_ctrl.shape}")
    n = spatial_ctrl.shape[0] - 1
    p = param_ctrl.shape[0]   - 1
    deg = n * p                                # final polynomial degree
    dim1 = spatial_ctrl.shape[1]               # D + 1  (spatial coords + weight)

    # ---------------------------------------------------------------
    # 1.  Build SB polynomials for sigma(t) and omega(t)
    # ---------------------------------------------------------------
    sigma_num = param_ctrl[:, 0] * param_ctrl[:, 1]   # sigma·omega
    omega_den = param_ctrl[:, 1]                      # omega

    sigma_sb = to_scaled(sigma_num)
    omega_sb = to_scaled(omega_den)
    omega_m_sigma_sb = omega_sb - sigma_sb                              # (omega - sigma)

    # Pre‑compute needed powers (full lists 0..n → cumulative convolution)
    pow_sigma         = _all_powers(sigma_sb, n)              # pow_sigma[i] = sigma^i
    pow_omega_m_sigma = _all_powers(omega_m_sigma_sb, n)[::-1]  # [...][i] = (omega-sigma)^{n-i}

    # ---------------------------------------------------------------
    # 2.  Scale spatial control points once to remove binomials (× C(n,i))
    # ---------------------------------------------------------------
    spatial_sb = to_scaled(spatial_ctrl.astype(float))

    # ---------------------------------------------------------------
    # 3.  Assemble numerator polynomial in every homogeneous coordinate.
    #     B_i^n(sigma/omega) ∝ sigma^i (omega - sigma)^{n-i}; multiply that
    #     SB basis by the (scaled) homogeneous control point P_i and sum.
    #     The trailing column carries the weight, so the denominator falls
    #     out of the same accumulation for free.
    # ---------------------------------------------------------------
    num = np.zeros((deg + 1, dim1))
    for i in range(n + 1):
        basis = sb_convolve(pow_omega_m_sigma[i], pow_sigma[i])     # SB coefficients
        num[:len(basis)] += np.outer(basis, spatial_sb[i])

    # ---------------------------------------------------------------
    # 4.  Back to ordinary Bernstein control points
    # ---------------------------------------------------------------
    homog = from_scaled(num)                          # (deg+1, D+1)

    if return_cartesian:
        return homog[:, :-1] / homog[:, -1:]
    return homog
# ----------------------------------------------------------------------
#  Composition (patch ∘ curve) completely in SB form
# ----------------------------------------------------------------------

def compose_patch_curve(patch_ctrl: np.ndarray,
                                 curve_ctrl: np.ndarray,
                                 return_cartesian: bool = False,
                                 curve_ctrl_homogeneous: bool | None = None) -> np.ndarray:
    """
    Exact composition  c(t) = patch(u(t)/w(t), v(t)/w(t))

    Parameters
    ----------
    patch_ctrl : ndarray (m+1, n+1, D+1)
        Homogeneous control lattice of the rational Bézier patch; the last
        column is the weight (D=3 ⇒ columns (x*w, y*w, z*w, w)).  Any D >= 1.
    curve_ctrl : ndarray (p+1, 3)
        Rational Bézier control points in parameter space. Two accepted
        layouts:
          • (u, v, w)              — plain parameters (previous behaviour).
          • (u*w, v*w, w)          — already homogeneous (e.g. output of
            `nurbs_bezier_to_bern` on a rational curve).
    curve_ctrl_homogeneous : bool
        Must be explicitly provided.
        - True  → columns are (u*w, v*w, w)  (homogeneous form)
        - False → columns are (u,  v,  w)    (plain parameters)
    return_cartesian : bool, default False
        If True, divides by the weights and returns Cartesian control
        points (xyz).  Otherwise returns homogeneous (xw,yw,zw,w).

    Returns
    -------
    ctrl_out : ndarray ((m+n)*p + 1, D+1)  or  (..., D)
        Control points of the composed rational Bézier curve.
    """
    # ------------------------------------------------------------------
    # 0.  Validate + sizes and degrees
    # ------------------------------------------------------------------
    if patch_ctrl.ndim != 3 or patch_ctrl.shape[2] < 2:
        raise ValueError(f"patch_ctrl must be (m+1, n+1, D+1) with D >= 1; got shape {patch_ctrl.shape}")
    if curve_ctrl.ndim != 2 or curve_ctrl.shape[1] != 3:
        raise ValueError(f"curve_ctrl must be (p+1, 3) with columns (u, v, w); got shape {curve_ctrl.shape}")
    m, n = patch_ctrl.shape[0] - 1, patch_ctrl.shape[1] - 1
    p    = curve_ctrl.shape[0]   - 1
    deg  = (m + n) * p                          # (total) polynomial degree
    dim1 = patch_ctrl.shape[2]                  # D + 1  (spatial coords + weight)

    # ------------------------------------------------------------------
    # 1.  Parameter curve  —  build SB polynomials U(t), V(t), W(t)
    #     We first convert (u, v, w) ⇒ homogeneous (u·w, v·w, w)
    # ------------------------------------------------------------------
    # Decide whether incoming parameter curve is already homogeneous
    if curve_ctrl_homogeneous is None:
        raise ValueError(
            "curve_ctrl_homogeneous must be explicitly set: "
            "True if curve_ctrl columns are (u*w, v*w, w); False if they are (u, v, w)."
        )

    if curve_ctrl_homogeneous:
        # Already (u*w, v*w, w)
        uw = curve_ctrl[:, 0]
        vw = curve_ctrl[:, 1]
        ww = curve_ctrl[:, 2]
    else:
        # Plain (u, v, w)
        uw = curve_ctrl[:, 0] * curve_ctrl[:, 2]     # u*w
        vw = curve_ctrl[:, 1] * curve_ctrl[:, 2]     # v*w
        ww = curve_ctrl[:, 2]                        # w

    U_sb = to_scaled(uw)
    V_sb = to_scaled(vw)
    W_sb = to_scaled(ww)

    Wu_sb = W_sb - U_sb                          # (W - U)
    Wv_sb = W_sb - V_sb                          # (W - V)

    # Pre‑compute powers that will be reused (full lists → cumulative convolution)
    pow_U  = _all_powers(U_sb, m)            # pow_U[i]  = U^i
    pow_Wu = _all_powers(Wu_sb, m)[::-1]     # pow_Wu[i] = (W-U)^{m-i}
    pow_V  = _all_powers(V_sb, n)
    pow_Wv = _all_powers(Wv_sb, n)[::-1]

    # ------------------------------------------------------------------
    # 2.  Scale the patch lattice once so that binomials disappear (× C(m,i)·C(n,j))
    # ------------------------------------------------------------------
    patch_sb = to_scaled_2d(patch_ctrl.astype(float))

    # ------------------------------------------------------------------
    # 3.  Assemble the homogeneous numerator (last column = denominator/weight).
    #     B_i^m(U/W)·B_j^n(V/W) ∝ U^i (W-U)^{m-i} V^j (W-V)^{n-j}; the i-only and
    #     j-only factors are loop-invariant, so hoist them out of the (i,j) loop.
    # ------------------------------------------------------------------
    basis_u = [sb_convolve(pow_Wu[i], pow_U[i]) for i in range(m + 1)]
    basis_v = [sb_convolve(pow_Wv[j], pow_V[j]) for j in range(n + 1)]

    num = np.zeros((deg + 1, dim1))
    for i in range(m + 1):
        for j in range(n + 1):
            basis_uv = sb_convolve(basis_u[i], basis_v[j])
            num[:len(basis_uv)] += np.outer(basis_uv, patch_sb[i, j])

    # ------------------------------------------------------------------
    # 4.  Back to ordinary Bernstein control points
    # ------------------------------------------------------------------
    homog = from_scaled(num)                           # (deg+1, dim1)

    if return_cartesian:
        return homog[:, :-1] / homog[:, -1:]
    return homog


# ---------------------------------------------------------------------
#  Helpers for general NURBS composition
# ---------------------------------------------------------------------

def _bernstein_to_power(coeff: np.ndarray) -> np.ndarray:
    """Convert Bernstein coefficients (degree n) to power basis coeffs a_k x^k."""
    n = len(coeff) - 1
    power = np.zeros(n + 1, dtype=float)
    for r in range(n + 1):
        s = 0.0
        for i in range(r + 1):
            s += coeff[i] * math.comb(n, i) * math.comb(n - i, r - i) * ((-1) ** (r - i))
        power[r] = s
    return power


def _segment_interval(crv: NURBSCurveTuple) -> tuple[float, float]:
    deg = crv.order - 1
    return crv.knot[deg], crv.knot[-deg - 1]


def _roots_against_constant(seg: NURBSCurveTuple, const: float, comp_idx: int) -> list[float]:
    """
    Return global parameter values t where component comp_idx of the rational Bézier
    segment equals const.
    """
    start, end = _segment_interval(seg)
    cpts = seg.control_points[:, comp_idx]
    w = seg.weights
    num_minus_c = cpts * w - const * w  # Bernstein coefficients of numerator - const*den
    power = _bernstein_to_power(num_minus_c)
    # np.roots expects descending order
    roots = np.roots(power[::-1])
    params = []
    for r in roots:
        if abs(r.imag) < 1e-12:
            s = r.real
            if -1e-12 < s < 1 + 1e-12:
                t_global = start + (end - start) * s
                if start + 1e-12 < t_global < end - 1e-12:
                    params.append(t_global)
    return params


def _collect_split_parameters(curve: NURBSCurveTuple,
                              u_knots: np.ndarray,
                              v_knots: np.ndarray) -> list[float]:
    """Find all curve parameters where (u(t), v(t)) crosses surface knot lines."""
    params = []
    if len(u_knots) == 0 and len(v_knots) == 0:
        return params
    bez_segs = decompose_curve(curve)
    for seg in bez_segs:
        for ku in u_knots:
            params.extend(_roots_against_constant(seg, ku, 0))
        for kv in v_knots:
            params.extend(_roots_against_constant(seg, kv, 1))
    # unique + sorted with tolerance
    params = sorted({round(p, 12): p for p in params}.values())
    return params


def _merge_bezier_segments(segments: list[NURBSCurveTuple]) -> NURBSCurveTuple:
    """Concatenate Bézier curve segments of the same degree into one NURBSCurveTuple."""
    if not segments:
        raise ValueError("No segments to merge.")
    deg = segments[0].order - 1
    boundaries = []
    ctrlpts = []
    weights = []

    for idx, seg in enumerate(segments):
        s0, s1 = _segment_interval(seg)
        if idx == 0:
            boundaries.append(s0)
        boundaries.append(s1)
        if idx == 0:
            ctrlpts.append(seg.control_points)
            weights.append(seg.weights)
        else:
            ctrlpts.append(seg.control_points[1:])
            weights.append(seg.weights[1:])

    knots = [boundaries[0]] * (deg + 1)
    for b in boundaries[1:-1]:
        knots.extend([b] * deg)
    knots.extend([boundaries[-1]] * (deg + 1))

    ctrlpts_arr = np.vstack(ctrlpts)
    weights_arr = np.hstack(weights)

    merged = NURBSCurveTuple(deg + 1, np.array(knots, dtype=float), ctrlpts_arr, weights_arr)

    # Try to remove superfluous knots introduced by splitting
    for b in boundaries[1:-1]:
        merged, _ = remove_knot_curve_max(merged, b, num=deg)
    return merged


# ---------------------------------------------------------------------
#  2‑D scaled‑Bernstein helpers
# ---------------------------------------------------------------------

def _binom_vec(n: int) -> np.ndarray:
    """Return vector [C(n,0) … C(n,n)] as float (cached, exact via math.comb)."""
    return _binom_scale(n)


def to_scaled_2d(ctrl: np.ndarray) -> np.ndarray:
    """
    (p+1,q+1,[…]) Bernstein  →  scaled‑Bernstein:
    multiplies each coefficient by C(p,i)·C(q,j).
    """
    p, q = ctrl.shape[0] - 1, ctrl.shape[1] - 1
    scale = _binom_vec(p)[:, None] * _binom_vec(q)[None, :]
    if ctrl.ndim == 3:
        return ctrl * scale[:, :, None]
    return ctrl * scale


def from_scaled_2d(coeff: np.ndarray) -> np.ndarray:
    """
    Scaled‑Bernstein  →  ordinary Bernstein (divides by the binomials).
    """
    p, q = coeff.shape[0] - 1, coeff.shape[1] - 1
    scale = _binom_vec(p)[:, None] * _binom_vec(q)[None, :]
    if coeff.ndim == 3:
        return coeff / scale[:, :, None]
    return coeff / scale


try:
    from scipy.signal import convolve2d as _scipy_convolve2d
except Exception:  # pragma: no cover - scipy is a declared dependency
    _scipy_convolve2d = None


def conv2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    2‑D discrete convolution of small coefficient grids (mode='full').

    Delegates to ``scipy.signal.convolve2d``, which uses the C-level *direct*
    method (not FFT) and therefore matches the previous pure‑Python loop to
    machine precision while being ~5x faster — conv2d dominated
    ``compose_patch_patch`` runtime (~95%).  Falls back to the NumPy loop if
    scipy is somehow unavailable.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if _scipy_convolve2d is not None:
        return _scipy_convolve2d(a, b)
    out = np.zeros((a.shape[0] + b.shape[0] - 1,
                    a.shape[1] + b.shape[1] - 1))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] != 0.0:
                out[i:i + b.shape[0], j:j + b.shape[1]] += a[i, j] * b
    return out


def sb_pow2d(poly: np.ndarray, k: int) -> np.ndarray:
    """k‑fold self‑convolution in the scaled‑Bernstein setting."""
    if k == 0:
        return np.array([[1.0]])
    result = np.array([[1.0]])
    base = poly.copy()
    e = k
    while e:
        if e & 1:
            result = conv2d(result, base)
        if e > 1:
            base = conv2d(base, base)
        e >>= 1
    return result




def compose_patch_patch(outer_ctrl: np.ndarray,
                                   param_ctrl: np.ndarray,
                                   return_cartesian: bool = False) -> np.ndarray:
    """
    Exact composition C(s,t) = B(u/w, v/w).

    Parameters
    ----------
    outer_ctrl : ndarray (m+1, n+1, D+1)
        Homogeneous control lattice of the outer rational Bézier surface; the
        last column is the weight (D=3 ⇒ (x*w, y*w, z*w, w)).  Any D >= 1.
    param_ctrl : ndarray (p+1, q+1, 3)
        Control lattice of the rational parameter patch (u,v,w).  Columns
        are plain (u, v, w) values, *not* multiplied by w.
    return_cartesian : bool, default False
        If True, divides by weights and returns Cartesian XYZ controls.
        Otherwise returns homogeneous controls.

    Returns
    -------
    ctrl_out : ndarray (deg_s+1, deg_t+1, D+1)   or  (..., D)
        Homogeneous (or Cartesian) control lattice of the composed
        rational Bézier surface.  Degrees:
            deg_s = (m + n) * p
            deg_t = (m + n) * q
    """
    # --------------------------------------------------------------
    # 0.  Validate + degrees / spatial dimension
    # --------------------------------------------------------------
    if outer_ctrl.ndim != 3 or outer_ctrl.shape[2] < 2:
        raise ValueError(f"outer_ctrl must be (m+1, n+1, D+1) with D >= 1; got shape {outer_ctrl.shape}")
    if param_ctrl.ndim != 3 or param_ctrl.shape[2] != 3:
        raise ValueError(f"param_ctrl must be (p+1, q+1, 3) with columns (u, v, w); got shape {param_ctrl.shape}")
    m, n = outer_ctrl.shape[0] - 1, outer_ctrl.shape[1] - 1
    p, q = param_ctrl.shape[0] - 1, param_ctrl.shape[1] - 1
    deg_s = (m + n) * p
    deg_t = (m + n) * q
    dim1 = outer_ctrl.shape[2]                  # D + 1  (spatial coords + weight)

    # --------------------------------------------------------------
    # 1.  Homogeneous (U,V,W) of the parameter patch in SB form
    # --------------------------------------------------------------
    U_ctrl = param_ctrl[:, :, 0] * param_ctrl[:, :, 2]   #  u·w
    V_ctrl = param_ctrl[:, :, 1] * param_ctrl[:, :, 2]   #  v·w
    W_ctrl = param_ctrl[:, :, 2]                         #  w

    U_sb   = to_scaled_2d(U_ctrl)
    V_sb   = to_scaled_2d(V_ctrl)
    W_sb   = to_scaled_2d(W_ctrl)
    WmU_sb = W_sb - U_sb
    WmV_sb = W_sb - V_sb

    # Pre‑compute powers that will be reused many times
    pow_U    = [sb_pow2d(U_sb,     i) for i in range(m + 1)]
    pow_WmU  = [sb_pow2d(WmU_sb, m - i) for i in range(m + 1)]
    pow_V    = [sb_pow2d(V_sb,     j) for j in range(n + 1)]
    pow_WmV  = [sb_pow2d(WmV_sb, n - j) for j in range(n + 1)]

    # --------------------------------------------------------------
    # 2.  Scale the outer lattice once to remove binomials (× C(m,i)·C(n,j))
    # --------------------------------------------------------------
    outer_sb = to_scaled_2d(outer_ctrl.astype(float))

    # --------------------------------------------------------------
    # 3.  Accumulate the homogeneous numerator grid (last channel = weight).
    #     i-only and j-only factors are loop-invariant → hoist them out.
    # --------------------------------------------------------------
    basis_u = [conv2d(pow_WmU[i], pow_U[i]) for i in range(m + 1)]
    basis_v = [conv2d(pow_WmV[j], pow_V[j]) for j in range(n + 1)]

    num = np.zeros((deg_s + 1, deg_t + 1, dim1))
    for i in range(m + 1):
        for j in range(n + 1):
            basis_uv = conv2d(basis_u[i], basis_v[j])      # SB coeffs
            su, sv = basis_uv.shape
            num[:su, :sv, :] += basis_uv[:, :, None] * outer_sb[i, j]

    # --------------------------------------------------------------
    # 4.  Convert back to ordinary Bernstein control lattices
    # --------------------------------------------------------------
    homog = from_scaled_2d(num)                         # (deg_s+1, deg_t+1, dim1)

    if return_cartesian:
        return homog[..., :-1] / homog[..., -1:]
    return homog


def compose_nurbs_surface_curve(surface: NURBSSurfaceTuple,
                                curve: NURBSCurveTuple) -> NURBSCurveTuple:
    """
    Exact composition of a general NURBS surface with a NURBS parameter curve.

    Steps:
      1) Decompose the surface into Bézier sub‑patches.
      2) Split the parameter curve so that each piece stays inside a single
         sub‑patch param rectangle (u‑knots × v‑knots).
      3) Decompose each sub‑curve into Bézier segments.
      4) Compose each Bézier pair with ``compose_patch_curve`` (using homogeneous
         parameter controls).
      5) Merge the resulting Bézier spatial curves and remove redundant knots.

    Returns a NURBSCurveTuple representing the composed 3‑D rational curve.
    """
    # 0) Surface interior knots
    du, dv = surface.order_u - 1, surface.order_v - 1
    ku_int = np.unique(surface.knot_u[du + 1:-(du + 1)])
    kv_int = np.unique(surface.knot_v[dv + 1:-(dv + 1)])

    # 1) Split the parameter curve where it crosses surface knot lines
    split_params = _collect_split_parameters(curve, ku_int, kv_int)
    curve_parts = split_curve_multiple(curve, split_params) if split_params else [curve]

    # 2) Fully decompose each part into Bézier segments
    curve_beziers = []
    for c in curve_parts:
        curve_beziers.extend(decompose_curve(c))

    # 3) Decompose surface into Bézier patches and cache their domains/controls
    bez_patches = decompose_surface(surface, "uv")
    patch_data = []
    for p in bez_patches:
        u0, u1 = p.knot_u[p.order_u - 1], p.knot_u[-p.order_u]
        v0, v1 = p.knot_v[p.order_v - 1], p.knot_v[-p.order_v]
        # normalize patch domain to [0,1] for SB composition
        p_norm = p._replace(
            knot_u=normalize_knots(np.array(p.knot_u, float), p.order_u - 1),
            knot_v=normalize_knots(np.array(p.knot_v, float), p.order_v - 1),
        )
        patch_data.append((u0, u1, v0, v1, nurbs_bezier_to_bern(p_norm)))

    composed_segments: list[NURBSCurveTuple] = []

    for seg_orig in curve_beziers:
        # pick containing patch using midpoint
        s0, s1 = _segment_interval(seg_orig)
        mid = 0.5 * (s0 + s1)
        uv_mid = evaluate_nurbs_curve(seg_orig, mid)["C"]
        u_mid, v_mid = uv_mid[0], uv_mid[1]

        target_patch = None
        target_bounds = None
        for u0, u1, v0, v1, pbern in patch_data:
            tol_u = _LOCATE_RTOL * max(1.0, u1 - u0)
            tol_v = _LOCATE_RTOL * max(1.0, v1 - v0)
            if (u0 - tol_u) <= u_mid <= (u1 + tol_u) and (v0 - tol_v) <= v_mid <= (v1 + tol_v):
                target_patch = pbern
                target_bounds = (u0, u1, v0, v1)
                break

        if target_patch is None:
            raise ValueError(
                f"Curve segment [{s0}, {s1}] (midpoint u,v = {u_mid}, {v_mid}) "
                f"not inside any surface sub‑patch."
            )

        # Compose using homogeneous parameter controls (u*w, v*w, w)
        seg = normalize_knots_curve(seg_orig)
        u0, u1, v0, v1 = target_bounds
        du = u1 - u0
        dv = v1 - v0
        if du == 0 or dv == 0:
            raise ValueError("Degenerate patch bounds encountered during composition.")

        u_local = (seg.control_points[:, 0] - u0) / du
        v_local = (seg.control_points[:, 1] - v0) / dv

        curve_bern = np.stack(
            [u_local * seg.weights,
             v_local * seg.weights,
             seg.weights],
            axis=1,
        )
        bern_out = compose_patch_curve(target_patch, curve_bern,
                                       return_cartesian=False,
                                       curve_ctrl_homogeneous=True)

        composed_segments.append(
            bern_to_nurbs_bezier(bern_out, interval=(s0, s1), rational=True)
        )

    return _merge_bezier_segments(composed_segments)


def compose_nurbs_curve_curve(spatial_curve: NURBSCurveTuple,
                              param_curve: NURBSCurveTuple) -> NURBSCurveTuple:
    """
    Exact composition of a general NURBS curve with a 1‑D NURBS parameter curve.

    Given a spatial curve ``C(s)`` (2‑D or 3‑D, rational) and a scalar‑valued
    parameter curve ``s = sigma(t) = sigma_num(t)/omega(t)`` whose values lie in
    ``C``'s knot domain ``[s_min, s_max]``, builds the composed curve

        C_new(t) = C(sigma(t))

    exactly, as a single NURBSCurveTuple.  This is an exact reparameterisation:
    each Bézier piece is degree‑elevated to ``deg(C) * deg(sigma)``.

    ``param_curve`` is a 1‑D NURBS curve: its ``control_points`` have shape
    ``(p+1, 1)`` (the scalar s‑values) with the usual ``weights``.

    Steps (mirror of :func:`compose_nurbs_surface_curve`):
      1) Find every ``t`` where ``sigma(t)`` crosses an interior knot of ``C``.
      2) Split ``sigma`` at those crossings, then fully decompose each part into
         Bézier segments.  This makes every ``sigma`` segment respect BOTH the
         crossings with ``C``'s knots AND ``sigma``'s own interior knots, so each
         segment maps into a single Bézier piece of ``C``.
      3) Decompose ``C`` into Bézier segments and cache each segment's
         ``[s0, s1]`` interval and homogeneous control net.
      4) For each ``sigma`` Bézier segment, locate the containing ``C`` segment
         (via the segment midpoint value), affinely remap ``sigma``'s control
         values into that segment's local ``[0, 1]`` domain (weights untouched),
         and compose with :func:`compose_curve_curve`.
      5) Merge the resulting Bézier spatial curves, removing redundant knots.

    Returns a NURBSCurveTuple representing the composed rational curve C(sigma(t)).
    """
    # 0) Interior knots of the spatial curve C(s)
    d = spatial_curve.order - 1
    ks_int = np.unique(spatial_curve.knot[d + 1:-(d + 1)])

    # 1) Split the parameter curve where sigma(t) crosses C's interior knots.
    #    _collect_split_parameters tests component 0 against the given knots and
    #    component 1 against the second list; passing an empty second list makes
    #    it a clean 1‑D crossing finder for the scalar curve sigma.
    split_params = _collect_split_parameters(param_curve, ks_int, np.array([], dtype=float))
    param_parts = split_curve_multiple(param_curve, split_params) if split_params else [param_curve]

    # 2) Fully decompose each part into Bézier segments (respects sigma's own knots too)
    param_beziers = []
    for c in param_parts:
        param_beziers.extend(decompose_curve(c))

    # 3) Decompose C into Bézier segments; cache domain + homogeneous control net
    seg_data = []
    for seg in decompose_curve(spatial_curve):
        s0 = seg.knot[seg.order - 1]
        s1 = seg.knot[-seg.order]
        seg_data.append((s0, s1, nurbs_bezier_to_bern(seg)))   # bern is (n+1, D+1) homogeneous

    composed_segments: list[NURBSCurveTuple] = []

    for seg_orig in param_beziers:
        # pick the containing C segment using the midpoint value of sigma
        t0, t1 = _segment_interval(seg_orig)
        s_mid = evaluate_nurbs_curve(seg_orig, 0.5 * (t0 + t1))["C"][0]

        target_bern = None
        target_bounds = None
        for s0, s1, bern in seg_data:
            tol = _LOCATE_RTOL * max(1.0, s1 - s0)
            if (s0 - tol) <= s_mid <= (s1 + tol):
                target_bern = bern
                target_bounds = (s0, s1)
                break

        if target_bern is None:
            raise ValueError(
                f"sigma segment [{t0}, {t1}] (mid value {s_mid}) is outside the "
                f"spatial curve domain [{seg_data[0][0]}, {seg_data[-1][1]}]."
            )

        s0, s1 = target_bounds
        ds = s1 - s0
        if ds == 0:
            raise ValueError("Degenerate Bézier segment of the spatial curve encountered.")

        # affine remap of sigma's Cartesian values into the local [0,1] domain of
        # the chosen C segment; the map s -> (s - s0)/ds leaves the weights intact
        s_local = (seg_orig.control_points[:, 0] - s0) / ds
        param_ctrl = np.stack([s_local, seg_orig.weights], axis=1)

        bern_out = compose_curve_curve(target_bern, param_ctrl)

        composed_segments.append(
            bern_to_nurbs_bezier(bern_out, interval=(t0, t1), rational=True)
        )

    return _merge_bezier_segments(composed_segments)


if __name__=="__main__":

    # Cubic rational Bézier curve (homogeneous control points)
    C_ctrl = np.array([[0.0, 0.0, 0.0, 1.0],
                       [1.0, 2.0, 0.0, 1.0],
                       [2.0, 2.0, 0.0, 1.0],
                       [3.0, 0.0, 0.0, 1.0]])

    # Quadratic re‑parameterisation  u(t)
    #u_ctrl = np.array([[0.0], [0.3], [1.0]])            # maps t∈[0,1] → u∈[0,1]
    #
    #composite = compose_curve_curve(C_ctrl, u_ctrl, return_cartesian=False)
    #print("Homogeneous control points of C(u(t)):")
    #print(composite)
    u_ctrlv2 = np.array([[0.0,1.], [0.3,1.], [1.0,1.]])
    composite2 = compose_curve_curve(C_ctrl, u_ctrlv2)
    print('v2:')
    print(composite2)
    st1 = NURBSSurfaceTuple(
        order_u=2,
        order_v=2,
        knot_u=np.array([  0.        ,   0.        , 1., 1]),
        knot_v=np.array([  0.        ,   0.        , 1, 1]),
        control_points=np.array([[[-128.25004889, -129.85828719,   67.43742325],
                [-128.25004889,  129.85828719,    0.        ]],
    
               [[ 128.25004889,  -46.98266257,    0.        ],
                [ 128.25004889,  129.85828719,    0.        ]]]),
        weights=np.array([[1., 1.],
               [1., 1.]])
    )

    crv = NURBSCurveTuple(order=4, knot=np.array([0., 0., 0., 0., 1,
    
                                                  1, 1, 1]), control_points=np.array([[0.3401254, 1. ],
                                                                                      [0.34273043, 0.55703351 ],
                                                                                      [0.58499832, 0.28611028],
                                                                                      [1., 0.5]]),
                          weights=np.array([1., 1., 1., 1.]))
    # plain (u, v, w) parameter net for the curve_ctrl_homogeneous=False path
    curve_bern=np.column_stack([crv.control_points, crv.weights])
    patch_bern = to_homogeneous_2d(st1.control_points, st1.weights)
    curve_3d=compose_patch_curve(patch_bern,curve_bern, curve_ctrl_homogeneous=False)
    cpts, weights = from_homogeneous_1d(curve_3d)
    curve_3d_rat = compose_patch_curve(nurbs_bezier_to_bern(st1),nurbs_bezier_to_bern(crv), curve_ctrl_homogeneous=True)

    result=NURBSCurveTuple(order=cpts.shape[0], knot=generate_knots(cpts.shape[0],cpts.shape[0]-1),control_points=cpts,weights=weights)
    print(result)
    print(bern_to_nurbs_bezier(curve_3d_rat))

    surf = NURBSSurfaceTuple(
        order_u=4,
        order_v=4,
        knot_u=np.array([0., 0., 0., 0., 0.5, 1., 1., 1., 1.]),
        knot_v=np.array([0., 0., 0., 0., 0.5, 1., 1., 1., 1.]),
        control_points=np.array([[[-5., -5., 0.],
                                  [-5., -3.333, 0.],
                                  [-5., 0., 0.],
                                  [-5., 3.333, 1.653],
                                  [-5., 5., 1.653]],

                                 [[-3.333, -5., 0.],
                                  [-3.333, -3.333, 0.],
                                  [-3.333, -0., 0.],
                                  [-3.333, 3.333, 1.653],
                                  [-3.333, 5., 1.653]],

                                 [[0., -5., 3.306],
                                  [-0., -3.333, 3.306],
                                  [0., 0., -3.3],
                                  [-0., 3.333, -3.3],
                                  [-0., 5., -3.306]],

                                 [[3.333, -5., 0.],
                                  [3.333, -3.333, 0.],
                                  [3.333, -0., 0.],
                                  [3.333, 3.333, 1.653],
                                  [3.333, 5., 1.653]],

                                 [[5., -5., 0.],
                                  [5., -3.333, 0.],
                                  [5., 0., 0.],
                                  [5., 3.333, 1.653],
                                  [5., 5., 1.653]]]),
        weights=np.array([[1., 1., 1., 1., 1.],
                          [1., 1., 1., 1., 1.],
                          [1., 1., 1., 1., 1.],
                          [1., 1., 1., 1., 1.],
                          [1., 1., 1., 1., 1.]])
    )

    curve = NURBSCurveTuple(
        order=4,
        knot=np.array([0., 0., 0., 0., 8.868, 8.868, 8.868, 17.737,
                       17.737, 17.737, 17.737]),
        control_points=np.array([[0.5, 0.135, 0.],
                                 [-0.046, 0.597, 0.],
                                 [0.231, 1.095, 0.],
                                 [0.5, 0.721, 0.],
                                 [0.769, 1.095, 0.],
                                 [1.046, 0.597, 0.],
                                 [0.5, 0.135, 0.]]),
        weights=np.array([1.791, 2.102, 0.881, 1., 0.881, 2.102, 1.791])
    )

    result=compose_nurbs_surface_curve(surf, curve)
    print(result)

    # compose_nurbs_curve_curve: reparameterise a 2-D rational curve C(s) by a
    # 1-D NURBS sigma(t) whose values sweep C's [0, 1] knot domain
    spatial = NURBSCurveTuple(
        order=3, knot=np.array([0., 0., 0., 0.5, 1., 1., 1.]),
        control_points=np.array([[0., 0.], [1., 2.], [2., 2.], [3., 0.]]),
        weights=np.array([1., 2., 0.5, 1.]),
    )
    sigma = NURBSCurveTuple(
        order=3, knot=np.array([0., 0., 0., 0.4, 1., 1., 1.]),
        control_points=np.array([[0.0], [0.3], [0.6], [1.0]]),
        weights=np.array([1., 2., 0.5, 1.]),
    )
    print(compose_nurbs_curve_curve(spatial, sigma))
