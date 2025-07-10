from __future__ import annotations


from mmcore.geom._nurbs_knots import generate_knots

# ----------------------------------------------------------------------
#  Helpers for the scaled‑Bernstein (SB) representation
# ----------------------------------------------------------------------
import math
import numpy as np

from mmcore.geom._nurbs_eval import NURBSSurfaceTuple, NURBSCurveTuple, to_homogeneous_2d, from_homogeneous_1d, to_homogeneous_1d, \
    from_homogeneous_2d
from mmcore.numeric.binom import binomial_coefficient_py
# ------------------------------------------------------------------
#  Scaled‑Bernstein helpers
# ------------------------------------------------------------------


def to_scaled(ctrl: np.ndarray) -> np.ndarray:
    """
    Standard Bernstein control points  ->  scaled‑Bernstein coefficients
    (multiplies by the binomial C(n,i)).
    """
    n = len(ctrl) - 1
    scale = np.fromiter((binomial_coefficient_py(n, i) for i in range(n + 1)),
                        dtype=float, count=n + 1)
    return ctrl * scale[..., None] if ctrl.ndim > 1 else ctrl * scale


def from_scaled(coeff: np.ndarray) -> np.ndarray:
    """
    Scaled‑Bernstein  ->  ordinary Bernstein control points.
    Divides by C(n,i).
    """
    n = len(coeff) - 1
    scale = np.fromiter((binomial_coefficient_py(n, i) for i in range(n + 1)),
                        dtype=float, count=n + 1)
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


def compose_curve_curve(spatial_ctrl: np.ndarray,
                                 param_ctrl:   np.ndarray,
                                 return_cartesian: bool = False) -> np.ndarray:
    """
    Parameters
    ----------
    spatial_ctrl : (n+1, 4) ndarray
        Homogeneous control points of the original 3‑D rational curve
        (x*w, y*w, z*w, w).  Degree n.
    param_ctrl   : (p+1, 2) ndarray
        Control points of the 1‑D rational re‑parametrisation curve
        in the (sigma, omega) form.  Degree p.
    return_cartesian : bool, default False
        If True, divides by the weights and returns Cartesian XYZ points.
        Otherwise returns homogeneous coordinates.

    Returns
    -------
    ctrl_out : ndarray ((n*p)+1, 4)  or  (..., 3)
        Control points of the composed rational Bézier curve
        C_new(t) = C(sigma(t)/omega(t)).
    """
    # ---------------------------------------------------------------
    # 0.  Degrees
    # ---------------------------------------------------------------
    n = spatial_ctrl.shape[0] - 1
    p = param_ctrl.shape[0]   - 1
    deg = n * p                                # final polynomial degree

    # ---------------------------------------------------------------
    # 1.  Build SB polynomials for sigma(t) and omega(t)
    # ---------------------------------------------------------------
    sigma_num = param_ctrl[:, 0] * param_ctrl[:, 1]   # sigma·omega
    omega_den = param_ctrl[:, 1]                      # omega

    sigma_sb = to_scaled(sigma_num)
    omega_sb = to_scaled(omega_den)
    omega_m_sigma_sb = omega_sb - sigma_sb                              # (omega - sigma)

    # Pre‑compute needed powers
    pow_sigma   = [sb_pow(sigma_sb,     i) for i in range(n + 1)]
    pow_omega_m_sigma = [sb_pow(omega_m_sigma_sb, n - i) for i in range(n + 1)]

    # ---------------------------------------------------------------
    # 2.  Scale spatial control points once to remove binomials
    # ---------------------------------------------------------------
    spatial_sb = spatial_ctrl.astype(float).copy()
    for i in range(n + 1):
        spatial_sb[i] *= math.comb(n, i)

    # ---------------------------------------------------------------
    # 3.  Assemble numerator and denominator polynomials (SB coeffs)
    # ---------------------------------------------------------------
    num_x = np.zeros(deg + 1)
    num_y = np.zeros(deg + 1)
    num_z = np.zeros(deg + 1)
    den   = np.zeros(deg + 1)

    for i in range(n + 1):
        basis = sb_convolve(pow_omega_m_sigma[i], pow_sigma[i])     # SB coefficients

        xw, yw, zw, w = spatial_sb[i]
        num_x[:len(basis)] += xw * basis
        num_y[:len(basis)] += yw * basis
        num_z[:len(basis)] += zw * basis
        den  [:len(basis)] +=  w * basis

    # ---------------------------------------------------------------
    # 4.  Back to ordinary Bernstein control points
    # ---------------------------------------------------------------
    cx = from_scaled(num_x)
    cy = from_scaled(num_y)
    cz = from_scaled(num_z)
    cw = from_scaled(den)

    homog = np.stack([cx, cy, cz, cw], axis=-1)       # (deg+1, 4)

    if return_cartesian:
        xyz = homog[:, :3] / homog[:, 3:4]
        return xyz
    return homog
# ----------------------------------------------------------------------
#  Composition (patch ∘ curve) completely in SB form
# ----------------------------------------------------------------------

def compose_patch_curve_non_rational(patch_ctrl: np.ndarray,
                           curve_ctrl: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    patch_ctrl : (m+1, n+1, 4)
        Homogeneous control lattice of the rational Bézier patch.
    curve_ctrl : (p+1, 2)
        Planar Bézier control points (u,v) inside [0,1]×[0,1].



    Returns
    -------
    (deg+1, 4) ndarray
        Control points of the composed spatial curve in homogeneous coords, still in Bernstein
        form.  Degree `deg = (m+n)*p`.
    """
    m, n = patch_ctrl.shape[0]-1, patch_ctrl.shape[1]-1
    p    = curve_ctrl.shape[0]-1
    deg  = (m + n) * p

    # 1.  Put everything in scaled‑Bernstein (SB) form
    u_sb = to_scaled(curve_ctrl[:, 0])           # length p+1
    v_sb = to_scaled(curve_ctrl[:, 1])

    one_sb = to_scaled(np.ones(p + 1))           # constant "1" of deg p

    um_sb = one_sb - u_sb                        # (1-u(t))  in SB form
    vm_sb = one_sb - v_sb

    # Pre‑compute powers we will need later
    pow_u   = [sb_pow(u_sb,      i) for i in range(m + 1)]
    pow_um  = [sb_pow(um_sb, m - i) for i in range(m + 1)]
    pow_v   = [sb_pow(v_sb,      j) for j in range(n + 1)]
    pow_vm  = [sb_pow(vm_sb, n - j) for j in range(n + 1)]

    # Patch lattice in SB form so that binomials disappear
    patch_sb = patch_ctrl.copy()
    for i in range(m + 1):
        for j in range(n + 1):
            patch_sb[i, j] *= math.comb(m, i) * math.comb(n, j)


    # 2.  Build numerator and denominator polynomials (still SB)
    num_x = np.zeros(deg + 1)
    num_y = np.zeros(deg + 1)
    num_z = np.zeros(deg + 1)
    den   = np.zeros(deg + 1)

    for i in range(m + 1):
        for j in range(n + 1):
            basis_u  = np.convolve(pow_um[i],  pow_u[i])
            basis_v  = np.convolve(pow_vm[j],  pow_v[j])
            basis_uv = np.convolve(basis_u, basis_v)   # SB coeffs

            xw, yw, zw, w = patch_sb[i, j]
            num_x[:len(basis_uv)] += xw * basis_uv
            num_y[:len(basis_uv)] += yw * basis_uv
            num_z[:len(basis_uv)] += zw * basis_uv
            den  [:len(basis_uv)] +=  w * basis_uv

    # 3.  Convert the 1‑D SB lists back to ordinary Bernstein control pts
    cx = from_scaled(num_x)
    cy = from_scaled(num_y)
    cz = from_scaled(num_z)
    cw = from_scaled(den)

    homog = np.stack([cx, cy, cz, cw], axis=-1)  # (deg+1, 4)

    return homog



def compose_patch_curve(patch_ctrl: np.ndarray,
                                 curve_ctrl: np.ndarray,
                                 return_cartesian: bool = False) -> np.ndarray:
    """
    Exact composition  c(t) = patch(u(t)/w(t), v(t)/w(t))

    Parameters
    ----------
    patch_ctrl : ndarray (m+1, n+1, 4)
        Homogeneous control lattice of the rational Bézier patch:
        columns = (x*w, y*w, z*w, w).
    curve_ctrl : ndarray (p+1, 3)
        Rational Bézier control points in parameter space:
        columns = (u, v, w). *These are NOT pre‑multiplied by w.*
    return_cartesian : bool, default False
        If True, divides by the weights and returns Cartesian control
        points (xyz).  Otherwise returns homogeneous (xw,yw,zw,w).

    Returns
    -------
    ctrl_out : ndarray ((m+n)*p + 1, 4)  or  (..., 3)
        Control points of the composed rational Bézier curve.
    """
    # ------------------------------------------------------------------
    # 0.  Sizes and degrees
    # ------------------------------------------------------------------
    m, n = patch_ctrl.shape[0] - 1, patch_ctrl.shape[1] - 1
    p    = curve_ctrl.shape[0]   - 1
    deg  = (m + n) * p                          # (total) polynomial degree

    # ------------------------------------------------------------------
    # 1.  Parameter curve  —  build SB polynomials U(t), V(t), W(t)
    #     We first convert (u, v, w) ⇒ homogeneous (u·w, v·w, w)
    # ------------------------------------------------------------------
    uw = curve_ctrl[:, 0] * curve_ctrl[:, 2]     # u*w
    vw = curve_ctrl[:, 1] * curve_ctrl[:, 2]     # v*w
    ww = curve_ctrl[:, 2]                        # w

    U_sb = to_scaled(uw)
    V_sb = to_scaled(vw)
    W_sb = to_scaled(ww)

    Wu_sb = W_sb - U_sb                          # (W - U)
    Wv_sb = W_sb - V_sb                          # (W - V)

    # Pre‑compute powers that will be reused
    pow_U    = [sb_pow(U_sb,    i)      for i in range(m + 1)]
    pow_Wu   = [sb_pow(Wu_sb, m - i)    for i in range(m + 1)]
    pow_V    = [sb_pow(V_sb,    j)      for j in range(n + 1)]
    pow_Wv   = [sb_pow(Wv_sb, n - j)    for j in range(n + 1)]

    # ------------------------------------------------------------------
    # 2.  Scale the patch lattice once so that binomials disappear
    # ------------------------------------------------------------------
    patch_sb = patch_ctrl.copy().astype(float)
    for i in range(m + 1):
        for j in range(n + 1):
            patch_sb[i, j] *= math.comb(m, i) * math.comb(n, j)

    # ------------------------------------------------------------------
    # 3.  Assemble numerator and denominator polynomials (SB coeffs)
    # ------------------------------------------------------------------
    num_x = np.zeros(deg + 1)
    num_y = np.zeros(deg + 1)
    num_z = np.zeros(deg + 1)
    den   = np.zeros(deg + 1)

    for i in range(m + 1):
        for j in range(n + 1):
            #   B_i^m(U/W) · B_j^n(V/W)  ∝  U^i (W-U)^{m-i} V^j (W-V)^{n-j}
            basis_u  = sb_convolve(pow_Wu[i], pow_U[i])
            basis_v  = sb_convolve(pow_Wv[j], pow_V[j])
            basis_uv = sb_convolve(basis_u,  basis_v)

            xw, yw, zw, w = patch_sb[i, j]
            num_x[:len(basis_uv)] += xw * basis_uv
            num_y[:len(basis_uv)] += yw * basis_uv
            num_z[:len(basis_uv)] += zw * basis_uv
            den  [:len(basis_uv)] +=  w * basis_uv

    # ------------------------------------------------------------------
    # 4.  Back to ordinary Bernstein control points
    # ------------------------------------------------------------------
    cx = from_scaled(num_x)
    cy = from_scaled(num_y)
    cz = from_scaled(num_z)
    cw = from_scaled(den)

    homog = np.stack([cx, cy, cz, cw], axis=-1)        # (deg+1, 4)

    if return_cartesian:
        xyz = homog[:, :3] / homog[:, 3:4]
        return xyz
    return homog


import math
import numpy as np


# ---------------------------------------------------------------------
#  2‑D scaled‑Bernstein helpers
# ---------------------------------------------------------------------

def _binom_vec(n: int) -> np.ndarray:
    """Return vector [C(n,0) … C(n,n)] as float."""
    return np.fromiter((binomial_coefficient_py(n, i) for i in range(n + 1)),
                       dtype=float, count=n + 1)


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


def conv2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    2‑D discrete convolution of small coefficient grids.
    Pure‑NumPy, O(N³) but fast for CAD degrees (≲ 20×20).
    """
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
    outer_ctrl : ndarray (m+1, n+1, 4)
        Homogeneous control lattice of the outer rational Bézier surface.
    param_ctrl : ndarray (p+1, q+1, 3)
        Control lattice of the rational parameter patch (u,v,w).  Columns
        are plain (u, v, w) values, *not* multiplied by w.
    return_cartesian : bool, default False
        If True, divides by weights and returns Cartesian XYZ controls.
        Otherwise returns homogeneous controls.

    Returns
    -------
    ctrl_out : ndarray (deg_s+1, deg_t+1, 4)   or  (..., 3)
        Homogeneous (or Cartesian) control lattice of the composed
        rational Bézier surface.  Degrees:
            deg_s = (m + n) * p
            deg_t = (m + n) * q
    """
    # --------------------------------------------------------------
    # 0.  Degrees
    # --------------------------------------------------------------
    m, n = outer_ctrl.shape[0] - 1, outer_ctrl.shape[1] - 1
    p, q = param_ctrl.shape[0] - 1, param_ctrl.shape[1] - 1
    deg_s = (m + n) * p
    deg_t = (m + n) * q

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
    # 2.  Scale the outer lattice once to remove binomials
    # --------------------------------------------------------------
    outer_sb = outer_ctrl.astype(float).copy()
    for i in range(m + 1):
        for j in range(n + 1):
            outer_sb[i, j] *= math.comb(m, i) * math.comb(n, j)

    # --------------------------------------------------------------
    # 3.  Accumulate numerator and denominator SB coefficient grids
    # --------------------------------------------------------------
    num_x = np.zeros((deg_s + 1, deg_t + 1))
    num_y = np.zeros_like(num_x)
    num_z = np.zeros_like(num_x)
    den   = np.zeros_like(num_x)

    for i in range(m + 1):
        for j in range(n + 1):
            basis_u  = conv2d(pow_WmU[i], pow_U[i])
            basis_v  = conv2d(pow_WmV[j], pow_V[j])
            basis_uv = conv2d(basis_u,    basis_v)     # SB coeffs

            xw, yw, zw, w = outer_sb[i, j]
            num_x[:basis_uv.shape[0], :basis_uv.shape[1]] += xw * basis_uv
            num_y[:basis_uv.shape[0], :basis_uv.shape[1]] += yw * basis_uv
            num_z[:basis_uv.shape[0], :basis_uv.shape[1]] += zw * basis_uv
            den  [:basis_uv.shape[0], :basis_uv.shape[1]] +=  w * basis_uv

    # --------------------------------------------------------------
    # 4.  Convert back to ordinary Bernstein control lattices
    # --------------------------------------------------------------
    cx = from_scaled_2d(num_x)
    cy = from_scaled_2d(num_y)
    cz = from_scaled_2d(num_z)
    cw = from_scaled_2d(den)

    homog = np.stack([cx, cy, cz, cw], axis=-1)         # (deg_s+1,deg_t+1,4)

    if return_cartesian:
        xyz = homog[..., :3] / homog[..., 3:4]
        return xyz
    return homog

if __name__=="__main__":

    # Cubic rational Bézier curve (homogeneous control points)
    C_ctrl = np.array([[0.0, 0.0, 0.0, 1.0],
                       [1.0, 2.0, 0.0, 1.0],
                       [2.0, 2.0, 0.0, 1.0],
                       [3.0, 0.0, 0.0, 1.0]])

    # Quadratic re‑parameterisation  u(t)
    u_ctrl = np.array([0.0, 0.3, 1.0])            # maps t∈[0,1] → u∈[0,1]

    composite = compose_curve_curve(C_ctrl, u_ctrl, return_cartesian=False)
    print("Homogeneous control points of C(u(t)):")
    print(composite)
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
    curve_bern=crv.control_points
    patch_bern = to_homogeneous_2d(st1.control_points, st1.weights)
    curve_3d=compose_patch_curve(patch_bern,curve_bern)
    cpts, weights = from_homogeneous_1d(curve_3d)
    curve_3d_rat = compose_patch_curve(nurbs_bezier_to_bern(st1),nurbs_bezier_to_bern(crv))

    result=NURBSCurveTuple(order=cpts.shape[0], knot=generate_knots(cpts.shape[0],cpts.shape[0]-1),control_points=cpts,weights=weights)
    print(result)
    print(bern_to_nurbs_bezier(curve_3d_rat))
