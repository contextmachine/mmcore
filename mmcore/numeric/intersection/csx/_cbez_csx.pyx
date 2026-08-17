# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: infer_types=True

"""
Cythonized Newton solvers for Bézier curve-surface intersection.

Drop-in replacements for the pure-Python versions in ``_bez_csx3.py``.
All inner loops are stack-allocated, nogil, with inline 3×3 Cramer solves
to eliminate numpy overhead in the tight Newton iteration.
"""

cimport cython
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs, cos, acos
from libc.stdlib cimport malloc, free

cnp.import_array()

ctypedef cnp.float64_t f64

cdef enum:  # compile-time constant: a C 'const' variable is not a constant
    # expression, so const-sized stack arrays are VLAs — a GCC/Clang extension
    # MSVC rejects (C2057/C2466). An enum member is a true constant expression.
    _STACK_MAX = 32  # stack basis buffers up to this degree (fast for small splines)

# Contact type codes (must match _bez_csx3.py)
DEF CONTACT_ISOLATED = 0
DEF CONTACT_OVERLAP = 1
DEF CONTACT_AMBIGUOUS = 2

DEF _TWO_THIRDS_PI = 2.0943951023931953


# ===========================================================================
# Evaluation primitives (duplicated from _bern_homog.pyx – cdef inline
# cannot be cimported without embedding bodies in .pxd)
# ===========================================================================

cdef inline f64 _powi(f64 x, int n) noexcept nogil:
    cdef f64 res = 1.0
    cdef f64 base = x
    cdef int e = n
    while e > 0:
        if e & 1:
            res *= base
        base *= base
        e >>= 1
    return res


cdef inline void _bernstein_basis_fill(int n, f64 t, f64* out) noexcept nogil:
    cdef int i
    cdef f64 omt, r, b
    cdef f64 t2, t3, t4
    cdef f64 omt2, omt3, omt4

    if n < 0:
        return

    if n == 0:
        out[0] = 1.0
        return

    if t <= 0.0:
        out[0] = 1.0
        for i in range(1, n + 1):
            out[i] = 0.0
        return

    if t >= 1.0:
        for i in range(0, n):
            out[i] = 0.0
        out[n] = 1.0
        return

    omt = 1.0 - t

    if n == 1:
        out[0] = omt
        out[1] = t
        return

    if n == 2:
        omt2 = omt * omt
        t2 = t * t
        out[0] = omt2
        out[1] = 2.0 * t * omt
        out[2] = t2
        return

    if n == 3:
        omt2 = omt * omt
        omt3 = omt2 * omt
        t2 = t * t
        t3 = t2 * t
        out[0] = omt3
        out[1] = 3.0 * t * omt2
        out[2] = 3.0 * t2 * omt
        out[3] = t3
        return

    if n == 4:
        omt2 = omt * omt
        omt3 = omt2 * omt
        omt4 = omt3 * omt
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        out[0] = omt4
        out[1] = 4.0 * t * omt3
        out[2] = 6.0 * t2 * omt2
        out[3] = 4.0 * t3 * omt
        out[4] = t4
        return

    # General stable recurrence
    if t <= 0.5:
        b = _powi(omt, n)
        out[0] = b
        r = t / omt
        for i in range(0, n):
            b = b * (n - i) / (i + 1) * r
            out[i + 1] = b
    else:
        b = _powi(t, n)
        out[n] = b
        r = omt / t
        for i in range(n, 0, -1):
            b = b * i / (n - i + 1) * r
            out[i - 1] = b


cdef inline void _bernstein_basis_deriv_fill(int n, f64 t, f64* out) noexcept nogil:
    cdef int m, i
    cdef f64 omt, r
    cdef f64 prev, curr

    if n <= 0:
        out[0] = 0.0
        return

    if n == 1:
        out[0] = -1.0
        out[1] = 1.0
        return

    m = n - 1
    omt = 1.0 - t

    if t <= 0.5:
        prev = _powi(omt, m)
        out[0] = -n * prev
        r = t / omt if omt != 0.0 else 0.0
        for i in range(1, m + 1):
            curr = prev * (m - (i - 1)) / i * r
            out[i] = n * (prev - curr)
            prev = curr
        out[n] = n * prev
    else:
        prev = _powi(t, m)
        out[n] = n * prev
        r = omt / t if t != 0.0 else 0.0
        for i in range(m, 0, -1):
            curr = prev * i / (m - i + 1) * r
            out[i] = n * (curr - prev)
            prev = curr
        out[0] = -n * prev


# ---------------------------------------------------------------------------
# Curve evaluation helpers
# ---------------------------------------------------------------------------

cdef inline void _eval_curve_point(const f64[:, ::1] Pw, int n, int dh,
                                   f64 t, f64* out) noexcept nogil:
    cdef int i, i2
    cdef f64 omt, r, b
    cdef const f64* p

    if n <= 0:
        p = &Pw[0, 0]
        for i in range(dh):
            out[i] = p[i]
        return

    if t <= 0.0:
        p = &Pw[0, 0]
        for i in range(dh):
            out[i] = p[i]
        return

    if t >= 1.0:
        p = &Pw[n, 0]
        for i in range(dh):
            out[i] = p[i]
        return

    omt = 1.0 - t

    if t <= 0.5:
        b = _powi(omt, n)
        r = t / omt
        p = &Pw[0, 0]
        if dh == 4:
            out[0] = b * p[0]; out[1] = b * p[1]
            out[2] = b * p[2]; out[3] = b * p[3]
        elif dh == 3:
            out[0] = b * p[0]; out[1] = b * p[1]; out[2] = b * p[2]
        else:
            for i in range(dh):
                out[i] = b * p[i]
        for i in range(0, n):
            b = b * (n - i) / (i + 1) * r
            p = &Pw[i + 1, 0]
            if dh == 4:
                out[0] += b * p[0]; out[1] += b * p[1]
                out[2] += b * p[2]; out[3] += b * p[3]
            elif dh == 3:
                out[0] += b * p[0]; out[1] += b * p[1]; out[2] += b * p[2]
            else:
                for i2 in range(dh):
                    out[i2] += b * p[i2]
    else:
        b = _powi(t, n)
        r = omt / t
        p = &Pw[n, 0]
        if dh == 4:
            out[0] = b * p[0]; out[1] = b * p[1]
            out[2] = b * p[2]; out[3] = b * p[3]
        elif dh == 3:
            out[0] = b * p[0]; out[1] = b * p[1]; out[2] = b * p[2]
        else:
            for i in range(dh):
                out[i] = b * p[i]
        for i in range(n, 0, -1):
            b = b * i / (n - i + 1) * r
            p = &Pw[i - 1, 0]
            if dh == 4:
                out[0] += b * p[0]; out[1] += b * p[1]
                out[2] += b * p[2]; out[3] += b * p[3]
            elif dh == 3:
                out[0] += b * p[0]; out[1] += b * p[1]; out[2] += b * p[2]
            else:
                for i2 in range(dh):
                    out[i2] += b * p[i2]


cdef inline void _eval_curve_d1(const f64[:, ::1] Pw, int n, int dh,
                                f64 t, f64* out) noexcept nogil:
    cdef int m, i, k
    cdef f64 omt, r, b, scale
    cdef const f64* p0
    cdef const f64* p1

    if n <= 0:
        for k in range(dh):
            out[k] = 0.0
        return

    m = n - 1
    scale = <f64>n
    omt = 1.0 - t

    if t <= 0.5:
        b = _powi(omt, m)
        r = t / omt if omt != 0.0 else 0.0
        p0 = &Pw[0, 0]
        p1 = &Pw[1, 0]
        if dh == 4:
            out[0] = b * scale * (p1[0] - p0[0])
            out[1] = b * scale * (p1[1] - p0[1])
            out[2] = b * scale * (p1[2] - p0[2])
            out[3] = b * scale * (p1[3] - p0[3])
        elif dh == 3:
            out[0] = b * scale * (p1[0] - p0[0])
            out[1] = b * scale * (p1[1] - p0[1])
            out[2] = b * scale * (p1[2] - p0[2])
        else:
            for k in range(dh):
                out[k] = b * scale * (p1[k] - p0[k])
        for i in range(0, m):
            b = b * (m - i) / (i + 1) * r
            p0 = &Pw[i + 1, 0]
            p1 = &Pw[i + 2, 0]
            if dh == 4:
                out[0] += b * scale * (p1[0] - p0[0])
                out[1] += b * scale * (p1[1] - p0[1])
                out[2] += b * scale * (p1[2] - p0[2])
                out[3] += b * scale * (p1[3] - p0[3])
            elif dh == 3:
                out[0] += b * scale * (p1[0] - p0[0])
                out[1] += b * scale * (p1[1] - p0[1])
                out[2] += b * scale * (p1[2] - p0[2])
            else:
                for k in range(dh):
                    out[k] += b * scale * (p1[k] - p0[k])
    else:
        b = _powi(t, m)
        r = omt / t if t != 0.0 else 0.0
        p0 = &Pw[n - 1, 0]
        p1 = &Pw[n, 0]
        if dh == 4:
            out[0] = b * scale * (p1[0] - p0[0])
            out[1] = b * scale * (p1[1] - p0[1])
            out[2] = b * scale * (p1[2] - p0[2])
            out[3] = b * scale * (p1[3] - p0[3])
        elif dh == 3:
            out[0] = b * scale * (p1[0] - p0[0])
            out[1] = b * scale * (p1[1] - p0[1])
            out[2] = b * scale * (p1[2] - p0[2])
        else:
            for k in range(dh):
                out[k] = b * scale * (p1[k] - p0[k])
        for i in range(m, 0, -1):
            b = b * i / (m - i + 1) * r
            p0 = &Pw[i - 1, 0]
            p1 = &Pw[i, 0]
            if dh == 4:
                out[0] += b * scale * (p1[0] - p0[0])
                out[1] += b * scale * (p1[1] - p0[1])
                out[2] += b * scale * (p1[2] - p0[2])
                out[3] += b * scale * (p1[3] - p0[3])
            elif dh == 3:
                out[0] += b * scale * (p1[0] - p0[0])
                out[1] += b * scale * (p1[1] - p0[1])
                out[2] += b * scale * (p1[2] - p0[2])
            else:
                for k in range(dh):
                    out[k] += b * scale * (p1[k] - p0[k])


# ---------------------------------------------------------------------------
# Surface evaluation helpers
# ---------------------------------------------------------------------------

cdef inline void _eval_surface_point_and_derivs(
    const f64[:, :, ::1] Pw, int nu, int nv, int dh,
    f64 u, f64 v,
    f64* Sh, f64* Shu, f64* Shv,
    f64* Bu, f64* Bud, f64* Bv, f64* Bvd,
) noexcept nogil:
    """Evaluate surface point + first partial derivatives in homogeneous space."""
    cdef int i, j, k
    cdef f64 bu, bud_i, bv, bvd_j
    cdef f64 w0, wu, wv
    cdef const f64* p

    _bernstein_basis_fill(nu, u, Bu)
    _bernstein_basis_deriv_fill(nu, u, Bud)
    _bernstein_basis_fill(nv, v, Bv)
    _bernstein_basis_deriv_fill(nv, v, Bvd)

    for k in range(dh):
        Sh[k] = 0.0
        Shu[k] = 0.0
        Shv[k] = 0.0

    for i in range(nu + 1):
        bu = Bu[i]
        bud_i = Bud[i]
        for j in range(nv + 1):
            bv = Bv[j]
            bvd_j = Bvd[j]
            w0 = bu * bv
            wu = bud_i * bv
            wv = bu * bvd_j
            p = &Pw[i, j, 0]
            if dh == 4:
                Sh[0] += w0 * p[0]; Sh[1] += w0 * p[1]
                Sh[2] += w0 * p[2]; Sh[3] += w0 * p[3]
                Shu[0] += wu * p[0]; Shu[1] += wu * p[1]
                Shu[2] += wu * p[2]; Shu[3] += wu * p[3]
                Shv[0] += wv * p[0]; Shv[1] += wv * p[1]
                Shv[2] += wv * p[2]; Shv[3] += wv * p[3]
            elif dh == 3:
                Sh[0] += w0 * p[0]; Sh[1] += w0 * p[1]; Sh[2] += w0 * p[2]
                Shu[0] += wu * p[0]; Shu[1] += wu * p[1]; Shu[2] += wu * p[2]
                Shv[0] += wv * p[0]; Shv[1] += wv * p[1]; Shv[2] += wv * p[2]
            else:
                for k in range(dh):
                    Sh[k] += w0 * p[k]
                    Shu[k] += wu * p[k]
                    Shv[k] += wv * p[k]


cdef inline void _eval_surface_point_only(
    const f64[:, :, ::1] Pw, int nu, int nv, int dh,
    f64 u, f64 v,
    f64* Sh, f64* Bu, f64* Bv,
) noexcept nogil:
    """Evaluate surface point only (no derivatives), for line search."""
    cdef int i, j, k
    cdef f64 w0
    cdef const f64* p

    _bernstein_basis_fill(nu, u, Bu)
    _bernstein_basis_fill(nv, v, Bv)

    for k in range(dh):
        Sh[k] = 0.0

    for i in range(nu + 1):
        for j in range(nv + 1):
            w0 = Bu[i] * Bv[j]
            p = &Pw[i, j, 0]
            if dh == 4:
                Sh[0] += w0 * p[0]; Sh[1] += w0 * p[1]
                Sh[2] += w0 * p[2]; Sh[3] += w0 * p[3]
            elif dh == 3:
                Sh[0] += w0 * p[0]; Sh[1] += w0 * p[1]; Sh[2] += w0 * p[2]
            else:
                for k in range(dh):
                    Sh[k] += w0 * p[k]


# ===========================================================================
# Dehomogenization
# ===========================================================================

cdef inline void _dehomog_point(f64* Ph, int dh, f64* p) noexcept nogil:
    """Dehomogenize a single point: p = Ph[:-1] / Ph[-1]."""
    cdef f64 invw = 1.0 / Ph[dh - 1]
    cdef int d = dh - 1
    cdef int k
    for k in range(d):
        p[k] = Ph[k] * invw


cdef inline void _dehomog_curve_d1(f64* Ch, f64* Chd, int dh,
                                   f64* c, f64* ct) noexcept nogil:
    """Quotient rule for curve point + first derivative."""
    cdef int d = dh - 1
    cdef f64 w = Ch[dh - 1]
    cdef f64 wd = Chd[dh - 1]
    cdef f64 invw = 1.0 / w
    cdef f64 invw2 = invw * invw
    cdef int k
    for k in range(d):
        c[k] = Ch[k] * invw
        ct[k] = (w * Chd[k] - Ch[k] * wd) * invw2


cdef inline void _dehomog_surface_d1(f64* Sh, f64* Shu, f64* Shv, int dh,
                                     f64* s, f64* su, f64* sv) noexcept nogil:
    """Quotient rule for surface point + first partial derivatives."""
    cdef int d = dh - 1
    cdef f64 w = Sh[dh - 1]
    cdef f64 wu = Shu[dh - 1]
    cdef f64 wv = Shv[dh - 1]
    cdef f64 invw = 1.0 / w
    cdef f64 invw2 = invw * invw
    cdef int k
    for k in range(d):
        s[k] = Sh[k] * invw
        su[k] = (w * Shu[k] - Sh[k] * wu) * invw2
        sv[k] = (w * Shv[k] - Sh[k] * wv) * invw2


# ===========================================================================
# Inline solvers
# ===========================================================================

cdef inline int _solve3x3_cramer(f64* A, f64* b, f64* x) noexcept nogil:
    """Solve 3x3 system Ax = b via Cramer's rule.

    A is stored row-major: A[i*3+j].
    Returns 1 on success, 0 if singular.
    """
    cdef f64 a00 = A[0], a01 = A[1], a02 = A[2]
    cdef f64 a10 = A[3], a11 = A[4], a12 = A[5]
    cdef f64 a20 = A[6], a21 = A[7], a22 = A[8]

    cdef f64 det = (a00 * (a11 * a22 - a12 * a21)
                  - a01 * (a10 * a22 - a12 * a20)
                  + a02 * (a10 * a21 - a11 * a20))

    if fabs(det) < 1e-300:
        x[0] = 0.0; x[1] = 0.0; x[2] = 0.0
        return 0

    cdef f64 inv_det = 1.0 / det

    x[0] = ((b[0] * (a11 * a22 - a12 * a21)
           - a01 * (b[1] * a22 - a12 * b[2])
           + a02 * (b[1] * a21 - a11 * b[2])) * inv_det)

    x[1] = ((a00 * (b[1] * a22 - a12 * b[2])
           - b[0] * (a10 * a22 - a12 * a20)
           + a02 * (a10 * b[2] - b[1] * a20)) * inv_det)

    x[2] = ((a00 * (a11 * b[2] - b[1] * a21)
           - a01 * (a10 * b[2] - b[1] * a20)
           + b[0] * (a10 * a21 - a11 * a20)) * inv_det)

    return 1


cdef inline void _solve2x2_cramer(f64 a00, f64 a01, f64 a10, f64 a11,
                                  f64 b0, f64 b1,
                                  f64* x0, f64* x1) noexcept nogil:
    """Solve 2x2 system via Cramer's rule."""
    cdef f64 det = a00 * a11 - a01 * a10
    if fabs(det) < 1e-300:
        x0[0] = 0.0; x1[0] = 0.0
        return
    cdef f64 inv_det = 1.0 / det
    x0[0] = (b0 * a11 - a01 * b1) * inv_det
    x1[0] = (a00 * b1 - b0 * a10) * inv_det


cdef inline void _form_normal_equations_3x3(
    f64* J, f64* G, f64 damp,
    f64* JTJ, f64* JTG,
) noexcept nogil:
    """Form J^T @ J + damp*I and -J^T @ G for a 3x3 Jacobian.

    J is row-major 3x3: J[row*3+col].
    JTJ output is row-major 3x3.
    JTG output is 3-vector.
    """
    cdef f64 j00 = J[0], j01 = J[1], j02 = J[2]
    cdef f64 j10 = J[3], j11 = J[4], j12 = J[5]
    cdef f64 j20 = J[6], j21 = J[7], j22 = J[8]

    # J^T @ J  (symmetric, but store full for Cramer)
    JTJ[0] = j00*j00 + j10*j10 + j20*j20 + damp
    JTJ[1] = j00*j01 + j10*j11 + j20*j21
    JTJ[2] = j00*j02 + j10*j12 + j20*j22
    JTJ[3] = JTJ[1]
    JTJ[4] = j01*j01 + j11*j11 + j21*j21 + damp
    JTJ[5] = j01*j02 + j11*j12 + j21*j22
    JTJ[6] = JTJ[2]
    JTJ[7] = JTJ[5]
    JTJ[8] = j02*j02 + j12*j12 + j22*j22 + damp

    # -J^T @ G
    JTG[0] = -(j00*G[0] + j10*G[1] + j20*G[2])
    JTG[1] = -(j01*G[0] + j11*G[1] + j21*G[2])
    JTG[2] = -(j02*G[0] + j12*G[1] + j22*G[2])


# ===========================================================================
# Clamp helper
# ===========================================================================

cdef inline f64 _clamp01(f64 x) noexcept nogil:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


# ===========================================================================
# Newton solvers
# ===========================================================================

cpdef tuple c_G_and_J_curve_surface(
    const f64[:, ::1] C_ctrl,
    const f64[:, :, ::1] S_ctrl,
    f64 t, f64 u, f64 v,
    bint rational,
):
    """Evaluate G = C(t) - S(u,v) and Jacobian J = [Ct, -Su, -Sv]."""
    cdef int nc = C_ctrl.shape[0] - 1
    cdef int nu = S_ctrl.shape[0] - 1
    cdef int nv = S_ctrl.shape[1] - 1
    cdef int dh = C_ctrl.shape[1]

    cdef f64 Ch[5]
    cdef f64 Chd[5]
    cdef f64 Sh[5]
    cdef f64 Shu_h[5]
    cdef f64 Shv_h[5]

    cdef f64 c_pt[3]
    cdef f64 ct_arr[3]
    cdef f64 s_pt[3]
    cdef f64 su_arr[3]
    cdef f64 sv_arr[3]

    cdef f64 Bu_s[_STACK_MAX + 1]
    cdef f64 Bud_s[_STACK_MAX + 1]
    cdef f64 Bv_s[_STACK_MAX + 1]
    cdef f64 Bvd_s[_STACK_MAX + 1]

    cdef f64* Bu = Bu_s
    cdef f64* Bud = Bud_s
    cdef f64* Bv = Bv_s
    cdef f64* Bvd = Bvd_s

    cdef f64* uheap = NULL
    cdef f64* vheap = NULL

    if nu > _STACK_MAX:
        uheap = <f64*>malloc(2 * (nu + 1) * sizeof(f64))
        Bu = uheap
        Bud = uheap + (nu + 1)
    if nv > _STACK_MAX:
        vheap = <f64*>malloc(2 * (nv + 1) * sizeof(f64))
        Bv = vheap
        Bvd = vheap + (nv + 1)

    # Pre-allocate output arrays and get memoryviews BEFORE try
    G_out = np.empty(3, dtype=np.float64)
    J_out = np.empty((3, 3), dtype=np.float64)
    cdef f64[::1] G_mv = G_out
    cdef f64[:, ::1] J_mv = J_out

    try:
        _eval_curve_point(C_ctrl, nc, dh, t, Ch)
        _eval_curve_d1(C_ctrl, nc, dh, t, Chd)
        _eval_surface_point_and_derivs(S_ctrl, nu, nv, dh, u, v,
                                       Sh, Shu_h, Shv_h, Bu, Bud, Bv, Bvd)

        if rational:
            _dehomog_curve_d1(Ch, Chd, dh, c_pt, ct_arr)
            _dehomog_surface_d1(Sh, Shu_h, Shv_h, dh, s_pt, su_arr, sv_arr)
        else:
            c_pt[0] = Ch[0]; c_pt[1] = Ch[1]; c_pt[2] = Ch[2]
            ct_arr[0] = Chd[0]; ct_arr[1] = Chd[1]; ct_arr[2] = Chd[2]
            s_pt[0] = Sh[0]; s_pt[1] = Sh[1]; s_pt[2] = Sh[2]
            su_arr[0] = Shu_h[0]; su_arr[1] = Shu_h[1]; su_arr[2] = Shu_h[2]
            sv_arr[0] = Shv_h[0]; sv_arr[1] = Shv_h[1]; sv_arr[2] = Shv_h[2]

        G_mv[0] = c_pt[0] - s_pt[0]
        G_mv[1] = c_pt[1] - s_pt[1]
        G_mv[2] = c_pt[2] - s_pt[2]

        J_mv[0, 0] = ct_arr[0]; J_mv[0, 1] = -su_arr[0]; J_mv[0, 2] = -sv_arr[0]
        J_mv[1, 0] = ct_arr[1]; J_mv[1, 1] = -su_arr[1]; J_mv[1, 2] = -sv_arr[1]
        J_mv[2, 0] = ct_arr[2]; J_mv[2, 1] = -su_arr[2]; J_mv[2, 2] = -sv_arr[2]

        return G_out, J_out
    finally:
        if uheap != NULL:
            free(uheap)
        if vheap != NULL:
            free(vheap)


cpdef object c_G_only_curve_surface(
    const f64[:, ::1] C_ctrl,
    const f64[:, :, ::1] S_ctrl,
    f64 t, f64 u, f64 v,
    bint rational,
):
    """Evaluate G = C(t) - S(u,v) only (no Jacobian)."""
    cdef int nc = C_ctrl.shape[0] - 1
    cdef int nu = S_ctrl.shape[0] - 1
    cdef int nv = S_ctrl.shape[1] - 1
    cdef int dh = C_ctrl.shape[1]

    cdef f64 Ch[5]
    cdef f64 Sh[5]
    cdef f64 c_pt[3]
    cdef f64 s_pt[3]

    cdef f64 Bu_s[_STACK_MAX + 1]
    cdef f64 Bv_s[_STACK_MAX + 1]
    cdef f64* Bu = Bu_s
    cdef f64* Bv = Bv_s
    cdef f64* uheap = NULL
    cdef f64* vheap = NULL

    if nu > _STACK_MAX:
        uheap = <f64*>malloc((nu + 1) * sizeof(f64))
        Bu = uheap
    if nv > _STACK_MAX:
        vheap = <f64*>malloc((nv + 1) * sizeof(f64))
        Bv = vheap

    G_out = np.empty(3, dtype=np.float64)
    cdef f64[::1] G_mv = G_out

    try:
        _eval_curve_point(C_ctrl, nc, dh, t, Ch)
        _eval_surface_point_only(S_ctrl, nu, nv, dh, u, v, Sh, Bu, Bv)

        if rational:
            _dehomog_point(Ch, dh, c_pt)
            _dehomog_point(Sh, dh, s_pt)
        else:
            c_pt[0] = Ch[0]; c_pt[1] = Ch[1]; c_pt[2] = Ch[2]
            s_pt[0] = Sh[0]; s_pt[1] = Sh[1]; s_pt[2] = Sh[2]

        G_mv[0] = c_pt[0] - s_pt[0]
        G_mv[1] = c_pt[1] - s_pt[1]
        G_mv[2] = c_pt[2] - s_pt[2]
        return G_out
    finally:
        if uheap != NULL:
            free(uheap)
        if vheap != NULL:
            free(vheap)


cpdef tuple c_newton_project_G0_curve_surface(
    const f64[:, ::1] C_ctrl,
    const f64[:, :, ::1] S_ctrl,
    f64 t0, f64 u0, f64 v0,
    f64 tol=1e-12,
    int it=15,
    f64 lm_damp=1e-12,
    f64 step_tol=1e-9,
    f64 delta_tol=1e-10,
    bint rational=False,
):
    """Levenberg-Marquardt Newton corrector to G(t,u,v) = C(t) - S(u,v) = 0.

    Clamps parameters to [0,1]^3.  Returns (t, u, v, G, J).
    """
    cdef int nc = C_ctrl.shape[0] - 1
    cdef int nu = S_ctrl.shape[0] - 1
    cdef int nv = S_ctrl.shape[1] - 1
    cdef int dh = C_ctrl.shape[1]

    cdef f64 t = t0, u = u0, v = v0
    cdef f64 delta_tol_sq = delta_tol * delta_tol
    cdef f64 tol_sq = tol * tol

    cdef f64 Ch[5], Chd[5]
    cdef f64 Sh[5], Shu_buf[5], Shv_buf[5]

    cdef f64 c_pt[3], ct_v[3]
    cdef f64 s_pt[3], su_v[3], sv_v[3]
    cdef f64 G[3]

    cdef f64 J[9]
    cdef f64 JTJ[9], JTG[3], delta[3]

    cdef f64 Ch_ls[5], Sh_ls[5]
    cdef f64 c_ls[3], s_ls[3], dgj[3]

    cdef f64 Bu_s[_STACK_MAX + 1]
    cdef f64 Bud_s[_STACK_MAX + 1]
    cdef f64 Bv_s[_STACK_MAX + 1]
    cdef f64 Bvd_s[_STACK_MAX + 1]
    cdef f64* Bu = Bu_s
    cdef f64* Bud = Bud_s
    cdef f64* Bv = Bv_s
    cdef f64* Bvd = Bvd_s

    cdef f64* uheap = NULL
    cdef f64* vheap = NULL

    if nu > _STACK_MAX:
        uheap = <f64*>malloc(2 * (nu + 1) * sizeof(f64))
        if uheap == NULL:
            raise MemoryError()
        Bu = uheap
        Bud = uheap + (nu + 1)
    if nv > _STACK_MAX:
        vheap = <f64*>malloc(2 * (nv + 1) * sizeof(f64))
        if vheap == NULL:
            if uheap != NULL:
                free(uheap)
            raise MemoryError()
        Bv = vheap
        Bvd = vheap + (nv + 1)

    cdef f64 prev_g2 = 1e300
    cdef int stall_count = 0
    cdef f64 g2, step, tn, un, vn, dgj_sq
    cdef int iter_i, ls_i

    # Pre-allocate output arrays
    G_out = np.empty(3, dtype=np.float64)
    J_out = np.empty((3, 3), dtype=np.float64)
    cdef f64[::1] G_mv = G_out
    cdef f64[:, ::1] J_mv = J_out

    try:
        for iter_i in range(it):
            _eval_curve_point(C_ctrl, nc, dh, t, Ch)
            _eval_curve_d1(C_ctrl, nc, dh, t, Chd)
            _eval_surface_point_and_derivs(S_ctrl, nu, nv, dh, u, v,
                                           Sh, Shu_buf, Shv_buf, Bu, Bud, Bv, Bvd)

            if rational:
                _dehomog_curve_d1(Ch, Chd, dh, c_pt, ct_v)
                _dehomog_surface_d1(Sh, Shu_buf, Shv_buf, dh, s_pt, su_v, sv_v)
            else:
                c_pt[0] = Ch[0]; c_pt[1] = Ch[1]; c_pt[2] = Ch[2]
                ct_v[0] = Chd[0]; ct_v[1] = Chd[1]; ct_v[2] = Chd[2]
                s_pt[0] = Sh[0]; s_pt[1] = Sh[1]; s_pt[2] = Sh[2]
                su_v[0] = Shu_buf[0]; su_v[1] = Shu_buf[1]; su_v[2] = Shu_buf[2]
                sv_v[0] = Shv_buf[0]; sv_v[1] = Shv_buf[1]; sv_v[2] = Shv_buf[2]

            G[0] = c_pt[0] - s_pt[0]
            G[1] = c_pt[1] - s_pt[1]
            G[2] = c_pt[2] - s_pt[2]

            J[0] = ct_v[0]; J[1] = -su_v[0]; J[2] = -sv_v[0]
            J[3] = ct_v[1]; J[4] = -su_v[1]; J[5] = -sv_v[1]
            J[6] = ct_v[2]; J[7] = -su_v[2]; J[8] = -sv_v[2]

            _form_normal_equations_3x3(J, G, lm_damp, JTJ, JTG)
            _solve3x3_cramer(JTJ, JTG, delta)

            g2 = G[0]*G[0] + G[1]*G[1] + G[2]*G[2]
            if g2 > 0.9 * prev_g2:
                stall_count += 1
                if stall_count > 2:
                    break
            else:
                stall_count = 0
            prev_g2 = g2

            step = 1.0
            for ls_i in range(8):
                tn = _clamp01(t + step * delta[0])
                un = _clamp01(u + step * delta[1])
                vn = _clamp01(v + step * delta[2])

                _eval_curve_point(C_ctrl, nc, dh, tn, Ch_ls)
                _eval_surface_point_only(S_ctrl, nu, nv, dh, un, vn, Sh_ls, Bu, Bv)

                if rational:
                    _dehomog_point(Ch_ls, dh, c_ls)
                    _dehomog_point(Sh_ls, dh, s_ls)
                else:
                    c_ls[0] = Ch_ls[0]; c_ls[1] = Ch_ls[1]; c_ls[2] = Ch_ls[2]
                    s_ls[0] = Sh_ls[0]; s_ls[1] = Sh_ls[1]; s_ls[2] = Sh_ls[2]

                dgj[0] = c_ls[0] - s_ls[0]
                dgj[1] = c_ls[1] - s_ls[1]
                dgj[2] = c_ls[2] - s_ls[2]
                dgj_sq = dgj[0]*dgj[0] + dgj[1]*dgj[1] + dgj[2]*dgj[2]

                if dgj_sq <= g2:
                    t = tn; u = un; v = vn
                    break
                step *= 0.5

            if g2 < tol_sq:
                break
            if step < step_tol and (delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]) < delta_tol_sq:
                break

        # Final evaluation for return
        _eval_curve_point(C_ctrl, nc, dh, t, Ch)
        _eval_curve_d1(C_ctrl, nc, dh, t, Chd)
        _eval_surface_point_and_derivs(S_ctrl, nu, nv, dh, u, v,
                                       Sh, Shu_buf, Shv_buf, Bu, Bud, Bv, Bvd)
        if rational:
            _dehomog_curve_d1(Ch, Chd, dh, c_pt, ct_v)
            _dehomog_surface_d1(Sh, Shu_buf, Shv_buf, dh, s_pt, su_v, sv_v)
        else:
            c_pt[0] = Ch[0]; c_pt[1] = Ch[1]; c_pt[2] = Ch[2]
            ct_v[0] = Chd[0]; ct_v[1] = Chd[1]; ct_v[2] = Chd[2]
            s_pt[0] = Sh[0]; s_pt[1] = Sh[1]; s_pt[2] = Sh[2]
            su_v[0] = Shu_buf[0]; su_v[1] = Shu_buf[1]; su_v[2] = Shu_buf[2]
            sv_v[0] = Shv_buf[0]; sv_v[1] = Shv_buf[1]; sv_v[2] = Shv_buf[2]

        G_mv[0] = c_pt[0] - s_pt[0]
        G_mv[1] = c_pt[1] - s_pt[1]
        G_mv[2] = c_pt[2] - s_pt[2]

        J_mv[0, 0] = ct_v[0]; J_mv[0, 1] = -su_v[0]; J_mv[0, 2] = -sv_v[0]
        J_mv[1, 0] = ct_v[1]; J_mv[1, 1] = -su_v[1]; J_mv[1, 2] = -sv_v[1]
        J_mv[2, 0] = ct_v[2]; J_mv[2, 1] = -su_v[2]; J_mv[2, 2] = -sv_v[2]

        return t, u, v, G_out, J_out
    finally:
        if uheap != NULL:
            free(uheap)
        if vheap != NULL:
            free(vheap)


cpdef tuple c_project_G0_fixed_t(
    const f64[:, ::1] C_ctrl,
    const f64[:, :, ::1] S_ctrl,
    f64 t_fixed,
    f64 u0, f64 v0,
    f64 tol=1e-12,
    int it=30,
    f64 lm_damp=1e-12,
    bint rational=False,
):
    """Solve min ||C(t_fixed) - S(u,v)|| with (u,v) in [0,1]^2.

    Returns (u, v, G_residual, converged).
    """
    cdef int nc = C_ctrl.shape[0] - 1
    cdef int nu = S_ctrl.shape[0] - 1
    cdef int nv = S_ctrl.shape[1] - 1
    cdef int dh = C_ctrl.shape[1]

    cdef f64 sq_tol = tol * tol

    cdef f64 Ch[5]
    cdef f64 p1[3]

    _eval_curve_point(C_ctrl, nc, dh, t_fixed, Ch)
    if rational:
        _dehomog_point(Ch, dh, p1)
    else:
        p1[0] = Ch[0]; p1[1] = Ch[1]; p1[2] = Ch[2]

    cdef f64 u = u0, v = v0

    cdef f64 Sh[5], Shu_buf[5], Shv_buf[5]
    cdef f64 s_pt[3], su_v[3], sv_v[3]
    cdef f64 G_arr[3]
    cdef f64 Sh_ls[5], s_ls[3], dgj[3]

    cdef f64 Bu_s[_STACK_MAX + 1]
    cdef f64 Bud_s[_STACK_MAX + 1]
    cdef f64 Bv_s[_STACK_MAX + 1]
    cdef f64 Bvd_s[_STACK_MAX + 1]
    cdef f64* Bu = Bu_s
    cdef f64* Bud = Bud_s
    cdef f64* Bv = Bv_s
    cdef f64* Bvd = Bvd_s

    cdef f64* uheap = NULL
    cdef f64* vheap = NULL

    if nu > _STACK_MAX:
        uheap = <f64*>malloc(2 * (nu + 1) * sizeof(f64))
        if uheap == NULL:
            raise MemoryError()
        Bu = uheap
        Bud = uheap + (nu + 1)
    if nv > _STACK_MAX:
        vheap = <f64*>malloc(2 * (nv + 1) * sizeof(f64))
        if vheap == NULL:
            if uheap != NULL:
                free(uheap)
            raise MemoryError()
        Bv = vheap
        Bvd = vheap + (nv + 1)

    cdef f64 jtj00, jtj01, jtj10, jtj11
    cdef f64 jtg0, jtg1
    cdef f64 du, dv
    cdef f64 step, g0, un, vn, dgj_sq
    cdef int iter_i

    # Pre-allocate output array
    G_out = np.empty(3, dtype=np.float64)
    cdef f64[::1] G_mv = G_out

    try:
        for iter_i in range(it):
            _eval_surface_point_and_derivs(S_ctrl, nu, nv, dh, u, v,
                                           Sh, Shu_buf, Shv_buf, Bu, Bud, Bv, Bvd)
            if rational:
                _dehomog_surface_d1(Sh, Shu_buf, Shv_buf, dh, s_pt, su_v, sv_v)
            else:
                s_pt[0] = Sh[0]; s_pt[1] = Sh[1]; s_pt[2] = Sh[2]
                su_v[0] = Shu_buf[0]; su_v[1] = Shu_buf[1]; su_v[2] = Shu_buf[2]
                sv_v[0] = Shv_buf[0]; sv_v[1] = Shv_buf[1]; sv_v[2] = Shv_buf[2]

            G_arr[0] = s_pt[0] - p1[0]
            G_arr[1] = s_pt[1] - p1[1]
            G_arr[2] = s_pt[2] - p1[2]

            jtj00 = su_v[0]*su_v[0] + su_v[1]*su_v[1] + su_v[2]*su_v[2] + lm_damp
            jtj01 = su_v[0]*sv_v[0] + su_v[1]*sv_v[1] + su_v[2]*sv_v[2]
            jtj10 = jtj01
            jtj11 = sv_v[0]*sv_v[0] + sv_v[1]*sv_v[1] + sv_v[2]*sv_v[2] + lm_damp

            jtg0 = -(su_v[0]*G_arr[0] + su_v[1]*G_arr[1] + su_v[2]*G_arr[2])
            jtg1 = -(sv_v[0]*G_arr[0] + sv_v[1]*G_arr[1] + sv_v[2]*G_arr[2])

            _solve2x2_cramer(jtj00, jtj01, jtj10, jtj11, jtg0, jtg1, &du, &dv)

            step = 1.0
            g0 = G_arr[0]*G_arr[0] + G_arr[1]*G_arr[1] + G_arr[2]*G_arr[2]

            while step > 1e-6:
                un = _clamp01(u + step * du)
                vn = _clamp01(v + step * dv)

                _eval_surface_point_only(S_ctrl, nu, nv, dh, un, vn, Sh_ls, Bu, Bv)
                if rational:
                    _dehomog_point(Sh_ls, dh, s_ls)
                else:
                    s_ls[0] = Sh_ls[0]; s_ls[1] = Sh_ls[1]; s_ls[2] = Sh_ls[2]

                dgj[0] = s_ls[0] - p1[0]
                dgj[1] = s_ls[1] - p1[1]
                dgj[2] = s_ls[2] - p1[2]
                dgj_sq = dgj[0]*dgj[0] + dgj[1]*dgj[1] + dgj[2]*dgj[2]

                if dgj_sq <= g0 + 1e-18:
                    u = un; v = vn
                    break
                step *= 0.5

            # Check convergence
            _eval_surface_point_only(S_ctrl, nu, nv, dh, u, v, Sh_ls, Bu, Bv)
            if rational:
                _dehomog_point(Sh_ls, dh, s_ls)
            else:
                s_ls[0] = Sh_ls[0]; s_ls[1] = Sh_ls[1]; s_ls[2] = Sh_ls[2]

            dgj[0] = s_ls[0] - p1[0]
            dgj[1] = s_ls[1] - p1[1]
            dgj[2] = s_ls[2] - p1[2]
            dgj_sq = dgj[0]*dgj[0] + dgj[1]*dgj[1] + dgj[2]*dgj[2]

            if dgj_sq < sq_tol:
                G_mv[0] = dgj[0]; G_mv[1] = dgj[1]; G_mv[2] = dgj[2]
                return u, v, G_out, True

        # Not converged — compute final residual
        _eval_surface_point_only(S_ctrl, nu, nv, dh, u, v, Sh_ls, Bu, Bv)
        if rational:
            _dehomog_point(Sh_ls, dh, s_ls)
        else:
            s_ls[0] = Sh_ls[0]; s_ls[1] = Sh_ls[1]; s_ls[2] = Sh_ls[2]

        dgj[0] = s_ls[0] - p1[0]
        dgj[1] = s_ls[1] - p1[1]
        dgj[2] = s_ls[2] - p1[2]
        dgj_sq = dgj[0]*dgj[0] + dgj[1]*dgj[1] + dgj[2]*dgj[2]

        G_mv[0] = dgj[0]; G_mv[1] = dgj[1]; G_mv[2] = dgj[2]
        return u, v, G_out, (sqrt(dgj_sq) < 5.0 * tol)
    finally:
        if uheap != NULL:
            free(uheap)
        if vheap != NULL:
            free(vheap)


# ===========================================================================
# SVD / Classification
# ===========================================================================

cdef inline void _svd3_singular_values(f64* J, f64* s_max, f64* s_mid, f64* s_min) noexcept nogil:
    """Singular values of a 3x3 matrix via eigenvalues of J^T @ J.

    Uses Smith (1961) / Kopp (2006) for eigenvalues of symmetric 3x3 PSD.
    J is row-major.
    """
    cdef f64 j00 = J[0], j10 = J[3], j20 = J[6]
    cdef f64 j01 = J[1], j11 = J[4], j21 = J[7]
    cdef f64 j02 = J[2], j12 = J[5], j22 = J[8]

    # A = J^T @ J  (symmetric)
    cdef f64 a00 = j00*j00 + j10*j10 + j20*j20
    cdef f64 a01 = j00*j01 + j10*j11 + j20*j21
    cdef f64 a02 = j00*j02 + j10*j12 + j20*j22
    cdef f64 a11 = j01*j01 + j11*j11 + j21*j21
    cdef f64 a12 = j01*j02 + j11*j12 + j21*j22
    cdef f64 a22 = j02*j02 + j12*j12 + j22*j22

    cdef f64 p1 = a01*a01 + a02*a02 + a12*a12
    cdef f64 q = (a00 + a11 + a22) / 3.0

    cdef f64 e0, e1, e2
    cdef f64 p2, p, inv_p
    cdef f64 b00, b11, b22, b01, b02, b12
    cdef f64 r, phi, two_p

    if p1 < 1e-30:
        e0 = a00; e1 = a11; e2 = a22
    else:
        p2 = (a00 - q)*(a00 - q) + (a11 - q)*(a11 - q) + (a22 - q)*(a22 - q) + 2.0*p1
        p = sqrt(p2 / 6.0)
        inv_p = 1.0 / p
        b00 = (a00 - q) * inv_p
        b11 = (a11 - q) * inv_p
        b22 = (a22 - q) * inv_p
        b01 = a01 * inv_p
        b02 = a02 * inv_p
        b12 = a12 * inv_p

        r = 0.5 * (b00*(b11*b22 - b12*b12)
                  - b01*(b01*b22 - b12*b02)
                  + b02*(b01*b12 - b11*b02))

        if r >= 1.0:
            phi = 0.0
        elif r <= -1.0:
            phi = _TWO_THIRDS_PI
        else:
            phi = acos(r) / 3.0

        two_p = 2.0 * p
        e0 = q + two_p * cos(phi)
        e2 = q + two_p * cos(phi + _TWO_THIRDS_PI)
        e1 = 3.0 * q - e0 - e2

    # Clamp to non-negative
    if e0 < 0.0:
        e0 = 0.0
    if e1 < 0.0:
        e1 = 0.0
    if e2 < 0.0:
        e2 = 0.0

    # Sort descending
    if e0 < e1:
        e0, e1 = e1, e0
    if e0 < e2:
        e0, e2 = e2, e0
    if e1 < e2:
        e1, e2 = e2, e1

    s_max[0] = sqrt(e0)
    s_mid[0] = sqrt(e1)
    s_min[0] = sqrt(e2)


cpdef tuple c_classify_contact_curve_surface(object J_arr, f64 sv_thresh=1e-8, object rel_thresh_obj=None):
    """Returns (type_code, s_max, s_mid, s_min).

    type_code: 0=ISOLATED, 1=OVERLAP, 2=AMBIGUOUS.
    """
    cdef cnp.ndarray[f64, ndim=2] J_np = np.asarray(J_arr, dtype=np.float64, order='C')
    cdef f64[:, ::1] J_mv = J_np

    cdef f64 J_buf[9]
    cdef int i, j
    for i in range(3):
        for j in range(3):
            J_buf[i * 3 + j] = J_mv[i, j]

    cdef f64 sm, sd, sn
    _svd3_singular_values(J_buf, &sm, &sd, &sn)

    cdef bint has_rel = (rel_thresh_obj is not None)
    cdef f64 rel_thresh = 0.0
    if has_rel:
        rel_thresh = <f64>rel_thresh_obj

    cdef bint mid_abs_ok = sd > 1e-10
    cdef f64 guard = 1e-12 * sm
    if sv_thresh * 1e-2 > guard:
        guard = sv_thresh * 1e-2
    cdef bint mid_rel_ok = sd > guard

    cdef bint overlap = (sn < sv_thresh) and (mid_abs_ok or mid_rel_ok)

    if (not overlap) and has_rel:
        if sm > 0.0 and mid_rel_ok:
            if (sn / sm) < rel_thresh:
                overlap = True

    if overlap:
        return CONTACT_OVERLAP, sm, sd, sn
    if sn >= sv_thresh and ((not has_rel) or (sn / sm) >= rel_thresh):
        return CONTACT_ISOLATED, sm, sd, sn
    return CONTACT_AMBIGUOUS, sm, sd, sn


cpdef bint c_overlap_like_svd(object J_arr, f64 sv_thresh=1e-8, f64 rel_thresh=1e-6):
    """Returns True if J indicates an overlap-like contact."""
    cdef cnp.ndarray[f64, ndim=2] J_np = np.asarray(J_arr, dtype=np.float64, order='C')
    cdef f64[:, ::1] J_mv = J_np

    cdef f64 J_buf[9]
    cdef int i, j
    for i in range(3):
        for j in range(3):
            J_buf[i * 3 + j] = J_mv[i, j]

    cdef f64 sm, sd, sn
    _svd3_singular_values(J_buf, &sm, &sd, &sn)

    if sm <= 0.0:
        return False
    cdef f64 guard = 1e-12 * sm
    if sv_thresh * 1e-2 > guard:
        guard = sv_thresh * 1e-2
    if sd <= guard:
        return False
    return (sn < sv_thresh) or (sn / sm < rel_thresh)
