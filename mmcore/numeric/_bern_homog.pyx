# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: infer_types=True

cimport cython

from functools import lru_cache

import numpy as np
cimport numpy as cnp


from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free

cnp.import_array()

ctypedef cnp.float64_t f64

cdef enum:  # compile-time constant: a C 'const' variable is not a constant
    # expression, so const-sized stack arrays are VLAs — a GCC/Clang extension
    # MSVC rejects (C2057/C2466). An enum member is a true constant expression.
    _STACK_MAX = 32  # stack basis buffers up to this degree (fast for small splines)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

cdef inline f64 _powi(f64 x, int n) noexcept nogil:
    """Integer power by exponentiation-by-squaring (fast, no libm pow)."""
    cdef f64 res = 1.0
    cdef f64 base = x
    cdef int e = n
    while e > 0:
        if e & 1:
            res *= base
        base *= base
        e >>= 1
    return res


cdef inline void _set_zero(f64* out, int n) noexcept nogil:
    cdef int i
    for i in range(n):
        out[i] = 0.0


# ---------------------------------------------------------------------------
# Bernstein basis fill (no allocations)
# ---------------------------------------------------------------------------

cdef inline void _bernstein_basis_fill(int n, f64 t, f64* out) noexcept nogil:
    """
    Fill out[0..n] with Bernstein basis B_i^n(t).
    Uses unrolled fast paths for n<=4 and stable recurrence otherwise.
    """
    cdef int i
    cdef f64 omt, r, b
    cdef f64 t2, t3, t4
    cdef f64 omt2, omt3, omt4

    if n < 0:
        # can't raise in nogil; caller checks
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

    # Unrolled small degrees (very common for splines)
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
    """
    Fill out[0..n] with first derivative d/dt B_i^n(t).
    Streaming computation without allocating B^{n-1}.
    """
    cdef int m, i
    cdef f64 omt, r
    cdef f64 prev, curr

    if n <= 0:
        out[0] = 0.0
        return

    if n == 1:
        out[0] = -1.0
        out[1] =  1.0
        return

    m = n - 1
    omt = 1.0 - t

    if t <= 0.5:
        prev = _powi(omt, m)         # b0
        out[0] = -n * prev
        r = t / omt if omt != 0.0 else 0.0
        for i in range(1, m + 1):
            curr = prev * (m - (i - 1)) / i * r
            out[i] = n * (prev - curr)
            prev = curr
        out[n] = n * prev
    else:
        prev = _powi(t, m)           # b_m
        out[n] = n * prev
        r = omt / t if t != 0.0 else 0.0
        for i in range(m, 0, -1):
            curr = prev * i / (m - i + 1) * r   # b_{i-1}
            out[i] = n * (curr - prev)          # Bd[i] = n*(b_{i-1} - b_i)
            prev = curr
        out[0] = -n * prev


cdef inline void _bernstein_basis_2nd_fill(int n, f64 t, f64* out) noexcept nogil:
    """
    Fill out[0..n] with second derivative d2/dt2 B_i^n(t).
    Streaming computation without allocating B^{n-2}.
    """
    cdef int m, i, k
    cdef f64 factor
    cdef f64 omt, r
    cdef f64 b0, b1, b
    cdef f64 bim2, bim1
    cdef f64 hi1, hi2, curr

    if n <= 1:
        for i in range(0, n + 1):
            out[i] = 0.0
        return

    factor = (<f64>n) * (<f64>(n - 1))
    m = n - 2
    omt = 1.0 - t

    if t <= 0.5:
        b0 = _powi(omt, m)
        out[0] = factor * b0

        if m == 0:
            # n==2 case
            out[1] = factor * (-2.0 * b0)
            out[2] = factor * b0
            return

        r = t / omt if omt != 0.0 else 0.0
        b1 = b0 * m * r
        out[1] = factor * (b1 - 2.0 * b0)

        bim2 = b0
        bim1 = b1
        for i in range(2, m + 1):
            b = bim1 * (m - (i - 1)) / i * r
            out[i] = factor * (bim2 - 2.0 * bim1 + b)
            bim2 = bim1
            bim1 = b

        out[n - 1] = factor * (bim2 - 2.0 * bim1)
        out[n]     = factor * bim1
    else:
        # Reverse streaming; fills indices 2..n via out[k+2]
        r = omt / t if t != 0.0 else 0.0
        curr = _powi(t, m)   # b_m

        hi1 = 0.0  # b_{k+1}
        hi2 = 0.0  # b_{k+2}

        k = m
        while True:
            out[k + 2] = factor * (curr - 2.0 * hi1 + hi2)
            hi2 = hi1
            hi1 = curr
            if k == 0:
                break
            curr = curr * k / (m - k + 1) * r   # next lower basis value
            k -= 1

        # now hi1=b0, hi2=b1
        out[1] = factor * (hi2 - 2.0 * hi1)
        out[0] = factor * hi1


# ---------------------------------------------------------------------------
# Public Bernstein API
#   - *_inplace: fastest (no allocations)
#   - *_fast: allocate once and fill (no caching)
#   - cached bernstein_basis / deriv / 2nd: matches your original API (read-only)
# ---------------------------------------------------------------------------

cpdef void bernstein_basis_inplace(int n, f64 t, f64[::1] out):
    if n < 0:
        raise ValueError("n must be >= 0")
    if out.shape[0] != n + 1:
        raise ValueError("out must have length n+1")
    with nogil:
        _bernstein_basis_fill(n, t, &out[0])


cpdef void bernstein_basis_deriv_inplace(int n, f64 t, f64[::1] out):
    if n < 0:
        raise ValueError("n must be >= 0")
    if out.shape[0] != n + 1:
        raise ValueError("out must have length n+1")
    with nogil:
        _bernstein_basis_deriv_fill(n, t, &out[0])


cpdef void bernstein_basis_2nd_inplace(int n, f64 t, f64[::1] out):
    if n < 0:
        raise ValueError("n must be >= 0")
    if out.shape[0] != n + 1:
        raise ValueError("out must have length n+1")
    with nogil:
        _bernstein_basis_2nd_fill(n, t, &out[0])


cpdef cnp.ndarray[f64, ndim=1] bernstein_basis_fast(int n, f64 t):
    cdef cnp.ndarray[f64, ndim=1] arr
    cdef f64[::1] mv
    if n < 0:
        raise ValueError("n must be >= 0")
    arr = np.empty(n + 1, dtype=np.float64)
    mv = arr
    with nogil:
        _bernstein_basis_fill(n, t, &mv[0])
    return arr


cpdef cnp.ndarray[f64, ndim=1] bernstein_basis_deriv_fast(int n, f64 t):
    cdef cnp.ndarray[f64, ndim=1] arr
    cdef f64[::1] mv
    if n < 0:
        raise ValueError("n must be >= 0")
    arr = np.empty(n + 1, dtype=np.float64)
    mv = arr
    with nogil:
        _bernstein_basis_deriv_fill(n, t, &mv[0])
    return arr


cpdef cnp.ndarray[f64, ndim=1] bernstein_basis_2nd_fast(int n, f64 t):
    cdef cnp.ndarray[f64, ndim=1] arr
    cdef f64[::1] mv
    if n < 0:
        raise ValueError("n must be >= 0")
    arr = np.empty(n + 1, dtype=np.float64)
    mv = arr
    with nogil:
        _bernstein_basis_2nd_fill(n, t, &mv[0])
    return arr


@lru_cache(maxsize=8192, typed=False)
def bernstein_basis(int n, float t):
    """Cached, returns read-only float64 array."""
    cdef cnp.ndarray[f64, ndim=1] arr = bernstein_basis_fast(n, <f64>t)
    arr.flags.writeable = False
    return arr


@lru_cache(maxsize=8192, typed=False)
def bernstein_basis_deriv(int n, float t):
    """Cached, returns read-only float64 array."""
    cdef cnp.ndarray[f64, ndim=1] arr = bernstein_basis_deriv_fast(n, <f64>t)
    arr.flags.writeable = False
    return arr


@lru_cache(maxsize=8192, typed=False)
def bernstein_basis_2nd(int n, float t):
    """Cached, returns read-only float64 array."""
    cdef cnp.ndarray[f64, ndim=1] arr = bernstein_basis_2nd_fast(n, <f64>t)
    arr.flags.writeable = False
    return arr


def bernstein_row(int n, float t):
    return bernstein_basis(n, t)


# ---------------------------------------------------------------------------
# Internal: fast Bézier curve evaluation (no Bernstein arrays, no NumPy dot)
# ---------------------------------------------------------------------------

cdef inline void _eval_curve_point(const f64[:, ::1] Pw, int n, int dh, f64 t, f64* out) noexcept nogil:
    cdef int i
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
            out[0] = b * p[0]
            out[1] = b * p[1]
            out[2] = b * p[2]
            out[3] = b * p[3]
        else:
            for i in range(dh):
                out[i] = b * p[i]

        for i in range(0, n):
            b = b * (n - i) / (i + 1) * r
            p = &Pw[i + 1, 0]
            if dh == 4:
                out[0] += b * p[0]
                out[1] += b * p[1]
                out[2] += b * p[2]
                out[3] += b * p[3]
            else:
                for i2 in range(dh):
                    out[i2] += b * p[i2]
    else:
        b = _powi(t, n)
        r = omt / t
        p = &Pw[n, 0]
        if dh == 4:
            out[0] = b * p[0]
            out[1] = b * p[1]
            out[2] = b * p[2]
            out[3] = b * p[3]
        else:
            for i in range(dh):
                out[i] = b * p[i]

        for i in range(n, 0, -1):
            b = b * i / (n - i + 1) * r
            p = &Pw[i - 1, 0]
            if dh == 4:
                out[0] += b * p[0]
                out[1] += b * p[1]
                out[2] += b * p[2]
                out[3] += b * p[3]
            else:
                for i2 in range(dh):
                    out[i2] += b * p[i2]


cdef inline void _eval_curve_d1(const f64[:, ::1] Pw, int n, int dh, f64 t, f64* out) noexcept nogil:
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
            else:
                for k in range(dh):
                    out[k] += b * scale * (p1[k] - p0[k])


cdef inline void _eval_curve_d2(const f64[:, ::1] Pw, int n, int dh, f64 t, f64* out) noexcept nogil:
    cdef int m, i, k
    cdef f64 omt, r, b, scale
    cdef const f64* p0
    cdef const f64* p1
    cdef const f64* p2

    if n <= 1:
        for k in range(dh):
            out[k] = 0.0
        return

    m = n - 2
    scale = (<f64>n) * (<f64>(n - 1))
    omt = 1.0 - t

    if t <= 0.5:
        b = _powi(omt, m)
        r = t / omt if omt != 0.0 else 0.0

        p0 = &Pw[0, 0]
        p1 = &Pw[1, 0]
        p2 = &Pw[2, 0]
        if dh == 4:
            out[0] = b * scale * (p2[0] - 2.0 * p1[0] + p0[0])
            out[1] = b * scale * (p2[1] - 2.0 * p1[1] + p0[1])
            out[2] = b * scale * (p2[2] - 2.0 * p1[2] + p0[2])
            out[3] = b * scale * (p2[3] - 2.0 * p1[3] + p0[3])
        else:
            for k in range(dh):
                out[k] = b * scale * (p2[k] - 2.0 * p1[k] + p0[k])

        for i in range(0, m):
            b = b * (m - i) / (i + 1) * r
            p0 = &Pw[i + 1, 0]
            p1 = &Pw[i + 2, 0]
            p2 = &Pw[i + 3, 0]
            if dh == 4:
                out[0] += b * scale * (p2[0] - 2.0 * p1[0] + p0[0])
                out[1] += b * scale * (p2[1] - 2.0 * p1[1] + p0[1])
                out[2] += b * scale * (p2[2] - 2.0 * p1[2] + p0[2])
                out[3] += b * scale * (p2[3] - 2.0 * p1[3] + p0[3])
            else:
                for k in range(dh):
                    out[k] += b * scale * (p2[k] - 2.0 * p1[k] + p0[k])
    else:
        b = _powi(t, m)
        r = omt / t if t != 0.0 else 0.0

        p0 = &Pw[n - 2, 0]
        p1 = &Pw[n - 1, 0]
        p2 = &Pw[n, 0]
        if dh == 4:
            out[0] = b * scale * (p2[0] - 2.0 * p1[0] + p0[0])
            out[1] = b * scale * (p2[1] - 2.0 * p1[1] + p0[1])
            out[2] = b * scale * (p2[2] - 2.0 * p1[2] + p0[2])
            out[3] = b * scale * (p2[3] - 2.0 * p1[3] + p0[3])
        else:
            for k in range(dh):
                out[k] = b * scale * (p2[k] - 2.0 * p1[k] + p0[k])

        for i in range(m, 0, -1):
            b = b * i / (m - i + 1) * r
            p0 = &Pw[i - 1, 0]
            p1 = &Pw[i, 0]
            p2 = &Pw[i + 1, 0]
            if dh == 4:
                out[0] += b * scale * (p2[0] - 2.0 * p1[0] + p0[0])
                out[1] += b * scale * (p2[1] - 2.0 * p1[1] + p0[1])
                out[2] += b * scale * (p2[2] - 2.0 * p1[2] + p0[2])
                out[3] += b * scale * (p2[3] - 2.0 * p1[3] + p0[3])
            else:
                for k in range(dh):
                    out[k] += b * scale * (p2[k] - 2.0 * p1[k] + p0[k])


# ---------------------------------------------------------------------------
# Public: homogeneous curve evaluation
# ---------------------------------------------------------------------------


cpdef void eval_bezier_homogeneous_curve_inplace(const f64[:, ::1] Pw, f64 t, f64[::1] out):
    cdef int n = Pw.shape[0] - 1
    cdef int dh = Pw.shape[1]
    if out.shape[0] != dh:
        raise ValueError("out must have length equal to Pw.shape[1]")
    with nogil:
        _eval_curve_point(Pw, n, dh, t, &out[0])


def eval_bezier_homogeneous_curve(Pw, double t):
    """
    Convenience wrapper (accepts array-like).
    For max speed: pass a float64 C-contiguous ndarray and use *_inplace.
    """
    cdef cnp.ndarray[f64, ndim=2] arr = np.asarray(Pw, dtype=np.float64, order="C")
    cdef int dh = arr.shape[1]
    cdef cnp.ndarray[f64, ndim=1] out = np.empty(dh, dtype=np.float64)
    eval_bezier_homogeneous_curve_inplace(arr, <f64>t, out)
    return out


cpdef tuple eval_bezier_curve_homog_with_derivs_fast(const f64[:, ::1] Pw, f64 t, bint want_second=True):
    """
    Fast, no allocations besides outputs, no Bernstein arrays, no NumPy dot.
    """
    cdef int n = Pw.shape[0] - 1
    cdef int dh = Pw.shape[1]

    cdef cnp.ndarray[f64, ndim=1] Ch  = np.empty(dh, dtype=np.float64)
    cdef cnp.ndarray[f64, ndim=1] Chd = np.empty(dh, dtype=np.float64)
    cdef cnp.ndarray[f64, ndim=1] Ch2

    cdef f64[::1] mvCh  = Ch
    cdef f64[::1] mvChd = Chd
    cdef f64[::1] mvCh2

    if want_second:
        Ch2 = np.empty(dh, dtype=np.float64)
        mvCh2 = Ch2
        with nogil:
            _eval_curve_point(Pw, n, dh, t, &mvCh[0])
            _eval_curve_d1(Pw, n, dh, t, &mvChd[0])
            _eval_curve_d2(Pw, n, dh, t, &mvCh2[0])
        return Ch, Chd, Ch2
    else:
        with nogil:
            _eval_curve_point(Pw, n, dh, t, &mvCh[0])
            _eval_curve_d1(Pw, n, dh, t, &mvChd[0])
        return Ch, Chd


def eval_bezier_curve_homog_with_derivs(Pw, double t, bint want_second=True):
    cdef cnp.ndarray[f64, ndim=2] arr = np.asarray(Pw, dtype=np.float64, order="C")
    return eval_bezier_curve_homog_with_derivs_fast(arr, <f64>t, want_second)


# ---------------------------------------------------------------------------
# Surface evaluation (homogeneous) — point + derivatives in tight loops
# ---------------------------------------------------------------------------
cpdef cnp.ndarray[f64, ndim=1] eval_bezier_surface_homog_fast(const f64[:, :, ::1] Pw, f64 u, f64 v):
    cdef int nu = Pw.shape[0] - 1
    cdef int nv = Pw.shape[1] - 1
    cdef int dh = Pw.shape[2]

    cdef cnp.ndarray[f64, ndim=1] Sh  = np.empty(dh, dtype=np.float64)



    cdef f64[::1] mvSh  = Sh


    # Basis buffers (stack for small degrees, heap otherwise)
    cdef f64 Bu_s[_STACK_MAX + 1]
    cdef f64 Bv_s[_STACK_MAX + 1]


    cdef f64* Bu = Bu_s

    cdef f64* Bv = Bv_s


    cdef f64* uheap = NULL
    cdef f64* vheap = NULL

    cdef int need_u =  (nu + 1)
    cdef int need_v =  (nv + 1)

    if nu > _STACK_MAX:
        uheap = <f64*> malloc(need_u * sizeof(f64))
        if uheap == NULL:
            raise MemoryError()
        Bu = uheap



    if nv > _STACK_MAX:
        vheap = <f64*> malloc(need_v * sizeof(f64))
        if vheap == NULL:
            if uheap != NULL: free(uheap)
            raise MemoryError()
        Bv = vheap




    cdef int i, j, k
    cdef f64 w0
    cdef f64 bu
    cdef f64 bv
    cdef const f64* p

    try:
        with nogil:
            _bernstein_basis_fill(nu, u, Bu)
            _bernstein_basis_fill(nv, v, Bv)

            # zero outputs
            for k in range(dh):
                mvSh[k]  = 0.0


            for i in range(nu + 1):
                bu = Bu[i]

                for j in range(nv + 1):
                    bv = Bv[j]


                    w0 = bu * bv



                    p = &Pw[i, j, 0]
                    if dh == 4:
                        mvSh[0]  += w0 * p[0]
                        mvSh[1]  += w0 * p[1]
                        mvSh[2]  += w0 * p[2]
                        mvSh[3]  += w0 * p[3]
                    else:
                        for k in range(dh):
                            mvSh[k]  += w0 * p[k]

    finally:
        if uheap != NULL:
            free(uheap)
        if vheap != NULL:
            free(vheap)

    return Sh

cpdef tuple eval_bezier_surface_homog_with_derivs_fast(const f64[:, :, ::1] Pw, f64 u, f64 v, bint want_second=True):
    cdef int nu = Pw.shape[0] - 1
    cdef int nv = Pw.shape[1] - 1
    cdef int dh = Pw.shape[2]

    cdef cnp.ndarray[f64, ndim=1] Sh  = np.empty(dh, dtype=np.float64)
    cdef cnp.ndarray[f64, ndim=1] Shu = np.empty(dh, dtype=np.float64)
    cdef cnp.ndarray[f64, ndim=1] Shv = np.empty(dh, dtype=np.float64)

    cdef cnp.ndarray[f64, ndim=1] Shuu
    cdef cnp.ndarray[f64, ndim=1] Shuv
    cdef cnp.ndarray[f64, ndim=1] Shvv

    cdef f64[::1] mvSh  = Sh
    cdef f64[::1] mvShu = Shu
    cdef f64[::1] mvShv = Shv
    cdef f64[::1] mvShuu
    cdef f64[::1] mvShuv
    cdef f64[::1] mvShvv

    # Basis buffers (stack for small degrees, heap otherwise)
    cdef f64 Bu_s[_STACK_MAX + 1]
    cdef f64 Bud_s[_STACK_MAX + 1]
    cdef f64 Bu2_s[_STACK_MAX + 1]
    cdef f64 Bv_s[_STACK_MAX + 1]
    cdef f64 Bvd_s[_STACK_MAX + 1]
    cdef f64 Bv2_s[_STACK_MAX + 1]

    cdef f64* Bu = Bu_s
    cdef f64* Bud = Bud_s
    cdef f64* Bu2 = Bu2_s
    cdef f64* Bv = Bv_s
    cdef f64* Bvd = Bvd_s
    cdef f64* Bv2 = Bv2_s

    cdef f64* uheap = NULL
    cdef f64* vheap = NULL

    cdef int need_u = (3 if want_second else 2) * (nu + 1)
    cdef int need_v = (3 if want_second else 2) * (nv + 1)

    if nu > _STACK_MAX:
        uheap = <f64*> malloc(need_u * sizeof(f64))
        if uheap == NULL:
            raise MemoryError()
        Bu = uheap
        Bud = uheap + (nu + 1)
        if want_second:
            Bu2 = uheap + 2 * (nu + 1)

    if nv > _STACK_MAX:
        vheap = <f64*> malloc(need_v * sizeof(f64))
        if vheap == NULL:
            if uheap != NULL: free(uheap)
            raise MemoryError()
        Bv = vheap
        Bvd = vheap + (nv + 1)
        if want_second:
            Bv2 = vheap + 2 * (nv + 1)

    if want_second:
        Shuu = np.empty(dh, dtype=np.float64)
        Shuv = np.empty(dh, dtype=np.float64)
        Shvv = np.empty(dh, dtype=np.float64)
        mvShuu = Shuu
        mvShuv = Shuv
        mvShvv = Shvv

    cdef int i, j, k
    cdef f64 w0, wu, wv, wuu, wuv, wvv
    cdef f64 bu, bud, bu2v
    cdef f64 bv, bvd, bv2v
    cdef const f64* p

    try:
        with nogil:
            _bernstein_basis_fill(nu, u, Bu)
            _bernstein_basis_deriv_fill(nu, u, Bud)
            if want_second:
                _bernstein_basis_2nd_fill(nu, u, Bu2)

            _bernstein_basis_fill(nv, v, Bv)
            _bernstein_basis_deriv_fill(nv, v, Bvd)
            if want_second:
                _bernstein_basis_2nd_fill(nv, v, Bv2)

            # zero outputs
            for k in range(dh):
                mvSh[k]  = 0.0
                mvShu[k] = 0.0
                mvShv[k] = 0.0
            if want_second:
                for k in range(dh):
                    mvShuu[k] = 0.0
                    mvShuv[k] = 0.0
                    mvShvv[k] = 0.0

            for i in range(nu + 1):
                bu = Bu[i]
                bud = Bud[i]
                bu2v = Bu2[i] if want_second else 0.0
                for j in range(nv + 1):
                    bv = Bv[j]
                    bvd = Bvd[j]
                    bv2v = Bv2[j] if want_second else 0.0

                    w0 = bu * bv
                    wu = bud * bv
                    wv = bu * bvd

                    if want_second:
                        wuu = bu2v * bv
                        wuv = bud * bvd
                        wvv = bu * bv2v

                    p = &Pw[i, j, 0]
                    if dh == 4:
                        mvSh[0]  += w0 * p[0]
                        mvSh[1]  += w0 * p[1]
                        mvSh[2]  += w0 * p[2]
                        mvSh[3]  += w0 * p[3]

                        mvShu[0] += wu * p[0]
                        mvShu[1] += wu * p[1]
                        mvShu[2] += wu * p[2]
                        mvShu[3] += wu * p[3]

                        mvShv[0] += wv * p[0]
                        mvShv[1] += wv * p[1]
                        mvShv[2] += wv * p[2]
                        mvShv[3] += wv * p[3]

                        if want_second:
                            mvShuu[0] += wuu * p[0]
                            mvShuu[1] += wuu * p[1]
                            mvShuu[2] += wuu * p[2]
                            mvShuu[3] += wuu * p[3]

                            mvShuv[0] += wuv * p[0]
                            mvShuv[1] += wuv * p[1]
                            mvShuv[2] += wuv * p[2]
                            mvShuv[3] += wuv * p[3]

                            mvShvv[0] += wvv * p[0]
                            mvShvv[1] += wvv * p[1]
                            mvShvv[2] += wvv * p[2]
                            mvShvv[3] += wvv * p[3]
                    else:
                        for k in range(dh):
                            mvSh[k]  += w0 * p[k]
                            mvShu[k] += wu * p[k]
                            mvShv[k] += wv * p[k]
                        if want_second:
                            for k in range(dh):
                                mvShuu[k] += wuu * p[k]
                                mvShuv[k] += wuv * p[k]
                                mvShvv[k] += wvv * p[k]
    finally:
        if uheap != NULL:
            free(uheap)
        if vheap != NULL:
            free(vheap)

    if want_second:
        return Sh, Shu, Shv, Shuu, Shuv, Shvv
    else:
        return Sh, Shu, Shv


def eval_bezier_homogeneous_surface(Pw, double u, double v):
    cdef cnp.ndarray[f64, ndim=3] arr = np.asarray(Pw, dtype=np.float64, order="C")
    # compute just point via fast deriv routine (no extra basis cost in practice)
    Sh, _, _ = eval_bezier_surface_homog_with_derivs_fast(arr, <f64>u, <f64>v, False)
    return Sh


def eval_bezier_surface_homog_with_derivs(Pw, double u, double v, bint want_second=True):
    cdef cnp.ndarray[f64, ndim=3] arr = np.asarray(Pw, dtype=np.float64, order="C")
    return eval_bezier_surface_homog_with_derivs_fast(arr, <f64>u, <f64>v, want_second)


# ---------------------------------------------------------------------------
# Homogeneous -> Cartesian projections (quotient rules)
# ---------------------------------------------------------------------------

cpdef tuple project_curve_homog_to_cartesian(Ch, Chd, Ch2=None):
    cdef cnp.ndarray[f64, ndim=1] a  = np.asarray(Ch,  dtype=np.float64, order="C")
    cdef cnp.ndarray[f64, ndim=1] ad = np.asarray(Chd, dtype=np.float64, order="C")
    cdef cnp.ndarray[f64, ndim=1] a2

    cdef int dh = a.shape[0]
    cdef int d = dh - 1

    cdef cnp.ndarray[f64, ndim=1] C  = np.empty(d, dtype=np.float64)
    cdef cnp.ndarray[f64, ndim=1] Cp = np.empty(d, dtype=np.float64)
    cdef cnp.ndarray[f64, ndim=1] Cpp

    cdef f64[::1] mvA = a
    cdef f64[::1] mvAd = ad
    cdef f64[::1] mvC = C
    cdef f64[::1] mvCp = Cp
    cdef f64[::1] mvA2
    cdef f64[::1] mvCpp

    cdef f64 w, wd, w2
    cdef f64 invw, invw2, invw3
    cdef int k

    if Ch2 is None:
        with nogil:
            w = mvA[dh - 1]
            wd = mvAd[dh - 1]
            invw = 1.0 / w
            invw2 = invw * invw
            for k in range(d):
                mvC[k] = mvA[k] * invw
                mvCp[k] = (w * mvAd[k] - mvA[k] * wd) * invw2
        return C, Cp

    a2 = np.asarray(Ch2, dtype=np.float64, order="C")
    mvA2 = a2
    Cpp = np.empty(d, dtype=np.float64)
    mvCpp = Cpp

    with nogil:
        w = mvA[dh - 1]
        wd = mvAd[dh - 1]
        w2 = mvA2[dh - 1]

        invw = 1.0 / w
        invw2 = invw * invw
        invw3 = invw2 * invw

        for k in range(d):
            mvC[k] = mvA[k] * invw
            mvCp[k] = (w * mvAd[k] - mvA[k] * wd) * invw2
            mvCpp[k] = ((w * w) * mvA2[k] - 2.0 * w * wd * mvAd[k] - w * w2 * mvA[k] + 2.0 * (wd * wd) * mvA[k]) * invw3

    return C, Cp, Cpp


cpdef tuple project_surface_homog_to_cartesian(Sh, Shu, Shv, Shuu=None, Shuv=None, Shvv=None):
    cdef cnp.ndarray[f64, ndim=1] a   = np.asarray(Sh,  dtype=np.float64, order="C")
    cdef cnp.ndarray[f64, ndim=1] au  = np.asarray(Shu, dtype=np.float64, order="C")
    cdef cnp.ndarray[f64, ndim=1] av  = np.asarray(Shv, dtype=np.float64, order="C")

    cdef int dh = a.shape[0]
    cdef int d = dh - 1

    cdef cnp.ndarray[f64, ndim=1] S  = np.empty(d, dtype=np.float64)
    cdef cnp.ndarray[f64, ndim=1] Su = np.empty(d, dtype=np.float64)
    cdef cnp.ndarray[f64, ndim=1] Sv = np.empty(d, dtype=np.float64)

    cdef f64[::1] mvA  = a
    cdef f64[::1] mvAu = au
    cdef f64[::1] mvAv = av
    cdef f64[::1] mvS  = S
    cdef f64[::1] mvSu = Su
    cdef f64[::1] mvSv = Sv

    cdef f64 w, wu, wv
    cdef f64 invw, invw2, invw3
    cdef int k

    if Shuu is None:
        with nogil:
            w  = mvA[dh - 1]
            wu = mvAu[dh - 1]
            wv = mvAv[dh - 1]
            invw = 1.0 / w
            invw2 = invw * invw
            for k in range(d):
                mvS[k]  = mvA[k] * invw
                mvSu[k] = (w * mvAu[k] - mvA[k] * wu) * invw2
                mvSv[k] = (w * mvAv[k] - mvA[k] * wv) * invw2
        return S, Su, Sv

    cdef cnp.ndarray[f64, ndim=1] auu = np.asarray(Shuu, dtype=np.float64, order="C")
    cdef cnp.ndarray[f64, ndim=1] auv = np.asarray(Shuv, dtype=np.float64, order="C")
    cdef cnp.ndarray[f64, ndim=1] avv = np.asarray(Shvv, dtype=np.float64, order="C")

    cdef f64[::1] mvAuu = auu
    cdef f64[::1] mvAuv = auv
    cdef f64[::1] mvAvv = avv

    cdef cnp.ndarray[f64, ndim=1] Suu = np.empty(d, dtype=np.float64)
    cdef cnp.ndarray[f64, ndim=1] Suv = np.empty(d, dtype=np.float64)
    cdef cnp.ndarray[f64, ndim=1] Svv = np.empty(d, dtype=np.float64)

    cdef f64[::1] mvSuu = Suu
    cdef f64[::1] mvSuv = Suv
    cdef f64[::1] mvSvv = Svv

    cdef f64 wuu, wuv, wvv

    with nogil:
        w   = mvA[dh - 1]
        wu  = mvAu[dh - 1]
        wv  = mvAv[dh - 1]
        wuu = mvAuu[dh - 1]
        wuv = mvAuv[dh - 1]
        wvv = mvAvv[dh - 1]

        invw = 1.0 / w
        invw2 = invw * invw
        invw3 = invw2 * invw

        for k in range(d):
            mvS[k]  = mvA[k] * invw
            mvSu[k] = (w * mvAu[k] - mvA[k] * wu) * invw2
            mvSv[k] = (w * mvAv[k] - mvA[k] * wv) * invw2

            mvSuu[k] = (w*w*mvAuu[k] - 2.0*w*wu*mvAu[k] - w*wuu*mvA[k] + 2.0*(wu*wu)*mvA[k]) * invw3
            mvSvv[k] = (w*w*mvAvv[k] - 2.0*w*wv*mvAv[k] - w*wvv*mvA[k] + 2.0*(wv*wv)*mvA[k]) * invw3
            mvSuv[k] = (w*w*mvAuv[k] - w*(wu*mvAv[k] + wv*mvAu[k]) - w*wuv*mvA[k] + 2.0*wu*wv*mvA[k]) * invw3

    return S, Su, Sv, Suu, Suv, Svv


# ---------------------------------------------------------------------------
# Backward-compatible helpers from your file
# ---------------------------------------------------------------------------

def elevate_derivative_net_homog(P4, int order=2):
    cdef cnp.ndarray[f64, ndim=2] P = np.asarray(P4, dtype=np.float64, order="C")
    cdef int dh = P.shape[1]
    cdef list nets = [P.copy()]
    cdef size_t nets_cnt=P.shape[0]
    cdef int nets_last_i
    cdef int k, i, j, n
    cdef cnp.ndarray[f64, ndim=2] prev
    cdef cnp.ndarray[f64, ndim=2] d
    cdef f64[:, ::1] mvPrev
    cdef f64[:, ::1] mvD

    for k in range(1, order + 1):
        nets_cnt=len(nets)
        nets_last_i = nets_cnt-1
        prev = nets[nets_last_i]
        n = prev.shape[0] - 1
        if n <= 0:
            nets.append(prev[:1].copy())
            continue

        d = np.empty((n, dh), dtype=np.float64)
        mvPrev = prev
        mvD = d
        with nogil:
            for i in range(n):
                for j in range(dh):
                    mvD[i, j] = (<f64>n) * (mvPrev[i + 1, j] - mvPrev[i, j])
        nets.append(d)

    return nets


def eval_homog_derivatives(P4, float t, int order=2):
    cdef cnp.ndarray[f64, ndim=2] P = np.asarray(P4, dtype=np.float64, order="C")

    if order <= 0:
        Ch = eval_bezier_homogeneous_curve(P, t)
        return [Ch]

    if order == 1:
        Ch, Chd = eval_bezier_curve_homog_with_derivs_fast(P, <f64>t, False)
        return [Ch, Chd]

    if order == 2:
        Ch, Chd, Ch2 = eval_bezier_curve_homog_with_derivs_fast(P, <f64>t, True)
        return [Ch, Chd, Ch2]

    # Generic fallback for higher orders: build derivative nets
    nets = elevate_derivative_net_homog(P, order=order)
    out = []
    for net in nets:
        out.append(eval_bezier_homogeneous_curve(net, t))
    return out


def dehomogenize_chain(Hvals):
    # Supports up to 2nd derivative like your original
    if not Hvals:
        return []

    H0 = np.asarray(Hvals[0], dtype=np.float64, order="C")
    dh = H0.shape[0]
    d = dh - 1

    N0 = H0[:d]
    W0 = H0[d]
    pos = N0 / W0
    out = [pos]

    if len(Hvals) >= 2:
        H1 = np.asarray(Hvals[1], dtype=np.float64, order="C")
        N1 = H1[:d]
        W1 = H1[d]
        out.append((N1 * W0 - N0 * W1) / (W0 * W0))

    if len(Hvals) >= 3:
        H2 = np.asarray(Hvals[2], dtype=np.float64, order="C")
        N2 = H2[:d]
        W2 = H2[d]
        num = (N2 * (W0 * W0)) - 2.0 * N1 * W0 * W1 + 2.0 * N0 * (W1 * W1) - N0 * W2 * W0
        out.append(num / (W0 * W0 * W0))

    return out
