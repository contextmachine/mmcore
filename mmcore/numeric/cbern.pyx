# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
cimport cython
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport pow
from libc.math cimport pow

cnp.import_array()


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void bernstein_basis_funs_c(int n, double t, double * B) noexcept nogil:
    """
    Compute Bernstein basis functions B_i^n(t) for i=0..n.
    Writes results directly into the pointer B (size n+1).
    """
    # 2. Variable declaration
    cdef int i
    cdef double r
    cdef double omt = 1.0 - t
    # 1. Handle explicit 0 and 1 cases for speed and stability
    if t == 0.0:
        memset(B, 0, (n + 1) * sizeof(double))
        B[0] = 1.0
        return

    if t == 1.0:
        memset(B, 0, (n + 1) * sizeof(double))
        B[n] = 1.0
        return



    # 3. Stable Recurrence (Forward or Backward)
    # We do not need malloc here; we can write sequentially into B.

    if t <= 0.5:
        # Forward recurrence: Calculate B[0] ... B[n]
        # B[0] = (1-t)^n
        B[0] = pow(omt, n)

        # Ratio r = t / (1-t)
        r = t / omt

        for i in range(n):
            # B[i+1] = B[i] * (n-i)/(i+1) * r
            B[i + 1] = (B[i] * <double> (n - i) / <double> (i + 1)) * r

    else:
        # Backward recurrence: Calculate B[n] ... B[0]
        # B[n] = t^n
        B[n] = pow(t, n)

        # Ratio r = (1-t) / t
        r = omt / t

        for i in range(n, 0, -1):
            # B[i-1] = B[i] * i / (n-i+1) * r
            B[i - 1] = (B[i] * <double> i / <double> (n - i + 1)) * r


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def bernstein_basis(int n, double t):
        """
        Python wrapper that allocates memory and calls the C kernel.
        """
        if n < 0:
            raise ValueError("n must be >= 0")

        # Allocate output array (uninitialized is fine as we overwrite specific paths,
        # but strictly we should zero it if we aren't sure we fill every index.
        # Our C logic handles all indices via the recurrence or memset, so empty is safe).

        cdef double[:] B = np.empty(n + 1, dtype=np.float64)

        bernstein_basis_funs_c(n, t, &B[0])

        return B

@cython.cdivision(True)
cdef inline void _basis_core(int n, double t, double * B) noexcept nogil:
    """
    Computes B_i^n(t) into the provided pointer B.
    B must have size (n+1).
    """
    # Handle explicit edges for stability
    if t == 0.0:
        memset(B, 0, (n + 1) * sizeof(double))
        B[0] = 1.0
        return
    if t == 1.0:
        memset(B, 0, (n + 1) * sizeof(double))
        B[n] = 1.0
        return

    cdef int i
    cdef double r
    cdef double omt = 1.0 - t

    if t <= 0.5:
        # Forward
        B[0] = pow(omt, n)
        r = t / omt
        for i in range(n):
            B[i + 1] = (B[i] * <double> (n - i) / <double> (i + 1)) * r
    else:
        # Backward
        B[n] = pow(t, n)
        r = omt / t
        for i in range(n, 0, -1):
            B[i - 1] = (B[i] * <double> i / <double> (n - i + 1)) * r

# -----------------------------------------------------------------------------
# 2. Derivative Kernels
# -----------------------------------------------------------------------------

cdef void _deriv_core(int n, double t, double * Bd) noexcept nogil:
    """
    Computes first derivative of B^n(t).
    Bd must have size (n+1).
    """
    if n <= 0:
        memset(Bd, 0, (n + 1) * sizeof(double))
        return

    # 1. Alloc scratch for degree (n-1) -> size n
    cdef double * B_sub = <double *> malloc(n * sizeof(double))
    if B_sub == NULL:
        return  # Memory error (unlikely)

    # 2. Compute lower degree basis
    _basis_core(n - 1, t, B_sub)

    # 3. Apply formula: n * (B_{i-1}^{n-1} - B_i^{n-1})
    # Unrolled for speed and boundary safety

    # i = 0: n * (0 - B_sub[0])
    Bd[0] = -1.0 * <double> n * B_sub[0]

    # i = 1..n-1
    cdef int i
    cdef double nd = <double> n
    for i in range(1, n):
        Bd[i] = nd * (B_sub[i - 1] - B_sub[i])

    # i = n: n * (B_sub[n-1] - 0)
    Bd[n] = nd * B_sub[n - 1]

    free(B_sub)

cdef void _second_deriv_core(int n, double t, double * out) noexcept nogil:
    """
    Computes second derivative of B^n(t).
    out must have size (n+1).
    """
    if n <= 1:
        memset(out, 0, (n + 1) * sizeof(double))
        return

    # 1. Alloc scratch for degree (n-2) -> size n-1
    cdef double * B_sub = <double *> malloc((n - 1) * sizeof(double))
    if B_sub == NULL:
        return

    # 2. Compute lower degree basis
    _basis_core(n - 2, t, B_sub)

    # 3. Apply formula with factor n*(n-1)
    # diffs are [1, -2, 1] kernel over B_sub

    cdef double factor = <double> (n * (n - 1))
    cdef int i

    # i = 0: factor * (0 - 0 + B_sub[0])
    out[0] = factor * B_sub[0]

    # i = 1: factor * (0 - 2*B_sub[0] + B_sub[1])
    # Note: If n=2, B_sub has size 1 (index 0 only). We must handle that.
    if n == 2:
        # Special case for quadratic:
        # i=0 -> B_sub[0] (done)
        # i=1 -> -2*B_sub[0]
        # i=2 -> B_sub[0]
        out[1] = factor * (-2.0 * B_sub[0])
        out[2] = factor * B_sub[0]
    else:
        # Standard loop for n > 2

        # Boundary start (i=1)
        out[1] = factor * (B_sub[1] - 2.0 * B_sub[0])

        # Main body (i=2 .. n-2)
        # B_sub indices: i-2, i-1, i
        for i in range(2, n - 1):
            out[i] = factor * (B_sub[i - 2] - 2.0 * B_sub[i - 1] + B_sub[i])

        # Boundary end (i=n-1)
        # B_{n-3} - 2 B_{n-2} + 0
        out[n - 1] = factor * (B_sub[n - 3] - 2.0 * B_sub[n - 2])

        # Boundary end (i=n)
        # B_{n-2} - 0 + 0
        out[n] = factor * B_sub[n - 2]

    free(B_sub)

# -----------------------------------------------------------------------------
# 3. Python Wrappers
# -----------------------------------------------------------------------------

def bernstein_basis_deriv(int n, double t):
    """
    High-performance Cython wrapper for 1st derivative.
    """
    if n < 0: raise ValueError("n must be >= 0")

    cdef cnp.ndarray[cnp.float64_t, ndim=1] res = np.empty(n + 1, dtype=np.float64)

    with nogil:
        _deriv_core(n, t, <double *> res.data)

    res.setflags(write=False)
    return res

def bernstein_basis_2nd(int n, double t):
    """
    High-performance Cython wrapper for 2nd derivative.
    """
    if n < 0: raise ValueError("n must be >= 0")

    cdef cnp.ndarray[cnp.float64_t, ndim=1] res = np.empty(n + 1, dtype=np.float64)

    with nogil:
        _second_deriv_core(n, t, <double *> res.data)

    res.setflags(write=False)
    return res