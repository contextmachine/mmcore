cimport cython
cimport numpy as cnp
import numpy as np

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.double_t, ndim=1] eval_bezier_raw(object P_in, double t):
    """
    Evaluate a Bézier curve in Bernstein form using an in-place De Casteljau scheme.

    Parameters
    ----------
    P_in : array_like, shape (n+1, d)
        Control points (Cartesian or homogeneous).
    t : float
        Parameter in [0, 1].

    Returns
    -------
    ndarray, shape (d,)
        Evaluated point in the same control-space (homogeneous safe).
    """
    cdef cnp.ndarray[cnp.double_t, ndim=2] P = np.ascontiguousarray(P_in, dtype=np.float64)
    cdef Py_ssize_t n = P.shape[0] - 1
    cdef Py_ssize_t d = P.shape[1]
    cdef cnp.ndarray[cnp.double_t, ndim=2] Q = np.array(P, copy=True, dtype=np.float64, order="C")
    cdef double[:, ::1] q = Q
    cdef cnp.ndarray[cnp.double_t, ndim=1] out = np.empty((d,), dtype=np.float64)
    cdef Py_ssize_t r, i, j, m
    cdef double omt = 1.0 - t

    if n <= 0:
        for j in range(d):
            out[j] = q[0, j]
        return out

    for r in range(1, n + 1):
        m = n + 1 - r
        for i in range(m):
            for j in range(d):
                q[i, j] = omt * q[i, j] + t * q[i + 1, j]

    for j in range(d):
        out[j] = q[0, j]
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple eval_bezier_raw_d1(object P_in, double t):
    """
    Evaluate a Bézier curve and its first derivative (control-space).

    Returns (P(t), P'(t)).
    """
    cdef cnp.ndarray[cnp.double_t, ndim=2] P = np.ascontiguousarray(P_in, dtype=np.float64)
    cdef Py_ssize_t n = P.shape[0] - 1
    cdef Py_ssize_t d = P.shape[1]
    cdef cnp.ndarray[cnp.double_t, ndim=2] Q = np.array(P, copy=True, dtype=np.float64, order="C")
    cdef double[:, ::1] q = Q
    cdef cnp.ndarray[cnp.double_t, ndim=1] pt = np.empty((d,), dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] dpt = np.zeros((d,), dtype=np.float64)
    cdef Py_ssize_t r, i, j, m
    cdef double omt = 1.0 - t
    cdef double qi0, qi1

    if n <= 0:
        for j in range(d):
            pt[j] = q[0, j]
        return pt, dpt

    if n == 1:
        for j in range(d):
            pt[j] = omt * q[0, j] + t * q[1, j]
            dpt[j] = q[1, j] - q[0, j]
        return pt, dpt

    # Run De Casteljau, capture the penultimate level (m == 2) for dP/dt.
    for r in range(1, n):
        m = n + 1 - r
        for i in range(m):
            for j in range(d):
                q[i, j] = omt * q[i, j] + t * q[i + 1, j]

    # Now q[0] and q[1] correspond to Q0^{n-1}, Q1^{n-1}.
    for j in range(d):
        qi0 = q[0, j]
        qi1 = q[1, j]
        dpt[j] = (<double>n) * (qi1 - qi0)
        # final step for point: Q0^n = lerp(q0, q1)
        pt[j] = omt * qi0 + t * qi1
    return pt, dpt


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple eval_bezier_raw_d2(object P_in, double t):
    """
    Evaluate a Bézier curve and its first and second derivatives (control-space).

    Returns (P(t), P'(t), P''(t)).
    """
    cdef cnp.ndarray[cnp.double_t, ndim=2] P = np.ascontiguousarray(P_in, dtype=np.float64)
    cdef Py_ssize_t n = P.shape[0] - 1
    cdef Py_ssize_t d = P.shape[1]
    cdef cnp.ndarray[cnp.double_t, ndim=2] Q = np.array(P, copy=True, dtype=np.float64, order="C")
    cdef double[:, ::1] q = Q
    cdef cnp.ndarray[cnp.double_t, ndim=1] pt = np.empty((d,), dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] dpt = np.zeros((d,), dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] ddpt = np.zeros((d,), dtype=np.float64)
    cdef Py_ssize_t r, i, j, m
    cdef double omt = 1.0 - t
    cdef double q0, q1, q2, q0m1, q1m1
    cdef double nn = <double>n
    cdef double t2, omt2, b0, b1, b2

    if n <= 0:
        for j in range(d):
            pt[j] = q[0, j]
        return pt, dpt, ddpt

    if n == 1:
        for j in range(d):
            pt[j] = omt * q[0, j] + t * q[1, j]
            dpt[j] = q[1, j] - q[0, j]
        return pt, dpt, ddpt

    if n == 2:
        # Quadratic: use explicit stable formulas (still homogeneous-safe).
        t2 = t * t
        omt2 = omt * omt
        b0 = omt2
        b1 = 2.0 * omt * t
        b2 = t2
        for j in range(d):
            pt[j] = b0 * q[0, j] + b1 * q[1, j] + b2 * q[2, j]
            dpt[j] = 2.0 * (omt * (q[1, j] - q[0, j]) + t * (q[2, j] - q[1, j]))
            ddpt[j] = 2.0 * (q[2, j] - 2.0 * q[1, j] + q[0, j])
        return pt, dpt, ddpt

    # General n>=3: run De Casteljau, capture m==3 and m==2 levels.
    # Stop at r == n-2 so that m == 3.
    for r in range(1, n - 1):
        m = n + 1 - r
        for i in range(m):
            for j in range(d):
                q[i, j] = omt * q[i, j] + t * q[i + 1, j]
        if m == 3:
            # q[0], q[1], q[2] correspond to Q0^{n-2}, Q1^{n-2}, Q2^{n-2}
            for j in range(d):
                q0 = q[0, j]
                q1 = q[1, j]
                q2 = q[2, j]
                ddpt[j] = nn * (nn - 1.0) * (q2 - 2.0 * q1 + q0)

    # Now we are at r == n-2 completed, m should be 3, next step would make m == 2.
    # Perform one more De Casteljau step to get Q^{n-1} (m==2).
    m = 2
    for i in range(m):
        for j in range(d):
            q[i, j] = omt * q[i, j] + t * q[i + 1, j]

    # q[0], q[1] are Q0^{n-1}, Q1^{n-1}
    for j in range(d):
        q0m1 = q[0, j]
        q1m1 = q[1, j]
        dpt[j] = nn * (q1m1 - q0m1)
        pt[j] = omt * q0m1 + t * q1m1

    return pt, dpt, ddpt
