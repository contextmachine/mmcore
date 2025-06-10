cimport cython
import numpy as np
cimport numpy as cnp

from libc.math cimport fabs


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def poly2d_mul(double[:, :] polyA,double[:, :] polyB):
    """
    Multiply two 2D polynomials in (u,v).
    polyA and polyB are NumPy arrays of shape (a_u+1, a_v+1) and (b_u+1, b_v+1),
    whose entries are the coefficients for u^i v^j. Returns an array of shape
    (a_u+b_u+1, a_v+b_v+1).
    """
    cdef int a_u = polyA.shape[0] - 1
    cdef int a_v = polyA.shape[1] - 1
    cdef int b_u = polyB.shape[0] - 1
    cdef int b_v = polyB.shape[1] - 1
    
    cdef int result_u = a_u + b_u + 1
    cdef int result_v = a_v + b_v + 1
    
    cdef double[:, :] result = np.zeros((result_u, result_v), dtype=np.double)
    cdef double[:, :] result_view = result
    
    cdef int i, j, k, l
    cdef double coeffA
    
    for i in range(a_u + 1):
        for j in range(a_v + 1):
            coeffA = polyA[i, j]
            if fabs(coeffA) < 1e-18:
                continue
            for k in range(b_u + 1):
                for l in range(b_v + 1):
                    result_view[i + k, j + l] += coeffA * polyB[k, l]
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def poly2d_pow(cnp.double_t[:, :] poly, int exponent):
    """
    Raise a 2D polynomial (in u,v) to a nonnegative integer exponent.
    Uses repeated squaring / repeated multiplication.
    If exponent = 0, returns the constant 1 polynomial [[1.0]].
    """
    if exponent == 0:
        return np.array([[1.0]], dtype=np.double)
    if exponent == 1:
        return np.asarray(poly).copy()
    
    cdef double[:, :] half = poly2d_pow(poly, exponent // 2)
    cdef double[:, :] half2 = poly2d_mul(half, half)
    
    if exponent % 2 == 0:
        return half2
    return poly2d_mul(half2, poly)

