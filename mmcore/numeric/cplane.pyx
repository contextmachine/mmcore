import numpy as np
cimport cython
cimport numpy as np
np.import_array()
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void lu_decomposition(double[:,:] A, double[:,:] L, double[:,:] U) noexcept nogil:

    cdef int i, j, k
    cdef double _sum
    for i in range(A.shape[0]):
        # Upper Triangular
        for k in range(i, A.shape[0]):
            _sum= 0.
            for j in range(i):
                _sum+= (L[i][j] * U[j][k])
            U[i][k] = A[i][k] - _sum
        # Lower Triangular
        for k in range(i, A.shape[0]):
            if i == k:
                L[i][i] = 1. # Diagonal as 1
            else:
                _sum= 0.
                for j in range(i):
                    _sum+= (L[k][j] * U[j][i])
                L[k][i] = (A[k][i] - _sum) / U[i][i]
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void forward_substitution(double[:,:] L, double[:] b, double[:] y)noexcept nogil:
    cdef Py_ssize_t n = L.shape[0]
    cdef Py_ssize_t i, j
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void back_substitution(double[:,:] U, double[:] y, double[:] x)noexcept nogil:
    cdef int n = U.shape[0]
    cdef int i,j

    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void lu_solve(double[:,:] A, double[:] b, double[:,:] L, double[:,:] U,double[:] y,double[:] x):

    cdef int i, j
    lu_decomposition(A,L,U)
    for i in range(L.shape[0]):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]

    for i in range(L.shape[0]-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, U.shape[0]):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]




@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian_elimination(double[:, ::1] A, np.ndarray[ndim=1,dtype=double] b):
    cdef int n = A.shape[0]
    cdef double[:, ::1] aug_matrix = np.hstack((A, b.reshape(n, 1)))
    cdef int i, j, k, max_row
    cdef double max_element, c

    for i in range(n):
        # Find pivot
        max_element = abs(aug_matrix[i, i])
        max_row = i
        for k in range(i + 1, n):
            if abs(aug_matrix[k, i]) > max_element:
                max_element = abs(aug_matrix[k, i])
                max_row = k

        # Swap maximum row with current row
        aug_matrix[i, i:n+1], aug_matrix[max_row, i:n+1] = aug_matrix[max_row, i:n+1].copy(), aug_matrix[i, i:n+1].copy()

        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            c = -aug_matrix[k, i] / aug_matrix[i, i]
            for j in range(i, n + 1):
                if i == j:
                    aug_matrix[k, j] = 0
                else:
                    aug_matrix[k, j] += c * aug_matrix[i, j]

    # Solve equation Ax=b using back substitution
    cdef double[::1] x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = aug_matrix[i, n] / aug_matrix[i, i]
        for k in range(i - 1, -1, -1):
            aug_matrix[k, n] -= aug_matrix[k, i] * x[i]

    return np.asarray(x)