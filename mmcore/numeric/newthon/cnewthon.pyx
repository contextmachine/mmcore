# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
cimport cython
import numpy as np
cimport numpy as cnp
cnp.import_array()
from libc.math cimport fabs

from cpython cimport PyErr_WarnEx


ctypedef cnp.float64_t DTYPE_t

cdef double DEFAULT_H = 1e-5

cdef inline void swap_rows(double[:, ::1] LU, Py_ssize_t row1, Py_ssize_t row2) noexcept nogil:
    cdef Py_ssize_t n = LU.shape[1]
    cdef Py_ssize_t j
    cdef double temp
    for j in range(n):
        temp = LU[row1, j]
        LU[row1, j] = LU[row2, j]
        LU[row2, j] = temp

cdef inline bint solve(double[:, ::1] A, double[::1] b, double[::1] result):
    """
    Solve the linear system Ax = b using LU decomposition with partial pivoting.
    Returns True if successful, False if the matrix is singular.
    """
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i, j, k, max_row
    cdef double sum, temp, pivot
    cdef int temp_pivot
    cdef double[::1] y = np.empty(n, dtype=np.float64)
    cdef double[:, ::1] LU = np.copy(A)
    cdef int[::1] pivots = np.arange(n, dtype=np.intc)

    # LU Decomposition with partial pivoting
    for k in range(n):
        # Find pivot
        pivot = fabs(LU[k, k])
        max_row = k
        for i in range(k + 1, n):
            temp = fabs(LU[i, k])
            if temp > pivot:
                pivot = temp
                max_row = i
        if pivot == 0:
            # Singular matrix
            return False
        if max_row != k:
            # Swap rows in LU
            swap_rows(LU, k, max_row)
            # Swap pivot indices
            temp_pivot = pivots[k]
            pivots[k] = pivots[max_row]
            pivots[max_row] = temp_pivot
        # Continue with LU decomposition
        for i in range(k + 1, n):
            LU[i, k] /= LU[k, k]
            for j in range(k + 1, n):
                LU[i, j] -= LU[i, k] * LU[k, j]

    # Forward substitution to solve L y = P b
    # Apply the permutation to b
    cdef double[::1] b_permuted = np.empty(n, dtype=np.float64)
    for i in range(n):
        b_permuted[i] = b[pivots[i]]

    # Forward substitution
    for i in range(n):
        sum = b_permuted[i]
        for j in range(i):
            sum -= LU[i, j] * y[j]
        y[i] = sum

    # Backward substitution to solve U x = y
    for i in reversed(range(n)):
        sum = y[i]
        for j in range(i + 1, n):
            sum -= LU[i, j] * result[j]
        if LU[i, i] == 0:
            # Singular matrix
            return False
        result[i] = sum / LU[i, i]

    return True

def hessian(f, double[::1] point, double h=DEFAULT_H):
    """
    Calculate the Hessian matrix of a given function `f` at a given `point`.

    :param f: The function to calculate the Hessian matrix for.
    :param point: The point at which to calculate the Hessian matrix.
    :param h: The step size for numerical differentiation.
    :return: The Hessian matrix of `f` at `point`.
    """
    cdef Py_ssize_t n = point.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=2] H = np.zeros((n, n), dtype=np.float64)
    cdef double[:, ::1] H_view = H
    cdef double fp = f(point)
    cdef Py_ssize_t i, j
    cdef double f1, f2, f3, f4
    cdef double temp_i, temp_j
    cdef double[::1] point_copy = point.copy()

    for i in range(n):
        temp_i = point_copy[i]
        # Diagonal elements
        point_copy[i] = temp_i + h
        f1 = f(point_copy)
        point_copy[i] = temp_i - h
        f2 = f(point_copy)
        point_copy[i] = temp_i  # restore point[i]
        H_view[i, i] = (f1 - 2 * fp + f2) / (h * h)

        for j in range(i + 1, n):
            temp_j = point_copy[j]
            # f(x_i + h, x_j + h)
            point_copy[i] = temp_i + h
            point_copy[j] = temp_j + h
            f1 = f(point_copy)
            # f(x_i + h, x_j - h)
            point_copy[j] = temp_j - h
            f2 = f(point_copy)
            # f(x_i - h, x_j + h)
            point_copy[i] = temp_i - h
            point_copy[j] = temp_j + h
            f3 = f(point_copy)
            # f(x_i - h, x_j - h)
            point_copy[j] = temp_j - h
            f4 = f(point_copy)
            # Restore point[i], point[j]
            point_copy[i] = temp_i
            point_copy[j] = temp_j
            H_view[i, j] = H_view[j, i] = (f1 - f2 - f3 + f4) / (4 * h * h)
    return H

def gradient(f, double[::1] point, double h=DEFAULT_H):
    """
    Compute the gradient of a function at a given point.

    :param f: The function to compute the gradient of.
    :param point: The point at which to compute the gradient.
    :param h: The step size for numerical differentiation.
    :return: The gradient vector.
    """
    cdef Py_ssize_t n = point.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] grad = np.zeros(n, dtype=np.float64)
    cdef double[::1] grad_view = grad
    cdef double f_plus_h, f_minus_h
    cdef Py_ssize_t i
    cdef double temp_i
    cdef double[::1] point_copy = point.copy()

    for i in range(n):
        temp_i = point_copy[i]
        point_copy[i] = temp_i + h
        f_plus_h = f(point_copy)
        point_copy[i] = temp_i - h
        f_minus_h = f(point_copy)
        point_copy[i] = temp_i  # restore point[i]
        grad_view[i] = (f_plus_h - f_minus_h) / (2 * h)
    return grad

def newtons_method(f, double[::1] initial_point, double tol=1e-5, int max_iter=100, bint no_warn=False, bint full_return=False, grad=None, hess=None):
    """
    Apply Newton's method to find the root of a function.

    :param f: The function for which the root is to be found.
    :param initial_point: The initial point for the iteration.
    :param tol: Tolerance for the stopping criterion.
    :param max_iter: Maximum number of iterations.
    :param no_warn: If True, suppress warnings.
    :param full_return: If True, return all intermediate variables.
    :param grad: The gradient function. If None, use the gradient function defined above.
    :param hess: The Hessian function. If None, use the hessian function defined above.
    :return: The root of the function if found, None otherwise.
    """
    cdef Py_ssize_t n = initial_point.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] point = np.copy(initial_point)
    cdef double[::1] point_view = point
    cdef int k
    cdef cnp.ndarray[DTYPE_t, ndim=1] grad_vec
    cdef double[::1] grad_view
    cdef cnp.ndarray[DTYPE_t, ndim=2] H_mat
    cdef double[:, ::1] H_view
    cdef cnp.ndarray[DTYPE_t, ndim=1] step
    cdef double[::1] step_view
    cdef double norm_diff
    cdef bint success

    if grad is None:
        _grad = gradient
    else:
        _grad = grad

    if hess is None:
        _hess = hessian
    else:
        _hess = hess

    for k in range(max_iter):
        grad_vec = _grad(f, point_view)
        grad_view = grad_vec
        H_mat = _hess(f, point_view)
        H_view = H_mat
        # Allocate step vector
        step = np.empty_like(grad_vec)
        step_view = step

            # Solve H_mat * step = grad_vec
        success = solve(H_view, grad_view, step_view)
        if not success:
             break
        # Update the point: point = point - step
        for i in range(n):
            point_view[i] -= step_view[i]
        # Check convergence
        norm_diff = 0.0
        for i in range(n):
            norm_diff += step_view[i] * step_view[i]
        norm_diff = norm_diff ** 0.5
        if norm_diff < tol:
            if full_return:
                return point, grad_view, H_view, k
            return point
    if full_return:
        return None, grad_view, H_view, max_iter
    return None
