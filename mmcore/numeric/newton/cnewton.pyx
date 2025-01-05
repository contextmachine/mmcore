# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
#cython: boundscheck=False, wraparound=False, cdivision=True
cimport cython
import numpy as np
cimport numpy as cnp

cimport numpy as np

# We will use typed memoryviews instead of raw NumPy arrays for speed.
# However, we still need to import the numpy module on the Python side.
cnp.import_array()

# For convenience, define a couple of type aliases:
ctypedef double DTYPE_t
# We will store vectors as 1D memoryviews of doubles
# and matrices as 2D memoryviews of doubles.



from libc.math cimport fabs





cdef double DEFAULT_H = 1e-7

cdef inline void swap_rows(double[:, :] LU, Py_ssize_t row1, Py_ssize_t row2) noexcept nogil:
    cdef Py_ssize_t n = LU.shape[1]
    cdef Py_ssize_t j
    cdef double temp
    for j in range(n):
        temp = LU[row1, j]
        LU[row1, j] = LU[row2, j]
        LU[row2, j] = temp

cdef inline bint solve(double[:, :] A, double[:] b, double[:] result):
    """
    Solve the linear system Ax = b using LU decomposition with partial pivoting.
    Returns True if successful, False if the matrix is singular.
    """
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i, j, k, max_row
    cdef double sum, temp, pivot
    cdef int temp_pivot
    cdef double[:] y = np.empty(n, dtype=np.float64)
    cdef double[:, :] LU = np.copy(A)
    cdef int[:] pivots = np.arange(n, dtype=np.intc)

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
    cdef double[:] b_permuted = np.empty(n, dtype=np.float64)
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
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def hessian(f, double[:] point, double h=DEFAULT_H):
    """
    Calculate the Hessian matrix of a given function `f` at a given `point`.

    :param f: The function to calculate the Hessian matrix for.
    :param point: The point at which to calculate the Hessian matrix.
    :param h: The step size for numerical differentiation.
    :return: The Hessian matrix of `f` at `point`.
    """
    cdef Py_ssize_t n = point.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=2] H = np.zeros((n, n), dtype=np.float64)
    cdef double[:, :] H_view = H
    cdef double fp = f(point)
    cdef Py_ssize_t i, j
    cdef double f1, f2, f3, f4
    cdef double temp_i, temp_j
    cdef double[:] point_copy = point.copy()

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
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gradient(f, double[:] point, double h=DEFAULT_H):
    """
    Compute the gradient of a function at a given point.

    :param f: The function to compute the gradient of.
    :param point: The point at which to compute the gradient.
    :param h: The step size for numerical differentiation.
    :return: The gradient vector.
    """
    cdef Py_ssize_t n = point.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] grad = np.zeros(n, dtype=np.float64)
    cdef double[:] grad_view = grad
    cdef double f_plus_h, f_minus_h
    cdef Py_ssize_t i
    cdef double temp_i
    cdef double[:] point_copy = point.copy()

    for i in range(n):
        temp_i = point_copy[i]
        point_copy[i] = temp_i + h
        f_plus_h = f(point_copy)
        point_copy[i] = temp_i - h
        f_minus_h = f(point_copy)
        point_copy[i] = temp_i  # restore point[i]
        grad_view[i] = (f_plus_h - f_minus_h) / (2 * h)
    return grad


def newtons_method(f, double[:] initial_point, double tol=1e-5, int max_iter=100, bint no_warn=False, bint full_return=False):
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
    cdef double[:] point = np.copy(initial_point)
    cdef double[:] point_view = point
    cdef int k
    cdef double[:] grad_vec
    cdef double[:] grad_view
    cdef double[:, :] H_mat
    cdef double[:, :] H_view
    cdef double[:] step=np.zeros((initial_point.shape[0],))
    cdef double[:] step_view
    cdef double norm_diff
    cdef bint success

  

    for k in range(max_iter):
        grad_vec = gradient(f, point_view)
        grad_view = grad_vec
        H_mat = hessian(f, point_view)
        H_view = H_mat
        # Allocate step vector
        for i in range(step.shape[0]):
            step[i] = 0
      

            # Solve H_mat * step = grad_vec
        success = solve(H_view, grad_view, step)
        if not success:
             break
        # Update the point: point = point - step
        for i in range(n):
            point_view[i] -= step[i]
        # Check convergence
        norm_diff = 0
        for i in range(n):
            norm_diff += step[i] * step[i]

        if norm_diff < (tol*tol):
            if full_return:
                return point, grad_view, H_view, k
            return point
    if full_return:
        return None, grad_view, H_view, max_iter
    return None




############################################################
# 2. Linear Algebra Routines
############################################################


cdef inline double _abs_val(double x) noexcept nogil :
    """Compute absolute value of x (double)."""
    if x < 0:
        return -x
    return x



cdef inline int lu_factorize(double[:, ::1] A,
                 int n,
                 int[::1] pivot_indices) noexcept nogil:
    """
    Perform an LU decomposition of A in place with partial pivoting.

    A becomes the combined L and U factors (Doolittle's method):
       A[i, j] for i > j : L[i, j]
       A[i, j] for i <= j: U[i, j]

    pivot_indices will store the row swaps performed during pivoting.
    Return 0 if successful, or 1 if a singular matrix is encountered.
    """
    cdef:
        int i, j, k, pivot
        double max_val, tmp

    # Initialize the pivot indices as the identity permutation
    for i in range(n):
        pivot_indices[i] = i

    for k in range(n):
        # 1. Find the pivot row
        pivot = k
        max_val = _abs_val(A[pivot_indices[k], k])
        for i in range(k+1, n):
            tmp = _abs_val(A[pivot_indices[i], k])
            if tmp > max_val:
                pivot = i
                max_val = tmp

        # Check for singular matrix
        if max_val == 0.0:
            return 1  # singular

        # 2. Swap pivot row if needed
        if pivot != k:
            tmp = pivot_indices[k]
            pivot_indices[k] = pivot_indices[pivot]
            pivot_indices[pivot] = <int> tmp

        # 3. Perform elimination
        for i in range(k+1, n):
            # L factor
            A[pivot_indices[i], k] /= A[pivot_indices[k], k]
            tmp = A[pivot_indices[i], k]
            for j in range(k+1, n):
                A[pivot_indices[i], j] -= tmp * A[pivot_indices[k], j]

    return 0



cdef inline int lu_solve(double[:, ::1] A,
             int n,
             int[::1] pivot_indices,
             double[::1] b) noexcept nogil:
    """
    Solve A x = b after A has been LU-factorized in place.
    pivot_indices is the list of row permutations done during factorization.
    b is modified in place to become the solution x.
    Return 0 if successful, 1 if unsuccessful (e.g., singular).
    """
    cdef:
        int i, j
        double sum_val

    # 1. Forward substitution for L
    for i in range(n):
        sum_val = b[pivot_indices[i]]
        for j in range(i):
            sum_val -= A[pivot_indices[i], j] * b[pivot_indices[j]]
        b[pivot_indices[i]] = sum_val

    # 2. Back substitution for U
    for i in reversed(range(n)):
        sum_val = b[pivot_indices[i]]
        for j in range(i+1, n):
            sum_val -= A[pivot_indices[i], j] * b[pivot_indices[j]]
        if A[pivot_indices[i], i] == 0.0:
            return 1  # singular
        b[pivot_indices[i]] = sum_val / A[pivot_indices[i], i]

    return 0



cdef inline int solve_linear_system(double[:, ::1] A,
                        double[::1] b,
                        int n) :
    """
    Helper that factors A (n x n) in place, then solves for x in Ax=b.
    The solution is written back into b.
    Return 0 if successful, 1 if singular or error.
    """
    cdef int[::1] pivot_indices = np.zeros(n, dtype=np.intc)  # for row pivots
    cdef int info = lu_factorize(A, n, pivot_indices)
    if info != 0:
        return 1  # singular
    info = lu_solve(A, n, pivot_indices, b)
    if info != 0:
        return 1  # singular
    return 0



cdef inline int invert_matrix(double[:, ::1] A,
                  double[:, ::1] A_inv,
                  int n) :
    """
    Compute the inverse of A (n x n) into A_inv by:
      1) Factorizing A (in-place).
      2) Solving n systems A * e_i = x_i (the columns of the identity).
      3) x_i becomes the corresponding column of A_inv.

    We do *not* want to clobber the user-provided A, so we make a local copy.
    Return 0 if successful, 1 if singular or error.
    """
    cdef:
        int i, j, info
        int[::1] pivot_indices = np.zeros(n, dtype=np.intc)
        double[:, ::1] A_copy = np.zeros((n, n), dtype=np.float64)
        double[::1] col = np.zeros(n, dtype=np.float64)

    # Copy A into A_copy
    for i in range(n):
        for j in range(n):
            A_copy[i, j] = A[i, j]

    # Factor A_copy in place
    info = lu_factorize(A_copy, n, pivot_indices)
    if info != 0:
        return 1  # singular

    # For each column of the identity, solve
    # A_copy * x = e_i. The solution x is the i-th column of A_inv.
    for i in range(n):
        # Set up e_i in col
        for j in range(n):
            col[j] = 0.0
        col[i] = 1.0

        # Solve the system
        info = lu_solve(A_copy, n, pivot_indices, col)
        if info != 0:
            return 1  # singular

        # Write col to A_inv[:, i]
        for j in range(n):
            A_inv[j, i] = col[j]

    return 0


############################################################
# 3. Norm and Newton's Method
############################################################


cdef inline double euclidean_norm(double[::1] vec, int n) noexcept nogil:
    """
    Compute the Euclidean norm of vec of length n.
    """
    cdef:
        int i
        double sum_sq = 0.0

    for i in range(n):
        sum_sq += vec[i] * vec[i]
    return sum_sq**0.5


@cython.boundscheck(True)
@cython.wraparound(True)
def newton_method2(
        object f,      # Python callable: f(x) -> np.ndarray of shape (n,)
        object jac,    # Python callable: jac(x) -> np.ndarray of shape (n,n)
        np.ndarray[np.float64_t, ndim=1] x0,
        double tol=1e-7,
        int max_iter=100
    ):
    """
    Perform Newton's method to find a root of f(x)=0 using the user-supplied
    Jacobian. Both f and jac must be callables.  x0 is the initial guess.
    tol is the tolerance. max_iter is the maximum number of iterations.

    Returns the solution as a NumPy array of shape (n,).
    Raises a ValueError if the method fails (e.g., singular Jacobian).
    """
    cdef:
        int i, k, n
        double[:, ::1] J
        double[:, ::1] J_inv
        double[::1] fx
        double[::1] delta
        double norm_delta

    # Extract dimension
    n = x0.shape[0]

    # We'll keep x in a typed memoryview for speed
    cdef double[::1] x = np.array(x0, dtype=np.float64)  # copy initial guess

    # Allocate needed workspace
    fx = np.zeros(n, dtype=np.float64)
    delta = np.zeros(n, dtype=np.float64)
    J = np.zeros((n, n), dtype=np.float64)
    J_inv = np.zeros((n, n), dtype=np.float64)

    for k in range(max_iter):
        # 1) Evaluate f at x
        fx_np = f(np.asarray(x))  # call Python function
        if fx_np.shape[0] != n:
            raise ValueError("f(x) returned an array of incorrect shape.")
        # Copy fx_np into fx
        for i in range(n):
            fx[i] = fx_np[i]

        # Check stopping condition
        if euclidean_norm(fx, n) < tol:
            # We are close enough to root
            return np.asarray(x)

        # 2) Evaluate the Jacobian at x
        jac_np = jac(np.asarray(x))  # call Python Jacobian
        if jac_np.shape != (n, n):
            raise ValueError("jac(x) returned an array of incorrect shape.")
        # Copy jac_np into J
        for i in range(n):
            for j in range(n):
                J[i, j] = jac_np[i, j]

        # 3) Compute inverse of the Jacobian
        if invert_matrix(J, J_inv, n) != 0:
            raise ValueError("Jacobian is singular or nearly singular at iteration %d" % k)

        # 4) delta = J_inv * f(x)
        #    We'll use delta as a stand-in for that product
        #    i.e., delta <- J_inv @ fx
        for i in range(n):
            delta[i] = 0.0
        for i in range(n):
            for j in range(n):
                delta[i] += J_inv[i, j] * fx[j]

        # 5) x_{k+1} = x_k - delta
        for i in range(n):
            x[i] = x[i] - delta[i]

        # Check if the step itself is small
        norm_delta = euclidean_norm(delta, n)
        if norm_delta < tol:
            return np.asarray(x)

    # If we exit the loop without returning, we failed to converge
    raise ValueError("Newton's method did not converge in %d iterations." % max_iter)