#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
#cython: initializedcheck=False

cimport cython
cimport numpy as cnp
cnp.import_array()

cdef int find_span(int n, int p, double u, double[:] U, bint is_periodic) noexcept nogil

cdef void basis_funs(int i, double u, int p, double[:] U, double* N) noexcept nogil

cdef void curve_point(int n, int p, double[:] U, double[:, :] P, double u, double* result,bint is_periodic) noexcept nogil

cpdef double[:, :] all_basis_funs(int span, double u, int p, double[:] U)  

cpdef double[:, :] ders_basis_funs(int i, double u, int p, int n, double[:] U) 

cpdef void curve_derivs_alg1(int n, int p, double[:] U, double[:, :] P, double u, int d, double[:, :] CK,bint is_periodic) 

cpdef void curve_deriv_cpts(int p, double[:] U, double[:, :] P, int d, int r1, int r2, double[:, :, :] PK)  
cpdef void curve_derivs_alg2(int n, int p, double[:] U, double[:, :] P, double u, int d, double[:, :] CK, double[:, :,:] PK,bint is_periodic) 

cdef void projective_to_cartesian(double[:] point, double[:] result)  noexcept nogil
cdef void projective_to_cartesian_ptr_ptr(double* point, double* result)  noexcept nogil
cdef void projective_to_cartesian_ptr_mem(double* point, double[:] result)  noexcept nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int find_span_inline(int n, int p, double u, double[:] U, bint is_periodic) nogil:
    """
    Determine the knot span index for a given parameter value `u`.

    This function finds the knot span index `i` such that the parameter `u` 
    lies within the interval [U[i], U[i+1]] in the knot vector `U`. 
    The knot vector `U` is assumed to be non-decreasing and the parameter 
    `u` is within the range `[U[p], U[n+1]]`.

    Parameters
    ----------
    n : int
        The maximum index of the knot span, typically the number of basis functions minus one.
    p : int
        The degree of the B-spline or NURBS.
    u : float
        The parameter value for which the span index is to be found.
    U : list of float
        The knot vector, a non-decreasing sequence of real numbers.

    Returns
    -------
    int
        The index `i` such that `U[i] <= u < U[i+1]`, where `i` is the knot span.

    Raises
    ------
    ValueError
        If the parameter `u` is outside the bounds of the knot vector `U` or 
        if the function fails to find a valid span within the maximum iterations.

    Notes
    -----
    The function employs a binary search algorithm to efficiently locate 
    the knot span. It handles special cases where `u` is exactly equal to 
    the last value in `U` and when `u` is outside the range of `U`.

    Example
    -------
    >>> U = [0, 0, 0, 0.5, 1, 1, 1]
    >>> find_span(4, 2, 0.3, U)
    2

    >>> find_span(4, 2, 0.5, U)
    3
    """
    cdef double U_min = U[p]
    cdef double U_max = U[n+1]
    cdef double period


    if is_periodic :
        # Wrap u to be within the valid range for periodic and closed curves

        period= U_max - U_min
        while u < U_min:
            u += period
        while u > U_max:
            u -= period

    else:
        # Clamp u to be within the valid range for open curves

        if u >= U[n+1]:

            return n

        elif u < U[0]:

            return p

        # Handle special case for the upper boundary
    if u == U[n + 1]:
        return n


    # Binary search for the correct knot span
    cdef int low = p
    cdef int high = n + 1
    cdef int mid = (low + high) // 2

    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2

    return mid
cpdef int find_multiplicity(double knot, double[:] knot_vector, double tol=*)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double knot_insertion_alpha(double u, double[:] knotvector, int span, int idx, int leg)   nogil:
    return (u - knotvector[leg + idx]) / (knotvector[idx + span + 1] - knotvector[leg + idx])
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double knot_removal_alpha_i(double u, int degree, double[:] knotvector, int num, int idx)   nogil:
    return (u - knotvector[idx]) / (knotvector[idx + degree + 1 + num] - knotvector[idx])
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double knot_removal_alpha_j(double u, int degree, double[:] knotvector, int num, int idx)  nogil:
    return (u - knotvector[idx - num]) / (knotvector[idx + degree + 1] - knotvector[idx - num])

cpdef tuple knot_refinement(int degree, double[:] knotvector, double[:, :] ctrlpts, double[:] knot_list=?,  double[:] add_knot_list=?, int density=*, bint is_periodic=*)

cpdef knot_removal(int degree, double[:] knotvector, double[:, :] ctrlpts, double u, double tol=*, int num=*,bint is_periodic=*)

cpdef double[:] knot_insertion_kv(double[:] knotvector, double u, int span, int r) 

cpdef knot_insertion(int degree, double[:] knotvector, double[:, :] ctrlpts, double u, int num=*, int s=*, int span=*, bint is_periodic=*) 

cdef void surface_point(int n, int p, double[:] U, int m, int q, double[:] V, double[:, :, :] Pw, double u, double v, double* result) noexcept nogil