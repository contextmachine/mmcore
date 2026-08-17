cimport cython
from libc.stdlib cimport malloc,free
from libcpp.vector cimport vector
from libcpp.cmath cimport fabs,fmin,fmax,round
from libc.string cimport memcpy
from libcpp.limits cimport numeric_limits



@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int find_span_linear(int degree, double[:] knot_vector, int num_ctrlpts, double knot) nogil:
    cdef int span = degree + 1  # knot span index starts from zero

    while span < num_ctrlpts and knot_vector[span] <= knot:
        span += 1
    return span - 1

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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double calc_epsilon(double x) noexcept nogil:
    cdef double relative_epsilon = numeric_limits[double].epsilon() * fabs(x)
    cdef double absolute_epsilon =  numeric_limits[double].denorm_min();
    cdef double delta = 10e-15;
    if (fabs(x) < delta):
        return absolute_epsilon
    else:
        return relative_epsilon

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int find_multiplicity(double knot, double[:] knot_vector, double tol) noexcept nogil:
    cdef int mult=0
    cdef int l=knot_vector.shape[0]
    cdef int i
    cdef double difference
    cdef double eps
    for i in range(l):
        eps=calc_epsilon(knot_vector[i])
        difference = knot - knot_vector[i]
        if fabs(difference) <= calc_epsilon(knot_vector[i]):
            mult += 1
    return mult

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double knot_insertion_alpha(double u, double[:] knotvector, int span, int idx, int leg)  noexcept nogil:
    return (u - knotvector[leg + idx]) / (knotvector[idx + span + 1] - knotvector[leg + idx])
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double knot_removal_alpha_i(double u, int degree, double[:] knotvector, int num, int idx) noexcept  nogil:
    return (u - knotvector[idx]) / (knotvector[idx + degree + 1 + num] - knotvector[idx])
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double knot_removal_alpha_j(double u, int degree, double[:] knotvector, int num, int idx) noexcept  nogil:
    return (u - knotvector[idx - num]) / (knotvector[idx + degree + 1] - knotvector[idx - num])

