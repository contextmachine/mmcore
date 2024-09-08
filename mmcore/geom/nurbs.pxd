#cython: language_level=3
# distutils: language = c++

cimport cython
cimport numpy as cnp

from mmcore.geom.parametric cimport ParametricCurve,ParametricSurface
from mmcore.numeric.binom cimport binomial_coefficient,binomial_coefficients
from libc.stdlib cimport malloc,free
from libcpp.vector cimport vector
from libcpp.cmath cimport fabs
from libc.string cimport memcpy



cpdef int find_span(int n, int p, double u, double[:] U, bint is_periodic) noexcept nogil



@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void basis_funs(int i, double u, int p, double[:] U, double* N) noexcept nogil:
    """
    Compute the nonvanishing basis functions.
    """

    cdef int pp = p + 1

    #cdef double[:] N = <double*>malloc(sizeof(double)*pp)
    cdef double* left = <double*>malloc(sizeof(double)*pp)
    cdef double* right = <double*>malloc(sizeof(double)*pp)
    N[0] = 1.0

    cdef int j, r
    cdef double saved, temp
    for j in range(1, pp):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0

        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        N[j] = saved
    free(left)
    free(right)


@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void curve_point(int n, int p, double[:] U, double[:, :] P, double u, double* result,bint is_periodic) noexcept nogil:
    """
    Compute a point on a B-spline curve.

    Parameters:
    n (int): The number of basis functions minus one.
    p (int): The degree of the basis functions.
    U (double[:]): The knot vector.
    P (double[:, :]): The control points of the B-spline curve.
    u (double): The parameter value.

    Returns:
    ndarray: The computed curve point.
    """

    cdef int pp = p + 1
    cdef int i, j
    cdef int span = find_span(n, p, u, U,is_periodic)
    cdef double* N = <double*>malloc(sizeof(double)*pp)
    cdef double sum_of_weights=0.
    cdef double b



    basis_funs(span, u, p, U, N)

    for i in range(pp):
        b = N[i] * P[span - p + i, 3]
        sum_of_weights +=  b
        result[0] += (b * P[span - p + i, 0])
        result[1] += (b * P[span - p + i, 1])
        result[2] += (b * P[span - p + i, 2])

    result[0]/=sum_of_weights
    result[1]/=sum_of_weights
    result[2]/=sum_of_weights
    result[3] = 1.


    free(N)





@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double[:, :] all_basis_funs_inline(int span, double u, int p, double* U, double[:, :] N) noexcept nogil:
    """
    Compute all nonzero basis functions and their derivatives up to the ith-degree basis function.

    Parameters:
    span (int): The knot span index.
    u (double): The parameter value.
    p (int): The degree of the basis functions.
    U (double[:]): The knot vector.

    Returns:
    double[:, :]: The basis functions.
    """

    cdef int pp = p+1

    cdef double* left = <double*>malloc(sizeof(double)*pp)
    cdef double* right = <double*>malloc(sizeof(double)*pp)
    N[0, 0] = 1.0

    cdef int j, r
    cdef double saved, temp
    for j in range(1, pp):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            N[j, r] = right[r + 1] + left[j - r]
            temp = N[r, j - 1] / N[j, r]
            N[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j, j] = saved
    free(left)
    free(right)
    return N




cpdef double[:, :] all_basis_funs(int span, double u, int p, double[:] U)

cpdef double[:, :] ders_basis_funs(int i, double u, int p, int n, double[:] U, double[:,:] ders=?)

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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline int find_multiplicity(double knot, double[:] knot_vector, double tol=1e-07) noexcept nogil:
    cdef int mult=0
    cdef int l=knot_vector.shape[0]
    cdef int i
    cdef double difference
    for i in range(l):
        difference=knot - knot_vector[i]
        if fabs(difference) <= tol:
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

cpdef tuple knot_refinement(int degree, double[:] knotvector, double[:, :] ctrlpts, double[:] knot_list=?,  double[:] add_knot_list=?, int density=*, bint is_periodic=*)

cpdef knot_removal(int degree, double[:] knotvector, double[:, :] ctrlpts, double u, double tol=*, int num=*,bint is_periodic=*)

cpdef double[:] knot_insertion_kv(double[:] knotvector, double u, int span, int r) noexcept nogil

cpdef double[:,:] knot_insertion(int degree, double[:] knotvector, double[:, :] ctrlpts, double u, int num, int s, int span, bint is_periodic=*, double[:,:] result=?) noexcept nogil


cdef void surface_derivatives(int[2] degree, double[:] knots_u,  double[:] knots_v, double[:, :] ctrlpts, int[2] size, double u, double v, int deriv_order, double[:, :, :] SKL) noexcept nogil

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void surface_point(int n, int p, double[:] U, int m, int q, double[:] V, double[:, :, :] Pw, double u, double v,  bint periodic_u , bint periodic_v, double[:] result) noexcept nogil :
    cdef int uspan, vspan, k, l,i,cur_l
    cdef int num_cols=4
    cdef int temp_length=(q + 1)*num_cols
    cdef double* Nu = <double*>malloc((p + 1) * sizeof(double))
    cdef double* Nv = <double*>malloc((q + 1) * sizeof(double))
    cdef double* temp = <double*>malloc(temp_length * sizeof(double))
    cdef double  w = 0.0
    result[0]=0.
    result[1]=0.
    result[2]=0.
    uspan = find_span_inline(n, p, u, U, periodic_u)
    basis_funs(uspan, u, p, U, Nu)

    vspan = find_span_inline(m, q, v, V, periodic_v)
    basis_funs(vspan, v, q, V, Nv)

    for l in range(q + 1):
        cur_l=l * num_cols
        temp[cur_l + 0] = 0.0
        temp[cur_l + 1] = 0.0
        temp[cur_l + 2] = 0.0
        temp[cur_l + 3] = 0.0
        for k in range(p + 1):
            for i in range(num_cols):
                temp[cur_l+i] += Nu[k] * Pw[uspan - p + k][vspan - q + l][i]

    for l in range(q + 1):
        cur_l=l*num_cols
        for i in range(num_cols-1):
            result[i] += Nv[l] * temp[cur_l+i]
        w+= Nv[l] * temp[cur_l+3]



    result[0] /= w
    result[1] /= w
    result[2] /= w


    free(Nu)
    free(Nv)
    free(temp)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline void rat_surface_derivs(double[:, :, :] SKLw, int deriv_order, double[:, :, :] SKL) noexcept nogil:
    """
    Computes the derivatives of a rational B-spline surface.

    This function calculates the derivatives of a rational B-spline surface based on the given weighted surface derivative values (`SKLw`). 
    The results are stored in the `SKL` array. The computation takes into account the specified derivative order, 
    and optimizations are applied to speed up the calculation.

    Parameters
    ----------
    SKLw : double[:, :, :]
        A 3D array containing the weighted surface derivatives. 
        The array dimensions are (deriv_order + 1, deriv_order + 1, dimension).
    deriv_order : int
        The highest order of derivative to compute.
    SKL : double[:, :, :]
        A 3D array where the computed derivatives will be stored. 
        The array dimensions are (deriv_order + 1, deriv_order + 1, dimension - 1).

    Notes
    -----
    - The function uses binomial coefficients to adjust the derivatives according to the rational formulation.
    - The function is marked with `nogil` to allow multi-threading in Cython and to avoid Python's Global Interpreter Lock (GIL).
    - Cython compiler directives are used to disable certain checks (e.g., bounds check, wraparound)
     and enable C division for performance.

    No Return
    ---------
        The function operates in-place and does not return any values. The results are directly stored in the provided `SKL` array.

    Examples
    --------
    Consider a scenario where you have a rational B-spline surface and you need to compute its derivatives up to the second order. 
    You would call the function as follows:

    .. code-block:: python

        import numpy as np
        cdef double[:, :, :] SKLw = np.zeros((3, 3, 4))
        cdef double[:, :, :] SKL = np.zeros((3, 3, 3))
        cdef int deriv_order = 2

        rat_surface_derivs(SKLw, deriv_order, SKL)
    """
    cdef int dimension = 4
    cdef int dm=3
    cdef int do=deriv_order+1
    cdef int k, l, j, i,d
    cdef double* v = <double*>malloc(dimension * sizeof(double))
    cdef double* v2 = <double*>malloc((dm) * sizeof(double))
    cdef double bin_kl, bin_ki, bin_lj

    for k in range(do):
        for l in range(do):
            memcpy(v, &SKLw[k, l, 0], dimension * sizeof(double))

            for j in range(1, l + 1):
                bin_lj = binomial_coefficient(l, j)
                for d in range(dm):
                    v[d] = v[d] - (bin_lj * SKLw[0, j, dm] * SKL[k, l-j, d])

            for i in range(1, k + 1):
                bin_ki = binomial_coefficient(k, i)
                for d in range(dm):
                    v[d] = v[d] - (bin_ki * SKLw[i, 0, dm] * SKL[k-i, l, d])

                for d in range(dimension - 1):
                    v2[d] = 0.0

                for j in range(1, l + 1):
                    bin_lj = binomial_coefficient(l, j)
                    for d in range(dimension - 1):
                        v2[d] = v2[d] + (bin_lj * SKLw[i, j, dimension-1] * SKL[k-i, l-j, d])

                for d in range(dimension - 1):
                    v[d] = v[d] - (bin_ki * v2[d])

            for d in range(dimension - 1):
                SKL[k, l, d] = v[d] / SKLw[0, 0, dimension-1]

    free(v)
    free(v2)



cdef class NURBSCurve(ParametricCurve):
    cdef public double[:,:] _control_points
    cdef public int _degree
    cdef double[:] _knots
    cdef bint _periodic
    cdef public object _evaluate_cached
    cdef double[:] _greville_abscissae


    cpdef void set_degree(self, int val)

    cpdef int get_degree(self)
    cpdef bint is_periodic(self)

    cdef void generate_knots(self)

    cpdef knots_update_hook(self)


    cdef void generate_knots_periodic(self)
    cpdef void make_periodic(self)

    cdef _update_interval(self)

    cpdef double[:,:] generate_control_points_periodic(self, double[:,:] cpts)



    cpdef void make_open(self)



    cdef void ctangent(self, double t,double[:] result)



    cdef void ccurvature(self, double t,double[:] result)



    cdef void cevaluate(self, double t, double[:] result) noexcept nogil

    cpdef evaluate4d(self, double t)


    cpdef set(self, double[:,:] control_points, double[:] knots )


    cdef void cevaluate_ptr(self, double t, double *result ) noexcept nogil



    cdef void cderivative(self, double t, double[:] result)


    cdef void csecond_derivative(self, double t, double[:] result)

    cdef void cderivatives1(self, double t, int d, double[:,:] CK )
    cdef void cderivatives2(self, double t, int d, double[:,:] CK )
    cdef void cplane(self, double t, double[:,:] result)
    cdef void cnormal(self, double t, double[:] result)
    cpdef void insert_knot(self, double param, int num)

    cdef NURBSCurve ccopy(self)
    cdef bytes cserialize(self)
    @staticmethod
    cdef NURBSCurve cdeserialize(const unsigned char[:] data)

cpdef double[:] greville_abscissae(double[:] knots, int degree)
cpdef tuple split_curve(NURBSCurve obj, double param, double tol=*)


cdef class NURBSSurface(ParametricSurface):
    cdef double* _control_points_arr
    cdef double[:] _knots_u
    cdef double[:] _knots_v

    cdef int[2] _size
    cdef int[2] _degree


    cdef double[:,:,:] control_points_view
    cdef double[:,:] control_points_flat_view

    cdef void _update_interval(self) noexcept nogil
    cdef void generate_knots_u(self)
    cdef void generate_knots_v(self)
    cdef void realloc_control_points(self, size_t new_size_u, size_t new_size_v) noexcept nogil
    cdef NURBSSurface ccopy(self)
    cpdef void insert_knot_u(self, double t, int count)
    cpdef void insert_knot_v(self, double t, int count)
    cdef void cnormalize_knots(self) noexcept nogil
    cdef void cnormalize_knots_u(self) noexcept nogil
    cdef void cnormalize_knots_v(self) noexcept nogil
    cdef void cbbox(self, double[:,:] result) noexcept nogil
    cpdef double[:,:] bbox(self)

cpdef tuple split_surface_u(NURBSSurface obj, double param)
cpdef tuple split_surface_v(NURBSSurface obj, double param)
cpdef tuple subdivide_surface(NURBSSurface surface, double u=*,double v=*, bint normalize_knots=*)
