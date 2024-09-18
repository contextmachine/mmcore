# cython: language_level=3
# distutils: language = c++
import functools

cimport cython
from libcpp.vector cimport vector

from libc.stdlib cimport malloc,free,realloc

cimport numpy as cnp
import numpy as np
from libc.string cimport memcpy,memcmp
from libc.math cimport sqrt,fabs,fmax,fminf,fmin,pow
from libc.stdint cimport uint32_t,int32_t

cimport mmcore.geom.nurbs

from mmcore.numeric.algorithms.quicksort cimport uniqueSorted
from  mmcore.numeric cimport calgorithms,vectors

cnp.import_array()

cdef extern from "_nurbs.cpp" nogil:
    cdef cppclass NURBSSurfaceData:
        double* control_points;
        double* knots_u ;
        double* knots_v;
        int size_u;
        int size_v;
        int degree_u;
        int degree_v;
        NURBSSurfaceData();
        NURBSSurfaceData(double* control_points, double* knots_u ,double* knots_v,int size_u,int size_v,int degree_u,int degree_v);

    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void aabb(double[:,:] points, double[:,:] min_max_vals) noexcept nogil:
    """
    AABB (Axis-Aligned Bounding Box) of a point collection.
    :param points: Points
    :rtype: np.ndarray[(2, K), np.dtype[float]] where:
        - N is a points count.
        - K is the number of dims. For example in 3d case (x,y,z) K=3.
    :return: AABB of a point collection.
    :rtype: np.ndarray[(2, K), np.dtype[float]] at [a1_min, a2_min, ... an_min],[a1_max, a2_max, ... an_max],
    """

    cdef int K = 3
    cdef int N = points.shape[0]
    #cdef double[:,:] min_max_vals = np.empty((2,K), dtype=np.float64)
    cdef double p
    cdef int i, j

    # Initialize min_vals and max_vals with the first point's coordinates
    for i in range(K):
        min_max_vals[0][i] = (points[0, i]/points[0, 3])
        min_max_vals[1][i] = (points[0, i]/points[0, 3])

    # Find the min and max for each dimension
    for j in range(1, N):
        for i in range(K):
            p=(points[j, i]/points[j, 3])
            if  p < min_max_vals[0][i]:
                min_max_vals[0][i] =  p
            if  p > min_max_vals[1][i]:
                min_max_vals[1][i] =  p

        

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int find_span(int n, int p, double u, double[:] U, bint is_periodic) noexcept nogil:
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


@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def basis_functions(int i, double u, int p, double[:] U):
    cdef double[:] Nu =np.zeros(p+1)
    basis_funs(i,u,p,U, &Nu[0])
    return np.array(Nu)


@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, :] all_basis_funs(int span, double u, int p, double[:] U):
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
    cdef double[:, :] N = np.zeros((pp, pp))
    cdef double[:] left = np.zeros(pp)
    cdef double[:] right = np.zeros(pp)
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

    return N

@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, :] ders_basis_funs(int i, double u, int p, int n, double[:] U, double[:,:] ders =None):
    """
    Compute the nonzero basis functions and their derivatives for a B-spline.

    This function calculates the nonzero basis functions and their derivatives 
    for a given parameter value `u` in a B-spline curve. The derivatives are 
    computed up to the `n`-th derivative.

    Parameters
    ----------
    i : int
        The knot span index such that `U[i] <= u < U[i+1]`.
    u : double
        The parameter value at which the basis functions and their derivatives 
        are evaluated.
    p : int
        The degree of the B-spline basis functions.
    n : int
        The number of derivatives to compute (0 for just the basis functions).
    U : double[:]
        The knot vector.

    Returns
    -------
    double[:, :]
        A 2D array `ders` of shape `(n+1, p+1)` where `ders[k, j]` contains the 
        `k`-th derivative of the `j`-th nonzero basis function at `u`.

    Notes
    -----
    - The algorithm is based on the approach described in "The NURBS Book" by 
      Piegl and Tiller, which efficiently computes the derivatives using 
      divided differences and recursive evaluation.
    - The function utilizes several optimizations provided by Cython, such as 
      disabling bounds checking and wraparound for faster execution.

    Example
    -------
    Suppose we have a degree 3 B-spline (cubic) with a knot vector `U`, and we 
    want to evaluate the basis functions and their first derivative at `u=0.5` 
    for the knot span `i=3`:

    .. code-block:: python

        import numpy as np
        cdef double[:] U = np.array([0, 0, 0, 0.5, 1, 1, 1], dtype=np.float64)
        ders = ders_basis_funs(3, 0.5, 3, 1, U)
        print(ders)

    This will output the basis functions and their first derivative for the 
    specified parameters.
    """

    cdef int pp=p+1
    cdef int nn = n + 1
    cdef int s1, s2

    cdef double[:, :] ndu = np.zeros((pp,pp))
    cdef double[:] left = np.zeros(pp)
    cdef double[:] right = np.zeros(pp)
    cdef double[:, :] a = np.zeros((2, pp))
    ndu[0, 0] = 1.0

    cdef int j, r, k, rk, pk, j1, j2
    cdef double saved, temp, d
    if ders is None:
        ders = np.zeros((nn, pp))
    for j in range(1, pp):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    for j in range(pp):
        ders[0, j] = ndu[j, p]



    for r in range(pp):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0
        for k in range(1, n + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            j = s1
            s1 = s2
            s2 = j

    r = p
    for k in range(1, n + 1):
        for j in range(pp):
            ders[k, j] *= r
        r *= (p - k)

    return ders

@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void curve_derivs_alg1(int n, int p, double[:] U, double[:, :] P, double u, int d, double[:, :] CK,bint is_periodic):
    """
    Compute the derivatives of a B-spline curve.

    Parameters:
    n (int): The number of basis functions minus one.
    p (int): The degree of the basis functions.
    U (double[:]): The knot vector.
    P (double[:, :]): The control points of the B-spline curve.
    u (double): The parameter value.
    d (int): The number of derivatives to compute.

    Returns:
    ndarray: The computed curve derivatives.
    """


    cdef int du = min(d, p)
    #cdef double[:, :] CK = np.zeros((du + 1, P.shape[1]))
    cdef int pp=p+1
    cdef int span = find_span_inline(n, p, u, U,is_periodic)
    cdef double[:, :] nders = ders_basis_funs(span, u, p, du, U)

    cdef int k, j, l
    for k in range(du + 1):
        for j in range(pp):
            for l in range(CK.shape[1]):
                CK[k, l] += nders[k, j] * P[span - p + j, l]

    #return np.asarray(CK)

@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void curve_deriv_cpts(int p, double[:] U, double[:, :] P, int d, int r1, int r2, double[:, :, :] PK) :
    """
    Compute control points of curve derivatives.

    Parameters:
    p (int): The degree of the basis functions.
    U (double[:]): The knot vector.
    P (double[:, :]): The control points of the B-spline curve.
    d (int): The number of derivatives to compute.
    r1 (int): The start index for control points.
    r2 (int): The end index for control points.
    PK (double[:, :, :]): The computed control points of curve derivatives.

    Returns:
    void
    """


    cdef int r = r2 - r1
    #cdef double[:, :, :] PK = np.zeros((d + 1, r + 1, P.shape[1]))

    cdef int i, k, j,pp
    pp=p+1
    cdef double tmp
    for i in range(r + 1):

        PK[0, i, :] = P[r1 + i, :]

    for k in range(1, d + 1):
        tmp = p - k + 1
        for i in range(r - k + 1):
            for j in range(P.shape[1]):

                PK[k, i, j] = tmp * (PK[k - 1, i + 1, j] - PK[k - 1, i, j]) / (U[r1 + i + pp] - U[r1 + i + k])



@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void curve_derivs_alg2(int n, int p, double[:] U, double[:, :] P, double u, int d, double[:, :] CK, double[:, :,:] PK,bint is_periodic):
    """
    Compute the derivatives of a B-spline curve.

    Parameters:
    n (int): The number of basis functions minus one.
    p (int): The degree of the basis functions.
    U (double[:]): The knot vector.
    P (double[:, :]): The control points of the B-spline curve.
    u (double): The parameter value.
    d (int): The number of derivatives to compute.

    Returns:
    ndarray: The computed curve derivatives
    """


    cdef int dimension = P.shape[1]
    cdef int degree = p
    cdef double[:] knotvector = U

    cdef int du = min(degree, d)

    #cdef double[:, :] CK = np.zeros((d + 1, dimension))

    cdef int span = find_span_inline(n, degree, u, knotvector,is_periodic)
    #cdef double[:, :, :] PK = np.zeros((d + 1, degree + 1, P.shape[1]))

    cdef double[:, :] bfuns = all_basis_funs(span, u, degree, knotvector)

    curve_deriv_cpts(degree, knotvector, P, d, span - degree, span, PK)

    cdef int k, j, i
    for k in range(du + 1):
        for j in range(degree - k + 1):
            for i in range(P.shape[1]):
                CK[k, i] += bfuns[j, degree - k] * PK[k, j, i]

@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void projective_to_cartesian(double[:] point, double[:] result)  noexcept nogil:
    cdef double w = point[3]
    result[0]=point[0]/w
    result[1]=point[1]/w
    result[2]=point[2]/w

@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void projective_to_cartesian_ptr_ptr(double* point, double* result)  noexcept nogil:
    cdef double w = point[3]
    result[0]=point[0]/w
    result[1]=point[1]/w
    result[2]=point[2]/w
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void projective_to_cartesian_ptr_mem(double* point, double[:] result)  noexcept nogil:
    cdef double w = point[3]
    result[0]=point[0]/w
    result[1]=point[1]/w
    result[2]=point[2]/w




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,:] knot_insertion(int degree, double[:] knotvector, double[:, :] ctrlpts, double u, int num, int s, int span, bint is_periodic=0,double[:, :] result=None) noexcept nogil:

    cdef int n = ctrlpts.shape[0]
    cdef int nq = n + num
    cdef int dim = ctrlpts.shape[1]


    cdef double* temp = <double*>malloc(sizeof(double) * (degree + 1) * dim)

    cdef int i, j, L, idx
    cdef double alpha
    if result is None:
        with gil:
         result = np.zeros((nq, dim), dtype=np.float64)


    for i in range(span - degree + 1):
        result[i] = ctrlpts[i]
    for i in range(span - s, n):
        result[i + num] = ctrlpts[i]

    for i in range(degree - s + 1):
        memcpy(&temp[i * dim], &ctrlpts[span - degree + i, 0], sizeof(double) * dim)

    for j in range(1, num + 1):
        L = span - degree + j
        for i in range(degree - j - s + 1):
            alpha = knot_insertion_alpha(u, knotvector, span, i, L)
            for idx in range(dim):
                temp[i * dim + idx] = alpha * temp[(i + 1) * dim + idx] + (1.0 - alpha) * temp[i * dim + idx]
        memcpy(&result[L, 0], &temp[0], sizeof(double) * dim)
        memcpy(&result[span + num - j - s, 0], &temp[(degree - j - s) * dim], sizeof(double) * dim)

    L = span - degree + num
    for i in range(L + 1, span - s):
        memcpy(&result[i, 0], &temp[(i - L) * dim], sizeof(double) * dim)

    free(temp)

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] knot_insertion_kv(double[:] knotvector, double u, int span, int r) noexcept nogil:
    cdef int kv_size = knotvector.shape[0]
    cdef double[:]  kv_updated
    with gil:
      kv_updated  = np.zeros(kv_size + r, dtype=np.float64)

    cdef int i
    for i in range(span + 1):
        kv_updated[i] = knotvector[i]
    for i in range(1, r + 1):
        kv_updated[span + i] = u
    for i in range(span + 1, kv_size):
        kv_updated[i + r] = knotvector[i]

    return kv_updated
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double point_distance(double* a, double* b ,int dim) noexcept nogil:
    cdef int i
    cdef double temp=0.
    for i in range(dim):

        temp+= pow(a[i]+b[i], 2)

    return sqrt(temp)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef knot_removal(int degree, double[:] knotvector, double[:, :] ctrlpts, double u, double tol=1e-4, int num=1,bint is_periodic=0):
    cdef int s = find_multiplicity(u, knotvector,1e-12)
    cdef int n = ctrlpts.shape[0]
    #n, degree,  u, knotvector, is_periodic
    cdef int r = find_span_inline(n, degree,  u, knotvector,is_periodic)

    cdef int first = r - degree
    cdef int last = r - s


    cdef int dim = ctrlpts.shape[1]
    cdef double[:, :] ctrlpts_new = np.zeros((n, dim), dtype=np.float64)
    memcpy(&ctrlpts_new[0, 0], &ctrlpts[0, 0], sizeof(double) * n * dim)

    cdef double* temp = <double*>malloc(sizeof(double) * ((2 * degree) + 1) * dim)

    cdef int t, i, j, ii, jj, k
    cdef bint remflag
    cdef double alpha_i, alpha_j
    cdef double[:] ptn = np.zeros(dim, dtype=np.float64)

    for t in range(num):
        memcpy(&temp[0], &ctrlpts[first - 1, 0], sizeof(double) * dim)
        memcpy(&temp[(last - first + 2) * dim], &ctrlpts[last + 1, 0], sizeof(double) * dim)
        i = first
        j = last
        ii = 1
        jj = last - first + 1
        remflag = False

        while j - i >= t:
            alpha_i = knot_removal_alpha_i(u, degree, knotvector, t, i)
            alpha_j = knot_removal_alpha_j(u, degree, knotvector, t, j)
            for k in range(dim):
                temp[ii * dim + k] = (ctrlpts[i, k] - (1.0 - alpha_i) * temp[(ii - 1) * dim + k]) / alpha_i
                temp[jj * dim + k] = (ctrlpts[j, k] - alpha_j * temp[(jj + 1) * dim + k]) / (1.0 - alpha_j)
            i += 1
            j -= 1
            ii += 1
            jj -= 1

        if j - i < t:
            if point_distance(&temp[(ii - 1) * dim], &temp[(jj + 1) * dim], dim) <= tol:
                remflag = True
        else:
            alpha_i = knot_removal_alpha_i(u, degree, knotvector, t, i)
            for k in range(dim):
                ptn[k] = (alpha_i * temp[(ii + t + 1) * dim + k]) + ((1.0 - alpha_i) * temp[(ii - 1) * dim + k])
            if point_distance(&ctrlpts[i, 0], &ptn[0], dim) <= tol:
                remflag = True

        if remflag:
            i = first
            j = last
            while j - i > t:
                memcpy(&ctrlpts_new[i, 0], &temp[(i - first + 1) * dim], sizeof(double) * dim)
                memcpy(&ctrlpts_new[j, 0], &temp[(j - first + 1) * dim], sizeof(double) * dim)
                i += 1
                j -= 1

        first -= 1
        last += 1

    t += 1
    j = (2 * r - s - degree) // 2
    i = j
    for k in range(1, t):
        if k % 2 == 1:
            i += 1
        else:
            j -= 1
    for k in range(i + 1, n):
        memcpy(&ctrlpts_new[j, 0], &ctrlpts[k, 0], sizeof(double) * dim)
        j += 1

    free(temp)
    return np.asarray(ctrlpts_new[:-t])


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple knot_refinement(int degree, double[:] knotvector, double[:, :] ctrlpts, double[:] knot_list=None,  double[:]  add_knot_list=None, int density=1, bint is_periodic=0) :
    cdef int n = ctrlpts.shape[0] - 1
    cdef int m = n + degree + 1
    cdef int dim = ctrlpts.shape[1]
    cdef double alpha
    cdef int d, i

    if knot_list is None:
        knot_list = np.array(knotvector[degree:-degree], dtype=np.float64)

    if add_knot_list is not None:

        knot_list = np.concatenate((knot_list, add_knot_list))

    cdef int usz=knot_list.shape[0]
    cdef int new_knot_len

    cdef double* sorted_knots=uniqueSorted(&knot_list[0],usz, &new_knot_len )

    knot_list = <double[:new_knot_len]>sorted_knots


    cdef double[:] rknots
    cdef int rknots_size

    for d in range(density - 1):
        rknots_size = usz * 2 - 1
        rknots = np.zeros(rknots_size, dtype=np.float64)

        for i in range(usz - 1):
            rknots[2 * i] = knot_list[i]
            rknots[2 * i + 1] = knot_list[i] + (knot_list[i + 1] - knot_list[i]) / 2.0

        rknots[rknots_size-1] = knot_list[new_knot_len-1]
        knot_list = rknots
        usz = rknots_size
    cdef double[:] X = np.zeros(knot_list.shape[0] * degree, dtype=np.float64)
    cdef int x_count = 0
    cdef int s, r ,ki

    for ki in range(knot_list.shape[0]):
        mk=knot_list[ki]
        s = find_multiplicity(mk, knotvector,1e-12)
        r = degree - s
        for _ in range(r):
            X[x_count] = mk
            x_count += 1
    X = X[:x_count]

    if x_count == 0:
        raise Exception("Cannot refine knot vector on this parametric dimension")

    cdef int r_val = x_count - 1
    cdef int a = find_span_inline(n, degree,  X[0],knotvector, is_periodic)
    #TODO !!!!! Проверить это место если возникнут проблемы
    #n, degree,  u, knotvector, is_periodic
    cdef int b = find_span_inline(n, degree, X[r_val],knotvector,is_periodic) + 1

    cdef double[:, :] new_ctrlpts = np.zeros((n + r_val + 2, dim), dtype=np.float64)
    cdef double[:] new_kv = np.zeros(m + r_val + 2, dtype=np.float64)

    cdef int j, k, l,idx,idx2
    for j in range(a - degree + 1):
        new_ctrlpts[j] = ctrlpts[j]
    for j in range(b - 1, n + 1):
        new_ctrlpts[j + r_val + 1] = ctrlpts[j]

    for j in range(a + 1):
        new_kv[j] = knotvector[j]
    for j in range(b + degree, m + 1):
        new_kv[j + r_val + 1] = knotvector[j]

    i = b + degree - 1
    k = b + degree + r_val
    j = r_val


    while j >= 0:
        while X[j] <= knotvector[i] and i > a:
            new_ctrlpts[k - degree - 1] = ctrlpts[i - degree - 1]
            new_kv[k] = knotvector[i]
            k -= 1
            i -= 1
        memcpy(&new_ctrlpts[k - degree - 1, 0], &new_ctrlpts[k - degree, 0], sizeof(double) * dim)
        for l in range(1, degree + 1):
            idx = k - degree + l
            alpha = new_kv[k + l] - X[j]
            if abs(alpha) < 1e-8:
                memcpy(&new_ctrlpts[idx - 1, 0], &new_ctrlpts[idx, 0], sizeof(double) * dim)
            else:
                alpha = alpha / (new_kv[k + l] - knotvector[i - degree + l])
                for idx2 in range(dim):
                    new_ctrlpts[idx - 1, idx2] = alpha * new_ctrlpts[idx - 1, idx2] + (1.0 - alpha) * new_ctrlpts[idx, idx2]
        new_kv[k] = X[j]
        k -= 1
        j -= 1

    return np.asarray(new_ctrlpts), np.asarray(new_kv)







@cython.cdivision(True)
@cython.boundscheck(False)
cpdef void surface_deriv_cpts(int dim, int[:] degree, double[:] kv0, double[:] kv1, double[:, :] cpts, int[:] cpsize, int[:] rs, int[:] ss, int deriv_order, double[:, :, :, :, :] PKL) :
    """
    Compute the control points of the partial derivatives of a B-spline surface.

    This function calculates the control points of the partial derivatives up to a given order 
    for a surface defined by tensor-product B-splines. The derivatives are computed with respect to the 
    parametric directions (U and V) of the surface.

    Parameters
    ----------
    dim : int
        The dimensionality of the control points (e.g., 2 for 2D, 3 for 3D).

    degree : int[:]
        The degrees of the B-spline in the U and V directions.
        `degree[0]` corresponds to the degree in the U direction, and 
        `degree[1]` to the degree in the V direction.

    kv0 : double[:]
        Knot vector for the U direction.

    kv1 : double[:]
        Knot vector for the V direction.

    cpts : double[:, :, :]
        Array of control points for the B-spline surface. It has a shape 
        of (nV, nU, dim), where nU and nV are the numbers of control points 
        in the U and V directions, respectively.

    cpsize : int[:]
        Size of the control points array in each direction (U and V).
        `cpsize[0]` gives the number of control points in the U direction, 
        and `cpsize[1]` gives the number in the V direction.

    rs : int[:]
        Range of indices in the U direction to consider for derivative computation.
        `rs[0]` is the start index, and `rs[1]` is the end index.

    ss : int[:]
        Range of indices in the V direction to consider for derivative computation.
        `ss[0]` is the start index, and `ss[1]` is the end index.

    deriv_order : int
        The order of the highest derivative to compute. For example, if `deriv_order` is 2,
        the function will compute the first and second derivatives.

    PKL : double[:, :, :, :, :]
        Output array to store the computed derivatives of the control points.
        It has a shape of (du+1, dv+1, r+1, s+1, dim), where `du` and `dv` are 
        the minimum of `degree[0]` and `degree[1]` with `deriv_order`, and `r` and `s` 
        are the ranges of indices specified by `rs` and `ss`.

    Notes
    -----
    - The function first computes the U-directional derivatives of the control points
      for each V-curve and stores them in `PKL`.
    - Then, it computes the V-directional derivatives of the already U-differentiated 
      control points to complete the tensor-product differentiation.
    - Cython decorators are used to disable bounds checking and enable division optimizations
      for enhanced performance.
    """
    cdef int du = min(degree[0], deriv_order)
    cdef int dv = min(degree[1], deriv_order)
    cdef int r = rs[1] - rs[0]
    cdef int s = ss[1] - ss[0]
    cdef int i, j, k, l, d,dd
    cdef double[:, :, :] PKu = np.zeros((du + 1, r + 1, dim), dtype=np.double)
    cdef double[:, :, :] PKuv = np.zeros((dv + 1, s + 1, dim), dtype=np.double)
    cdef double[:, :] temp_cpts = np.zeros((cpsize[0], dim), dtype=np.double)

    # Control points of the U derivatives of every U-curve
    for j in range(ss[0], ss[1] + 1):
        for i in range(cpsize[0]):
            temp_cpts[i] = cpts[j + (cpsize[1] * i)]

        curve_deriv_cpts(degree[0], kv0, temp_cpts, du, rs[0], rs[1], PKu)

        # Copy into output as the U partial derivatives
        for k in range(du + 1):
            for i in range(r - k + 1):
                for d in range(dim):
                    PKL[k, 0, i, j - ss[0], d] = PKu[k, i, d]

    # Control points of the V derivatives of every U-differentiated V-curve
    for k in range(du):
        for i in range(r - k + 1):
            dd = min(deriv_order - k, dv)

            for j in range(s + 1):
                for d in range(dim):
                    temp_cpts[j, d] = PKL[k, 0, i, j, d]

            curve_deriv_cpts(degree[1], kv1[ss[0]:], temp_cpts, dd, 0, s, PKuv)

            # Copy into output
            for l in range(1, dd + 1):
                for j in range(s - l + 1):
                    for d in range(dim):
                        PKL[k, l, i, j, d] = PKuv[l, j, d]
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def surface_point_py(int n, int p, double[:] U, int m, int q, double[:] V, double[:, :, :] Pw, double u, double v,  bint periodic_u=0 , bint periodic_v=0, double[:] result=None):
    if result is None:
        result=np.zeros((3,))

    surface_point(n,p,U,m,q,V,Pw,u,v,periodic_u,periodic_v,result)
    return np.array(result)





@cython.cdivision(True)
cdef inline int min_int(int a, int b) nogil:
    return a if a < b else b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void surface_derivatives(int[2] degree, double[:] knots_u,  double[:] knots_v, double[:, :] ctrlpts, int[2] size, double u, double v, int deriv_order, double[:, :, :] SKL) noexcept nogil:
    """
    Compute surface derivatives for a NURBS (Non-Uniform Rational B-Splines) surface.

    This function calculates the derivatives of a NURBS surface at a given parametric point (u, v).
    The derivatives up to a specified order are computed and stored in the provided array `SKL`.

    Parameters
    ----------
    degree : int[2]
        The degrees of the NURBS surface in the u and v directions.
    knotvector : double[:, :]
        The knot vectors for the u and v directions.
    ctrlpts : double[:, :]
        The control points of the NURBS surface, stored as a 2D array.
    size : int[2]
        The number of control points in the u and v directions.
    u : double
        The parametric coordinate in the u direction where derivatives are evaluated.
    v : double
        The parametric coordinate in the v direction where derivatives are evaluated.
    deriv_order : int
        The maximum order of derivatives to be computed.
    SKL : double[:, :, :]
        An array to store the computed derivatives. `SKL[k][l][i]` contains the derivative 
        of order k in the u direction and l in the v direction for the ith coordinate 
        of the surface point.

    Notes
    -----
    - The function uses Cython directives to optimize performance, including disabling 
      bounds checking, wraparound, and enabling C division.
    - The algorithm follows the standard approach for computing NURBS surface derivatives 
      using basis function derivatives.

    Usage
    -----
    Example usage of this function might look like the following:

    .. code-block:: python

        import numpy as np

        # Define parameters
        degree = np.array([3, 3], dtype=np.int32)
        knotvector = np.array([...], dtype=np.double)  # Define appropriate knot vectors
        ctrlpts = np.array([...], dtype=np.double)    # Define appropriate control points
        size = np.array([num_ctrlpts_u, num_ctrlpts_v], dtype=np.int32)
        u = 0.5
        v = 0.5
        deriv_order = 2
        SKL = np.zeros((deriv_order + 1, deriv_order + 1, 4), dtype=np.double)

        # Call the function
        surface_derivatives(degree, knotvector, ctrlpts, size, u, v, deriv_order, SKL)

        # SKL now contains the derivatives of the surface at (u, v)

    Performance Considerations
    --------------------------
    - Memory management is done manually using malloc/free to ensure efficiency.
    - The `nogil` directive allows the function to be run in parallel threads in Python.

    References
    ----------
    - Piegl, L., & Tiller, W. (1997). The NURBS Book (2nd ed.). Springer.
    """
    cdef int dimension = 4
    cdef int pdimension = 2
    cdef double[2] uv
    uv[0] = u
    uv[1] = v

    cdef int[2] d
    d[0] = min_int(degree[0], deriv_order)
    d[1] = min_int(degree[1], deriv_order)

    cdef int[2] span
    cdef double* basisdrv_u
    cdef double* basisdrv_v
    cdef double[:,:] mbasisdrv_u
    cdef double[:,:] mbasisdrv_v
    cdef int idx, k, i,s, r, cu, cv, l, dd
    cdef double* temp
    cdef int basisdrv_size_u, basisdrv_size_v

    # Allocate memory and compute basis function derivatives for u and v directions
    span[0] = find_span_inline(size[0], degree[0], u, knots_u, 0)
    basisdrv_size_u = (d[0] + 1) * (degree[0] + 1)
    basisdrv_u = <double*>malloc(basisdrv_size_u * sizeof(double))
    with gil:
        mbasisdrv_u=<double[:(d[0] + 1),:(degree[0] + 1)]>basisdrv_u

        ders_basis_funs(span[0], u, degree[0], d[0], knots_u, mbasisdrv_u)

    span[1] = find_span_inline(size[1], degree[1], v, knots_v, 0)
    basisdrv_size_v = (d[1] + 1) * (degree[1] + 1)
    basisdrv_v = <double*>malloc(basisdrv_size_v * sizeof(double))
    with gil:
        mbasisdrv_v=<double[:(d[1] + 1),:(degree[1] + 1)]>basisdrv_v

        ders_basis_funs(span[1], v, degree[1], d[1], knots_v, mbasisdrv_v)

    temp = <double*>malloc((degree[1] + 1) * dimension * sizeof(double))

    for k in range(d[0] + 1):
        for s in range(degree[1] + 1):
            for i in range(dimension):
                temp[s * dimension + i] = 0.0
            for r in range(degree[0] + 1):
                cu = span[0] - degree[0] + r
                cv = span[1] - degree[1] + s
                for i in range(dimension):
                    temp[s * dimension + i] += basisdrv_u[k * (degree[0] + 1) + r] * ctrlpts[cv + (size[1] * cu)][i]

        dd = min_int(deriv_order, d[1])
        for l in range(dd + 1):
            for s in range(degree[1] + 1):
                for i in range(dimension):
                    SKL[k][l][i] += basisdrv_v[l * (degree[1] + 1) + s] * temp[s * dimension + i]

    free(temp)
    free(basisdrv_u)
    free(basisdrv_v)



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def surface_derivatives_py(tuple degree,
            double[:] knots_u,
            double[:] knots_v,
            double[:, :] ctrlpts,
            tuple size,
            double u, double v,
            int deriv_order=0,
            double[:, :, :] SKL=None):
    cdef int[2] deg
    cdef int[2] sz
    deg[0]=degree[0]
    deg[1]=degree[1]
    sz[0]=size[0]
    sz[1]=size[1]
    if SKL is None:
        SKL = np.zeros((deriv_order + 1, deriv_order + 1, 4), dtype=np.double)

        # Call the function

    surface_derivatives(deg,knots_u,knots_v,ctrlpts,sz,u,v,deriv_order,SKL)
    return np.array(SKL)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def rat_surface_derivs_py(double[:, :, :] SKLw, int deriv_order=0, double[:, :, :] SKL=None):
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
    if SKL is None:


        SKL = np.zeros((SKLw.shape[0], SKLw.shape[1], 3))
    rat_surface_derivs(SKLw,deriv_order,SKL)
    return np.array(SKL)



cdef public char* MAGIC_BYTES=b"NRBC"
cdef public uint32_t VERSION=1
cdef int MAGIC_BYTES_SIZE=4
cdef class NURBSCurve(ParametricCurve):


    def __cinit__(self):
        self._evaluate_cached = functools.lru_cache(maxsize=None)(self._evaluate)
        self._knots=np.zeros((0,))
        self._periodic=0
        self._control_points=np.zeros((0,4))
        self._interval=np.zeros((2,))
    def __reduce__(self):
        return (self.__class__, (np.asarray(self._control_points),self._degree,  np.asarray(self._knots),  self._periodic))
    def __init__(self, double[:,:] control_points, int degree=3, double[:] knots=None, bint periodic=0):
        super().__init__()
        self._degree = degree
        self._periodic = periodic
        self._control_points = np.ones((control_points.shape[0], 4))

        if control_points.shape[1]==4:

            self._control_points[:,:] = control_points
        else:
            self._control_points[:,:-1]=control_points


        if knots is None:
                self.generate_knots()
        else:
            self._knots=knots
            self.knots_update_hook()

        if  periodic:
            self.make_periodic()





    def __deepcopy__(self, memodict={}):
        obj=self.__class__(control_points=self.control_points.copy(), degree=self._degree, knots=np.asarray(self._knots).copy())

        obj.weights=self.weights


        return obj
    cpdef void set_degree(self, int val):
        self._degree=val


    cpdef int get_degree(self):
        return self._degree
    @property
    def degree(self):
        return self._degree
    @degree.setter
    def degree(self,v):
        cdef int val=int(v)
        self.set_degree(val)

    cpdef bint is_periodic(self):
        """
        Check if the NURBS curve is periodic
        """
        cdef bint res = True
        cdef int i,j
        cdef double[:,:] part1=self._control_points[:self._degree]
        cdef double[:,:] part2= self._control_points[-self._degree:]
        for j in range(part1.shape[0]):

            for i in range(4):

                res = part1[j][i] == part2[j][i]
                if not res:
                    break

        return res

    def __getstate__(self):
        state=dict()
        state['_control_points']=np.asarray(self._control_points)
        state['_knots'] = np.asarray(self._knots)
        state['_degree'] = self._degree
        state['_periodic']=self._periodic
        state['_interval']=np.asarray(self._interval)
        state['_weights'] = np.asarray(self.weights)
        return state

    def __setstate__(self,state):
        cdef int i
        self._control_points=np.ones((len(state['_control_points']),4))
        for i in range(len(state['_control_points'])):
            self._control_points[i,0]=state['_control_points'][i][  0]
            self._control_points[i, 1] = state['_control_points'][i][1]
            self._control_points[i, 2] = state['_control_points'][i][2]
            self._control_points[i, 3] = state['_weights'][i]

        self._knots = state['_knots']
        self._degree = state['_degree']
        self._periodic = state['_periodic']
        self.weights = state['_control_points'][:,3]
        self._interval = state['_interval']


    @property
    def periodic(self):
        return self._periodic



    @property
    def control_points(self):
        return np.asarray(self._control_points[:,:-1])


    @control_points.setter
    def control_points(self, control_points):
        self._control_points = np.ones((control_points.shape[0],4))
        if control_points.shape[1]==4:
            self._control_points[:]=control_points
        else:

            self._control_points[:, :control_points.shape[1]] = control_points
        self._evaluate_cached.cache_clear()



    @property
    def knots(self):
        return np.asarray(self._knots)
    @knots.setter
    def knots(self, double[:] v):
        self._knots=v
        self.knots_update_hook()
        self._evaluate_cached.cache_clear()

    @property
    def weights(self):
        return np.asarray(self._control_points[:, 3])
    @weights.setter
    def weights(self, double[:] v):
        self._control_points[:, 3]=v
        self._evaluate_cached.cache_clear()

    @property
    def greville_abscissae(self):
        return np.asarray(self._greville_abscissae)


    cdef void generate_knots(self):
        """
        This function generates default knots based on the number of control points
        :return: A numpy array of knots

        Notes
        ------
        **Difference with OpenNURBS**

        OpenNURBS uses a knots vector shorter by one knot on each side. 
        The original explanation can be found in `opennurbs/opennurbs_evaluate_nurbs.h`.
        [source](https://github.com/mcneel/opennurbs/blob/19df20038249fc40771dbd80201253a76100842c/opennurbs_evaluate_nurbs.h#L116-L148)
        mmcore uses the standard knotvector length according to DeBoor and The NURBS Book.

        **Difference with geomdl**

        Unlike geomdl, the knots vector is not automatically normalised from 0 to 1.
        However, there are no restrictions on the use of the knots normalised vector. 

        """
        cdef int n = len(self._control_points)
        self._knots = np.concatenate((
            np.zeros(self._degree + 1),
            np.arange(1, n - self._degree),
            np.full(self._degree + 1, n - self._degree)
        ))

        self.knots_update_hook()
    cpdef knots_update_hook(self):
        self._update_interval()
        self._greville_abscissae = greville_abscissae(self.knots,self.degree
                                                      )


    cdef void generate_knots_periodic(self):
        """
        This function generates knots for a periodic NURBS curve
        """
        cdef int i
        cdef int n = len(self._control_points)
        cdef int m = n + self.degree + 1
        self._knots = np.zeros(m)
        for i in range(m):
            self._knots[i] = i - self.degree
        self.knots_update_hook()

    cdef _update_interval(self):
        self._interval[0] =  np.min(self._knots)
        self._interval[1] = np.max(self._knots)
    cpdef double[:,:] generate_control_points_periodic(self, double[:,:] cpts):
        cdef int n = len(cpts)
        cdef int i
        cdef int new_n = n + self.degree
        cdef double[:,:] new_control_points = np.zeros((new_n, 4))
        new_control_points[:n, :] = cpts
        for i in range(self.degree):
            new_control_points[n + i, :] = cpts[i, :]
        return new_control_points
    cpdef void make_periodic(self):
        """
        Modify the NURBS curve to make it periodic
        """

        if self.is_periodic():
            return
        cdef int n = len(self.control_points)
        cdef int new_n = n + self.degree
        cdef double[:,:] new_control_points = np.zeros((new_n, 4))
        cdef int i
        # Copy the original control points

        new_control_points[:n, :] = self._control_points

        # Add the first degree control points to the end to make it periodic
        for i in range(self.degree):
            new_control_points[n + i, :] = self._control_points[i, :]

        self._control_points = new_control_points
        self.generate_knots_periodic()

        self._periodic=True
        self._evaluate_cached.cache_clear()





    cpdef void make_open(self):
        """
        Modify the NURBS curve to make it open
        """
        if not self.is_open():
            return
        cdef int n = len(self._control_points) - self._degree  # Calculate the original number of control points
        self._control_points = self._control_points[:n, :]  # Trim the extra control points
        self.generate_knots()  # Generate an open knot vector

        self._periodic = False
        self._evaluate_cached.cache_clear()

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void ctangent(self, double t,double[:] result):
        cdef double[:,:] ders=np.zeros((3,3))
        self.cderivatives2(t,2, ders)
        calgorithms.evaluate_tangent(ders[1],ders[2],result)


    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void ccurvature(self, double t,double[:] result):
           cdef double[:,:] ders=np.zeros((3,3))
           cdef double nrm=0
           self.cderivatives2(t,2, ders)
           calgorithms.evaluate_curvature(ders[1],ders[2],ders[0],result)

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cevaluate(self, double t, double[:] result) noexcept nogil:
        """
        Compute a point on a NURBS-spline curve.

        :param t: The parameter value.
        :return: np.array with shape (3,).
        """
        cdef double w
        cdef double* _result_buffer=<double*>malloc(sizeof(double)*4)
        cdef int n= len(self._control_points)-1
        #cdef double * res = <double *> malloc(sizeof(double) * 4)

        _result_buffer[0] = 0.
        _result_buffer[1] = 0.
        _result_buffer[2] = 0.
        _result_buffer[3] = 0.

        curve_point(n, self._degree, self._knots, self._control_points, t, _result_buffer, self._periodic)

        result[0] = _result_buffer[0]
        result[1] = _result_buffer[1]
        result[2] = _result_buffer[2]

        free(_result_buffer)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef evaluate4d(self, double t) :

        cdef double * _result_buffer = <double *> malloc(sizeof(double) * 4)
        cdef int n = len(self._control_points) - 1
        #cdef double * res = <double *> malloc(sizeof(double) * 4)
        cdef double[:] result=np.zeros((4,))
        _result_buffer[0] = 0.
        _result_buffer[1] = 0.
        _result_buffer[2] = 0.
        _result_buffer[3] = 0.

        curve_point(n, self._degree, self._knots, self._control_points, t, _result_buffer, self._periodic)

        result[0] = _result_buffer[0]
        result[1] = _result_buffer[1]
        result[2] = _result_buffer[2]
        result[3] = _result_buffer[3]

        free(_result_buffer)
        return np.asarray(result)


    cpdef set(self, double[:,:] control_points, double[:] knots ):
        self._control_points=control_points
        self._knots=knots
        self._evaluate_cached.cache_clear()

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cevaluate_ptr(self, double t, double *result ) noexcept nogil:
        """
        Compute a point on a NURBS-spline curve.

        :param t: The parameter value.
        :return: np.array with shape (3,).
        """
        cdef double w
        cdef int n = len(self._control_points) - 1

        result[0]=0.
        result[1] = 0.
        result[2]=0.
        result[3] = 0.


        curve_point(n, self._degree, self._knots, self._control_points, t, result, self._periodic)


        result[3]=1.

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _evaluate(self, double t):
        cdef cnp.ndarray[double,ndim=1] result =np.zeros((3,))
        self.cevaluate(t, result)

        return result

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate(self, double t):
        return self._evaluate_cached(t)

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_multi(self, double[:] t):
        cdef cnp.ndarray[double, ndim=2] result=np.empty((t.shape[0],4))
        cdef int i;
        for i in range(t.shape[0]):


            self.cevaluate(t[i],result[i])
        return result[:,:3]

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def derivative(self, t):
        cdef cnp.ndarray[double, ndim=1] result =np.zeros((3,))
        self.cderivative(t,result)
        return result


    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cderivatives1(self, double t, int d, double[:,:] CK ) :
        """
        :param t: The parameter value.
        :type t: float
        :param d: The number of derivatives to compute.
        :type d: int
        :return: np.array with shape (d+1,M) where M is the number of vector components.
        """
        cdef int n = len(self._control_points) - 1
        cdef int i
        #cdef double[:, :]  CK = np.zeros((du + 1, 4))
        #cdef double[:, :, :] PK = np.zeros((d + 1, self._degree + 1,  self._control_points.shape[1]-1))
        curve_derivs_alg1(n, self._degree, self._knots, self._control_points[:,:-1], t, d, CK,self._periodic)

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cderivatives2(self, double t, int d, double[:,:] CK ) :
           """
           :param t: The parameter value.
           :type t: float
           :param d: The number of derivatives to compute.
           :type d: int
           :return: np.array with shape (d+1,M) where M is the number of vector components.
           """
           cdef int n = len(self._control_points) - 1
           cdef int i
           #cdef double[:, :]  CK = np.zeros((du + 1, 4))
           #cdef double[:, :, :] PK = np.zeros((d + 1, self._degree + 1,  self._control_points.shape[1]-1))
           cdef double[:,:,:] PK= np.zeros((d + 1, self._degree + 1, self._control_points.shape[1]-1 ))

           curve_derivs_alg2(n, self._degree, self._knots, self._control_points[:,:-1], t, d, CK, PK,self._periodic)

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cplane(self, double t, double[:,:] result):

        self.cderivatives2(t,2, result[:3,:])


        cdef double nrm = vectors.scalar_norm(result[1])
        result[1,0]/= nrm
        result[1,1] /= nrm
        result[1,2] /= nrm

        vectors.scalar_gram_schmidt_emplace(result[1],result[2])
        nrm=vectors.scalar_norm(result[2])

        result[2, 0] /= nrm
        result[2, 1] /= nrm
        result[2, 2] /= nrm

        result[3,0] = (result[1][1] * result[2][2]) - (result[1][2] * result[2][1])
        result[3,1] = (result[1][2] * result[2][0]) - (result[1][0] * result[2][2])
        result[3,2] = (result[1][0] * result[2][1]) - (result[1][1] * result[2][0])
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cnormal(self, double t, double[:] result):
        cdef double[:,:] vecs=np.zeros((3,3))
        self.cplane(t,vecs)
        result[:]=vecs[2,:]

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def derivatives2(self, double t, int d=1 ) :
        """
        :param t: The parameter value.
        :type t: float
        :param d: The number of derivatives to compute.
        :type d: int
        :return: np.array with shape (d+1,M) where M is the number of vector components.
        """

        cdef int du = min(d, self._degree)
        cdef cnp.ndarray[double, ndim=2]  CK = np.zeros((du + 1, 3))


        self.cderivatives2(t,d,CK)
        return CK
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def derivatives1(self, double t, int d=1 ) :
            """
            :param t: The parameter value.
            :type t: float
            :param d: The number of derivatives to compute.
            :type d: int
            :return: np.array with shape (d+1,M) where M is the number of vector components.
            """
            cdef int du = min(d, self._degree)
            cdef cnp.ndarray[double, ndim=2] CK = np.zeros((du + 1, 3))

            self.cderivatives1(t,d, CK)
            return CK
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def derivative_control_points(self, double t, int d=1 ) :
            """
               :param t: The parameter value.
               :type t: float
               :param d: The number of derivatives to compute.
               :type d: int
               :return: np.array with shape (d+1,M) where M is the number of vector components.
            """
            cdef int n = len(self._control_points) - 1
            cdef int span = find_span(n, self._degree, t,self._knots,self._periodic)
            cdef cnp.ndarray[double, ndim=3] PK = np.zeros((d + 1, self._degree + 1, self._control_points.shape[1]-1))
            curve_deriv_cpts(self._degree, self._knots,self._control_points, d, span - self._degree, span, PK)

            return PK
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cderivative(self, double t, double[:] result):
        cdef double[:,:] res=np.zeros((2,3))
        self.cderivatives2(t, 1, res)
        result[0]=res[1][0]
        result[1] = res[1][1]
        result[2] = res[1][2]

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void csecond_derivative(self, double t, double[:] result):
        cdef double[:,:] res=np.zeros((3,3))
        self.cderivatives2(t, 2, res)
        result[0]= res[2][0]
        result[1]= res[2][1]
        result[2]= res[2][2]

    cpdef void insert_knot(self, double t, int count):
        """ Inserts knots n-times to a spline geometry.


        """

        # Start curve knot insertion
        cdef int n = self._control_points.shape[0]
        cdef int new_count = n + count
        cdef double[:,:] cpts = self._control_points.copy()

        # Find knot span
        cdef int span = find_span_inline(n-1, self._degree, t, self._knots,0)
        cdef double[:] k_v = knot_insertion_kv(self._knots, t, span, count)
        cdef int s_u = find_multiplicity(t, self._knots, 1e-12)
        # Compute new knot vector
        self._control_points=np.empty((new_count,4))


        # Compute new control points

        knot_insertion(self._degree,
                                              self._knots,
                                              cpts,
                                              t,
                                                count, s_u, span,0,self._control_points  )

        # Update curve

        self._knots=k_v
        self._update_interval()

        self._evaluate_cached.cache_clear()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef NURBSCurve ccopy(self):
        cdef int n_ctrlpts = self._control_points.shape[0]
        cdef int dim = self._control_points.shape[1]
        cdef int n_knots = self._knots.shape[0]

        # Allocate memory for new arrays
        cdef double[:, :] new_control_points = np.empty((n_ctrlpts, dim), dtype=np.float64)
        cdef double[:] new_knots = np.empty(n_knots, dtype=np.float64)

        # Copy data using memcpy for efficiency
        cdef double* src_ptr
        cdef double* dst_ptr

        # Copy control points
        src_ptr = &self._control_points[0, 0]
        dst_ptr = &new_control_points[0, 0]
        memcpy(dst_ptr, src_ptr, n_ctrlpts * dim * sizeof(double))

        # Copy knots
        src_ptr = &self._knots[0]
        dst_ptr = &new_knots[0]
        memcpy(dst_ptr, src_ptr, n_knots * sizeof(double))

        # Create new NURBSCurve object
        cdef NURBSCurve new_curve = NURBSCurve.__new__(NURBSCurve)
        new_curve._control_points = new_control_points
        new_curve.knots = new_knots
        new_curve._degree = self._degree
        new_curve._periodic = self._periodic
        new_curve._interval[0] = self._interval[0]
        new_curve._interval[1] = self._interval[1]

        # Copy Greville abscissae if they exist
        if self._greville_abscissae is not None:
            new_curve._greville_abscissae = np.array(self._greville_abscissae, dtype=np.float64)

        return new_curve

    # Method to call ccopy from Python
    def copy(self):
        cdef NURBSCurve crv = self.ccopy()

        crv._evaluate_cached = functools.lru_cache(maxsize=None)(self._evaluate)
        return crv

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef bytes cserialize(self):
        cdef int n_ctrlpts = self._control_points.shape[0]
        cdef int dim = self._control_points.shape[1]
        cdef int n_knots = self._knots.shape[0]

        # Calculate total size of the byte array
        cdef size_t total_size = (
            sizeof(MAGIC_BYTES) +
            sizeof(uint32_t) + # Magic bytes and version
            sizeof(uint32_t) +  # degree
            sizeof(bint) +  # periodic flag
            sizeof(uint32_t)+  # number of control points
            sizeof(uint32_t) +  # number of knots
            n_ctrlpts * dim * sizeof(double) +  # control points
            n_knots * sizeof(double)  # knots
        )
        cdef char* buffer = <char*>malloc(total_size)
        # Allocate memory for the byte array

        cdef char* current = buffer
        cdef char* magic
        magic=<char*>&(MAGIC_BYTES[0])
        cdef uint32_t vers=<uint32_t> VERSION
        # Write magic bytes and version
        memcpy(buffer, magic, MAGIC_BYTES_SIZE)
        current += MAGIC_BYTES_SIZE
        (<uint32_t*>current)[0] = vers  # version
        current += sizeof(uint32_t)



        # Write degree and periodic flag
        (<uint32_t*>current)[0] = self._degree
        current += sizeof(uint32_t)
        (<uint32_t*>current)[0] = self._periodic
        current += sizeof(uint32_t)

        # Write number of control points and knots
        (<uint32_t*>current)[0] = n_ctrlpts
        current += sizeof(uint32_t)
        (<uint32_t*>current)[0] = n_knots
        current += sizeof(uint32_t)
        cdef double* src_ptr
        cdef double* dst_ptr

        # Copy control points
        src_ptr = &self._control_points[0, 0]
        dst_ptr = &self._knots[0]
        # Write control points
        memcpy(current,src_ptr, n_ctrlpts * dim * sizeof(double))
        current += (n_ctrlpts * dim * sizeof(double))

        # Write knots
        memcpy(current, dst_ptr, n_knots * sizeof(double))

        # Create Python bytes object and free the buffer
        cdef bytes result = buffer[:total_size]
        free(buffer)
        return result



    def serialize(self):

        cdef bytes res = self.cserialize()


        return res
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cnormalize_knots(self):
        cdef double start=np.min(self._knots)
        cdef double end = np.max(self._knots)
        cdef double d=1/(end-start)
        cdef int i
        for i in range(len(self._knots)):
            self._knots[i]=((self._knots[i]-start)*d)
        self.knots_update_hook()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def normalize_knots(self):
        self.cnormalize_knots()
        
        self._evaluate_cached.cache_clear()

    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef NURBSCurve cdeserialize(const unsigned char[:] data):
        cdef const char* buffer = <const char*>&data[0]
        cdef const char* current = buffer
        cdef int vers
        # Check magic bytes and version
        if memcmp(current, MAGIC_BYTES, MAGIC_BYTES_SIZE) != 0:
            raise ValueError(f"Invalid magic bytes: {current}")
        current += MAGIC_BYTES_SIZE
        vers=(<uint32_t*>current)[0]
        if (<uint32_t*>current)[0] != VERSION:
            raise ValueError(f"Unsupported version: {vers}")
        current += sizeof(uint32_t)

        # Read degree and periodic flag
        cdef int degree = (<uint32_t*>current)[0]
        current += sizeof(uint32_t)
        cdef bint periodic = (<uint32_t*>current)[0]
        current += sizeof(uint32_t)

        # Read number of control points and knots
        cdef int n_ctrlpts = (<uint32_t*>current)[0]
        current += sizeof(uint32_t)
        cdef int n_knots = (<uint32_t*>current)[0]
        current += sizeof(uint32_t)

        # Allocate memory for control points and knots
        cdef double* control_points_data = <double*>malloc(n_ctrlpts * 4 * sizeof(double))
        cdef double* knots_data = <double*>malloc(n_knots * sizeof(double))

        if control_points_data == NULL or knots_data == NULL:
            free(control_points_data)
            free(knots_data)
            raise MemoryError("Failed to allocate memory for deserialization")

        # Copy control points data
        memcpy(control_points_data, current, n_ctrlpts * 4 * sizeof(double))
        current += n_ctrlpts * 4 * sizeof(double)

        # Copy knots data
        memcpy(knots_data, current, n_knots * sizeof(double))

        # Create memory views for the allocated arrays
        cdef double[:, :] control_points = <double[:n_ctrlpts, :4]>control_points_data
        cdef double[:] knots = <double[:n_knots]>knots_data

        # Create new NURBSCurve object
        cdef NURBSCurve new_curve = NURBSCurve.__new__(NURBSCurve)
        new_curve._control_points = control_points
        new_curve._knots = knots
        new_curve._degree = degree
        new_curve._periodic = periodic

        # Update interval and other necessary properties
        new_curve.knots_update_hook()

        return new_curve

    @staticmethod
    def deserialize(data):
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("Input must be bytes or bytearray")
        return NURBSCurve.cdeserialize(data)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def bbox(self,double[:,:] result=None):
        if result is None:
            result=np.zeros((2,3))
        aabb(self._control_points, result)
        return result

    def astuple(self):
        cdef tuple res=(self.control_points.tolist(), self.knots.tolist(), self._degree, self._periodic)
        return res
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] greville_abscissae(double[:] knots, int degree):
    cdef int n = knots.shape[0] - degree - 1
    cdef double[:] greville=np.empty((n,))
    cdef double temp
    cdef int i,j,k
    for i in range(n):
        temp=0.
        for k in range(degree):
            j = i + 1 + k
            temp+=knots[j]
        temp/= degree
        greville[i]=temp
    return greville


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple split_curve(NURBSCurve obj, double param, double tol=1e-7,bint normalize_knots=False):

    cdef int degree = obj._degree
    cdef double[:] knotvector = obj._knots
    cdef double[:, :] ctrlpts = obj._control_points
    cdef int n_ctrlpts = ctrlpts.shape[0]
    cdef int dim = ctrlpts.shape[1]
    cdef int ks, s, r, knot_span
    cdef int i, j
    cdef bint is_periodic = 0

    if param<=obj._interval[0] or param>=obj._interval[1] or fabs(param - obj._interval[0])<=tol or fabs(param -obj._interval[1])<=tol:


        raise ValueError("Cannot split from the domain edge ")

    ks = find_span_inline(n_ctrlpts , degree, param, knotvector, 0) - degree + 1

    s = find_multiplicity(param, knotvector,1e-12)
    r = degree - s


    # Insert knot
    cdef NURBSCurve temp_obj = obj.ccopy()


    temp_obj.insert_knot(param, r)
    cdef double[:, :] tcpts = temp_obj._control_points
    cdef double[:] temp_knots = temp_obj._knots
    # Knot vectors
    knot_span = find_span_inline(temp_obj._control_points.shape[0] , degree, param, temp_knots, 0) + 1
    #cdef double[:] curve1_kv = np.empty((knot_span + 1,), dtype=np.float64)
    #cdef double[:] curve2_kv = np.empty((temp_obj._knots.shape[0] - knot_span+  degree+ 1,), dtype=np.float64)
    cdef vector[double] surf1_kv = vector[double](knot_span )
    cdef vector[double] surf2_kv = vector[double](temp_knots.shape[0] - knot_span)


    for i in range(knot_span):
        surf1_kv[i] = temp_knots[i]
    for i in range(temp_knots.shape[0] - knot_span):

        surf2_kv[i] = temp_knots[i + knot_span]

    # Add param to the end of surf1_kv and beginning of surf2_kv

    surf1_kv.push_back(param)
    for j in range(degree+1):

        surf2_kv.insert(surf2_kv.begin(), param);





    # Create control points for the two new surfaces

    cdef double[:] surf1_kvm=np.empty(surf1_kv.size())
    cdef double[:] surf2_kvm=np.empty(surf2_kv.size())
    for i in range(surf1_kv.size()):
        surf1_kvm[i]=surf1_kv[i]
    for i in range(surf2_kv.size()):
        surf2_kvm[i]=surf2_kv[i]



    # Control points

    cdef double[:, :] curve1_ctrlpts = tcpts[:ks + r,  :]
    cdef double[:, :] curve2_ctrlpts = tcpts[ks + r - 1:,:]



    # Create new curves
    cdef NURBSCurve curve1 = NURBSCurve(curve1_ctrlpts.copy(), degree, surf1_kvm,0)
    cdef NURBSCurve curve2 = NURBSCurve(curve2_ctrlpts.copy(), degree,  surf2_kvm,0)
    if normalize_knots:
        curve1.cnormalize_knots()
        curve1.knots_update_hook()
        curve2.cnormalize_knots()
        curve2.knots_update_hook()

    return curve1, curve2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple split_curve2(NURBSCurve obj, double param, double tol=1e-12):
    cdef int degree = obj._degree
    cdef double[:] knotvector = obj._knots
    cdef double[:, :] ctrlpts = obj._control_points
    cdef int n_ctrlpts = ctrlpts.shape[0]
    cdef int dim = ctrlpts.shape[1]
    cdef int ks, s, r, knot_span
    cdef int i
    cdef bint is_periodic = 0

    if (param <= obj._interval[0] or param >= obj._interval[1] or
        fabs(param - obj._interval[0]) <= tol or fabs(param - obj._interval[1]) <= tol):
        raise ValueError("Cannot split from the domain edge")

    ks = find_span_inline(n_ctrlpts, degree, param, knotvector, 0) - degree + 1
    s = find_multiplicity(param, knotvector, 1e-12)
    r = degree - s

    # Insert knot
    cdef NURBSCurve temp_obj = obj.ccopy()
    temp_obj.insert_knot(param, r)
    cdef double[:, :] tcpts = temp_obj._control_points
    cdef double[:] temp_knots = temp_obj._knots

    # Knot vectors
    knot_span = find_span_inline(temp_obj._control_points.shape[0], degree, param, temp_knots, 0) + 1

    # Calculate sizes of new knot vectors
    cdef int nknots1 = knot_span + 1  # +1 to include param at the end
    cdef int nknots2 = (temp_knots.shape[0] - knot_span) + (degree + 1)  # + degree+1 for param repetitions

    cdef double[:] surf1_kv = np.empty(nknots1, dtype=np.float64)
    cdef double[:] surf2_kv = np.empty(nknots2, dtype=np.float64)

    # Fill surf1_kv
    for i in range(knot_span):
        surf1_kv[i] = temp_knots[i]
    surf1_kv[knot_span] = param  # Add param at the end

    # Fill surf2_kv
    for i in range(degree + 1):
        surf2_kv[i] = param  # Insert param at the beginning
    for i in range(temp_knots.shape[0] - knot_span):
        surf2_kv[i + degree + 1] = temp_knots[i + knot_span]

    # Control points
    cdef double[:, :] curve1_ctrlpts = tcpts[:ks + r, :]
    cdef double[:, :] curve2_ctrlpts = tcpts[ks + r - 1:, :]

    # Create new curves
    cdef NURBSCurve curve1 = NURBSCurve(curve1_ctrlpts.copy(), degree, surf1_kv, 0)
    cdef NURBSCurve curve2 = NURBSCurve(curve2_ctrlpts.copy(), degree, surf2_kv, 0)

    return curve1, curve2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef list split_curve_multiple(NURBSCurve crv, double[:] params):
    cdef list crvs = []
    cdef NURBSCurve temp=crv
    cdef int i
    cdef tuple tpl
    for i in range(params.shape[0]):
        tpl = split_curve2(crv, params[i] )
        crv=tpl[1]
        crvs.append(tpl[0])
    crvs.append(crv)
    return crvs


cdef class NURBSSurface(ParametricSurface):
    def __init__(self, double[:,:,:] control_points, tuple degree, double[:] knots_u=None, double[:] knots_v=None ):
        super().__init__()
        self._interval=np.zeros((2,2))
        self._size =[control_points.shape[0],control_points.shape[1]]
        self._degree=[degree[0],degree[1]]

        cdef int cpt_count=  self._size[0]*  self._size[1]
        self._control_points_arr=<double*>malloc(self._size[0]*self._size[1]*4*sizeof(double))
        self.control_points_view=<double[:control_points.shape[0],:control_points.shape[1],:4 ]>self._control_points_arr
        self.control_points_flat_view=<double[:cpt_count,:4 ]>self._control_points_arr
        cdef int i, j, k
        if control_points.shape[2]<4:
            for i in range(cpt_count):
                self.control_points_flat_view[i][3]=1.

        for i in range(control_points.shape[0]):
            for j in range(control_points.shape[1]):
                for k in range(control_points.shape[2]):
                    self.control_points_view[i][j][k]=control_points[i][j][k]




        if knots_u is None:
            self.generate_knots_u()
        else:
            self._knots_u=knots_u
        if knots_v is None:
            self.generate_knots_v()
        else:
            self._knots_v=knots_v
        self._update_interval()
    @property
    def knots_u(self):
        return np.array(self._knots_u)
    @knots_u.setter
    def knots_u(self,val):
        self._knots_u=val
        self._update_interval()

    @property
    def knots_v(self):
        return np.array(self._knots_v)
    @property
    def control_points(self):
        cdef double w;
        cdef int i,j
        cdef double[:,:,:] pts=self.control_points_view[...,:3].copy()
        for i in range(self.control_points_view.shape[0]):
            for j in range(self.control_points_view.shape[1]):
                w=self.control_points_view[i,j,3]
                if w!=1.:

                    pts[i,j,0]/=w
                    pts[i,j,1]/=w
                    pts[i,j,2]/=w

        return pts


    @property
    def control_points_flat(self):
        cdef double w;
        cdef int i
        cdef double[:,:] pts=self.control_points_flat_view[...,:3].copy()
        for i in range(self.control_points_flat_view.shape[0]):
            w=self.control_points_flat_view[i,3]
            if w!=1.:
                pts[i,0]/=w
                pts[i,1]/=w
                pts[i,2]/=w
        return pts
    @property
    def control_points_flat_w(self):

        return self.control_points_flat_view
    @property
    def shape(self):
        return tuple(self._size)
    @property
    def control_points_w(self):
        return self.control_points_view

    @knots_v.setter
    def knots_v(self,val):
        self._knots_v=val
        self._update_interval()
    @property
    def degree(self):
        cdef int[:] dg=self._degree
        return dg
    @degree.setter
    def degree(self, val):
        self._degree[0]=val[0]
        self._degree[1]=val[1]
        self.generate_knots_u()
        self.generate_knots_v()


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _update_interval(self) noexcept nogil:
        self._interval[0][0] = self._knots_u[self._degree[0]]
        self._interval[0][1] = self._knots_u[self._knots_u.shape[0] - self._degree[0]-1 ]
        self._interval[1][0] = self._knots_v[self._degree[1]]
        self._interval[1][1] = self._knots_v[self._knots_v.shape[0] - self._degree[1]-1 ]
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void generate_knots_u(self):
        """
        This function generates default knots based on the number of control points
        :return: A numpy array of knots

        Notes
        ------
        **Difference with OpenNURBS**

        OpenNURBS uses a knots vector shorter by one knot on each side. 
        The original explanation can be found in `opennurbs/opennurbs_evaluate_nurbs.h`.
        [source](https://github.com/mcneel/opennurbs/blob/19df20038249fc40771dbd80201253a76100842c/opennurbs_evaluate_nurbs.h#L116-L148)
        mmcore uses the standard knotvector length according to DeBoor and The NURBS Book.

        **Difference with geomdl**

        Unlike geomdl, the knots vector is not automatically normalised from 0 to 1.
        However, there are no restrictions on the use of the knots normalised vector. 

        """
        cdef int nu = self._size[0]


        self._knots_u = np.concatenate((
            np.zeros(self._degree[0] + 1),
            np.arange(1, nu - self._degree[0]),
            np.full(self._degree[0] + 1, nu - self._degree[0])
        ))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void generate_knots_v(self):
        """
        This function generates default knots based on the number of control points
        :return: A numpy array of knots

        Notes
        ------
        **Difference with OpenNURBS**

        OpenNURBS uses a knots vector shorter by one knot on each side. 
        The original explanation can be found in `opennurbs/opennurbs_evaluate_nurbs.h`.
        [source](https://github.com/mcneel/opennurbs/blob/19df20038249fc40771dbd80201253a76100842c/opennurbs_evaluate_nurbs.h#L116-L148)
        mmcore uses the standard knotvector length according to DeBoor and The NURBS Book.

        **Difference with geomdl**

        Unlike geomdl, the knots vector is not automatically normalised from 0 to 1.
        However, there are no restrictions on the use of the knots normalised vector. 

        """
        cdef int nv = self._size[1]


        self._knots_v = np.concatenate((
            np.zeros(self._degree[1]+ 1),
            np.arange(1, nv - self._degree[1]),
            np.full(self._degree[1] + 1, nv - self._degree[1])
        ))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cevaluate(self, double u, double v,double[:] result) noexcept nogil:
        surface_point(self._size[0]-1,self._degree[0],self._knots_u,self._size[1]-1,self._degree[1],self._knots_v, self.control_points_view, u,v, 0, 0, result)



    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef NURBSSurface ccopy(self) :
        cdef int n_ctrlpts_u = self._size[0]
        cdef int n_ctrlpts_v = self._size[1]
        cdef int n_knots_u = self._knots_u.shape[0]
        cdef int n_knots_v = self._knots_v.shape[0]

        # Allocate memory for new arrays
        cdef double* new_control_points_arr = <double*>malloc(n_ctrlpts_u * n_ctrlpts_v * 4 * sizeof(double))
        cdef double* new_knots_u = <double*>malloc(n_knots_u * sizeof(double))
        cdef double* new_knots_v = <double*>malloc(n_knots_v * sizeof(double))

        if new_control_points_arr == NULL or new_knots_u == NULL or new_knots_v == NULL:
            free(new_control_points_arr)
            free(new_knots_u)
            free(new_knots_v)

            raise MemoryError("Failed to allocate memory for surface copy")

        # Copy data using memcpy for efficiency
        memcpy(new_control_points_arr, self._control_points_arr, n_ctrlpts_u * n_ctrlpts_v * 4 * sizeof(double))
        memcpy(new_knots_u, &self._knots_u[0], n_knots_u * sizeof(double))
        memcpy(new_knots_v, &self._knots_v[0], n_knots_v * sizeof(double))

        # Create new NURBSSurface object
        cdef NURBSSurface new_surface = NURBSSurface.__new__(NURBSSurface)
        new_surface._control_points_arr = new_control_points_arr
        new_surface.control_points_view = <double[:n_ctrlpts_u, :n_ctrlpts_v, :4]>new_control_points_arr
        new_surface.control_points_flat_view = <double[:n_ctrlpts_u*n_ctrlpts_v, :4]>new_control_points_arr
        new_surface._knots_u = <double[:n_knots_u]>new_knots_u
        new_surface._knots_v = <double[:n_knots_v]>new_knots_v
        new_surface._degree[0] = self._degree[0]
        new_surface._degree[1] = self._degree[1]
        new_surface._size[0] = self._size[0]
        new_surface._size[1] = self._size[1]

        # Copy interval
        new_surface._interval = self._interval.copy()

        return new_surface

    cdef void realloc_control_points(self, size_t new_size_u, size_t new_size_v ) noexcept nogil:

        self._control_points_arr= <double*> realloc(self._control_points_arr, new_size_u*new_size_v*4*sizeof(double))
        with gil:
            self.control_points_view=<double[:new_size_u,:new_size_v,:4]>self._control_points_arr
            self.control_points_flat_view = <double[:(new_size_u*new_size_v), :4]>self._control_points_arr
        self._size[0]=new_size_u
        self._size[1] = new_size_v

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cnormalize_knots_u(self) noexcept nogil:

        cdef double mx=self._knots_u[self._knots_u.shape[0]-1]
        cdef double mn=self._knots_u[0]
        cdef double d=mx-mn
        cdef int i

        for i in range(self._knots_u.shape[0]):
            self._knots_u[i]=((self._knots_u[i]-mn)/d)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cnormalize_knots_v(self) noexcept nogil:
        cdef double mx = self._knots_v[self._knots_v.shape[0] - 1]
        cdef double mn = self._knots_v[0]
        cdef double d = mx - mn
        cdef int i
        for i in range(self._knots_v.shape[0]):
            self._knots_v[i] = ((self._knots_v[i] - mn) / d)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cnormalize_knots(self) noexcept nogil:
        self.cnormalize_knots_u()
        self.cnormalize_knots_v()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def normalize_knots(self) :
        self.cnormalize_knots()
        self._update_interval()



    def copy(self):
        cdef NURBSSurface new_surface = self.ccopy()
        # If there are any Python-level attributes that need to be copied, do it here
        # For example, if there was a cached evaluation function:
        # new_surface._evaluate_cached = functools.lru_cache(maxsize=None)(new_surface._evaluate)
        return new_surface



    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void insert_knot_u(self, double t, int count):
        cdef int v

        cdef int new_count_u = self._size[0] + count
        cdef int new_count_v = self._size[1]

        cdef double[:,:,:] cpts=self.control_points_view.copy()
        cdef int span = find_span_inline(self._size[0]-1,
            self._degree[0], t, self._knots_u, 0
        )

        # Compute new knot vector
        cdef double[:] k_v = knot_insertion_kv(self._knots_u, t, span, count)
        cdef int s_u = find_multiplicity(t, self._knots_u,1e-12)

        self._control_points_arr= <double*> realloc(self._control_points_arr, new_count_u*new_count_v*4*sizeof(double))

        self.control_points_view=<double[:new_count_u,:new_count_v,:4]>self._control_points_arr
        self.control_points_flat_view = <double[:(new_count_u*new_count_v), :4]>self._control_points_arr


        for v in range(self._size[1]):

            knot_insertion(
                self._degree[0],
            self._knots_u,
            cpts[:,v,:],
            t,
            count,
            s_u,
            span,0,  self.control_points_view[:,v,:])

        # Update surface properties
        #free(&self._knots_u[0])
        self._knots_u = k_v
        self._size[0] = new_count_u
        self._size[1] = new_count_v
        self._update_interval()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cbbox(self, double[:,:] result) noexcept nogil:
        result[:]=0.
        aabb(self.control_points_flat_view, result)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double[:,:] bbox(self):

        cdef double[:,:] bb=np.empty((2,3))


        aabb(self.control_points_flat_view, bb)
        return bb

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void insert_knot_v(self, double t, int count) :
        cdef int u


        cdef int new_count_u = self._size[0]
        cdef int new_count_v = self._size[1] + count
        cdef double[:,:,:] cpts=self.control_points_view.copy()

        cdef int span = find_span_inline(self._size[1]-1,
            self._degree[1], t, self._knots_v, 0
        )

        # Compute new knot vector
        cdef double[:] k_v = knot_insertion_kv(self._knots_v, t, span, count)
        cdef int s_v = find_multiplicity(t, self._knots_v,1e-12)

        self._control_points_arr= <double*> realloc(self._control_points_arr, new_count_u*new_count_v*4*sizeof(double))

        self.control_points_view=<double[:new_count_u,:new_count_v,:4]>self._control_points_arr
        self.control_points_flat_view = <double[:(new_count_u*new_count_v), :4]>self._control_points_arr


        for u in range(self._size[0]):

            knot_insertion(self._degree[1],
            self._knots_v,
            cpts[u,:,:],
            t,
            count,
            s_v,
            span,0,self.control_points_view[u,:,:])

        #free(&self._knots_v[0])
        self._knots_v = k_v
        self._size[0] =new_count_u
        self._size[1] =new_count_v
        self._update_interval()

    def __dealloc__(self):
        if self._control_points_arr !=NULL:
            free(self._control_points_arr)





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple split_surface_u(NURBSSurface obj, double param, double tol=1e-7) :
    cdef int size_u = obj.control_points_view.shape[0]
    cdef int size_v = obj.control_points_view.shape[1]
    cdef int ks, s, r, knot_span
    cdef double[:, :, :] cpts = obj.control_points_view
    cdef double[:] knots_u = obj._knots_u
    cdef double[:] knots_v = obj._knots_v
    cdef int degree_u = obj._degree[0]
    cdef int degree_v = obj._degree[1]
    if param<=obj._interval[0][0] or param>=obj._interval[0][1] or fabs(param - obj._interval[0][0])<=tol or fabs(param -obj._interval[0][1])<=tol:


        raise ValueError("Cannot split from the domain edge")
    ks = find_span_inline(size_u, degree_u, param, knots_u, 0) - degree_u + 1
    s = find_multiplicity(param, knots_u, 1e-12)
    r = degree_u - s

    # Create a copy of the original surface and insert knot
    cdef NURBSSurface temp_obj = obj.ccopy()
    temp_obj.insert_knot_u(param, r)

    cdef double[:, :, :] tcpts = temp_obj.control_points_view
    cdef double[:] temp_knots_u = temp_obj._knots_u

    knot_span = find_span_inline(temp_obj._size[0], degree_u, param, temp_knots_u, 0) + 1

    # Create knot vectors for the two new surfaces
    cdef vector[double] surf1_kv = vector[double](knot_span )
    cdef vector[double] surf2_kv = vector[double](temp_knots_u.shape[0] - knot_span)

    cdef int i
    for i in range(knot_span):
        surf1_kv[i] = temp_knots_u[i]
    for i in range(temp_knots_u.shape[0] - knot_span):

        surf2_kv[i] = temp_knots_u[i + knot_span]

    # Add param to the end of surf1_kv and beginning of surf2_kv
    cdef int j
    surf1_kv.push_back(param)
    for j in range(degree_u+1):

        surf2_kv.insert(surf2_kv.begin(), param);





    # Create control points for the two new surfaces
    cdef double[:, :, :] surf1_ctrlpts = tcpts[:ks + r, :, :]
    cdef double[:, :, :] surf2_ctrlpts = tcpts[ks + r - 1:, :, :]

    cdef double[:] surf1_kvm=np.empty(surf1_kv.size())
    cdef double[:] surf2_kvm=np.empty(surf2_kv.size())
    for i in range(surf1_kv.size()):
        surf1_kvm[i]=surf1_kv[i]
    for i in range(surf2_kv.size()):
        surf2_kvm[i]=surf2_kv[i]

    # Create new surfaces
    cdef NURBSSurface surf1 = NURBSSurface(np.asarray(surf1_ctrlpts.copy()), (degree_u, degree_v),surf1_kvm,knots_v.copy() )
    cdef NURBSSurface surf2 = NURBSSurface(np.asarray(surf2_ctrlpts.copy()), (degree_u, degree_v), surf2_kvm, knots_v.copy())


    return surf1, surf2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple split_surface_v(NURBSSurface obj, double param,double tol=1e-7) :
    cdef int size_u = obj.control_points_view.shape[0]
    cdef int size_v = obj.control_points_view.shape[1]
    cdef int ks, s, r, knot_span
    cdef double[:, :, :] cpts = obj.control_points_view
    cdef double[:] knots_u = obj._knots_u
    cdef double[:] knots_v = obj._knots_v
    cdef int degree_u = obj._degree[0]
    cdef int degree_v = obj._degree[1]
    if param <= obj._interval[1][0] or param >= obj._interval[1][1] or fabs(param - obj._interval[1][0]) <= tol or fabs(
            param - obj._interval[1][1]) <= tol:
        raise ValueError("Cannot split from the domain edge")
    ks = find_span_inline(size_v, degree_v, param, knots_v, 0) - degree_v + 1
    s = find_multiplicity(param, knots_v,1e-12)
    r = degree_v - s

    # Create a copy of the original surface and insert knot
    cdef NURBSSurface temp_obj = obj.ccopy()
    temp_obj.insert_knot_v(param, r)

    cdef double[:, :, :] tcpts = temp_obj.control_points_view
    cdef double[:] temp_knots_v = temp_obj._knots_v

    knot_span = find_span_inline(temp_obj._size[1], degree_v, param, temp_knots_v, 0) + 1

    # Create knot vectors for the two new surfaces
    cdef vector[double] surf1_kv = vector[double](knot_span )
    cdef vector[double] surf2_kv = vector[double](temp_knots_v.shape[0] - knot_span)

    cdef int i
    for i in range(knot_span):
        surf1_kv[i] = temp_knots_v[i]
    for i in range(temp_knots_v.shape[0] - knot_span):

        surf2_kv[i] = temp_knots_v[i + knot_span]

    # Add param to the end of surf1_kv and beginning of surf2_kv
    cdef int j
    surf1_kv.push_back(param)
    for j in range(degree_v+1):

        surf2_kv.insert(surf2_kv.begin(), param);





    # Create control points for the two new surfaces
    cdef double[:, :, :] surf1_ctrlpts = tcpts[:,:ks + r, :]
    cdef double[:, :, :] surf2_ctrlpts = tcpts[:,ks + r - 1:, :]

    cdef double[:] surf1_kvm=np.empty(surf1_kv.size())
    cdef double[:] surf2_kvm=np.empty(surf2_kv.size())

    for i in range(surf1_kv.size()):
        surf1_kvm[i]=surf1_kv[i]
    for i in range(surf2_kv.size()):
            surf2_kvm[i]=surf2_kv[i]

    # Create new surfaces
    cdef NURBSSurface surf1 = NURBSSurface(surf1_ctrlpts.copy(), (degree_u, degree_v), knots_u.copy(), surf1_kvm)
    cdef NURBSSurface surf2 = NURBSSurface(surf2_ctrlpts.copy(), (degree_u, degree_v), knots_u.copy(),surf2_kvm)

    return surf1, surf2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple subdivide_surface(NURBSSurface surface, double u=0.5,double v=0.5,double tol=1e-7,bint normalize_knots=True):
    cdef tuple surfs1 = split_surface_u(surface, u,tol)
    cdef NURBSSurface surf1, surf2, surf11, surf12, surf21, surf22
    surf1= surfs1[0]
    surf2 = surfs1[1]
    if normalize_knots:
        surf1.normalize_knots()
        surf2.normalize_knots()


    surf11, surf12 = split_surface_v(surf1,v,tol)
    surf21, surf22 = split_surface_v(surf2,v,tol)
    if normalize_knots:
        surf11.normalize_knots()
        surf12.normalize_knots()
        surf21.normalize_knots()
        surf22.normalize_knots()
    return surf11,surf12,surf21,surf22


cdef inline list decompose_direction(NURBSSurface srf, int idx,double tol=1e-7):
        cdef list srf_list = []
        cdef double[:] knots = srf._knots_u if idx == 0 else srf._knots_v
        cdef int degree = srf._degree[idx]
        cdef double[:]  unique_knots = np.sort(np.unique(knots[degree + 1 : -(degree + 1)]))

        while unique_knots.shape[0]>0:
            knot = unique_knots[0]
            if idx == 0:
                srfs = split_surface_u(srf, knot, tol)
            else:
                srfs = split_surface_v(srf, knot,tol)
            srf_list.append(srfs[0])
            srf = srfs[1]
            unique_knots = unique_knots[1:]
        srf_list.append(srf)
        return srf_list



@cython.boundscheck(False)
@cython.cdivision(True)
def decompose_surface(surface, decompose_dir="uv",normalize_knots=True):
    def decompose_direction(srf, idx):
        srf_list = []
        knots = srf.knots_u if idx == 0 else srf.knots_v
        degree = srf.degree[idx]
        unique_knots = sorted(set(knots[degree + 1 : -(degree + 1)]))

        while unique_knots:
            knot = unique_knots[0]
            if idx == 0:
                srfs = split_surface_u(srf, knot)
            else:
                srfs = split_surface_v(srf, knot)
            srf_list.append(srfs[0])
            srf = srfs[1]
            unique_knots = unique_knots[1:]

        srf_list.append(srf)
        return srf_list

    if not isinstance(surface, NURBSSurface):
        raise ValueError("Input must be an instance of NURBSSurface class")

    surf = surface.copy()
    surf.normalize_knots()
    if decompose_dir == "u":
        surfs_u=decompose_direction(surf, 0)
        if normalize_knots:
            for srg in surfs_u:
                srg.normalize_knots()
        return surfs_u
    elif decompose_dir == "v":
        surfs_v=decompose_direction(surf, 1)
        if normalize_knots:
            for srg in surfs_v:
                srg.normalize_knots()
        return surfs_v


    elif decompose_dir == "uv":
        multi_surf = []
        surfs_u = decompose_direction(surf, 0)

        for sfu in surfs_u:
            dsf=decompose_direction(sfu, 1)
            if normalize_knots:
                for srg in dsf:
                    srg.normalize_knots()
                    multi_surf.append(srg)
            else:
                multi_surf+=dsf
        return multi_surf
    else:
        raise ValueError(
            f"Cannot decompose in {decompose_dir} direction. Acceptable values: u, v, uv"
        )


cdef class CurveCurveEq:
    cdef public NURBSCurve curve1
    cdef public NURBSCurve curve2
    __slots__=['curve1', 'curve2']

    def __init__(self, NURBSCurve curve1, NURBSCurve curve2):
        self.curve1=curve1
        self.curve2=curve2

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double evaluate(self, double[:] x):

        cdef double t=x[0]
        cdef double s = x[1]

        cdef double[:,:] pts = np.zeros((3,3))
        self.curve1.cevaluate(t,pts[0])
        self.curve2.cevaluate(s,pts[1])
        pts[2,0]= pts[0,0]-pts[1,0]
        pts[2,1]= pts[0,1] - pts[1,1]
        pts[2, 2] = pts[0, 2] - pts[1, 2]
        cdef double res=pts[2,0]*pts[2,0]+pts[2,1]*pts[2,1]+pts[2,2]*pts[2,2]
        return res
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __call__(self,double[:] x):
        return self.evaluate(x)

cdef class CurveSurfaceEq:
    cdef public NURBSCurve curve
    cdef public NURBSSurface surface
    __slots__=['curve', 'surface']

    def __init__(self, NURBSCurve curve, NURBSSurface surface):
        self.curve=curve
        self.surface=surface
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double evaluate(self, double[:] x):

        cdef double t=x[0]
        cdef double u = x[1]
        cdef double v = x[2]
        cdef double[:,:] pts = np.zeros((3,3))
        self.curve.cevaluate(t,pts[0])
        self.surface.cevaluate(u,v,pts[1])
        pts[2,0]= pts[0,0]-pts[1,0]
        pts[2,1]= pts[0,1] - pts[1,1]
        pts[2, 2] = pts[0, 2] - pts[1, 2]
        cdef double res=pts[2,0]*pts[2,0]+pts[2,1]*pts[2,1]+pts[2,2]*pts[2,2]
        return res
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __call__(self,double[:] x):
        return self.evaluate(x)


cdef class SurfaceSurfaceEq:
    cdef public NURBSSurface surface1
    cdef public NURBSSurface surface2
    __slots__=['surface1', 'surface2']
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __init__(self, NURBSSurface surf1, NURBSSurface surf2):
        self.surface1=surf1
        self.surface2=surf2
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double evaluate(self, double[:] x):

        cdef double s =x[0]
        cdef double t = x[1]
        cdef double u = x[2]
        cdef double v = x[3]
        cdef double[:,:] pts = np.zeros((3,3))
        self.surface1.cevaluate(s,t,pts[0])
        self.surface2.cevaluate(u,v,pts[1])
        pts[2,0]= pts[0,0]-pts[1,0]
        pts[2,1]= pts[0,1] - pts[1,1]
        pts[2, 2] = pts[0, 2] - pts[1, 2]
        cdef double res=pts[2,0]*pts[2,0]+pts[2,1]*pts[2,1]+pts[2,2]*pts[2,2]
        return res
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __call__(self,double[:] x):
        return self.evaluate(x)




def extract_isocurve(
        surface: NURBSSurface, param: float, direction: str = "u"
) -> NURBSCurve:
    """
    Extract an isocurve from a NURBS surface at a given parameter in the u or v direction.

    Args:
    surface (NURBSSurface): The input NURBS surface.
    param (float): The parameter value at which to extract the isocurve.
    direction (str): The direction of the isocurve, either 'u' or 'v'. Default is 'u'.

    Returns:
    NURBSCurve: The extracted isocurve as a NURBS curve.

    Raises:
    ValueError: If the direction is not 'u' or 'v', or if the param is out of range.
    """
    if direction not in ["u", "v"]:
        raise ValueError("Direction must be either 'u' or 'v'.")

    cdef double[:,:] interval = surface._interval
    if direction == "u":
        knots = surface.knots_v
        degree = surface.degree[0]
        param_range = interval[1]
        n = surface.shape[1] - 1
        m = surface.shape[0]
    else:  # direction == 'v'
        knots = surface.knots_u
        degree = surface.degree[1]
        param_range = interval[0]
        n = surface.shape[0] - 1
        m = surface.shape[1]

    if param < param_range[0] or param > param_range[1]:
        raise ValueError(f"Parameter {param} is out of range {param_range}")

    span = find_span(n, degree, param, knots, 0)
    basis = basis_functions(span, param, degree, knots)

    control_points = np.zeros((m, 4))

    if direction == "u":
        for i in range(m):
            for j in range(degree + 1):
                idx = min(max(span - degree + j, 0), n)
                control_points[i] += np.asarray(basis[j]) * np.asarray(surface.control_points_w[i, idx, :])
    else:  # direction == 'v'
        for i in range(m):
            for j in range(degree + 1):
                idx = min(max(span - degree + j, 0), n)

                control_points[i] += np.asarray(basis[j]) * np.asarray(surface.control_points_w[idx, i, :])

    return NURBSCurve(control_points, degree, knots)
