#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
#cython: initializedcheck=False
# distutils: language = c++
import functools

cimport cython
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc,free
from libcpp.vector cimport vector
from mmcore.geom.parametric cimport ParametricCurve
from mmcore.numeric cimport vectors,calgorithms
cimport mmcore.geom.nurbs.algorithms
from mmcore.numeric.algorithms.quicksort cimport uniqueSorted
from libc.math cimport fabs, sqrt,fmin,fmax,pow
from libc.string cimport memcpy
cnp.import_array()


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int find_span(int n, int p, double u, double[:] U, bint is_periodic) noexcept nogil:
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
cdef void basis_funs(int i, double u, int p, double[:] U, double* N) noexcept nogil:
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
cdef void curve_point(int n, int p, double[:] U, double[:, :] P, double u, double* result,bint is_periodic) noexcept nogil:
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
cpdef double[:, :] ders_basis_funs(int i, double u, int p, int n, double[:] U):
    """
    Compute the nonzero basis functions and their derivatives.
    """

    cdef int pp=p+1
    cdef int nn = n + 1
    cdef int s1, s2
    cdef double[:, :] ders = np.zeros((nn, pp))
    cdef double[:, :] ndu = np.zeros((pp,pp))
    cdef double[:] left = np.zeros(pp)
    cdef double[:] right = np.zeros(pp)
    cdef double[:, :] a = np.zeros((2, pp))
    ndu[0, 0] = 1.0

    cdef int j, r, k, rk, pk, j1, j2
    cdef double saved, temp, d
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
    cdef int span = find_span(n, p, u, U,is_periodic)
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

    cdef int span = find_span(n, degree, u, knotvector,is_periodic)
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
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef int find_multiplicity(double knot, double[:] knot_vector, double tol=1e-07):
    cdef int mult=0
    cdef int l=knot_vector.shape[0]
    cdef int i
    cdef double difference
    for i in range(l):
        difference=knot - knot_vector[i]
        if fabs(difference) <= tol:
            mult += 1
    return mult
from libcpp.vector cimport vector
cdef class KnotVector:
   
    cdef vector[double] data
    def __init__(self, int size=0) -> None:
        
        self.data.resize(size)
    @staticmethod
    cdef KnotVector from_array(double[:] arr):
        cdef KnotVector vec=KnotVector.__new__(KnotVector)
        cdef int sz=arr.shape[0]
        vec.__init__(sz)
        cdef int i
        for i in range(sz):
            vec.data[i]=arr[i]
        return vec
    
    def __getitem__(self, item):
        cdef int i, j,sz
        cdef KnotVector sliced_vec
        if isinstance(item, int):
            if item < 0:
                item += self.data.size()
            if item < 0 or item >= self.data.size():
                raise IndexError("Index out of range")
            return self.data[item]
        elif isinstance(item, slice):
            start, stop, step = item.indices(self.data.size())
            sliced_vec = KnotVector.__new__(KnotVector)
            sz = len(range(start, stop, step))
            sliced_vec.__init__(sz)
           
            for i, j in enumerate(range(start, stop, step)):
                sliced_vec.data[i] = self.data[j]
            return sliced_vec
        else:
            raise TypeError("Invalid argument type")

    def __setitem__(self, item, value):
        cdef int i, j
        if isinstance(item, int):
            if item < 0:
                item += self.data.size()
            if item < 0 or item >= self.data.size():
                raise IndexError("Index out of range")
            self.data[item] = value
        elif isinstance(item, slice):
           
            start, stop, step = item.indices(self.data.size())
            if not isinstance(value, KnotVector) or len(range(start, stop, step)) != value.get_size():
                raise ValueError("Slice assignment must be of the same size")
            
            for i, j in enumerate(range(start, stop, step)):
                self.data[j] = value.data[i]
        else:
            raise TypeError("Invalid argument type")
   
            
    
    @cython.wraparound(False)
    cpdef void extend_array(self, double[:] arr):
        cdef int n=arr.shape[0]
        cdef int current=self.data.size()
        cdef int new_size=n+current
        self.data.resize(new_size)
        cdef int i
        for i in range( new_size):
            self.data[current+i]=arr[i]
    
    cpdef int get_size(self):
        return self.data.size()
    @cython.cdivision(True) 
    cpdef double[:] to_memview(self):

      
        cdef int current=self.data.size()
        # cython: error=False
        cdef double[:] arr=<double[:current]>(<double*>malloc(sizeof(double)* current))
        cdef int i
        for i in range( current):
            self.data[current+i]=arr[i]
        return arr

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef knot_insertion(int degree, double[:] knotvector, double[:, :] ctrlpts, double u, int num=1,bint is_periodic=0):
    cdef int s = find_multiplicity(u, knotvector)
    cdef int n = ctrlpts.shape[0]
    cdef int k = find_span_inline( n, degree,  u, knotvector, is_periodic)
    
    
    cdef int nq = n + num
    cdef int dim = ctrlpts.shape[1]
    
    cdef double[:, :] ctrlpts_new = np.zeros((nq, dim), dtype=np.float64)
    cdef double* temp = <double*>malloc(sizeof(double) * (degree + 1) * dim)
    
    cdef int i, j, L, idx
    cdef double alpha
    
    for i in range(k - degree + 1):
        ctrlpts_new[i] = ctrlpts[i]
    for i in range(k - s, n):
        ctrlpts_new[i + num] = ctrlpts[i]
    
    for i in range(degree - s + 1):
        memcpy(&temp[i * dim], &ctrlpts[k - degree + i, 0], sizeof(double) * dim)
    
    for j in range(1, num + 1):
        L = k - degree + j
        for i in range(degree - j - s + 1):
            alpha = knot_insertion_alpha(u, knotvector, k, i, L)
            for idx in range(dim):
                temp[i * dim + idx] = alpha * temp[(i + 1) * dim + idx] + (1.0 - alpha) * temp[i * dim + idx]
        memcpy(&ctrlpts_new[L, 0], &temp[0], sizeof(double) * dim)
        memcpy(&ctrlpts_new[k + num - j - s, 0], &temp[(degree - j - s) * dim], sizeof(double) * dim)
    
    L = k - degree + num
    for i in range(L + 1, k - s):
        memcpy(&ctrlpts_new[i, 0], &temp[(i - L) * dim], sizeof(double) * dim)
    
    free(temp)
    return np.asarray(ctrlpts_new)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] knot_insertion_kv(double[:] knotvector, double u, int span, int r):
    cdef int kv_size = knotvector.shape[0]
    cdef double[:] kv_updated = np.zeros(kv_size + r, dtype=np.float64)
    
    cdef int i
    for i in range(span + 1):
        kv_updated[i] = knotvector[i]
    for i in range(1, r + 1):
        kv_updated[span + i] = u
    for i in range(span + 1, kv_size):
        kv_updated[i + r] = knotvector[i]
    
    return kv_updated

cdef inline double point_distance(double* a, double* b ,int dim):
    cdef int i
    cdef double temp=0.
    for i in range(dim):

        temp+= pow(a[i]+b[i], 2)

    return sqrt(temp)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef knot_removal(int degree, double[:] knotvector, double[:, :] ctrlpts, double u, double tol=1e-4, int num=1,bint is_periodic=0) noexcept:
    cdef int s = find_multiplicity(u, knotvector)
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
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple knot_refinement(int degree, double[:] knotvector, double[:, :] ctrlpts, double[:] knot_list=None,  double[:]  add_knot_list=None, int density=1, bint is_periodic=0) :
    cdef int n = ctrlpts.shape[0] - 1
    cdef int m = n + degree + 1
    cdef int dim = ctrlpts.shape[1]
    cdef double alpha
    cdef int d, i,idx

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
            
        rknots[-1] = knot_list[-1]
        knot_list = rknots
        usz = rknots_size
    cdef double[:] X = np.zeros(knot_list.shape[0] * degree, dtype=np.float64)
    cdef int x_count = 0
    cdef int s, r
    for mk in knot_list:
        s = find_multiplicity(mk, knotvector)
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
    
    cdef int j, k, l
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

