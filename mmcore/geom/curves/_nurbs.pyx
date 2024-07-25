# cython: language_level=3
import functools

cimport cython
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc,free
from mmcore.geom.parametric cimport ParametricCurve
from mmcore.numeric cimport vectors,calgorithms
from mmcore.geom.curves.deboor cimport cdeboor,cevaluate_nurbs,xyz
from libc.math cimport fabs, sqrt,fmin,fmax,pow
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int find_span(int n, int p, double u, double[:] U, bint is_periodic) noexcept nogil:
    """
    Determine the knot span index.
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

    #return np.asarray(PK)

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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void projective_to_cartesian(double[:] point, double[:] result)  noexcept nogil:
    cdef double w = point[3]
    result[0]=point[0]/w
    result[1]=point[1]/w
    result[2]=point[2]/w

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void projective_to_cartesian_ptr_ptr(double* point, double* result)  noexcept nogil:
    cdef double w = point[3]
    result[0]=point[0]/w
    result[1]=point[1]/w
    result[2]=point[2]/w
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void projective_to_cartesian_ptr_mem(double* point, double[:] result)  noexcept nogil:
    cdef double w = point[3]
    result[0]=point[0]/w
    result[1]=point[1]/w
    result[2]=point[2]/w



cdef class NURBSpline(ParametricCurve):
    cdef public double[:,:] _control_points
    cdef public int _degree
    cdef double[:] _knots
    cdef bint _periodic
    cdef public object _evaluate_cached


    def __reduce__(self):
        return (self.__class__, (np.asarray(self._control_points),self._degree,np.asarray(self._knots),self._periodic))
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
        self._update_interval()
        self._evaluate_cached = functools.lru_cache(maxsize=None)(self._evaluate)
        if  periodic:
            self.make_periodic()





    def __deepcopy__(self, memodict={}):
        obj=self.__class__(control_points=self.control_points.copy(), degree=self._degree, knots=np.asarray(self._knots).copy())

        obj.weights=self.weights
        if self._periodic:
            obj.make_periodic()

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
        state['_degree'] = self._degree,
        state['_periodic']=self._periodic

        return state

    def __setstate__(self,state):
        self._control_points=state['_control_points']
        self._knots = state['_knots']
        self._degree = state['_degree']
        self._periodic = state['_periodic']

        self._update_interval()

    @property
    def periodic(self):
        return self._periodic
    @property
    def control_points(self):
        return np.asarray(self._control_points[:,:-1])
    @control_points.setter
    def control_points(self, double[:,:] control_points):
        self._control_points = np.ones((control_points.shape[0],4))

        self._control_points[:, :-1] = control_points
        self._evaluate_cached.cache_clear()


    @property
    def knots(self):
        return np.asarray(self._knots)
    @knots.setter
    def knots(self, double[:] v):
        self._knots=v
        self._evaluate_cached.cache_clear()

    @property
    def weights(self):
        return np.asarray(self._control_points[:, 3])
    @weights.setter
    def weights(self, double[:] v):
        self._control_points[:, 3]=v
        self._evaluate_cached.cache_clear()
    cdef void generate_knots(self):
        """
        This function generates default knots based on the number of control points
        :return: A numpy array of knots
        """
        cdef int n = len(self._control_points)
        self._knots = np.concatenate((
            np.zeros(self._degree + 1),
            np.arange(1, n - self._degree),
            np.full(self._degree + 1, n - self._degree)
        ))


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

    cdef _update_interval(self):
        self._interval[0] = self._knots[self._degree]
        self._interval[1] = self._knots[self._knots.shape[0] - self._degree-1 ]
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
        self._update_interval()
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

        self._update_interval()
        self._periodic = False
        self._evaluate_cached.cache_clear()


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void ctangent(self, double t,double[:] result):
        cdef double[:,:] ders=np.zeros((3,3))
        self.cderivatives2(t,2, ders)
        calgorithms.evaluate_tangent(ders[1],ders[2],result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void ccurvature(self, double t,double[:] result):
           cdef double[:,:] ders=np.zeros((3,3))
           cdef double nrm=0
           self.cderivatives2(t,2, ders)
           calgorithms.evaluate_curvature(ders[1],ders[2],ders[0],result)

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







    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _evaluate(self, double t):
        cdef double[:] result =np.zeros((3,))
        self.cevaluate(t, result)

        return np.asarray(result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate(self, double t):
        return self._evaluate_cached(t)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_multi(self, double[:] t):
        cdef double[:,:] result=np.empty((t.shape[0],4))
        cdef size_t i;
        for i in range(t.shape[0]):


            self.cevaluate(t[i],result[i])
        return np.asarray(result)[:,:3]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def derivative(self, t):
        cdef double[:] result =np.zeros((3,))
        self.cderivative(t,result)
        return np.asarray(result)
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
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cnormal(self, double t, double[:] result):
        cdef double[:,:] vecs=np.zeros((3,3))
        self.cplane(t,vecs)
        result[:]=vecs[2,:]


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
        cdef double[:, :]  CK = np.zeros((du + 1, 3))


        self.cderivatives2(t,d,CK)
        return np.asarray(CK)
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
            cdef double[:, :]  CK = np.zeros((du + 1, 3))

            self.cderivatives1(t,d, CK)
            return np.asarray(CK)
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
            cdef double[:, :, :] PK = np.zeros((d + 1, self._degree + 1, self._control_points.shape[1]-1))
            curve_deriv_cpts(self._degree, self._knots,self._control_points, d, span - self._degree, span, PK)

            return np.asarray(PK)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cderivative(self, double t, double[:] result):
        cdef double[:,:] res=np.zeros((2,3))
        self.cderivatives2(t, 1, res)
        result[0]=res[1][0]
        result[1] = res[1][1]
        result[2] = res[1][2]
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void csecond_derivative(self, double t, double[:] result):
            cdef double[:,:] res=np.zeros((3,3))
            self.cderivatives2(t, 2, res)
            result[0]= res[2][0]
            result[1]= res[2][1]
            result[2]= res[2][2]


