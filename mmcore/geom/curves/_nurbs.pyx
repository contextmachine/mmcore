# cython: language_level=3
cimport cython
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc,free
from mmcore.geom.parametric cimport ParametricCurve
from mmcore.numeric cimport vectors,calgorithms
from libc.math cimport fabs, sqrt,fmin,fmax,pow
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int find_span(int n, int p, double u, double[:] U) noexcept nogil:
    """
    Determine the knot span index.
    """

    if u == U[n + 1]:
        return n  # Special case
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
cdef void curve_point(int n, int p, double[:] U, double[:, :] P, double u, double[:] result) noexcept nogil:
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
    cdef int span = find_span(n, p, u, U)
    cdef double* N = <double*>malloc(sizeof(double)*pp)

    basis_funs(span, u, p, U, N)

    for i in range(pp):
        for j in range(result.shape[0]):
            result[j] += N[i] * P[span - p + i, j]


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
cpdef void curve_derivs_alg1(int n, int p, double[:] U, double[:, :] P, double u, int d, double[:, :] CK):
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
    cdef int span = find_span(n, p, u, U)
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

    Returns:
    ndarray: The computed control points of curve derivatives.
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
cpdef void curve_derivs_alg2(int n, int p, double[:] U, double[:, :] P, double u, int d, double[:, :] CK, double[:, :,:] PK):
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

    cdef int span = find_span(n, degree, u, knotvector)
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


cdef class NURBSpline(ParametricCurve):
    cdef double[:,:] _control_points
    cdef public int degree
    cdef public double[:] knots
    cdef public int n
    cdef double[:, :, :] _PK


    def __init__(self, double[:,:] control_points, int degree=3, double[:] knots=None):
        super().__init__()
        self._control_points = np.ones((control_points.shape[0], 4))

        self._control_points[:, :-1] = control_points

        self.degree = degree
        if knots is None:
            self.generate_knots()
        else:
            self.knots=knots

        self.n = len(self._control_points) - 1
        self._interval[0] = np.min(self.knots)
        self._interval[1] = np.max(self.knots)
        self._PK = np.zeros((2 + 1, self.degree + 1, self._control_points.shape[1]-1 ))

    cdef void _pknull(self):

        self._PK= np.zeros((2 + 1, self.degree + 1, self._control_points.shape[1]-1 ))

    @property
    def control_points(self):
        return self._control_points[..., :-1]
    @control_points.setter
    def control_points(self, double[:,:] control_points):
        self._control_points = np.ones((control_points.shape[0],4))

        self._control_points[:, :-1] = control_points

    cdef void generate_knots(self):
        """
        This function generates default knots based on the number of control points
        :return: A numpy array of knots
        """
        cdef int n = len(self.control_points)
        self.knots = np.concatenate((
            np.zeros(self.degree + 1),
            np.arange(1, n - self.degree),
            np.full(self.degree + 1, n - self.degree)
        ))
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

        curve_point(self.n, self.degree, self.knots, self._control_points, t, result)
        projective_to_cartesian(result,result)


    def evaluate(self, t):
        if not (self._interval[0]<=t<=self._interval[1]):
            raise ValueError("t out of bounds")
        cdef double[:] result =np.zeros((4,))
        self.cevaluate(t,result)
        return np.asarray(result[:3])
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

        cdef int i
        #cdef double[:, :]  CK = np.zeros((du + 1, 4))
        #cdef double[:, :, :] PK = np.zeros((d + 1, self.degree + 1,  self._control_points.shape[1]-1))
        curve_derivs_alg1(self.n, self.degree, self.knots, self._control_points[:,:-1], t, d, CK)

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

           cdef int i
           #cdef double[:, :]  CK = np.zeros((du + 1, 4))
           #cdef double[:, :, :] PK = np.zeros((d + 1, self.degree + 1,  self._control_points.shape[1]-1))
           self._pknull()

           curve_derivs_alg2(self.n, self.degree, self.knots, self._control_points[:,:-1], t, d, CK, self._PK)




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
        result[:]=vecs[3,:]












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

        cdef int du = min(d, self.degree)
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

            cdef int du = min(d, self.degree)
            cdef double[:, :]  CK = np.zeros((du + 1, 3))

            self.cderivatives1(t,d,CK)
            return np.asarray(CK)
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
            cdef double[:,:] res=np.zeros((2,3))
            self.cderivatives2(t, 2, res)
            result[0]=res[2][0]
            result[1]= res[2][1]
            result[2]= res[2][2]
    cpdef double[:,:] get_control_points_4d(self):
        return self._control_points
    cpdef void set_control_points_4d(self, double[:,:] cpts):
            self._control_points=cpts
            self.generate_knots()
            self.n = self._control_points.shape[0] - 1
            self._interval[0] = np.min(self.knots)
            self._interval[1] = np.max(self.knots)