
#cython: language_level=3
cimport cython
cimport mmcore.numeric.vectors
import numpy as np
cimport numpy as np

from libc.math cimport fabs, sqrt,fmin,fmax,pow

cdef extern from "math.h":
    long double sin(long double u)nogil
    long double cos(long double v)nogil
    long double atan2(long double a, double b)nogil
    long double atan(long double a)nogil

cdef double RTOL=1.e-15
cdef double ATOL=1.e-18
cdef bint cis_close(double a, double b, double atol, double rtol):
    cdef bint result=fabs(a - b) <= (atol + rtol * fabs(b))
    return result
cpdef bint is_close(double a, double b=0., double atol=ATOL, double rtol=RTOL):
    cdef bint result=fabs(a - b) <= (atol + rtol * fabs(b))
    return result


cdef extern from "limits.h":
    double DBL_MAX
    double DBL_MIN




ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef dot_array_x_array(double[:,:]vec_a, double[:,:]vec_b):
    cdef double[:]res = np.empty((vec_a.shape[0],))
    cdef double item = 0.0
    cdef int i,j
    for i in range(vec_a.shape[0]):

        item = 0.0
        for j in range(vec_a.shape[1]):
            item += vec_a[i, j] * vec_b[i, j]
        res[i] += item
    return res
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef matmul_array(double[:,:,:] vec_a, double[:,:,:] vec_b):

    cdef int i, j,k,w
    cdef int l=vec_a.shape[0]
    cdef int rows_A = vec_a.shape[1]
    cdef int cols_A = vec_a.shape[2]
    cdef int rows_B = vec_b.shape[1]
    cdef int cols_B = vec_b.shape[2]
    cdef double[:,:,:] result=np.empty((l,rows_A,cols_B))
    for w in range(l):
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[w,i,j] += vec_a[w,i,k] * vec_b[w,k,j]

    return np.array(result)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef dot_vec_x_array(double[:]vec_a, double[:,:]vec_b):
    cdef double[:]res = np.empty((vec_b.shape[0],))
    cdef double item = 0.0
    cdef int i, j
    for i in range(vec_b.shape[0]):
        item = 0.0
        for j in range(vec_a.shape[0]):
            item += vec_a[j] * vec_b[i, j]
        res[i] += item
    return res
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef dot_array_x_vec(double[:,:]vec_a, double[:]vec_b):
    cdef double[:]res = np.empty((vec_a.shape[0],))
    cdef double item = 0.0
    cdef int i, j
    for i in range(vec_a.shape[0]):
        item = 0.0
        for j in range(vec_b.shape[0]):
            item += vec_a[i, j] * vec_b[j]
        res[i] += item
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef dot(double[:,:]vec_a, double[:,:]vec_b):
    cdef double[:]res = np.empty((vec_a.shape[0],))
    cdef double item = 0.0
    cdef int i, j
    for i in range(vec_a.shape[0]):

        item = 0.0
        for j in range(vec_a.shape[1]):
            item += vec_a[i, j] * vec_b[i, j]
        res[i] = item
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double scalar_dot(double[:] vec_a, double[:]  vec_b) noexcept nogil:
    cdef double res = 0.0
    cdef int  j
    for j in range(vec_a.shape[0]):
        res += vec_a[j] * vec_b[j]
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double _dot_inl(double[:] vec_a,
                            double[:]  vec_b) noexcept nogil:
    cdef double res = 0.0
    cdef int  j
    for j in range(vec_a.shape[0]):
        res += vec_a[j] * vec_b[j]
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef scalar_norm(double[:]vec):
    cdef DTYPE_t res = 0.0
    cdef DTYPE_t res2 = 0.0
    cdef int j
    for j in range(vec.shape[0]):
        res += (vec[j] ** 2)

    return sqrt(res)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef scalar_normalize(double[:]vec):
    cdef double res = 0.
    cdef Py_ssize_t j

    for j in range(vec.shape[0]):
        res += (vec[j] ** 2)

    res=sqrt(res)

    if res>1e-15:
        for j in range(vec.shape[0]):
            vec[j]=vec[j]/res
        return 1
    return 0
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef norm(double[:,:]vec):
    cdef double[:]res = np.empty((vec.shape[0],))
    cdef long double item = 0.0
    cdef long double component_sq = 0.0
    cdef int i,j
    for i in range(vec.shape[0]):
        item = 0.0
        for j in range(vec.shape[1]):
            component_sq = vec[i, j] ** 2
            item += component_sq
        res[i] = sqrt(item)

    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef unit(double[:,:]vec):
    cdef double[:,:]res = np.empty((vec.shape[0], vec.shape[1]))
    cdef long double item
    cdef long double component_sq = 0.0
    cdef long double nrm = 0.0
    cdef int i, j,k
    for i in range(vec.shape[0]):
        item = 0.0
        for j in range(vec.shape[1]):
            component_sq = vec[i, j] ** 2
            item += component_sq
        nrm = sqrt(item)
        for k in range(vec.shape[1]):
            res[i, k] = vec[i, k] / nrm
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cross(double[:,:]vec_a,
          double[:,:]vec_b):
    cdef double[:,:]res = np.empty((vec_a.shape[0], 3))
    cdef int i
    for i in range(vec_a.shape[0]):
        res[i, 0] = (vec_a[i, 1] * vec_b[i, 2]) - (vec_a[i, 2] * vec_b[i, 1])
        res[i, 1] = (vec_a[i, 2] * vec_b[i, 0]) - (vec_a[i, 0] * vec_b[i, 2])
        res[i, 2] = (vec_a[i, 0] * vec_b[i, 1]) - (vec_a[i, 1] * vec_b[i, 0])
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef support_vector(double[:,:]vertices, double[:]direction):
    cdef double[:]support = np.zeros((direction.shape[0],))
    cdef  highest = -np.inf
    cdef long double dot_value = 0.0
    cdef int i
    for i in range(vertices.shape[0]):
        dot_value = scalar_dot(vertices[i], direction)
        if dot_value > highest:
            highest = dot_value
            support = vertices[i]
    return support

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef multi_support_vector(double[:,:]vertices, double[:,:]directions):
    cdef double[:,:]support = np.zeros((directions.shape[0], directions.shape[1]))
    cdef  highest = -np.inf
    cdef long double dot_value = 0.0
    cdef int i,j
    for j in range(directions.shape[0]):
        highest = -np.inf
        for i in range(vertices.shape[0]):
            dot_value = scalar_dot(vertices[i], directions[j])
            if dot_value > highest:
                highest = dot_value
                support[j, :] = vertices[i]

    return support

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef gram_schmidt(double[:,:]vec_a,
                 double[:,:]vec_b):
    cdef double[:,:]res = np.empty((vec_a.shape[0], vec_a.shape[1]))
    cdef long double item_dot
    cdef int i,j,k
    for i in range(vec_a.shape[0]):
        item_dot = 0.0
        for j in range(vec_a.shape[1]):
            item_dot += vec_b[i, j] * vec_a[i, j]
        for k in range(vec_a.shape[1]):
            res[i, k] = vec_b[i, j] - (vec_a[i, j] * item_dot)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef spherical_to_cartesian(double[:,:]rtp):
    cdef double[:,:]pts = np.empty((rtp.shape[0], 3))
    cdef int i
    for i in range(rtp.shape[0]):
        pts[i, 0] = rtp[i, 0] * sin(rtp[i, 1]) * cos(rtp[i, 2])
        pts[i, 1] = rtp[i, 0] * sin(rtp[i, 1]) * sin(rtp[i, 2])
        pts[i, 2] = rtp[i, 0] * cos(rtp[i, 1])
    return pts

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cartesian_to_spherical(double[:,:]xyz):
    cdef double[:,:]pts = np.empty((xyz.shape[0], 3))
    cdef long double XsqPlusYsq
    cdef int i
    for i in range(xyz.shape[0]):
        XsqPlusYsq = xyz[i, 0] ** 2 + xyz[i, 1] ** 2
        pts[i, 0] = sqrt(XsqPlusYsq + xyz[i, 2] ** 2)
        pts[i, 1] = atan2(sqrt(XsqPlusYsq), xyz[i, 2])
        pts[i, 2] = atan2(xyz[i, 1], xyz[i, 0])
    return pts

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cylindrical_to_xyz(double[:,:]rpz):
    cdef double[:,:]pts = np.empty((rpz.shape[0], 3))
    cdef long double XsqPlusYsq
    cdef int i
    for i in range(rpz.shape[0]):
        pts[i, 0] = rpz[i, 0] * cos(rpz[i, 1])
        pts[i, 1] = rpz[i, 0] * sin(rpz[i, 1])

        pts[i, 2] = rpz[i, 2]
    return pts

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef courtesan_to_cylindrical(double[:,:]xyz):
    cdef double[:,:]pts = np.empty((xyz.shape[0], 3))
    cdef long double XsqPlusYsq
    cdef int i
    for i in range(xyz.shape[0]):
        XsqPlusYsq = xyz[i, 0] ** 2 + xyz[i, 1] ** 2
        pts[i, 0] = sqrt(XsqPlusYsq + xyz[i, 2] ** 2)
        pts[i, 1] = atan2(sqrt(XsqPlusYsq), xyz[i, 2])
        pts[i, 2] = xyz[i, 2]
    return pts
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef long double cdet(double[:,:]arr) :
    cdef long double res = 0.0
    cdef int i
    for i in range(arr.shape[0] - 1):
        res += ((arr[i + 1][0] - arr[i][0]) * (arr[i + 1][1] + arr[i][1]))
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef det(double[:,:]arr):
    return cdet(arr)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef points_order(double[:,:]points):
    determinant = cdet(points)
    cdef long res = -1
    if determinant > 0:
        res = 0
    elif determinant < 0:
        res = 1
    return res

cpdef multi_points_order(list points_list):
    cdef np.ndarray[long, ndim = 1] res = np.empty((len(points_list),), int)
    cdef int i
    for i in range(len(points_list)):
        res[i] = points_order(points_list[i])
    return res
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef scalar_cross(double[:] vec_a,
          double[:] vec_b):
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty((3,))


    res[ 0] = (vec_a[ 1] * vec_b[ 2]) - (vec_a[ 2] * vec_b[ 1])
    res[ 1] = (vec_a[ 2] * vec_b[ 0]) - (vec_a[ 0] * vec_b[ 2])
    res[ 2] = (vec_a[ 0] * vec_b[ 1]) - (vec_a[ 1] * vec_b[ 0])
    return res
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef  scalar_unit(double[:]vec):
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty((vec.shape[0],))
    cdef long double item=0.
    cdef long double component_sq = 0.0
    cdef long double nrm = 0.0
    cdef int j,k
    for j in range(vec.shape[0]):
        component_sq = vec[j] ** 2
        item += component_sq
    nrm = sqrt(item)
    for k in range(vec.shape[0]):
        res[k] = vec[ k] / nrm
    return res



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef int cinvert_jacobian(double[:,:] J, double[:,:] J_inv) noexcept nogil:
    cdef double a=J[0][0]
    cdef double b=J[0][1]
    cdef double c=J[1][0]
    cdef double d=J[1][1]
    cdef double jac_det = a * d - b * c
    cdef int return_code = 0

    if jac_det==0:
        return_code=-1

    else:
        J_inv[0][0]=d / jac_det
        J_inv[0][1] = -b / jac_det
        J_inv[1][0] =-c  / jac_det
        J_inv[1][1] = a / jac_det

    return return_code

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void cinvert_jacobian_vec(double[:,:,:] J, double[:,:,:] J_inv, int[:] status) noexcept nogil:
    cdef size_t i
    for i in range(J_inv.shape[0]):
        status[i]=cinvert_jacobian(J[i], J_inv[i])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef invert_jacobian(J):
    cdef double[:,:] J_inv_m
    cdef double[:,:, :] J_inv
    cdef int[:] status
    cdef int status_m
    if J.ndim==2:
        J_inv_m=np.empty((2,2))
        status_m=cinvert_jacobian(J,J_inv_m)
        return J_inv_m,status_m

    else:
        J_inv = np.empty((J.shape[0], 2, 2))
        status=np.empty((J.shape[0], 2, 2), dtype=np.intc)
        with nogil:
            cinvert_jacobian_vec(J, J_inv,  status)
        return J_inv, status
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef vector_projection(double[:] a, double[:] b):
    cdef double[:] res=np.empty((3,))
    bn=(b[0]**2 + b[1]**2 + b[2]**2)

    res[0]=a[0]*b[0]*b[0]/bn + a[1]*b[0]*b[1]/bn + a[2]*b[0]*b[2]/bn
    res[1]=a[0]*b[0]*b[1]/bn + a[1]*b[1]*b[1]/bn + a[2]*b[1]*b[2]/bn
    res[2]=a[0]*b[0]*b[2]/bn + a[1]*b[1]*b[2]/bn + a[2]*b[2]*b[2]/bn
    return res
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef closest_point_on_ray( double[:] start, double[:] direction,  double[:] point):
    cdef double [:] res=np.empty((3,))
    cdef double [:] a=np.empty((3,))
    cdef double directionn = (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    a[0]=point[0]-start[0]
    a[1] = point[1] - start[1]
    a[2] = point[2] - start[2]

    res[0]=a[0]*direction[0]*direction[0]/directionn + a[1]*direction[0]*direction[1]/directionn + a[2]*direction[0]*direction[2]/directionn
    res[1]=a[0]*direction[0]*direction[1]/directionn + a[1]*direction[1]*direction[1]/directionn + a[2]*direction[1]*direction[2]/directionn
    res[2]=a[0]*direction[0]*direction[2]/directionn + a[1]*direction[1]*direction[2]/directionn + a[2]*direction[2]*direction[2]/directionn
    return res
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef closest_point_on_line(double[:] start,double[:] end, double[:] point):
    cdef double [:] direction=np.empty((3,))
    cdef double [:] p=np.empty((3,))
    direction[0] = end[0] - start[0]
    direction[1] = end[1] - start[1]
    direction[2] = end[2] - start[2]
    p[0] = point[0] - start[0]
    p[1] = point[1] - start[1]
    p[2] = point[2] - start[2]
    return vector_projection(p, direction)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef bint solve2x2(double[:,:] matrix, double[:] y,  double[:] result) noexcept nogil:

    cdef bint res
    cdef double det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # matrix[1][0]hematrix[1][0]k if the determinmatrix[0][0]nt is zero
    if det == 0:

        return 0
    else:
        # matrix[1][0]matrix[0][0]lmatrix[1][0]ulmatrix[0][0]te x matrix[0][0]nd y using the dirematrix[1][0]t method
        result[0] = (y[0] * matrix[1][1] - matrix[0][1] * y[1]) / det
        result[1] = (matrix[0][0] * y[1] - y[0] * matrix[1][0]) / det

        return 1
cdef inline double scalar_norm3d(double[3] vec_a):
    cdef double res = sqrt(vec_a[0] * vec_a[0] + vec_a[1] * vec_a[1] + vec_a[2] * vec_a[2])

    return res
cdef inline double scalar_norm2d(double[3] vec_a):
    cdef double res = sqrt(vec_a[0] * vec_a[0] + vec_a[1] * vec_a[1] )

    return res
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void scalar_gram_schmidt(double[:] vec_a,
                 double[:] vec_b, double[:] result) :
    cdef double[3] temp=np.zeros(3)
    scalar_mul3d(vec_a, scalar_dot(vec_b, vec_a), result)
    sub3d(vec_b, result, result)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void scalar_gram_schmidt_emplace(double[:] vec_a,
                 double[:] vec_b) noexcept nogil:
    cdef double delta=vec_b[0]* vec_a[0]+vec_b[1]* vec_a[1]+vec_b[2]* vec_a[2]
    vec_b[0] -= vec_a[0] * delta
    vec_b[1] -= vec_a[1] * delta
    vec_b[2] -= vec_a[2] * delta

"""
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:] scalar_gram_schmidt(double[:] vec_a,
                 double[:] vec_b) :

    return scalar_unit(vec_b - vec_a * scalar_dot(vec_b, vec_a))
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:] make_perpendicular(double[:] vec_a,
                 double[:] vec_b):
    cdef double [:] res=np.empty(vec_a.shape)
    if vec_a.shape[0] == 2:
        res[0]= -v1[1]
        res[1] = v1[0]
        return res
    else:
        return scalar_gram_schmidt(vec_a,vec_b)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:] linear_combination_3d(double a, double[:] v1, double b, double[:] v2) :
    cdef double[:] res = np.empty( v1.shape)
    res[0]=v1[0]*a + b * v2[0]
    res[1]=v1[1] * a + b * v2[1]
    res[2]=v1[2] * a + b * v2[2]
    return res


cpdef rotate(self, deg):
    sin, cos = function_DegToSinCos(deg)
    return Vector2D(self.x * cos - self.y * sin, self.x * sin + self.y * cos)

    cdef rotateAroundPoint(self, v2d, deg):
        sin, cos = function_DegToSinCos(deg)
        v2d0 = self.sub(v2d)
        return Vector2D(v2d0.x * cos - v2d0.y * sin, v2d0.x * sin + v2d0.y * cos).add(v2d)



cpdef void function_linearCombine2D(double[2] v3d0, double r0, double[2] v3d1, double r1,
                                    double[2] result):
    result[0] = r0 * v3d0[0] + r1 * v3d1[0]
    result[1] = r0 * v3d0[1] + r1 * v3d1[1]


cpdef void function_linearCombine3D(double[3] v3d0, double r0, double[3] v3d1, double r1, double[3] v3d2, double r2=0, double[3] result):

    result[0]=r0 * v3d0[0] + r1 * v3d1[0] + r2 * v3d2[0]
    result[1]=r0 * v3d0[1] + r1 * v3d1[1] + r2 * v3d2[1]
    result[2]= r0 * v3d0[2]+ r1 * v3d1[2] + r2 * v3d2[2]



cpdef function_DegToSinCos(deg):
    deg = deg * pi / 180
    return (sin(deg), math.cos(deg))


cpdef double function_SolveEquationDeg1(double a, double b):
    return -b / a

cpdef function_SolveEquationDeg2(double a, double b, double c, result[:]):  # sqrt exception will be raised
    cdef static double
    if a!=0:  # wenn a!=0
        h0 = sqrt((b * b - 4 * a * c) / 4 * a * a)
        h1 = b / -2 * a

        return (h1 + h0, h1 - h0)
    return function_SolveEquationDeg1(b, c)


cpdef function_PointOverPlane(v3d, n_v3d, d):  #{...stellt fest, ob der Punkt p "vor" der Ebene  nv*x-d=0 liegt.}
    return v3d.scalarProduct(n_v3d) - d >= 0


cpdef function_GetPlaneEquation(v3d0, v3d1, v3d2):  #Exception wenn keine ebene
    n_v3d = ((v3d1.sub(v3d0)).vectorProduct(v3d2.sub(v3d0))).normalize()
    if not n_v3d.equals(Vector3D()): return (n_v3d, n_v3d.scalarProduct(v3d0))
    raise
"""



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double dot3d(double [:]  vec_a, double [:]  vec_b) noexcept nogil:
    cdef double res = vec_a[0] * vec_b[0]+ vec_a[1] * vec_b[1]+ vec_a[2] * vec_b[2]

    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double norm3d(double [:] vec)noexcept nogil:
    cdef double res = sqrt(vec[0] ** 2+vec[1] ** 2+vec[2] ** 2)
    return res
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void min3d(double [:]  vec_a, double [:]  vec_b, double[:] res)noexcept nogil:

    res[0] = fmin(vec_a[0] ,vec_b[0])
    res[1] = fmin(vec_a[1] , vec_b[1])
    res[2] = fmin(vec_a[2] , vec_b[2])
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void max3d(double [:]  vec_a, double [:]  vec_b, double[:] res)noexcept nogil:

    res[0] = fmax(vec_a[0] ,vec_b[0])
    res[1] = fmax(vec_a[1] , vec_b[1])
    res[2] = fmax(vec_a[2] , vec_b[2])
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void bavg3d(double [:]  vec_a, double [:]  vec_b, double[:] res)noexcept nogil:

    res[0] = (vec_a[0] +vec_b[0])/2
    res[1] = (vec_a[1] +vec_b[1])/2
    res[2] = (vec_a[2] +vec_b[2])/2
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sqrt3d(double [3]  vec_a,  double[3] res)noexcept nogil:

    res[0] = sqrt(vec_a[0])
    res[1] = sqrt(vec_a[1])
    res[2] = sqrt(vec_a[2])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mul3d(double[3]  vec_a, double[3]  vec_b, double[:] res)noexcept nogil:

    res[0] = vec_a[0] * vec_b[0]
    res[1] = vec_a[1] * vec_b[1]
    res[2] = vec_a[2] * vec_b[2]
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void add3d(double [:]  vec_a, double [:]  vec_b, double[:] res)noexcept nogil:

    res[0] = vec_a[0] + vec_b[0]
    res[1] = vec_a[1] + vec_b[1]
    res[2] = vec_a[2] + vec_b[2]
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void sub3d(double [:]  vec_a, double [:]  vec_b, double[:] res)noexcept nogil:

    res[0] = vec_a[0] - vec_b[0]
    res[1] = vec_a[1] - vec_b[1]
    res[2] = vec_a[2] - vec_b[2]
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void scalar_sub3d(double [:]  vec_a, double  b, double[:] res)noexcept nogil:

    res[0] = vec_a[0] - b
    res[1] = vec_a[1] - b
    res[2] = vec_a[2] - b
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void scalar_add3d(double [:]  vec_a, double  b, double[:] res) noexcept nogil:

    res[0] = vec_a[0] + b
    res[1] = vec_a[1] + b
    res[2] = vec_a[2] + b
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void scalar_mul3d(double [:]  vec_a, double  b, double[:] res)noexcept nogil:

    res[0] = vec_a[0] * b
    res[1] = vec_a[1] * b
    res[2] = vec_a[2] * b
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void scalar_div3d(double [:]  vec_a, double  b, double[:] res) noexcept nogil:

    res[0] = vec_a[0] / b
    res[1] = vec_a[1] / b
    res[2] = vec_a[2] / b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef decompose_vector(double [:] v,double [:] a, double [:] b):

    cdef double[:,:] A=np.empty((2,2))
    cdef double[:] y=np.empty((2,))
    cdef double[:] xy=np.empty((2,))

    A[0,0]=scalar_dot(a,a)
    A[0,1]=scalar_dot(a, b)
    A[1, 0] = A[0,1]
    A[1, 1] =   scalar_dot(b, b)
    y[0] = scalar_dot(a, v)
    y[1] = scalar_dot(b,v)
    solve2x2(A,y,xy)
    return xy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef decompose_vectors(double[:,:] v, double[:] a, double[:] b, double[:,:] result=None):
    cdef double[:,:] A=np.empty((2, 2))
    cdef double[:] y=np.empty((2, ))
    cdef int i;
    if result is None:
        result=np.empty((v.shape[0], 2))

    for i in range(v.shape[0]):

            A[0, 0]=_dot_inl(a, a)
            A[0, 1]=_dot_inl(a, b)
            A[1, 0] = A[0, 1]
            A[1, 1] =  _dot_inl(b, b)
            y[0] = _dot_inl(a, v[i])
            y[1] = _dot_inl(b, v[i])
            solve2x2(A, y, result[i])
    return result
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double[:] normal_from_4pt(double[:] a, double[:] b, double[:] c, double[:] d, double[:] result=None) noexcept nogil:
    cdef double[3] temp1
    cdef double[3] temp2
    if result is None:
        with gil:
            result=np.empty((3,))
    temp1[0] = c[0] - a[0]
    temp1[1] = c[1] - a[1]
    temp1[2] = c[2] - a[2]
    temp2[0] = d[0] - b[0]
    temp2[1] = d[1] - b[1]
    temp2[2] = d[2] - b[2]
    cross_d1_3d(temp1, temp2, result)
    return result




