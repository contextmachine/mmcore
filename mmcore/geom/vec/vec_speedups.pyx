from libc.math cimport pi,fabs
cdef extern from "math.h":
    long double sqrt(long double xx)
    long double sin(long double u)
    long double cos(long double v)
    long double atan2(long double a, double b)
    long double atan(long double a)


    double fmax(double a)


cdef extern from "limits.h":
    double DBL_MAX
    double DBL_MIN


import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def dot_array_x_array(np.ndarray[DTYPE_t, ndim=2] vec_a, np.ndarray[DTYPE_t, ndim=2] vec_b):
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty((vec_a.shape[0],))
    cdef DTYPE_t item = 0.0
    for i in range(vec_a.shape[0]):

        item = 0.0
        for j in range(vec_a.shape[1]):
            item += vec_a[i, j] * vec_b[i, j]
        res[i] += item
    return res
@cython.boundscheck(False)
@cython.wraparound(False)
def dot_vec_x_array(np.ndarray[DTYPE_t, ndim=1] vec_a, np.ndarray[DTYPE_t, ndim=2] vec_b):
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty((vec_b.shape[0],))
    cdef DTYPE_t item = 0.0
    for i in range(vec_b.shape[0]):
        item = 0.0
        for j in range(vec_a.shape[0]):
            item += vec_a[j] * vec_b[i, j]
        res[i] += item
    return res
@cython.boundscheck(False)
@cython.wraparound(False)
def dot_array_x_vec(np.ndarray[DTYPE_t, ndim=2] vec_a, np.ndarray[DTYPE_t, ndim=1] vec_b):
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty((vec_a.shape[0],))
    cdef DTYPE_t item = 0.0
    for i in range(vec_a.shape[0]):
        item = 0.0
        for j in range(vec_b.shape[0]):
            item += vec_a[i, j] * vec_b[j]
        res[i] += item
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def dot(np.ndarray[DTYPE_t, ndim=2] vec_a, np.ndarray[DTYPE_t, ndim=2] vec_b):
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty((vec_a.shape[0],))
    cdef DTYPE_t item = 0.0
    for i in range(vec_a.shape[0]):

        item = 0.0
        for j in range(vec_a.shape[1]):
            item += vec_a[i, j] * vec_b[i, j]
        res[i] = item
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_dot(np.ndarray[DTYPE_t, ndim=1] vec_a,
                            np.ndarray[DTYPE_t, ndim=1] vec_b):
    cdef long double res = 0.0
    for j in range(vec_a.shape[0]):
        res += vec_a[j] * vec_b[j]
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_norm(np.ndarray[DTYPE_t, ndim=1] vec):
    cdef DTYPE_t res = 0.0
    cdef DTYPE_t res2 = 0.0
    for j in range(vec.shape[0]):
        res += (vec[j] ** 2)
    if res<=1e-15:
        return res2
    return sqrt(res)
@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_normalize(np.ndarray[DTYPE_t, ndim=1] vec):
    cdef DTYPE_t res = 0.
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
def norm(np.ndarray[DTYPE_t, ndim=2] vec):
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty((vec.shape[0],))
    cdef long double item = 0.0
    cdef long double component_sq = 0.0
    for i in range(vec.shape[0]):
        item = 0.0
        for j in range(vec.shape[1]):
            component_sq = vec[i, j] ** 2
            item += component_sq
        res[i] = sqrt(item)

    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def unit(np.ndarray[DTYPE_t, ndim=2] vec):
    cdef np.ndarray[DTYPE_t, ndim=2] res = np.empty((vec.shape[0], vec.shape[1]))
    cdef long double item
    cdef long double component_sq = 0.0
    cdef long double nrm = 0.0
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
def cross(np.ndarray[DTYPE_t, ndim=2] vec_a,
          np.ndarray[DTYPE_t, ndim=2] vec_b):
    cdef np.ndarray[DTYPE_t, ndim=2] res = np.empty((vec_a.shape[0], 3))

    for i in range(vec_a.shape[0]):
        res[i, 0] = (vec_a[i, 1] * vec_b[i, 2]) - (vec_a[i, 2] * vec_b[i, 1])
        res[i, 1] = (vec_a[i, 2] * vec_b[i, 0]) - (vec_a[i, 0] * vec_b[i, 2])
        res[i, 2] = (vec_a[i, 0] * vec_b[i, 1]) - (vec_a[i, 1] * vec_b[i, 0])
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def support_vector(np.ndarray[DTYPE_t, ndim=2] vertices, np.ndarray[DTYPE_t, ndim=1] direction):
    cdef np.ndarray[DTYPE_t, ndim=1] support = np.zeros((direction.shape[0],))
    cdef  highest = -np.inf
    cdef long double dot_value = 0.0
    for i in range(vertices.shape[0]):
        dot_value = scalar_dot(vertices[i], direction)
        if dot_value > highest:
            highest = dot_value
            support = vertices[i]
    return support

@cython.boundscheck(False)
@cython.wraparound(False)
def multi_support_vector(np.ndarray[DTYPE_t, ndim=2] vertices, np.ndarray[DTYPE_t, ndim=2] directions):
    cdef np.ndarray[DTYPE_t, ndim=2] support = np.zeros((directions.shape[0], directions.shape[1]))
    cdef  highest = -np.inf
    cdef long double dot_value = 0.0
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
def gram_schmidt(np.ndarray[DTYPE_t, ndim=2] vec_a,
                 np.ndarray[DTYPE_t, ndim=2] vec_b):
    cdef np.ndarray[DTYPE_t, ndim=2] res = np.empty((vec_a.shape[0], vec_a.shape[1]))
    cdef long double item_dot
    for i in range(vec_a.shape[0]):
        item_dot = 0.0
        for j in range(vec_a.shape[1]):
            item_dot += vec_b[i, j] * vec_a[i, j]
        for k in range(vec_a.shape[1]):
            res[i, k] = vec_b[i, j] - (vec_a[i, j] * item_dot)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def spherical_to_cartesian(np.ndarray[DTYPE_t, ndim=2] rtp):
    cdef np.ndarray[DTYPE_t, ndim=2] pts = np.empty((rtp.shape[0], 3))
    for i in range(rtp.shape[0]):
        pts[i, 0] = rtp[i, 0] * sin(rtp[i, 1]) * cos(rtp[i, 2])
        pts[i, 1] = rtp[i, 0] * sin(rtp[i, 1]) * sin(rtp[i, 2])
        pts[i, 2] = rtp[i, 0] * cos(rtp[i, 1])
    return pts

@cython.boundscheck(False)
@cython.wraparound(False)
def cartesian_to_spherical(np.ndarray[DTYPE_t, ndim=2] xyz):
    cdef np.ndarray[DTYPE_t, ndim=2] pts = np.empty((xyz.shape[0], 3))
    cdef long double XsqPlusYsq
    for i in range(xyz.shape[0]):
        XsqPlusYsq = xyz[i, 0] ** 2 + xyz[i, 1] ** 2
        pts[i, 0] = sqrt(XsqPlusYsq + xyz[i, 2] ** 2)
        pts[i, 1] = atan2(sqrt(XsqPlusYsq), xyz[i, 2])
        pts[i, 2] = atan2(xyz[i, 1], xyz[i, 0])
    return pts

@cython.boundscheck(False)
@cython.wraparound(False)
def cylindrical_to_xyz(np.ndarray[DTYPE_t, ndim=2] rpz):
    cdef np.ndarray[DTYPE_t, ndim=2] pts = np.empty((rpz.shape[0], 3))
    cdef long double XsqPlusYsq
    for i in range(rpz.shape[0]):
        pts[i, 0] = rpz[i, 0] * cos(rpz[i, 1])
        pts[i, 1] = rpz[i, 0] * sin(rpz[i, 1])

        pts[i, 2] = rpz[i, 2]
    return pts

@cython.boundscheck(False)
@cython.wraparound(False)
def courtesan_to_cylindrical(np.ndarray[DTYPE_t, ndim=2] xyz):
    cdef np.ndarray[DTYPE_t, ndim=2] pts = np.empty((xyz.shape[0], 3))
    cdef long double XsqPlusYsq
    for i in range(xyz.shape[0]):
        XsqPlusYsq = xyz[i, 0] ** 2 + xyz[i, 1] ** 2
        pts[i, 0] = sqrt(XsqPlusYsq + xyz[i, 2] ** 2)
        pts[i, 1] = atan2(sqrt(XsqPlusYsq), xyz[i, 2])
        pts[i, 2] = xyz[i, 2]
    return pts
@cython.boundscheck(False)
@cython.wraparound(False)
cdef long double cdet(np.ndarray[DTYPE_t, ndim=2] arr) :
    cdef long double res = 0.0
    for i in range(arr.shape[0] - 1):
        res += ((arr[i + 1][0] - arr[i][0]) * (arr[i + 1][1] + arr[i][1]))
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def det(np.ndarray[DTYPE_t, ndim=2] arr):
    return cdet(arr)

@cython.boundscheck(False)
@cython.wraparound(False)
def points_order(np.ndarray[DTYPE_t, ndim=2] points):
    determinant = cdet(points)
    cdef long res = -1
    if determinant > 0:
        res = 0
    elif determinant < 0:
        res = 1
    return res

def multi_points_order(list points_list):
    cdef np.ndarray[long, ndim = 1] res = np.empty((len(points_list),), int)
    for i in range(len(points_list)):
        res[i] = points_order(points_list[i])
    return res
@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_cross(np.ndarray[DTYPE_t, ndim=1] vec_a,
          np.ndarray[DTYPE_t, ndim=1] vec_b):
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty((3,))


    res[ 0] = (vec_a[ 1] * vec_b[ 2]) - (vec_a[ 2] * vec_b[ 1])
    res[ 1] = (vec_a[ 2] * vec_b[ 0]) - (vec_a[ 0] * vec_b[ 2])
    res[ 2] = (vec_a[ 0] * vec_b[ 1]) - (vec_a[ 1] * vec_b[ 0])
    return res
@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_unit(np.ndarray[DTYPE_t, ndim=1] vec):
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty((vec.shape[0],))
    cdef long double item=0.
    cdef long double component_sq = 0.0
    cdef long double nrm = 0.0
    for j in range(vec.shape[0]):
        component_sq = vec[j] ** 2
        item += component_sq
    nrm = sqrt(item)
    for k in range(vec.shape[0]):
        res[k] = vec[ k] / nrm
    return res



@cython.boundscheck(False)
@cython.wraparound(False)
cdef int cinvert_jacobian(double[:,:] J, double[:,:] J_inv)noexcept nogil:
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
cdef void cinvert_jacobian_vec(double[:,:,:] J, double[:,:,:] J_inv, int[:] status) noexcept nogil:
    cdef size_t i
    for i in range(J_inv.shape[0]):
        status[i]=cinvert_jacobian(J[i], J_inv[i])

@cython.boundscheck(False)
@cython.wraparound(False)
def invert_jacobian(J):
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
def vector_projection(double[:] a, double[:] b):
    cdef double[:] res=np.empty((3,))
    bn=(b[0]**2 + b[1]**2 + b[2]**2)

    res[0]=a[0]*b[0]*b[0]/bn + a[1]*b[0]*b[1]/bn + a[2]*b[0]*b[2]/bn
    res[1]=a[0]*b[0]*b[1]/bn + a[1]*b[1]*b[1]/bn + a[2]*b[1]*b[2]/bn
    res[2]=a[0]*b[0]*b[2]/bn + a[1]*b[1]*b[2]/bn + a[2]*b[2]*b[2]/bn
    return res
@cython.boundscheck(False)
@cython.wraparound(False)
def closest_point_on_ray( double[:] start, double[:] direction,  double[:] point):
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
def closest_point_on_line(double[:] start,double[:] end, double[:] point):
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
cpdef bint solve2x2(double[:,:] matrix, double[:] y,  double[:] result) noexcept nogil:

    cdef bint res
    cdef double det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # matrix[1][0]hematrix[1][0]k if the determinmatrix[0][0]nt is zero
    if det == 0:
        res=0
        return res
    else:
        # matrix[1][0]matrix[0][0]lmatrix[1][0]ulmatrix[0][0]te x matrix[0][0]nd y using the dirematrix[1][0]t method
        result[0] = (y[0] * matrix[1][1] - matrix[0][1] * y[1]) / det
        result[1] = (matrix[0][0] * y[1] - y[0] * matrix[1][0]) / det
        res=1
        return res
"""
def rotate(self, deg):
    sin, cos = function_DegToSinCos(deg)
    return Vector2D(self.x * cos - self.y * sin, self.x * sin + self.y * cos)

    def rotateAroundPoint(self, v2d, deg):
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



def function_DegToSinCos(deg):
    deg = deg * pi / 180
    return (sin(deg), math.cos(deg))


cpdef double function_SolveEquationDeg1(double a, double b):
    return -b / a

def function_SolveEquationDeg2(double a, double b, double c, result[:]):  # sqrt exception will be raised
    cdef static double
    if a!=0:  # wenn a!=0
        h0 = sqrt((b * b - 4 * a * c) / 4 * a * a)
        h1 = b / -2 * a

        return (h1 + h0, h1 - h0)
    return function_SolveEquationDeg1(b, c)


def function_PointOverPlane(v3d, n_v3d, d):  #{...stellt fest, ob der Punkt p "vor" der Ebene  nv*x-d=0 liegt.}
    return v3d.scalarProduct(n_v3d) - d >= 0


def function_GetPlaneEquation(v3d0, v3d1, v3d2):  #Exception wenn keine ebene
    n_v3d = ((v3d1.sub(v3d0)).vectorProduct(v3d2.sub(v3d0))).normalize()
    if not n_v3d.equals(Vector3D()): return (n_v3d, n_v3d.scalarProduct(v3d0))
    raise
"""