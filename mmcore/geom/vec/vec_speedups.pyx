cdef extern from "math.h":
    long double sqrt(long double xx)
    long double sin(long double u)
    long double cos(long double v)
    long double atan2(long double a, double b)
    long double atan(long double a)

cdef extern from "limits.h":
    double DBL_MAX


import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dot_array_x_array(np.ndarray[DTYPE_t, ndim=2] vec_a, np.ndarray[DTYPE_t, ndim=2] vec_b):
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty((vec_a.shape[0],))
    cdef DTYPE_t item = 0.0
    for i in range(vec_a.shape[0]):

        item = 0.0
        for j in range(vec_a.shape[1]):
            item += vec_a[i, j] * vec_b[i, j]
        res[i] += item
    return res

cdef dot_vec_x_array(np.ndarray[DTYPE_t, ndim=1] vec_a, np.ndarray[DTYPE_t, ndim=2] vec_b):
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty((vec_b.shape[0],))
    cdef DTYPE_t item = 0.0
    for i in range(vec_b.shape[0]):
        item = 0.0
        for j in range(vec_a.shape[0]):
            item += vec_a[j] * vec_b[i, j]
        res[i] += item
    return res

cdef dot_array_x_vec(np.ndarray[DTYPE_t, ndim=2] vec_a, np.ndarray[DTYPE_t, ndim=1] vec_b):
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
cdef long double scalar_dot(np.ndarray[DTYPE_t, ndim=1] vec_a,
                            np.ndarray[DTYPE_t, ndim=1] vec_b):
    cdef long double res = 0.0
    for j in range(vec_a.shape[0]):
        res += vec_a[j] * vec_b[j]
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long double scalar_norm(np.ndarray[DTYPE_t, ndim=1] vec):
    cdef DTYPE_t res = 0.0
    for j in range(vec.shape[0]):
        res += (vec[j] ** 2)
    return sqrt(res)

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
cdef long double cdet(np.ndarray[DTYPE_t, ndim=2] arr):
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
