cdef extern from "math.h":
    long double sqrt(long double xx)
    long double sin(long double u)
    long double cos(long double v)
    long double atan2(long double a, double b)

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def cartesian(np.ndarray[DTYPE_t, ndim=2] rtp):
    cdef np.ndarray[DTYPE_t, ndim=2] pts = np.empty((rtp.shape[0], 3))
    for i in range(rtp.shape[0]):
        pts[i, 0] = rtp[i, 0] * sin(rtp[i, 1]) * cos(rtp[i, 2])
        pts[i, 1] = rtp[i, 0] * sin(rtp[i, 1]) * sin(rtp[i, 2])
        pts[i, 2] = rtp[i, 0] * cos(rtp[i, 1])
    return pts

@cython.boundscheck(False)
@cython.wraparound(False)
def spherical(np.ndarray[DTYPE_t, ndim=2] xyz):
    cdef np.ndarray[DTYPE_t, ndim=2] pts = np.empty((xyz.shape[0], 3))
    cdef long double XsqPlusYsq
    for i in range(xyz.shape[0]):
        XsqPlusYsq = xyz[i, 0] ** 2 + xyz[i, 1] ** 2
        pts[i, 0] = sqrt(XsqPlusYsq + xyz[i, 2] ** 2)
        pts[i, 1] = atan2(sqrt(XsqPlusYsq), xyz[i, 2])
        pts[i, 2] = atan2(xyz[i, 1], xyz[i, 0])
    return pts
