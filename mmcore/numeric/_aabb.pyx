
cimport cython
import numpy as np
cimport numpy as cnp
from libc.math cimport fmin,fmax
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void caabb(double[:,:] points, double[:] min_point, double[:] max_point) noexcept nogil:
    """
    AABB (Axis-Aligned Bounding Box) of a point collection.
    :param points: Points
    :rtype: np.ndarray[(2, K), np.dtype[float]] where:
        - N is a points count.
        - K is the number of dims. For example in 3d case (x,y,z) K=3.
    :return: AABB of a point collection.
    :rtype: np.ndarray[(2, K), np.dtype[float]] at [a1_min, a2_min, ... an_min],[a1_max, a2_max, ... an_max],
    """

    cdef int K = points.shape[1]
    cdef int N = points.shape[0]
    #cdef double[:,:] min_max_vals = np.empty((2,K), dtype=np.float64)
    cdef double p
    cdef int i, j

    # Initialize min_vals and max_vals with the first point's coordinates
    for i in range(K):
        min_point[i] = points[0, i]
        max_point[i] = points[0, i]

    # Find the min and max for each dimension
    for j in range(1, N):
        for i in range(K):
            p=points[j, i]
            if  p <   min_point[i]:
                  min_point[i] =  p
            if  p >   max_point[i]:
                  max_point[i] =  p

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def aabb(double[:,:] pts, double[:,:] bb=None):
    if bb is None:
        bb   =np.zeros((2,3))

    caabb(pts,bb[0],bb[1])
    return bb

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline bint caabb_intersect(double[:,:] bb1, double[:,:] bb2) noexcept nogil:
    cdef bint temp
    for i in range(bb1.shape[1]):
        temp = (bb1[0][i] <= bb2[1][i] ) and (bb1[1][i] >= bb2[0][i])
        if temp ==0:
            return temp
    return temp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline bint caabb_intersect_3d(double[:,:] bb1, double[:,:] bb2) noexcept nogil:
    cdef bint temp=(bb1[0][0] <= bb2[1][0] ) and (bb1[1][0] >= bb2[0][0])  and (bb1[0][1] <= bb2[1][1] ) and (bb1[1][1] >= bb2[0][1]) and (bb1[0][2] <= bb2[1][2] ) and (bb1[1][2] >= bb2[0][2])
    return temp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def aabb_intersect(double[:,:] bb1, double[:,:] bb2):
    cdef bint result
    if bb1.shape[1]==3:
        result=caabb_intersect_3d(bb1,bb2)
    else:
        result=caabb_intersect(bb1,bb2)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline bint caabb_intersection_3d(double[:,:] self, double[:,:] other, double[:,:] result) noexcept nogil:
    cdef double max_min_x = fmax(self[0][0], other[0][0])
    cdef double max_min_y = fmax(self[0][1], other[0][1])
    cdef double max_min_z = fmax(self[0][2], other[0][2])


    cdef double min_max_x = fmin(self[1][0], other[1][0])
    cdef double min_max_y = fmin(self[1][1], other[1][1])
    cdef double min_max_z = fmin(self[1][2], other[1][2])
    cdef bint r=0
    if max_min_x > min_max_x or max_min_y > min_max_y or max_min_z > min_max_z:
        return r
    result[0,0]=max_min_x
    result[0,1]=max_min_y
    result[0,2]=max_min_z
    result[1,0]=min_max_x
    result[1,1]=min_max_y
    result[1,2]=min_max_z
    r =1
    return r

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def aabb_intersection(double[:,:] bb1, double[:,:] bb2, cnp.ndarray[double, ndim=2] result=None):
    cdef bint success
    if result is None:
        result = np.zeros((2,3))
    success=caabb_intersection_3d(bb1,bb2,result)
    if success:
        return result
    else:
        return None
