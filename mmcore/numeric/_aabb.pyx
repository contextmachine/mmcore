
cimport cython
import numpy as np
cimport numpy as cnp
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
