
cimport cython
import numpy as np
cimport numpy as cnp
from libc.math cimport fmin,fmax,fabs
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





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double _ray_aabb_intersect(
        double[3] bmin,
        double[3] bmax,
        double[3] ro,  # ray origin
        double[3] rd, # ray direction
        double* tmin_ref,
        double* tmax_ref
) noexcept nogil:
    """
    Slab intersection for a ray vs. AABB. Returns t >= 0 for the NEAREST forward intersection,
    or a negative value if no valid intersection. 
    (Returning negative is easier for the calling function to interpret "no hit".)
    """
    cdef double tmin = -1e300
    cdef double tmax = 1e300
    cdef double t1, t2, invD
    cdef int i
    cdef double tmp
    tmin_ref[0]=-1.
    tmax_ref[0]=-1.
    # For each dimension x=0, y=1, z=2
    for i in range(3):
        if fabs(rd[i]) < 1e-15:
            # Ray direction in this axis is ~0 => must be within [bmin[i], bmax[i]]
            if ro[i] < bmin[i] or ro[i] > bmax[i]:
                return -1.0  # no intersection
        else:
            invD = 1.0 / rd[i]
            t1 = (bmin[i] - ro[i]) * invD
            t2 = (bmax[i] - ro[i]) * invD

            # "Near" and "far" in param
            if t1 > t2:
                # swap
                tmp = t1
                t1 = t2
                t2 = tmp

            if t1 > tmin:
                tmin = t1
            if t2 < tmax:
                tmax = t2

            if tmax < tmin:
                return -1.0  # no intersection

    # If we reach here, [tmin, tmax] is the param interval for the entire intersection
    # We want the earliest intersection >= 0
    if tmax < 0:
        return -1.0  # entire intersection is behind the ray origin

    # If tmin >= 0, that's the first intersection
    tmin_ref[0] = tmin
    tmax_ref[0] = tmax
    if tmin >= 0:

        return tmin

    # otherwise, if tmin < 0 < tmax, the ray origin is inside the AABB
    # The first intersection in front is tmax
    if tmax >= 0:

        return tmax

    return -1.0
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ray_aabb_intersect(
        cnp.ndarray[double, ndim=2] bb,
        cnp.ndarray[double, ndim=1] ray_origin,
        cnp.ndarray[double, ndim=1] ray_dir
):
    """
    Ray vs. 3D AABB intersection.

    Parameters
    ----------
    aabb_min    : (3,) float64
    aabb_max    : (3,) float64
    ray_origin  : (3,) float64
    ray_dir     : (3,) float64

    Returns
    -------
    float or None
        If there's a valid forward intersection, returns the 't' parameter
        (distance along the ray_dir from ray_origin). Otherwise, None.
    """
    cdef:
        double[3] bmin, bmax, ro, rd
        double t
        int i
        double tmin
        double tmax

    # Copy input
    for i in range(3):
        bmin[i] = bb[0][i]
        bmax[i] = bb[1][i]
        ro[i] = ray_origin[i]
        rd[i] = ray_dir[i]

    t = _ray_aabb_intersect(bmin, bmax, ro, rd, &tmin,&tmax)
    if t >= 0:

            if tmin<= 0:
                tmin = 0.
            if tmax >= 1.:
                tmax = 1.
            return [tmin,tmax]

    else:
        return None

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int _segment_aabb_intersect(
        double[3] bmin,
        double[3] bmax,
        double[3] s,  # segment start
        double[3] e,  # segment end
        double out_t[2]  # up to 2 solutions
) noexcept nogil:
    """
    Slab intersection for segment vs. AABB. 
    Returns the number of intersection solutions in [0,1], or 0 if none.

    The param is t in [0,1] for S + t*(E-S).
    If there's 1 solution, out_t[0] holds it.
    If there's 2 solutions, out_t[0] < out_t[1].
    """
    cdef:
        double[3] d  # direction = e - s
        double tmin = 0.0
        double tmax = 1.0

        double t1, t2, invD
        int i
        double tmp

    d[0] = e[0] - s[0]
    d[1] = e[1] - s[1]
    d[2] = e[2] - s[2]

    for i in range(3):
        if fabs(d[i]) < 1e-15:
            # If the segment is parallel to this axis and out of bounds => no intersection
            if s[i] < bmin[i] or s[i] > bmax[i]:
                return 0
        else:
            invD = 1.0 / d[i]
            t1 = (bmin[i] - s[i]) * invD
            t2 = (bmax[i] - s[i]) * invD

            # Ensure t1 <= t2
            if t1 > t2:
                tmp = t1
                t1 = t2
                t2 = tmp

            # Narrow the segment param range
            if t1 > tmin:
                tmin = t1
            if t2 < tmax:
                tmax = t2

            if tmax < tmin:
                return 0

    # Now we have [tmin, tmax] as intersection interval with the infinite line
    # We intersect that with [0,1] for the segment
    if tmax < 0 or tmin > 1:
        return 0  # entirely out of [0,1]
    out_t[0] = tmin
    out_t[1] = tmax
    cdef double clip_min = tmin if (tmin > 0) else 0.0
    cdef double clip_max = tmax if (tmax < 1) else 1.0


    if clip_max < clip_min:
        return 0

    if fabs(clip_min - clip_max) < 1e-15:
        # It's effectively a single intersection
        return 1
    else:

        return 2
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def segment_aabb_intersect(
        cnp.ndarray[double, ndim=2] bb,
        cnp.ndarray[double, ndim=2] seg

):
    """
    Segment vs. 3D AABB intersection.

    Parameters
    ----------
    aabb_min  : (3,) float64
    aabb_max  : (3,) float64
    seg_start : (3,) float64
    seg_end   : (3,) float64

    Returns
    -------
    list of float or None
        The intersection parameter(s) t in [0,1]. Could be of length 1 or 2.
        If no intersection, returns None.

    The intersection point(s) can be computed as:
        P(t) = seg_start + t*(seg_end - seg_start)
    """
    cdef:
        double[3] bmin, bmax, s, e
        double t_out[2]
        int i, n
    for i in range(3):
        bmin[i] = bb[0][i]
        bmax[i] = bb[1][i]
        s[i] = seg[0][i]
        e[i] = seg[1][i]
    n = _segment_aabb_intersect(bmin, bmax, s, e, t_out)
    if n > 0:
        if t_out[0] <= 0:
            t_out[0]=0.
        if t_out[1]>=1.:
            t_out[1]=1.
        return [t_out[0],t_out[1]]


    return None
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def segment_aabb_clip(
        cnp.ndarray[double, ndim=2] bb,
        cnp.ndarray[double, ndim=2] seg,
        cnp.ndarray[double, ndim=2] out=None):
    cdef:
        double[3] bmin, bmax, s, e
        double t_out[2]
        int i, n
    if out is None:

        out=np.zeros((2,3))


    for i in range(3):
        bmin[i] = bb[0][i]
        bmax[i] = bb[1][i]
        s[i] =  seg[0][i]
        e[i] = seg[1][i]
    n = _segment_aabb_intersect(bmin, bmax, s, e, t_out)
    if n > 0:
        if t_out[0] <= 0:
            t_out[0] = 0.
        if t_out[1] >= 1.:
            t_out[1] = 1.
        for i in range(3):
            out[0][i]=s[i]+(e[i]-s[i])*t_out[0]
            out[1][i] = s[i] + (e[i] - s[i]) * t_out[1]
        return out
    return None
