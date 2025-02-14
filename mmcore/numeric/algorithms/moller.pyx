# cython: boundscheck=False, wraparound=False, cdivision=True,language_level=3
# ^ Some typical directives to speed things up further or suppress Python overhead.

from cython.parallel import prange
cimport cython
cimport numpy as cnp

from libc.math cimport fabs

cnp.import_array()


# Define a small epsilon for floating point comparisons
DEF EPSILON = 1e-8

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _is_close(double a, double b, double eps=EPSILON)  noexcept nogil:
    cdef bint res=fabs(a - b) < eps
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _point_equals(double[3] p, double[3] q) noexcept nogil:
    cdef bint res=_is_close(p[0], q[0], EPSILON) and _is_close(p[1], q[1], EPSILON) and _is_close(p[2], q[2], EPSILON)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _point_on_edge(double[3] P, double[3] V0, double[3] V1) noexcept nogil:
    """
    Check if point P lies on the edge defined by V0 and V1.
    """

    cdef double cross_x, cross_y, cross_z, len_sq, dot_prod
    cdef double vec_px = P[0] - V0[0]
    cdef double vec_py = P[1] - V0[1]
    cdef double vec_pz = P[2] - V0[2]
    cdef double edge_x = V1[0] - V0[0]
    cdef double edge_y = V1[1] - V0[1]
    cdef double edge_z = V1[2] - V0[2]

    # Compute cross product to check collinearity
    cross_x = vec_py * edge_z - vec_pz * edge_y
    cross_y = vec_pz * edge_x - vec_px * edge_z
    cross_z = vec_px * edge_y - vec_py * edge_x

    # If cross product is not near zero, P is not on the line
    cdef bint cond=_is_close(cross_x, 0.0, EPSILON) and  _is_close(cross_y, 0.0, EPSILON) and _is_close(cross_z, 0.0, EPSILON)
    if not cond:
        return 0

    # Compute dot product to check if P is between V0 and V1
    dot_prod = vec_px * edge_x + vec_py * edge_y + vec_pz * edge_z
    len_sq = edge_x * edge_x + edge_y * edge_y + edge_z * edge_z

    if dot_prod < -EPSILON or dot_prod > len_sq + EPSILON:
        return 0

    return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _classify_intersection(double[3] P, double[3] V0, double[3] V1, double[3] V2) noexcept nogil:
    """
    Classify the intersection point P with the triangle defined by V0, V1, V2.
    Returns:
        1: Intersection within the interior
        2: Intersection at a vertex
        3: Intersection along an edge
    """
    if _point_equals(P, V0):
        return 2
    if _point_equals(P, V1):
        return 2
    if _point_equals(P, V2):
        return 2
    if _point_on_edge(P, V0, V1) :
        return 3
    if _point_on_edge(P, V1, V2):
        return 3
    if _point_on_edge(P, V2, V0):
        return 3

    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _intersect_triangle_segment(
    double[3] V0, double[3] V1, double[3] V2,
    double[3] S, double[3] E,
    double[3] I
) noexcept nogil:
    """
    Möller–Trumbore intersection algorithm adapted for segments.
    Returns:
        1 if intersection exists and is within [0,1] segment parameters, else 0.
        If intersection exists, fills I with intersection point.
    """
    cdef double dir_x, dir_y, dir_z
    cdef double edge1_x, edge1_y, edge1_z
    cdef double edge2_x, edge2_y, edge2_z
    cdef double h_x, h_y, h_z
    cdef double a, f, u, v
    cdef double s_x, s_y, s_z
    cdef double q_x, q_y, q_z
    cdef double t

    # Compute direction vector of segment
    dir_x = E[0] - S[0]
    dir_y = E[1] - S[1]
    dir_z = E[2] - S[2]

    # Find vectors for two edges sharing V0
    edge1_x = V1[0] - V0[0]
    edge1_y = V1[1] - V0[1]
    edge1_z = V1[2] - V0[2]

    edge2_x = V2[0] - V0[0]
    edge2_y = V2[1] - V0[1]
    edge2_z = V2[2] - V0[2]

    # Begin calculating determinant - also used to calculate u parameter
    h_x = dir_y * edge2_z - dir_z * edge2_y
    h_y = dir_z * edge2_x - dir_x * edge2_z
    h_z = dir_x * edge2_y - dir_y * edge2_x

    a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z

    if -EPSILON < a < EPSILON:
        return 0  # This means parallel

    f = 1.0 / a
    s_x = S[0] - V0[0]
    s_y = S[1] - V0[1]
    s_z = S[2] - V0[2]

    u = f * (s_x * h_x + s_y * h_y + s_z * h_z)
    if u < -EPSILON or u > 1.0 + EPSILON:
        return 0

    q_x = s_y * edge1_z - s_z * edge1_y
    q_y = s_z * edge1_x - s_x * edge1_z
    q_z = s_x * edge1_y - s_y * edge1_x

    v = f * (dir_x * q_x + dir_y * q_y + dir_z * q_z)
    if v < -EPSILON or u + v > 1.0 + EPSILON:
        return 0

    # At this stage we can compute t to find out where the intersection point is on the line
    t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z)

    if t < -EPSILON or t > 1.0 + EPSILON:
        return 0  # Intersection not within the segment

    # Compute intersection point
    I[0] = S[0] + t * dir_x
    I[1] = S[1] + t * dir_y
    I[2] = S[2] + t * dir_z

    return 1
@cython.boundscheck(False)
@cython.wraparound(False)
def intersect_triangle_segment(
    double[:] V0,
      double[:] V1,
      double[:]V2,
      double[:] S,
      double[:] E,
      double[:] out
):
    """
    Intersect a 3D triangle with a segment.

    Parameters
    ----------
    V0, V1, V2 : ndarray of shape (3,), dtype float64
        Triangle vertices.
    S, E : ndarray of shape (3,), dtype float64
        Segment start and end points.


    """
    cdef double[3] c_V0, c_V1, c_V2, c_S, c_E, c_I
    cdef int exists, flag

    # Copy inputs to C arrays
    cdef int i
    
    c_V0[0] = V0[0]
    c_V1[0] = V1[0]
    c_V2[0] = V2[0]
    c_S[0]  = S[0]
    c_E[0]  = E[0]

    c_V0[1] = V0[1]       
    c_V1[1] = V1[1]       
    c_V2[1] = V2[1]       
    c_S[1]  = S[1]        
    c_E[1]  = E[1]        
    
    c_V0[2] = V0[2]       
    c_V1[2] = V1[2]       
    c_V2[2] = V2[2]       
    c_S[2]  = S[2]        
    c_E[2]  = E[2]        

    # Perform intersection
    exists = _intersect_triangle_segment(c_V0, c_V1, c_V2, c_S, c_E, c_I)

    if not exists:
        return  0

    # Classify the intersection
    #flag = _classify_intersection(c_I, c_V0, c_V1, c_V2)
    flag=1
    out[:]=c_I
    return flag
@cython.boundscheck(False)
@cython.wraparound(False)
def intersect_triangles_segment_one(
    double[:,:] V0,
    double[:,:] V1,
    double[:,:] V2,
    double[:] S,
    double[:] E,
        double[:] out
):
    """
    Intersect a 3D triangle with a segment.

    Parameters
    ----------
    V0, V1, V2 : ndarray of shape (3,), dtype float64
        Triangle vertices.
    S, E : ndarray of shape (3,), dtype float64
        Segment start and end points.

    Returns
    -------
    intersection_point : ndarray of shape (3,) or None
        The intersection point in 3D space.
    flag : int
        Intersection type flag:
            0: No intersection
            1: Intersection at a vertex
            2: Intersection along an edge
            4: Intersection within the interior
    """
    cdef int tris=V0.shape[0]

    cdef double[3] c_V0, c_V1, c_V2, c_S, c_E, c_I
    cdef int exists, flag
    cdef int ixs,i
    
    c_S[0] = S[0]
    c_E[0] = E[0]
    c_S[1] = S[1]  
    c_E[1] = E[1]  
    c_S[ 2] = S[ 2]  
    c_E[ 2] = E[ 2]  
    # Copy inputs to C arrays
    for ixs in range(tris):

        
        c_V0[0] = V0[ixs][0]
        c_V1[0] = V1[ixs][0]
        c_V2[0] = V2[ixs][0]
        
        c_V0[1] = V0[ixs][1]
        c_V1[1] = V1[ixs][1]
        c_V2[1] = V2[ixs][1]
        
        c_V0[2] = V0[ixs][2]
        c_V1[2] = V1[ixs][2]
        c_V2[2] = V2[ixs][2]

        # Perform intersection
        exists = _intersect_triangle_segment(c_V0, c_V1, c_V2, c_S, c_E, c_I)
        if exists:
            # Classify the intersection
            #flag = _classify_intersection(c_I, c_V0, c_V1, c_V2)
            flag=1
            out[:]=c_I
            return (out, flag)

    return (None, 0)

@cython.boundscheck(False)
@cython.wraparound(False)
def intersect_triangles_segment(
    double[:,:] V0,
    double[:,:] V1,
    double[:,:] V2,
    double[:] S,
    double[:] E,
        double[:,:] outs,
        int[:] flags
):
    """
    Intersect a 3D triangle with a segment.

    Parameters
    ----------
    V0, V1, V2 : ndarray of shape (N,3), dtype float64
        Triangle vertices.
    S, E : ndarray of shape (3,), dtype float64
        Segment start and end points.
    outs, flags : ndarray of shape (N,3)  dtype float64 and (N,) dtype intc
        Intersection Points, Intersection Type

    """
    cdef int tris=V0.shape[0]

    cdef double[3] c_V0, c_V1, c_V2, c_S, c_E, c_I
    cdef int exists, flag
    cdef int ixs,i

    c_S[0] = S[0]
    c_E[0] = E[0]
    c_S[1] = S[1]
    c_E[1] = E[1]
    c_S[ 2] = S[ 2]
    c_E[ 2] = E[ 2]

    # Copy inputs to C arrays
    for ixs in range(tris):


        c_V0[0] = V0[ixs][0]
        c_V1[0] = V1[ixs][0]
        c_V2[0] = V2[ixs][0]

        c_V0[1] = V0[ixs][1]
        c_V1[1] = V1[ixs][1]
        c_V2[1] = V2[ixs][1]

        c_V0[2] = V0[ixs][2]
        c_V1[2] = V1[ixs][2]
        c_V2[2] = V2[ixs][2]

        # Perform intersection
        exists = _intersect_triangle_segment(c_V0, c_V1, c_V2, c_S, c_E, c_I)
        if exists:
            # Classify the intersection
            flags[ixs] = _classify_intersection(c_I, c_V0, c_V1, c_V2)

            outs[ixs,0]=c_I[0]
            outs[ixs, 1] = c_I[1]
            outs[ixs, 2] = c_I[2]
        else:
            flags[ixs]=0


@cython.boundscheck(False)
@cython.wraparound(False)

def intersect_triangles_segment_parallel(
    double[:,:] V0,
    double[:,:] V1,
    double[:,:] V2,
    double[:] S,
    double[:] E,
        double[:,:] outs,
        int[:] flags
):
    """
    Intersect a 3D triangle with a segment.

    Parameters
    ----------
    V0, V1, V2 : ndarray of shape (N,3), dtype float64
        Triangle vertices.
    S, E : ndarray of shape (3,), dtype float64
        Segment start and end points.
    outs, flags : ndarray of shape (N,3)  dtype float64 and (N,1) dtype intc
        Intersection Points, Intersection Type

    """
    cdef int tris=V0.shape[0]

    cdef double[3] c_V0, c_V1, c_V2, c_S, c_E, c_I
    cdef int exists, flag
    cdef int ixs,i
    with nogil:
        c_S[0] = S[0]
        c_E[0] = E[0]
        c_S[1] = S[1]
        c_E[1] = E[1]
        c_S[ 2] = S[ 2]
        c_E[ 2] = E[ 2]

        # Copy inputs to C arrays
        for ixs in prange(tris, schedule='static'):


            c_V0[0] = V0[ixs][0]
            c_V1[0] = V1[ixs][0]
            c_V2[0] = V2[ixs][0]

            c_V0[1] = V0[ixs][1]
            c_V1[1] = V1[ixs][1]
            c_V2[1] = V2[ixs][1]

            c_V0[2] = V0[ixs][2]
            c_V1[2] = V1[ixs][2]
            c_V2[2] = V2[ixs][2]

            # Perform intersection
            exists = _intersect_triangle_segment(c_V0, c_V1, c_V2, c_S, c_E, c_I)
            if exists:
                # Classify the intersection
                flags[ixs] = _classify_intersection(c_I, c_V0, c_V1, c_V2)

                outs[ixs,0]=c_I[0]
                outs[ixs, 1] = c_I[1]
                outs[ixs, 2] = c_I[2]
            else:
                flags[ixs]=0



