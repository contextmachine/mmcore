#cython: boundscheck=False, wraparound=False, cdivision=True
cimport cython
from libc.math cimport fabs
cimport numpy as cnp
import numpy as np

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline int find_max_row(double[5][5] m, int i, int n) noexcept nogil:
    cdef int l= i
    cdef int r = n - 1
    cdef int mid

    while l < r:
        mid = (l + r) // 2
        if fabs(m[mid][i]) > fabs(m[mid + 1][i]):
            r = mid
        else:
            l = mid + 1
    return l
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline int find_max_row_3x3(double[3][3] m, int i, int n) noexcept nogil:
    cdef int l= i
    cdef int r = n - 1
    cdef int mid

    while l < r:
        mid = (l + r) // 2
        if fabs(m[mid][i]) > fabs(m[mid + 1][i]):
            r = mid
        else:
            l = mid + 1
    return l
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef void cevaluate_plane(double[:,:] pln, double[:] point, double[:] result) noexcept nogil:
    cdef int i
    for i in range(3):
        result[i]=  pln[0,i] +  pln[1, i] * point[0] + pln[2, i] * point[1] + pln[3, i] * point[2]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef void cinverse_evaluate_plane(double[:,:] pln, double[:] point, double[:] result) noexcept nogil:
    cdef int i, j
    for i in range(3):
        result[i] = 0.0
        for j in range(3):
            result[i] += ((point[j]-pln[0,j]) * pln[i+1, j])



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double[:] evaluate_plane(double[:,:] pln, double[:] point):

    cdef double[:] result=np.empty((3,))
    cevaluate_plane(pln,point,result)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double[:] inverse_evaluate_plane(double[:,:] pln, double[:] point):
    cdef double[:] result=np.empty((3,))
    cinverse_evaluate_plane(pln,point,result)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double[:,:] evaluate_plane_arr(double[:,:] pln, double[:,:] points):
    cdef int i
    cdef double[:,:] result=np.empty((points.shape[0], 3,))


    for i in range(points.shape[0]):
        cevaluate_plane(pln, points[i], result[i])
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double[:,:] inverse_evaluate_plane_arr(double[:,:] pln, double[:,:]  points):
    cdef int i
    cdef double[:,:] result=np.empty((points.shape[0], 3,))


    for i in range(points.shape[0]):
            cinverse_evaluate_plane(pln, points[i], result[i])
    return result








@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef  inline void solve_system_5x5(double[5][5] m, double[5] v, double[5] x) noexcept nogil:
    cdef int i, j, k,h, max_row
    cdef int n = 5
    cdef double factor
    cdef double temp1,temp2

    for i in range(n):
        max_row = find_max_row(m, i, n)
        if i != max_row:
            for h in range(n):
                temp1=m[i][h]
                temp2 =m[max_row][h]

                m[i][h]=temp2
                m[max_row][h]=temp1

            temp1=v[max_row]
            temp2=v[i]
            v[i]=temp1
            v[max_row]=temp2



        for j in range(i + 1, n):
            factor = m[j][i] / m[i][i]

            for k in range(i, n):
                m[j][k] -= factor * m[i][k]
            v[j] -= factor * v[i]

    for i in range(n):

        x[n - 1 - i] = v[n - 1 - i] / m[n - 1 - i][n - 1 - i]
        for j in range(n - 1 - i):

            v[j] -= m[j][n - 1 - i] * x[n - 1 - i]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline void solve_system_3x3(double[3][3] m, double[3] v, double[:] x) noexcept nogil:
    cdef int i, j, k,h, max_row
    cdef int n = 3
    cdef double factor
    cdef double temp1,temp2

    for i in range(n):
        max_row = find_max_row_3x3(m, i, n)
        if i != max_row:
            for h in range(n):
                temp1=m[i][h]
                temp2 =m[max_row][h]

                m[i][h]=temp2
                m[max_row][h]=temp1

            temp1=v[max_row]
            temp2=v[i]
            v[i]=temp1
            v[max_row]=temp2



        for j in range(i + 1, n):
            factor = m[j][i] / m[i][i]

            for k in range(i, n):
                m[j][k] -= factor * m[i][k]
            v[j] -= factor * v[i]

    for i in range(n):

        x[n - 1 - i] = v[n - 1 - i] / m[n - 1 - i][n - 1 - i]
        for j in range(n - 1 - i):

            v[j] -= m[j][n - 1 - i] * x[n - 1 - i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline void cross_product(double[:] v1, double[:] v2, double[:] res) noexcept nogil:
    res[0] = (v1[1] * v2[2]) - (v1[2] * v2[1])
    res[1] = (v1[2] * v2[0]) - (v1[0] * v2[2])
    res[2] = (v1[0] * v2[1]) - (v1[1] * v2[0])
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline double dot_product(double[:] v1, double[:] v2) noexcept nogil:
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void cplane_plane_intersect(double[:,:] plane1, double[:,:] plane2, double[:,:] result) noexcept nogil:
    cdef double[5][5] A;
    cdef double[5] b;
    cdef double[5] solution;
    cdef double[:] normal1=plane1[3];
    cdef double[:] normal2=plane2[3];
    cdef int i,j
    # Create augmented matrix
    for i in range(3):
        for j in range(3):
            if i==j:
                A[i][j]=2.
            else:
                A[i][j]=0.
    A[3][3] = 0.
    A[3][4] = 0.
    A[4][4] = 0.
    A[4][3] = 0.

    for i in range(3):
        A[i][3] = normal1[i]
        A[i][4] = normal2[i]
        A[3][i] = normal1[i]
        A[4][i] = normal2[i]


    b[0]=plane1[0][0]
    b[1]=plane1[0][1]
    b[2]=plane1[0][2]
    b[3] = dot_product(plane1[0], normal1)
    b[4] = dot_product(plane2[0], normal2)
    solve_system_5x5(A, b, solution)
    result[0][0] = solution[0]
    result[0][1] = solution[1]
    result[0][2] = solution[2]
    cross_product(normal1, normal2, result[1])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void cplane_plane_normal_intersect(double[:] origin1, double[:] normal1,double[:]  origin2, double[:] normal2,double[:,:] result) noexcept nogil:
    cdef double[5][5] A;
    cdef double[5] b;
    cdef double[5] solution;

    cdef int i,j
    # Create augmented matrix
    for i in range(3):
        for j in range(3):
            if i==j:
                A[i][j]=2.
            else:
                A[i][j]=0.
    A[3][3] = 0.
    A[3][4] = 0.
    A[4][4] = 0.
    A[4][3] = 0.

    for i in range(3):
        A[i][3] = normal1[i]
        A[i][4] = normal2[i]
        A[3][i] = normal1[i]
        A[4][i] = normal2[i]


    b[0]=origin1[0]
    b[1]=origin1[1]
    b[2]=origin1[2]
    b[3] = dot_product(origin1, normal1)
    b[4] = dot_product(origin2, normal2)
    solve_system_5x5(A, b, solution)
    result[0][0] = solution[0]
    result[0][1] = solution[1]
    result[0][2] = solution[2]
    cross_product(normal1, normal2, result[1])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,:] plane_plane_intersect(double[:,:] plane1, double[:,:] plane2):
    cdef double[:,:] result=np.empty((2,3))
    cplane_plane_intersect(plane1,plane2,result)
    return result
@cython.boundscheck(False)
@cython.wraparound(False)
def plane_plane_normal_intersect(double[:] origin1, double[:] normal1,double[:]  origin2, double[:] normal2,double[:,:] result=None):
    if result is None:
        result=np.empty((2,3))
    cplane_plane_normal_intersect(origin1,normal1, origin2, normal2, result)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void cplanes_plane_intersect(double[:,:,:] planes1, double[:,:] plane2, double[:,:,:] result) noexcept nogil:
    cdef int i
    for i in range(planes1.shape[0]):
        cplane_plane_intersect(planes1[i],plane2,result[i])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,:,:] planes_plane_intersect(double[:,:,:] planes1, double[:,:] plane2):
    cdef int i
    cdef double[:,:,:] result=np.empty((planes1.shape[0],2,3))

    cplanes_plane_intersect(planes1,plane2,result)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void cplanes_planes_intersect(double[:,:,:] planes1, double[:,:,:] planes2, double[:,:,:] result) noexcept nogil:
    cdef int i
    for i in range(planes1.shape[0]):
        cplane_plane_intersect(planes1[i],planes2[i],result[i])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,:,:] planes_planes_intersect(double[:,:,:] planes1, double[:,:,:] planes2):
    cdef int i
    cdef double[:,:,:] result=np.empty((planes1.shape[0],2,3))
    if planes1.shape[0] != planes2.shape[0]:
        raise ValueError("planes1 and planes2 length should be equal!")

    cplanes_planes_intersect(planes1,planes2,result)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void cplane_plane_plane_intersect(double[:,:] plane1, double[:,:] plane2,double[:,:] plane3, double[:] result) noexcept nogil:
    cdef double[3][3] A;
    cdef double[3] b;
    cdef double[:] normal1=plane1[3];
    cdef double[:] normal2=plane2[3];
    cdef double[:] normal3=plane3[3];
    A[0][0] = normal1[0]
    A[0][1] = normal1[1]
    A[0][2] = normal1[2]
    A[1][0] = normal2[0]
    A[1][1] = normal2[1]
    A[1][2] = normal2[2]
    A[2][0] = normal3[0]
    A[2][1] = normal3[1]
    A[2][2] = normal3[2]

    b[0] = dot_product(plane1[0], normal1)
    b[1] = dot_product(plane2[0], normal2)
    b[2] =  dot_product(plane3[0], normal3)

    solve_system_3x3(A, b, result)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void cplane_plane_plane_intersect_points_and_normals(double[:] pt1,double[:] normal1, double[:] pt2, double[:] normal2, double[:] pt3, double[:] normal3,double[:] result) noexcept nogil:
    """
    :param pt1: A 1D array representing the origin of the first plane. Shape should be (3,).
    :param normal1: A 1D array representing the normal of the first plane. Shape should be (3,).
    :param pt2: A 1D array representing the origin of the second plane. Shape should be (3,).
    :param normal2: A 1D array representing the normal of the second plane. Shape should be (3,).
    :param pt3: A 1D array representing the origin of the third plane. Shape should be (3,).
    :param normal3: A 1D array representing the normal of the third plane. Shape should be (3,).
    :return: A 1D array representing the intersection point of the three planes. Shape is (3,).

    """
    cdef double[3][3] A;
    cdef double[3] b;
    A[0][0] = normal1[0]
    A[0][1] = normal1[1]
    A[0][2] = normal1[2]
    A[1][0] = normal2[0]
    A[1][1] = normal2[1]
    A[1][2] = normal2[2]
    A[2][0] = normal3[0]
    A[2][1] = normal3[1]
    A[2][2] = normal3[2]

    b[0] = dot_product(pt1, normal1)
    b[1] = dot_product(pt2, normal2)
    b[2] =  dot_product(pt3, normal3)

    solve_system_3x3(A, b, result)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] plane_plane_plane_intersect(double[:,:] plane1, double[:,:] plane2,double[:,:] plane3):

    cdef int i
    cdef double[:] result=np.empty((3,))
    cplane_plane_plane_intersect(plane1,plane2,plane3,result)
    return result
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] plane_plane_plane_intersect_points_and_normals(double[:] pt1, double[:] normal1, double[:] pt2, double[:] normal2, double[:] pt3, double[:] normal3):
    """
    :param pt1: A 1D array representing the origin of the first plane. Shape should be (3,).
    :param normal1: A 1D array representing the normal of the first plane. Shape should be (3,).
    :param pt2: A 1D array representing the origin of the second plane. Shape should be (3,).
    :param normal2: A 1D array representing the normal of the second plane. Shape should be (3,).
    :param pt3: A 1D array representing the origin of the third plane. Shape should be (3,).
    :param normal3: A 1D array representing the normal of the third plane. Shape should be (3,).
    :return: A 1D array representing the intersection point of the three planes. Shape is (3,).

    """
    cdef int i
    cdef double[:] result=np.empty((3,))
    cplane_plane_plane_intersect_points_and_normals( pt1,normal1, pt2,normal2, pt3, normal3,result)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def plane_plane_intersection(a, b, result=None):
    cdef tuple a_initial_shape = a.shape
    cdef tuple  b_initial_shape = b.shape
    cdef bint a_reshaped=False
    cdef bint b_reshaped = False
    cdef tuple a_new_shape = a_initial_shape
    cdef tuple b_new_shape = b_initial_shape
    cdef cnp.ndarray[double, ndim=1] a_flt
    cdef cnp.ndarray[double, ndim=1] b_flt

    if len(a.shape) > 3:
        a_flt = a.flatten()
        a_new_shape=int(len(a_flt)//12),4,3
        a=a_flt.reshape( a_new_shape)
        a_reshaped = True
        if result is not None:
            result.reshape((b_new_shape[0], 2, 3))
    if len(b.shape) > 3:
        b_flt = b.flatten()
        b_new_shape=int(len(b_flt)//12),4,3
        b=a_flt.reshape( b_new_shape)
        b_reshaped = True
        if (result is not None) and (not a_reshaped):
            result.reshape((b_new_shape[0],2,3))

    if len(a.shape)==3 and len(b.shape)==3:
        if result is None:
            result = np.empty((len(a),2,3))

        cplanes_planes_intersect(a, b, result)
    elif len(a.shape) == 3 and len(b.shape)==2:
        if result is None:
            result = np.empty((len(a), 2, 3))

        cplanes_plane_intersect(a, b,result)
    elif len(b.shape) == 3 and len(a.shape) == 2:
        if result is None:
            result = np.empty((len(b), 2, 3))
        cplanes_plane_intersect( b,a,result)
    elif len(a.shape)==2 and len(b.shape)==2:
        if result is None:
            result = np.empty(( 2, 3))
        cplane_plane_intersect(a,b,result)
    else:
        raise ValueError("The arguments are totally incorrect!")
    if a_reshaped or b_reshaped:

        return result.reshape( (*(a_initial_shape[:(len(a_initial_shape)-2)]),2,3) if a_reshaped else (*(b_initial_shape[:(len(a_initial_shape)-2)]),2,3))
    else:
        return result



"""
Example usage:

Use the intersection function in Python:


   plane_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
   plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
   ray_origin   = np.array([0.0, -1.0, 0.0], dtype=np.float64)
   ray_dir      = np.array([0.0,  1.0, 0.0], dtype=np.float64)

   intersection_point = ray_plane_intersection(plane_origin,
                                               plane_normal,
                                               ray_origin,
                                               ray_dir)
   if intersection_point is not None:
       print("Intersection:", intersection_point)
   else:
       print("No intersection or behind the ray origin.")
"""


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline bint _ray_plane_intersect(
    double[3] plane_origin,
    double[3] plane_normal,
    double[3] ray_origin,
    double[3] ray_direction,
    double[3] out_intersection
) noexcept nogil:
    """
    Returns True (1) if intersection exists (t >= 0), and fills out_intersection.
    Returns False (0) otherwise.
    """

    cdef double denom
    cdef double numer
    cdef double t

    # dot(plane_normal, ray_direction)
    denom = (plane_normal[0]*ray_direction[0] +
             plane_normal[1]*ray_direction[1] +
             plane_normal[2]*ray_direction[2])

    # If denom ~ 0 => ray is parallel to the plane
    if fabs(denom) < 1e-15:
        return False

    # dot(plane_normal, (plane_origin - ray_origin))
    numer = (plane_normal[0]*(plane_origin[0] - ray_origin[0]) +
             plane_normal[1]*(plane_origin[1] - ray_origin[1]) +
             plane_normal[2]*(plane_origin[2] - ray_origin[2]))

    # Parameter t
    t = numer / denom

    # If t < 0 => intersection is "behind" the ray origin => no valid intersection
    if t < 0:
        return False

    # Intersection = ray_origin + t * ray_direction
    out_intersection[0] = ray_origin[0] + t * ray_direction[0]
    out_intersection[1] = ray_origin[1] + t * ray_direction[1]
    out_intersection[2] = ray_origin[2] + t * ray_direction[2]

    return True


def ray_plane_intersection(
    cnp.ndarray[cnp.float64_t, ndim=2] plane,
    cnp.ndarray[cnp.float64_t, ndim=2] ray
):
    """
    A Python-visible function to intersect a 3D ray with a plane.

    Parameters
    ----------
    plane  : ndarray[float64, ndim=1] of shape (2,3)

    ray    : ndarray[float64, ndim=1] of shape (2,3)


    Returns
    -------
    intersection_point : np.ndarray of shape (3,) or None
        If intersection exists and t >= 0, returns the [x, y, z] intersection.
        If the ray is parallel or intersection is behind origin, returns None.
    """
    cdef:
        double[3] p_o, p_n, r_o, r_d, result
        bint hit
        int i

    #cython: boundscheck=False, wraparound=False, cdivision=True

    # Copy data into fixed-size C arrays for faster access in Cython
    for i in range(3):
        p_o[i] = plane[0][i]
        p_n[i] = plane[1][i]
        r_o[i] = ray[0][i]
        r_d[i] = ray[1][i]

    hit = _ray_plane_intersect(p_o, p_n, r_o, r_d, result)
    if hit:
        return np.array([result[0], result[1], result[2]], dtype=np.float64)
    else:
        return None



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline bint _ray_param_plane_intersect(
    double[3] plane_origin,
    double[3] plane_xaxis,
    double[3] plane_yaxis,
    double[3] plane_normal,
    double[3] ray_origin,
    double[3] ray_direction,
    double[3] out_intersection,
    double[2] out_uv
) noexcept nogil:
    """
    Returns True (1) if intersection exists (t >= 0). Fills out_intersection and out_uv[0], out_uv[1].
    Returns False (0) otherwise.

    Parametric plane: P(u, v) = plane_origin + u*plane_xaxis + v*plane_yaxis.
    We also store the (u, v) in out_uv if intersection is valid.
    """
    cdef double denom, numer, t
    cdef double ix, iy, iz   # intersection coords
    cdef double diffx, diffy, diffz
    cdef double xd, yd
    # For solving u, v from intersection in plane basis:
    #   M = [[ X·X,  X·Y ],
    #        [ Y·X,  Y·Y ]]
    #   RHS = [ X·diff, Y·diff ]
    cdef double a, b, c_, det  # a= X·X, b= X·Y, c_=Y·Y

    ###################
    # 1) Ray-plane intersection (same as above)
    ###################
    denom = (plane_normal[0] * ray_direction[0] +
             plane_normal[1] * ray_direction[1] +
             plane_normal[2] * ray_direction[2])

    if fabs(denom) < 1e-15:
        return False

    numer = (plane_normal[0] * (plane_origin[0] - ray_origin[0]) +
             plane_normal[1] * (plane_origin[1] - ray_origin[1]) +
             plane_normal[2] * (plane_origin[2] - ray_origin[2]))

    t = numer / denom
    if t < 0:
        return False

    # Intersection = ray_origin + t * ray_direction
    ix = ray_origin[0] + t * ray_direction[0]
    iy = ray_origin[1] + t * ray_direction[1]
    iz = ray_origin[2] + t * ray_direction[2]

    out_intersection[0] = ix
    out_intersection[1] = iy
    out_intersection[2] = iz

    ###################
    # 2) Compute (u, v) such that:
    #    ix, iy, iz  =  plane_origin + u * plane_xaxis + v * plane_yaxis
    # => diff = I - plane_origin = u*X + v*Y
    ###################
    diffx = ix - plane_origin[0]
    diffy = iy - plane_origin[1]
    diffz = iz - plane_origin[2]

    # X·X, X·Y, Y·Y
    a = (plane_xaxis[0]*plane_xaxis[0] +
         plane_xaxis[1]*plane_xaxis[1] +
         plane_xaxis[2]*plane_xaxis[2])

    b = (plane_xaxis[0]*plane_yaxis[0] +
         plane_xaxis[1]*plane_yaxis[1] +
         plane_xaxis[2]*plane_yaxis[2])

    c_ = (plane_yaxis[0]*plane_yaxis[0] +
          plane_yaxis[1]*plane_yaxis[1] +
          plane_yaxis[2]*plane_yaxis[2])

    # dot(X, diff)
    xd = (plane_xaxis[0]*diffx +
          plane_xaxis[1]*diffy +
          plane_xaxis[2]*diffz)

    # dot(Y, diff)
    yd = (plane_yaxis[0]*diffx +
          plane_yaxis[1]*diffy +
          plane_yaxis[2]*diffz)

    # Solve the 2x2 system:
    #   [ a  b ] [ u ] = [ xd ]
    #   [ b  c ] [ v ]   [ yd ]
    #
    #   det = a*c - b^2
    det = a * c_ - b * b
    if fabs(det) < 1e-15:
        return False  # X and Y are degenerate or very close to collinear

    out_uv[0] = (xd * c_ - b * yd) / det   # u
    out_uv[1] = (-b * xd + a * yd) / det  # v

    return True


def ray_plane_parametric_intersection(

        cnp.ndarray[cnp.float64_t, ndim=2] plane,
        cnp.ndarray[cnp.float64_t, ndim=2] ray
):
    """
    Intersect a 3D ray with a parametric plane:
        P(u,v) = plane_origin + u*plane_xaxis + v*plane_yaxis

    Parameters
    ----------
    plane_origin  : (3,) float64
        A known point on the plane.
    plane_xaxis   : (3,) float64
        The 'X' direction vector of the plane.
    plane_yaxis   : (3,) float64
        The 'Y' direction vector of the plane.
    plane_normal  : (3,) float64
        The normal to the plane.
    ray_origin    : (3,) float64
    ray_direction : (3,) float64

    Returns
    -------
    (point_3d, (u, v)) or None
        point_3d : ndarray of shape (3,)
        (u, v)   : the parametric coordinates in the plane basis
        If there's no valid forward intersection, returns None.
    """
    cdef:
        double[3] p_o, p_x, p_y, p_n, r_o, r_d, result
        double[2] uv
        bint hit
        int i

    # Copy input arrays to fixed-size C arrays
    for i in range(3):
        p_o[i] = plane[0][i]
        p_x[i] = plane[1][i]
        p_y[i] = plane[2][i]
        p_n[i] = plane[3][i]
        r_o[i] = ray[0][i]
        r_d[i] = ray[1][i]

    hit = _ray_param_plane_intersect(p_o, p_x, p_y, p_n, r_o, r_d, result, uv)
    if hit:
        # Return (intersection_point, (u, v))
        point_3d = np.array([result[0], result[1], result[2]], dtype=np.float64)
        return (point_3d, (uv[0], uv[1]))
    else:
        return None
