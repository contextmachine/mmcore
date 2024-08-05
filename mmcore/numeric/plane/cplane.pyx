cimport cython
from libc.math cimport fabs
cimport numpy as cnp
import numpy as np

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int find_max_row(double[5][5] m, int i, int n) noexcept nogil:
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
cdef int find_max_row_3x3(double[3][3] m, int i, int n) noexcept nogil:
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
cpdef void cevaluate_plane(double[:,:] pln, double[:] point, double[:] result) noexcept nogil:
    cdef int i
    for i in range(3):
        result[i]=  pln[0,i] +  pln[1, i] * point[0] + pln[2, i] * point[1] + pln[3, i] * point[2]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void cinverse_evaluate_plane(double[:,:] pln, double[:] point, double[:] result) noexcept nogil:
    cdef int i, j
    for i in range(3):
        result[i] = 0.0
        for j in range(3):
            result[i] += ((point[j]-pln[0,j]) * pln[i+1, j])




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] evaluate_plane(double[:,:] pln, double[:] point):

    cdef double[:] result=np.empty((3,))
    cevaluate_plane(pln,point,result)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] inverse_evaluate_plane(double[:,:] pln, double[:] point):
    cdef double[:] result=np.empty((3,))
    cinverse_evaluate_plane(pln,point,result)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,:] evaluate_plane_arr(double[:,:] pln, double[:,:] points):
    cdef int i
    cdef double[:,:] result=np.empty((points.shape[0], 3,))


    for i in range(points.shape[0]):
        cevaluate_plane(pln, points[i], result[i])
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,:] inverse_evaluate_plane_arr(double[:,:] pln, double[:,:]  points):
    cdef int i
    cdef double[:,:] result=np.empty((points.shape[0], 3,))


    for i in range(points.shape[0]):
            cinverse_evaluate_plane(pln, points[i], result[i])
    return result








@cython.boundscheck(False)
@cython.wraparound(False)
cdef void solve_system_5x5(double[5][5] m, double[5] v, double[5] x) noexcept nogil:
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
cdef void solve_system_3x3(double[3][3] m, double[3] v, double[:] x) noexcept nogil:
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
cdef inline void cross_product(double[:] v1, double[:] v2, double[:] res) noexcept nogil:
    res[0] = (v1[1] * v2[2]) - (v1[2] * v2[1])
    res[1] = (v1[2] * v2[0]) - (v1[0] * v2[2])
    res[2] = (v1[0] * v2[1]) - (v1[1] * v2[0])
@cython.boundscheck(False)
@cython.wraparound(False)
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
cpdef double[:,:] plane_plane_intersect(double[:,:] plane1, double[:,:] plane2):
    cdef double[:,:] result=np.empty((2,3))
    cplane_plane_intersect(plane1,plane2,result)
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


