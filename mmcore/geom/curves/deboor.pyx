cimport cython

cimport numpy as np
import numpy as np
ctypedef struct xyz:
    double x,y,z;
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cdeboor(double[:] knots, double t, int i, int k) noexcept nogil:
    """
    Calculating basis function with de Boor algorithm
    """
    # print(t,i,k)


    cdef double c1,c2;

    if k == 0:
        return 1.0 if knots[i] <= t <= knots[i + 1] else 0.0
    if knots[i + k] == knots[i]:
        c1 = 0.0
    else:
        c1 = (t  - knots[i]) / (knots[i + k] - knots[i]) * cdeboor(knots, t, i, k - 1)

    if knots[i + k + 1] == knots[i + 1]:
        c2 = 0.0
    else:
        c2 = ((knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) * cdeboor(knots,t, i + 1, k - 1))

    return c1 + c2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef xyz cevaluate_nurbs(double t, double[:,:] cpts, double[:] knots, double[:] weights, int degree  ):
    cdef Py_ssize_t n=cpts.shape[0]
    cdef double b;
    cdef double crd;
    cdef double sum_of_weights =0.;

    cdef xyz result=xyz(0.,0.,0.);
    for i in range(n):
        b = cdeboor(knots, t, i, degree)* weights[i]
        result.x += b * cpts[i][0]
        result.y  += b * cpts[i][1]
        result.z  += b *  cpts[i][2]
        sum_of_weights += b
    result.x = result.x / sum_of_weights
    result.y = result.y / sum_of_weights
    result.z = result.z / sum_of_weights
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cevaluate_nurbs_multi( double[:] t,double[:,:] cpts, double[:] knots, double[:] weights, int degree,double[:,:] result) :

    cdef xyz pt;

    for i in range(t.shape[0]):
        pt=cevaluate_nurbs(t[i],cpts,knots,weights,degree)
        result[i][0]=pt.x
        result[i][1] = pt.y
        result[i][2] = pt.z


def deboor(double[:] knots, double t, int i, int k)   :
    cdef double d=cdeboor(knots,t,i,k)
    return d

def evaluate_nurbs(double t, double[:,:] cpts, double[:] knots, double[:] weights, int degree )   :
    cdef double[:] result=np.empty((3,))
    cdef xyz pt=cevaluate_nurbs(t,cpts,knots,weights,degree)
    result[0] = pt.x
    result[1] = pt.y
    result[2] = pt.z

    return  result
def evaluate_nurbs_multi(double[:] t, double[:,:] cpts, double[:] knots, double[:] weights, int degree )   :
    cdef double[:,:] result=np.empty((t.shape[0],3))
    cevaluate_nurbs_multi(t,cpts,knots,weights,degree,result)
    return  result
