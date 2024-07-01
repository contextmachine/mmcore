
ctypedef struct xyz:
    double x,y,z;

cdef double cdeboor(double[:] knots, double t, int i, int k) noexcept nogil


cdef void cevaluate_nurbs(double t, double[:,:] cpts, double[:] knots, double[:] weights, int degree , xyz* result )


cdef void cevaluate_nurbs_multi( double[:] t,double[:,:] cpts, double[:] knots, double[:] weights, int degree, double[:,:] result)



