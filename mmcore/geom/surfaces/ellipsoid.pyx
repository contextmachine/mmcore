
from libc.math cimport sin, cos,pi

cpdef void ellipsoid(double u,double  v, double a, double b, double c, double ox,double oy, double oz, double[:] output) :
    cdef double sin_u=sin(u)
    #if output is None:
    #    # Creating a default view, e.g.
    #
    #    cyarr = cvarray(shape=( 3,), itemsize=sizeof(double))
    #    output = cyarr
    #
    output[0]=   ox+a *  sin_u * cos(v)
    output[1] =  oy+b * sin_u* sin(v)
    output[2] =  oz+c * cos(u)


