
cimport numpy as cnp

cdef void c_derivatives(callback, double u, double v, double[:,:] result)
cpdef cnp.ndarray derivatives(callback, double u, double v)
cpdef cnp.ndarray derivatives_array( callback, double[:] u, double[:] v)
cdef void c_second_derivatives(callback, double u, double v, double[:,:] result)
cpdef cnp.ndarray second_derivatives(callback, double u, double v)
cpdef cnp.ndarray  second_derivatives_array( callback, double[:] u, double[:] v)