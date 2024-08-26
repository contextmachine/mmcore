cimport cython
from mmcore.geom.parametric cimport ParametricSurface

cdef class NURBSSurface(ParametricSurface):
    cdef double* _control_points_arr
    cdef double[:] _knots_u
    cdef double[:] _knots_v
    
    cdef int[2] _size
    cdef int[2] _degree

   
    cdef double[:,:,:] control_points_view
    cdef double[:,:] control_points_flat_view
 
    cpdef void _update_interval(self)
    cdef void generate_knots_u(self)
    cdef void generate_knots_v(self)