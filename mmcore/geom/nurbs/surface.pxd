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
    cdef void realloc_control_points(self, size_t new_size_u, size_t new_size_v) noexcept nogil
    cdef NURBSSurface ccopy(self)
    cpdef void insert_knot_u(self, double t, int r)
    cpdef void insert_knot_v(self, double t, int r)
    cdef void cnormalize_knots(self) noexcept nogil
    cdef void cnormalize_knots_u(self) noexcept nogil
    cdef void cnormalize_knots_v(self) noexcept nogil