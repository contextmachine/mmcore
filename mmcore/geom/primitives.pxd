cdef class Implicit3D:
    cdef double[:,:] _bounds
    cdef double cimplicit(self, double x, double y, double z)  noexcept nogil
    cdef void cnormal(self, double x, double y, double z, double[:] result)  noexcept nogil
cdef class Sphere(Implicit3D):
    cdef double ox
    cdef double oy
    cdef double oz
    cdef double _radius

    cdef void _calculate_bounds(self) noexcept nogil
cdef class Cylinder(Implicit3D):
    cdef double ox
    cdef double oy
    cdef double oz
    cdef double dx
    cdef double dy
    cdef double dz
    cdef double ex
    cdef double ey
    cdef double ez
    cdef double _radius



    cdef void _calculate_direction(self)  noexcept nogil

    cdef void _calculate_bounds(self)  noexcept  nogil
    cdef void _normal_not_unit(self,double x,double y, double z,double[:] res)  noexcept nogil
cdef class Tube(Cylinder):
    """
    Straight cylindrical pipe with adjustable thickness.

    """
    cdef double _thickness

