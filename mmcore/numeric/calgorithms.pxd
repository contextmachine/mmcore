

from mmcore.geom.primitives cimport Implicit3D


cpdef void evaluate_tangent(double[:] D1, double[:] D2, double[:] result) noexcept nogil



cpdef bint evaluate_curvature(
    double[:] derivative, double[:] second_derivative, double[:] unit_tangent_vector, double[:] curvature_vector) noexcept nogil


cdef inline bint cintersection_curve_point(Implicit3D surf1, Implicit3D surf2, double[:] q0,  double[:] f, double[:] g1, double[:] g2, double[:] result, double tol, size_t max_iter )



cpdef bint intersection_curve_point(Implicit3D surf1, Implicit3D surf2,  double[:] q0, double[:] result, double tol,size_t max_iter)


