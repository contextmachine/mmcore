# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.math cimport fabs,sqrt

cimport numpy as cnp
cnp.import_array()

from mmcore.geom.primitives cimport Implicit3D
cimport mmcore.numeric.calgorithms
cimport cython
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef void evaluate_tangent(double[:] D1, double[:] D2, double[:] result) noexcept nogil:
    """
    D1 - first derivative vector
    D2 - second derivative vector

    :math:`\\dfrac{D2}{||D1||}}  \\cos(\\omega x)f(x)dx` or
    :math:`\\int^b_a \\sin(\\omega x)f(x)dx`
    :param D1:
    :param D2:
    :return:

    """

    cdef double d1 =sqrt(D1[0]*D1[0]+D1[1]*D1[1]+D1[2]*D1[2])

    if fabs(d1)<=1e-15:
        d1 = sqrt(D2[0]*D2[0]+D2[1]*D2[1]+D2[2]*D2[2])
        if d1 > 0.0:

            result[0] = D2[0] / d1
            result[1] = D2[1] / d1
            result[2] = D2[2] / d1
        else:
            result[0]= 0.0
            result[1] = 0.0
            result[2] = 0.0

    else:
        result[0] = D1[0] / d1
        result[1] = D1[1] / d1
        result[2] = D1[2] / d1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef surface_jacobian(double[:] du, double[:] dv):
    cdef double[:,:] J=np.empty((2,2))
    J[0, 0] = scalar_dot(du, du)
    J[0, 1] = scalar_dot(du, dv)
    J[1, 0] = scalar_dot(dv, du)
    J[1, 1] = scalar_dot(dv, dv)
    return J


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef bint intersection_curve_point(Implicit3D surf1, Implicit3D surf2,  double[:] q0, double[:] result, double tol,size_t max_iter) :
    cdef double[:] g1=np.zeros((3,), dtype=np.double)
    cdef double[:] g2=np.zeros((3,), dtype=np.double)
    cdef double[:] f=np.zeros((2,), dtype=np.double)
    cdef bint success=cintersection_curve_point(surf1,surf2,q0,f,g1,g2,result,tol,max_iter)

    return success

@cython.boundscheck(False)  # Turn off bounds-checking for array indexing
@cython.wraparound(False)   # Turn off negative indexing
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef bint evaluate_curvature(
    double[:] derivative,
    double[:] second_derivative, 
    double[:] unit_tangent_vector, 
    double[:] curvature_vector) noexcept nogil:
    """
    Calculates the unit tangent vector, curvature vector, and a recalculate condition for a given derivative and
    second derivative.
    """
    cdef int n = derivative.shape[0]
    cdef double norm_derivative, negative_second_derivative_dot_tangent, inverse_norm_derivative_squared
    cdef bint recalculate_condition
    cdef double zero_tolerance = 0.0
    cdef int i
    # Norm of derivative
    norm_derivative = 0.0
    for i in range(n):
        norm_derivative += derivative[i] * derivative[i]
    norm_derivative = norm_derivative ** 0.5

    # Check if norm of derivative is too small
    if norm_derivative == zero_tolerance:
        norm_derivative = 0.0
        for i in range(n):
            norm_derivative += second_derivative[i] * second_derivative[i]
        norm_derivative = norm_derivative ** 0.5

        if norm_derivative > zero_tolerance:
            for i in range(n):
                unit_tangent_vector[i] = second_derivative[i] / norm_derivative
        else:
            for i in range(n):
                unit_tangent_vector[i] = 0.0

        # Set curvature vector to zero, we will not recalculate
        recalculate_condition = False
    else:
        for i in range(n):
            unit_tangent_vector[i] = derivative[i] / norm_derivative

        # Compute scalar component of curvature
        negative_second_derivative_dot_tangent = 0.0
        for i in range(n):
            negative_second_derivative_dot_tangent -= second_derivative[i] * unit_tangent_vector[i]

        inverse_norm_derivative_squared = 1.0 / (norm_derivative * norm_derivative)

        # Calculate curvature vector
        for i in range(n):
            curvature_vector[i] = inverse_norm_derivative_squared * (
                second_derivative[i] + negative_second_derivative_dot_tangent * unit_tangent_vector[i]
            )

        # We will recalculate
        recalculate_condition = True

    return recalculate_condition