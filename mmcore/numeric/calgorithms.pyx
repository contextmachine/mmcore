cimport cython


from libc.math cimport fabs,sqrt
from mmcore.numeric.vectors cimport scalar_dot,scalar_norm,solve2x2
from mmcore.geom.primitives cimport Implicit3D


import numpy as np
cimport numpy as np
@cython.boundscheck(False)
@cython.wraparound(False)
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
cpdef bint evaluate_curvature(
    double[:] derivative, double[:] second_derivative, double[:] unit_tangent_vector, double[:] curvature_vector,
)noexcept nogil:
    """
    Calculates the unit tangent vector, curvature vector, and a recalculate condition for a given derivative and
    second derivative.

    :param derivative: The derivative vector.
    :param second_derivative: The second derivative vector.
    :return: A tuple containing the unit tangent vector, curvature vector, and recalculate condition.

    Example usage:
        derivative = np.array([1, 0, 0])
        second_derivative = np.array([0, 1, 0])
        evaluate_curvature2(derivative, second_derivative)
    """
    # Norm of derivative
    cdef double norm_derivative = sqrt(derivative[0]*derivative[0]+derivative[1]*derivative[1]+derivative[2]*derivative[2])
    cdef double zero_tolerance = 0.0
    cdef double negative_second_derivative_dot_tangent,inverse_norm_derivative_squared

    cdef bint recalculate_condition

    # Check if norm of derivative is too small
    if norm_derivative == zero_tolerance:
        norm_derivative = sqrt(second_derivative[0]*second_derivative[0]+second_derivative[1]*second_derivative[1]+second_derivative[2]*second_derivative[2])

        # If norm of second_derivative is above tolerance, calculate the unit tangent
        # If not, set unit tangent as zeros_like second_derivative
        if norm_derivative > zero_tolerance:
            unit_tangent_vector[0] = second_derivative[0] / norm_derivative
            unit_tangent_vector[1] = second_derivative[1] / norm_derivative
            unit_tangent_vector[2] = second_derivative[2] / norm_derivative
        else:
            unit_tangent_vector[0]=0.
            unit_tangent_vector[1] = 0.
            unit_tangent_vector[2] = 0.


        # Set curvature vector to zero, we will not recalculate
        curvature_vector[0]=0.
        curvature_vector[1] = 0.
        curvature_vector[2] = 0.
        recalculate_condition = False
    else:
        unit_tangent_vector[0] = derivative[0] / norm_derivative
        unit_tangent_vector[1] = derivative[1] / norm_derivative
        unit_tangent_vector[2] = derivative[2] / norm_derivative


        # Compute scalar component of curvature
        negative_second_derivative_dot_tangent = - (second_derivative[0]* unit_tangent_vector[0]+second_derivative[1]* unit_tangent_vector[1]+second_derivative[2]* unit_tangent_vector[2])
        inverse_norm_derivative_squared = 1.0 / (norm_derivative * norm_derivative)

        # Calculate curvature vector
        curvature_vector[0] = inverse_norm_derivative_squared * (
            second_derivative[0]
            + negative_second_derivative_dot_tangent * unit_tangent_vector[0]
        )
        curvature_vector[1] = inverse_norm_derivative_squared * (
                second_derivative[1]
                + negative_second_derivative_dot_tangent * unit_tangent_vector[1]
        )
        curvature_vector[2] = inverse_norm_derivative_squared * (
                second_derivative[2]
                + negative_second_derivative_dot_tangent * unit_tangent_vector[2]
        )

        # We will recalculate
        recalculate_condition = True

    return recalculate_condition



@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint cintersection_curve_point(Implicit3D surf1, Implicit3D surf2, double[:] q0,  double[:] f, double[:] g1, double[:] g2, double[:] result, double tol, size_t max_iter ) :
    cdef double[:] qk=np.zeros((3,), dtype=np.double)
    cdef double[:] qk_next=np.zeros((3,), dtype=np.double)
    cdef double[:] alpha_beta=np.zeros((2,), dtype=np.double)
    cdef double[:] delta=np.zeros((3,), dtype=np.double)


    cdef double[:] g=np.zeros((2,), dtype=np.double)
    cdef double[:,:] J=np.zeros((2,2), dtype=np.double)
    cdef size_t i=0
    cdef bint success = 1
    cdef bint _success = 0
    qk[0]=q0[0]
    qk[1]=q0[1]
    qk[2]=q0[2]
    f[0]=surf1.cimplicit(qk[0], qk[1], qk[2],)
    f[1]=surf2.cimplicit(qk[0], qk[1], qk[2],)
    surf1.cgradient(qk[0], qk[1], qk[2], g1)
    surf2.cgradient(qk[0], qk[1], qk[2], g2)


    J[0][0] = scalar_dot(g1, g1)
    J[0][1]=scalar_dot(g2, g1)
    J[1][0]=scalar_dot(g1, g2)
    J[1][1]=scalar_dot(g2, g2)

    g[0]= -f[0]
    g[1]= -f[1]
    success = solve2x2(J, g, alpha_beta)
    surf1.cgradient(qk[0], qk[1], qk[2], g1)
    surf2.cgradient(qk[0], qk[1], qk[2], g2)

    delta[0] = alpha_beta[0] * g1[0] + alpha_beta[1] *g2[0]
    delta[1] = alpha_beta[0] * g1[1] + alpha_beta[1] * g2[1]
    delta[2] = alpha_beta[0] * g1[2] + alpha_beta[1] * g2[2]
    qk_next[0] = delta[0] + qk[0]
    qk_next[1] = delta[1] + qk[1]
    qk_next[2] = delta[2] + qk[2]
    d = scalar_norm(delta)


    while d > tol:

        if i > max_iter:


            success =0
            return success

        qk[0] = qk_next[0]
        qk[1] = qk_next[1]
        qk[2] = qk_next[2]

        f[0] = surf1.cimplicit(qk[0], qk[1], qk[2],)
        f[1] = surf2.cimplicit(qk[0], qk[1], qk[2],)
        surf1.cgradient(qk[0],qk[1],qk[2], g1)
        surf2.cgradient(qk[0],qk[1],qk[2], g2)

        J[0][0] = scalar_dot(g1, g1)
        J[0][1] = scalar_dot(g2, g1)
        J[1][0] = scalar_dot(g1, g2)
        J[1][1] = scalar_dot(g2, g2)

        g[0]= -f[0]
        g[1]= -f[1]
        _success = solve2x2(J, g, alpha_beta)

        #alpha, beta = newton_step(qk, alpha, beta, f1, f2, g1, g2)
        #alpha, beta = newton_step(qk, alpha, beta, f1, f2, g1, g2)
        delta[0] = alpha_beta[0] * g1[0] + alpha_beta[1] * g2[0]
        delta[1] = alpha_beta[0] * g1[1] + alpha_beta[1] * g2[1]
        delta[2] = alpha_beta[0] * g1[2] + alpha_beta[1] * g2[2]
        qk_next[0] = delta[0] + qk[0]
        qk_next[1] = delta[1] + qk[1]
        qk_next[2] = delta[2] + qk[2]

        d = scalar_norm(delta)

        i += 1
    result[0] = qk_next[0]
    result[1] = qk_next[1]
    result[2] = qk_next[2]
    return success

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint intersection_curve_point(Implicit3D surf1, Implicit3D surf2,  double[:] q0, double[:] result, double tol,size_t max_iter) :
    cdef double[:] g1=np.zeros((3,), dtype=np.double)
    cdef double[:] g2=np.zeros((3,), dtype=np.double)
    cdef double[:] f=np.zeros((2,), dtype=np.double)
    cdef bint success=cintersection_curve_point(surf1,surf2,q0,f,g1,g2,result,tol,max_iter)

    return success

