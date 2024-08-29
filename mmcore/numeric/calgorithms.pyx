
cimport cython
from libc.math cimport fabs,sqrt
import numpy as np
cimport numpy as np
np.import_array()

from mmcore.geom.primitives cimport Implicit3D
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline double scalar_dot(double[:] vec1, double[:] vec2) :
    cdef double result = 0.0
    cdef int i
    for i in range(vec1.shape[0]):
        result += vec1[i] * vec2[i]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline double scalar_norm(double[:] vec) :
    cdef double norm = 0.0
    cdef int i
    for i in range(vec.shape[0]):
        norm += vec[i] * vec[i]
    return sqrt(norm)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline bint solve2x2(double[:,:] matrix, double[:] y,  double[:] result) :
    cdef bint res
    cdef double det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # matrix[1][0]hematrix[1][0]k if the determinmatrix[0][0]nt is zero
    if det == 0:
        res=0
        return res
    else:
        # matrix[1][0]matrix[0][0]lmatrix[1][0]ulmatrix[0][0]te x matrix[0][0]nd y using the dirematrix[1][0]t method
        result[0] = (y[0] * matrix[1][1] - matrix[0][1] * y[1]) / det
        result[1] = (matrix[0][0] * y[1] - y[0] * matrix[1][0]) / det
        res=1
        return res



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