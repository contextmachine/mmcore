cimport cython

from libc.stdlib cimport malloc,free
from mmcore.numeric.vectors cimport scalar_dot,scalar_norm,solve2x2
from mmcore.geom.primitives cimport Implicit3D
import numpy as np
cimport numpy as np


ctypedef  struct IntersectionCurvePointFullOutput:
    bint success

    double f1
    double f2
    double d
    double result_x
    double result_y
    double result_z
    double[3] g1
    double[3] g2

    size_t iter_count



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
    surf1.cnormal(qk[0], qk[1], qk[2], g1)
    surf2.cnormal(qk[0], qk[1], qk[2], g2)


    J[0][0] = scalar_dot(g1, g1)
    J[0][1]=scalar_dot(g2, g1)
    J[1][0]=scalar_dot(g1, g2)
    J[1][1]=scalar_dot(g2, g2)

    g[0]= -f[0]
    g[1]= -f[1]
    success = solve2x2(J, g, alpha_beta)
    surf1.cnormal(qk[0], qk[1], qk[2], g1)
    surf2.cnormal(qk[0], qk[1], qk[2], g2)

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
        surf1.cnormal(qk[0],qk[1],qk[2], g1)
        surf2.cnormal(qk[0],qk[1],qk[2], g2)

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

