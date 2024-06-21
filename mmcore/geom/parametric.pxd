

cdef class ParametricCurve:
    cdef double[2] _interval


    cdef void cevaluate(self, double t , double[:] result)

    cpdef double[:] interval(self)


    cdef void cderivative(self, double t, double[:] result)

    cdef void csecond_derivative(self, double t, double[:] result)
    cdef void ccurvature(self, double t,double[:] result)
    cdef void cnormal(self, double t,double[:] result)
    cdef void ctangent(self, double t,double[:] result)
    cdef void cplane(self, double t,double[:,:] result)
    cdef void cplanes(self, double[:] t,double[:,:,:] result)
    cpdef double[:] normal(self,double t)

    cpdef double[:] tangent(self,double t)

    cpdef double[:] curvature(self,double t)