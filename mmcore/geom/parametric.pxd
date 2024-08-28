

cdef class ParametricCurve:
    cdef double[2] _interval


    cdef void cevaluate(self, double t , double[:] result)  noexcept nogil

    cpdef double[:] interval(self)

    cpdef double[:] start(self)
    cpdef double[:]  end(self)
    cpdef bint is_closed(self)
    cpdef bint is_open(self)
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

cdef class ParametricSurface:
    cdef double[:,:] _interval

    cdef void cderivative_u(self, double u, double v,double[:] result)
    cdef void cderivative_v(self, double u, double v,double[:] result)
    cdef void csecond_derivative_uu(self, double u, double v,double[:] result)
    cdef void csecond_derivative_vv(self, double u, double v,double[:] result)
    cdef void csecond_derivative_uv(self, double u, double v,double[:] result)
    cdef void cevaluate(self, double u, double v,double[:] result) noexcept nogil;
    cpdef evaluate_v2(self, double u, double v)
    cdef void cplane_at(self, double u, double v, double[:,:] result)

cdef class Ruled(ParametricSurface):
    cdef public ParametricCurve c0
    cdef public ParametricCurve c1
    