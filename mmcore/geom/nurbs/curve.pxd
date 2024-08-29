#cython: language_level=3




cimport cython

from mmcore.geom.parametric cimport ParametricCurve


cdef class NURBSCurve(ParametricCurve):
    cdef public double[:,:] _control_points
    cdef public int _degree
    cdef double[:] _knots
    cdef bint _periodic
    cdef public object _evaluate_cached
    cdef double[:] _greville_abscissae


    cpdef void set_degree(self, int val)

    cpdef int get_degree(self)
    cpdef bint is_periodic(self)

    cdef void generate_knots(self)
     
    cpdef knots_update_hook(self)
       

    cdef void generate_knots_periodic(self)
    cpdef void make_periodic(self)

    cdef _update_interval(self)

    cpdef double[:,:] generate_control_points_periodic(self, double[:,:] cpts)



    cpdef void make_open(self)
    


    cdef void ctangent(self, double t,double[:] result)
     


    cdef void ccurvature(self, double t,double[:] result)
  


    cdef void cevaluate(self, double t, double[:] result) noexcept nogil

    cpdef evaluate4d(self, double t)


    cpdef set(self, double[:,:] control_points, double[:] knots )


    cdef void cevaluate_ptr(self, double t, double *result ) noexcept nogil



    cdef void cderivative(self, double t, double[:] result)
   

    cdef void csecond_derivative(self, double t, double[:] result)   

    cdef void cderivatives1(self, double t, int d, double[:,:] CK ) 
    cdef void cderivatives2(self, double t, int d, double[:,:] CK )
    cdef void cplane(self, double t, double[:,:] result)
    cdef void cnormal(self, double t, double[:] result)
    cpdef void insert_knot(self, double param, int num)

    cdef NURBSCurve ccopy(self)
    cdef bytes cserialize(self)
    @staticmethod
    cdef NURBSCurve cdeserialize(const unsigned char[:] data)

cpdef double[:] greville_abscissae(double[:] knots, int degree)
cpdef tuple split_curve(NURBSCurve obj, double param) 