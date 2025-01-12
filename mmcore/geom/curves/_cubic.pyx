
cimport cython
import numpy as np
from mmcore.geom.parametric cimport ParametricCurve
from libc.math cimport pow

cdef class CubicSpline(ParametricCurve):
    cdef public int degree
    cdef public double[:] p0
    cdef public double[:] c0
    cdef public double[:] c1
    cdef public double[:] p1

    def __init__(self, double[:] p0,double[:] c0,double[:] c1,double[:] p1):
        super().__init__()
        self.degree=3
        self.p0=p0
        self.c0 = c0
        self.c1 = c1
        self.p1 = p1
    def __reduce__(self):
        return (self.__class__, (np.asarray(self.p0), np.asarray(self.c0),np.asarray(self.c1),np.asarray(self.p1)))
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cevaluate(self, double t, double[:] result) noexcept nogil:
        cdef double x0 = 1 - t;
        cdef double x1 = pow(x0, 2)
        cdef double x2= pow(t, 2)
        cdef double x3=pow(x0, 3)
        cdef double x4= pow(t, 3)
        result[0] = 3 * self.c0[0] * t * x1 + 3 * self.c1[0] * x2 * x0 + self.p0[0] *x3 + self.p1[0] * x4
        result[1] = 3 * self.c0[1] * t * x1 + 3 * self.c1[1] * x2 * x0 + self.p0[1] * x3 + self.p1[1] * x4
        result[2] = 3 * self.c0[2] * t * x1 + 3 * self.c1[2] * x2 * x0 + self.p0[2]  * x3 + self.p1[2]* x4

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cderivative(self, double t, double[:] result) :
        cdef double t0 =  (2 * t - 2)

        cdef double x0 = 3 * pow(t, 2);
        cdef double x1 = 1 - t;
        cdef double x2 = 3 * pow(x1, 2);
        result[0] = 3 * self.c0[0] * t * t0 + self.c0[0] * x2 + 6 * self.c1[0] * t * x1 - self.c1[0] * x0 - self.p0[0] * x2 + self.p1[0] * x0;
        result[1] = 3 * self.c0[1] * t * t0 + self.c0[1] * x2 + 6 * self.c1[1] * t * x1 - self.c1[1] * x0 - self.p0[1] * x2 + self.p1[1] * x0;
        result[2] = 3 * self.c0[2] * t * t0 + self.c0[2] * x2 + 6 * self.c1[2] * t * x1 - self.c1[2] * x0 - self.p0[2] * x2 + self.p1[2] * x0;

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void csecond_derivative(self, double t, double[:] result) :
        cdef double x0 = t - 1;
        result[0] = 6 * self.c0[0] * t + 12 * self.c0[0] * x0 - 12 * self.c1[0] * t - 6 * self.c1[0] * x0 - 6 * self.p0[0] * x0 + 6 * self.p1[0] * t
        result[1] = 6 * self.c0[1] * t + 12 * self.c0[1] * x0 - 12 * self.c1[1] * t - 6 * self.c1[1] * x0 - 6 * self.p0[1] * x0 + 6 * self.p1[1] * t
        result[2] = 6 * self.c0[2] * t + 12 * self.c0[2] * x0 - 12 * self.c1[2] * t - 6 * self.c1[2] * x0 - 6 * self.p0[2] * x0 + 6 * self.p1[2] * t



