cimport cython
import numpy as np
cimport numpy as np
cdef double DEFAULT_H=1e-3
from mmcore.numeric cimport vectors
from mmcore.numeric cimport calgorithms
from libc.math cimport fabs



cdef class ParametricCurve:

    def __init__(self):
        self._interval=np.zeros((2,))
        self._interval[1] = 1.0
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cevaluate(self, double t , double[:] result):
        pass
    cpdef double[:] interval(self):
        return self._interval
    def evaluate(self, t):
        cdef double[:] result =np.zeros((3,))
        self.cevaluate(t,result)
        return result
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cderivative(self, double t, double[:] result) :
        """
        :param t:float
        :return: vector of first derivative   as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.
        """
        print(np.asarray(result))
        cdef double[:] v1=np.zeros(result.shape)
        cdef double[:] v2=np.zeros(result.shape)
        cdef double t1,t2
        cdef start=DEFAULT_H+self._interval[0]
        cdef end = self._interval[1]-DEFAULT_H
        if end >= t >= start:
            t1=t+DEFAULT_H
            t2 = t - DEFAULT_H
            self.cevaluate(t1,v1)
            self.cevaluate(t2,v2)
            vectors.sub3d(v1,v2,result)
            vectors.scalar_div3d(result, 2*DEFAULT_H, result)


        elif t <= start:
            t1 = t + DEFAULT_H
            self.cevaluate(t1, v1)
            self.cevaluate(t, v2)
            vectors.sub3d(v1, v2, result)
            vectors.scalar_div3d(result,  DEFAULT_H, result)

        else:
            t2 = t - DEFAULT_H
            self.cevaluate(t, v1)
            self.cevaluate(t2, v2)
            vectors.sub3d(v1, v2, result)
            vectors.scalar_div3d(result, DEFAULT_H, result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void csecond_derivative(self, double t, double[:] result):
            """
            :param t:float
            :return: vector of first derivative   as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.
            """
            cdef double[:] v1=np.zeros(result.shape)
            cdef double[:] v2=np.zeros(result.shape)
            cdef double t1,t2

            t1=t+DEFAULT_H
            t2 = t - DEFAULT_H
            self.cderivative(t1,v1)
            self.cderivative(t2,v2)

            vectors.sub3d(v1,v2,result)
            vectors.scalar_div3d(result, 2*DEFAULT_H, result)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def derivative(self, t):
        cdef double[:] result =np.zeros((3,))
        self.cderivative(t,result)
        return np.asarray(result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def second_derivative(self, t):
        cdef double[:] result =np.zeros((3,))
        self.csecond_derivative(t,result)
        return np.asarray(result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void ccurvature(self, double t,double[:] result):
        cdef double[:,:] ders = np.zeros((3,3))
        self.cderivative(t, ders[0])
        self.csecond_derivative(t, ders[1])
        calgorithms.evaluate_curvature(ders[0],ders[1],ders[2],result)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void ctangent(self, double t,double[:] result):
        cdef double[:,:] ders = np.zeros((2,3))
        self.cderivative(t, ders[0])
        self.csecond_derivative(t, ders[1])
        calgorithms.evaluate_tangent(ders[0],ders[1],result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cnormal(self, double t,double[:] result):
        cdef double[:,:] ders = np.zeros((2,3))
        self.cderivative(t, ders[0])
        self.csecond_derivative(t, ders[1])

        cdef double nrm = vectors.scalar_norm(ders[0])
        ders[0, 0] /= nrm
        ders[0, 1] /= nrm
        ders[0, 2] /= nrm

        vectors.scalar_gram_schmidt_emplace(ders[0], ders[1])
        nrm = vectors.scalar_norm(ders[1])

        ders[1, 0] /= nrm
        ders[1, 1] /= nrm
        ders[1, 2] /= nrm

        result[0] = (ders[0][1] * ders[1][2]) - (ders[0][2] * ders[1][1])
        result[1] = (ders[0][2] * ders[1][0]) - (ders[0][0] * ders[1][2])
        result[2] = (ders[0][0] * ders[1][1]) - (ders[0][1] * ders[1][0])
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cplanes(self, double[:] t, double[:,:,:] result):
        cdef int i
        for i in range(t.shape[0]):
            self.cplane(t[i],result[i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cplane(self, double t, double[:,:] result):
        self.cevaluate(t, result[0])
        self.cderivative(t,  result[1])
        self.csecond_derivative(t, result[2])

        cdef double nrm = vectors.scalar_norm(result[1])
        result[1, 0] /= nrm
        result[1, 1] /= nrm
        result[1, 2] /= nrm

        vectors.scalar_gram_schmidt_emplace(result[1], result[2])
        nrm = vectors.scalar_norm(result[2])

        result[2, 0] /= nrm
        result[2, 1] /= nrm
        result[2, 2] /= nrm

        result[3, 0] = (result[1][1] * result[2][2]) - (result[1][2] * result[2][1])
        result[3, 1] = (result[1][2] * result[2][0]) - (result[1][0] * result[2][2])
        result[3, 2] = (result[1][0] * result[2][1]) - (result[1][1] * result[2][0])

    cpdef double[:] normal(self,double t):
        cdef double[:]result=np.zeros(3)
        self.cnormal(t,result)
        return result
    cpdef double[:] tangent(self,double t):
        cdef double[:]result=np.zeros(3)
        self.ctangent(t,result)
        return result
    cpdef double[:] curvature(self,double t):
        cdef double[:]result=np.zeros(3)
        self.ccurvature(t,result)
        return result
    def plane_at(self,double t):
        cdef double[:,:]result=np.zeros((4,3))
        self.cplane(t,result)
        return result
    def planes_at(self,double[:] t):
        cdef double[:,:,:]result=np.zeros((t.shape[0], 4,3))
        self.cplanes(t,result)
        return result