
cimport cython
import numpy as np
cimport numpy as cnp

from mmcore.numeric.interval import _interval

cnp.import_array()
from mmcore.numeric.vectors cimport sub3d

cdef double DEFAULT_H=1e-3
from mmcore.numeric cimport vectors
from mmcore.numeric cimport calgorithms
from mmcore.geom.evaluator cimport surface_evaluator
from libc.math cimport fabs
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef  inline void cross_inl(double[:] vec_a,double[:] vec_b,double[:] res ) noexcept nogil:

    res[ 0] = (vec_a[ 1] * vec_b[ 2]) - (vec_a[ 2] * vec_b[ 1])
    res[ 1] = (vec_a[ 2] * vec_b[ 0]) - (vec_a[ 0] * vec_b[ 2])
    res[ 2] = (vec_a[ 0] * vec_b[ 1]) - (vec_a[ 1] * vec_b[ 0])


cdef class ParametricCurve:

    def __init__(self):
        self._interval=np.zeros((2,))
        self._interval[1] = 1.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void cevaluate(self, double t , double[:] result) noexcept nogil:
        pass

    def interval(self):
        cdef tuple interv=(self._interval[0],self._interval[1])
        return interv

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def evaluate(self, t):
        cdef double[:] result =np.zeros((3,))
        self.cevaluate(t,result)
        return result
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void cderivative(self, double t, double[:] result) :
        """
        :param t:float
        :return: vector of first derivative   as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.
        """
        #print(np.asarray(result))
        cdef double[:] v1=np.zeros(result.shape)
        cdef double[:] v2=np.zeros(result.shape)
        cdef double t1,t2
        cdef start=DEFAULT_H+self._interval[0]
        cdef end = self._interval[1]-DEFAULT_H

        t1=t+DEFAULT_H
        t2 = t - DEFAULT_H
        self.cevaluate(t1,v1)
        self.cevaluate(t2,v2)
        result[0]=(v1[0]- v2[0])*(0.5*DEFAULT_H)
        result[1] = (v1[1] - v2[1]) * (0.5 * DEFAULT_H)
        result[2] = (v1[2] - v2[2]) * (0.5 * DEFAULT_H)


    cpdef double[:] end(self):
        cdef double[:] pt=np.empty((3,))

        self.cevaluate(self._interval[1], pt)
        return pt
    cpdef double[:] start(self):
        cdef double[:] pt=np.empty((3,))

        self.cevaluate(self._interval[0], pt)
        return pt

    cpdef bint is_closed(self):
        cdef double[:] end=self.end()
        cdef double[:] start=self.start()
        cdef bint res=fabs(end[0]-start[0])<=1e-7 and fabs(end[1]-start[1])<=1e-7 and fabs(end[2]-start[2])<=1e-7
        return res
    cpdef bint is_open(self):
        cdef bint res= not self.is_closed()
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
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
            result[0] = (v1[0] - v2[0]) * (0.5 * DEFAULT_H)
            result[1] = (v1[1] - v2[1]) * (0.5 * DEFAULT_H)
            result[2] = (v1[2] - v2[2]) * (0.5 * DEFAULT_H)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def derivative(self, t):
        cdef double[:] result =np.zeros((3,))
        self.cderivative(t,result)
        return np.asarray(result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def second_derivative(self, t):
        cdef double[:] result =np.zeros((3,))
        self.csecond_derivative(t,result)
        return np.asarray(result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void ccurvature(self, double t,double[:] result):
        cdef double[:,:] ders = np.zeros((3,3))
        self.cderivative(t, ders[0])
        self.csecond_derivative(t, ders[1])
        calgorithms.evaluate_curvature(ders[0],ders[1],ders[2],result)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void ctangent(self, double t,double[:] result):
        cdef double[:,:] ders = np.zeros((2,3))
        self.cderivative(t, ders[0])
        self.csecond_derivative(t, ders[1])
        calgorithms.evaluate_tangent(ders[0],ders[1],result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
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
        result[0] =   ders[1, 0]
        result[1] =   ders[1, 1]
        result[2] =   ders[1, 2]

        #
        #result[0] = (ders[0][1] * ders[1][2]) - (ders[0][2] * ders[1][1])
        #result[1] = (ders[0][2] * ders[1][0]) - (ders[0][0] * ders[1][2])
        #result[2] = (ders[0][0] * ders[1][1]) - (ders[0][1] * ders[1][0])
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void cplanes(self, double[:] t, double[:,:,:] result):
        cdef int i
        for i in range(t.shape[0]):
            self.cplane(t[i],result[i])



    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef double[:] normal(self,double t):
        cdef double[:]result=np.zeros(3)
        self.cnormal(t,result)
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef double[:] tangent(self,double t):
        cdef double[:]result=np.zeros(3)
        self.ctangent(t,result)
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef double[:] curvature(self,double t):
        cdef double[:]result=np.zeros(3)
        self.ccurvature(t,result)
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def plane_at(self,double t):
        cdef double[:,:]result=np.zeros((4,3))
        self.cplane(t,result)
        return result
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def planes_at(self,double[:] t):
        cdef double[:,:,:]result=np.zeros((t.shape[0], 4,3))
        self.cplanes(t,result)
        return result


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def points(self, count=50):
        """
        :param count: The number of points to generate (default is 50)
        :return: A numpy array of shape (count, 3) containing the evaluated points
        """
        cdef int cnt=count
        cdef double[:,:] result=np.empty((cnt,3))

        cdef double[:] t=   np.linspace(  self._interval[0],   self._interval[1], cnt)
        cdef int i


        for i in range(cnt):
            self.cevaluate(t[i],result[i])

        return np.asarray(result)

    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __call__(self, t, cnp.ndarray[double, ndim=2] res=None) :
        cdef int l = 1 if isinstance(t,(float,int)) else len(t)
        cdef int i

        if res is None:  
            res=np.zeros((l,3))
        if l>1:
            for i in range(l):
                self.cevaluate(t[i],res[i])
            return res
        else:
            self.cevaluate(t,res[0])
            return res
        

cdef class ReparametrizedCurve(ParametricCurve):
    cdef public ParametricCurve curve
    def __init__(self, ParametricCurve curve, tuple new_interval):
        super().__init__()
        self._interval[0]= new_interval[0]
        self._interval[1] = new_interval[1]
        self.curve=curve


cdef class ParametricSurface:

    def __init__(self):

        self._interval=np.zeros((2,2))
        self._interval[0][1]=1.
        self._interval[1][1] = 1.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void cderivative_u(self, double u, double v,double[:] result):
        cdef double[:] dd=np.empty((3,))
        if (1. - DEFAULT_H) >= u >= DEFAULT_H:
            self.cevaluate(u + DEFAULT_H, v, dd)
            self.cevaluate(u- DEFAULT_H , v, result)
            sub3d(dd,result,result)
            vectors.scalar_div3d(result,2*DEFAULT_H,result)

        elif u < DEFAULT_H:


            self.cevaluate(u + DEFAULT_H, v, dd)
            self.cevaluate(u , v, result)
            sub3d(dd,result,result)
            vectors.scalar_div3d(result,DEFAULT_H,result)


        else:
            self.cevaluate(u , v, dd)
            self.cevaluate(u- DEFAULT_H, v, result)
            sub3d(dd, result, result)
            vectors.scalar_div3d(result, DEFAULT_H, result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void cderivative_v(self, double u, double v,double[:] result):
        cdef double[:] dd=np.empty((3,))
        if (1. - DEFAULT_H) >= v >= DEFAULT_H:
            self.cevaluate(u , v+ DEFAULT_H, dd)
            self.cevaluate(u , v- DEFAULT_H, result)
            sub3d(dd,result,result)
            vectors.scalar_div3d(result,2*DEFAULT_H,result)

        elif v< DEFAULT_H:


            self.cevaluate(u , v+ DEFAULT_H, dd)
            self.cevaluate(u , v, result)
            sub3d(dd,result,result)
            vectors.scalar_div3d(result,DEFAULT_H,result)


        else:
            self.cevaluate(u , v, dd)
            self.cevaluate(u, v- DEFAULT_H, result)
            sub3d(dd, result, result)
            vectors.scalar_div3d(result, DEFAULT_H, result)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void csecond_derivative_uu(self, double u, double v,double[:] result):
        cdef double[:] dd=np.empty((3,))
        if (1. - DEFAULT_H) >= u >= DEFAULT_H:
            self.cderivative_u(u + DEFAULT_H, v, dd)
            self.cderivative_u(u- DEFAULT_H , v, result)
            sub3d(dd,result,result)
            vectors.scalar_div3d(result,2*DEFAULT_H,result)

        elif u < DEFAULT_H:


            self.cderivative_u(u + DEFAULT_H, v, dd)
            self.cderivative_u(u , v, result)
            sub3d(dd,result,result)
            vectors.scalar_div3d(result,DEFAULT_H,result)


        else:
            self.cderivative_u(u , v, dd)
            self.cderivative_u(u- DEFAULT_H, v, result)
            sub3d(dd, result, result)
            vectors.scalar_div3d(result, DEFAULT_H, result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void csecond_derivative_vv(self, double u, double v,double[:] result):
        cdef double[:] dd=np.empty((3,))
        if (1. - DEFAULT_H) >= v >= DEFAULT_H:
            self.cderivative_v(u , v+ DEFAULT_H, dd)
            self.cderivative_v(u , v- DEFAULT_H, result)
            sub3d(dd,result,result)
            vectors.scalar_div3d(result,2*DEFAULT_H,result)

        elif v< DEFAULT_H:


            self.cderivative_v(u , v+ DEFAULT_H, dd)
            self.cderivative_v(u , v, result)
            sub3d(dd,result,result)
            vectors.scalar_div3d(result,DEFAULT_H,result)


        else:
            self.cderivative_v(u , v, dd)
            self.cderivative_v(u, v- DEFAULT_H, result)
            sub3d(dd, result, result)
            vectors.scalar_div3d(result, DEFAULT_H, result)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void csecond_derivative_uv(self, double u, double v,double[:] result):
        cdef double[:] dd=np.empty((3,))
        if (1. - DEFAULT_H) >= v >= DEFAULT_H:
            self.cderivative_u(u , v+ DEFAULT_H, dd)
            self.cderivative_u(u , v- DEFAULT_H, result)
            sub3d(dd,result,result)
            vectors.scalar_div3d(result,2*DEFAULT_H,result)

        elif v< DEFAULT_H:


            self.cderivative_u(u , v+ DEFAULT_H, dd)
            self.cderivative_u(u , v, result)
            sub3d(dd,result,result)
            vectors.scalar_div3d(result,DEFAULT_H,result)


        else:
            self.cderivative_u(u , v, dd)
            self.cderivative_u(u, v- DEFAULT_H, result)
            sub3d(dd, result, result)
            vectors.scalar_div3d(result, DEFAULT_H, result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cevaluate(self, double u, double v,double[:] result) noexcept nogil:
        pass
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void cplane_at(self, double u, double v, double[:,:] result):

        orig = self.cevaluate(u, v,result[0] )
        self.cderivative_u(u,v,result[1])
        result[1,:] = vectors.scalar_unit(result[1])
        self.cderivative_v(u,v,result[3])
        result[3,:] = vectors.scalar_unit(vectors.scalar_cross(result[1], result[3] ))
        result[2,:] = vectors.scalar_cross(result[3], result[1])
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def evaluate(self, double[:] uv):
        cdef double[:] res=np.empty(3)
        self.cevaluate(uv[0],uv[1],res)
        return np.asarray(res)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef evaluate_v2(self, double u, double v):
        cdef double[:] result=np.zeros((3,))
        self.cevaluate(u, v, result)
        return np.array(result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def plane_at(self, double[:] uv):
        cdef double[:,:] res=np.empty((4,3))
        self.cplane_at(uv[0],uv[1],res)
        return np.asarray(res)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def derivative_u(self, double[:] uv):
        cdef double[:] res=np.empty((3,))
        self.cderivative_u(uv[0],uv[1],res)

        return np.asarray(res)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def derivative_v(self, double[:] uv):
        cdef double[:] res=np.empty((3,))
        self.cderivative_v(uv[0],uv[1],res)

        return np.asarray(res)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def second_derivative_vv(self, double[:] uv):
        cdef double[:] res=np.empty((3,))
        self.csecond_derivative_vv(uv[0],uv[1],res)

        return np.asarray(res)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def second_derivative_uu(self, double[:] uv):
        cdef double[:] res=np.empty((3,))
        self.csecond_derivative_uu(uv[0],uv[1],res)

        return np.asarray(res)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def second_derivative_uv(self, double[:] uv):
        cdef double[:] res=np.empty((3,))
        self.csecond_derivative_uv(uv[0],uv[1],res)

        return np.asarray(res)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def evaluate_multi(self, double[:,:] uv):
        cdef int i
        cdef double[:,:] res=np.empty((uv.shape[0],3))
        for i in range(uv.shape[0]):
            self.cevaluate(uv[i,0],uv[i,1],res[i])
        return np.asarray(res)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def derivatives(self, double[:] uv, double[:,:] result=None):

        cdef double[:] a
        cdef double[:] b
        cdef double h2= 2 *DEFAULT_H
        cdef double u1,v1,u2,v2
        cdef u=uv[0]
        cdef v=uv[1]
        if result is None:
            result=np.empty((2,3))
        u1=u-DEFAULT_H
        u2=u+DEFAULT_H
        v1=v-DEFAULT_H
        v2=v+DEFAULT_H
        #result[0] =callback(u, v)
        a=self.evaluate_v2(u1,v)
        b=self.evaluate_v2(u2,v)

        result[0, 0]= (b[0] - a[0]) /h2
        result[0, 1]= (b[1] - a[1]) / h2
        result[0, 2] = (b[2] - a[2]) / h2
        a = self.evaluate_v2(u, v1)
        b = self.evaluate_v2(u, v2)

        result[1, 0] = (b[0] - a[0]) / h2
        result[1, 1] = (b[1] - a[1]) / h2
        result[1, 2] = (b[2] - a[2]) /h2
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def interval(self):
        cdef tuple inter=((self._interval[0][0],self._interval[0][1]),(self._interval[1][0],self._interval[1][1]))
        return inter

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef  cderivatives(self,double u, double v, double[:,:] result ):
        cdef double[:] a=np.zeros(3)
        cdef double[:] b=np.zeros(3)
        cdef double h2= 2 *DEFAULT_H
        cdef double u1,v1,u2,v2
        u1=u-DEFAULT_H
        u2=u+DEFAULT_H
        v1=v-DEFAULT_H
        v2=v+DEFAULT_H
        #result[0] =callback(u, v)
        self.cevaluate(u1,v,a)
        self.cevaluate(u2,v,b)


        result[0, 0]= (b[0] - a[0]) /h2
        result[0, 1]= (b[1] - a[1]) / h2
        result[0, 2] = (b[2] - a[2]) / h2
        self.cevaluate(u, v1, a)
        self.cevaluate(u, v2, b)

        result[1, 0] = (b[0] - a[0]) / h2
        result[1, 1] = (b[1] - a[1]) / h2
        result[1, 2] = (b[2] - a[2]) /h2

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef cnormal(self, double u, double v, double[:] result):
        cdef double[:,:] ders=np.zeros([2,3])
        self.cderivatives(u,v,ders)
        cross_inl(ders[0],ders[1],result)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def normal(self, double[:] uv, cnp.ndarray[double, ndim=1] result=None):

        if result is None:
            result=np.empty(3)
        self.cnormal(uv[0],uv[1],result)


        return result



cdef class Ruled(ParametricSurface):
    #cdef double _remap_u
    #cdef double _remap_v
    def __init__(self, ParametricCurve c0, ParametricCurve c1):

        super().__init__()
        self.c0=c0
        self.c1=c1
        #self._remap_u = vectors.scalar_dot(np.array((0., 1.)), self._interval[0])
        #self._remap_v = vectors.scalar_dot(np.array((0., 1.)), self._interval[1])
    def __reduce__(self):
        return (self.__class__, (self.c0, self.c1))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cevaluate(self, double u,double v, double[:] result) noexcept nogil:
        cdef double u1=(u*(self.c0._interval[1]-self.c0._interval[0]))+self.c0._interval[0]
        cdef double u2=(u*(self.c1._interval[1]-self.c1._interval[0]))+self.c1._interval[0]
        
        cdef double[:] temp
        with gil:
            temp=np.zeros((3,))
        self.c0.cevaluate(u1,  temp)
        self.c1.cevaluate(u2, result)

        result[0] = (1. - v) * temp[0] + v * result[0]
        result[1] = (1. - v) * temp[1] + v * result[1]
        result[2] = (1. - v) * temp[2] + v * result[2]
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void cderivative_v(self, double u, double v, double[:] result ):
        cdef double u1 = (u * (self.c0._interval[1] - self.c0._interval[0])) + self.c0._interval[0]
        cdef double u2 = (u * (self.c1._interval[1] - self.c1._interval[0])) + self.c1._interval[0]
        cdef double[:] temp=np.empty(3)
        self.c1.cevaluate(u2, temp)
        self.c0.cevaluate(u1, result)
        result[0] = temp[0] - result[0]
        result[1] = temp[1] - result[1]
        result[2] = temp[2] - result[2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void cderivative_u(self, double u, double v, double[:] result ):
        cdef double u1 = (u * (self.c0._interval[1] - self.c0._interval[0])) + self.c0._interval[0]
        cdef double u2 = (u * (self.c1._interval[1] - self.c1._interval[0])) + self.c1._interval[0]
        cdef double[:] temp=np.empty(3)
        cdef double dv=1.-v

        self.c1.cderivative(u2,temp)
        temp[0]*=v
        temp[1] *= v
        temp[2] *= v
        self.c0.cderivative(u1,result)
        result[0]*=dv
        result[1] *= dv
        result[2] *= dv
        result[0]+=temp[0]
        result[1] += temp[1]
        result[2] += temp[2]
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void csecond_derivative_uu(self, double u, double v, double[:] result ):
            cdef double u1 = (u * (self.c0._interval[1] - self.c0._interval[0])) + self.c0._interval[0]
            cdef double u2 = (u * (self.c1._interval[1] - self.c1._interval[0])) + self.c1._interval[0]
            cdef double[:] temp=np.empty(3)
            cdef double dv=1.-v

            self.c1.csecond_derivative(u2,temp)
            temp[0]*=v
            temp[1] *= v
            temp[2] *= v
            self.c0.csecond_derivative(u1,result)
            result[0]*=dv
            result[1] *= dv
            result[2] *= dv
            result[0]+=temp[0]
            result[1] += temp[1]
            result[2] += temp[2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void csecond_derivative_vv(self, double u, double v, double[:] result ):
        result[0] = 0.
        result[1] = 0.
        result[2] = 0.

    def second_derivative_vv(self, double[:] uv):
        return np.zeros(3)

cdef class RationalRuled(Ruled):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void cevaluate(self, double u,double v, double[:] result) noexcept nogil:
        cdef double u1=(u*(self.c0._interval[1]-self.c0._interval[0]))+self.c0._interval[0]
        cdef double u2=(u*(self.c1._interval[1]-self.c1._interval[0]))+self.c1._interval[0]
        cdef double[:] temp
        cdef double[:] res
        with gil:

            temp=np.empty(4)
            res=np.empty(4)

        self.c0.cevaluate(u1,  temp)
        self.c1.cevaluate(u2, res)
        result[0] = (1. - v) * temp[0] + v * res[0]
        result[1] = (1. - v) * temp[1] + v * res[1]
        result[2] = (1. - v) * temp[2] + v * res[2]
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void cderivative_v(self, double u, double v, double[:] result ):
        cdef double u1 = (u * (self.c0._interval[1] - self.c0._interval[0])) + self.c0._interval[0]
        cdef double u2 = (u * (self.c1._interval[1] - self.c1._interval[0])) + self.c1._interval[0]
        cdef double[:] temp=np.empty(4)
        cdef double[:] res=np.empty(4)
        self.c1.cevaluate(u2, temp)
        self.c0.cevaluate(u1, res)
        result[0] = temp[0] - res[0]
        result[1] = temp[1] - res[1]
        result[2] = temp[2] - res[2]


cdef class BiLinear(ParametricSurface):
    cdef double[:] b00
    cdef double[:] b10
    cdef double[:] b11
    cdef double[:] b01
    def __init__(self, a, b, c, d):
        super().__init__()
        self.b00, self.b10, self.b11, self.b01 = np.array([a, b, c, d], dtype=float)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void cevaluate(self, double u,double v, double[:] result) noexcept nogil:
        result[0]=v * (u * self.b11[0] + self.b01[0] * (1 - u)) + (1 - v) * (
                    u * self.b10[0] + self.b00[0] * (1 - u))
        result[1] = v * (u * self.b11[1] + self.b01[1] * (1 - u)) + (1 - v) * (
                u * self.b10[1] + self.b00[1] * (1 - u))
        result[2] = v * (u * self.b11[2] + self.b01[2] * (1 - u)) + (1 - v) * (
                u * self.b10[2] + self.b00[2] * (1 - u))
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void cderivative_u(self, double u,double v, double[:] result):
        
        result[0]= v * (-self.b01[0] + self.b11[0]) + (1 - v) * (-self.b00[0] + self.b10[0])
        result[1] = v * (-self.b01[1] + self.b11[1]) + (1 - v) * (-self.b00[1] + self.b10[1])

        result[2] = v * (-self.b01[2] + self.b11[2]) + (1 - v) * (-self.b00[2] + self.b10[2])
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void cderivative_v(self, double u,double v, double[:] result):
        result[0] = -u * self.b10[0] + u * self.b11[0] - self.b00[0] * (1 - u) + self.b01[0] * (1 - u)
        result[1] = -u * self.b10[1] + u * self.b11[1] - self.b00[1] * (1 - u) + self.b01[1] * (1 - u)
        result[2] = -u * self.b10[2] + u * self.b11[2] - self.b00[2] * (1 - u) + self.b01[2] * (1 - u)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void csecond_derivative_uv(self, double u,double v, double[:] result):

        result[0] =self.b00[0] - self.b01[0] - self.b10[0] + self.b11[0]    
        result[1] =self.b00[1] - self.b01[1] - self.b10[1] + self.b11[1]    
        result[2] =self.b00[2] - self.b01[2] - self.b10[2] + self.b11[2]    
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void csecond_derivative_vv(self, double u,double v, double[:] result):
        result[0] =0.0
        result[1] =0.0
        result[2] =0.0


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void csecond_derivative_uu(self, double u,double v, double[:] result):
        result[0] =0.0
        result[1] =0.0
        result[2] =0.0
    def __reduce__(self):
        return (self.__class__, (np.asarray(self.b00),np.asarray(self.b10),np.asarray(self.b11),np.asarray(self.b01)))