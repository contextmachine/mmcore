cimport cython
import numpy as np
from libc.math cimport sqrt
from libc.stdlib cimport malloc,free
cimport numpy as np
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double scalar_dot(double [:]  vec_a, double [:]  vec_b) nogil:
    cdef double res = 0.0
    cdef size_t j
    for j in range(vec_a.shape[0]):
        res += vec_a[j] * vec_b[j]
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double scalar_norm(double [:] vec) nogil:
    cdef double res = 0.0
    cdef double res2 = 0.0
    cdef size_t j
    for j in range(vec.shape[0]):
        res += (vec[j] ** 2)
    if res==0:
        return res2
    return sqrt(res)

cdef class CImplicit3D:
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double cimplicit(self, double x,double y, double z) noexcept nogil:
        cdef double res=0.
        return res
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cnormal(self, double x, double y, double z, double[:] result) noexcept nogil:
        cdef double n
        result[0] = (self.cimplicit(x+1e-3,y,z) - self.cimplicit(x-1e-3,y,z)) / 2 / 1e-3
        result[1] = (self.cimplicit(x,y+1e-3,z)- self.cimplicit(x,y-1e-3,z)) / 2 / 1e-3
        result[2] = (self.cimplicit(x,y,z+1e-3) - self.cimplicit(x,y,z-1e-3)) / 2 / 1e-3
        n=scalar_norm(result)
        if n>0:

            result[0] /=n
            result[1] /= n
            result[2] /= n
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def normal(self, double[:] point):
        cdef double x=point[0]
        cdef double y = point[1]
        cdef double z = point[2]
        cdef double[:] result=np.empty((3,))
        self.cnormal(x,y,z,result)
        return np.array(result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def implicit( self, double[:] point):
        cdef double x=point[0]
        cdef double y = point[1]
        cdef double z = point[2]
        cdef double result = self.cimplicit(x,y,z)
        return result


cdef class Sphere(CImplicit3D):
    cdef double[:] origin
    cdef double radius
    cdef double[:,:] _bounds
    def __init__(self, double[:] origin, double radius ):
        self.origin=origin
        self.radius=radius
        self._bounds=np.empty((2,3))
        self._bounds[0][0] = self.origin[0] - self.radius
        self._bounds[0][1] = self.origin[1] - self.radius
        self._bounds[0][2] = self.origin[2] - self.radius
        self._bounds[1][0] = self.origin[0] + self.radius
        self._bounds[1][1] = self.origin[1] + self.radius
        self._bounds[1][2] = self.origin[2] + self.radius
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double cimplicit(self, double x,double y, double z) noexcept nogil:
        cdef double result = sqrt((x-self.origin[0])**2+(y-self.origin[1])**2+(z-self.origin[2])**2)-self.radius
        return result
    def bounds(self):

        return self._bounds