cimport cython
import numpy as np
from libc.math cimport fabs, sqrt,fmin,fmax
from libc.stdlib cimport malloc,free
from cpython cimport PyTuple_New,PyTuple_Pack,PyTuple_GetItem,PyTuple_GET_ITEM
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double dot3d(double [:]  vec_a, double [:]  vec_b) noexcept nogil:
    cdef double res = vec_a[0] * vec_b[0]+ vec_a[1] * vec_b[1]+ vec_a[2] * vec_b[2]
   
    return res
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double norm3d(double [:] vec)noexcept nogil:
    cdef double res = sqrt(vec[0] ** 2+vec[1] ** 2+vec[2] ** 2)
    return res
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void min3d(double [:]  vec_a, double [:]  vec_b, double[:] res)noexcept nogil:
    
    res[0] = fmin(vec_a[0] ,vec_b[0])
    res[1] = fmin(vec_a[1] , vec_b[1])
    res[2] = fmin(vec_a[2] , vec_b[2])
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void max3d(double [:]  vec_a, double [:]  vec_b, double[:] res)noexcept nogil:
    
    res[0] = fmax(vec_a[0] ,vec_b[0])
    res[1] = fmax(vec_a[1] , vec_b[1])
    res[2] = fmax(vec_a[2] , vec_b[2]) 
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void bavg3d(double [:]  vec_a, double [:]  vec_b, double[:] res)noexcept nogil:
    
    res[0] = (vec_a[0] +vec_b[0])/2
    res[1] = (vec_a[1] +vec_b[1])/2
    res[2] = (vec_a[2] +vec_b[2])/2 
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sqrt3d(double [3]  vec_a,  double[3] res)noexcept nogil:
    
    res[0] = sqrt(vec_a[0])
    res[1] = sqrt(vec_a[1])
    res[2] = sqrt(vec_a[2])

    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mul3d(double[3]  vec_a, double[3]  vec_b, double[:] res)noexcept nogil:
  
    res[0] = vec_a[0] * vec_b[0]
    res[1] = vec_a[1] * vec_b[1]
    res[2] = vec_a[2] * vec_b[2]
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void add3d(double [:]  vec_a, double [:]  vec_b, double[:] res)noexcept nogil:
  
    res[0] = vec_a[0] + vec_b[0]
    res[1] = vec_a[1] + vec_b[1]
    res[2] = vec_a[2] + vec_b[2]
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sub3d(double [:]  vec_a, double [:]  vec_b, double[:] res)noexcept nogil:
  
    res[0] = vec_a[0] - vec_b[0]
    res[1] = vec_a[1] - vec_b[1]
    res[2] = vec_a[2] - vec_b[2]
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void scalar_sub3d(double [:]  vec_a, double  b, double[:] res)noexcept nogil:
  
    res[0] = vec_a[0] - b
    res[1] = vec_a[1] - b
    res[2] = vec_a[2] - b
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void scalar_add3d(double [:]  vec_a, double  b, double[:] res) noexcept nogil:
  
    res[0] = vec_a[0] + b
    res[1] = vec_a[1] + b
    res[2] = vec_a[2] + b
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void scalar_mul3d(double [:]  vec_a, double  b, double[:] res)noexcept nogil:
  
    res[0] = vec_a[0] * b
    res[1] = vec_a[1] * b
    res[2] = vec_a[2] * b
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void scalar_div3d(double [:]  vec_a, double  b, double[:] res) noexcept nogil:
  
    res[0] = vec_a[0] / b
    res[1] = vec_a[1] / b
    res[2] = vec_a[2] / b


cdef class Implicit3D:
    cdef double[:,:] _bounds
    def __init__(self):
        self._bounds=np.empty((2,3))
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
        n=norm3d(result)
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
    @property
    def bounds(self):
        return self._bounds

cdef class Sphere(Implicit3D):
    cdef double ox
    cdef double oy
    cdef double oz
    cdef double _radius

    def __init__(self, double ox,double oy,double oz, double radius ):
        super().__init__()
        self.ox=ox
        self.oy=oy
        self.oz=oz
        self._radius=radius
        self._calculate_bounds()
        
    


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _calculate_bounds(self) noexcept nogil:
        self._bounds[0][0] = self.ox - self._radius
        self._bounds[0][1] = self.oy - self._radius
        self._bounds[0][2] = self.oz - self._radius
        self._bounds[1][0] = self.ox + self._radius
        self._bounds[1][1] = self.oy + self._radius
        self._bounds[1][2] = self.oz + self._radius

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double cimplicit(self, double x,double y, double z) noexcept nogil:
        cdef double result = sqrt((x-self.ox)**2+(y-self.oy)**2+(z-self.oz)**2)-self._radius
        return result
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cnormal(self, double x,double y, double z, double[:] result) noexcept nogil:
        cdef double n
        result[0]=x-self.ox
        result[1]=y-self.oy
        result[2]=z-self.oz
        n=norm3d(result)
        result[0]/=n
        result[1] /= n
        result[2] /= n

    @property
    def bounds(self):
        return self._bounds
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void set(self, double ox, double oy, double oz, double radius ):
        self.ox=ox
        self.oy=oy
        self.oz=oz
        self._radius=radius
        self._calculate_bounds()

  
    
    def astuple(self):
        cdef tuple tpl=PyTuple_Pack(4,self.ox,self.oy,self.oz,self._radius )
        return tpl
@cython.boundscheck(False)
@cython.wraparound(False)     
cdef void cylinder_aabb(double pax,double pay,double paz, double pbx,double pby,double pbz,double ax,double ay,double az, double r, double[:,:] res) noexcept nogil:
    cdef:
        double axsq=ax*ax
        double aysq=ax*ay
        double azsq=az*az
        double a_norm_sq=axsq+aysq+azsq
        double ex=r*sqrt(1-axsq / a_norm_sq)
        double ey=r*sqrt(1-aysq / a_norm_sq)
        double ez=r*sqrt(1-azsq / a_norm_sq)
   
    res[0][0]=fmin(pax - ex,pbx - ex)
    res[0][1]=fmin(pay - ey,pby - ey)
    res[0][2]=fmin(paz - ez,pbz - ez)
    res[1][0]=fmax(pax + ex,pbx + ex)
    res[1][1]=fmax(pay + ey,pby + ey)
    res[1][2]=fmax(paz + ez,pbz + ez)

    
cdef class Cylinder(Implicit3D):
    cdef double ox
    cdef double oy
    cdef double oz
    cdef double dx
    cdef double dy
    cdef double dz
    cdef double ex
    cdef double ey
    cdef double ez
    cdef double _radius
    def __init__(self, double[:] start, double[:] end, double radius) -> None:
        super().__init__()
        self.ox=start[0]
        self.oy=start[1]
        self.oz=start[2]
        self.ex=end[0]
        self.ey=end[1]
        self.ez=end[2]
        self._radius=radius
        self._calculate_direction()
        self._calculate_bounds()

    @cython.boundscheck(False)
    @cython.wraparound(False)  
    cdef double cimplicit(self, double x,double y, double z) noexcept nogil:
       
        cdef double ax = x - self.ox
        cdef double  ay = y - self.oy
        cdef double  az = z - self.oz
        cdef double  bn = self.dx**2 + self.dy**2 + self.dz**2
        cdef double n = sqrt((x-(ax * self.dx * self.dx / bn + ay * self.dx * self.dy / bn + az * self.dx * self.dz / bn + self.ox))**2+(y- (ax * self.dx * self.dy / bn + ay * self.dy * self.dy / bn + az * self.dy * self.dz / bn + self.oy))**2+(z- (ax * self.dx * self.dz / bn + ay * self.dy * self.dz / bn + az * self.dz * self.dz / bn + self.oz))**2)
        return n
            
    @cython.boundscheck(False)
    @cython.wraparound(False)  
    cdef void cnormal(self, double x,double y, double z, double[:] result) noexcept nogil:
        cdef double n
        self._normal_not_unit(x,y,z,result)
        n=norm3d(result)
        result[0]/=n
        result[1]/=n
        result[2]/=n
    


    cdef void _calculate_direction(self) noexcept nogil:
        self.dx=self.ex-self.ox
        self.dy=self.ey-self.oy
        self.dz=self.ez-self.oz


    cdef void _calculate_bounds(self) noexcept nogil:
        cylinder_aabb(self.ox,self.oy,self.oz,self.ex,self.ey,self.ez,self.dx,self.dy,self.dz,self._radius,self._bounds)

    def astuple(self):
        cdef tuple tpl=PyTuple_Pack(4,self.ox,self.oy,self.oz,self.ex,self.ey,self.ez,self._radius )
        return tpl
    
    @property
    def start(self):
        cdef double[:] orig = np.empty(3)
        orig[0]=self.ox
        orig[1]=self.oy
        orig[2]=self.oz
        return orig
    @start.setter
    def start(self, double[:] end):

        self.ox=end[0]
        self.oy=end[1]
        self.oz=end[2]
        self._calculate_direction()
        self._calculate_bounds()
    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, double val):
        self._radius=val
        self._calculate_bounds()
    @property
    def end(self):
        cdef double[:] orig = np.empty(3)
        orig[0]=self.ex
        orig[1]=self.ey
        orig[2]=self.ez
        return orig
    
    @end.setter
    def end(self, double[:] end):

        self.ex=end[0]
        self.ey=end[1]
        self.ez=end[2]
        self._calculate_direction()
        self._calculate_bounds()
     
    @property
    def direction(self):
        cdef double[:] orig = np.empty(3)
        orig[0]=self.dx
        orig[1]=self.dy
        orig[2]=self.dz
        return orig
    @property
    def bounds(self):
        return self._bounds
    cpdef void set(self, double[:] start, double[:] end,double radius):
        self.ox=start[0]
        self.oy=start[1]
        self.oz=start[2]
        self.ex=end[0]
        self.ey=end[1]
        self.ez=end[2]
        self._radius=radius
        self._calculate_direction()
        self._calculate_bounds()
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _normal_not_unit(self,double x,double y, double z,double[:] res) noexcept nogil:
        cdef double ax = x - self.ox
        cdef double  ay = y - self.oy
        cdef double  az = z - self.oz
        cdef double  bn = self.dx**2 + self.dy**2 + self.dz**2
        # return start + vector_projection(point - start, direction)
        res[0]=x-(ax * self.dx * self.dx / bn + ay * self.dx * self.dy / bn + az * self.dx * self.dz / bn + self.ox)
        res[1]=y- (ax * self.dx * self.dy / bn + ay * self.dy * self.dy / bn + az * self.dy * self.dz / bn + self.oy)
        res[2]=z- (ax * self.dx * self.dz / bn + ay * self.dy * self.dz / bn + az * self.dz * self.dz / bn + self.oz)
        
cdef class CylinderPipe(Cylinder):
    """
    Straight cylindrical pipe with adjustable thickness.
    
    """
    cdef double _thickness
 
    def __init__(self, double[:] start, double[:] end, double radius,  double thickness) -> None:
        self._thickness = thickness
        super().__init__(start, end, radius) 


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double cimplicit(self, double x, double y, double z) noexcept nogil:
        cdef double r=Cylinder.cimplicit(self, x,y,z)
        cdef double res = fabs(r)-self._thickness/2
        return res
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cnormal(self, double x, double y, double z, double[:] result) noexcept nogil:
        Implicit3D.cnormal(self, x,y,z, result)
    cdef void _calculate_bounds(self) noexcept nogil:
        cylinder_aabb(self.ox,self.oy,self.oz,self.ex,self.ey,self.ez,self.dx,self.dy,self.dz,self._radius+self._thickness/2, self._bounds)
