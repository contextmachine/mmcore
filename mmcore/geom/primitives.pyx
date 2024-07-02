cimport cython
import numpy as np
from libc.math cimport fabs, sqrt,fmin,fmax,pow,pi,sin,cos
from libc.stdlib cimport malloc,free
from cpython cimport PyTuple_New,PyTuple_Pack,PyTuple_GetItem,PyTuple_GET_ITEM
cimport numpy as np
from mmcore.geom.parametric cimport ParametricSurface
from mmcore.numeric.vectors cimport scalar_cross
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint solve2x2(double[:,:] matrix, double[:] y,  double[:] result) noexcept nogil:
  
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

    def __init__(self):
        self._bounds=np.empty((2,3))
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double cimplicit(self, double x,double y, double z) noexcept nogil:
        cdef double res=0.
        return res
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cgradient(self, double x, double y, double z, double[:] result) noexcept nogil:
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
    def gradient(self, double[:] point):
        cdef double x=point[0]
        cdef double y = point[1]
        cdef double z = point[2]
        cdef double[:] result=np.empty((3,))
        self.cgradient(x,y,z,result)
        return np.array(result)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def implicit( self, double[:] point):
        cdef double x=point[0]
        cdef double y = point[1]
        cdef double z = point[2]
        cdef double result = self.cimplicit(x,y,z)
        return result

    def bounds(self):
        return self._bounds

cdef class Sphere(Implicit3D):


    def __init__(self,double[:] origin, double radius ):
        super().__init__()
        self.ox=origin[0]
        self.oy=origin[1]
        self.oz=origin[2]
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
    cdef void cgradient(self, double x,double y, double z, double[:] result) noexcept nogil:
        cdef double n
        result[0]=x-self.ox
        result[1]=y-self.oy
        result[2]=z-self.oz
        n=norm3d(result)
        result[0]/=n
        result[1] /= n
        result[2] /= n


    def bounds(self):
        return self._bounds



    def astuple(self):
        cdef tuple tpl=PyTuple_Pack(4,self.ox,self.oy,self.oz,self._radius )
        return tpl

    @property
    def radius(self):
        cdef double r=self._radius

        return r
    @radius.setter
    def radius(self, double r):
        self._radius=r
        self._calculate_bounds()


    @property
    def origin(self):
        cdef double[:] orig=np.empty((3,))
        orig[0]=self.ox
        orig[1]=self.oy
        orig[2]=self.oz
        
        return orig
    @origin.setter
    def origin(self,  double[:] orig):

        self.ox=orig[0]
        self.oy=orig[1]
        self.oz=orig[2]
        self._calculate_bounds()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cylinder_aabb(double pax,double pay,double paz, double pbx,double pby,double pbz,double ax,double ay,double az, double r, double[:,:] res) noexcept nogil:
    cdef:
        double axsq=ax*ax
        double aysq=ay*ay
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


cdef class ImplicitCylinder(Implicit3D):

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
        cdef double n = sqrt((x-(ax * self.dx * self.dx / bn + ay * self.dx * self.dy / bn + az * self.dx * self.dz / bn + self.ox))**2+(y- (ax * self.dx * self.dy / bn + ay * self.dy * self.dy / bn + az * self.dy * self.dz / bn + self.oy))**2+(z- (ax * self.dx * self.dz / bn + ay * self.dy * self.dz / bn + az * self.dz * self.dz / bn + self.oz))**2)-self._radius
        return n

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cgradient(self, double x,double y, double z, double[:] result) noexcept nogil:
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
        cdef tuple tpl=PyTuple_Pack(7,self.ox, self.oy, self.oz, self.ex, self.ey, self.ez, self._radius )
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

    def bounds(self):
        return self._bounds

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




cdef class Cylinder(ParametricSurface):
    cdef ImplicitCylinder _implicit_prim
    cdef public  double[:] start
    cdef public  double[:] end
    cdef public  double[:] direction
    cdef public  double radius
    cdef public double[:] u
    cdef public double[:] v
    cdef double[:] W
    cdef double[:] D_hat
    cdef double  D_norm
    cdef double[:] U_hat

    def __init__(self, start, end, radius):
        self._implicit_prim=ImplicitCylinder(start,end,radius)
        self._interval=np.zeros((2,2))
        self.start = start
        self.end = end
        self.radius = radius
        self._interval[0][1]=pi*2
        self.direction =np.zeros(3)
        # Calculate direction vector D and its normalized form D_hat
        sub3d(self.end ,self.start,self.direction)
        self.D_norm = norm3d(self.direction)
        self.D_hat = np.asarray(self.direction)/ self.D_norm
        self._interval[1][1] =  self.D_norm

        # Choose an arbitrary vector W that is not collinear with D_hat
        if np.allclose(self.D_hat, [1, 0, 0]):
            self.W = np.array([0., 1., 0.])
        else:
            self.W = np.array([1., 0., 0.])

        # Calculate U and its normalized form U_hat
        self.u = scalar_cross(self.D_hat, self.W)
        self.U_hat = np.asarray(self.u) / norm3d(self.u)

        # Calculate V
        self.v = scalar_cross(self.D_hat, self.U_hat)
    def interval(self):
        return np.asarray(self._interval)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def gradient(self, double[:] point):

        return self._implicit_prim.gradient(point)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def implicit( self, double[:] point):

        return self._implicit_prim.implicit(point)

    def bounds(self):
        return self._implicit_prim.bounds()
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cevaluate(self, double u, double v, double[:] result):
            # Calculate the point on the cylinder surface for given t and theta
            cdef double cos_u=cos(u)
            cdef double sin_u = sin(u)

            result[0] = self.start[0] + v * self.D_hat[0] + self.radius * (cos_u * self.U_hat[0] + sin_u * self.v[0])
            result[1] = self.start[1] + v * self.D_hat[1] + self.radius * (cos_u* self.U_hat[1] + sin_u * self.v[1])
            result[2] = self.start[2] + v * self.D_hat[2] + self.radius * (cos_u * self.U_hat[2] + sin_u  * self.v[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double cimplicit(self, double x, double y, double z) noexcept nogil:
        cdef double n=self._implicit_prim.cimplicit(x,y,z)
        return n
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cgradient(self, double x, double y, double z, double[:] result) noexcept nogil:
        self._implicit_prim.cgradient(x,y,z, result)


cdef class Tube(ImplicitCylinder):
    """
    Straight cylindrical pipe with adjustable thickness.

    """


    def __init__(self, double[:] start, double[:] end, double radius,  double thickness) -> None:
        self._thickness = thickness
        super().__init__(start, end, radius)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double cimplicit(self, double x, double y, double z) noexcept nogil:
        cdef double x0 = pow(self.dx, 2);
        cdef double x1 = pow(self.dy, 2);
        cdef double x2 = pow(self.dz, 2);
        cdef double x3 = 1.0 / (x0 + x1 + x2);
        cdef double x4 = x3 * (self.oy - y);
        cdef double x5 = self.dx * self.dy;
        cdef double x6 = x3 * (self.oz - z);
        cdef double x7 = self.dz * x6;
        cdef double x8 = x3 * (self.ox - x);
        cdef double res = -1.0 / 2.0 * self._thickness + fabs(self._radius - sqrt(
            pow(self.dx * x7 - self.ox + x + x0 * x8 + x4 * x5, 2) + pow(self.dy * x7 - self.oy + x1 * x4 + x5 * x8 + y, 2) + pow(
                self.dx * self.dz * x8 + self.dy * self.dz * x4 - self.oz + x2 * x6 + z, 2)));
        return res
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cgradient(self, double x, double y, double z, double[:] result) noexcept nogil:
        cdef double x0 = pow(self.dx, 2);
        cdef double x1 = pow(self.dy, 2);
        cdef double x2 = pow(self.dz, 2);
        cdef double x3 = 1.0 / (x0 + x1 + x2);
        cdef double x4 = self.dy * x3;
        cdef double x5 = -self.ox + x;
        cdef double x6 = x5 + 0.001;
        cdef double x7 = -x6;
        cdef double x8 = self.dx * x7;
        cdef double x9 = self.oz - z;
        cdef double x10 = self.dz * x4;
        cdef double x11 = x10 * x9;
        cdef double x12 = self.oy - y;
        cdef double x13 = x1 * x3;
        cdef double x14 = -self.oy + y;
        cdef double x15 = x12 * x13 + x14;
        cdef double x16 = x11 + x15;
        cdef double x17 = self.dz * x3;
        cdef double x18 = x10 * x12;
        cdef double x19 = x2 * x3;
        cdef double x20 = -self.oz + z;
        cdef double x21 = x19 * x9 + x20;
        cdef double x22 = x18 + x21;
        cdef double x23 = x0 * x3;
        cdef double x24 = self.dx * x17;
        cdef double x25 = x24 * x9;
        cdef double x26 = self.dx * x4;
        cdef double x27 = x12 * x26;
        cdef double x28 = x25 + x27;
        cdef double x29 = self.ox - x;
        cdef double x30 = x29 + 0.001;
        cdef double x31 = x14 + 0.001;
        cdef double x32 = -x31;
        cdef double x33 = x23 * x29 + x5;
        cdef double x34 = x25 + x33;
        cdef double x35 = x24 * x29;
        cdef double x36 = x21 + x35;
        cdef double x37 = x26 * x29;
        cdef double x38 = x11 + x37;
        cdef double x39 = x12 + 0.001;
        cdef double x40 = x20 + 0.001;
        cdef double x41 = -x40;
        cdef double x42 = x27 + x33;
        cdef double x43 = x15 + x37;
        cdef double x44 = x18 + x35;
        cdef double x45 = x9 + 0.001;
        result[0] = -500.0 * fabs(self._radius - sqrt(
            pow(x16 + x26 * x30, 2) + pow(x22 + x24 * x30, 2) + pow(x23 * x30 + x28 + x5 - 0.001, 2))) + 500.0 * fabs(
            self._radius - sqrt(pow(x16 + x4 * x8, 2) + pow(x17 * x8 + x22, 2) + pow(x23 * x7 + x28 + x6, 2)));
        result[1]  = 500.0 * fabs(self._radius - sqrt(
            pow(x10 * x32 + x36, 2) + pow(x26 * x32 + x34, 2) + pow(x13 * x32 + x31 + x38, 2))) - 500.0 * fabs(
            self._radius - sqrt(pow(x10 * x39 + x36, 2) + pow(x26 * x39 + x34, 2) + pow(x13 * x39 + x14 + x38 - 0.001, 2)));
        result[2]  = 500.0 * fabs(self._radius - sqrt(
            pow(x10 * x41 + x43, 2) + pow(x24 * x41 + x42, 2) + pow(x19 * x41 + x40 + x44, 2))) - 500.0 * fabs(
            self._radius - sqrt(pow(x10 * x45 + x43, 2) + pow(x24 * x45 + x42, 2) + pow(x19 * x45 + x20 + x44 - 0.001, 2)));

    cdef void _calculate_bounds(self) noexcept nogil:
        cylinder_aabb(self.ox,self.oy,self.oz,self.ex,self.ey,self.ez,self.dx,self.dy,self.dz,self._radius+self._thickness/2, self._bounds)
    def astuple(self):
        cdef tuple tpl = PyTuple_Pack(8, self.ox, self.oy, self.oz, self.ex, self.ey, self.ez, self._radius, self._thickness)
        return tpl


