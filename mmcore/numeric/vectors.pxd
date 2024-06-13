
import numpy as np

cpdef dot_array_x_array(double[:,:] vec_a, double[:,:] vec_b)


cpdef dot_vec_x_array(double[:] vec_a, double[:,:] vec_b)


cpdef dot_array_x_vec(double[:,:] vec_a, double[:] vec_b)



cpdef dot(double[:,:] vec_a, double[:,:] vec_b)



cpdef scalar_dot(double[:] vec_a,double[:] vec_b)


cpdef scalar_norm(double[:] vec)


cpdef scalar_normalize(double[:] vec)


cpdef norm(double[:,:] vec)



cpdef unit(double[:,:] vec)



cpdef cross(double[:,:] vec_a,
          double[:,:] vec_b)



cpdef support_vector(double[:,:] vertices, double[:] direction)



cpdef multi_support_vector(double[:,:] vertices, double[:,:] directions)



cpdef gram_schmidt(double[:,:] vec_a,
                 double[:,:] vec_b)



cpdef spherical_to_cartesian(double[:,:] rtp)



cpdef cartesian_to_spherical(double[:,:] xyz)
   


cpdef cylindrical_to_xyz(double[:,:] rpz)


cpdef courtesan_to_cylindrical(double[:,:] xyz)

cpdef long double cdet(double[:,:] arr)




cpdef det(double[:,:] arr)



cpdef points_order(double[:,:] points)


cpdef multi_points_order(list points_list)


cpdef scalar_cross(double[:] vec_a,
          double[:] vec_b)


cpdef scalar_unit(double[:] vec)





cpdef int cinvert_jacobian(double[:,:] J, double[:,:] J_inv) noexcept nogil


cpdef void cinvert_jacobian_vec(double[:,:,:] J, double[:,:,:] J_inv, int[:] status) noexcept nogil



cpdef invert_jacobian(J)


cpdef vector_projection(double[:] a, double[:] b)
 

cpdef closest_point_on_ray( double[:] start, double[:] direction,  double[:] point)
  

cpdef closest_point_on_line(double[:] start,double[:] end, double[:] point)
 

cpdef bint solve2x2(double[:,:] matrix, double[:] y,  double[:] result)  noexcept nogil


cdef inline double scalar_dot3d(double* vec_a,
                            double*  vec_b):
    cdef double res = vec_a[0]*vec_b[0]+vec_a[1]*vec_b[1]+vec_a[2]*vec_b[2]

    return res
"""
cpdef double[:] scalar_gram_schmidt(double[:] vec_a,double[:] vec_b) noexcept nogil
cpdef double[:] make_perpendicular(double[:] vec_a, double[:] vec_b) noexcept nogil
cpdef double[:] linear_combination_3d(double a, double[:] v1, double b,double[:] v2) noexcept nogil

cpdef rotate(self, deg)
    sin, cos = function_DegToSinCos(deg)
    return Vector2D(self.x * cos - self.y * sin, self.x * sin + self.y * cos)

    cpdef rotateAroundPoint(self, v2d, deg)
        sin, cos = function_DegToSinCos(deg)
        v2d0 = self.sub(v2d)
        return Vector2D(v2d0.x * cos - v2d0.y * sin, v2d0.x * sin + v2d0.y * cos).add(v2d)



cpdef void function_linearCombine2D(double[2] v3d0, double r0, double[2] v3d1, double r1,
                                    double[2] result)
    result[0] = r0 * v3d0[0] + r1 * v3d1[0]
    result[1] = r0 * v3d0[1] + r1 * v3d1[1]


cpdef void function_linearCombine3D(double[3] v3d0, double r0, double[3] v3d1, double r1, double[3] v3d2, double r2=0, double[3] result)

    result[0]=r0 * v3d0[0] + r1 * v3d1[0] + r2 * v3d2[0]
    result[1]=r0 * v3d0[1] + r1 * v3d1[1] + r2 * v3d2[1]
    result[2]= r0 * v3d0[2]+ r1 * v3d1[2] + r2 * v3d2[2]



cpdef function_DegToSinCos(deg)
    deg = deg * pi / 180
    return (sin(deg), math.cos(deg))


cpdef double function_SolveEquationDeg1(double a, double b)
    return -b / a

cpdef function_SolveEquationDeg2(double a, double b, double c, result[:]) # sqrt exception will be raised
    cpdef static double
    if a!=0:  # wenn a!=0
        h0 = sqrt((b * b - 4 * a * c) / 4 * a * a)
        h1 = b / -2 * a

        return (h1 + h0, h1 - h0)
    return function_SolveEquationDeg1(b, c)


cpdef function_PointOverPlane(v3d, n_v3d, d) #{...stellt fest, ob der Punkt p "vor" der Ebene  nv*x-d=0 liegt.}
    return v3d.scalarProduct(n_v3d) - d >= 0


cpdef function_GetPlaneEquation(v3d0, v3d1, v3d2) #Exception wenn keine ebene
    n_v3d = ((v3d1.sub(v3d0)).vectorProduct(v3d2.sub(v3d0))).normalize()
    if not n_v3d.equals(Vector3D())return (n_v3d, n_v3d.scalarProduct(v3d0))
    raise
"""