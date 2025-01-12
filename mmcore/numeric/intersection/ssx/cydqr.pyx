# distutils: language = c++
cimport cython
from libcpp.vector cimport vector
from mmcore.numeric.vectors cimport scalar_norm, scalar_dot,sub3d,    cross_d1_3d,scalar_normalize


from mmcore.numeric.algorithms.cygjk cimport gjk_collision_detection,Vec3
from libc.math cimport  fmin,fmax, sqrt,fabs,ceil

from mmcore.geom.nurbs cimport NURBSSurface,subdivide_surface
cimport numpy as cnp
import numpy as np
from scipy.spatial import ConvexHull


cdef extern from "_dqr.cpp" nogil:
    cdef cppclass UV12:
        double u1;
        double v1;
        double u2;
        double v2;
        UV12() except +
        UV12(double u1,double v1, double u2,double v2) except +



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
cdef inline bint gjk(double[:,:] v1, double[:,:] v2, double tol) noexcept nogil:
    cdef vector[Vec3[double]] vf1=vector[Vec3[double]](v1.shape[0])
    cdef vector[Vec3[double]] vf2=vector[Vec3[double]](v2.shape[0])
    cdef int i;
    cdef size_t max_iter=v1.shape[0]*v2.shape[0]
    for i in range(v1.shape[0]):
        vf1[i][0]=v1[i][0]
        vf1[i][1]=v1[i][1]
        vf1[i][2]=v1[i][2]
    for i in range(v2.shape[0]):
        vf2[i][0]=v2[i][0]
        vf2[i][1]=v2[i][1]
        vf2[i][2]=v2[i][2]
    cdef bint result= <bint>gjk_collision_detection(vf1, vf2, tol, max_iter)
    return result
cnp.import_array()
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void normal_from_4pt(double[:] a, double[:] b, double[:] c, double[:] d, double[:] result) noexcept nogil:
    cdef double[3] temp1
    cdef double[3] temp2
    
    temp1[0] = c[0] - a[0]
    temp1[1] = c[1] - a[1]
    temp1[2] = c[2] - a[2]
    temp2[0] = d[0] - b[0]
    temp2[1] = d[1] - b[1]
    temp2[2] = d[2] - b[2]
    cross_d1_3d(temp1, temp2, result)
  
cdef inline bint bounding_boxes_intersection(self, other):
    """
    Calculate the intersection of this bounding box with another bounding box.

    :param other: The other bounding box to intersect with.
    :return: A new BoundingBox representing the intersection, or None if there is no intersection.
    """
    # Calculate the maximum of the minimum points for each dimension
    cdef double max_min_x = fmax(self[0][0], other[0][0])
    cdef double  max_min_y = fmax(self[0][1], other[0][1])
    cdef double max_min_z = fmax(self[0][2], other[0][2])

    # Calculate the minimum of the maximum points for each dimension
    cdef double min_max_x = fmin(self[1][0], other[1][0])
    cdef double min_max_y = fmin(self[1][1], other[1][1])
    cdef double min_max_z = fmin(self[1][2], other[1][2])

    # Check if the bounding boxes intersect
    if (max_min_x > min_max_x) or (max_min_y > min_max_y) or (max_min_z > min_max_z):
        return 0




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef bint find_intersections(
        NURBSSurface surface1, double[2] u1_range, double[2] v1_range, NURBSSurface surface2, double[2] u2_range, double[2] v2_range, double tolerance, int depth, int max_depth, vector[UV12]& result
):
    s1_control_points = surface1.control_points
    #s2_control_points = surface2.control_points
    s1_control_points_flat = surface1.control_points_flat
    s2_control_points_flat = surface2.control_points_flat

    #print(kpu,kpv)
    #print(u1_range,v1_range,u2_range,v2_range)
    # Compute bounding boxes

    #bbox1_min, bbox1_max = bbox(surface1, *u1_range, *v1_range)
    #bbox2_min, bbox2_max = bbox(surface2, *u2_range, *v2_range)

    #bbox1_min, bbox1_max=aabb(np.array([a1,b1,c1,d1]))
    #bbox2_min, bbox2_max=aabb(np.array([a2,b2,c2,d2]))
    cdef NURBSSurface s11, s12, s21, s22,s31, s32, s41, s42
    cdef list srfs
    cdef double[:,:] bbox1 = np.zeros((2,3))
    cdef double[:,:] bbox2 = np.zeros((2,3))
    cdef double[:] temp=np.zeros((3,))
    cdef double[:] a1,b1,c1,d1,o1
    cdef object h1,h2
    cdef double[:] n1 
    cdef double dd1,dd2,tempd1,tempd2,u1_mid,v1_mid,u2_mid,v2_mid
    cdef bint res=0
    cdef int i,j,k,l
   
    cdef double[2][2] sub_u1_range
    cdef double[2][2] sub_v1_range
    cdef double[2][2] sub_u2_range
    cdef double[2][2] sub_v2_range









    surface1.cbbox( bbox1 )
    surface2.cbbox(bbox2)
    # Check if bounding boxes intersect

    if not bounding_boxes_intersect(bbox1[0], bbox1[1], bbox2[0], bbox2[1]):
        #print('bb',(u1_range,v1_range),(u2_range,v2_range))

        return 0  # No intersection in this subdivision
    if max_depth <= depth:
        u1_mid = (u1_range[0] + u1_range[1]) / 2
        v1_mid = (v1_range[0] + v1_range[1]) / 2
        u2_mid = (u2_range[0] + u2_range[1]) / 2
        v2_mid = (v2_range[0] + v2_range[1]) / 2
        result.emplace_back(u1_mid, v1_mid,u2_mid, v2_mid)
        return 1
    
    #bb1_min,bb1_max=bounding_boxes_intersection((bbox1_min, bbox1_max), (bbox2_max, bbox2_min))
    sub3d(bbox1[1],bbox1[0],temp)
    dd1=fmax(temp[2],fmax(temp[0],temp[1]))
    sub3d(bbox2[1],bbox2[0],temp)
    dd2=fmax(temp[2],fmax(temp[0],temp[1]))

    
    if (dd1 <= tolerance) and (dd2 <= tolerance):
        # Return a representative point (e.g., midpoint)
        u1_mid = (u1_range[0] + u1_range[1]) / 2
        v1_mid = (v1_range[0] + v1_range[1]) / 2
        u2_mid = (u2_range[0] + u2_range[1]) / 2
        v2_mid = (v2_range[0] + v2_range[1]) / 2
        result.emplace_back(u1_mid, v1_mid,u2_mid, v2_mid)
        return 1

    n1 = np.zeros(3)
    o1=np.zeros(3)
    #n2 = np.zeros(3)
    a1 = s1_control_points[0, 0, :]
    b1 = s1_control_points[-1, 0, :]
    c1 = s1_control_points[-1, -1, :]
    d1 = s1_control_points[0, -1, :]
    surface1.cevaluate(0.5, 0.5, o1)
    #a2 = s2_control_points[0,0,:]
    #b2 = s2_control_points[-1,0,:]
    #c2 = s2_control_points[-1,-1,:]
    #d2 = s2_control_points[0,-1,:]

    #o1 = (a1+b1+c1+d1)/4
    #o2 = (a2 + b2 + c2 + d2) * 0.25

    normal_from_4pt(a1, b1, c1, d1, n1)
    #ff=find_mean_plane(s1_control_points_flat)
    scalar_normalize(n1)
    #normal_from_4pt(a2, b2, c2, d2,n2)
    dd1 = scalar_dot(n1,o1)

    dd1*= -1
    #d2 =-n2.dot(o2)
    tempd1 = fabs(n1[0] * s1_control_points_flat[..., 0] + n1[1] * s1_control_points_flat[..., 1] + n1[2] * \
           s1_control_points_flat[..., 2] + dd1)
    #res1=np.abs(np.array(dot_array_x_vec(s1_control_points_flat,n1 ))+d1   )

    if np.all(tempd1 <= tolerance):

        #res2 = np.array(dot_array_x_vec(s2_control_points_flat, n1)) + d1
        tempd1 = n1[0] * s2_control_points_flat[..., 0] + n1[1] * s2_control_points_flat[..., 1] + n1[2] * \
               s2_control_points_flat[..., 2] + dd1

        if np.all(tempd1 < 0) or np.all(tempd1 > 0):
            #print("p n", res2)
            return 0

    #    n2 /= scalar_norm(n2)
    #    rt = scalar_dot(n2, n1)
    #
    #    d1 = scalar_dot(o2-o1, n2) - o1
    #    dst = scalar_norm(d1)
    #    if abs(rt)>=D :
    #        print('rt',rt,dst)
    #        return []

    h1 = ConvexHull(s1_control_points_flat)
    h2 = ConvexHull(s2_control_points_flat)
    
    res = gjk(h1.points[h1.vertices], h2.points[h2.vertices], tol=1e-8)

    if res==0:
        
        return 0

    # Check stopping criterion
    if (u1_range[1] - u1_range[0]) < (tolerance) and (v1_range[1] - v1_range[0]) < (tolerance):
        # Return a representative point (e.g., midpoint)
        u1_mid = (u1_range[0] + u1_range[1]) / 2
        v1_mid = (v1_range[0] + v1_range[1]) / 2
        u2_mid = (u2_range[0] + u2_range[1]) / 2
        v2_mid = (v2_range[0] + v2_range[1]) / 2
        result.emplace_back(u1_mid, v1_mid,u2_mid, v2_mid)
        return 1 # This is a candidate intersection point
    #
    if (h1.volume <= (tolerance)) or (h2.volume <= (tolerance)):
        u1_mid = (u1_range[0] + u1_range[1]) / 2
        v1_mid = (v1_range[0] + v1_range[1]) / 2
        u2_mid = (u2_range[0] + u2_range[1]) / 2
        v2_mid = (v2_range[0] + v2_range[1]) / 2
        result.emplace_back(u1_mid, v1_mid,u2_mid, v2_mid)
        return 1 # This is a candidate intersection point
    # Otherwise, subdivide the parameter domains
    u1_mid = (u1_range[0] + u1_range[1]) / 2
    v1_mid = (v1_range[0] + v1_range[1]) / 2

    u2_mid = (u2_range[0] + u2_range[1]) / 2
    v2_mid = (v2_range[0] + v2_range[1]) / 2

   

    # Recursive calls for each pair of subdomains

    s11, s12, s21, s22 = subdivide_surface(surface1)
    s31, s32, s41, s42 = subdivide_surface(surface2)

    srfs = [[s11, s12], [s21, s22]], [[s31, s32], [s41, s42]]
    sub_u1_range=[(u1_range[0], u1_mid), (u1_mid, u1_range[1])]
    sub_v1_range=[(v1_range[0], v1_mid), (v1_mid, v1_range[1])]
    sub_u2_range=[(u2_range[0], u2_mid), (u2_mid, u2_range[1])]
    sub_v2_range=[(v2_range[0], v2_mid), (v2_mid, v2_range[1])]
    
    res=0
    for i in range(2):
        for j in range(2):
            for k, in range(2):
                for l in range(2):
                    
                    rrrr=find_intersections(srfs[0][i][j],
                            sub_u1_range[i],
                            sub_v1_range[j],
                            srfs[1][k][l],
                            sub_u2_range[k],
                            sub_v2_range[l],
                            tolerance,
                            depth+1,
                            max_depth,
                            result
                        )
                    if rrrr==1:
                        res=1
                    

    return res


cdef inline bint bounding_boxes_intersect(double[:] bbox1_min, double[:] bbox1_max, double[:] bbox2_min, double[:] bbox2_max) noexcept nogil:
    # Check if bounding boxes intersect in all three dimensions
    cdef bint res=(bbox1_max[0] >= bbox2_min[0]) and (bbox1_min[0] <= bbox2_max[0]) and (bbox1_max[1] >= bbox2_min[1])and (bbox1_min[1] <= bbox2_max[1])and (bbox1_max[2] >= bbox2_min[2]) and (bbox1_min[2] <= bbox2_max[2])
    return res





cdef void cdetect_intersection(NURBSSurface surf1, NURBSSurface surf2, tolerance,vector[UV12]& intersections):
    surf1.normalize_knots()
    surf2.normalize_knots()
    cdef int ku1 = len(np.unique(surf1.knots_u)) * surf1.degree[0]
    cdef int  kv1 = len(np.unique(surf1.knots_v)) * surf1.degree[1]
    cdef int ku2 = len(np.unique(surf2.knots_u)) * surf2.degree[0]
    cdef int kv2 = len(np.unique(surf2.knots_v)) * surf2.degree[1]
    cdef int max_depth = <int>ceil(fmax(fmax(kv1, ku2), fmax(ku1, kv2)))
    intersections.reserve(max_depth)
    cdef double[2] su1=[surf1._interval[0][0],surf1._interval[0][1]]
    cdef double[2] sv1=[surf1._interval[1][0],surf1._interval[1][1]]
    cdef double[2] su2=[surf2._interval[0][0],surf2._interval[0][1]]
    cdef double[2] sv2=[surf2._interval[1][0],surf2._interval[1][1]]
    cdef bint ints = find_intersections(
        surf1,
        su1,
        sv1,
        surf2,
       su2,
         sv2,
  
        tolerance,
        0,
        max_depth,
        intersections

    )

  
def detect_intersection(NURBSSurface surf1, NURBSSurface surf2,double tolerance=0.1):
    
    cdef vector[UV12] intersections=vector[UV12]()
    cdetect_intersection(surf1,surf2,tolerance,intersections)
    cdef double[:,:,:] ixs = np.zeros((2,intersections.size(),2))
    for i in range(intersections.size()):
        ixs[0,i,0]=intersections[i].u1
        ixs[0,i,1]=intersections[i].v1   
        ixs[1,i,0]=intersections[i].u2     
        ixs[1,i,1]=intersections[i].v2   

    return np.array(ixs)










