import math

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from mmcore.geom.nurbs.surface import NURBSSurface,split_surface_u,split_surface_v,subdivide_surface

from mmcore.geom.nurbs.curve import NURBSCurve
from mmcore.numeric import uvs, evaluate_length
from mmcore.numeric.aabb import aabb
from mmcore.numeric.algorithms.gjk import gjk_collision_detection
from mmcore.numeric.algorithms.cygjk  import gjk
from mmcore.numeric.vectors import normal_from_4pt, scalar_unit,dot_vec_x_array,dot_array_x_vec
np.set_printoptions(suppress=True)
from mmcore.numeric.vectors import scalar_norm, scalar_dot

def find_mean_plane(points):
    # Step 1: Compute the centroid (mean of the points)
    centroid = np.mean(points, axis=0)

    # Step 2: Subtract the centroid from each point (center the points)
    centered_points = points - centroid

    # Step 3: Construct the covariance matrix
    covariance_matrix = np.cov(centered_points, rowvar=False)

    # Step 4: Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Step 5: The eigenvector corresponding to the smallest eigenvalue
    # is the normal to the plane
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

    # Step 6: Formulate the plane equation (n_x * x + n_y * y + n_z * z = d)
    # Here, d = -normal_vector.dot(centroid)
    d = -normal_vector.dot(centroid)

    # The plane equation can be represented as (n_x, n_y, n_z, d)
    plane_equation = np.append(normal_vector, d)

    return plane_equation

SUBDS=0
from mmcore.numeric.plane import project_point_by_normal
from mmcore.numeric.plane.cplane import planes_plane_intersect, plane_plane_intersect
_uvs=uvs(4,4)
D=0.9
def bounding_boxes_intersection(self, other):
    """
    Calculate the intersection of this bounding box with another bounding box.

    :param other: The other bounding box to intersect with.
    :return: A new BoundingBox representing the intersection, or None if there is no intersection.
    """
    # Calculate the maximum of the minimum points for each dimension
    max_min_x = max(self[0][0], other[0][0])
    max_min_y = max(self[0][1], other[0][1])
    max_min_z = max(self[0][2], other[0][2])

    # Calculate the minimum of the maximum points for each dimension
    min_max_x = min(self[1][0], other[1][0])
    min_max_y = min(self[1][1], other[1][1])
    min_max_z = min(self[1][2], other[1][2])

    # Check if the bounding boxes intersect
    if max_min_x > min_max_x or max_min_y > min_max_y or max_min_z > min_max_z:
        return None

    # Create and return the intersection bounding box
    intersection_min = (max_min_x, max_min_y, max_min_z)
    intersection_max = (min_max_x, min_max_y, min_max_z)
    return intersection_min, intersection_max
def calculate_parametric_tol(self:NURBSSurface,tol=0.1):
    crv_u = NURBSCurve(np.copy(self.control_points[:, 0, :]), degree=self.degree[0], knots=self.knots_u)
    crv_v = NURBSCurve(np.copy(self.control_points[0, :, :]), degree=self.degree[1], knots=self.knots_v)
    lu,_=evaluate_length(crv_u.derivative,*crv_u.interval(),tol)
    lv,_=evaluate_length(crv_v.derivative,*crv_v.interval(),tol)
    return tol/lu,tol/lv


def find_intersections(
    surface1, u1_range, v1_range, surface2, u2_range, v2_range, tolerance,depth=0,max_depth=10
):
        s1_control_points=surface1.control_points
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

        bbox1_min, bbox1_max=bbox(surface1)
        bbox2_min, bbox2_max=bbox(surface2)
        # Check if bounding boxes intersect

        if not bounding_boxes_intersect(bbox1_min, bbox1_max, bbox2_min, bbox2_max):
            #print('bb',(u1_range,v1_range),(u2_range,v2_range))

            return []  # No intersection in this subdivision
        if max_depth <= depth:

            u1_mid = (u1_range[0] + u1_range[1]) / 2
            v1_mid = (v1_range[0] + v1_range[1]) / 2
            u2_mid = (u2_range[0] + u2_range[1]) / 2
            v2_mid = (v2_range[0] + v2_range[1]) / 2
            return [ ((u1_mid, v1_mid),(u2_mid, v2_mid))]

        #bb1_min,bb1_max=bounding_boxes_intersection((bbox1_min, bbox1_max), (bbox2_max, bbox2_min))
        d1=np.array(bbox1_max) - np.array(bbox1_min)
        d2=np.array(bbox2_max) - np.array(bbox2_min)
        if (max(d1) <=tolerance) and( max(d2)<=tolerance):
            # Return a representative point (e.g., midpoint)
            u1_mid = (u1_range[0] + u1_range[1]) / 2
            v1_mid = (v1_range[0] + v1_range[1]) / 2
            u2_mid = (u2_range[0] + u2_range[1]) / 2
            v2_mid = (v2_range[0] + v2_range[1]) / 2
            return [((u1_mid, v1_mid), (u2_mid, v2_mid))]

        n1=np.zeros(3)
        #n2 = np.zeros(3)
        a1 = s1_control_points[0,0,:]
        b1 = s1_control_points[-1,0,:]
        c1 = s1_control_points[-1,-1,:]
        d1 = s1_control_points[0,-1,:]
        o1=surface1.evaluate_v2(0.5,0.5)
        #a2 = s2_control_points[0,0,:]
        #b2 = s2_control_points[-1,0,:]
        #c2 = s2_control_points[-1,-1,:]
        #d2 = s2_control_points[0,-1,:]

        #o1 = (a1+b1+c1+d1)/4
        #o2 = (a2 + b2 + c2 + d2) * 0.25

        normal_from_4pt(a1,b1,c1,d1,n1)
        #ff=find_mean_plane(s1_control_points_flat)
        n1/=scalar_norm(n1)
        #normal_from_4pt(a2, b2, c2, d2,n2)
        d1 =-n1.dot(o1)

        #d2 =-n2.dot(o2)
        res1=n1[0]*s1_control_points_flat[...,0]+ n1[1]*s1_control_points_flat[...,1]+ n1[2]*s1_control_points_flat[...,2]+d1
        #res1=np.abs(np.array(dot_array_x_vec(s1_control_points_flat,n1 ))+d1   )

        if np.all(np.abs(res1)<=tolerance):

            #res2 = np.array(dot_array_x_vec(s2_control_points_flat, n1)) + d1
            res2 = n1[0] * s2_control_points_flat[..., 0] + n1[1] * s2_control_points_flat[..., 1] + n1[2] * \
                   s2_control_points_flat[..., 2] + d1

            if np.all(res2<0) or np.all(res2>0):
                #print("p n", res2)
                return []




        #    n2 /= scalar_norm(n2)
        #    rt = scalar_dot(n2, n1)
        #
        #    d1 = scalar_dot(o2-o1, n2) - o1
        #    dst = scalar_norm(d1)
        #    if abs(rt)>=D :
        #        print('rt',rt,dst)
        #        return []

        h1=ConvexHull(s1_control_points_flat)
        h2=ConvexHull(s2_control_points_flat)


        gjk_res= gjk(h1.points[h1.vertices],h2.points[h2.vertices],tol=1e-8)

        if not gjk_res:
           #print("g n")
           return []


        # Check stopping criterion
        if (u1_range[1] - u1_range[0]) <(tolerance) and (v1_range[1] - v1_range[0]) < (tolerance):
            # Return a representative point (e.g., midpoint)
            u1_mid = (u1_range[0] + u1_range[1]) / 2
            v1_mid = (v1_range[0] + v1_range[1]) / 2
            u2_mid = (u2_range[0] + u2_range[1]) / 2
            v2_mid = (v2_range[0] + v2_range[1]) / 2
            return [((u1_mid, v1_mid), (u2_mid, v2_mid))] # This is a candidate intersection point
        #
        if (h1.volume <= (tolerance)) or (h2.volume <= (tolerance)):
            u1_mid = (u1_range[0] + u1_range[1]) / 2
            v1_mid = (v1_range[0] + v1_range[1]) / 2
            u2_mid = (u2_range[0] + u2_range[1]) / 2
            v2_mid = (v2_range[0] + v2_range[1]) / 2
            return [((u1_mid, v1_mid), (u2_mid, v2_mid))] # This is a candidate intersection point
        # Otherwise, subdivide the parameter domains
        u1_mid = (u1_range[0] + u1_range[1]) / 2
        v1_mid = (v1_range[0] + v1_range[1]) / 2

        u2_mid = (u2_range[0] + u2_range[1]) / 2
        v2_mid = (v2_range[0] + v2_range[1]) / 2

        intersections = []

        # Recursive calls for each pair of subdomains

        s11, s12,s21,s22 = subdivide_surface(surface1)
        s31,s32, s41,s42= subdivide_surface(surface2)

        srfs=[[s11, s12],[s21,s22]],[[s31,s32],[s41,s42]
        ]


        for i,sub_u1_range in enumerate([(u1_range[0], u1_mid), (u1_mid, u1_range[1])]):
            for j,sub_v1_range in  enumerate([(v1_range[0], v1_mid), (v1_mid, v1_range[1])]):
                for k,sub_u2_range in  enumerate([(u2_range[0], u2_mid), (u2_mid, u2_range[1])]):
                    for l,sub_v2_range in  enumerate([(v2_range[0], v2_mid), (v2_mid, v2_range[1])]):



                        intersections.extend(
                                find_intersections(
                                    srfs[0][i][j],
                                    sub_u1_range,
                                    sub_v1_range,
                                    srfs[1][k][l],
                                    sub_u2_range,
                                    sub_v2_range,
                                    tolerance,
                                )
                            )

        return intersections


def bounding_boxes_intersect(bbox1_min, bbox1_max, bbox2_min, bbox2_max):
    # Check if bounding boxes intersect in all three dimensions
    return (
        bbox1_max[0] >= bbox2_min[0]
        and bbox1_min[0] <= bbox2_max[0]
        and bbox1_max[1] >= bbox2_min[1]
        and bbox1_min[1] <= bbox2_max[1]
        and bbox1_max[2] >= bbox2_min[2]
        and bbox1_min[2] <= bbox2_max[2]
    )
def next_power_of_two(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


def next_power_of_4(n):
    # Step 1: Check if `n` is already a power of 4
    if n > 0 and (n & (n - 1)) == 0 and (n - 1) % 3 == 0:
        return n

    # Step 2: Find the next power of 2 greater than or equal to `n`
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1

    # Step 3: Ensure it's a power of 4
    while (n - 1) % 3 != 0:
        n <<= 1

    return n


def detect_intersection(surf1,surf2,tolerance=1e-3):
    surf1.normalize_knots()
    surf2.normalize_knots()
    ku1=len(np.unique(surf1.knots_u))*surf1.degree[0]
    kv1 = len(np.unique(surf1.knots_v)) * surf1.degree[1]
    ku2 = len(np.unique(surf2.knots_u)) * surf2.degree[0]
    kv2 = len(np.unique(surf2.knots_v)) * surf2.degree[1]
    max_depth=max((kv1,ku2,ku1,kv2))

    ints = find_intersections(
        surf1,
        (0.0, 1.0),
        (0.0, 1.0),
        surf2,
        (0.0, 1.0),
        (0.0, 1.0),
        tolerance,
        0,
        max_depth


    )
    return ints

if __name__ == "__main__":
    # runfile('/Users/andrewastakhov/dev/mmcore-dev/mmcore/numeric/intersection/ssx/dqr4.py', wdir='/Users/andrewastakhov/dev/mmcore-dev')
    # python3 mmcore/numeric/intersection/ssx/dqr4.py
    def bbox_(surface, u0, u1, v0, v1):
        return aabb(
            np.array(
                [
                    surface.evaluate_v2((u0 + u1) * 0.5, (v0 + v1) * 0.5),
                    surface.evaluate_v2(u0, v0),
                    surface.evaluate((u1, v1)),
                    surface.evaluate((u0, v1)),
                    surface.evaluate((u1, v0)),
                ]
            )
        )




    def bbox(surf: NURBSSurface):
        return surf.bbox()


    def geomdl_convex_hull(surf: NURBSSurface):

        #print(np.array(surf.control_points_flat[...,:3]))

        return ConvexHull(np.array(surf.control_points_flat[...,:3]))


    from mmcore._test_data import ssx as ssx_test_data

    import time
    S1,S2=ssx_test_data[2]
    #import yappi
    s = time.time()
    import yappi

    yappi.set_clock_type("wall")  # Use set_clock_type("wall") for wall time
    yappi.start()
    ints=    detect_intersection(S1,
                        S2,
                        0.1
                        )



    yappi.stop()
    func_stats = yappi.get_func_stats()
    func_stats.save(f"{__file__.replace('.py', '')}_{int(time.time())}.pstat", type='pstat')

    print(time.time() - s)
    pts1 = []
    pts2 = []
    curvatures = []
    for (i1, i2),(j1,j2) in ints:
        pts1.append(S1.evaluate_v2(i1,i2).tolist())
        pts2.append(S2.evaluate_v2(j1, j2).tolist())
    pts=[pts1,pts2]