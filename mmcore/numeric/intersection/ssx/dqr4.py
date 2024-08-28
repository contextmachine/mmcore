import numpy as np
from scipy.spatial import ConvexHull
from mmcore.geom.nurbs.surface import NURBSSurface,split_surface_u,split_surface_v,subdivide_surface
from mmcore.numeric import uvs
from mmcore.numeric.aabb import aabb
from mmcore.numeric.algorithms.gjk import gjk_collision_detection
from mmcore.numeric.vectors import normal_from_4pt, scalar_unit

from mmcore.numeric.vectors import scalar_norm, scalar_dot


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

def find_intersections(
    surface1, u1_range, v1_range, surface2, u2_range, v2_range, tolerance
):

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
        #bb1_min,bb1_max=bounding_boxes_intersection((bbox1_min, bbox1_max), (bbox2_max, bbox2_min))
        #if scalar_norm(bbox1_max-bbox1_min) <=tolerance and scalar_norm(bbox2_max-bbox2_min) <=tolerance:
        #    # Return a representative point (e.g., midpoint)
        #    u1_mid = (u1_range[0] + u1_range[1]) / 2
        #    v1_mid = (v1_range[0] + v1_range[1]) / 2
        #    return [(u1_mid, v1_mid)]  # This is a candidate intersection point
        #if (u11 - u10) < 1/(len(set(surface1.knotvector_u))*surface1.degree_u):
        #
        #    n1=np.zeros(3)
        #    n2 = np.zeros(3)
        #    a1 = surface1.evaluate_v2(0., 0.)
        #    b1 = surface1.evaluate_v2(1., 0)
        #    c1 = surface1.evaluate_v2(1., 1.)
        #    d1 = surface1.evaluate_v2(0., 1.)
        #    a2 = surface2.evaluate_v2(0., 0.)
        #    b2 = surface2.evaluate_v2(1., 0)
        #    c2 = surface2.evaluate_v2(1., 1.)
        #    d2 = surface2.evaluate_v2(0., 1.)
        #    o1 = (a1+b1+c1+d1)*0.25
        #    o2 = (a2 + b2 + c2 + d2) * 0.25
        #    normal_from_4pt(a1,b1,c1,d1,n1)
        #    normal_from_4pt(a2, b2, c2, d2,n2)
        #    n1 /= scalar_norm(n1)
        #    n2 /= scalar_norm(n2)
        #    rt = scalar_dot(n2, n1)
        #
        #    d1 = scalar_dot(o2-o1, n2) - o1
        #    dst = scalar_norm(d1)
        #    if abs(rt)>=D :
        #        print('rt',rt,dst)
        #        return []

        h1=geomdl_convex_hull(surface1)
        h2=geomdl_convex_hull(surface2)



        gjk_res=gjk_collision_detection(h1.points,h2.points)
        if not gjk_res:

            return []




        # Check stopping criterion
        if (u1_range[1] - u1_range[0]) < tolerance and (
            v1_range[1] - v1_range[0]
        ) < tolerance:
            # Return a representative point (e.g., midpoint)
            u1_mid = (u1_range[0] + u1_range[1]) / 2
            v1_mid = (v1_range[0] + v1_range[1]) / 2
            return [(u1_mid, v1_mid)]  # This is a candidate intersection point

        # Otherwise, subdivide the parameter domains
        u1_mid = (u1_range[0] + u1_range[1]) / 2
        v1_mid = (v1_range[0] + v1_range[1]) / 2

        u2_mid = (u2_range[0] + u2_range[1]) / 2
        v2_mid = (v2_range[0] + v2_range[1]) / 2

        intersections = []
        visited=np.zeros((2,2,2,2),dtype=bool)
        # Recursive calls for each pair of subdomains

        s11, s12,s21,s22=subdivide_surface(surface1)
        s31,s32, s41,s42= subdivide_surface(surface2)

        srfs=[[s11, s12],[s21,s22]],[[s31,s32],[s41,s42]
        ]


        for i,sub_u1_range in enumerate([(u1_range[0], u1_mid), (u1_mid, u1_range[1])]):
            for j,sub_v1_range in  enumerate([(v1_range[0], v1_mid), (v1_mid, v1_range[1])]):
                for k,sub_u2_range in  enumerate([(u2_range[0], u2_mid), (u2_mid, u2_range[1])]):
                    for l,sub_v2_range in  enumerate([(v2_range[0], v2_mid), (v2_mid, v2_range[1])]):
                        if visited[i,j,k,l]:
                            pass
                        else:
                            visited[i,j,k,l]=True


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


if __name__ == "__main__":
    from mmcore.geom.surfaces import Surface, Coons

    from mmcore.geom.curves.bspline import NURBSpline

    pts1 = np.array(
        [
            [
                (-6.0558943035701525, -13.657656200983698, 1.0693341635684721),
                (-1.5301574718208828, -12.758430585795727, -2.4497481670182113),
                (4.3625055618617772, -14.490138754852163, -0.052702347089249368),
                (7.7822965141636233, -13.958097981505476, 1.1632592672736894),
            ],
            [
                (7.7822965141636233, -13.958097981505476, 1.1632592672736894),
                (9.3249111495947457, -9.9684277340655711, -2.3272399773510646),
                (9.9156785503454081, -4.4260877770435245, -4.0868275118021469),
                (13.184366571517304, 1.1076098797323481, 0.55039832538794542),
            ],
            [
                (-3.4282810787748206, 2.5976227512567878, -4.1924897351083787),
                (5.7125793432806686, 3.1853804927764848, -3.1997049666908506),
                (9.8891692556257418, 1.2744489476398368, -7.2890391724273922),
                (13.184366571517304, 1.1076098797323481, 0.55039832538794542),
            ],
            [
                (-6.0558943035701525, -13.657656200983698, 1.0693341635684721),
                (-2.1677078000821663, -4.2388638567221646, -3.2149413059589502),
                (-3.5823721281354479, -1.1684651343084738, 3.3563417199639680),
                (-3.4282810787748206, 2.5976227512567878, -4.1924897351083787),
            ],
        ]
    )

    pts2 = np.array(
        [
            [
                (-9.1092663228073292, -12.711321277810857, -0.77093266173210928),
                (-1.5012583168504101, -15.685662924609387, -6.6022178296290024),
                (0.62360921189203689, -15.825362292273830, 2.9177845739234654),
                (7.7822965141636233, -14.858282311330257, -5.1454157090841059),
            ],
            [
                (7.7822965141636233, -14.858282311330257, -5.1454157090841059),
                (9.3249111495947457, -9.9684277340655711, -1.3266123160614773),
                (12.689851531339878, -4.4260877770435245, -8.9585086671785774),
                (10.103825228355211, 1.1076098797323481, -5.6331564229411617),
            ],
            [
                (-5.1868371621186844, 4.7602528056675295, 0.97022697723726137),
                (-0.73355849180427846, 3.1853804927764848, 1.4184540026745367),
                (1.7370638323127894, 4.7726088993795681, -3.7548102282588882),
                (10.103825228355211, 1.1076098797323481, -5.6331564229411617),
            ],
            [
                (-9.1092663228073292, -12.711321277810857, -0.77093266173210928),
                (-3.9344403681487776, -6.6256134176686521, -6.3569364954962628),
                (-3.9413840306534453, -1.1684651343084738, 0.77546233191951042),
                (-5.1868371621186844, 4.7602528056675295, 0.97022697723726137),
            ],
        ]
    )

    patch1 = Coons(*(NURBSpline(pts) for pts in pts1))
    patch2 = Coons(*(NURBSpline(pts) for pts in pts2))
    pts1 = np.array([[-25.0, -25.0, -10.0], [-25.0, -15.0, -5.0], [-25.0, -5.0, 0.0], [-25.0, 5.0, 0.0], [-25.0, 15.0, -5.0],
            [-25.0, 25.0, -10.0], [-15.0, -25.0, -8.0], [-15.0, -15.0, -4.0], [-15.0, -5.0, -4.0], [-15.0, 5.0, -4.0],
            [-15.0, 15.0, -4.0], [-15.0, 25.0, -8.0], [-5.0, -25.0, -5.0], [-5.0, -15.0, -3.0], [-5.0, -5.0, -8.0],
            [-5.0, 5.0, -8.0], [-5.0, 15.0, -3.0], [-5.0, 25.0, -5.0], [5.0, -25.0, -3.0], [5.0, -15.0, -2.0],
            [5.0, -5.0, -8.0], [5.0, 5.0, -8.0], [5.0, 15.0, -2.0], [5.0, 25.0, -3.0], [15.0, -25.0, -8.0],
            [15.0, -15.0, -4.0], [15.0, -5.0, -4.0], [15.0, 5.0, -4.0], [15.0, 15.0, -4.0], [15.0, 25.0, -8.0],
            [25.0, -25.0, -10.0], [25.0, -15.0, -5.0], [25.0, -5.0, 2.0], [25.0, 5.0, 2.0], [25.0, 15.0, -5.0],
            [25.0, 25.0, -10.0]])
    pts1=pts1.reshape((6,len(pts1) // 6, 3))
    pts2 =  np.array([[25.0, 14.774795467423544, 5.5476189978794661], [25.0, 10.618169208735296, -15.132510312735601], [25.0, 1.8288992061686002, -13.545426491756078], [25.0, 9.8715747661086723, 14.261864686419623], [25.0, -15.0, 5.0], [25.0, -25.0, 5.0], [15.0, 25.0, 1.8481369394623908], [15.0, 15.0, 5.0], [15.0, 5.0, -1.4589623860307768], [15.0, -5.0, -1.9177595746260625], [15.0, -15.0, -30.948650572598954], [15.0, -25.0, 5.0], [5.0, 25.0, 5.0], [5.0, 15.0, -29.589097491066767], [3.8028908181980938, 5.0, 5.0], [5.0, -5.0, 5.0], [5.0, -15.0, 5.0], [5.0, -25.0, 5.0], [-5.0, 25.0, 5.0], [-5.0, 15.0, 5.0], [-5.0, 5.0, 5.0], [-5.0, -5.0, -27.394523521151221], [-5.0, -15.0, 5.0], [-5.0, -25.0, 5.0], [-15.0, 25.0, 5.0], [-15.0, 15.0, -23.968082282285287], [-15.0, 5.0, 5.0], [-15.0, -5.0, 5.0], [-15.0, -15.0, -18.334465891060319], [-15.0, -25.0, 5.0], [-25.0, 25.0, 5.0], [-25.0, 15.0, 14.302789083068138], [-25.0, 5.0, 5.0], [-25.0, -5.0, 5.0], [-25.0, -15.0, 5.0], [-25.0, -25.0, 5.0]]

                     )
    pts2=pts2.reshape((6, len(pts2) // 6, 3))


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

    print(pts1.shape,pts2.shape)

    S1 = NURBSSurface(pts1,(3,3))
    S2 = NURBSSurface(pts2,(3,3))
    S1.normalize_knots()
    S2.normalize_knots()
    import time

    s = time.time()
    ints = find_intersections(
        S1,
        (0.0, 1.0),
        (0.0, 1.0),
        S2,
        (0.0, 1.0),
        (0.0, 1.0),
        0.1,
    
    )
    print(time.time() - s)
    pts = []
    curvatures = []
    for i1, i2 in ints:
        pts.append(S1.evaluate_v2(i1,i2).tolist())
    