from mmcore.geom.surfaces.fundamental_forms import classify_point_on_surface


import numpy as np

from mmcore.numeric.aabb import aabb
from mmcore.numeric.vectors import scalar_dot



def find_intersections(
    surface1,
    u1_range,
    v1_range,
    surface2,
    u2_range,
    v2_range,
    tol=0.1,
    ptol=1e-4,
    htol=1e-2,
):
    def inner(u1_range, v1_range, u2_range, v2_range, step=0):
        if (u1_range[1] - u1_range[0]) < tol and (v1_range[1] - v1_range[0]) < ptol:
            u1_mid = (u1_range[0] + u1_range[1]) / 2
            v1_mid = (v1_range[0] + v1_range[1]) / 2
            u2_mid = (u2_range[0] + u2_range[1]) / 2
            v2_mid = (v2_range[0] + v2_range[1]) / 2
            return [((u1_mid, v1_mid), (u2_mid, v2_mid))]

        # bbox_mid_min,bbox_mid_max= np.array(bounding_boxes_intersection((bbox1_min, bbox1_max), (bbox2_min, bbox2_max)))

        # Check stopping criterion

        # Return a representative point (e.g., midpoint)

        # This is a candidate intersection point

        # Otherwise, subdivide the parameter domains
        u1_mid = (u1_range[0] + u1_range[1]) / 2
        v1_mid = (v1_range[0] + v1_range[1]) / 2

        u2_mid = (u2_range[0] + u2_range[1]) / 2
        v2_mid = (v2_range[0] + v2_range[1]) / 2

        intersections = []
        #
        ## Recursive calls for each pair of subdomains
        #
        fb = [
            ((u1_range[0], u1_mid), (v1_range[0], v1_mid)),
            ((u1_mid, u1_range[1]), (v1_range[0], v1_mid)),
            ((u1_mid, u1_range[1]), (v1_mid, v1_range[1])),
            ((u1_range[0], u1_mid), (v1_mid, v1_range[1])),
        ]
        sb = [
            ((u2_range[0], u1_mid), (v2_range[0], v1_mid)),
            ((u2_mid, u2_range[1]), (v2_range[0], v1_mid)),
            ((u2_mid, u2_range[1]), (v2_mid, v2_range[1])),
            ((u2_range[0], u2_mid), (v2_mid, v2_range[1])),
        ]
        bboxes1 = []
        bboxes2 = []
        for i in range(4):
            bboxes1.append(bbox(surface1, *fb[i][0], *fb[i][1]))
            bboxes2.append(bbox(surface2, *sb[i][0], *sb[i][1]))
        #

        ixs = np.zeros((4, 4), dtype=bool)
        visited = np.zeros((4, 4), dtype=bool)
        candidates = []
        for i in range(4):
            for j in range(4):
                if not visited[i, j]:
                    visited[i, j] = visited[j, i] = True
                    if np.any(ixs[i]) and step>=2:
                        continue
                    else:
                        res = bounding_boxes_intersection(bboxes1[i], bboxes2[j])
                        ixs[i, j] = res
                        if res:
                            candidates.append((i, j))

        # print(visited)
        # print(ixs)
        # print(candidates)
        #
        #
        #
        # [(u1_range[0], u1_mid), (u1_mid, u1_range[1])]
        # [(v1_range[0], v1_mid), (u1_mid, u1_range[1])]
        for i, j in candidates:
            bbox1_min, bbox1_max = bboxes1[i]
            bbox2_min, bbox2_max = bboxes2[j]

            bbox_int_min = np.maximum(bbox1_min, bbox2_min)
            bbox_int_max = np.minimum(bbox1_max, bbox2_max)
            direction = bbox_int_max - bbox_int_min

            dst = scalar_dot(direction, direction)
            if (dst < tol) or (direction[-1] < htol):
                u1_mid = (u1_range[0] + u1_range[1]) / 2
                v1_mid = (v1_range[0] + v1_range[1]) / 2
                u2_mid = (u2_range[0] + u2_range[1]) / 2
                v2_mid = (v2_range[0] + v2_range[1]) / 2
                intersections.append(((u1_mid, v1_mid), (u2_mid, v2_mid)))
            else:
                intersections.extend(inner(*fb[i], *sb[j], step + 1))

            # intersections.extend(
            #    find_intersections(surface1, sub_u1_range, sub_v1_range, surface2, sub_u2_range, sub_v2_range,
            #                       tol, ptol, htol)
            # )
        # (candidates)
        return intersections

    # Compute bounding boxes
    bbox1_min, bbox1_max = bbox(surface1, *u1_range, *v1_range)
    bbox2_min, bbox2_max = bbox(surface2, *u2_range, *v2_range)

    # Check if bounding boxes intersect
    if not bounding_boxes_intersect(bbox1_min, bbox1_max, bbox2_min, bbox2_max):
        return []  # No intersection in this subdivision
    else:
        return inner(u1_range, v1_range, u2_range, v2_range, step=1)


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


def bbox(surface, u0, u1, v0, v1):
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


    import time

    s = time.time()
    ints = find_intersections(
        patch1,
        (0.0, 1.0),
        (0.0, 1.0),
        patch2,
        (0.0, 1.0),
        (0.0, 1.0),
        0.1,
        0.001,
        0.01,
    )
    print(time.time() - s)
    pts = []
    curvatures = []
    for i1, i2 in ints:
        pts.append((patch1.evaluate_v2(*i1).tolist(), patch2.evaluate_v2(*i2).tolist()))
        curvatures.append(
            (
                int(classify_point_on_surface(patch1, *i1)),
                int(classify_point_on_surface(patch2, *i2)),
            )
        )
