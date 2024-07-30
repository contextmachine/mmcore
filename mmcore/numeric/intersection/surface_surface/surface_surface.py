import functools
import itertools
import time
import warnings

import numpy as np
from scipy.spatial import KDTree

from mmcore.geom.bvh import BoundingBox, intersect_bvh_objects, BVHNode
from mmcore.geom.surfaces import Surface, Coons
from mmcore.numeric.closest_point import closest_point_on_ray, closest_point_on_surface_batched, \
    closest_point_on_surface, closest_points_on_surface
from mmcore.numeric.algorithms.point_inversion import point_inversion_surface
from mmcore.numeric.intersection.surface_surface._terminator import (
    TerminatorType,
    Terminator,
)

from mmcore.numeric.plane import plane_plane_intersect, plane_plane_plane_intersect

from mmcore.geom.curves.bspline import NURBSpline, interpolate_nurbs_curve
from mmcore.numeric.vectors import (
    scalar_norm,
    scalar_cross,
    scalar_unit,
    det,
    solve2x2,
    norm,
    scalar_dot,
)


def compute_intersection_curvature(
    Su1, Sv1, Suu1, Suv1, Svv1, Su2, Sv2, Suu2, Suv2, Svv2
):
    """
    Compute the curvature of the intersection curve between two parametric surfaces given their partial derivatives.

    Parameters:
    -  Su1, Sv1, Suu1, Suv1, Svv1: partial derivatives of the first surface at the intersection point
    -  Su2, Sv2, Suu2, Suv2, Svv2: partial derivatives of the second surface at the intersection point

    Returns:
    - curvature: curvature of the intersection curve at the given point
    """

    # Compute normal vectors
    N1 = np.array(scalar_cross(Su1, Sv1))
    N1 /= scalar_norm(N1)
    N2 = np.array(scalar_cross(Su2, Sv2))
    N2 /= scalar_norm(N2)

    # Check if the surfaces intersect tangentially
    cos_theta = scalar_dot(N1, N2)
    if np.isclose(np.abs(cos_theta), 1):
        raise ValueError("The surfaces intersect tangentially at the given point")

    # Compute tangent vector of the intersection curve
    T = np.array(scalar_cross(N1, N2))
    T /= scalar_norm(T)
    uSu1 = scalar_dot(scalar_unit(Su1), T)
    uSu2 = scalar_dot(scalar_unit(Su2), T)
    uSv1 = scalar_dot(scalar_unit(Sv1), T)
    uSv2 = scalar_dot(scalar_unit(Sv2), T)
    # Compute the curvature vector
    L1 = scalar_dot(scalar_unit(Suu1), N1)
    M1 = scalar_dot(scalar_unit(Suv1), N1)
    N1_2 = scalar_dot(scalar_unit(Svv1), N1)
    L2 = scalar_dot(scalar_unit(Suu2), N2)
    M2 = scalar_dot(scalar_unit(Suv2), N2)
    N2_2 = scalar_dot(scalar_unit(Svv2), N2)
    k1 = L1 * uSu1**2 + 2 * M1 * uSu1 * uSv1 + N1_2 * (uSv1**2)
    k2 = L2 * uSu2**2 + 2 * M2 * uSu2 * uSv2 + N2_2 * (uSv2**2)
    curvature_vector = (k1 * N1 + k2 * N2) / (1 - cos_theta**2)

    # Compute the curvature magnitude
    # curvature = np.linalg.norm(curvature_vector)

    return curvature_vector, T


def get_plane(origin, du, dv):
    duu = du / scalar_norm(du)

    dn = scalar_unit(scalar_cross(duu, dv))
    dvu = scalar_cross(dn, duu)
    return np.array([origin, duu, dvu, dn])


def get_normal(du, dv):
    duu = scalar_unit(du)
    dn = scalar_unit(scalar_cross(duu, dv))

    return duu, dn


def improve_uv(du, dv, xyz_old, xyz_better):
    dxdu, dydu, dzdu = du
    dxdv, dydv, dzdv = dv

    delta = xyz_better - xyz_old

    xy = np.array([[dxdu, dxdv], [dydu, dydv]]), [delta[0], delta[1]]
    xz = np.array([[dxdu, dxdv], [dzdu, dzdv]]), [delta[0], delta[2]]
    yz = np.array([[dydu, dydv], [dzdu, dzdv]]), [delta[1], delta[2]]

    max_det = max([xy, xz, yz], key=lambda Ab: det(Ab[0]))
    res = np.zeros(2)
    solve2x2(max_det[0], np.array(max_det[1]), res)
    return res


def freeform_step(pt1, pt2, du1, dv1, du2, dv2):
    pl1, pl2 = get_plane(pt1, du1, dv1), get_plane(pt2, du2, dv2)
    ln = np.array(plane_plane_intersect(pl1, pl2))

    np1 = np.asarray(closest_point_on_ray((ln[0], ln[1]), pt1))
    np2 = np.asarray(closest_point_on_ray((ln[0], ln[1]), pt2))

    return np1, np1 + (np2 - np1) / 2, np2


def freeform_method(s1, s2, uvb1, uvb2, tol=1e-6, cnt=0, max_cnt=200):
    pt1 = s1.evaluate(uvb1)
    pt2 = s2.evaluate(uvb2)
    if scalar_norm(pt1 - pt2) < tol:
        return (pt1, uvb1), (pt2, uvb2)
    du1 = s1.derivative_u(uvb1)
    dv1 = s1.derivative_v(uvb1)

    du2 = s2.derivative_u(uvb2)
    dv2 = s2.derivative_v(uvb2)

    p1_better, xyz_better, p2_better = freeform_step(pt1, pt2, du1, dv1, du2, dv2)

    if xyz_better is None:
        return

    uvb1_better = uvb1 + improve_uv(du1, dv1, pt1, xyz_better)
    uvb2_better = uvb2 + improve_uv(du2, dv2, pt2, xyz_better)

    if (
        any(uvb1_better < 0.0)
        or any(uvb2_better < 0.0)
        or any(uvb1_better > 1.0)
        or any(uvb2_better > 1.0)
    ):
        return None

    else:
        if cnt < max_cnt:
            return freeform_method(s1, s2, uvb1_better, uvb2_better, tol, cnt + 1)
        else:
            warnings.warn("freeform not convergence")
            return


def find_uv_intersection(us, vs, ut, vt, u0, v0, u1, v1):
    def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
        # Calculate the denominator
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

        if denom == 0:
            return None  # Lines are parallel

        # Calculate the numerators
        ua_num = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
        ub_num = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)

        ua = ua_num / denom
        ub = ub_num / denom

        # Check if the intersection is within the line segments
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            intersection_x = x1 + ua * (x2 - x1)
            intersection_y = y1 + ua * (y2 - y1)
            return (intersection_x, intersection_y)

        return None

    # Define the boundaries as line segments
    boundaries = [
        (u0, v0, u1, v0),  # Bottom boundary
        (u1, v0, u1, v1),  # Right boundary
        (u1, v1, u0, v1),  # Top boundary
        (u0, v1, u0, v0),  # Left boundary
    ]

    for ua, va, ub, vb in boundaries:
        intersection = line_intersection(us, vs, ut, vt, ua, va, ub, vb)
        if intersection:
            return intersection

    return None


@functools.lru_cache(maxsize=None)
def solve_step(r, tol):
    return np.sqrt(r**2 - (r - tol) ** 2) * 2


def solve_marching(
    pt1,
    pt2,
    du1,
    dv1,
    duu1,
    duv1,
    dvv1,
    du2,
    dv2,
    duu2,
    duv2,
    dvv2,
    tol,
    side=1,
    curvature_step=False,
):
    pl1, pl2 = get_plane(pt1, du1, dv1), get_plane(pt2, du2, dv2)

    marching_direction = np.array(scalar_unit(scalar_cross(pl1[-1], pl2[-1])))

    # tng = np.zeros((2, 3))
    # calgorithms.evaluate_curvature(marching_direction * side, pl1[-1], tng[0], tng[1])
    # marching_direction=tng[0]
    # K = tng[1]
    if not curvature_step:
        r = 1
        step = solve_step(r, tol)
    else:
        curvature_vector, tangent_vector = compute_intersection_curvature(
            du1, dv1, duu1, duv1, dvv1, du2, dv2, duu2, duv2, dvv2
        )

        # print(scalar_norm(curvature_vector))
        # r = 1 / (scalar_norm(K))
        r = 1 / np.sqrt(scalar_norm(curvature_vector))
        # print(r,1/r)
        # r=1/r
        # print((r**2)-(r-tol)**2)

        step = np.sqrt(r**2 - (r - tol) ** 2) * 2
        # print(step)
    new_pln = np.array(
        [pt1 + marching_direction * side * step, pl1[-1], pl2[-1], marching_direction]
    )

    return np.array(plane_plane_plane_intersect(pl1, pl2, new_pln)), step


def marching_step(
    s1: Surface,
    s2,
    uvb1,
    uvb2,
    tol,
    cnt=0,
    side=1,
    curvature_step=False,
    boundary_terminators: Terminator = None,
):
    pt1, xax1, yax1, n1 = pl1 = s1.plane_at(uvb1)
    pt2, xax2, yax2, n2 = pl2 = s2.plane_at(uvb2)

    du1 = s1.derivative_u(uvb1)
    dv1 = s1.derivative_v(uvb1)
    du2 = s2.derivative_u(uvb2)
    dv2 = s2.derivative_v(uvb2)
    T = np.array(scalar_cross(n1, n2))
    T /= np.linalg.norm(T)

    def sectional_tangent(veps):
        uv1_new = uvb1 + improve_uv(du1, dv1, pt1, pt1 + veps)
        uv2_new = uvb2 + improve_uv(du2, dv2, pt2, pt2 + veps)
        n1, n2 = s1.normal(uv1_new), s2.normal(uv2_new)
        Tn = np.array(scalar_cross(n1, n2))
        Tn /= np.linalg.norm(Tn)
        return Tn

    vesp = T * 1e-3
    curvature_vector = (sectional_tangent(vesp) - T) / (np.linalg.norm(vesp))

    # tangent, curvature_vector = sectional_curvature_vector(s1, s2, pt1, eps=1e-5, tol=tol)

    K = np.linalg.norm(curvature_vector)

    r = 1 / K

    # print(r,1/r)
    # r=1/r
    # print((r**2)-(r-tol)**2)

    step = np.sqrt(r**2 - (r - tol) ** 2) * 2

    new_pln = np.array([(pt1 + pt2) / 2 + T * side * step, n1, n2, T])
    # print(step)

    xyz_better = np.array(plane_plane_plane_intersect(pl1, pl2, new_pln))

    # print([[xyz_better.tolist()],pl1.tolist(),pl2.tolist(),new_pln.tolist()], end=',')
    if xyz_better is None:
        return
    uvb1_better = uvb1 + improve_uv(du1, dv1, pt1, xyz_better)
    uvb2_better = uvb2 + improve_uv(du2, dv2, pt2, xyz_better)

    if (
        any(uvb1_better < 0.0)
        or any(uvb2_better < 0.0)
        or any(uvb1_better > 1.0)
        or any(uvb2_better > 1.0)
    ):
        if boundary_terminators is not None:
            xyz_better, uvb1_better, uvb2_better = boundary_terminators.get_closest(
                xyz_better
            )


        # uvb1_better,uvb2_better= constrained_uv_int(s1, s2, uvb1, uvb1_better, uvb2, uvb2_better, pt1,pt2,tol)

        # xyz_better = (s1.evaluate(uvb1_better) + s2.evaluate(uvb2_better)) / 2
        # uvb1_better = np.clip(uvb1_better, 0., 1.)
        # uvb2_better = np.clip(uvb2_better, 0., 1.)

        return (
            (xyz_better, uvb1_better),
            (xyz_better, uvb2_better),
            step,
            TerminatorType.EDGE,
        )
    res = freeform_method(s1, s2, uvb1_better, uvb2_better, tol)
    if res is None:
        return
    else:
        (xyz1_better, uvb1_better), (xyz2_better, uvb2_better) = res
        return (
            ((xyz1_better + xyz2_better) * 0.5, uvb1_better),
            ((xyz1_better + xyz2_better) * 0.5, uvb2_better),
            step,
            TerminatorType.STEP,
        )


def constrained_uv_int(surf1, surf2, uv11, uv12, uv21, uv22, pt1, pt2, tol):
    (u0, u1), (v0, v1) = surf1.interval()
    res1 = find_uv_intersection(*uv11, *uv12, u0, v0, u1, v1)
    (u0, u1), (v0, v1) = surf2.interval()
    res2 = find_uv_intersection(*uv21, *uv22, u0, v0, u1, v1)
    if res1 is None and res2 is None:
        raise ValueError("No intersection between uv1 and uv2")
    elif res1 is not None and res2 is not None:
        uv1 = res1
        uv2 = res2
        p1n = surf1.evaluate(uv1)
        p2n = surf2.evaluate(uv2)
        t1 = p1n - pt1
        t2 = p2n - pt2

        if np.linalg.norm(t1) <= np.linalg.norm(t2):
            uv2 = point_inversion_surface(surf2, p1n, *uv2, tol1=1e-6, tol2=tol)
        else:
            uv1 = point_inversion_surface(surf1, p2n, *uv1, tol1=1e-6, tol2=tol)

        return uv1, uv2

    elif res1 is not None:
        uv1 = res1
        p1n = surf1.evaluate(uv1)

        # uv2 = newtons_method(wrp1, uv22)
        uv2 = point_inversion_surface(surf2, p1n, *uv22, tol1=1e-6, tol2=tol)

        return uv1, uv2

    elif res2 is not None:
        uv2 = res2
        p2n = surf2.evaluate(uv2)

        uv1 = point_inversion_surface(surf1, p2n, *uv12, tol1=1e-6, tol2=tol)

        return uv1, uv2

    else:
        raise ValueError("No intersection between uv1 and uv2")
    # xyz_second_better = surf2.evaluate(np.array(uv_second)) - pt2

    # xyz3 = min(((xyz_second_better,pt1), (xyz_first_better, pt2)), key=lambda x: scalar_norm(x[0]))


def marching_method(
    s1,
    s2,
    initial_uv1,
    initial_uv2,
    kd=None,
    tol=1e-3,
    max_iter=500,
    no_ff=False,
    side=1,
    curvature_step=False,
    boundary_terminators=None,
):
    terminator = None
    use_kd = kd is not None
    ixss = set()
    xyz1_init, xyz2_init = s1.evaluate(initial_uv1), s2.evaluate(initial_uv2)
    res = marching_step(
        s1,
        s2,
        initial_uv1,
        initial_uv2,
        tol=tol,
        side=side,
        curvature_step=curvature_step,
        boundary_terminators=boundary_terminators
    )
    if res is None:
        terminator = TerminatorType.EDGE
        return

    (xyz1, uv1_new), (xyz2, uv2_new), step, terminator = res
    if use_kd:
        ixss.update(kd.query_ball_point(xyz1_init, step * 2))
    # print()

    uvs = [(uv1_new, uv2_new)]
    pts = [xyz1]

    # print(uv1_new, uv2_new)

    steps = [step]
    if terminator is TerminatorType.EDGE:
        return uvs, pts, steps, list(ixss), terminator
    for i in range(max_iter):
        uv1, uv2 = uv1_new, uv2_new

        res = marching_step(
            s1, s2, uv1, uv2, tol=tol, side=side, curvature_step=curvature_step,boundary_terminators=boundary_terminators
        )
        if res is None:
            terminator = TerminatorType.EDGE
            break
        (xyz1, uv1_new), (xyz2, uv2_new), step, terminator = res
        pts.append(xyz1)
        uvs.append((uv1_new, uv2_new))
        steps.append(step)
        if use_kd:
            # print(   len(kd.data),ixss,step,tol,kd.query(xyz1, 3))

            ixss.update(kd.query_ball_point(xyz1, step * 2))
            # print(ixss)

        if terminator is TerminatorType.EDGE:
            break

        if scalar_norm(xyz1 - xyz1_init) < step:
            # print("b", xyz1_init, xyz1, np.linalg.norm(xyz1 - xyz1_init), step)
            pts.append(pts[0])

            uvs.append((initial_uv1, initial_uv1))
            terminator = TerminatorType.LOOP
            break

        # print("I", np.linalg.norm(xyz1 - xyz1_init), step*2)

    # print(len(pts))
    return uvs, pts, steps, list(ixss), terminator


from collections import namedtuple

SurfaceStuff = namedtuple("SurfaceStuff", ["surf", "kd", "pts", "uv", "bbox"])
ClosestSurfaces = namedtuple("ClosestSurfaces", ["a", "b"])


def find_closest_points(surf1: Surface, surf2: Surface, tol=1e-3):
    min1max1: BoundingBox = surf1.tree.bounding_box
    min2max2: BoundingBox = surf2.tree.bounding_box

    if min1max1.intersect(min2max2):
        pts = []
        uvs1 = []
        uvs2 = []

        for first, second in intersect_bvh_objects(surf1.tree, surf2.tree):
            first: BVHNode
            second: BVHNode
            # bb=first.bounding_box.intersection(second.bounding_box)
            uv1 = np.clip(np.average(first.object.uvs, axis=0), 0.0, 1.0)
            uv2 = np.clip(np.average(second.object.uvs, axis=0), 0.0, 1.0)

            if any([any(uv1 < 0), any(uv1 > 1), any(uv2 < 0), any(uv2 > 1)]):
                pass
            else:
                # print(uv1, uv2)

                res = freeform_method(surf1, surf2, uv1, uv2, tol=tol)

                if res is not None:
                    (xyz1_new, uvb1_better), (xyz2_new, uvb2_better) = res
                    if any(
                        [
                            any(uvb1_better < 0),
                            any(uvb1_better > 1),
                            any(uvb2_better < 0),
                            any(uvb2_better > 1),
                        ]
                    ):
                        pass
                    # print(uvb1_better, uvb2_better)
                    pts.append(xyz1_new)
                    uvs1.append(uvb1_better)
                    uvs2.append(uvb2_better)
        # print(np.array(pts).tolist())
        return KDTree(np.array(pts)), np.array(uvs1), np.array(uvs2)


from mmcore.numeric.intersection.surface_surface._terminator import surface_surface_intersection_edge_terminator


def surface_ppi(
    surf1: Surface, surf2: Surface, tol=0.1, max_iter=500, curvature_step=False
):
    res = find_closest_points(surf1, surf2, tol=tol)
    if res is None:
        return
    edge_terminator = surface_surface_intersection_edge_terminator(surf1, surf2
                                                                   )


    kd, uvs1, uvs2 = res

    curves = []
    curves_uvs = []
    terminators = []
    stepss = []
    data = kd.data
    ii = 0
    l = len(uvs2)

    def _next():
        nonlocal ii, kd, data, uvs1, uvs2, l

        ress = marching_method(
            surf1,
            surf2,
            uvs1[0],
            uvs2[0],
            kd=kd,
            tol=tol,
            side=1,
            curvature_step=curvature_step,
            boundary_terminators=edge_terminator,
        )
        ii += 1
        if ress is not None:
            start = np.copy(data[0])
            start_uv = np.array([np.copy(uvs1[0]), np.copy(uvs2[0])])

            if ress[-1] != TerminatorType.LOOP:
                ress_back = marching_method(
                    surf1,
                    surf2,
                    uvs1[0],
                    uvs2[0],
                    kd=kd,
                    tol=tol,
                    side=-1,
                    curvature_step=curvature_step,
                    boundary_terminators=edge_terminator,
                )

                if ress_back is not None:
                    uv_s, pts, steps, ixss, terminator = ress
                    uvsb, ptsb, stepsb, ixssb, terminator_back = ress_back
                    rmv = np.unique(ixss + ixssb)
                    data = np.delete(data, rmv, axis=0)
                    uvs1 = np.delete(uvs1, rmv, axis=0)
                    uvs2 = np.delete(uvs2, rmv, axis=0)
                    if len(data.shape) == 2:
                        kd = KDTree(data)
                    else:
                        kd = None
                    terminators.append([terminator_back, terminator])
                    curves_uvs.append(
                        list(itertools.chain(reversed(uvsb), [start_uv], uv_s))
                    )
                    stepss.append(list(itertools.chain(reversed(stepsb), steps)))

                    return list(itertools.chain(reversed(ptsb), [start], pts))

            uv_s, pts, steps, ixss, terminator = ress
            rmv = np.array(ixss, dtype=int)
            data = np.delete(data, rmv, axis=0)
            uvs1 = np.delete(uvs1, rmv, axis=0)
            uvs2 = np.delete(uvs2, rmv, axis=0)
            if len(data.shape) == 2:
                kd = KDTree(data)
            else:
                kd = None
            terminators.append([terminator])
            curves_uvs.append([start_uv] + list(uv_s))
            stepss.append(steps)
            return [start] + list(pts)
        else:
            uvs1 = np.delete(uvs1, 0, axis=0)
            uvs2 = np.delete(uvs2, 0, axis=0)
            data = np.delete(data, 0, axis=0)
            if len(data.shape) == 2:
                kd = KDTree(data)
            else:
                kd = None

    for i in range(max_iter):
        if data.size == 0:
            break
        if ii >= l:
            break

        res = _next()
        if res is None:
            continue
        else:
            curves.append(res)

    return (
        curves,
        [
            [np.array(crv, dtype=float) for crv in zip(*curve_uv)]
            for curve_uv in curves_uvs
        ],
        stepss,
        terminators,
    )


from mmcore.geom.surfaces import CurveOnSurface, Surface


def surface_intersection(
    surf1: Surface,
    surf2: Surface,
    tol: float = 0.01,
    max_iter: int = 500,
    curvature_step=False,
) -> list[tuple[NURBSpline, CurveOnSurface, CurveOnSurface]]:
    """
    Calculate the intersection of two parametric surfaces.

    :param surf1: The first surface.
    :type surf1: Surface
    :param surf2: The second surface.
    :type surf2: Surface
    :param tol: The tolerance value for the intersection algorithm (optional, default is 0.01).
    :type tol: float
    :param max_iter: The maximum number of iterations for the intersection algorithm (optional, default is 500). Now
    this parameter exists primarily to debug recursion
    :type max_iter: int
    :param curvature_step:  Use curvature dependent step (experimental, default is False). At the moment it does not give an increase in speed.
    :type curvature_step: bool
    :return: A list of tuples, where each tuple contains an interpolated spatial NURBS curve intersection and the corresponding objects
             CurveOnSurface objects for surf1 and surf2.
    :rtype: list[tuple[NURBSpline, CurveOnSurface, CurveOnSurface]]

    Note
    -----
    If successful (intersection found), this function returns a list of intersection results because two surfaces can form as many separate intersection curves as desired.

     Since two surfaces can form as many separate intersection curves as desired, the list of intersection results.
     Each intersection result is a separate intersection curve in three views:
        1. A spatial NURBS curve (NURBSpline object).
        2. A curve in the parametric space of the first surface (CurveOnSurface object).
        3. A curve in the parametric space of the second surface (CurveOnSurface

    """
    res = surface_ppi(
        surf1, surf2, tol=tol, max_iter=max_iter, curvature_step=curvature_step
    )
    if res is None:
        return []

    curves, curves_uvs, stepss, terminators = res
    results = []
    for i, curve_pts in enumerate(curves):
        curve = interpolate_nurbs_curve(curve_pts, 3)

        if all([terminator is TerminatorType.LOOP for terminator in terminators[i]]):
            curve_on_surf1 = interpolate_nurbs_curve(curves_uvs[i][0][:-1], 3)
            curve_on_surf2 = interpolate_nurbs_curve(curves_uvs[i][1][:-1], 3)
            curve.make_periodic()
            curve_on_surf1.make_periodic()
            curve_on_surf2.make_periodic()
        else:
            curve_on_surf1 = interpolate_nurbs_curve(curves_uvs[i][0], 3)
            curve_on_surf2 = interpolate_nurbs_curve(curves_uvs[i][1], 3)
        results.append(
            (
                curve,
                CurveOnSurface(
                    surf1, curve_on_surf1, interval=curve_on_surf1.interval()
                ),
                CurveOnSurface(
                    surf2, curve_on_surf2, interval=curve_on_surf2.interval()
                ),
            )
        )

    return results


if __name__ == "__main__":
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
    # with open('tests/patch1.txt') as f:
    #    pts1=np.array(eval(f.read()))
    # with open('tests/patch2.txt') as f:
    #    pts2=np.array(eval(f.read()))
    # with open('tests/coons1.pkl', 'rb') as f:
    #    patch1 = dill.load(f)

    # with open('tests/coons2.pkl', 'rb') as f:
    #    patch2 = dill.load(f)

    patch1 = Coons(*(NURBSpline(pts) for pts in pts1))
    patch2 = Coons(*(NURBSpline(pts) for pts in pts2))
    # from mmcore.geom.curves.cubic import CubicSpline
    # patch1 = Coons(*(CubicSpline(*pts) for pts in pts1))
    # patch2 = Coons(*(CubicSpline(*pts) for pts in pts2))
    patch1.build_tree(10, 10)
    patch2.build_tree(10, 10)
    #print(
    #    patch1._rc,
    #)

    # yappi.set_clock_type("wall")  # Use set_clock_type("wall") for wall time
    # yappi.start()
    s = time.time()
    TOL = 0.001
    cc = surface_ppi(patch1, patch2, TOL)
    print(time.time() - s)
    # yappi.stop()
    # func_stats = yappi.get_func_stats()
    # func_stats.save(f"{__file__.replace('.py', '')}_{int(time.time())}.pstat", type='pstat')

    # tolerance checks

    pts_crv_1=np.array(cc[0][0])
    pts_crv_2 = np.array(cc[0][1])

    nrm=[np.all(
        np.linalg.norm(patch1(closest_points_on_surface(patch1, pts_crv_1))-pts_crv_1,axis=1)),
    np.all(
        np.linalg.norm(patch1(closest_points_on_surface(patch1, pts_crv_2)) - pts_crv_2, axis=1)),
        np.all(
            np.linalg.norm(patch2(closest_points_on_surface(patch2, pts_crv_1)) - pts_crv_1, axis=1)),
            np.all(
                np.linalg.norm(patch2(closest_points_on_surface(patch2, pts_crv_2)) - pts_crv_2, axis=1))]
    print(      all(nrm))



    print([np.array(c).tolist() for c in cc[0]])

    # print([patch1(uvs(20, 20)).tolist(), patch2(uvs(20, 20)).tolist()])
    res = surface_intersection(patch1, patch2, TOL)
