import functools
import itertools
import time
import warnings

from scipy.spatial import KDTree

from mmcore.geom.bvh import BoundingBox, intersect_bvh_objects, BVHNode
from mmcore.geom.surfaces import Surface, Coons
from mmcore.numeric.closest_point import closest_point_on_ray, closest_points_on_surface

from mmcore.numeric.plane import plane_plane_intersect

from mmcore.geom.curves.bspline import NURBSpline, interpolate_nurbs_curve

import numpy as np
from mmcore.numeric.vectors import det, solve2x2
from collections import namedtuple
from typing import NamedTuple

from mmcore.numeric import scalar_cross, scalar_norm
from mmcore.numeric.vectors import scalar_unit

from mmcore.numeric.fdm import DEFAULT_H
from mmcore.numeric.intersection.surface_surface._terminator import TerminatorType

from mmcore.numeric.plane import plane_plane_plane_intersect_points_and_normals

SurfaceDerivativesData = namedtuple('SurfaceDerivatives', ['uv', 'pt', 'du', 'dv', 'normal'])


class IntersectionStepData(NamedTuple):
    first: SurfaceDerivativesData
    second: SurfaceDerivativesData
    tangent: np.ndarray


class MarchingMethod:

    def __init__(self, s1,
                 s2,
                 kd=None,
                 tol=1e-3,
                 max_iter=500,
                 side=1,

                 boundary_terminators=None,
                 fdm_h=DEFAULT_H
                 ):
        self.s1 = s1
        self.s2 = s2

        self.kd = kd
        self.tol = tol
        self.max_iter = max_iter
        self.side = side
        self.boundary_terminators = boundary_terminators
        self.fdm_h = fdm_h

    @staticmethod
    def calculate_derivatives_data(surface, uv):
        # TODO: вероятно нужно сделать оптимизированную версию для некоторых примитивов дальше использовать plane_at, а если его нет то вычислять самостоятельно

        pt = surface.evaluate(uv)
        du = surface.derivative_u(uv)
        dv = surface.derivative_v(uv)
        n = np.array(scalar_unit(scalar_cross(du, dv)))

        return SurfaceDerivativesData(uv, pt, du, dv, n)

    @classmethod
    def calculate_intersection_step_data(cls, surface1, surface2, uv1,
                                         uv2) -> IntersectionStepData:
        s1_data = cls.calculate_derivatives_data(surface1, uv1)
        s2_data = cls.calculate_derivatives_data(surface2, uv2)
        T = np.array(scalar_cross(s1_data.normal, s2_data.normal))
        T /= np.linalg.norm(T)
        return IntersectionStepData(s1_data, s2_data, T)

    def step(self, uv1, uv2):
        intersection_step_data: IntersectionStepData = self.calculate_intersection_step_data(self.s1, self.s2, uv1, uv2)
        curvature_vector = self.calculate_sectional_curvature(intersection_step_data)
        step = self.calculate_step(curvature_vector)
        pt_better = self.refine_point(intersection_step_data, step=step)
        if pt_better is None:
            return
        uv1_better, uv2_better = self.improve_uvs(intersection_step_data, pt_better)

        is_edge, (pt_better, uv1_better, uv2_better) = self.handle_edge(pt_better, uv1_better, uv2_better)
        if is_edge:
            return (pt_better, uv1_better), (pt_better, uv2_better), step, TerminatorType.EDGE
        return (pt_better, uv1_better), (pt_better, uv2_better), step, TerminatorType.STEP

    def calculate_step(self, curvature_vector):
        K = np.linalg.norm(curvature_vector)

        r = 1 / K
        step = np.sqrt(r ** 2 - (r - self.tol) ** 2) * 2
        return step

    def calculate_new_plane(self, ders: IntersectionStepData, step):
        pt1, n1 = ders.first.pt, ders.first.normal
        pt2, n2 = ders.second.pt, ders.second.normal
        T = ders.tangent
        new_pln = np.array([(pt1 + pt2) / 2 + T * self.side * step, n1, n2, T])
        return new_pln

    def calculate_sectional_curvature(self, step_data: IntersectionStepData):
        def sectional_tangent(veps):
            uv1_new = step_data.first.uv + improve_uv(step_data.first.du, step_data.first.dv, step_data.first.pt,
                                                      step_data.first.pt + veps)
            uv2_new = step_data.second.uv + improve_uv(step_data.second.du, step_data.second.dv, step_data.second.pt,
                                                       step_data.second.pt + veps)

            n1, n2 = self.s1.normal(uv1_new), self.s2.normal(uv2_new)
            Tn = np.array(scalar_cross(n1, n2))
            Tn /= np.linalg.norm(Tn)
            return Tn

        vesp = step_data.tangent * self.fdm_h
        curvature_vector = (sectional_tangent(vesp) - step_data.tangent) / (np.linalg.norm(vesp))

        return curvature_vector

    def refine_point(self, ders: IntersectionStepData, step: float):
        new_pln = self.calculate_new_plane(ders, step=step)
        return np.array(plane_plane_plane_intersect_points_and_normals(ders.first.pt,
                                                                       ders.first.normal,
                                                                       ders.second.pt,
                                                                       ders.second.normal,
                                                                       new_pln[0],
                                                                       new_pln[-1]
                                                                       )

                        )

    @staticmethod
    def improve_uvs(step_data: IntersectionStepData, pt_better: np.ndarray):
        uvb1_better = step_data.first.uv + improve_uv(step_data.first.du, step_data.first.dv, step_data.first.pt,
                                                      pt_better)
        uvb2_better = step_data.second.uv + improve_uv(step_data.second.du, step_data.second.dv, step_data.second.pt,
                                                       pt_better)
        return uvb1_better, uvb2_better

    def handle_edge(self, xyz_better, uvb1_better, uvb2_better):

        if (
                any(uvb1_better < 0.0)
                or any(uvb2_better < 0.0)
                or any(uvb1_better > 1.0)
                or any(uvb2_better > 1.0)
        ):
            if self.boundary_terminators is not None:
                xyz_better, uvb1_better, uvb2_better = self.boundary_terminators.get_closest(
                    xyz_better
                )

            # uvb1_better,uvb2_better= constrained_uv_int(s1, s2, uvb1, uvb1_better, uvb2, uvb2_better, pt1,pt2,tol)

            # xyz_better = (s1.evaluate(uvb1_better) + s2.evaluate(uvb2_better)) / 2
            # uvb1_better = np.clip(uvb1_better, 0., 1.)
            # uvb2_better = np.clip(uvb2_better, 0., 1.)

            return True, (xyz_better, uvb1_better, uvb2_better)

        return False, (xyz_better, uvb1_better, uvb2_better)

    def solve(self, initial_uv1, initial_uv2

              ):
        terminator = None
        use_kd = self.kd is not None
        ixss = set()
        xyz1_init, xyz2_init = self.s1.evaluate(initial_uv1), self.s2.evaluate(initial_uv2)

        res = self.step(

            initial_uv1,
            initial_uv2
        )

        if res is None:
            terminator = TerminatorType.EDGE
            return

        (xyz1, uv1_new), (xyz2, uv2_new), step, terminator = res
        if use_kd:
            ixss.update(self.kd.query_ball_point(xyz1_init, step * 2))

        uvs = [(uv1_new, uv2_new)]
        pts = [xyz1]

        steps = [step]
        if terminator is TerminatorType.EDGE:
            return uvs, pts, steps, list(ixss), terminator
        for i in range(self.max_iter):
            uv1, uv2 = uv1_new, uv2_new

            res = self.step(
                uv1, uv2
            )
            if res is None:
                terminator = TerminatorType.EDGE
                break
            (xyz1, uv1_new), (xyz2, uv2_new), step, terminator = res
            pts.append(xyz1)
            uvs.append((uv1_new, uv2_new))
            steps.append(step)
            if use_kd:
                ixss.update(self.kd.query_ball_point(xyz1, step * 2))

            if terminator is TerminatorType.EDGE:
                break

            if scalar_norm(xyz1 - xyz1_init) < step:
                pts.append(pts[0])

                uvs.append((initial_uv1, initial_uv2))
                terminator = TerminatorType.LOOP
                break

        return uvs, pts, steps, list(ixss), terminator


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


def get_plane(origin, du, dv):
    duu = du / scalar_norm(du)

    dn = scalar_unit(scalar_cross(duu, dv))
    dvu = scalar_cross(dn, duu)
    return np.array([origin, duu, dvu, dn])


def get_normal(du, dv):
    duu = scalar_unit(du)
    dn = scalar_unit(scalar_cross(duu, dv))

    return duu, dn


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
        print(1)
        return
    edge_terminator = surface_surface_intersection_edge_terminator(surf1, surf2, tol=tol
                                                                   )

    kd, uvs1, uvs2 = res

    curves = []
    curves_uvs = []
    terminators = []
    stepss = []
    data = kd.data
    ii = 0
    l = len(uvs2)
    march = MarchingMethod(
        surf1,
        surf2,
        kd=kd,
        tol=tol,
        side=1,
        boundary_terminators=edge_terminator,
    )

    def _next():
        nonlocal ii, kd, data, uvs1, uvs2, l

        march.side = 1
        march.kd = kd

        ress = march.solve(uvs1[0], uvs2[0])
        ii += 1
        if ress is not None:
            start = np.copy(data[0])
            start_uv = np.array([np.copy(uvs1[0]), np.copy(uvs2[0])])

            if ress[-1] != TerminatorType.LOOP:
                march.side = -1
                march.kd = kd
                march.kd = kd
                march.boundary_terminators = edge_terminator

                ress_back = march.solve(uvs1[0], uvs2[0])

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

    pts_crv_1 = np.array(cc[0][0])
    pts_crv_2 = np.array(cc[0][1])

    nrm = [np.all(
        np.linalg.norm(patch1(closest_points_on_surface(patch1, pts_crv_1)) - pts_crv_1, axis=1)),
        np.all(
            np.linalg.norm(patch1(closest_points_on_surface(patch1, pts_crv_2)) - pts_crv_2, axis=1)),
        np.all(
            np.linalg.norm(patch2(closest_points_on_surface(patch2, pts_crv_1)) - pts_crv_1, axis=1)),
        np.all(
            np.linalg.norm(patch2(closest_points_on_surface(patch2, pts_crv_2)) - pts_crv_2, axis=1))]
    print(all(nrm))

    print([np.array(c).tolist() for c in cc[0]])

    # print([patch1(uvs(20, 20)).tolist(), patch2(uvs(20, 20)).tolist()])
    res = surface_intersection(patch1, patch2, TOL)

    #-----------------
