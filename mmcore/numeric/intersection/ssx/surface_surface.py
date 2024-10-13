import itertools
import time
import warnings

from mmcore.geom.nurbs import NURBSSurface
from mmcore.geom.surfaces import CurveOnSurface, Surface
from scipy.spatial import KDTree

from mmcore.geom.bvh import BoundingBox, intersect_bvh_objects, BVHNode
from mmcore.geom.surfaces import Surface, Coons
from mmcore.numeric.closest_point import closest_point_on_ray, closest_points_on_surface, closest_point_on_surface
from mmcore.numeric.intersection.csx import nurbs_csx

from mmcore.numeric.plane import plane_plane_intersect

from mmcore.geom.curves.bspline import NURBSpline, interpolate_nurbs_curve

import numpy as np
from mmcore.numeric.vectors import norm
from collections import namedtuple
from typing import NamedTuple, Optional

from mmcore.numeric import scalar_cross, scalar_norm
from mmcore.numeric.vectors import scalar_unit

from mmcore.numeric.fdm import DEFAULT_H
from mmcore.numeric.intersection.ssx._terminator import TerminatorType
from mmcore.geom.evaluator import surface_evaluator
from mmcore.numeric.plane import plane_plane_plane_intersect_points_and_normals
from mmcore.numeric.intersection.ssx._terminator import (
    surface_surface_intersection_edge_terminator,
)

SurfaceDerivativesData = namedtuple(
    "SurfaceDerivativesData", ["uv", "pt", "du", "dv", "normal"]
)


class IntersectionStepData(NamedTuple):
    first: SurfaceDerivativesData
    second: SurfaceDerivativesData
    tangent: Optional[np.ndarray]


#def improve_uv(du, dv, xyz_old, xyz_better):
#
#    return res

from mmcore.numeric.intersection.ssx._ssi import improve_uv



SurfaceStuff = namedtuple("SurfaceStuff", ["surf", "kd", "pts", "uv", "bbox"])
ClosestSurfaces = namedtuple("ClosestSurfaces", ["a", "b"])

from mmcore.numeric.gauss_map import detect_intersections, extract_isocurve


def find_closest_points_nurbs(surf1: NURBSSurface, surf2: NURBSSurface,tol=1e-3):
    uvsi = []
    uvsj = []
    pts=[]
    for i, j in detect_intersections(surf1, surf2):

        for ii,l in enumerate((extract_isocurve(i, 0., 'v'),
                  extract_isocurve(i, 0., 'u'),
                  extract_isocurve(i, 1., 'u'),
                  extract_isocurve(i, 1., 'v'))):
            try:
                res = nurbs_csx(l, j, tol=tol, ptol=1e-5)
            except Exception as e:
                continue

            if len(res)>0:
                _,ptt,res=zip(*res)
                res=np.array(res)
                uvj=res[0, 1:]
                if ii==0:
                    uvi=np.array([res[0,0],0.])
                elif ii == 1:
                    uvi = np.array([0., res[0, 0]])
                elif ii == 2:
                    uvi = np.array([1., res[0, 0]])
                else:
                    uvi = np.array([ res[0, 0],1.])
                pt=j.evaluate(uvj)
                pts.append(pt)

                uvsi.append(uvi)
                uvsj.append(uvj)
    if len(pts)>0:
        print(np.array(pts),np.array(uvsi), np.array(uvsj))
        return KDTree(np.array(pts)), np.array(uvsi), np.array(uvsj)

def find_closest_points(surf1: Surface, surf2: Surface,  freeform):
    if isinstance(surf1, NURBSSurface) and isinstance(surf2, NURBSSurface):
        return find_closest_points_nurbs(surf1, surf2)
    #print('\nfind_closest_points start\n---------------\n\n')
    min1max1: BoundingBox = surf1.tree.bounding_box
    min2max2: BoundingBox = surf2.tree.bounding_box
    interval1 = tuple(surf1.interval())
    interval2 = tuple(surf2.interval())

    if min1max1.intersect(min2max2):
        pts = []
        uvs1 = []
        uvs2 = []
        index=0

        for first, second in intersect_bvh_objects(surf1.tree, surf2.tree):
            first: BVHNode
            second: BVHNode
            # bb=first.bounding_box.intersection(second.bounding_box)

            #uv1 = np.average(first.object.uvs, axis=0)
            #uv2 = np.average(second.object.uvs, axis=0)

            uv1 = np.average(first.object.uvs, axis=0)
            uv2 = np.average(second.object.uvs, axis=0)

            if len(uv1) == 0 or len(uv2) == 0:
                pass

            else:
                # #print(uv1, uv2)

                    res = freeform.solve( uv1, uv2, return_edges=False)

                    if res is not None:
                        (xyz1_new, uvb1_better, uvb2_better) = res
                        if any(
                                handle_inside(*uvb1_better, interval1)
                                + handle_inside(*uvb2_better, interval2)

                        ):
                            pass
                        # #print(uvb1_better, uvb2_better)
                        pts.append(xyz1_new)
                        uvs1.append(uvb1_better)
                        uvs2.append(uvb2_better)
                        #print(f'find_closest_points end {index} loop')
                        index+=1
        ##print(np.array(pts).tolist())

        #print('\n\n---------------\n\nfind_closest_points end\n\n')
        return KDTree(np.array(pts)), np.array(uvs1), np.array(uvs2)


def handle_inside(u1, v1, bounds):
    return (
        not (bounds[0][0] <= u1 <= bounds[0][1]),
        not (bounds[1][0] <= v1 <= bounds[1][1])

    )


class SSXMethod:
    def __init__(
            self,
            s1,
            s2,
            tol=1e-3,
            max_iter=500,
            boundary_terminators=None,
            s1_interval=None,
            s2_interval=None,
    ):
        self.s1 = s1
        self.s2 = s2

        self.tol = tol
        self.max_iter = max_iter

        self.boundary_terminators = boundary_terminators

        self.s1_interval = s1_interval
        self.s2_interval = s2_interval
        if s1_interval is None:
            self.s1_interval = tuple(
                getattr(self.s1, "interval", lambda: ((0.0, 1.0), (0.0, 1.0)))()
            )
        if s2_interval is None:
            self.s2_interval = tuple(
                getattr(self.s2, "interval", lambda: ((0.0, 1.0), (0.0, 1.0)))()
            )

    @classmethod
    def calculate_derivatives_data(cls,surface, uv):
        # TODO: вероятно нужно сделать оптимизированную версию для некоторых примитивов дальше использовать plane_at, а если его нет то вычислять самостоятельно
        #print(f'{cls.__name__}.calculate_derivatives_data({surface,uv})')
        pt=surface.evaluate(uv)
        pt,du,dv,n=surface_evaluator.origin_derivatives_normal(surface.evaluate_v2,uv[0],uv[1])

        #n = np.array(scalar_unit(scalar_cross(du, dv)))

        return SurfaceDerivativesData(uv, pt, du, dv, n)

    @classmethod
    def calculate_intersection_step_data(
            cls, surface1, surface2, uv1, uv2
    ) -> IntersectionStepData:
        #print(f'{cls.__name__}.calculate_intersection_step_data({surface1, surface2, uv1, uv2})')
        s1_data = cls.calculate_derivatives_data(surface1, uv1)
        s2_data = cls.calculate_derivatives_data(surface2, uv2)
        T = np.array(scalar_cross(s1_data.normal, s2_data.normal))
        T /= scalar_norm(T)
        return IntersectionStepData(s1_data, s2_data, T)

    @staticmethod
    def improve_uvs(step_data: IntersectionStepData, pt_better: np.ndarray):
        uvb1_better = step_data.first.uv + improve_uv(
            step_data.first.du, step_data.first.dv, step_data.first.pt, pt_better
        )
        uvb2_better = step_data.second.uv + improve_uv(
            step_data.second.du, step_data.second.dv, step_data.second.pt, pt_better
        )
        return uvb1_better, uvb2_better

    def handle_inside(self, u1, v1, u2, v2):
        return (
            not (self.s1_interval[0][0] <= u1 <= self.s1_interval[0][1]),
            not (self.s1_interval[1][0] <= v1 <= self.s1_interval[1][1]),
            not (self.s2_interval[0][0] <= u2 <= self.s2_interval[0][1]),
            not (self.s2_interval[1][0] <= v2 <= self.s2_interval[1][1]),
        )

    def handle_edge(self, xyz_better, uvb1_better, uvb2_better):
        if any(self.handle_inside(*uvb1_better, *uvb2_better)):
            if self.boundary_terminators is not None:
                (
                    xyz_better,
                    uvb1_better,
                    uvb2_better,
                ) = self.boundary_terminators.get_closest(xyz_better)

            return True, (xyz_better, uvb1_better, uvb2_better)

        return False, (xyz_better, uvb1_better, uvb2_better)


    def solve(self, uv1, uv2):
        pass


class FreeFormMethod(SSXMethod):
    @staticmethod
    def calculate_intersection_line(step_data: IntersectionStepData):
        pln1, pln2 = np.empty((2, 4, 3))
        pln1[0] = step_data.first.pt
        pln2[0] = step_data.second.pt
        pln1[-1] = step_data.first.normal
        pln2[-1] = step_data.second.normal
        ln = np.array(plane_plane_intersect(pln1, pln2))
        return ln

    @classmethod
    def calculate_intersection_step_data(
            cls, surface1, surface2, uv1, uv2
    ) -> IntersectionStepData:
        #print(f'FreeFormMethod.calculate_intersection_step_data({surface1, surface2, uv1, uv2})')
        s1_data = cls.calculate_derivatives_data(surface1, uv1)
        s2_data = cls.calculate_derivatives_data(surface2, uv2)

        return IntersectionStepData(s1_data, s2_data, None)

    def refine_points(self, step_data: IntersectionStepData):
        #print(f'FreeFormMethod.refine_points({step_data})')
        ln = self.calculate_intersection_line(step_data)

        np1 = np.array(closest_point_on_ray((ln[0], ln[1]), step_data.first.pt))
        np2 = np.array(closest_point_on_ray((ln[0], ln[1]), step_data.second.pt))

        return np1, np1 + (np2 - np1) / 2, np2

    def solve(self, uv1, uv2, return_edges=False):
        #print(f'FreeFormMethod.solve({uv1, uv2})')

        # sn = scalar_norm(pt1 - pt2)
        # if sn <( tol/2):
        #    return ((pt1+pt2)/2, uvb1), ((pt1+pt2)/2, uvb2)
        uv1=np.array(uv1)
        uv2 = np.array(uv2)

        for i in range(self.max_iter):
            ##print(i)
            intersection_step_data: IntersectionStepData = self.calculate_intersection_step_data(self.s1, self.s2,
                                                                                                uv1,
                                                                                                uv2)

            p1_better, mid, p2_better = self.refine_points(intersection_step_data)



            uv1 += improve_uv(intersection_step_data.first.du,
                                              intersection_step_data.first.dv,
                                              intersection_step_data.first.pt,
                                              mid)

            uv2 += improve_uv(intersection_step_data.second.du,
                                              intersection_step_data.second.dv,
                                              intersection_step_data.second.pt,
                                              mid)

            if any(self.handle_inside(*uv1,*uv2)):
                if return_edges:
                    mid,uv1,uv2=self.boundary_terminators.get_closest(mid)
                    return  mid,uv1,uv2
                else:
                    return

            d1, d2 = scalar_norm(intersection_step_data.first.pt - p1_better), scalar_norm(
                intersection_step_data.second.pt - p2_better)

            if d1<self.tol and d2<self.tol:

                return mid, uv1, uv2





        warnings.warn("freeform not convergence")
        return


class MarchingMethod(SSXMethod):
    def __init__(
            self,
            s1,
            s2,
            kd=None,
            tol=1e-3,
            max_iter=500,
            side=1,
            boundary_terminators=None,
            fdm_h=DEFAULT_H,
            s1_interval=None,
            s2_interval=None,
            freeform:FreeFormMethod=None
    ):
        super().__init__(s1, s2, tol=tol, max_iter=max_iter, boundary_terminators=boundary_terminators,
                         s1_interval=s1_interval, s2_interval=s2_interval)
        self.kd = kd

        self.side = side
        self.fdm_h = fdm_h
        self.boundary_terminators = boundary_terminators
        self.freeform=freeform


    def step(self, uv1, uv2):
        #print(f'{self.__class__}.solve({uv1, uv2})')

        intersection_step_data: IntersectionStepData = (
            self.calculate_intersection_step_data(self.s1, self.s2, uv1, uv2)
        )
        curvature_vector = self.calculate_sectional_curvature(intersection_step_data)
        step = self.calculate_step(curvature_vector)
        pt_better = self.refine_point(intersection_step_data, step=step)
        if pt_better is None:
            return

        uv1_better, uv2_better = self.improve_uvs(intersection_step_data, pt_better)

        is_edge, (pt_better, uv1_better, uv2_better) = self.handle_edge(
            pt_better, uv1_better, uv2_better
        )
        if is_edge:
            return (
                (pt_better, uv1_better),
                (pt_better, uv2_better),
                step,
                TerminatorType.EDGE,
            )
        res = self.freeform.solve(uv1_better, uv2_better,return_edges=True)
        if res is not None:
            pt_better,uv1_better, uv2_better=            res

        return (
            (pt_better, uv1_better),
            (pt_better, uv2_better),
            step,
            TerminatorType.STEP,
        )

    def calculate_step(self, curvature_vector):
        K = scalar_norm(curvature_vector)

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
        #print(f'{self.__class__}.calculate_sectional_curvature({step_data})')
        def sectional_tangent(veps):
            uv1_new = step_data.first.uv + improve_uv(
                step_data.first.du,
                step_data.first.dv,
                step_data.first.pt,
                step_data.first.pt + veps,
            )
            uv2_new = step_data.second.uv + improve_uv(
                step_data.second.du,
                step_data.second.dv,
                step_data.second.pt,
                step_data.second.pt + veps,
            )


            n1, n2 = self.s1.normal(uv1_new), self.s2.normal(uv2_new)
            Tn = np.array(scalar_cross(n1, n2))
            Tn /= scalar_norm(Tn)
            return Tn

        vesp = step_data.tangent * self.fdm_h
        curvature_vector = (sectional_tangent(vesp) - step_data.tangent) / (
            scalar_norm(vesp)
        )

        return curvature_vector

    def refine_point(self, ders: IntersectionStepData, step: float):
        new_pln = self.calculate_new_plane(ders, step=step)
        return np.array(
            plane_plane_plane_intersect_points_and_normals(
                ders.first.pt,
                ders.first.normal,
                ders.second.pt,
                ders.second.normal,
                new_pln[0],
                new_pln[-1],
            )
        )

    def solve(self, initial_uv1, initial_uv2):
        #print(f'{self.__class__}.solve({initial_uv1,initial_uv2})')
        use_kd = self.kd is not None
        ixss = set()
        xyz1_init, xyz2_init = self.s1.evaluate(initial_uv1), self.s2.evaluate(
            initial_uv2
        )

        res = self.step(initial_uv1, initial_uv2)

        if res is None:
            return

        (xyz1, uv1_new), (xyz2, uv2_new), step, terminator = res


        if use_kd:
            ixss.update(self.kd.query_ball_point(xyz1, step * 2))

        uvs = [(uv1_new, uv2_new)]
        pts = [xyz1]

        steps = [step]
        if terminator is TerminatorType.EDGE:
            return uvs, pts, steps, list(ixss), terminator
        for i in range(self.max_iter):
            uv1, uv2 = uv1_new, uv2_new

            res = self.step(uv1, uv2)
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

times=[]
def surface_ppi(surf1: Surface, surf2: Surface, tol=0.1, max_iter=500):
    #s=time.perf_counter_ns()
    edge_terminator = surface_surface_intersection_edge_terminator(
        surf1, surf2, tol=tol
    )
    #times.append(time.perf_counter_ns()-s)

    freeform=FreeFormMethod(surf1,surf2,
        tol=tol,
        boundary_terminators=edge_terminator,
                   max_iter=9
    )
    #s = time.perf_counter_ns()
    res = find_closest_points(surf1, surf2, freeform)
    #times.append(time.perf_counter_ns() - s)

    if res is None:
        #print(1)
        return

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
        freeform=freeform
    )

    def _next():
        nonlocal ii, kd, data, uvs1, uvs2, l

        march.side = 1

        ress = march.solve(uvs1[0], uvs2[0])
        ii += 1
        if ress is not None:
            start = np.copy(data[0])
            start_uv = np.array([np.copy(uvs1[0]), np.copy(uvs2[0])])

            if ress[-1] != TerminatorType.LOOP:
                march.side = -1

                ress_back = march.solve(uvs1[0], uvs2[0])

                if ress_back is not None:
                    uv_s, pts, steps, ixss, terminator = ress
                    uvsb, ptsb, stepsb, ixssb, terminator_back = ress_back
                    rmv = np.unique(ixss + ixssb)
                    data = np.delete(data, rmv, axis=0)
                    uvs1 = np.delete(uvs1, rmv, axis=0)
                    uvs2 = np.delete(uvs2, rmv, axis=0)
                    if len(data.shape) == 2:
                        march.kd = kd = KDTree(data)
                    else:
                        march.kd = kd = None
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
                march.kd = kd = KDTree(data)
            else:
                march.kd = kd = None
            terminators.append([terminator])
            curves_uvs.append([start_uv] + list(uv_s))
            stepss.append(steps)
            return [start] + list(pts)
        else:
            uvs1 = np.delete(uvs1, 0, axis=0)
            uvs2 = np.delete(uvs2, 0, axis=0)
            data = np.delete(data, 0, axis=0)
            if len(data.shape) == 2:
                march.kd = kd = KDTree(data)
            else:
                march.kd = kd = None

    #s=time.perf_counter_ns()
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
    ##times.append(time.perf_counter_ns() - s)

    return (
        curves,
        [
            [np.array(crv, dtype=float) for crv in zip(*curve_uv)]
            for curve_uv in curves_uvs
        ],
        stepss,
        terminators,
    )




def surface_intersection(
        surf1: Surface, surf2: Surface, tol: float = 0.01, max_iter: int = 500
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
    res = surface_ppi(surf1, surf2, tol=tol, max_iter=max_iter)
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
    #patch1.build_tree(3, 3)
    #patch2.build_tree(3, 3)
    # #print(
    #    patch1._rc,
    # )
    #import yappi
    #yappi.set_clock_type("wall")  # Use set_clock_type("wall") for wall time
    #yappi.start()
    #s = time.perf_counter_ns()
    TOL = 0.01


    #cc = surface_ppi(patch1, patch2, TOL)
    #print((time.perf_counter_ns() - s)*1e-9)
    #yappi.stop()
    #func_stats = yappi.get_func_stats()
    #func_stats.save(f"{__file__.replace('.py', '')}_{int(time.time())}.pstat", type='pstat')

    # tolerance checks
    from mmcore._test_data import ssx as td

    #pts_crv_1 = np.array(cc[0][0])
    #pts_crv_2 = np.array(cc[0][1])
    #print([t*1e-9 for t in times])
    #nrm = [
    #    np.all(
    #        norm(
    #            patch1(closest_points_on_surface(patch1, pts_crv_1)) - pts_crv_1
    #        )
    #    ),
    #    np.all(
    #        norm(
    #            patch1(closest_points_on_surface(patch1, pts_crv_2)) - pts_crv_2
    #        )
    #    ),
    #    np.all(
    #        norm(
    #            patch2(closest_points_on_surface(patch2, pts_crv_1)) - pts_crv_1
    #        )
    #    ),
    #    np.all(
    #        norm(
    #            patch2(closest_points_on_surface(patch2, pts_crv_2)) - pts_crv_2
    #        )
    #    ),
    #]
    ##print(all(nrm))

    #print([np.array(c).tolist() for c in cc[0]])
    #
    ## #print([patch1(uvs(20, 20)).tolist(), patch2(uvs(20, 20)).tolist()])
    #res = surface_intersection(patch1, patch2, TOL)
    s1,s2=td[2]
    s = time.perf_counter_ns()

    cc =    surface_ppi(s1,s2,TOL)
    e=(time.perf_counter_ns()-s)*1e-9
    print(e)
    print([np.array(c).tolist() for c in cc[0]])
    # -----------------
