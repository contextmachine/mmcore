import itertools
import time
import warnings
from dataclasses import dataclass

from mmcore.geom.nurbs import NURBSSurface
from mmcore.geom.surfaces import CurveOnSurface, Surface
from scipy.spatial import KDTree

from mmcore.geom.bvh import BoundingBox, intersect_bvh_objects, BVHNode
from mmcore.geom.surfaces import Surface, Coons
from mmcore.numeric.vectors import solve2x2,det
from mmcore.numeric.intersection.ssx._ssi import improve_uv as cimprove_uv
from mmcore.numeric.algorithms.point_inversion import point_inversion_surface
from mmcore.numeric.closest_point import closest_point_on_ray, closest_points_on_surface, closest_point_on_surface, \
    closest_point_on_nurbs_surface
from mmcore.numeric.intersection.csx import nurbs_csx

from mmcore.numeric.plane import plane_plane_intersect

from mmcore.geom.curves.bspline import NURBSpline, interpolate_nurbs_curve

import numpy as np
from mmcore.numeric.vectors import norm, det
from collections import namedtuple
from typing import NamedTuple, Optional, Tuple

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


def improve_uv(du, dv, xyz_old, xyz_better, res):
    dxdu, dydu, dzdu = du
    dxdv, dydv, dzdv = dv

    delta = xyz_better - xyz_old

    xy = np.array([[dxdu, dxdv], [dydu, dydv]]), [delta[0], delta[1]]
    xz = np.array([[dxdu, dxdv], [dzdu, dzdv]]), [delta[0], delta[2]]
    yz = np.array([[dydu, dydv], [dzdu, dzdv]]), [delta[1], delta[2]]

    max_det = max([xy, xz, yz], key=lambda Ab: det(Ab[0]))


    return solve2x2(max_det[0], np.array(max_det[1]), res)

def improve_uv_robust(surf, uv_old,du, dv, xyz_old, xyz_better, uv_better=None):
        if uv_better is None:
            uv_better=np.zeros(2)

        success_first = cimprove_uv(du, dv, xyz_old, xyz_better, uv_better)

        if success_first == 1:
            uv_better[:] = point_inversion_surface(surf, xyz_better,*uv_old, 1e-6, 1e-6)
        else:
            uv_better += uv_old

        return uv_better


class IntersectionStepData(NamedTuple):
    first: SurfaceDerivativesData
    second: SurfaceDerivativesData
    tangent: Optional[np.ndarray]


#def improve_uv(du, dv, xyz_old, xyz_better):
#
#    return res



SurfaceStuff = namedtuple("SurfaceStuff", ["surf", "kd", "pts", "uv", "bbox"])
ClosestSurfaces = namedtuple("ClosestSurfaces", ["a", "b"])

from mmcore.numeric.gauss_map import detect_intersections
from mmcore.numeric.intersection.ssx.boundary_intersection import extract_isocurve, find_boundary_intersections


def find_closest_points(surf1: Surface, surf2: Surface,  freeform):

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
    def calculate_derivatives_data(cls,surface:NURBSSurface, uv):
        # TODO: вероятно нужно сделать оптимизированную версию для некоторых примитивов дальше использовать plane_at, а если его нет то вычислять самостоятельно
        #print(f'{cls.__name__}.calculate_derivatives_data({surface,uv})')
        pt=surface.evaluate(uv)
        du=surface.derivative_u(uv)
        dv=surface.derivative_v(uv)

        n=np.cross(du,dv)

        #n = np.array(scalar_unit(scalar_cross(du, dv)))

        return SurfaceDerivativesData(uv, pt, du, dv, n/np.linalg.norm(n))

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


    def improve_uvs(self,step_data: IntersectionStepData, pt_better: np.ndarray):

        uvb1_better = improve_uv_robust(self.s1, step_data.first.uv, step_data.first.du, step_data.first.dv,step_data.first.pt,pt_better )
        uvb2_better = improve_uv_robust(self.s2, step_data.second.uv,step_data.second.du, step_data.second.dv, step_data.second.pt, pt_better)



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

            uv1,uv2=self.improve_uvs(intersection_step_data,mid)


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
        #print('initial_data',intersection_step_data)
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
        #
        #res = self.freeform.solve(uv1_better, uv2_better,return_edges=True)
        #if res is not None:
        #    pt_better,uv1_better, uv2_better=            res

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


        mid_pt = (pt1 + pt2) * 0.5
        new_pln = np.array([mid_pt + T * self.side * step, n1, n2, T])
        return new_pln

    def calculate_sectional_curvature(self, step_data: IntersectionStepData):
        #print(f'{self.__class__}.calculate_sectional_curvature({step_data})')
        def sectional_tangent(veps):

            uv1_new = point_inversion_surface(self.s1, step_data.first.pt + veps, *step_data.first.uv,1e-6,1e-6)

            uv2_new = point_inversion_surface(self.s2, step_data.second.pt + veps, *step_data.second.uv,1e-6,1e-6)



            n1, n2 = self.s1.normal(uv1_new), self.s2.normal(uv2_new)
            Tn = np.array(scalar_cross(n1, n2))

            Tn /= scalar_norm(Tn)




            return Tn

        vesp = step_data.tangent

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
        ixss.add(0)
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
def find_start_points_nurbs(surf1, surf2,tol=1e-3):
    xyz=[]
    u1=[]
    u2=[]
    for s1,s2 in detect_intersections(surf1, surf2, tol=tol):
        for pt in find_boundary_intersections(s1,s2,tol):
            if tuple(pt.point) not in xyz:
                xyz.append(tuple(pt.point))

                u1.append(pt.surface1_params)
                u2.append(pt.surface2_params)


    return KDTree(np.array(xyz)),np.array(u1),np.array(u2)

times=[]
def surface_ppi(surf1: Surface, surf2: Surface, tol=0.001, max_iter=500):
    #s=time.perf_counter_ns()[(0.12254503038194443, 0.607421875), (0.12037037478552923, 0.6044921875),
    edge_terminator = surface_surface_intersection_edge_terminator(
        surf1, surf2, tol=tol
    )
    #times.append(time.perf_counter_ns()-s)

    freeform=FreeFormMethod(surf1,surf2,
        tol=tol,
        boundary_terminators=edge_terminator,
                   max_iter=19
    )
    #s = time.perf_counter_ns()
    if isinstance(surf1, NURBSSurface) and isinstance(surf2, NURBSSurface):
        res = find_start_points_nurbs(surf1,surf2,tol=tol)
        #print(res[0].data.tolist())
    else:
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
                    #print(ress_back)
                    #print(uvsb)
                    rmv = np.unique(ixss + ixssb)

                    #print(rmv)
                    if  len(rmv)==0:
                        rmv=np.array([0],dtype=int)

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
            #print(rmv)
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
    s1,s2=td[1]
    s = time.perf_counter_ns()

    cc =    surface_ppi(s1,s2,TOL)
    e=(time.perf_counter_ns()-s)*1e-9
    #s1= NURBSSurface(s1.control_points+np.array([[1000.,0.,1000.]]), degree=tuple(s1.degree))
    #s2= NURBSSurface(s2.control_points + np.array([[1000., 0., 1000.]]), degree=tuple(s2.degree))
    #st1 = time.perf_counter_ns()
    #cc2 =    surface_ppi(s1,s2,TOL)
    #e1=(time.perf_counter_ns()-st1)*1e-9

    print([np.array(c).tolist() for c in cc[0]])
    # -----------------

    with open('crz2.json', 'w') as f:
        import json

        res=[np.array(c).tolist() for c in cc[0]]
        print(res)
        json.dump(res, f)

    print("\n\n\n","-"*80,"\n",e)
    #print(e1)
