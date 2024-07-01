import itertools
import math
import time
from enum import Enum

import numpy as np
import scipy

from scipy.optimize import fsolve
from scipy.spatial import KDTree

from mmcore.geom.bvh import BoundingBox, intersect_bvh, intersect_bvh_objects, BVHNode
from mmcore.geom.surfaces import Surface, Coons
from mmcore.numeric import calgorithms
from mmcore.numeric.aabb import aabb_overlap, aabb
from mmcore.numeric.closest_point import closest_point_on_line, closest_point_on_ray
from mmcore.numeric.plane import plane_plane_intersect, plane_plane_plane_intersect
from mmcore.geom.curves.knot import interpolate_curve
from mmcore.geom.curves.bspline import NURBSpline
from mmcore.numeric.vectors import scalar_norm, scalar_cross, scalar_unit, det, solve2x2


class TerminatorType(int, Enum):
    FAIL = 0
    LOOP = 1
    EDGE = 2


def get_plane(origin, du, dv):
    duu = du / scalar_norm(du)

    dn = scalar_unit(scalar_cross(duu, dv))
    dvu = scalar_cross(dn, duu)
    return [origin, duu, dvu, dn]


def freeform_step_debug(pt1, pt2, du1, dv1, du2, dv2):
    pl1, pl2 = get_plane(pt1, du1, dv1), get_plane(pt2, du2, dv2)

    ln = np.array(plane_plane_intersect(pl1, pl2))

    np1 = np.asarray(closest_point_on_ray((ln[0], ln[1]), pt1))
    np2 = np.asarray(closest_point_on_ray((ln[0], ln[1]), pt2))

    return np1, np1 + (np2 - np1) / 2, np2


def get_normal(du, dv):
    duu = scalar_unit(du)
    dn = scalar_unit(scalar_cross(duu, dv))

    return duu, dn


def solve_marching(pt1, pt2, du1, dv1, du2, dv2, tol, side=1):
    pl1, pl2 = get_plane(pt1, du1, dv1), get_plane(pt2, du2, dv2
                                                   )

    marching_direction = np.array(scalar_unit(scalar_cross(pl1[-1], pl2[-1])))

    tng = np.zeros((2, 3))

    calgorithms.evaluate_curvature(marching_direction * side, pl1[-1], tng[0], tng[1])

    K = tng[1]

    r = 1 / scalar_norm(K)

    step = np.sqrt(abs(r ** 2 - (r - tol) ** 2)) * 2

    new_pln = np.array([pt1 + marching_direction * side * step, pl1[-1], pl2[-1], marching_direction * side])

    return plane_plane_plane_intersect(pl1, pl2, new_pln), step


def improve_uv(du, dv, xyz_old, xyz_better):
    dxdu, dydu, dzdu = du
    dxdv, dydv, dzdv = dv

    delta = xyz_better - xyz_old

    xy = np.array([[dxdu, dxdv], [dydu, dydv]]), [delta[0], delta[1]]
    xz = np.array([[dxdu, dxdv], [dzdu, dzdv]]), [delta[0], delta[2]]
    yz = np.array([[dydu, dydv], [dzdu, dzdv]]), [delta[1], delta[2]]

    max_det = max([xy, xz, yz], key=lambda Ab: det(Ab[0]))
    res=np.zeros(2)
    solve2x2(max_det[0], np.array(max_det[1]), res)
    return res


def freeform_step(s1, s2, uvb1, uvb2, tol, cnt=0):
    pt1 = s1.evaluate(uvb1)
    du1 = s1.derivative_u(uvb1)
    dv1 = s1.derivative_v(uvb1)
    pt2 = s2.evaluate(uvb2)
    du2 = s2.derivative_u(uvb2)
    dv2 = s2.derivative_v(uvb2)
    xyz_better = freeform_step_debug(pt1, pt2, du1, dv1, du2, dv2)[1]

    uvb1_better = uvb1 + improve_uv(du1, dv1, pt1, xyz_better)
    uvb2_better = uvb2 + improve_uv(du2, dv2, pt2, xyz_better)
    if any(uvb1_better < 0.) or any(uvb2_better < 0.) or any(uvb1_better > 1.) or any(uvb2_better > 1.):
        return
    xyz1_new = s1.evaluate(uvb1_better)
    xyz2_new = s2.evaluate(uvb2_better)
    #print(   np.linalg.norm(xyz1_new -  xyz2_new))
    if scalar_norm(xyz1_new - xyz2_new) <= tol:
        # print("e")
        return (xyz1_new, uvb1_better), (xyz2_new, uvb2_better)
    else:

        return freeform_step(s1, s2, uvb1_better, uvb2_better, tol, cnt + 1)


def marching_step(s1, s2, uvb1, uvb2, tol, cnt=0, side=1):
    pt1 = s1.evaluate(uvb1)
    pt2 = s2.evaluate(uvb2)

    du1 = s1.derivative_u(uvb1)
    dv1 = s1.derivative_v(uvb1)

    du2 = s2.derivative_u(uvb2)
    dv2 = s2.derivative_v(uvb2)
    xyz_better, step = solve_marching(pt1, pt2, du1, dv1, du2, dv2, tol, side=side)

    uvb1_better = uvb1 + improve_uv(du1, dv1, pt1, xyz_better)
    uvb2_better = uvb2 + improve_uv(du2, dv2, pt2, xyz_better)

    if any(uvb1_better < 0.) or any(uvb2_better < 0.) or any(uvb1_better > 1.) or any(uvb2_better > 1.):


        #uvb1_better=np.clip(uvb1_better, 0., 1.)
        #uvb2_better=np.clip(uvb2_better, 0., 1.)

        #pt=s1.evaluate(uvb1_better)

        return

    return (xyz_better, uvb1_better), (xyz_better, uvb2_better), step

    #else:

    #    return marching_step(s1, s2, uvb1_better, uvb2_better, tol, cnt + 1, side=side)


def marching_method(
        s1, s2, initial_uv1, initial_uv2, kd=None, tol=1e-3, max_iter=1000, no_ff=False, side=1
):
    terminator = None
    use_kd = kd is not None
    ixss = set()
    xyz1_init, xyz2_init = s1.evaluate(initial_uv1), s2.evaluate(initial_uv2)
    res = marching_step(s1, s2, initial_uv1, initial_uv2, tol=tol, side=side)

    if res is None:
        #print("N")
        return terminator

    (xyz1, uv1_new), (xyz2, uv2_new), step = res
    if use_kd:
        ixss.update(kd.query_ball_point(xyz1_init, step * 2))
    #print()

    uvs = [(uv1_new, uv2_new)]
    pts = [xyz1]

    #print(uv1_new, uv2_new)

    steps = [step]

    for i in range(max_iter):

        uv1, uv2 = uv1_new, uv2_new

        res = marching_step(s1, s2, uv1, uv2, tol=tol, side=side)

        if res is None:
            #print("N")
            terminator = TerminatorType.EDGE
            break
        else:

            (xyz1, uv1_new), (xyz2, uv2_new), step = res
            pts.append(xyz1)
            uvs.append((uv1_new, uv2_new))
            steps.append(step)
            if use_kd:
                #print(   len(kd.data),ixss,step,tol,kd.query(xyz1, 3))

                ixss.update(kd.query_ball_point(xyz1, step * 2))
                #print(ixss)

            if np.linalg.norm(xyz1 - xyz1_init) < step:
                #print("b", xyz1_init, xyz1, np.linalg.norm(xyz1 - xyz1_init), step)
                pts.append(pts[0])

                uvs.append((initial_uv1, initial_uv1))
                terminator = TerminatorType.LOOP
                break
        #print("I", np.linalg.norm(xyz1 - xyz1_init), step*2)

    #print(len(pts))
    return uvs, pts, steps, list(ixss), terminator



class SurfacePPI:
    def __init__(self, surf1, surf2, tol=1e-3):
        self.surf1 = surf1
        self.surf2 = surf2
        self.tol = tol
        self.success = False
        self.kd, self.uvs1, self.uvs2 = None, None, None

        self.curves = []
        self.curves_uvs = []
        self.stepss = []
        self.data=None
        self.i = 0
        self.l=0
    def _find_closest_pts(self):
        res = find_closest2(surf1=self.surf1, surf2=self.surf2, tol=self.tol)
        if res is None:
            self.success = False

        else:
            self.success = True
            self.kd, self.uvs1, self.uvs2 = res

    def prepare_data(self):

        self.curves = []
        self.curves_uvs = []
        self.stepss = []
        self.data = self.kd.data
        self.i = 0
        self.l = len(self.uvs1)


class ParametricSurfaceIntersectionIterator:
    def __init__(self, obj:SurfacePPI,degree=3):
        self._obj = obj
        self.degree = degree



    def __next__(self):
        if self._obj.data.shape[0] < 1:
            raise StopIteration
        if self._obj.i >= self._obj.l:
            raise StopIteration
        # print(ii)

        ress = marching_method(self._obj.surf1, self._obj.surf2, self._obj.uvs1[0], self._obj.uvs2[0], kd=self._obj.kd, tol=self._obj.tol, side=1)
        self._obj.i += 1
        if ress is not None:
            start = np.copy(self._obj.data[0])
            if ress[-1] != TerminatorType.LOOP:
                ress_back = marching_method(self._obj.surf1, self._obj.surf2, self._obj.uvs1[0], self._obj.uvs2[0], kd=self._obj.kd, tol=self._obj.tol, side=-1)

                if ress_back is not None:

                    uvs, pts, steps, ixss, terminator = ress
                    uvsb, ptsb, stepsb, ixssb, terminator = ress_back
                    rmv = np.unique(ixss + ixssb)
                    self._obj.data = np.delete(self._obj.data, rmv, axis=0)
                    self._obj.uvs1 = np.delete(self._obj.uvs1, rmv, axis=0)
                    self._obj.uvs2 = np.delete(self._obj.uvs2, rmv, axis=0)
                    if len(self._obj.data.shape) == 2:
                        self._obj.kd = KDTree(self._obj.data)
                    else:

                        self._obj.kd = None
                    return list(itertools.chain(reversed(ptsb), [start], pts))

            uvs, pts, steps, ixss, terminator = ress

            rmv = np.array(ixss, dtype=int)
            # print(rmv)
            self._obj.data = np.delete(self._obj.data, rmv, axis=0)
            self._obj.uvs1 = np.delete(self._obj.uvs1, rmv, axis=0)
            self._obj.uvs2 = np.delete(self._obj.uvs2, rmv, axis=0)
            if len(self._obj.data.shape) == 2:
                self._obj.kd = KDTree(self._obj.data)

            else:

                self._obj.kd = None

            return [start] + list(pts)



        else:
            self._obj.uvs1 = np.delete(self._obj.uvs1, 0, axis=0)
            self._obj.uvs2 = np.delete(self._obj.uvs2, 0, axis=0)
            self._obj.data = np.delete(self._obj.data, 0, axis=0)
            if len(self._obj.data.shape) == 2:
                self._obj.kd = KDTree(self._obj.data)

            else:
                self._obj.kd = None

def surface_ppi(surf1, surf2, tol=0.1):
    res = find_closest2(surf1, surf2, tol=tol)
    if res is None:
        return

    kd, uvs1, uvs2 = res

    curves = []
    curves_uvs = []
    stepss = []
    data = kd.data
    ii = 0
    l = len(uvs2)

    def _next():
        nonlocal ii, kd, data, uvs1, uvs2, l

        if data.shape[0] < 1:
            raise StopIteration
        if ii >= l:
            raise StopIteration
        #print(ii)

        ress = marching_method(surf1, surf2, uvs1[0], uvs2[0], kd=kd, tol=tol, side=1)
        ii += 1
        if ress is not None:
            start = np.copy(data[0])
            if ress[-1] != TerminatorType.LOOP:
                ress_back = marching_method(surf1, surf2, uvs1[0], uvs2[0], kd=kd, tol=tol, side=-1)

                if ress_back is not None:

                    uvs, pts, steps, ixss, terminator = ress
                    uvsb, ptsb, stepsb, ixssb, terminator = ress_back
                    rmv = np.unique(ixss + ixssb)
                    data = np.delete(data, rmv, axis=0)
                    uvs1 = np.delete(uvs1, rmv, axis=0)
                    uvs2 = np.delete(uvs2, rmv, axis=0)
                    if len(data.shape) == 2:
                        kd = KDTree(data)
                    else:

                        kd = None
                    return list(itertools.chain(reversed(ptsb), [start], pts))

            uvs, pts, steps, ixss, terminator = ress

            rmv = np.array(ixss, dtype=int)
            # print(rmv)
            data = np.delete(data, rmv, axis=0)
            uvs1 = np.delete(uvs1, rmv, axis=0)
            uvs2 = np.delete(uvs2, rmv, axis=0)
            if len(data.shape) == 2:
                kd = KDTree(data)

            else:

                kd = None

            return [start] + list(pts)



        else:
            uvs1 = np.delete(uvs1, 0, axis=0)
            uvs2 = np.delete(uvs2, 0, axis=0)
            data = np.delete(data, 0, axis=0)
            if len(data.shape) == 2:
                kd = KDTree(data)

            else:
                kd = None

    while True:
        try:
            res = _next()
            if res is None:
                pass
            else:

                curves.append(res)
        except StopIteration as err:
            break

    return curves, curves_uvs, stepss


def uvs(u_count, v_count):
    u = np.linspace(0., 1., u_count)
    v = np.linspace(0., 1., v_count)
    uv = []
    for i in range(u_count):
        for j in range(v_count):
            uv.append((u[i], v[j]))
    return np.array(uv)


from collections import namedtuple

SurfaceStuff = namedtuple("SurfaceStuff", ['surf', 'kd', 'pts', 'uv', 'bbox'])
ClosestSurfaces = namedtuple("ClosestSurfaces", ['a', 'b'])


def find_closest2(surf1: Surface, surf2: Surface, tol=1e-3):
    min1max1: BoundingBox = surf1.tree.bounding_box
    min2max2: BoundingBox = surf2.tree.bounding_box

    if min1max1.intersect(min2max2):
        pts = []
        uvs1 = []
        uvs2 = []
        for first, second in intersect_bvh_objects(surf1.tree, surf2.tree):
            first: BVHNode
            second: BVHNode
            #bb=first.bounding_box.intersection(second.bounding_box)

            uv1 = np.average(first.object.uvs, axis=0)
            uv2 = np.average(second.object.uvs, axis=0)

            res = freeform_step(surf1, surf2, uv1, uv2, tol=tol)
            if res is not None:
                (xyz1_new, uvb1_better), (xyz2_new, uvb2_better) = res
                #print(uvb1_better, uvb2_better)
                pts.append(xyz1_new)
                uvs1.append(uvb1_better)
                uvs2.append(uvb2_better)
        return KDTree(np.array(pts)), np.array(uvs1), np.array(uvs2)


if __name__ == '__main__':
    import yappi
    from mmcore.geom.curves.cubic import CubicSpline

    pts1 = np.array([
        [(-6.0558943035701525, -13.657656200983698, 1.0693341635684721),
         (-1.5301574718208828, -12.758430585795727, -2.4497481670182113),
         (4.3625055618617772, -14.490138754852163, -0.052702347089249368),
         (7.7822965141636233, -13.958097981505476, 1.1632592672736894)],
        [(7.7822965141636233, -13.958097981505476, 1.1632592672736894),
         (9.3249111495947457, -9.9684277340655711, -2.3272399773510646),
         (9.9156785503454081, -4.4260877770435245, -4.0868275118021469),
         (13.184366571517304, 1.1076098797323481, 0.55039832538794542)],
        [(-3.4282810787748206, 2.5976227512567878, -4.1924897351083787),
         (5.7125793432806686, 3.1853804927764848, -3.1997049666908506),
         (9.8891692556257418, 1.2744489476398368, -7.2890391724273922),
         (13.184366571517304, 1.1076098797323481, 0.55039832538794542)],
        [(-6.0558943035701525, -13.657656200983698, 1.0693341635684721),
         (-2.1677078000821663, -4.2388638567221646, -3.2149413059589502),
         (-3.5823721281354479, -1.1684651343084738, 3.3563417199639680),
         (-3.4282810787748206, 2.5976227512567878, -4.1924897351083787)]]
    )

    pts2 = np.array([
        [(-9.1092663228073292, -12.711321277810857, -0.77093266173210928),
         (-1.5012583168504101, -15.685662924609387, -6.6022178296290024),
         (0.62360921189203689, -15.825362292273830, 2.9177845739234654),
         (7.7822965141636233, -14.858282311330257, -5.1454157090841059)],
        [(7.7822965141636233, -14.858282311330257, -5.1454157090841059),
         (9.3249111495947457, -9.9684277340655711, -1.3266123160614773),
         (12.689851531339878, -4.4260877770435245, -8.9585086671785774),
         (10.103825228355211, 1.1076098797323481, -5.6331564229411617)],
        [(-5.1868371621186844, 4.7602528056675295, 0.97022697723726137),
         (-0.73355849180427846, 3.1853804927764848, 1.4184540026745367),
         (1.7370638323127894, 4.7726088993795681, -3.7548102282588882),
         (10.103825228355211, 1.1076098797323481, -5.6331564229411617)],
        [(-9.1092663228073292, -12.711321277810857, -0.77093266173210928),
         (-3.9344403681487776, -6.6256134176686521, -6.3569364954962628),
         (-3.9413840306534453, -1.1684651343084738, 0.77546233191951042),
         (-5.1868371621186844, 4.7602528056675295, 0.97022697723726137)]]
    )

    #with open('tests/coons1.pkl', 'rb') as f:
    #    patch1 = dill.load(f)

    #with open('tests/coons2.pkl', 'rb') as f:
    #    patch2 = dill.load(f)

    patch1 = Coons(*(NURBSpline(pts) for pts in pts1))
    patch2 = Coons(*(NURBSpline(pts) for pts in pts2))

    #patch1 = Coons(*(CubicSpline(*pts) for pts in pts1))
    #patch2 = Coons(*(CubicSpline(*pts) for pts in pts2))
    patch1.build_tree()
    patch2.build_tree()
    print(patch1._rc,)
    s = time.time()
    #yappi.set_clock_type("wall")  # Use set_clock_type("wall") for wall time
    #yappi.start()
    cc = surface_ppi(patch1, patch2, 0.01)
    #yappi.stop()
    #func_stats = yappi.get_func_stats()
    #func_stats.save(f"{__file__.replace('.py', '')}_{int(time.time())}.pstat", type='pstat')

    print(time.time() - s)
    print([np.array(c).tolist() for c in cc[0]])
