import warnings
import weakref
from collections import namedtuple

import numpy as np
from scipy.spatial import KDTree

from mmcore.geom.nurbs import NURBSSurface, extract_isocurve
from mmcore.geom.surfaces import CurveOnSurface, Surface

from enum import Enum

from mmcore.numeric.vectors import scalar_norm

from mmcore.numeric.closest_point import closest_points_on_surface, closest_point_on_surface_batched
from mmcore.numeric.intersection.csx import curve_surface_intersection, nurbs_csx


class TerminatorType(int, Enum):
    FAIL = 0
    LOOP = 1
    EDGE = 2
    STEP = 3


class Terminator:
    __slots__ = ('terminator_type', 'xyz', 'uv1', 'uv2', 'surf1', 'surf2', '_size','_kd')

    def __init__(self, terminator_type: TerminatorType, xyz, uv1, uv2, surf1, surf2):
        self._size = len(xyz)

        self.terminator_type = terminator_type
        self.xyz = np.array(xyz).reshape((self._size, 3))
        self.uv1 = np.array(uv1).reshape((self._size, 2))
        self.uv2 = np.array(uv2).reshape((self._size, 2))
        self._kd = KDTree(self.xyz)

        self.surf1 = surf1
        self.surf2 = surf2


    def check_xyz(self, other):
        d = self.xyz - np.array(other)
        return scalar_norm(d)

    def check_uv1(self, other):
        other = np.array(other)

        d = self.uv1 - np.array(other)
        return scalar_norm(d)

    def check_uv2(self, other):
        other = np.array(other)

        d = self.uv2 - np.array(other)
        return scalar_norm(d)
    def get_closest(self, other):
        d,i=self._kd.query(other,k=1)

        return self.xyz[i],self.uv1[i],self.uv2[i]
    def get_uv(self, other):
        i=self._kd.query(other,k=1)
        return self.xyz[i],self.uv1[i],self.uv2[i]

def build_boundary_if_not_present(surface: Surface):
    if surface.boundary is None:
        surface.build_boundary()


def surface_surface_intersection_edge_terminator(surf1: Surface, surf2: Surface,tol=1e-3) -> Terminator:
    """



    :param boundary:
      :param surf2:
    :return: t params, uv's params of boundary surface , uv's params of surf2
    """
    #print('surface_surface_intersection_edge_terminator')
    if isinstance(surf1, NURBSSurface) and isinstance(surf2, NURBSSurface):
        uv1=[]
        uv2 = []
        xyz=[]
        for l in (extract_isocurve(surf1, 0., 'v'),
                  extract_isocurve(surf1, 0., 'u'),
                  extract_isocurve(surf1, 1., 'u'),
                  extract_isocurve(surf1, 1., 'v')):
            print(l.interval(),surf2.interval())
            try:
                rr=nurbs_csx(l, surf2, tol=tol, ptol=1e-7)
                if len(rr)>0:
                    _,pts,res=zip(*rr)

                    res = np.array(res)

                    uvj = res[:, 1:]
                    #pt=np.array(surf2.evaluate_multi(res[:, 1:]))
                    xyz.extend(pts)
                    def fun(x):
                        return surf1.evaluate_multi(np.array(x))
                    fun.interval=surf1.interval

                    uvi = closest_point_on_surface_batched(fun, np.array(pts),1e-4)
                    uv1.extend(uvi)
                    uv2.extend(uvj)
            except Exception as e:
                pass
        for l in (extract_isocurve(surf2, 0., 'v'),
                  extract_isocurve(surf2, 0., 'u'),
                  extract_isocurve(surf2, 1., 'u'),
                  extract_isocurve(surf2, 1., 'v')):
            try:
                rr=nurbs_csx(l, surf1, tol=tol, ptol=1e-6)
                if len(rr) > 0:

                    _, pts, res = zip(*rr)

                    res = np.array(res)

                    uvj = res[:, 1:]
                    #pt = np.array(surf1.evaluate_multi(res[:, 1:]))
                    xyz.extend(pts)

                    def fun(x):
                        return surf2.evaluate_multi(np.array(x))

                    fun.interval = surf2.interval
                    uvi = closest_point_on_surface_batched(fun, np.array(pts),1e-4)
                    uv2.extend(uvi)
                    uv1.extend(uvj)
            except Exception as err:
                pass
    else:
        build_boundary_if_not_present(surf1)
        build_boundary_if_not_present(surf2)
        try:
            b1s2 = curve_surface_intersection(surf1.boundary, surf2,tol=tol)
        except Exception as e:
            warnings.warn(str(e))
            b1s2=[]
        try:
            b2s1 = curve_surface_intersection(surf2.boundary, surf1, tol=tol)
        except Exception as e:
            warnings.warn(str(e))
            b2s1 =[]

        trms = []
        xyz = []
        uv1 = []
        uv2 = []
        if len(b1s2) > 0:
            prms1 = np.array(b1s2)
            xyz = surf1.boundary(prms1[:, 0])
            uv1 = surf1.boundary.curve(prms1[:, 0])[...,:2]
            uv2 = prms1[:,1:]

        if len(b2s1) > 0:
            prms2 = np.array(b2s1)
            xyz = np.array([*xyz, *surf2.boundary(prms2[:, 0])])
            uv1 = np.array([*uv1, *prms2[:, 1:]])
            uv2 = np.array([*uv2, *surf2.boundary.curve(prms2[:,0])[:, :2]])

    return Terminator(TerminatorType.EDGE, xyz=xyz,
                      uv1=uv1,
                      uv2=uv2,
                      surf1=surf1, surf2=surf2)
