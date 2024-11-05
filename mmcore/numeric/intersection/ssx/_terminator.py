import warnings
import weakref
from collections import namedtuple

import numpy as np
from scipy.spatial import KDTree

from mmcore.geom.nurbs import NURBSSurface
from mmcore.geom.surfaces import CurveOnSurface, Surface
from enum import Enum
from mmcore.numeric.vectors import scalar_norm
from mmcore.numeric.closest_point import closest_points_on_surface, closest_point_on_surface_batched, \
    closest_point_on_nurbs_surface
from mmcore.numeric.intersection.csx import curve_surface_intersection, nurbs_csx
from mmcore.numeric.intersection.ssx.boundary_intersection import extract_isocurve, find_boundary_intersections


class TerminatorType(int, Enum):
    FAIL = 0
    LOOP = 1
    EDGE = 2
    STEP = 3

# Maybe Deprecated
class Terminator:
    __slots__ = ('terminator_type', 'xyz', 'uv1', 'uv2', 'surf1', 'surf2', '_size', '_kd')

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
        if self._kd is None:
            return self.xyz[0], self.uv1[0], self.uv2[0]
        d, i = self._kd.query(other, k=1)
        return self.xyz[i], self.uv1[i], self.uv2[i]

    def get_uv(self, other):

        i = self._kd.query(other, k=1)
        return self.xyz[i], self.uv1[i], self.uv2[i]
    def __len__(self):
        return self._size
    def __bool__(self):
        return self._size > 0
    def __delitem__(self, key):
        np.delete(self.xyz,key)
        np.delete(self.uv1,key)
        np.delete(self.uv2,key)
        self._size -= 1
        if self._size>1:
            self._kd = KDTree(self.xyz)
        else:
            self._kd = None


    def remove(self, xyz):
        if self._kd is None and self._size==1:
            if np.linalg.norm(self.xyz[0]-xyz)<1e-6:
                self.__delitem__(0

                )
            else:
                print(xyz,self.xyz)
                raise ValueError(f'{np.linalg.norm(self.xyz.reshape((3,))-xyz)}<1e-6')


        elif  self._kd is None and self._size==0:
            return
        else:
            try:



                d,i = self._kd.query(xyz, k=1)
            except Exception as err:
                print(xyz,self.xyz)
            if d==0:


                self.__delitem__(i)
            else :
                print(d)
                raise ValueError(f'{d}')

    def pop(self):

        pt=self.xyz[0]
        u1=self.uv1[0]
        u2=self.uv2[0]
        self._size -= 1
        self.xyz=self.xyz[1:]
        if self._size>1:



            self._kd=KDTree(self.xyz)
        else:
            self._kd = None
        self.uv1 = self.uv1[1:]
        self.uv2 = self.uv2[1:]
        return pt,u1,u2

# Deprecated!!!
def build_boundary_if_not_present(surface: Surface):
    if surface.boundary is None:
        surface.build_boundary()


def surface_surface_intersection_edge_terminator(surf1: Surface, surf2: Surface, tol=1e-3) -> Terminator:
    """
    Find intersection points between surface boundaries.
    
    Args:
        surf1: First surface
        surf2: Second surface
        tol: Tolerance for intersection detection
        
    Returns:
        Terminator: Object containing intersection points and their parameters
    """
    if isinstance(surf1, NURBSSurface) and isinstance(surf2, NURBSSurface):
        uv1 = []
        uv2 = []
        xyz = []
        for l in find_boundary_intersections(surf1, surf2, tol=tol):
            if not tuple(l.point) in xyz and not l.surface1_params in uv1 and not l.surface2_params  in uv2:

                xyz.append(tuple(l.point))
                #print(l.surface1_params,closest_point_on_nurbs_surface( surf1, l.point, tol=tol))
                #print(l.surface2_params, closest_point_on_nurbs_surface(surf2, l.point, tol=tol))
                uv1.append(l.surface1_params)
                uv2.append(l.surface2_params)
                #uv1.append(l.surface1_params)
                #uv2.append(l.surface2_params)

    else:
        build_boundary_if_not_present(surf1)
        build_boundary_if_not_present(surf2)
        try:
            b1s2 = curve_surface_intersection(surf1.boundary, surf2, tol=tol)
        except Exception as e:
            warnings.warn(str(e))
            b1s2 = []
        try:
            b2s1 = curve_surface_intersection(surf2.boundary, surf1, tol=tol)
        except Exception as e:
            warnings.warn(str(e))
            b2s1 = []

        xyz = []
        uv1 = []
        uv2 = []
        if len(b1s2) > 0:
            prms1 = np.array(b1s2)
            xyz = surf1.boundary(prms1[:, 0])
            uv1 = surf1.boundary.curve(prms1[:, 0])[..., :2]
            uv2 = prms1[:, 1:]

        if len(b2s1) > 0:
            prms2 = np.array(b2s1)
            xyz = np.array([*xyz, *surf2.boundary(prms2[:, 0])])
            uv1 = np.array([*uv1, *prms2[:, 1:]])
            uv2 = np.array([*uv2, *surf2.boundary.curve(prms2[:, 0])[:, :2]])

    return Terminator(TerminatorType.EDGE, xyz=xyz,
                     uv1=uv1,
                     uv2=uv2,
                     surf1=surf1, surf2=surf2)