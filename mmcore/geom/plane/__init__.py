from collections import namedtuple

import numpy as np

from mmcore.func import vectorize
from mmcore.geom import vec
from mmcore.geom.vec import cross, dot, unit

_Plane = namedtuple("Plane", ["origin", "xaxis", "yaxis", 'zaxis'])
_PlaneGeneral = namedtuple("Plane", ["origin", "axises"])
from mmcore.base.ecs.components import component


@component()
class ArrayComponent:
    arr: np.ndarray = None

class Plane:
    """
    zaxis=cross(xaxis, yaxis)
    yaxis=cross(zaxis,xaxis)
    xaxis=cross( yaxis,zaxis)
    """

    def __init__(self, arr=None):
        self._bytes = None
        if arr is None:
            self._arr_cmp = ArrayComponent(np.zeros((4, 3)))
        else:
            self._arr_cmp = ArrayComponent(np.array(arr))

        self._dirty = True

    @property
    def _arr(self):
        return self._arr_cmp.arr

    @_arr.setter
    def _arr(self, v):
        self._arr_cmp.arr = np.array(v)
        self._dirty = True

    def to_bytes(self):
        self._solve_dirty()
        return self._bytes

    def _solve_dirty(self):
        if self._dirty:
            self._bytes = self._arr.tobytes()
            self._hash = hash(self._bytes)
            self._dirty = False

    def __hash__(self):
        self._solve_dirty()
        return self._hash
    @property
    def normal(self):
        return self.zaxis

    @normal.setter
    def normal(self, v):
        self.zaxis = v
        self._dirty = True


    @property
    def xaxis(self):
        return self._arr[1]

    @property
    def yaxis(self):
        return self._arr[2]

    @property
    def zaxis(self):
        return self._arr[3]

    @property
    def origin(self):
        return self._arr[0]

    @origin.setter
    def origin(self, v):

        self._arr[0, np.arange(len(v))] = v
        self._dirty = True

    @xaxis.setter
    def xaxis(self, v):

        self._arr[1, np.arange(len(v))] = v
        self._dirty = True

    @yaxis.setter
    def yaxis(self, v):
        self._arr[2, np.arange(len(v))] = v
        self._dirty = True

    @zaxis.setter
    def zaxis(self, v):
        self._arr[3, np.arange(len(v))] = v
        self._dirty = True


def create_plane(x=(1, 0, 0), y=None, origin=(0, 0, 0)):
    pln = Plane()
    pln.origin = origin

    if y is None:

        x, y = axis = unit([x, [-x[1], x[0], 0]])
        pln._arr[1:] = x, y, (0, 0, 1)


    else:
        x, y = unit([x, y])

        pln._arr[1:] = x, y, np.cross(x, y)

    return pln


def create_plane_from_xaxis_and_normal(xaxis=(1, 0, 0), normal=(0, 0, 1), origin=(0, 0, 0)):
    pln = Plane()
    pln.origin = origin
    pln.xaxis = vec.unit(xaxis)
    pln.zaxis = vec.unit(normal)
    pln.yaxis = cross(pln.zaxis, pln.xaxis)

    return pln


def plane(origin, xaxis, yaxis, zaxis):
    return Plane((origin, xaxis, yaxis, zaxis))


@vectorize(excluded=[0], signature='(i)->(i)')
def project(pln, pt):
    return pt - (dot(pln.normal, pt - pln.origin) * pln.normal)


WXY = create_plane()

from mmcore.geom.transform import rotate_around_axis


def rotate_plane(pln, angle=0.0, axis=None, origin=None):
    if origin is None:
        origin = pln.origin
    if axis is None:
        axis = pln.normal
    else:
        axis = unit(axis)

    xyz = rotate_around_axis(pln.origin + np.array([pln.xaxis, pln.yaxis, pln.zaxis]), angle, origin=origin,
                             axis=axis)
    origin = rotate_around_axis(pln.origin, angle, origin=origin, axis=axis)
    xaxis, yaxis, zaxis = unit(xyz - pln.origin)
    return plane(origin, xaxis, yaxis, zaxis)




@vectorize(signature='(j),()->(i)')
def append_np(arr, val):
    return np.append(arr, val)


def translate_plane(pln: Plane, vec: np.ndarray):
    return Plane(np.array([pln.origin + vec, pln.xaxis, pln.yaxis, pln.zaxis]))


def translate_plane_inplace(pln: Plane, vec: np.ndarray):
    pln.origin += vec


def rotate_plane_inplace(pln: Plane, angle: float, axis: np.ndarray = None, origin=None):
    if origin is None:
        origin = pln.origin
    if axis is None:
        axis = pln.normal



    else:
        axis = unit(axis)

    xyz = rotate_around_axis(pln.origin + np.array([pln.xaxis, pln.yaxis, pln.zaxis]), angle, origin=origin,
                             axis=axis)
    pln.origin = rotate_around_axis(pln.origin, angle, origin=origin, axis=axis)
    pln.xaxis, pln.yaxis, pln.zaxis = unit(xyz - pln.origin)


@vectorize(otypes=[float], excluded=[1], signature='(i)->(i)')
def local_to_world(pt, pln: Plane = WXY):
    z = np.zeros(3)
    z += pln.origin
    z += pt[0] * pln.xaxis
    z += pt[1] * pln.yaxis
    z += pt[2] * pln.zaxis
    return z


@vectorize(excluded=[1], signature='(i)->(i)')
def world_to_local(pt, pln: Plane):
    return np.array([pln.xaxis, pln.yaxis, pln.zaxis]) @ (np.array(pt) - pln.origin)


@vectorize(excluded=[1, 2], signature='(i)->(i)')
def plane_to_plane(pt, plane_a: Plane, plane_b: Plane):
    return local_to_world(world_to_local(pt, plane_b), plane_a)


def gen_norms(spline, density=4, bounds=(0, 1)):
    for t in np.linspace(*bounds, len(spline.control_points) * density):
        pt = spline.tan(t)

        x = rotate_vector_2d(pt.normal)
        y = np.cross(pt.normal, x)

        yield plane(origin=pt.point, xaxis=x, yaxis=y, zaxis=pt.normal)


from mmcore.geom.parametric import algorithms as algo


def ray_intersection(ray: algo.Ray, plane: Plane):
    return algo.ray_plane_intersection(*ray, plane, full_return=True)
