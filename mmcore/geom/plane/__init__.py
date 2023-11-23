from collections import namedtuple

import numpy as np

from mmcore.func import vectorize
from mmcore.geom.vectors import unit

_Plane = namedtuple("Plane", ["origin", "xaxis", "yaxis", 'zaxis'])
_PlaneGeneral = namedtuple("Plane", ["origin", "axises"])


class Plane:
    def __init__(self, arr=None):
        if arr is None:
            self._arr = np.zeros((4, 3))
        else:
            self._arr = np.array(arr)

    @property
    def normal(self):
        return self.zaxis

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
        self._arr[0:len(v)] = v

    @xaxis.setter
    def xaxis(self, v):
        self._arr[1:len(v)] = v

    @yaxis.setter
    def yaxis(self, v):
        self._arr[2:len(v)] = v

    @zaxis.setter
    def zaxis(self, v):
        self._arr[3, :len(v)] = v


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


def plane(origin, xaxis, yaxis, zaxis):
    return Plane((origin, xaxis, yaxis, zaxis))


WXY = create_plane()

from mmcore.geom.transform import rotate_around_axis


def rotate_plane(pln, angle=0.0, axis=None):
    if axis is None:
        axis = pln.normal

        return plane(pln.origin,
                     rotate_around_axis(pln.xaxis, angle, origin=pln.origin, axis=axis),
                     rotate_around_axis(pln.yaxis, angle, origin=pln.origin, axis=axis),
                     pln.zaxis)
    else:
        axis = unit(axis)
        return plane(pln.origin,
                     rotate_around_axis(pln.xaxis, angle, origin=pln.origin, axis=axis),
                     rotate_around_axis(pln.yaxis, angle, origin=pln.origin, axis=axis),
                     rotate_around_axis(pln.zaxis, angle, origin=pln.origin, axis=axis))


@vectorize(signature='(j),()->(i)')
def append_np(arr, val):
    return np.append(arr, val)


def translate_plane(pln: Plane, vec: np.ndarray):
    return Plane(np.array([pln.origin + vec, pln.xaxis, pln.yaxis, pln.zaxis]))


def translate_plane_inplace(pln: Plane, vec: np.ndarray):
    pln.origin += vec


def rotate_plane_inplace(pln: Plane, angle: float, axis: np.ndarray, ):
    if axis is None:
        axis = pln.normal

        pln._arr[1:3] = rotate_around_axis(pln.xaxis, angle, origin=pln.origin, axis=axis), rotate_around_axis(
            pln.yaxis,
            angle,
            origin=pln.origin,

            axis=axis),


    else:
        axis = unit(axis)
        pln._arr[1:] = [rotate_around_axis(pln.xaxis, angle, origin=pln.origin, axis=axis),
                        rotate_around_axis(pln.yaxis, angle, origin=pln.origin, axis=axis),
                        rotate_around_axis(pln.zaxis, angle, origin=pln.origin, axis=axis)]


@vectorize(otypes=[float], excluded=['pln'], signature='(i)->(i)')
def local_to_world(pt, pln: Plane = WXY):
    z = np.zeros(3)
    z += pln.origin
    z += pt[0] * pln.xaxis
    z += pt[1] * pln.yaxis
    z += pt[2] * pln.zaxis
    return z


@vectorize(otypes=[float], excluded=['pln'], signature='(i)->(i)')
def world_to_local(pt, pln: Plane = WXY):
    return np.array([pln.xaxis, pln.yaxis, pln.zaxis]) @ (np.array(pt) - pln.origin)


@vectorize(otypes=[float], excluded=['plane_a', 'plane_b'], signature='(i)->(i)')
def plane_to_plane(pt, plane_a: Plane, plane_b: Plane):
    return local_to_world(world_to_local(pt, pln=plane_b), pln=plane_a)


def gen_norms(spline, density=4, bounds=(0, 1)):
    for t in np.linspace(*bounds, len(spline.control_points) * density):
        pt = spline.tan(t)

        x = rotate_vector_2d(pt.normal)
        y = np.cross(pt.normal, x)

        yield plane(origin=pt.point, xaxis=x, yaxis=y, zaxis=pt.normal)


from mmcore.geom.parametric import algorithms as algo


def ray_intersection(ray: algo.Ray, plane: Plane):
    return algo.ray_plane_intersection(*ray, plane, full_return=True)
