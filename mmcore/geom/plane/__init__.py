from collections import namedtuple

import numpy as np

from mmcore.geom.vectors import unit

_Plane = namedtuple("Plane", ["origin", "xaxis", "yaxis", 'zaxis'])


class Plane(_Plane):
    @property
    def normal(self):
        return self.zaxis


def create_plane(x=(1, 0, 0), y=(0, 1, 0), origin=(0, 0, 0)):
    if (x, y) == ((1, 0, 0), (0, 1, 0)):
        x = np.array(x)
        y = np.array(y)
        z = np.array((0, 0, 1))
    else:
        x = unit(np.array(x))
        y = unit(np.array(y))
        z = np.cross(x, y)
    return Plane(np.array(origin), x, y, z)


WXY = create_plane()


def transform_point3d(pt, trx):
    return (trx @ np.append(pt, [1]))[:3]


def transform_point2d(pt, trx):
    return (trx @ np.append(pt, [0, 1]))[:2]


import pyquaternion as pq


def rotate_plane(plane, angle=0.0, axis=None):
    if axis is None:
        axis = plane.normal

        return Plane(plane.origin,
                     rotate_from_origin(plane.xaxis, plane.origin, angle=angle, axis=axis),
                     rotate_from_origin(plane.yaxis, plane.origin, angle=angle, axis=axis),
                     plane.zaxis)
    else:
        axis = unit(axis)
        return Plane(plane.origin,
                     rotate_from_origin(plane.xaxis, plane.origin, angle=angle, axis=axis),
                     rotate_from_origin(plane.yaxis, plane.origin, angle=angle, axis=axis),
                     rotate_from_origin(plane.zaxis, plane.origin, angle=angle, axis=axis))


def rotate_from_origin(pt, origin, angle, axis=(0, 0, 1)):
    q = pq.Quaternion(axis=axis, angle=angle)
    ppt = np.array(pt)
    origin = np.array(origin)
    if len(pt) == 3:
        return transform_point3d(ppt - origin, q.transformation_matrix) + origin
    else:
        return transform_point2d(ppt - origin, q.transformation_matrix) + origin


def vectorize(**kws):
    def wrap(fun):
        return np.vectorize(fun, **kws)

    return wrap


def translate_plane(plane: Plane, vec: np.ndarray):
    return Plane(plane.origin + vec, plane.xaxis, plane.yaxis, plane.zaxis)


@vectorize(otypes=[float], excluded=['plane'], signature='(i)->(i)')
def local_to_world(pt, plane: Plane = WXY):
    z = np.zeros(3)
    z += plane.origin
    z += pt[0] * plane.xaxis
    z += pt[1] * plane.yaxis
    z += pt[2] * plane.zaxis
    return z


@vectorize(otypes=[float], excluded=['plane'], signature='(i)->(i)')
def world_to_local(pt, plane: Plane = WXY):
    return np.array([plane.xaxis, plane.yaxis, plane.zaxis]) @ (np.array(pt) - plane.origin)


@vectorize(otypes=[float], excluded=['plane_a', 'plane_b'], signature='(i)->(i)')
def plane_to_plane(pt, plane_a: Plane, plane_b: Plane):
    return local_to_world(world_to_local(pt, plane=plane_b), plane=plane_a)
