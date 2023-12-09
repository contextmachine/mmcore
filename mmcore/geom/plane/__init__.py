from collections import namedtuple

import numpy as np
from multipledispatch import dispatch

from mmcore.func import vectorize
from mmcore.geom import vec
from mmcore.geom.vec import cross, dot, norm, unit

_Plane = namedtuple("Plane", ["origin", "xaxis", "yaxis", 'zaxis'])
_PlaneGeneral = namedtuple("Plane", ["origin", "axises"])
from mmcore.base.ecs.components import EcsProto, component, EcsProperty


@component()
class PlaneComponent:
    ref: np.ndarray = None
    origin: int = 0
    xaxis: int = 1
    yaxis: int = 2
    zaxis: int = 3



@component()
class PlaneParamsArray:
    arr: np.ndarray = None


NpPlane = np.void(0, dtype=np.dtype([('origin', float, (3,)),
                                     ('xaxis', float, (3,)),
                                     ('yaxis', float, (3,)),
                                     ('zaxis', float, (3,))]))


@dispatch(object, object)
def np_plane(xaxis, yaxis):
    xaxis, yaxis = unit([xaxis, yaxis])
    zaxis = cross(xaxis, yaxis)
    pln = np.array(0, dtype=NpPlane)
    pln[1] = xaxis
    pln[2] = yaxis
    pln[3] = zaxis
    return pln


@dispatch(object, object, object)
def np_plane(xaxis, yaxis, origin):
    xaxis, yaxis = unit([xaxis, yaxis])
    zaxis = cross(xaxis, yaxis)
    pln = np.array(0, dtype=NpPlane)
    pln[0] = origin
    pln[1] = xaxis
    pln[2] = yaxis
    pln[3] = zaxis
    return pln



class Plane(EcsProto):
    """
    zaxis=cross(xaxis, yaxis)
    yaxis=cross(zaxis,xaxis)
    xaxis=cross( yaxis,zaxis)
    """

    ecs_plane = EcsProperty(type=PlaneComponent)

    def __init__(self, arr=None):

        super().__init__()
        self._bytes = None
        self.ecs_plane = PlaneComponent(ref=np.append([0., 0., 0.],
                                                      np.eye(3, 3)).reshape((4, 3)) if arr is None else arr)
        self._dirty = True

    def replace_component(self, comp: PlaneComponent):
        self.ecs_components[0] = comp

    @property
    def _arr_cmp(self):
        return self.ecs_components[0]

    @_arr_cmp.setter
    def _arr_cmp(self, v):
        self.ecs_components[0] = v
    @property
    def _arr(self):
        return self.ecs_plane.ref

    @_arr.setter
    def _arr(self, v):
        self.ecs_plane.ref[:] = np.array(v)
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
        return self._arr[self.ecs_plane.xaxis]

    @property
    def yaxis(self):
        return self._arr[self.ecs_plane.yaxis]

    @property
    def zaxis(self):
        return self._arr[self.ecs_plane.zaxis]

    @property
    def origin(self):
        return self._arr[self.ecs_plane.origin]

    @origin.setter
    def origin(self, v):
        self._arr[self.ecs_plane.origin, np.arange(len(v))] = v
        self._dirty = True

    @xaxis.setter
    def xaxis(self, v):
        self._arr[self.ecs_plane.xaxis, np.arange(len(v))] = v

        self._dirty = True

    @yaxis.setter
    def yaxis(self, v):
        self._arr[self.ecs_plane.yaxis, np.arange(len(v))] = v
        self._dirty = True

    @zaxis.setter
    def zaxis(self, v):
        self._arr[self.ecs_plane.zaxis, np.arange(len(v))] = v
        self._dirty = True

    @property
    def d(self):
        return -1 * (
                self.normal[0] * self.origin[0] + self.normal[1] * self.origin[1] + self.normal[2] * self.origin[2])

    def todict(self):
        return dict(origin=self.origin.tolist(), xaxis=self.xaxis.tolist(), yaxis=self.yaxis.tolist(),
                    zaxis=self.zaxis.tolist())


class SlimPlane:
    def __init__(self, arr: np.ndarray = None):
        if arr is None:
            arr = np.zeros((4, 3))
            arr[1:, :] = np.eye(3)

        self._array = arr if isinstance(arr, np.ndarray) else np.array(arr, dtype=float)

    @property
    def origin(self):
        return self._array[0]

    @origin.setter
    def origin(self, v):
        self._array[0] = v

    @property
    def xaxis(self):
        return self._array[1]

    @xaxis.setter
    def xaxis(self, v):
        self._array[1] = v

    @property
    def yaxis(self):
        return self._array[2]

    @yaxis.setter
    def yaxis(self, v):
        self._array[2] = v

    @property
    def zaxis(self):
        return self._array[3]

    @zaxis.setter
    def zaxis(self, v):
        self._array[3] = v

    @property
    def normal(self):
        return self._array[3]

    @normal.setter
    def normal(self, v):
        self._array[3] = v

    def __hash__(self):
        return hash(self._array.tobytes())

    @property
    def d(self):
        return -1 * (
                self.normal[0] * self.origin[0] + self.normal[1] * self.origin[1] + self.normal[2] * self.origin[2])

    def todict(self):
        return dict(origin=self.origin.tolist(), xaxis=self.xaxis.tolist(), yaxis=self.yaxis.tolist(),
                    zaxis=self.zaxis.tolist())

    @classmethod
    def from_normal(cls, normal: np.ndarray, origin: np.ndarray = np.zeros(3, float)):
        return cls(plane_from_normal_numeric(normal, origin))

    @classmethod
    def from_3pt(cls, origin: np.ndarray, pt1: np.ndarray, pt2: np.ndarray):
        normal = unit(
            cross(
                *unit(
                    (np.array(pt1) - np.array(origin),
                     np.array(pt2) - np.array(origin))
                )
            )
        )

        return cls.from_normal(normal=normal, origin=origin)

    def project(self, pts: np.ndarray):
        return project(self, pts)

    @vectorize(excluded=[0], signature='(i)->(i)')
    def point_from_local_to_world(self, pt: np.ndarray):
        return self._array[0] + self._array[1] * pt[0] + self._array[2] * pt[1] + self._array[3] * pt[2]

    @vectorize(excluded=[0], signature='(i)->(i)')
    def point_from_world_to_plane(self, pt: np.ndarray):
        return dot(pt - self._array[0], self._array[1:])


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
def local_to_world(pt, pln: 'Plane|SlimPlane' = WXY):
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


def rotate_vectors_around_plane(vecs, plane_a, angle):
    return norm(vecs) * rotate_around_axis(unit(vecs), angle, origin=(0., 0., 0.), axis=plane_a.normal)


@vectorize(excluded=[0, 1], signature='()->()')
def rotate_plane_around_plane(plane1, plane2, angle):
    """
    Parameters
    ----------
    :param plane1: Plane который планируется повернуть
    :param plane2: Plane в системе которого происходит поворот
    :param angle:  Угол поворота в радианах
    :type angle: float
    :returns new_plane: Новый Plane в глобальных координатах
    :rtype: Plane
    """
    origin = rotate_around_axis(plane1.origin, angle, plane2.origin, plane2.normal)
    xaxis, yaxis, zaxis = rotate_around_axis(np.array([plane1.xaxis, plane1.yaxis, plane1.zaxis]), angle,
                                             origin=(0., 0., 0.), axis=plane2.normal)
    return Plane(np.array([origin, xaxis, yaxis, zaxis], dtype=float))


@vectorize(signature='(i),(i)->(j,i)')
def plane_from_normal_numeric(vector=(2, 33, 1), origin=(0., 0.0, 0.)):
    Z = unit(vector)

    axises = cross(Z, sorted(np.eye(3), key=lambda x: dot(Z, x)))
    Y = axises[0]
    X = cross(Y, Z)
    return np.array([origin, X, Y, Z])


def plane_from_normal(vector=(2, 33, 1), origin=(0., 0.0, 0.)):
    return Plane(plane_from_normal_numeric(vector, origin))


@vectorize(signature='(j,i)->()')
def test_plane_num(pln):
    X, Y, Z = pln[1:]
    return np.allclose([dot(X, Y), dot(Y, Z), dot(Z, X)], 0)


def is_parallel(self, other):
    _cross = cross(unit(self.normal), unit(other.normal))
    A = np.array([self.normal, other.normal, _cross])
    return np.linalg.det(A) == 0
