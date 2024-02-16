from collections import namedtuple
from uuid import uuid4

import numpy as np
from multipledispatch import dispatch

from mmcore.func import vectorize
from mmcore.geom import vec
from mmcore.geom.plane.refine import PlaneRefine
from mmcore.geom.vec import cross, dot, norm, perp2d, unit

_Plane = namedtuple("Plane", ["origin", "xaxis", "yaxis", 'zaxis'])
_PlaneGeneral = namedtuple("Plane", ["origin", "axises"])
from mmcore.base.ecs.components import EcsProto, component, EcsProperty


def pln_eq_3pt(p0, p1, p2):
    matrix_a = np.array([[p0[1], p0[2], 1],
                         [p1[1], p1[2], 1],
                         [p2[1], p2[2], 1]])
    matrix_b = np.array([[-p0[0], p0[2], 1],
                         [-p1[0], p1[2], 1],
                         [-p2[0], p2[2], 1]])
    matrix_c = np.array([[p0[0], p0[1], 1],
                         [p1[0], p1[1], 1],
                         [p2[0], p2[1], 1]])
    matrix_d = np.array([[-p0[0], -p0[1], p0[2]],
                         [-p1[0], -p1[1], p1[2]],
                         [-p2[0], -p2[1], p2[2]]])
    det_a = np.linalg.det(matrix_a)
    det_b = np.linalg.det(matrix_b)
    det_c = np.linalg.det(matrix_c)
    det_d = -np.linalg.det(matrix_d)
    return np.array([det_a, det_b, det_c, det_d])



NpPlane = np.void(0, dtype=np.dtype([('origin', float, (3,)),
                                     ('xaxis', float, (3,)),
                                     ('yaxis', float, (3,)),
                                     ('zaxis', float, (3,))]))


@dispatch(object, object)
def np_plane(xaxis, yaxis):
    """
    :param xaxis: The x-axis vector of the plane.
    :type xaxis: object
    :param yaxis: The y-axis vector of the plane.
    :type yaxis: object
    :return: The plane defined by the given x-axis and y-axis vectors.
    :rtype: object

    """
    xaxis, yaxis = unit([xaxis, yaxis])
    zaxis = cross(xaxis, yaxis)
    pln = np.array(0, dtype=NpPlane)
    pln[1] = xaxis
    pln[2] = yaxis
    pln[3] = zaxis
    return pln


@dispatch(object, object, object)
def np_plane(xaxis, yaxis, origin):
    """
    :param xaxis: The x-axis vector of the plane.
    :type xaxis: object
    :param yaxis: The y-axis vector of the plane.
    :type yaxis: object
    :param origin: The origin point of the plane.
    :type origin: object
    :return: The plane represented as a numpy array.
    :rtype: object

    """
    xaxis, yaxis = unit([xaxis, yaxis])
    zaxis = cross(xaxis, yaxis)
    pln = np.array(0, dtype=NpPlane)
    pln[0] = origin
    pln[1] = xaxis
    pln[2] = yaxis
    pln[3] = zaxis
    return pln


@vectorize(excluded=[0], signature='(i)->()')
def distance(self, pt):
    return point_to_plane_distance(pt, self.origin, self.normal)


class Entity:
    def __init__(self, uuid=None):

        super().__init__()
        if uuid is None:
            uuid = uuid4().hex

        self._uuid = uuid

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, v):
        self._uuid = v


class Plane(Entity):
    """
    zaxis=cross(xaxis, yaxis)
    yaxis=cross(zaxis,xaxis)
    xaxis=cross( yaxis,zaxis)
    """

    def __init__(self, arr=None, origin=None, xaxis=None, yaxis=None, normal=None, uuid=None):
        super().__init__(uuid=uuid)

        if (xaxis is not None) and (normal is not None):
            xaxis = unit(xaxis)
            zaxis = unit(normal)
            yaxis = unit(cross(normal, xaxis))
            self._array = np.array([xaxis, yaxis, zaxis])

        elif (xaxis is not None) and (yaxis is not None):
            xaxis = unit(xaxis)
            yaxis = unit(yaxis)
            zaxis = unit(cross(xaxis, yaxis))

            self._array = np.array([xaxis, yaxis, zaxis])

        elif (normal is not None) and (yaxis is not None):
            yaxis = unit(yaxis)
            zaxis = unit(normal)
            xaxis = unit(cross(yaxis, zaxis))
            self._array = np.array([xaxis, yaxis, zaxis])

        else:
            if arr is None:

                self._array = np.append([0., 0., 0.], np.eye(3, 3)).reshape((4, 3))
            else:
                arr[1:] = unit(arr[1:])
                self._array = arr
        if origin is not None:
            self.origin = origin

        self._bytes = None
        self._dirty = True
        self._multiple = False
        if len(self._array.shape) > 2:
            self._multiple = True

    @property
    def local_axis(self):
        return self._array[1:]

    @local_axis.setter
    def local_axis(self, v):
        self._array[1:] = v

    def refine(self, proprity_axis=('y', 'x')):

        refine = PlaneRefine(*proprity_axis)

        self._array[1:, ...] = np.swapaxes(refine(np.swapaxes(self._array, 0, 1)[:, 1:, ...], inplace=False), 0, 1)

    def distance(self, pt):
        return distance(self, pt)

    def project(self, pt):
        return project(self, pt)


    @property
    def _arr(self):
        return self._array

    @_arr.setter
    def _arr(self, v):
        self._array[:] = np.array(v)
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
    def origin(self):
        return self.local_origin

    @origin.setter
    def origin(self, v):
        self.local_origin = v

    @property
    def local_origin(self):
        return self._arr[0]

    @local_origin.setter
    def local_origin(self, v):
        self._arr[0, np.arange(len(v))] = v
        self._dirty = True

    @property
    def axis(self):
        return self.local_axis

    @axis.setter
    def axis(self, v):
        self.local_axis = v

    def _get_axis(self, i):
        return self.axis[i]

    def _set_axis(self, i, v):
        self.axis[i] = v

    @property
    def xaxis(self):
        return self._get_axis(0)

    @property
    def yaxis(self):
        return self._get_axis(1)

    @xaxis.setter
    def xaxis(self, v):
        self._set_axis(0, v)


    @yaxis.setter
    def yaxis(self, v):
        self._set_axis(1, v)

    @property
    def zaxis(self):
        return self._get_axis(2)

    @zaxis.setter
    def zaxis(self, v):
        self._set_axis(2, v)

    @property
    def normal(self):
        return self.zaxis

    @normal.setter
    def normal(self, v):
        self.zaxis = v






    @property
    def d(self):
        return -1 * (
                self.normal[0] * self.origin[0] + self.normal[1] * self.origin[1] + self.normal[2] * self.origin[2])

    def todict(self):
        return dict(origin=self.origin.tolist(), xaxis=self.xaxis.tolist(), yaxis=self.yaxis.tolist(),
                    zaxis=self.zaxis.tolist())

    @vectorize(excluded=[0], signature='(i)->(i)')
    def point_from_local_to_world(self, pt: np.ndarray):
        """

             :param pts: Points in local coordinates
             :type pts: np.ndarray
             :return: Points in world coordinates
             :rtype: np.ndarray
             """
        return self._arr[0] + self._arr[1] * pt[0] + self._arr[2] * pt[1] + self._arr[3] * pt[2]

    @vectorize(excluded=[0], signature='(i)->(i)')
    def point_from_world_to_plane(self, pts: np.ndarray):
        """

              :param pts: Points in world coordinates
              :type pts: np.ndarray
              :return: Points in local coordinates
              :rtype: np.ndarray
              """
        return dot(pts - self._array[0], self._array[1:])

    @vectorize(excluded=[0], signature='(i)->( i)')
    def _evaluate(self, uvh):
        return self.origin + self.axis[0] * uvh[0] + self.axis[1] * uvh[1] + self.axis[2] * uvh[2]
    def __call__(self, uvh):
        if self._multiple:
            return self._evaluate_multi(uvh)
        else:
            return self._evaluate(uvh)

    @vectorize(excluded=[0], signature='(i)->( k, i)')
    def _evaluate_multi(self, uvh):
        return self.origin + self.axis[0] * uvh[0] + self.axis[1] * uvh[1] + self.axis[2] * uvh[2]

    @vectorize(excluded=[0], signature='(i)->(i)')
    def at_local(self, pt):
        return dot(self.axis, pt - self.origin)

    def create_relative(self, arr, **kwargs) -> 'ChildPln':
        return RelativePlane(self, arr, parent=self, **kwargs)

    def to_relative(self, parent) -> 'RelativePlane':
        pln = RelativePlane(parent=parent)
        pln._array = self._array
        return pln

    def point_at(self, pts):
        """

        :param pts: Points in local coordinates
        :type pts: np.ndarray
        :return: Points in world coordinates
        :rtype: np.ndarray
        """
        return self.point_from_local_to_world(pts)

    def in_plane_coordinates(self, pts):
        """

        :param pts: Points in world coordinates
        :type pts: np.ndarray
        :return: Points in local coordinates
        :rtype: np.ndarray
        """
        return self.point_from_world_to_plane(pts)

    def rotate(self, angle, axis=2, pln=None) -> 'Plane':
        if pln is None:
            pln = self
        return Plane(arr=np.array([
            rotate_around_axis(self.origin, angle, origin=pln.origin,
                               axis=pln.axis),
            *rotate_vectors_around_plane(self.axis, pln, angle, axis)
        ]))

    def translate(self, vector: np.ndarray) -> 'Plane':

        pln = Plane(self._array)
        pln.origin = self.origin + self.point_at(vector)
        return pln

class RelativePlane(Plane):
    def __init__(self, arr=None, parent: Plane = None, **kwargs):
        super().__init__(arr, **kwargs)
        self.parent = parent

    @property
    def origin(self):
        return self.parent(self.local_origin)

    @origin.setter
    def origin(self, v):
        self.local_origin = self.parent.at_local(np.array(v))

    @property
    def axis(self):
        return self.parent(self.local_axis)
    @origin.setter
    def axis(self, v):
        self.local_axis = self.parent.at_local(v + self.origin)

    def _set_axis(self, i, v):
        self.local_axis[i] = self.parent.at_local(np.array(v) + self.origin)


def create_plane(x=(1, 0, 0), y=None, origin=(0, 0, 0)):
    """

    """
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
    """

    """
    pln = Plane()
    pln.origin = origin
    pln.xaxis = vec.unit(xaxis)
    pln.zaxis = vec.unit(normal)
    pln.yaxis = cross(pln.zaxis, pln.xaxis)

    return pln


def plane(origin, xaxis, yaxis, zaxis):
    """

    """
    return Plane((origin, xaxis, yaxis, zaxis))


@vectorize(excluded=[0], signature='(i)->(i)')
def project(pln, pt):
    """
    Calculate the projection of a point onto a plane.

    :param pln: The plane to project the point onto.
    :type pln: Plane|SlimPlane
    :param pt: The point to be projected onto the plane.
    :type pt: ndarray (shape: (3,))
    :return: The projected point.
    :rtype: ndarray (shape: (3,))
    """
    return pt - (dot(pln.normal, pt - pln.origin) * pln.normal)


WXY = create_plane()

from mmcore.geom.transform import rotate_around_axis, axis_rotation_transform


def rotate_plane(pln, angle=0.0, axis=None, origin=None):
    """

    """
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
    """
    :param arr: numpy array
    :type arr: numpy.ndarray
    :param val: value to append to the array
    :type val: any
    :return: new numpy array with the value appended
    :rtype: numpy.ndarray
    """
    return np.append(arr, val)


def translate_plane(pln: Plane, vec: np.ndarray):
    """

    """
    return Plane(np.array([pln.origin + vec, pln.xaxis, pln.yaxis, pln.zaxis]))


def translate_plane_inplace(pln: Plane, vec: np.ndarray):
    """
    Translate the given plane by the given vector, inplace.

    :param pln: The plane object to be translated.
    :type pln: Plane|SlimPlane
    :param vec: The vector used for translation.
    :type vec: numpy.ndarray
    :return: None
    :rtype: None

    """
    pln.origin += vec


def rotate_plane_inplace(pln: Plane, angle: float, axis: np.ndarray = None, origin=None):
    """

    """
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
    """
    :param pt: The point in the local coordinate system that needs to be transformed to the world coordinate system.
    :type pt: numpy.ndarray

    :param pln: The reference plane in the world coordinate system that defines the transformation.
                It can be either a Plane or a SlimPlane object.
    :type pln: Plane or SlimPlane.

    :return: The transformed point in the world coordinate system.
    :rtype: numpy.ndarray
    """
    z = np.zeros(3)
    z += pln.origin
    z += pt[0] * pln.xaxis
    z += pt[1] * pln.yaxis
    z += pt[2] * pln.zaxis
    return z


@vectorize(excluded=[1], signature='(i)->(i)')
def world_to_local(pt, pln: Plane):
    """

    """
    return np.array([pln.xaxis, pln.yaxis, pln.zaxis]) @ (np.array(pt) - pln.origin)


def orient_plane(pln1: np.ndarray, pln2: np.ndarray):
    return np.vstack([pln2[0] - pln1[0], pln1[1:] @ pln2[1:].T])



@vectorize(excluded=[1, 2], signature='(i)->(i)')
def plane_to_plane(pt, plane_a: Plane, plane_b: Plane):
    """

    """
    return local_to_world(world_to_local(pt, plane_b), plane_a)


def gen_norms(spline, density=4, bounds=(0, 1)):
    """

    """
    for t in np.linspace(*bounds, len(spline.control_points) * density):
        pt = spline.tan(t)

        x = perp2d(pt.normal)
        y = cross(pt.normal, x)

        yield plane(origin=pt.point, xaxis=x, yaxis=y, zaxis=pt.normal)


from mmcore.geom.parametric import algorithms as algo, point_to_plane_distance


def ray_intersection(ray: algo.Ray, pln: Plane):
    """Find the intersection point between a ray and a plane.

    :param ray: The ray for which to find the intersection point.
    :type ray: algo.Ray
    :param pln: The plane with which to intersect the ray.
    :type pln: Plane|SlimPlane
    :return: The intersection point between the ray and the plane.
    :rtype: np.ndarray|tuple
    """
    return algo.ray_plane_intersection(*ray, pln, full_return=True)


def rotate_vectors_around_plane(vecs, plane_a, angle, axis=2):
    """
    Rotate vectors around a plane.

    :param vecs: The vectors to rotate.
    :type vecs: list or numpy.array
    :param plane_a: The plane around which to rotate the vectors.
    :type plane_a: Plane|SlimPlane
    :param angle: The angle in radians by which to rotate the vectors.
    :type angle: float
    :return: The rotated vectors.
    :rtype: numpy.array
    """
    return norm(vecs) * rotate_around_axis(unit(vecs), angle, origin=(0., 0., 0.), axis=plane_a.axis[axis])


@vectorize(excluded=[0, 1, 2], signature='()->()')
def rotate_plane_around_plane(plane1, plane2, angle, return_cls=Plane):
    """
    Rotate a plane around another plane.

    :param plane1: The plane to be rotated. Plane который планируется повернуть.
    :type plane1: Plane|SlimPlane
    :param plane2: The plane around which plane1 should be rotated. Plane в системе которого происходит поворот.
    :type plane2: Plane|SlimPlane
    :param angle: The angle of rotation in radians. Угол поворота в радианах.
    :type angle: float
    :param return_cls: The class to return
    :type return_cls: type[Plane]|type[SlimPlane] or other plane-like primitive.
    :return: The rotated plane. Новый Plane в глобальных координатах.
    :rtype: Plane|SlimPlane

    """
    origin = rotate_around_axis(plane1.origin, angle, plane2.origin, plane2.normal)
    xaxis, yaxis, zaxis = rotate_around_axis(np.array([plane1.xaxis, plane1.yaxis, plane1.zaxis]), angle,
                                             origin=(0., 0., 0.), axis=plane2.normal)
    return return_cls(np.array([origin, xaxis, yaxis, zaxis], dtype=float))


@vectorize(signature='(i),(i)->(j,i)')
def plane_from_normal_numeric(vector=(2., 33., 1.), origin=(0., 0.0, 0.)):
    """
    :param vector: The normal vector of the plane
    :type vector: tuple of floats or ndarray of floats
    :param origin: The origin point of the plane
    :type origin: tuple of floats or ndarray of floats
    :return: The plane defined by the normal vector and origin point
    :rtype: numpy array

    This method takes a normal vector and origin point as input and returns a plane defined by the normal vector and origin point. The normal vector should be a tuple of integers, and the
    * origin point should be a tuple of floats. The method calculates the unit normal vector, finds the axises perpendicular to the normal vector, and then calculates the X, Y, and Z components
    * of the plane. The result is returned as a numpy array.
    """
    Z = unit(vector)

    axises = cross(Z, sorted(np.eye(3), key=lambda x: dot(Z, x)))
    Y = axises[0]
    X = cross(Y, Z)
    return np.array([origin, X, Y, Z])


def plane_from_normal(vector=(2, 33, 1), origin=(0., 0.0, 0.), return_cls: 'Plane|SlimPlane' = Plane):
    """
    Create a Plane object from a normal vector and origin point.

    :param vector: The normal vector of the plane.
    :type vector: tuple, optional
    :param origin: The origin point of the plane.
    :type origin: tuple, optional
    :return: The Plane object.
    :rtype: Plane|SlimPlane

    """
    return return_cls(plane_from_normal_numeric(vector, origin))




def is_parallel(self, other):
    """
    Checks if two vectors are parallel.

    :param self: First vector.
    :type self: Vector

    :param other: Second vector.
    :type other: Vector

    :return: True if the vectors are parallel, False otherwise.
    :rtype: bool
    """
    _cross = cross(unit(self.normal), unit(other.normal))
    A = np.array([self.normal, other.normal, _cross])
    return np.linalg.det(A) == 0


import pyquaternion as pq


def orient_matrix(p1, p2):
    trx = np.eye(4)
    trx[:3, :3] = p1._array[1:] @ p2._array[1:].T
    trx[-1, :-1] = p2._array[0] - p1._array[0]
    return trx
