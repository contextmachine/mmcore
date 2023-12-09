from collections import namedtuple

import numpy as np
from multipledispatch import dispatch

from mmcore.func import vectorize
from mmcore.geom import vec
from mmcore.geom.vec import cross, dot, norm, perp2d, unit

_Plane = namedtuple("Plane", ["origin", "xaxis", "yaxis", 'zaxis'])
_PlaneGeneral = namedtuple("Plane", ["origin", "axises"])
from mmcore.base.ecs.components import EcsProto, component, EcsProperty


@component()
class PlaneComponent:
    """

    The `PlaneComponent` class represents a component in a plane. It is used to describe the reference, origin, and axes of the plane.

    Attributes:
        ref (numpy.ndarray): The reference array of the plane component.
        origin (int): The index of the origin axis.
        xaxis (int): The index of the x-axis.
        yaxis (int): The index of the y-axis.
        zaxis (int): The index of the z-axis.

    """
    ref: np.ndarray = None
    origin: int = 0
    xaxis: int = 1
    yaxis: int = 2
    zaxis: int = 3





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
    """
    SlimPlane

    Class representing a slim plane in 3D.

    Attributes:
        _array (numpy.ndarray): The array representing the slim plane.

    Methods:
        __init__(self, arr: np.ndarray = None):
            Initializes a SlimPlane object.

        origin:
            A property representing the origin of the slim plane.

        origin.setter:
            A property setter for the origin of the slim plane.

        xaxis:
            A property representing the x-axis of the slim plane.

        xaxis.setter:
            A property setter for the x-axis of the slim plane.

        yaxis:
            A property representing the y-axis of the slim plane.

        yaxis.setter:
            A property setter for the y-axis of the slim plane.

        zaxis:
            A property representing the z-axis of the slim plane.

        zaxis.setter:
            A property setter for the z-axis of the slim plane.

        normal:
            A property representing the normal vector of the slim plane.

        normal.setter:
            A property setter for the normal vector of the slim plane.

        __hash__(self):
            Returns the hash value of the slim plane.

        d:
            A property representing the 'd' value of the slim plane equation.

        todict(self):
            Returns a dictionary representation of the slim plane.

        from_normal(cls, normal: np.ndarray, origin: np.ndarray = np.zeros(3, float)):
            Creates a SlimPlane object from a normal vector and an origin point.

        from_3pt(cls, origin: np.ndarray, pt1: np.ndarray, pt2: np.ndarray):
            Creates a SlimPlane object from 3 points.

        project(self, pts: np.ndarray):
            Projects the given points onto the slim plane.

        point_from_local_to_world(self, pt: np.ndarray):
            Transforms a point from local coordinates to world coordinates.

        point_from_world_to_plane(self, pt: np.ndarray):
            Transforms a point from world coordinates to the slim plane coordinates.

        point_at(self, pt):
            Returns the world coordinates of a given point on the slim plane.
    """
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
        """Creates an instance of SlimPlane from 3 points.

        :param origin: The origin point.
        :type origin: numpy.ndarray
        :param pt1: The first point.
        :type pt1: numpy.ndarray
        :param pt2: The second point.
        :type pt2: numpy.ndarray
        :return: An instance of SlimPlane.
        :rtype: SlimPlane
        """
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

    def point_at(self, pt):
        """

        :param pt:
        :type pt:
        :return:
        :rtype:
        """
        return self.point_from_local_to_world(pt)

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
    :type pln: Plane
    :param pt: The point to be projected onto the plane.
    :type pt: ndarray (shape: (3,))
    :return: The projected point.
    :rtype: ndarray (shape: (3,))
    """
    return pt - (dot(pln.normal, pt - pln.origin) * pln.normal)


WXY = create_plane()

from mmcore.geom.transform import rotate_around_axis


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
    :type pln: Plane
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
    :type pln: Plane or SlimPlane

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


from mmcore.geom.parametric import algorithms as algo


def ray_intersection(ray: algo.Ray, pln: Plane):
    """Find the intersection point between a ray and a plane.

    :param ray: The ray for which to find the intersection point.
    :type ray: algo.Ray
    :param pln: The plane with which to intersect the ray.
    :type pln: Plane
    :return: The intersection point between the ray and the plane.
    :rtype: algo.Point
    """
    return algo.ray_plane_intersection(*ray, pln, full_return=True)


def rotate_vectors_around_plane(vecs, plane_a, angle):
    """
    Rotate vectors around a plane.

    :param vecs: The vectors to rotate.
    :type vecs: list or numpy.array
    :param plane_a: The plane around which to rotate the vectors.
    :type plane_a: Plane object
    :param angle: The angle in radians by which to rotate the vectors.
    :type angle: float
    :return: The rotated vectors.
    :rtype: numpy.array
    """
    return norm(vecs) * rotate_around_axis(unit(vecs), angle, origin=(0., 0., 0.), axis=plane_a.normal)


@vectorize(excluded=[0, 1], signature='()->()')
def rotate_plane_around_plane(plane1, plane2, angle):
    """
    Rotate a plane around another plane.

    :param plane1: The plane to be rotated. Plane который планируется повернуть.
    :type plane1: Plane
    :param plane2: The plane around which plane1 should be rotated. Plane в системе которого происходит поворот.
    :type plane2: Plane
    :param angle: The angle of rotation in radians. Угол поворота в радианах.
    :type angle: float
    :return: The rotated plane. Новый Plane в глобальных координатах.
    :rtype: Plane
    """
    origin = rotate_around_axis(plane1.origin, angle, plane2.origin, plane2.normal)
    xaxis, yaxis, zaxis = rotate_around_axis(np.array([plane1.xaxis, plane1.yaxis, plane1.zaxis]), angle,
                                             origin=(0., 0., 0.), axis=plane2.normal)
    return Plane(np.array([origin, xaxis, yaxis, zaxis], dtype=float))


@vectorize(signature='(i),(i)->(j,i)')
def plane_from_normal_numeric(vector=(2, 33, 1), origin=(0., 0.0, 0.)):
    """
    :param vector: The normal vector of the plane
    :type vector: tuple of ints
    :param origin: The origin point of the plane
    :type origin: tuple of floats
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


def plane_from_normal(vector=(2, 33, 1), origin=(0., 0.0, 0.)):
    """
    Create a Plane object from a normal vector and origin point.

    :param vector: The normal vector of the plane.
    :type vector: tuple, optional
    :param origin: The origin point of the plane.
    :type origin: tuple, optional
    :return: The Plane object.
    :rtype: Plane

    """
    return Plane(plane_from_normal_numeric(vector, origin))




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
