#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
import functools
import operator
import warnings
from collections import namedtuple
from typing import Any, ContextManager, Union

import numpy as np
import pyquaternion as pq
from compas.data import Data
from compas.geometry import Transformation
from numpy import ndarray

from mmcore.geom.vectors import unit


def add_crd(pt, value):
    if not isinstance(pt, np.ndarray):
        pt = np.array(pt, dtype=float)
    if len(pt.shape) == 1:
        pt = pt.reshape(1, pt.shape[0])

    return np.c_[pt, np.ones((pt.shape[0], 1)) * value]


def add_w(pt):
    return add_crd(pt, value=1)


def remove_crd(pt):
    return pt.flatten()[:-1]


Plane = namedtuple("Plane", ["xaxis", "yaxis", "normal", "origin"])
WorldXY = Plane([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0])


def transpose(matrix):
    """
    Transposes a 4x4 matrix.

    Parameters:
    matrix (list): A list of lists representing the matrix.

    Returns:
    list: A list of lists representing the transposed matrix.
    """
    # Create an empty matrix to hold the result
    result: list[list[float]] = [[0.0 for j in range(4)] for i in range(4)]

    # Transpose the matrix
    for i in range(4):
        for j in range(4):
            print(i, j)
            result[i][j] = matrix[j][i]

    return result


def zero_transform():
    return [[0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]]


def matmul(t1, t2):
    def mr(m1, m2):
        mmm = zero_transform()

        for i in range(len(m1)):

            for j in range(len(m2)):
                mmm[i][j] = functools.reduce(operator.add, map(lambda k: m1[j][k[1]] * m2[i][k[0]],
                                                               zip(range(len(m1[i])), range(len(m2[j])))))
        return mmm

    return mr(transpose(t1), t2)


def mirror(right):
    mirror = np.eye(3, 3)
    mirror[1, 1] = -1
    left = np.zeros((len(right), 3))
    for i, pt in enumerate(right):
        # #print(i,pt)
        left[i, ...] = mirror @ np.asarray(pt).T
    return left


TRANSFORM_WARN = True


class Transform:
    """
    WARNING
    This version contains a matrix ordering bug, it only exists because there is a fair amount of code compatible
    with this particular implementation. In new code, use TransformV2.
    Soon Transform will be completely replaced by TransformV2.
    --------------------------------------------------------------------------------------------------------------------
    [!] To disable this warning, set the mmcore.geom.transform.TRANSFORM_WARN constant to False, like this:

    >>> from mmcore.geom.transform import TRANSFORM_WARN
    >>> TRANSFORM_WARN = False

    """
    matrix: ndarray

    def __new__(cls, matrix=None, *args, **kwargs):
        if TRANSFORM_WARN:
            warnings.warn(DeprecationWarning(Transform.__doc__))
        inst = super().__new__(cls)
        if matrix is None:
            matrix = np.eye(4)
        inst.matrix = np.array(matrix).reshape((4, 4))
        return inst

    def __repr__(self):
        return "Transform:\n" + self.matrix.__str__()

    def __getitem__(self, item: Union[slice, int, ndarray, tuple, Any]) -> ndarray:
        return self.matrix[item]

    def __setitem__(self, key: Union[slice, int, ndarray, tuple, Any], value):
        self.matrix[key] = value

    def __matmul__(self, other):
        return Transform(self.__array__().__matmul__(np.array(other)))

    def __rmatmul__(self, other):
        if isinstance(other, (list, tuple)):
            other = np.array(other, dtype=float)
        if isinstance(other, ndarray):

            if (len(other.shape) == 1) and (other.shape[0] == 3):
                ppt = np.array([0, 0, 0, 1], dtype=float)
                ppt[0:3] = other
                other[:] = (ppt @ self.matrix.T)[0:-1]
                return other
            elif (len(other.shape) > 1) and (other.shape[1] == 3):

                print('!!')
                l = []
                for i in other:
                    l.append(self.__rmatmul__(i).tolist())
                return np.array(l, dtype=float)
            elif other.shape[0] == 4:
                print('other!', other)
                return remove_crd((other @ self.matrix).T)

            else:
                raise ValueError(f"Shape (3, ...) or (4, ...) was expected, but {other.shape} exist.")

        elif hasattr(other, 'matrix'):
            other.transform(self.matrix)
        elif hasattr(other, "transform"):
            if issubclass(other.__class__, Data):
                T = Transformation(self.matrix[:3, :3])
                other.transform(T)
                return other
        elif isinstance(other, Transform):
            return Transform(other.__array__() @ self.matrix)


        else:
            raise ValueError(f"{other} does not define any of the available interfaces for transformation.")

    def __array__(self):
        return np.array(self.matrix, dtype=float)

    def __iter__(self):
        return self.matrix.__iter__()

    def tolist(self):
        return self.matrix.tolist()

    def rotate(self, axis, angle):
        self.matrix = self.matrix @ pq.Quaternion(axis=axis, angle=angle).transformation_matrix

    def translate(self, direction):
        matrix = np.array([[1, 0, 0, direction[0]],
                           [0, 1, 0, direction[1]],
                           [0, 0, 1, direction[2]],
                           [0, 0, 0, 1]], dtype=float)
        self.matrix = self.matrix @ matrix

    @classmethod
    def from_plane(cls, plane):
        M = cls.__new__(cls)
        M.matrix[0][0], M.matrix[1][0], M.matrix[2][0] = plane.xaxis
        M.matrix[0][1], M.matrix[1][1], M.matrix[2][1] = plane.yaxis
        M.matrix[0][2], M.matrix[1][2], M.matrix[2][2] = plane.normal
        M.matrix[0][3], M.matrix[1][3], M.matrix[2][3] = plane.origin
        return M

    @classmethod
    def from_plane_to_plane(cls, plane_a, plane_b):
        # print(plane_a,plane_b)
        M1 = cls.from_plane(Plane(unit(plane_a.xaxis), unit(plane_a.yaxis), unit(plane_a.normal), plane_a.origin))
        M2 = cls.from_plane(Plane(unit(plane_b.xaxis), unit(plane_b.yaxis), unit(plane_b.normal), plane_b.origin))
        return cls(M2.matrix @ M1.matrix)

    @classmethod
    def from_world_to_plane(cls, plane):

        return cls.from_plane_to_plane(Plane([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]), plane)

    def __invert__(self):
        return Transform(np.linalg.inv(self.matrix))

    def inv(self):
        return self.__invert__()

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def T(self):
        return self.matrix.T

    def flatten(self):
        return self.matrix.flatten()

    @classmethod
    def plane_projection(cls, plane: Plane):
        t = Transform.from_plane_to_plane(WorldXY, plane)
        wxy_proj = Transform()
        wxy_proj.matrix[2, 2] = 0
        tinv = t.__invert__()
        aa = t @ wxy_proj
        aa1 = aa @ tinv
        return aa1

    @classmethod
    def mirror(cls, x=-1, y=1, z=1):
        return Transform([
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [0, 0, z, 0],
            [0, 0, 0, 1]
        ])

    @classmethod
    def scale(cls, x: float = 1, y: float = 1, z: float = 1):
        matrix = np.array([[x, 0, 0, 0],
                           [0, y, 0, 0],
                           [0, 0, z, 0],
                           [0, 0, 0, 1]], dtype=float)
        return Transform(matrix)


case2d = lambda pts: np.c_[np.zeros((len(pts), 1)), np.ones((len(pts), 1))]
case3d = lambda pts: np.ones((len(pts), 1))


def append_arr(arr, itm=case3d): return np.c_[arr, itm(arr)]


class TransformV2:
    """
    This version lacks one annoying mistake that the previous one has.
    Soon Transform will be completely replaced by TransformV2
    """

    def __new__(cls, matrix=None, *args, **kwargs):
        inst = super().__new__(cls)
        if matrix is None:
            matrix = np.eye(4)
        inst.matrix = np.array(matrix).reshape((4, 4))
        return inst

    def __matmul__(self, other):
        if isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, TransformV2):
            return TransformV2(other.matrix @ self.matrix)
        elif hasattr(other, 'transform'):
            return other.transform(self)

        elif isinstance(other, (np.ndarray, np.matrix)):
            if len(other.shape) == 2:
                if other.shape == (4, 4):
                    return self.__matmul__(TransformV2(other))

                elif other.shape[-1] == 3:
                    return ((self.matrix @ append_arr(other, case3d).T).T[..., :3]).tolist()
                elif other.shape[-1] == 2:
                    return (
                        (self.matrix @ np.c_[other, np.c_[np.zeros((len(other), 1)), np.ones((len(other), 1))]].T).T[
                        ...,
                        :3]).tolist()
            elif other.shape == (3,):
                return (self.matrix @ np.append(other, 1).T).T.tolist()[:-1]
            else:
                raise f"{other}"
        else:
            raise f"{other}"

    def __rmatmul__(self, other):
        self.matrix = other.matrix @ self.matrix
        return self

    def __array__(self):
        return np.array(self.matrix, dtype=float)

    def __iter__(self):
        return self.matrix.__iter__()

    def tolist(self):
        return self.matrix.tolist()

    def rotate(self, axis, angle):
        self.matrix = pq.Quaternion(axis=axis, angle=angle).transformation_matrix @ self.matrix

    def translate(self, direction):
        matrix = np.array([[1, 0, 0, direction[0]],
                           [0, 1, 0, direction[1]],
                           [0, 0, 1, direction[2]],
                           [0, 0, 0, 1]], dtype=float)
        self.matrix = self.matrix @ matrix

    def mirror(self, x=1, y=1, z=1):
        self.matrix = self.from_mirror(x, y, z).matrix @ self.matrix

    def scale(self, x=1, y=1, z=1):
        self.matrix = self.from_scale(x, y, z).matrix @ self.matrix

    @classmethod
    def from_plane(cls, plane):
        M = cls.__new__(cls)
        M.matrix[0][0], M.matrix[1][0], M.matrix[2][0] = plane.xaxis
        M.matrix[0][1], M.matrix[1][1], M.matrix[2][1] = plane.yaxis
        M.matrix[0][2], M.matrix[1][2], M.matrix[2][2] = plane.normal
        M.matrix[0][3], M.matrix[1][3], M.matrix[2][3] = plane.origin
        return M

    @classmethod
    def from_plane_to_plane(cls, plane_a, plane_b):
        # print(plane_a,plane_b)
        M1 = cls.from_plane(Plane(unit(plane_a.xaxis), unit(plane_a.yaxis), unit(plane_a.normal), plane_a.origin))
        M2 = cls.from_plane(Plane(unit(plane_b.xaxis), unit(plane_b.yaxis), unit(plane_b.normal), plane_b.origin))
        return cls(M2.matrix @ M1.matrix)

    @classmethod
    def from_world_to_plane(cls, plane):
        return cls.from_plane_to_plane(Plane([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]), plane)

    def __invert__(self):
        return TransformV2(np.linalg.inv(self.matrix))

    def inv(self):
        return self.__invert__()

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def T(self):
        return self.matrix.T

    def flatten(self):
        return self.matrix.flatten()

    @classmethod
    def plane_projection(cls, plane: Plane):
        t = Transform.from_plane_to_plane(WorldXY, plane)
        wxy_proj = TransformV2()
        wxy_proj.matrix[2, 2] = 0
        tinv = t.__invert__()
        aa = t @ wxy_proj
        aa1 = aa @ tinv
        return aa1

    @classmethod
    def from_mirror(cls, x=-1, y=1, z=1):
        return TransformV2([
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [0, 0, z, 0],
            [0, 0, 0, 1]
        ])

    @classmethod
    def from_scale(cls, x: float = 1, y: float = 1, z: float = 1):
        matrix = np.array([[x, 0, 0, 0],
                           [0, y, 0, 0],
                           [0, 0, z, 0],
                           [0, 0, 0, 1]], dtype=float)
        return cls(matrix)

    @classmethod
    def from_translate(cls, direction):
        m = cls()
        m.translate(direction)
        return m

    @classmethod
    def from_rotate(cls, axis, angle):
        m = cls()
        m.rotate(axis, angle)
        return m

    def __repr__(self):
        return f'{self.matrix.__repr__()}'.replace("array", "TrxV2")


class OwnerTransform:
    def __init__(self, f):
        self._f = f
        self.name = f.__name__

    def __get__(self, instance, owner):
        if instance is None:
            return self._f
        return self._f(instance) @ Transform(np.array(instance.matrix, dtype=float).reshape((4, 4)))


class TransformManager(ContextManager):
    def __init__(self, matrix: Transform, obj):
        self.matrix = matrix
        self.obj = obj

    def __enter__(self):
        self.obj @ self.matrix
        return self.obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.obj @ self.matrix.inv


def assign_transform(m):
    def assign_transform_wrapper(obj, *args, **kw):
        return ((np.array(m(obj, *args, **kw) + [1]) @ obj.matrix.T)[:3]).tolist()

    return assign_transform_wrapper


YZ_TO_XY = Transform(np.array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]).reshape((4, 4)))

XY_TO_YZ = Transform(np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1]).reshape((4, 4)))

YZ_TO_XY_V2 = TransformV2(np.array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]).reshape((4, 4)))

XY_TO_YZ_V2 = TransformV2(np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1]).reshape((4, 4)))


def world_coords_to_plane_coords(point, plane):
    """Transforms a point from world coordinates to construction plane coordinates.
    Parameters:
      point (point): A 3D point in world coordinates.
      plane (plane): The construction plane
    Returns:
      (point): 3D point in construction plane coordinates
    Example:
      import rhinoscriptsyntax as rs
      plane = rs.ViewCPlane()
      point = rs.XformWorldToCPlane([0,0,0], plane)
      if point: print "CPlane point:", point
    See Also:
      XformCPlaneToWorld
    """

    v = point - plane.origin

    return v * plane.xaxis + v * plane.yaxis + v * plane.normal


def plane_coords_to_world_coords(point, plane):
    """Transforms a point from world coordinates to construction plane coordinates.
    Parameters:
      point (point): A 3D point in world coordinates.
      plane (plane): The construction plane
    Returns:
      (point): 3D point in construction plane coordinates
    Example:
      import rhinoscriptsyntax as rs
      plane = rs.ViewCPlane()
      point = rs.XformWorldToCPlane([0,0,0], plane)
      if point: print "CPlane point:", point
    See Also:
      XformCPlaneToWorld
    """

    x, y, z = point
    return plane.origin + (x * plane.xaxis) + (y * plane.yaxis) + (z * plane.normal)


def custom_to_global(point, origin, x_axis, y_axis, z_axis):
    # Compute the inverse of the transformation matrix from custom to global coordinate system
    det = x_axis[0] * y_axis[1] * z_axis[2] + y_axis[0] * z_axis[1] * x_axis[2] + z_axis[0] * x_axis[1] * y_axis[2] - \
          x_axis[0] * z_axis[1] * y_axis[2] - y_axis[0] * x_axis[1] * z_axis[2] - z_axis[0] * y_axis[1] * x_axis[2]
    if det == 0:
        raise ValueError("Transformation matrix is not invertible")
    inv_transform_matrix = [
        [
            (y_axis[1] * z_axis[2] - z_axis[1] * y_axis[2]) / det,
            (z_axis[1] * x_axis[2] - x_axis[1] * z_axis[2]) / det,
            (x_axis[1] * y_axis[2] - y_axis[1] * x_axis[2]) / det
        ],
        [
            (z_axis[0] * y_axis[2] - y_axis[0] * z_axis[2]) / det,
            (x_axis[0] * z_axis[2] - z_axis[0] * x_axis[2]) / det,
            (y_axis[0] * x_axis[2] - x_axis[0] * y_axis[2]) / det
        ],
        [
            (y_axis[0] * z_axis[1] - z_axis[0] * y_axis[1]) / det,
            (z_axis[0] * x_axis[1] - x_axis[0] * z_axis[1]) / det,
            (x_axis[0] * y_axis[1] - y_axis[0] * x_axis[1]) / det
        ]
    ]

    # Subtract the origin from the point and transform the result using the inverse transformation matrix
    transformed_point = [
        inv_transform_matrix[0][0] * (point[0] - origin[0]) + inv_transform_matrix[1][0] * (point[1] - origin[1]) +
        inv_transform_matrix[2][0] * (point[2] - origin[2]),
        inv_transform_matrix[0][1] * (point[0] - origin[0]) + inv_transform_matrix[1][1] * (point[1] - origin[1]) +
        inv_transform_matrix[2][1] * (point[2] - origin[2]),
        inv_transform_matrix[0][2] * (point[0] - origin[0]) + inv_transform_matrix[1][2] * (point[1] - origin[1]) +
        inv_transform_matrix[2][2] * (point[2] - origin[2])
    ]

    # Return the transformed point as a tuple of three numbers
    return tuple(transformed_point)


def global_to_custom(point, origin, x_axis, y_axis, z_axis):
    """
    Convert a point from a global coordinate system to a custom coordinate system defined by an origin and three axes.

    :param point: tuple or list of three numbers representing the coordinates of a point in the global coordinate system
    :param origin: tuple or list of three numbers representing the origin of the custom coordinate system
    :param x_axis: tuple or list of three numbers representing the x-axis of the custom coordinate system
    :param y_axis: tuple or list of three numbers representing the y-axis of the custom coordinate system
    :param z_axis: tuple or list of three numbers representing the z-axis of the custom coordinate system
    :return: tuple of three numbers representing the coordinates of the point in the custom coordinate system
    """
    # Define the transformation matrix from global to custom coordinate system
    transform_matrix = [
        [x_axis[0], y_axis[0], z_axis[0]],
        [x_axis[1], y_axis[1], z_axis[1]],
        [x_axis[2], y_axis[2], z_axis[2]]
    ]

    # Subtract the origin from the point and transform the result using the transformation matrix
    transformed_point = [
        transform_matrix[0][0] * (point[0] - origin[0]) + transform_matrix[1][0] * (point[1] - origin[1]) +
        transform_matrix[2][0] * (point[2] - origin[2]),
        transform_matrix[0][1] * (point[0] - origin[0]) + transform_matrix[1][1] * (point[1] - origin[1]) +
        transform_matrix[2][1] * (point[2] - origin[2]),
        transform_matrix[0][2] * (point[0] - origin[0]) + transform_matrix[1][2] * (point[1] - origin[1]) +
        transform_matrix[2][2] * (point[2] - origin[2])
    ]

    # Return the transformed point as a tuple of three numbers
    return tuple(transformed_point)
