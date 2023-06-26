#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
import warnings
from collections import namedtuple
from typing import Any, Union, ContextManager

import numpy as np
import pyquaternion as pq
from compas.data import Data
from compas.geometry import Transformation
from mmcore.geom.vectors import unit
from numpy import ndarray


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
            warnings.warn(Transform.__doc__)
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
