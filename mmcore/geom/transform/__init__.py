#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
from collections import namedtuple
from functools import wraps
from typing import Any, Union, ContextManager

import numpy as np


from mmcore.addons import ModuleResolver
from mmcore.geom.vectors import unit

with ModuleResolver() as rsl:
    pass
import rhino3dm as rg
from compas.data import Data
from compas.geometry import Transformation
from numpy import ndarray

import pyquaternion as pq

Plane=namedtuple("Plane", ["xaxis","yaxis","normal","origin"])
def mirror(right):
    mirror = np.eye(3, 3)
    mirror[1, 1] = -1
    left = np.zeros((len(right), 3))

    for i, pt in enumerate(right):
        # print(i,pt)
        left[i, ...] = mirror @ np.asarray(pt).T
    return left


def create_Transform(flat_arr):
    tg = rg.Transform.ZeroTransformation()
    k = 0
    for i in range(4):
        for j in range(4):
            setattr(tg, "M{}{}".format(i, j), flat_arr[k])
            k += 1
    return tg


class Transform:
    matrix: ndarray

    def __new__(cls, matrix=None, *args, **kwargs):
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
        if hasattr(other, 'matrix'):
            other.transform(self.matrix)
        elif hasattr(other, "transform"):
            if issubclass(other.__class__, Data):
                T = Transformation(self.matrix[:3, :3])
                other.transform(T)
                return other
        elif isinstance(other, Transform):
            return Transform(other.__array__() @ self.matrix)
        elif isinstance(other, ndarray):
            if other.shape[0] == 3:
                ppt = np.array([0, 0, 0, 1], dtype=float)
                ppt[0:3] = other
                other[:] = (ppt @ self.matrix)[0:3]
                return other

            elif other.shape[0] == 4:
                return other @ self.matrix
            else:
                raise ValueError(f"Shape (3, ...) or (4, ...) was expected, but {other.shape} exist.")
        else:
            raise ValueError(f"{other} does not define any of the available interfaces for transformation.")

    def __array__(self, *args, **kwargs):
        return np.array(self.matrix, dtype=float, *args, **kwargs)

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
    def from_plane_to_plane(cls, plane_a,plane_b):
        M1 = cls.from_plane(Plane(unit(plane_a.xaxis),unit(plane_a.yaxis),unit(plane_a.normal),plane_a.origin))
        M2 = cls.from_plane(Plane(unit(plane_b.xaxis),unit(plane_b.yaxis),unit(plane_b.normal),plane_b.origin))
        return cls(M2.matrix @ M1.matrix.T)

    @classmethod
    def from_world_to_plane(cls, plane):

        return cls.from_plane_to_plane(Plane([1,0,0],[0,1,0],[0,0,1],[0,0,0]),plane)

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
        return ((np.array(m(obj, *args, **kw) + [1]) @ obj.matrix_to_square_form().T)[:3]).tolist()
    return assign_transform_wrapper