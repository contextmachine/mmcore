import dataclasses

import abc
import numpy as np

import typing

T = typing.TypeVar("T")


@typing.runtime_checkable
class ParametricObject(typing.Protocol[T]):
    @abc.abstractmethod
    def evaluate(self, t):
        ...


@dataclasses.dataclass
class NormalPoint:
    point: typing.Sequence[float] = (0, 0, 0)
    normal: typing.Sequence[float] = (0, 0, 1)

    @property
    def x(self):
        return self.x

    @property
    def y(self):
        return self.y

    @property
    def z(self):
        return self.z

    @property
    def pt(self):
        """alias"""
        return self.point

    def __iter__(self):
        return iter([self.point, self.normal])

    def __array__(self, **kwargs):
        return np.array([self.point, self.normal], **kwargs)

    def __add__(self, other):
        a, b = self.__array__() + other.__array__()
        return NormalPoint(point=a, normal=b)
    def __sud__(self, other):
        a, b = self.__array__() - other.__array__()
        return NormalPoint(point=a, normal=b)

    def __mul__(self, other):
        a, b = self.__array__() * other.__array__()
        return NormalPoint(point=a, normal=b)
    def __mat__(self, other):
        a,b = self.__array__()  @ other
        return NormalPoint(point=a, normal=b)
