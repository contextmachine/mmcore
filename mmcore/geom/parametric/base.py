import abc
import dataclasses
import itertools
import typing
from collections import namedtuple

import numpy as np
from scipy.spatial import distance

from mmcore.geom.transform import Transform

T = typing.TypeVar("T")

UVPoint = namedtuple("UVPoint", ["u", "v", "point"])
EvalPointTuple1D = namedtuple("EvalPointTuple1D", ["i", "t", "x", "y", "z"])
EvalPointTuple1D.__doc__ = """Represent t like 1d evaluation result.\n 
5e tuple[i: int, t: float, x: float, y: float, z: float]"""
EvalPointTuple2D = namedtuple("EvalPointTuple2D", ["i", "j", "u", "v", "x", "y", "z"])
EvalPointTuple2D.__doc__ = """Represent u,v like 2d evaluation result.\n 
 7e tuple[i: int, j: int, u: float, v: float, x: float, y: float, z: float]"""

from mmcore.geom.parametric.algorithms import ClosestPoint, ProximityPoints


class transform_manager:
    def __init__(self, trx):
        self._trx = trx
    @property
    def trx(self):
        return self._trx()
    def __call__(self, method):
        return lambda slf,*args, **kwargs: np.array(method(slf,*args, **kwargs)).tolist() @ slf.__trx__



@dataclasses.dataclass(eq=True, unsafe_hash=True)
class ParametricObject(typing.Protocol[T]):
    @transform_manager
    def transform(self, m: Transform=None):
        if m is None:
            m=Transform()
        self.__trx__ @ m
        return self.__trx__
    def __post_init__(self):
        self.__trx__ = Transform()

    @property
    def __params__(self) -> typing.Any:
        """
        Used for hashing. This can be useful for caching the evaluate method and other methods that require the object
        itself to be hashable.
        Default: dataclasses.asdict(self)
        """
        return dataclasses.asdict(self)

    @property
    def dim(self) -> int:
        """
        Defines the dimension of the vector that will be considered as the minimum when passed to the evaluate method.
        For example for dimension = 1, t=[0.0, 0.3] will call the method 2 times, while for dimension =2 once,
        as u,v = t = [0.0, 0.3]
        """
        return 1

    @abc.abstractmethod
    @transform
    def evaluate(self, t):
        ...

    def closest_point(self, point):
        return ClosestPoint(self, point)(x0=np.array(list(itertools.repeat(0.5, self.dim))),
                                         bounds=list(itertools.repeat((0.0, 1.0), self.dim)))

    def distance_at(self, point, t):
        return distance.euclidean(point, self.evaluate(t))

    def distance(self, point):
        return self.closest_point(point).distance

    def proximity(self, other: 'ParametricObject'):
        return ProximityPoints(self, other)(x0=np.array(list(itertools.repeat(0.5, self.dim + other.dim))),
                                            bounds=(itertools.repeat((0.0, 1.0), self.dim + other.dim)))




transform_manager(ParametricObject)


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
        a, b = self.__array__() @ other
        return NormalPoint(point=a, normal=b)


class ProxyAttributeDescriptor(typing.Generic[T]):
    def __init__(self, proxy_name=None, default=None, callback=lambda x: x, no_set=False):
        self.proxy_name = proxy_name
        self.callback = callback
        self.default = default
        self.no_set = no_set

    def __set_name__(self, owner, name):
        self.name = name
        if self.proxy_name is None:
            self.proxy_name = self.name

    def __get__(self, inst, own=None) -> T:

        if inst is None:
            return self.default

        else:
            res = self.callback(getattr(inst._proxy, self.proxy_name))
            return res if res is not None else self.default

    def __set__(self, inst: T, v):
        # #print(f"event: set {self.proxy_name}/{self.name}->{v}")
        if not self.no_set:
            try:

                setattr(inst.proxy, self.proxy_name, v)
            except  AttributeError:
                pass


@dataclasses.dataclass
class ProxyParametricObject(ParametricObject[T]):
    @transform_manager
    def transform(self, m: Transform=None):
        return super().transform(m)
    @abc.abstractmethod
    def prepare_proxy(self):
        ...

    @abc.abstractmethod
    @transform
    def evaluate(self, t) -> typing.Union[tuple[float, float, float], np.ndarray, list, typing.Any]:
        ...

    @property
    def proxy(self) -> T:
        try:
            return self._proxy
        except AttributeError as err:
            self.prepare_proxy()
            return self._proxy
