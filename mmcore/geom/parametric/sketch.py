import functools

import abc
import copy
import dataclasses
import geomdl
import math
import os
import sys
import timeit
import typing
from abc import ABCMeta, abstractmethod
from collections import namedtuple

from earcut.earcut import normal
from geomdl.operations import tangent
from itertools import starmap
from scipy.optimize import fsolve

from mmcore.base import geomdict
from mmcore.geom.transform import remove_crd, Transform, WorldXY
from mmcore.geom.vectors import *
from enum import Enum

from mmcore.geom.parametric.base import ParametricObject, NormalPoint, UVPoint, EvalPointTuple2D
from mmcore.geom.vectors import unit, add_translate, angle

from mmcore.collections import DoublyLinkedList
from mmcore.func.curry import curry

from pyquaternion import Quaternion
from compas.geometry.transformations import matrix_from_frame_to_frame
import mmcore
import zlib
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
from mmcore.geom.materials import ColorRGB
from mmcore.base.models.gql import MeshPhongMaterial

from geomdl import NURBS
from geomdl import utilities as geomdl_utils
from mmcore.collections import DCLL, DoublyLinkedList
from mmcore.collections.multi_description import EntityCollection, ElementSequence
import multiprocess as mp


def add_crd(pt, value):
    if not isinstance(pt, np.ndarray):
        pt = np.array(pt, dtype=float)
    if len(pt.shape) == 1:
        pt = pt.reshape(1, pt.shape[0])

    return np.c_[pt, np.ones((pt.shape[0], 1)) * value]


def add_w(pt):
    return add_crd(pt, value=1)


# mp.get_start_method = "swapn"
import uuid as _uuid

TOLERANCE = 1e-6

T = typing.TypeVar("T")


@dataclasses.dataclass
class Linear(ParametricObject):
    x0: float = 0
    y0: float = 0
    z0: float = 0
    a: float = 1
    b: float = 0
    c: float = 0

    def __post_init__(self):
        self.x = lambda t: self.x0 + self.a * t
        self.y = lambda t: self.y0 + self.b * t
        self.z = lambda t: self.z0 + self.c * t

    def extend(self, a, b):
        return Linear.from_two_points(self.start - unit(self.direction) * a, self.end + unit(self.direction) * b)

    @classmethod
    def from_two_points(cls, start, end):
        a, b, c = np.asarray(end) - np.asarray(start)
        x, y, z = start

        return cls(x0=x, y0=y, z0=z, a=a, b=b, c=c)

    def evaluate(self, t):
        return np.array([self.x(t), self.y(t), self.z(t)], dtype=float)

    @property
    def length(self):
        return euclidean(self.evaluate(0.0), self.evaluate(1.0))

    def divide_distance(self, step):
        tstep = 1 / (self.length / step)

        for i in np.arange(self.length // step) * tstep:
            yield self.evaluate(i)

    def divide_distance_dll(self, step):
        tstep = 1 / (self.length / step)
        dll = DoublyLinkedList()
        for i in np.arange(self.length // step) * tstep:
            dll.append(self.evaluate(i))
        return dll

    def divide_distance_planes(self, step):

        for d in self.divide_distance(step):
            yield PlaneLinear(d, self.direction)

    @property
    def start(self):
        return self.evaluate(0)

    @property
    def end(self):
        return self.evaluate(1)

    @property
    def direction(self):
        return self.end - self.start

    def to_repr(self, backend=None):
        if backend is None:
            return LineObject(
                points=[np.array(self.start, dtype=float).tolist(),
                        np.array(self.start, dtype=float).tolist()]

            )
        else:
            return backend(self)

    def prox(self, other, bounds=[(0, 1), (0, 1)]):
        res = ProximityPoints(self, other)([0.5, 0.5], bounds=bounds)

        return res

    def __hash__(self):
        return


class ParametricLineObject(): ...


def hhp():
    pts = [[-220175.307469, -38456.999234, 20521],
           [-211734.667469, -9397.999234, 13016.199506],
           [-171710.667469, -8217.999234, 5829],
           [-152984.00562, -28444.1358, 6172.86857]]
    import numpy as np
    pts = np.array(pts)

    return HyPar4pt(*pts)


def llll(hp, step=600):
    a = ClosestPoint(hp.b, hp.side_d)(x0=0.5, bounds=((0, 1),))
    uva = unit([hp.side_a.a, hp.side_a.b, hp.side_a.c])
    l = Linear.from_two_points(hp.a, np.asarray(a.pt).flatten())
    uvb = unit([l.a, l.b, l.c])

    dt = np.dot(uva, uvb)
    lll = []
    stp = (1 / dt) * step
    for pt in hp.side_a.divide_distance(stp).T:
        lll.append(
            Linear.from_two_points(pt, np.asarray(ClosestPoint(pt, hp.side_d)(x0=0.5, bounds=((0, 1),)).pt).flatten()))
    return lll


def l22(r):
    rec = EntityCollection(r)
    return list(zip(rec['start'], rec['end']))


def m():
    return np.asarray(l22(llll(hhp(), 600))).tolist()


class ProxyDescriptor(typing.Generic[T]):
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
            res = self.callback(getattr(inst.proxy, self.proxy_name))
            return res if res is not None else self.default

    def __set__(self, inst: T, v):
        # #print(f"event: set {self.proxy_name}/{self.name}->{v}")
        if not self.no_set:
            try:

                setattr(inst.proxy, self.proxy_name, v)
            except  AttributeError:
                pass


@dataclasses.dataclass
class ProxyParametricObject(ParametricObject):

    @abc.abstractmethod
    def prepare_proxy(self):
        ...

    @abc.abstractmethod
    def evaluate(self, t):
        ...

    @property
    def proxy(self):
        try:
            return self._proxy
        except AttributeError as err:
            self.prepare_proxy()
            return self._proxy

    def tessellate(self):
        ...


@dataclasses.dataclass
class NurbsCurve(ProxyParametricObject):
    """

    """

    control_points: typing.Iterable[typing.Iterable[float]] = ProxyDescriptor(proxy_name="ctrlpts")
    delta: float = 0.01
    degree: int = 3
    dimension: int = ProxyDescriptor(default=2)
    rational: bool = ProxyDescriptor()
    domain: typing.Optional[tuple[float, float]] = ProxyDescriptor()

    bbox: typing.Optional[list[list[float]]] = ProxyDescriptor()
    knots: typing.Optional[list[float]] = None

    def __post_init__(self):
        self.proxy.degree = self.degree

        self.proxy.ctrlpts = self.control_points

        if self.knots is None:
            self.knots = geomdl_utils.generate_knot_vector(self._proxy.degree, len(self._proxy.ctrlpts))
            self.proxy.knotvector = self.knots
        self.proxy.delta = self.delta
        self.proxy.evaluate()

    def prepare_proxy(self):
        self._proxy = NURBS.Curve()
        self._proxy.degree = self.degree

    def evaluate(self, t):
        if hasattr(t, "__len__"):
            if hasattr(t, "tolist"):
                t = t.tolist()
            else:
                t = list(t)
            return np.asarray(self._proxy.evaluate_list(t))
        else:
            return np.asarray(self._proxy.evaluate_single(t))

    @property
    def proxy(self):
        try:
            return self._proxy
        except AttributeError:
            self.prepare_proxy()
            return self._proxy

    def tan(self, t):
        pt = tangent(self.proxy, t)
        return NormalPoint(*pt)



class ProxyMethod:
    def __init__(self, fn):
        self.fn = curry(fn)
        self.name = fn.__name__

    def __get__(self, instance, owner):
        return self.fn(instance)


@dataclasses.dataclass
class NurbsSurface(ProxyParametricObject):
    _proxy = NURBS.Surface()
    control_points: typing.Iterable[typing.Iterable[float]]
    degree: tuple = ProxyDescriptor(proxy_name="degree", default=(3, 3))
    delta: float = 0.025
    degree_u: int = ProxyDescriptor(proxy_name="degree_u", default=3)
    degree_v: int = ProxyDescriptor(proxy_name="degree_v", default=3)
    size_u: int = ProxyDescriptor(proxy_name="ctrlpts_size_u", default=6)
    size_v: int = ProxyDescriptor(proxy_name="ctrlpts_size_v", default=6)
    dimentions: int = ProxyDescriptor(proxy_name="dimentions", default=3)
    knots_u: typing.Optional[list[list[float]]] = ProxyDescriptor(proxy_name="knotvector_u", no_set=True)
    knots_v: typing.Optional[list[list[float]]] = ProxyDescriptor(proxy_name="knotvector_v", no_set=True)
    knots: typing.Optional[list[list[float]]] = ProxyDescriptor(proxy_name="knotvector", no_set=True)
    domain: typing.Optional[list[list[float]]] = ProxyDescriptor(proxy_name="domain", no_set=True)
    trims: tuple = ()

    @property
    def proxy(self):
        try:
            return self._proxy
        except AttributeError:
            self.prepare_proxy()
            return self._proxy

    def __post_init__(self):

        self.proxy.ctrlpts = self.control_points

        self.proxy.knotvector_u = geomdl_utils.generate_knot_vector(self._proxy.degree[0], self.size_u)
        self.proxy.knotvector_v = geomdl_utils.generate_knot_vector(self._proxy.degree[1], self.size_v)
        self.proxy.delta = self.delta
        self.proxy.evaluate()

        # u,v=self.size_u, self.size_v

    def evaluate(self, t):
        if len(np.array(t).shape) > 2:

            t = np.array(t).tolist()

            return np.asarray(self._proxy.evaluate_list(t))
        else:
            return np.asarray(self._proxy.evaluate_single(t))

    def prepare_proxy(self):

        self._proxy.set_ctrlpts(list(self.control_points), (self.size_u, self.size_v))
        # self._proxy.ctrlpts=self.control_points
        # ##print(self)

        self._proxy.degree_u, self._proxy.degree_v = self.degree

    def normal(self, t):
        return geomdl.operations.normal(self.proxy, t)

    def tan(self, t):
        pt, tn = tangent(self.proxy, t)
        return NormalPoint(*pt, normal=tn)

    def tessellate(self, uuid=None):
        self.proxy.tessellate()
        vertseq = ElementSequence(self._proxy.vertices)
        faceseq = ElementSequence(self._proxy.faces)
        uv = np.round(np.asarray(vertseq["uv"]), decimals=5)
        normals = [v for p, v in normal(self._proxy, uv.tolist())]
        if uuid is None:
            uuid = _uuid.uuid4().__str__()
        return dict(vertices=np.array(vertseq['data']).flatten(),
                    normals=np.array(normals).flatten(),
                    indices=np.array(faceseq["vertex_ids"]).flatten(),
                    uv=uv,
                    uuid=uuid)


class NurbsSurfaceGeometry:
    material_type = MeshPhongMaterial
    castShadow: bool = True
    receiveShadow: bool = True
    geometry_type = ...

    def __new__(cls, *args, color=ColorRGB(0, 255, 40), control_points=(), **kwargs):
        inst = super().__new__(cls, *args, material=MeshPhongMaterial(color=color.decimal), **kwargs)
        inst.solve_proxy_view(control_points)

        return inst

    def __call__(self, *args, material=None, color=None, **kwargs):
        if material is None:
            if color is not None:
                self.color = color

                self.material = self.material_type(color=color.decimal)
            else:
                self.color = ColorRGB(125, 125, 125)
                self.material = self.material_type(color=self.color.decimal)
        # super(GeometryObject, self).__call__(*args, material=self.material, **kwargs)

    def solve_proxy_view(self, control_points, **kwargs):
        arr = np.array(control_points)
        su, sv, b = arr.shape
        degu = su - 1 if su >= 4 else 3
        degv = sv - 1 if sv >= 4 else 3

        self._proxy = NurbsSurface(control_points=arr.reshape((su * sv, b)).tolist(),
                                   size_u=su,
                                   size_v=sv,
                                   degree_u=degu,
                                   degree_v=degv,
                                   **kwargs)
        self.solve_geometry()

    def solve_geometry(self):
        self._geometry = self.uuid + "-geom"

        geomdict[self._geometry] = self._proxy.tessellate(uuid=self._geometry).create_buffer()


@dataclasses.dataclass
class LineSequence(ParametricObject):
    seq: dataclasses.InitVar[list[Linear]]
    lines: DoublyLinkedList[Linear]


@dataclasses.dataclass
class Polyline(ParametricObject):
    control_points: typing.Union[DCLL, DoublyLinkedList]
    closed: bool = False

    @classmethod
    def from_points(cls, pts):
        closed = False
        if pts[0] == pts[-1]:
            closed = True

        dll = DCLL()
        for pt in pts:
            dll.append(pt)
        inst = cls(dll)
        inst.closed = closed
        return inst

    @property
    def segments(self):
        h = self.control_points.head
        lnr = []
        for pt in self.control_points:
            lnr.append(Linear.from_two_points(copy.deepcopy(h.data), copy.deepcopy(h.next.data)))
            h = h.next
        return lnr

    def evaluate(self, t):
        segm, tt = divmod(t, 1)
        return self.segments[int(segm)].evaluate(tt)


@dataclasses.dataclass
class EvaluatedPoint:
    point: list[float]
    normal: typing.Optional[list[float]]
    direction: typing.Optional[list[typing.Union[float, list[float]]]]
    t: typing.Optional[list[typing.Union[float, list[float]]]]


@dataclasses.dataclass
class PlaneLinear(ParametricObject):
    origin: typing.Iterable[float]
    normal: typing.Optional[typing.Iterable[float]] = None
    xaxis: typing.Optional[typing.Iterable[float]] = None
    yaxis: typing.Optional[typing.Iterable[float]] = None

    def __post_init__(self):

        # #print(unit(self.normal), self.xaxis, self.yaxis)
        if self.xaxis is not None and self.yaxis is not None:
            self.normal = np.cross(unit(self.xaxis), unit(self.yaxis))
        elif self.normal is not None:
            self.normal = unit(self.normal)
            if np.allclose(self.normal, np.array([0.0, 0.0, 1.0])):

                if self.xaxis is not None:
                    self.xaxis = unit(self.xaxis)
                    self.yaxis = np.cross(self.normal, self.xaxis
                                          )

                elif self.yaxis is not None:
                    self.yaxis = unit(self.yaxis)
                    self.xaxis = np.cross(self.normal, self.yaxis)
                else:

                    self.xaxis = np.array([1, 0, 0], dtype=float)
                    self.yaxis = np.array([0, 1, 0], dtype=float)
            else:
                self.xaxis = np.cross(self.normal, np.array([0, 0, 1]))
                self.yaxis = np.cross(self.normal, self.xaxis
                                      )

    def evaluate(self, t):

        if len(t) == 2:
            u, v = t
            uu = np.array(self.x_linear.evaluate(u) - self.origin)
            vv = np.array(self.y_linear.evaluate(v) - self.origin)
            return np.array(self.origin) + uu + vv
        else:
            u, v, w = t
            uu = np.array(self.x_linear.evaluate(u) - self.origin)
            vv = np.array(self.y_linear.evaluate(v) - self.origin)
            ww = np.array(self.z_linear.evaluate(w) - self.origin)
            return np.array(self.origin) + uu + vv + ww

    @property
    def x_linear(self):
        return Linear.from_two_points(self.origin, np.array(self.origin) + unit(np.array(self.xaxis)))

    @property
    def y_linear(self):
        return Linear.from_two_points(self.origin, np.array(self.origin) + unit(np.array(self.yaxis)))

    @property
    def z_linear(self):
        return Linear.from_two_points(self.origin, np.array(self.origin) + unit(np.array(self.normal)))

    def point_at(self, pt):
        T = Transform.from_plane_to_plane(self, WorldXY)
        return remove_crd(add_w(pt) @ T.matrix.T)

    @property
    def x0(self):
        return self.origin[0]

    @property
    def y0(self):
        return self.origin[1]

    @property
    def z0(self):
        return self.origin[2]

    @property
    def a(self):
        return self.normal[0]

    @property
    def b(self):
        return self.normal[1]

    @property
    def c(self):
        return self.normal[2]

    @property
    def d(self):

        return -np.sum(self.normal * self.origin)

    def is_parallel(self, other):
        _cross = np.cross(unit(self.normal), unit(other.normal))
        A = np.array([self.normal, other.normal, _cross])
        return np.linalg.det(A) == 0

    @classmethod
    def from_tree_pt(cls, origin, pt2, pt3):

        x = np.array(pt2) - np.array(origin)
        nrm = np.cross(x, (pt3 - np.array(origin)))
        return PlaneLinear(normal=nrm, xaxis=x, origin=origin)

    def intersect(self, other):

        _cross = np.cross(unit(self.normal), unit(other.normal))
        A = np.array([self.normal, other.normal, _cross])
        d = np.array([-self.d, -other.d, 0.]).reshape(3, 1)

        # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

        # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

        p_inter = np.linalg.solve(A, d).T

        return Linear.from_two_points(p_inter[0], (p_inter + _cross)[0])

    def transform_from_other(self, other):
        return Transform.from_plane_to_plane(other, self)

    def transform_to_other(self, other):
        return Transform.from_plane_to_plane(self, other)

    @property
    def projection(self):
        return Transform.plane_projection(self)

    def project(self, gm):
        # return gm @ self.projection
        raise NotImplementedError

    def orient(self, gm, plane=WorldXY):

        # return remove_crd(add_crd(gm, 1).reshape((4,)) @ self.transform_from_other(plane).matrix.T)
        raise NotImplementedError

    def ray_intersect(self, ray: Linear):
        return line_plane_collision(self, ray, TOLERANCE)


@dataclasses.dataclass
class HyPar4pt(ParametricObject):
    a: typing.Iterable[float]
    b: typing.Iterable[float]
    c: typing.Iterable[float]
    d: typing.Iterable[float]

    side_a = property(fget=lambda self: Linear.from_two_points(self.a, self.b))
    side_b = property(fget=lambda self: Linear.from_two_points(self.b, self.c))
    side_c = property(fget=lambda self: Linear.from_two_points(self.d, self.c))
    side_d = property(fget=lambda self: Linear.from_two_points(self.a, self.d))
    sides_enum: 'typing.Optional[typing.Type[Enum,...]]' = None

    def generate_sides_enum(self):
        class HypSidesEnum(Enum):
            A = self.side_a
            B = self.side_b
            C = self.side_c
            D = self.side_d

        self.sides_enum = HypSidesEnum

    def __post_init__(self):
        self.generate_sides_enum()

    def evaluate(self, t):
        def evl(tt):
            u, v = tt

            lu = Linear.from_two_points(self.side_a.evaluate(u), self.side_c.evaluate(u))
            lv = Linear.from_two_points(self.side_b.evaluate(v), self.side_d.evaluate(v))
            cl = CurveCurveIntersect(lu, lv)

            a, b, c = np.cross(unit([lu.a, lu.b, lu.c]), unit([lv.a, lv.b, lv.c]))
            r = cl(tolerance=TOLERANCE)
            if tuple(r)[-1]:
                return EvaluatedPoint(point=r.pt.tolist(), normal=[a, b, c],
                                      direction=[unit([lu.a, lu.b, lu.c]), unit([lv.a, lv.b, lv.c])], t=r.t)
            else:
                raise Exception(r)

        if np.asarray(t).shape == (2,):
            return evl(t)
        else:
            u, v = t
            l = []
            for _u in u:
                for _v in v:
                    l.append(evl((_u, _v)))
            return EntityCollection(l)

    def intr(self, pln):

        d = []
        for i in [self.side_a, self.side_b, self.side_c, self.side_d]:
            res = line_plane_collision(pln, i)

            if res is not None:
                re = ClosestPoint(res, i)(x0=0.5, bounds=[(0.0, 1.0)]).t
                # #print("RE", re)
                if (re == 0.0) or (re == 1.0):
                    pass
                else:
                    d.append(res)

        return d

    @property
    def polyline(self):
        return Polyline.from_points([self.a, self.b, self.c, self.d, self.a])


IsCoDirectedResponse = namedtuple('IsCoDirectedResponse', ["do"])


def is_co_directed(a, b):
    dp = np.dot(a, b)
    return dp, dp * (1 if dp // 1 >= 0 else -1)


@dataclasses.dataclass
class Grid(ParametricObject):

    def evaluate(self, t):
        ...


@dataclasses.dataclass
class HypPar4ptGrid(HyPar4pt):
    def parallel_side_grid(self, step1, side: str = "D"):
        self._grd = DCLL()
        side = getattr(self.sides_enum, side).value
        d = []
        for pl in side.divide_distance_planes(step1):

            # #print("t",dd)
            r = self.intr(pl)
            if not (r == []):
                try:
                    a, b = r
                    d.append(Linear.from_two_points(b.tolist(), a.tolist()))
                except:
                    pass

        return d

    def parallel_vec_grid(self, step1, vec: Linear):

        d = []
        for pl in vec.divide_distance_planes(step1):

            # #print("t",dd)
            r = self.intr(pl)
            if not (r == []):
                a, b = r
                d.append(Linear.from_two_points(b.tolist(), a.tolist()))
        return d

    """
    def custom_plane_grid(self, step1, plane: PlaneLinear):
        self._grd = DCLL()
        plane.normal
        dct=[]
        
        for side in self.polyline.segments:
            dt=np.dot(unit(plane.normal),unit(side.direction))
            startplane=PlaneLinear(side.start,plane.normal)
            plan
            for i in np.linspace(0, 1, dt*side.length)
                startplane.
        for pt in [self.a, self.b, self.c, self.d]:
            ans=plane.point_at(pt)
            a = ans._asdict()
            a['distance']= ans.distance * np.array(ans.pt-np.array(pt)).dot(plane.normal)

            a["vec"]
            dct.append(ans)
        dct.sort(key=lambda x: x.distance)

        plane.point_at()
        side = getattr(self.sides_enum, self.side).value
        d = []
        for pl in side.divide_distance_planes(step1):

            # #print("t",dd)
            r = self.intr(pl)
            if not (r == []):
                a, b = r
                d.append(Linear.from_two_points(a.tolist(), b.tolist()))

        return d"""


@dataclasses.dataclass
class LineSequence(ParametricObject):
    axis: typing.Iterable[Linear]

    @classmethod
    def from_arr(cls, arr):
        ax = []
        for i in arr:
            if len(i) > 2:
                ax.append(Polyline(*i))
            else:
                ax.append(Linear.from_two_points(*i))
        return cls(ax)

    def evaluate(self, t) -> Polyline:
        dl = DCLL()
        for i in self.axis:
            dl.append(i.evaluate(t))
        return Polyline(dl)

    def surf(self, u=3) -> NurbsSurface:
        pts = []
        for p in np.linspace(0, 1, u):
            if isinstance(p, NurbsCurve):
                pts.append(p.control_points)
            else:
                pts.append(np.asarray(list(self.evaluate(p).control_points)
                                      ))
        print(pts)
        aa = np.asarray(pts).flatten()

        return NurbsSurface(control_points=aa.reshape((len(aa) // 3, 3)).tolist(), size_u=u, size_v=len(self.axis))


if sys.version_info.minor >= 10:
    from typing_extensions import Protocol
else:
    from typing import Protocol


class AbstractBuilder(Protocol):

    def build(self, geoms, params) -> ParametricObject:
        ...


class GeomDesc:
    def __init__(self, default=None, cls=LineSequence):
        self.default = default
        self._cls = cls

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, inst, own=None):
        if inst is None:
            return self.default
        return self._cls(getattr(inst, "_" + self.name))

    def __set__(self, inst, v):
        setattr(inst, "_" + self.name, v)


class Loft:
    geoms = GeomDesc()

    def __call__(self, geoms):
        geoms_seq = LineSequence(geoms)
        ax = geoms.axis[0]
        if hasattr(ax, "control_points"):
            size_u = len(self.geoms.axis)
            size_v = len(ax.control_points)

        else:
            size_u = len(self.geoms.axis)
            size_v = 2

        return geoms_seq.surf(size_u)

    def build(self, geoms, params) -> NurbsSurface:
        return LineSequence(geoms).surf(u=params)


class MinimizeSolution:
    solution_response: typing.Any

    @abc.abstractmethod
    def solution(self, t):
        ...

    def __call__(self,
                 x0: np.ndarray = np.asarray([0.5, 0.5]),
                 bounds: typing.Optional[typing.Iterable[tuple[float, float]]] = ((0, 1), (0, 1)),
                 *args,
                 **kwargs):
        res = minimize(self.solution, x0, bounds=bounds, **kwargs)
        return self.prepare_solution_response(res)

    def __init_subclass__(cls, solution_response=None, **kwargs):
        cls.solution_response = solution_response
        super().__init_subclass__(**kwargs)

    @abc.abstractmethod
    def prepare_solution_response(self, solution):
        ...


ClosestPointSolution = namedtuple("ClosestPointSolution", ["pt", "t", "distance"])
IntersectSolution = namedtuple("IntersectSolution", ["pt", "t", "is_intersect"])
IntersectFail = namedtuple("IntersectFail", ["pt", "t", "distance", "is_intersect"])

MultiSolutionResponse = namedtuple("MultiSolutionResponse", ["pts", "roots"])


class ProximityPoints(MinimizeSolution, solution_response=ClosestPointSolution):
    """

    >>> a=[[9.258697, -8.029476, 0],
    ...    [6.839202, -1.55593, -6.390758],
    ...    [18.258577, 16.93191, 11.876064],
    ...    [19.834301, 27.566156, 1.173745],
    ...    [-1.257139, 45.070784, 0]]
    >>> b=[[27.706367, 29.142311, 6.743523],
    ...    [29.702408, 18.6766, 19.107107],
    ...    [15.5427, 6.960314, 10.273386],
    ...    [2.420935, 26.07378, 18.666591],
    ...    [-3.542004, 3.424012, 11.066738]]
    >>> A,B=NurbsCurve(control_points=a),NurbsCurve(control_points=b)
    >>> prx=ProxPoints(A,B)
    >>> prx()
    ClosestPointSolution(pt=[array([16.27517685, 16.07437063,  4.86901707]), array([15.75918043, 14.67951531, 14.57947997])], t=array([0.52562605, 0.50105099]), distance=9.823693977393207)
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.c1, self.c2 = c1, c2

    def solution(self, t) -> float:
        t1, t2 = t

        return euclidean(self.c1.evaluate(t1), self.c2.evaluate(t2))

    def prepare_solution_response(self, solution):
        t1, t2 = solution.x
        return self.solution_response([self.c1.evaluate(t1),
                                       self.c2.evaluate(t2)],
                                      solution.x,
                                      solution.fun)

    def __call__(self, x0: np.ndarray = np.asarray([0.5, 0.5]),
                 bounds: typing.Optional[typing.Iterable[tuple[float, float]]] = ((0, 1), (0, 1)),
                 *args,
                 **kwargs):
        res = minimize(self.solution, x0, bounds=bounds, **kwargs)

        return self.prepare_solution_response(res)


ProxPoints = ProximityPoints  # Alies for me


class MultiSolution(MinimizeSolution, solution_response=MultiSolutionResponse):
    @abc.abstractmethod
    def solution(self, t): ...

    def __call__(self,
                 x0: np.ndarray = np.asarray([0.5, 0.5]),

                 **kwargs):
        res = fsolve(self.solution, x0, **kwargs)
        return self.prepare_solution_response(res)

    @abc.abstractmethod
    def prepare_solution_response(self, solution):
        ...


class ClosestPoint(MinimizeSolution, solution_response=ClosestPointSolution):
    """
    >>> a= [[-25.0, -25.0, -5.0],
    ... [-25.0, -15.0, 0.0],
    ... [-25.0, -5.0, 0.0],
    ... ...
    ... [25.0, 15.0, 0.0],
    ... [25.0, 25.0, -5.0]]
    >>> srf=NurbsSurface(control_points=a,size_u=6,size_v=6)
    >>> pt=np.array([13.197247, 21.228605, 0])
    >>> cpt=ClosestPoint(pt, srf)
    >>> cpt(x0=np.array((0.5,0.5)), bounds=srf.domain)

    ClosestPointSolution(pt=[array([13.35773142, 20.01771329, -7.31090389])], t=array([0.83568967, 0.9394131 ]), distance=7.41224187927431)
    """

    def __init__(self, point, geometry):
        super().__init__()
        self.point, self.gm = point, geometry

    def solution(self, t):
        r = self.gm.evaluate(t)
        r = np.array(r).T.flatten()
        # #print(self.point, r)

        return euclidean(self.point, r)

    def prepare_solution_response(self, solution):
        return self.solution_response([self.gm.evaluate(solution.x)], solution.x, solution.fun)

    def __call__(self, x0=(0.5,), **kwargs):
        return super().__call__(x0, **kwargs)


"""
def hyp(arr):
    d = arr[1].reshape((3, 1)) + ((arr[0] - arr[1]).reshape((3, 1)) * np.stack(
        [np.linspace(0, 1, num=10), np.linspace(0, 1, num=10), np.linspace(0, 1, num=10)]))
    c = arr[2].reshape((3, 1)) + ((arr[1] - arr[2]).reshape((3, 1)) * np.stack(
        [np.linspace(0, 1, num=10), np.linspace(0, 1, num=10), np.linspace(0, 1, num=10)]))
    f = arr[2].reshape((3, 1)) + ((arr[3] - arr[2]).reshape((3, 1)) * np.stack(
        [np.linspace(0, 1, num=10), np.linspace(0, 1, num=10), np.linspace(0, 1, num=10)]))
    g = arr[3].reshape((3, 1)) + ((arr[0] - arr[3]).reshape((3, 1)) * np.stack(
        [np.linspace(0, 1, num=10), np.linspace(0, 1, num=10), np.linspace(0, 1, num=10)]))

    grp = Group(name="grp")
    lns2 = list(zip(g.T, c.T))
    lns = list(zip(f.T, d.T))
    for i, (lna, lnb) in enumerate(lns):
        grp.add(LineObject(name=f"1-{i}", points=(lna.tolist(), lnb.tolist())))
    for i, (lna, lnb) in enumerate(lns2):
        grp.add(LineObject(name=f"2-{i}", points=(lna.tolist(), lnb.tolist())))

"""


class CurveCurveIntersect(ProximityPoints, solution_response=ClosestPointSolution):
    def __init__(self, c1, c2):
        super().__init__(c1, c2)

    def __call__(self, *args, tolerance=TOLERANCE, **kwargs):
        r = super().__call__(*args, **kwargs)
        is_intersect = np.allclose(r.distance, 0.0, atol=tolerance)
        if is_intersect:
            return IntersectSolution(np.mean(r.pt, axis=0), r.t, is_intersect)

        else:
            return IntersectFail(r.pt, r.t, r.distance, is_intersect)


def line_plane_collision(plane: PlaneLinear, ray: Linear, epsilon=1e-6):
    ray_dir = np.array(ray.direction)
    ndotu = np.array(plane.normal).dot(ray_dir)
    if abs(ndotu) < epsilon:
        return None
    w = ray.start - plane.origin
    si = -np.array(plane.normal).dot(w) / ndotu
    Psi = w + si * ray_dir + plane.origin
    return Psi


def test_cp():
    pts = np.random.random((100, 3)).tolist()
    pts2 = np.random.random((100, 3)).tolist()
    *res, = starmap(Linear.from_two_points, (pts, pts2))
    for i in res:
        for j in res:
            yield ProximityPoints(i, j)()


from mmcore.base.geom import LineObject

from mmcore.base.basic import iscollection


class HypGridLayer:
    hyp: HypPar4ptGrid
    prev: typing.Optional[list[Linear]] = None
    color: typing.Union[ColorRGB, tuple] = (70, 10, 240)
    name: str = "Foo"
    high: float = 0.0
    step: float = 0.0
    direction: typing.Union[str, Linear] = "D"  # self.prev.sort(key=lambda x: x.length)

    def offset_hyp(self, h=2.0):
        A1, B1, C1, D1 = self.hyp.evaluate((0, 0)), self.hyp.evaluate((1, 0)), self.hyp.evaluate(
            (1, 1)), self.hyp.evaluate((0, 1))
        hpp = []
        for item in [A1, B1, C1, D1]:
            hpp.append(np.array(item.point) + np.array(item.normal) * h)
        return HypPar4ptGrid(*hpp)

    def __init__(self, **kwargs):
        object.__init__(self)
        self.__call__(**kwargs)

    def __call__(self, **kwargs):
        self.__dict__ |= kwargs
        if self.prev is not None:
            self.prev.sort(key=lambda x: x.length)
            self.direction = self.prev[-1].extend(60, 60)
        return self.solve_grid()

    def solve_grid(self):
        hp_next = self.offset_hyp(self.high)

        if isinstance(self.direction, str):
            return hp_next.parallel_side_grid(self.step, side=self.direction)
        else:
            return hp_next.parallel_vec_grid(self.step, self.direction)


def to_repr(backend=None):
    def wrapper(obj):
        if backend is None:
            return obj
        else:
            if isinstance(obj, dict):
                return dict((k, wrapper(v)) for k, v in obj.items())
            elif iscollection(obj):
                return [wrapper(item) for item in obj]
            else:
                return backend(obj)

    return wrapper


class SubSyst2:
    A = [-31.02414546224999, -17.3277158585, 9.136232981]
    B = [-22.583505462250002, 11.731284141500002, 1.631432487]
    C = [17.44049453775, 12.911284141500003, -5.555767019000001]
    D = [36.167156386749994, -7.314852424499997, -5.211898449000001]

    def __call__(self, *args, trav=False, **kwargs):
        super().__call__(*args, trav=False, **kwargs)
        self.hyp = HypPar4ptGrid(self.A, self.B, self.C, self.D)
        self.grpr = []
        self.initial = HypGridLayer(name="Layer-Initial", step=0.6, direction="D", color=(70, 10, 240))(trav=False)

        self.layer1 = HypGridLayer(grid=self.initial(trav=False),
                                   hyp=self.hyp,
                                   high=0.3,
                                   step=0.6,
                                   color=(259, 49, 10),
                                   name="Layer-1"
                                   )
        self.layer2 = HypGridLayer(grid=self.layer1(trav=False),
                                   hyp=self.hyp,
                                   high=0.2,
                                   step=3,
                                   color=(259, 49, 10),
                                   name="Layer-2"
                                   )
        self.layer3 = HypGridLayer(grid=self.layer2(trav=False),
                                   hyp=self.hyp,
                                   high=0.5,
                                   step=1.5,
                                   color=(25, 229, 100),
                                   name="Layer-3"
                                   )
        return self


def webgl_line_backend(**props):
    def wrapper(item):
        return LineObject(points=[np.array(item.start, dtype=float).tolist(), np.array(item.end, dtype=float).tolist()],
                          **props)

    return wrapper


"""
def f(node):
    # *r,=range(1,len(dl)-2)
    if node.next is None:
        pass

    elif node.prev is not None:
        return list(zip(hyp_transform(node.data, node.next.data), hyp_transform(node.data, node.prev.data)))


def ff(dl):
    for i in range(1, len(dl)):
        item = dl.get(i)
        yield f(item)


def no_mp(dl):
    return list(ff(dl))

"""

nc = NurbsCurve([[0, 0, 0], [1, 0, 1], [2, 3, 4], [3, 3, 3]])


@dataclasses.dataclass
class Circle:
    r: float

    def evaluate(self, t):
        return np.array([self.r * np.cos(t * 2 * np.pi), self.r * np.sin(t * 2 * np.pi), 0.0], dtype=float)

    @property
    def plane(self):
        return WorldXY

import zlib
@dataclasses.dataclass
class Circle3D(Circle):
    origin: tuple[float, float, float] = (0, 0, 0)
    normal: tuple[float, float, float] = (0, 0, 1)
    torsion: tuple[float, float, float] = (1, 0, 0)  # The xaxis value of the base plane,

    # defines the point of origin of the circle
    @property
    def plane(self):
        return PlaneLinear(normal=unit(self.normal), origin=self.origin)

    @functools.lru_cache(maxsize=128)
    def evaluate(self, t):
        try:
            return self.plane.orient(super().evaluate(t), super().plane)
        except NotImplementedError:
            return remove_crd(
                add_crd(super().evaluate(t), 1).reshape((4,)) @ self.plane.transform_from_other(WorldXY).matrix.T)

    def __hash__(self):
        return zlib.adler32(self.__repr__().encode())
    def __eq__(self, other):
        return self.__repr__()==other.__repr__()
@dataclasses.dataclass
class Pipe:
    """
    >>> nb2=NurbsCurve([[0, 0, 0        ] ,
    ...                 [-47, -315, 0   ] ,
    ...                 [-785, -844, 0  ] ,
    ...                 [-704, -1286, 0 ] ,
    ...                 [-969, -2316, 0 ] ] )
    >>> r=Circle(r=10.5)
    >>> oo=Pipe(nb2, r)

    """
    path: ParametricObject
    shape: ParametricObject

    def evalplane(self, t):
        pt = self.path.tan(t)
        return PlaneLinear(origin=pt.point, normal=pt.normal)

    def evaluate(self, t):
        u, v = t

        pln = self.evalplane(u)

        return remove_crd(
            add_crd(self.shape.evaluate(v), 1).reshape((4,)).tolist() @ pln.transform_from_other(WorldXY).matrix.T)

    def geval(self, uvs=(20, 20), bounds=((0, 1), (0, 1))):
        for i, u in enumerate(np.linspace(*bounds[0], uvs[0])):
            for j, v in enumerate(np.linspace(*bounds[1], uvs[1])):
                yield EvalPointTuple2D(i, j, u, v, *self.evaluate([u, v]))

    def veval(self, uvs=(20, 20), bounds=((0, 1), (0, 1))):
        data = np.zeros(uvs + (3,), dtype=float)
        for i, j, u, v, x, y, z in self.geval(uvs, bounds):
            data[i, j, :] = [x, y, z]
        return data

    def mpeval(self, uvs=(20, 20), bounds=((0, 1), (0, 1)), workers=-1):
        """
        >>> path = NurbsCurve([[0, 0, 0],
        ...               [-47, -315, 0],
        ...               [-785, -844, 0],
        ...               [-704, -1286, 0],
        ...               [-969, -2316, 0]])

        >>> profile = Circle(r=10.5)
        >>> pipe = Pipe(path, profile)
        >>> pipe.veval(uvs=(2000, 200)) # 400,000 points
        time 40.84727501869202 s
        >>> pipe.mpeval(uvs=(2000, 200)) # 400,000 points
        time 8.37929892539978 s # Yes it's also too slow, but it's honest work
        """

        def inner(u):
            return [(u, v, *self.evaluate([u, v])) for v in np.linspace(*bounds[1], uvs[1])]

        if workers == -1:
            workers = os.cpu_count()

        with mp.Pool(workers) as p:
            return p.map(inner, np.linspace(*bounds[0], uvs[0]))
