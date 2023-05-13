import abc
import copy
import dataclasses
import math
import sys
import typing
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from itertools import starmap

import mmcore
from mmcore.base import geomdict, objdict
from mmcore.base.geom import MeshBufferGeometryBuilder, GeometryObject, MeshObject
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
from mmcore.geom.materials import ColorRGB
from mmcore.base.models.gql import MeshPhongMaterial
from mmcore.geom.basic import Point
from mmcore.base.basic import Group
from mmcore.base.geom import LineObject
from geomdl import NURBS
from geomdl import utilities as geomdl_utils
from mmcore.collections import DCLL, DoublyLinkedList
from mmcore.collections.multi_description import EntityCollection, ElementSequence

import uuid as _uuid
from geomdl.operations import normal, tangent

TOLERANCE = 0.001

T = typing.TypeVar("T")


@typing.runtime_checkable
class ParametricObject(typing.Protocol[T]):
    @abc.abstractmethod
    def evaluate(self, t):
        ...


class Linear(LineObject, ParametricObject):
    x0: float = 0
    y0: float = 0
    z0: float = 0
    a: float = 1
    b: float = 0
    c: float = 0
    _points = None
    x = lambda self, t: self.x0 + self.a * t
    y = lambda self, t: self.y0 + self.b * t
    z = lambda self, t: self.z0 + self.c * t
    _direction=None
    @property
    def points(self):
        return np.asarray([[self.evaluate(0)], [self.evaluate(1)]], dtype=float).tolist()

    @points.setter
    def points(self, v):
        start, end = np.asarray(v, dtype=float).tolist()
        self._points = start, end
        self.a, self.b, self.c = np.asarray(end) - np.asarray(start)
        self.x0, self.y0, self.z0 = start

        self.solve_geometry()

    @classmethod
    def from_two_points(cls, start, end):
        a, b, c = np.asarray(end) - np.asarray(start)
        x, y, z = start
        x0 = x + a * 0
        y0 = y + b * 0
        z0 = z + c * 0
        x1 = x + a * 1
        y1 = y + b * 1
        z1 = z + c * 1

        return cls(points=np.asarray([[x0, y0, z0], [x1, y1, z1]], dtype=float).tolist())

    def evaluate(self, t):
        return np.array([self.x(t), self.y(t), self.z(t)])

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

    def extend(self, a,b):
        self(points=[self.start-unit(self.direction)*a,self.end+unit(self.direction)*b])



class Line(LineObject):
    _color = ColorRGB(10, 170, 40)

    def __new__(cls, start, end, *args, **kwargs):
        proxy = Linear.from_two_points(start, end)
        inst = super().__new__(cls, points=[start, end], *args, **kwargs)
        inst._proxy = proxy

        return inst

    @property
    def proxy(self):
        return self._proxy

    @proxy.setter
    def proxy(self, v):
        self._proxy = v

    def from_points(self):
        start, end = self.points
        self.proxy = Linear.from_two_points(start, end)


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


T = typing.TypeVar("T")


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
        # print(f"event: set {self.proxy_name}/{self.name}->{v}")
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

    def tessellate(self):
        ...


from mmcore.collections import curry


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
        # #print(self)

        self._proxy.degree_u, self._proxy.degree_v = self.degree

    def normal_at(self, t):
        return normal(self.proxy, t)

    def tangent_at(self, t):
        return tangent(self.proxy, t)

    def tessellate(self, uuid=None):
        self.proxy.tessellate()
        vertseq = ElementSequence(self._proxy.vertices)
        faceseq = ElementSequence(self._proxy.faces)
        uv = np.round(np.asarray(vertseq["uv"]), decimals=5)
        normals = [v for p, v in normal(self._proxy, uv.tolist())]
        if uuid is None:
            uuid = _uuid.uuid4().__str__()
        return MeshBufferGeometryBuilder(vertices=np.array(vertseq['data']).flatten(),
                                         normals=np.array(normals).flatten(),
                                         indices=np.array(faceseq["vertex_ids"]).flatten(),
                                         uv=uv,
                                         uuid=uuid)


class NurbsSurfaceGeometry(MeshObject):
    material_type = MeshPhongMaterial
    castShadow: bool = True
    receiveShadow: bool = True
    geometry_type = MeshBufferGeometryBuilder

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
        super(GeometryObject, self).__call__(*args, material=self.material, **kwargs)

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
class Polyline(ParametricObject):
    control_points: typing.Union[DCLL, DoublyLinkedList]
    closed: bool = False

    @classmethod
    def from_points(cls, pts):
        closed = False
        if all(pts[0] == pts[-1]):
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
            lnr.append(Linear(points=[copy.deepcopy(h.data), copy.deepcopy(h.next.data)]))
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

    def __iter__(self):
        return iter(self.point)
    def __array__(self):
        return np.asarray(self.point,dtype=float)

from mmcore.geom.vectors import unit


@dataclasses.dataclass
class PlaneLinear(ParametricObject):
    origin: typing.Iterable[float]
    normal: typing.Optional[typing.Iterable[float]] = None
    xaxis: typing.Optional[typing.Iterable[float]] = None
    yaxis: typing.Optional[typing.Iterable[float]] = None

    def __post_init__(self):
        # print(unit(self.normal), self.xaxis, self.yaxis)
        self.xaxis = np.cross(unit(self.normal), np.array([0, 0, 1]))
        self.yaxis = np.cross(unit(self.normal), self.xaxis
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
        return ClosestPoint(pt, self)


from enum import Enum


@dataclasses.dataclass
class HyPar4pt(ParametricObject):
    a: typing.Iterable[float]
    b: typing.Iterable[float]
    c: typing.Iterable[float]
    d: typing.Iterable[float]

    sides_enum: 'typing.Optional[typing.Type[Enum,...]]' = None

    def generate_sides_enum(self):
        class HypSidesEnum(Enum):
            A = "side_a"
            B = "side_b"
            C = "side_c"
            D = "side_d"

        self.sides_enum = HypSidesEnum

    def __post_init__(self):
        self._side_a = Linear(points=[self.a, self.b])
        self._side_b = Linear(points=[self.b, self.c])
        self._side_c = Linear(points=[self.d, self.c])
        self._side_d = Linear(points=[self.a, self.d])
        self.generate_sides_enum()

    @property
    def side_a(self):
        self._side_a(points=[self.a, self.b])
        return self._side_a

    @property
    def side_b(self):
        self._side_b(points=[self.b, self.c])
        return self._side_b

    @property
    def side_c(self):
        self._side_c(points=[self.d, self.c])
        return self._side_c

    @property
    def side_d(self):
        self._side_d(points=[self.a, self.d])
        return self._side_d

    def evaluate(self, t):
        _u,_v=0.5,0.5
        lu = Linear(points=[self.side_a.evaluate(_u), self.side_c.evaluate(_u)])
        lv = Linear(points=[self.side_b.evaluate(_v), self.side_d.evaluate(_v)])

        def evl(tt):
            u, v = tt
            lu(points=[self.side_a.evaluate(u), self.side_c.evaluate(u)])
            lv(points=[self.side_b.evaluate(v), self.side_d.evaluate(v)])
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
        for i in self.polyline.segments:
            res = line_plane_collision(pln, i)

            if res is not None:
                re = ClosestPoint(res, i)(x0=0.5, bounds=[(0.0, 1.0)]).t
                # print("RE", re)
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
class HypPar4ptGrid(HyPar4pt):
    def parallel_side_grid(self, step1, side: str = "D"):
        if hasattr(self, "_grd"):
            for item in self._grd:
                objdict.__delitem__(item.uuid)
                del item
        self._grd = []
        side = getattr(self, getattr(self.sides_enum, side).value)

        for pl in side.divide_distance_planes(step1):

            # print("t",dd)
            r = self.intr(pl)
            if not (r == []):
                a, b = r
                self._grd.append(Linear(points=[a.tolist(), b.tolist()]))


        return self._grd

    def parallel_vec_grid(self, step1, name:str, vec: Linear):
        if hasattr(self, name):
            for item in getattr(self, name):
                objdict.__delitem__(item.uuid)
                del item

        d = []
        for pl in vec.divide_distance_planes(step1):

            # print("t",dd)
            r = self.intr(pl)
            if not (r == []):
                a, b = r
                d.append(Linear(points=[a.tolist(), b.tolist()]))
        setattr(self,name,d)

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

            # print("t",dd)
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
    >>> pt=np.array([13.197247, 21.228605, 0])
    >>> prx=ProxPoints(A,B)
    >>> prx()
    ClosestPointSolution(pt=[array([16.27517685, 16.07437063,  4.86901707]), array([15.75918043, 14.67951531, 14.57947997])], t=array([0.52562605, 0.50105099]), distance=9.823693977393207)
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.c1, self.c2 = c1, c2

    def solution(self, t) -> float:
        t1, t2 = t

        return euclidean(np.array(self.c1.evaluate(t1)), np.array(self.c2.evaluate(t2)))

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
from scipy.optimize import fsolve


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
        r = np.array(self.gm.evaluate(t))
        r = np.array(r).T.flatten()
        # print(self.point, r)

        return euclidean(self.point, r)

    def prepare_solution_response(self, solution):
        return self.solution_response([self.gm.evaluate(solution.x)], solution.x, solution.fun)

    def __call__(self, x0=(0.5,), **kwargs):
        return super().__call__(x0, **kwargs)


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
