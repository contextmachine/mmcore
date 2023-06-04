import functools

import copy
import dataclasses
import sys
import typing
from collections import namedtuple

from itertools import starmap

import dill
import scipy.optimize

from mmcore.geom.parametric.algorithms import EvaluatedPoint, ProximityPoints, ClosestPoint, CurveCurveIntersect

from mmcore.geom.parametric.nurbs import NurbsCurve, NurbsSurface
from mmcore.geom.transform import remove_crd, Transform, WorldXY
from enum import Enum

from mmcore.geom.parametric.base import ParametricObject
from mmcore.geom.vectors import unit

from mmcore.func.curry import curry
from scipy.optimize import minimize
import numpy as np
from scipy.spatial.distance import euclidean
from mmcore.geom.materials import ColorRGB

from mmcore.collections import DCLL, DoublyLinkedList
from mmcore.collections.multi_description import EntityCollection
import base64


def add_crd(pt, value):
    if not isinstance(pt, np.ndarray):
        pt = np.array(pt, dtype=float)
    if len(pt.shape) == 1:
        pt = pt.reshape(1, pt.shape[0])

    return np.c_[pt, np.ones((pt.shape[0], 1)) * value]


def add_w(pt):
    return add_crd(pt, value=1)


# mp.get_start_method = "swapn"

TOLERANCE = 1e-6

T = typing.TypeVar("T")


@dataclasses.dataclass
class PolyNominal(ParametricObject):
    func: typing.Callable

    def evaluate(self, t):
        return self.func(t)

    @property
    def __params__(self):
        return {"func": base64.b64encode(dill.dumps(self.func))}


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

    def distance_func(self, point):
        """
        return a distance function with t param.
        """

        def inner(t):
            x_0, y_0, z_0 = point
            x_1, y_1, z_1 = self.start
            x_2, y_2, z_2 = self.end
            return ((x_1 - x_0) + (x_2 - x_1) * t) ** 2 \
                + ((y_1 - y_0) + (y_2 - y_1) * t) ** 2 \
                + ((z_1 - z_0) + (z_2 - z_1) * t) ** 2

        return PolyNominal(inner)

    def distance_at(self, point, t):
        return self.distance_func(point).evaluate(t)

    def distance(self, point):
        return minimize(self.distance_func(point).evaluate, x0=np.array([0.5]), bounds=[0, 1])
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

    @property
    def __params__(self):
        return dataclasses.asdict(self)

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


def is_collinear(a, b):
    dp = np.dot(a, b)
    return dp, dp * (1 if dp // 1 >= 0 else -1)


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

    @property
    def __params__(self):
        return dataclasses.asdict(self)

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
    pass
else:
    pass

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


@dataclasses.dataclass
class Circle(ParametricObject):
    r: float

    @property
    def __params__(self):
        return [self.r]

    @functools.lru_cache
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

    @functools.lru_cache(maxsize=512)
    def evaluate(self, t):
        try:
            return self.plane.orient(super().evaluate(t), super().plane)
        except NotImplementedError:
            return remove_crd(
                add_crd(super().evaluate(t), 1).reshape((4,)) @ self.plane.transform_from_other(WorldXY).matrix.T)

    @property
    def __params__(self):
        return [self.origin, self.normal]

    def __hash__(self):
        return super(ParametricObject, self).__hash__()