import gzip
import string
import typing
import uuid as _uuid

import numpy as np

from mmcore.base import AGroup, APoints, Delegate
from mmcore.base.components import Component
from mmcore.base.models.gql import PointsMaterial
from mmcore.base.params import ParamGraph, pgraph
from mmcore.collections import DCLL
from mmcore.geom.materials import ColorRGB
from mmcore.geom.parametric import Linear, PlaneLinear
from mmcore.geom.parametric import line_plane_intersection, ray_triangle_intersection


def recsetter(obj, i, val):
    if isinstance(i, (int, str)):
        obj[i] = val
    elif len(i) == 1:
        obj[i[0]] = val
    else:
        return recsetter(obj[i[0]], i[1:], val)


def recgetter(obj, i):
    if isinstance(i, (int, str)):
        return obj[i]
    elif len(i) == 1:
        return obj[i[0]]
    else:
        return recgetter(obj[i[0]], i[1:])


class PointProtocol(typing.Protocol):
    x: typing.Any
    y: typing.Any
    z: typing.Any

    def __iter__(self) -> typing.Iterator[float]:
        # return iter([self.x, self.y, self.z])
        ...

    def __array__(self) -> np.ndarray:
        # return np.array([self.x, self.y, self.z])
        ...


class ClosestPointProtocol(typing.Protocol):
    def closest_point(self, point: PointProtocol) -> PointProtocol: ...


def __point_getitem__(self, i):
    return np.ndarray.__getitem__(self, i)


@Delegate(np.ndarray)
class PointArrayProxy:
    """
    Simple proxy object on a np.ndarray instance.
    >>> pr=PointArrayProxy(np.array([[1,2,3],
    ...                              [2,3,4]]))
    >>> pr
    array([ [1, 2, 3],
            [2, 3, 4]])
    >>> pr[0]
    <__main__.PointItem at 0x14e184610>
    >>> pt=pr[0]
    >>> pt.x
    1
    >>> pt.x=3
    >>>pr
    array([ [3, 2, 3],
            [2, 3, 4]])
    >>> pr[0,2]=7
    >>> pt.z
    7
    """

    def __init__(self, arr):
        super().__init__()
        if isinstance(arr, np.ndarray):
            self._ref = arr
        else:
            raise TypeError(f'{arr}')

    @property
    def wrapped_array(self):
        return self._ref

    @wrapped_array.setter
    def wrapped_array(self, v: np.ndarray):
        if isinstance(v, np.ndarray):
            self._ref = v
        else:
            raise TypeError(f'{v}')

    def __getitem__(self, item):
        res = self._ref.__getitem__(item)
        if len(res.shape) == 1 and res.shape[0] == 3:
            return PointProxy(item, self)
        else:
            return res

    def __setitem__(self, key, value):
        self._ref.__setitem__(key, value)

    def __repr__(self):
        return self._ref.__repr__()


class Point(PointProtocol):

    def __iter__(self):
        return iter([self.x, self.y, self.z])

    def __array__(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def perp_to(self, anything: ClosestPointProtocol):
        return anything.closest_point(self)


class PointProxy(Point):
    def __new__(cls, ptr, point_owner: PointArrayProxy, **kwargs):
        obj = super().__new__(cls)
        obj.point_owner = point_owner

        obj.point_ptr = ptr
        return obj

    def __iter__(self):
        return iter([self.x, self.y, self.z])

    def __array__(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def perp_to(self, anything: ClosestPointProtocol):
        return anything.closest_point(self)

    @property
    def x(self):
        return self.point_owner[self.point_ptr, 0]

    @x.setter
    def x(self, v):
        self.point_owner[self.point_ptr, 0] = v

    @property
    def y(self):
        return self.point_owner[self.point_ptr, 1]

    @y.setter
    def y(self, v):
        self.point_owner[self.point_ptr, 1] = v

    @property
    def z(self):
        return self.point_owner[self.point_ptr, 2]

    @z.setter
    def z(self, v):
        self.point_owner[self.point_ptr, 2] = v


class ControlPoint(Component, PointProtocol):
    x: float = 0
    y: float = 0
    z: float = 0
    size: float = 0.5
    color: tuple = (157, 75, 75)
    __exclude__ = Component.__exclude__

    def __call__(self, **kwargs):
        super().__call__(**kwargs)
        self.__repr3d__()
        return self

    def __repr3d__(self):
        self._repr3d = APoints(uuid=self.uuid,
                               name=self.name,
                               geometry=list(self),
                               material=PointsMaterial(color=ColorRGB(*self.color).decimal, size=self.size),
                               _endpoint=self.endpoint,
                               controls=self.param_node.todict())

        return self._repr3d

    def __iter__(self):
        return iter([self.x, self.y, self.z])


from dill import pointers


class ControlPointProxy(ControlPoint):
    __exclude__ = ["point"]

    def __new__(cls, point: PointProxy, *args, **kwargs):
        node = super().__new__(cls, point=point, x=point.x, y=point.y, z=point.z,
                               uuid=hex(pointers.reference(point.point_owner)) + f"-{point.point_ptr}", *args,
                               **kwargs)
        node.resolver.point = point

        return node

    @property
    def x(self):
        return self.point.x

    @x.setter
    def x(self, v):
        self.point.x = v

    @property
    def y(self):
        return self.point.y

    @y.setter
    def y(self, v):
        self.point.y = v

    @property
    def z(self):
        return self.point.z

    @z.setter
    def z(self, v):
        self.point.z = v

    def array_sync(self):
        self.param_node(x=self.x, y=self.y, z=self.z)


class ControlPointList(Component):
    __exclude__ = ["points_keys", "seq_names"]
    point_type = ControlPoint
    seq_names: list[str] = []

    def __class_getitem__(cls, item):

        return type(cls.__name__ + item.__name__, (cls,),
                    {"__qualname__": f'{cls.__name__}[{item.__name__}]', "point_type": item})

    def __new__(cls, points=(), *args, **kwargs):
        points_keys = list(string.ascii_lowercase[:len(points)])
        if not isinstance(points[0], (Component, ControlPoint)):
            cpts = []
            for i, pt in enumerate(points):
                print(pt)
                x, y, z = pt
                cpts.append(
                    cls.point_type(x=x, y=y, z=z, name="Point" + points_keys[i].upper(), uuid="point" + points_keys[i]))
            points = cpts
        node = super().__new__(cls, *args, **dict(zip(string.ascii_lowercase[:len(points) + 1], points)),
                               points_keys=points_keys, seq_names=points_keys, **kwargs)

        return node

    def __array__(self):
        return np.array([list(i) for i in self.points], dtype=float)

    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return self.points.__len__()

    def __getitem__(self, item):
        return self.points.__getitem__(item)

    @property
    def points(self):
        lst = []
        for k in self.points_keys:
            lst.append(getattr(self, k))
        return lst

    def __repr3d__(self):
        self._repr3d = AGroup(seq=self.points, uuid=self.uuid, name=self.name)

        return self._repr3d


def update_array_children(self, graph: ParamGraph = pgraph):
    for k in graph.item_table.keys():
        if k.startswith(hex(pointers.reference(self))):

            if isinstance(graph.item_table[k].resolver, ControlPointProxy):
                ptr, i = k.split("-")
                print(graph.item_table[k])
                graph.item_table[k](x=self._ref[int(i), 0], y=self._ref[int(i), 1], z=self._ref[int(i), 2])


ControlPointBaseList = ControlPointList[ControlPointList]


class ControlPointProxyList(ControlPointList):
    __exclude__ = ["points_keys", "seq_names"]
    seq_names: list[str]

    def __new__(cls, points=(), *args, **kwargs):
        points_keys = list(string.ascii_lowercase[:len(points)])
        if not isinstance(points[0], (Component, PointProxy, ControlPointProxy)):
            arr_proxy = PointArrayProxy(np.array(points, dtype=float))
            cpts = []
            for i in range(len(points)):
                cpts.append(ControlPointProxy(PointProxy(i, arr_proxy), name="Point" + points_keys[i].upper()))
            points = cpts
        node = Component.__new__(cls, *args, **dict(zip(string.ascii_lowercase[:len(points) + 1], points)),
                                 points_keys=points_keys, **kwargs)
        return node


class Triangle:
    def __init__(self, pta, ptb, ptc):
        self.points = DCLL.from_list([pta, ptb, ptc])
        self.plane = PlaneLinear.from_tree_pt(pta, ptb, ptc)

    def ray_intersection(self, ray):
        return ray_triangle_intersection(ray[0], ray[1], self.points)

    def plane_intersection(self, plane):
        sideA, sideB = self.divide_vertices_from_plane(plane)
        if (sideB == []) or (sideA == []):
            yield None

        else:
            for a in sideA:
                for b in sideB:
                    yield line_plane_intersection(plane,
                                                  Linear.from_two_points(self.points[a], self.points[b])).tolist()

    def plane_split(self, plane):
        sideA, sideB = self.divide_vertices_from_plane(plane)
        if (sideB == []) or (sideA == []):
            return [self.points[i] for i in sideA], [self.points[i] for i in sideB]

        else:
            if len(sideA) == 2:
                A0 = self.points[sideA[0]]
                A1 = self.points[sideA[1]]
                B0 = self.points[sideB[0]]
                AB1, AB2 = line_plane_intersection(plane,
                                                   Linear.from_two_points(A0, B0)).tolist(), line_plane_intersection(
                    plane, Linear.from_two_points(A1, B0)).tolist()
                return [A0, A1, AB2, AB1], [B0, AB1, AB2]
            else:
                A0 = self.points[sideA[0]]
                B0 = self.points[sideB[0]]
                B1 = self.points[sideB[1]]
                BA1, BA2 = line_plane_intersection(plane,
                                                   Linear.from_two_points(A0, B0)).tolist(), line_plane_intersection(
                    plane, Linear.from_two_points(A0, B1)).tolist()

                return [A0, BA1, BA2], [B0, B1, BA2, BA1],

    def plane_cut(self, plane):
        sideA, sideB = self.plane_split(plane)
        return sideA

    def divide_vertices_from_plane(self, plane):
        node = self.points.head
        la = []
        lb = []
        for i in range(3):
            if plane.side(node.data):
                la.append(i)
            else:
                lb.append(i)
            node = node.next
        return la, lb

    @property
    def lines(self):
        node = self.points.head
        lns = []
        for i in range(3):
            lns.append(Linear.from_two_points(node.data, node.next.data))
            node = node.next
        return lns


from mmcore.base.components import Component


class ComponentList(Component):
    __exclude__ = ["cmps", "comp_type", "prefix", "seq_names"]

    comp_type: type = Component
    prefix: str = "Component"

    def __new__(cls, cmps=(), *args, prefix="Component", **kwargs):
        items = {}
        names = list(string.ascii_lowercase[:len(cmps)])
        for i, pt in enumerate(cmps):
            items[names[i]] = pt

        return super().__new__(cls, *args, seq_names=names, **kwargs, **items)

    def __getitem__(self, item):
        return getattr(self, self.seq_names[item])


class CompShapeList(ComponentList):
    bounds: ControlPointProxyList

    def __new__(cls, bounds, hls=(), *args, **kwargs):
        node = super().__new__(cls, cmps=hls, bounds=bounds, *args, **kwargs)
        node.resolver.seq_names = ["bounds"] + node.resolver.seq_names
        return node


from mmcore.geom.reflection import Ptr, tables
BUFFERS = dict()
tables["buffers"] = BUFFERS
BUFFERS_PTR = Ptr("buffers")


class BufferPtr(Ptr):
    def __new__(cls, ref, parent_ptr=BUFFERS_PTR, **kwargs):
        return super().__new__(cls, ref, parent_ptr=parent_ptr, **kwargs)
class GeometryBuffer:
    _tolerance = -1
    _remove_duplicates = True
    __props__ = ['tolerance', 'remove_duplicates']
    _uuid = None

    def __init__(self, buffer=None, uuid=None, **kwargs):
        super().__init__()
        if (uuid is None) and len(BUFFERS) == 0:
            uuid = "default"

        self._buffer = [] if buffer is None else buffer
        self._uuid = _uuid.uuid4().hex if uuid is None else uuid
        BUFFERS[self._uuid] = self
        self.update_props(kwargs)

    def index(self, val):
        return self._buffer.index(val)

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, v):
        global BUFFERS
        self._uuid = v
        BUFFERS[v] = self

    def add_items(self, pts):
        ixs = []
        for pt in pts:
            ixs.append(self.add_item(pt))
        return ixs

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value

    @property
    def remove_duplicates(self):
        return self._remove_duplicates

    @remove_duplicates.setter
    def remove_duplicates(self, value):
        self._remove_duplicates = value

    def __getstate__(self):
        dct = dict(
            __props__=self.__props__,
            buffer=self._buffer,
            uuid=self.uuid)
        for k in self.__props__:
            dct[k] = getattr(self, k)

        return dct

    def update_props(self, props):
        for k in self.__props__:
            if k in props.keys():
                setattr(self, k, props[k])

    def __setstate__(self, state):
        u = state.pop("uuid")
        BUFFERS[u] = self
        self._uuid = u
        self.__props__ = state.pop('__props__')
        self._buffer = state.pop("buffer")

        self.update_props(state)

    def add_item(self, point):
        if self._tolerance >= 0:
            value = list(round(i, self.tolerance) for i in point)
        else:
            value = point
        if self.remove_duplicates:
            if value not in self._buffer:
                self._buffer.append(value)
                return len(self._buffer) - 1
            else:
                return self._buffer.index(value)
        else:
            self._buffer.append(value)
            return len(self._buffer) - 1

    def update_item(self, i, val):

        self._buffer[i] = val

    def update_all_items(self, vals):
        l = len(self._buffer)
        for i, pt in enumerate(vals):
            if i < l:
                self.update_item(i, pt)
            else:
                self.add_item(pt)

    def update_all(self, vals):

        for i, pt in enumerate(vals):
            self._buffer[i] = pt

    def get_all_points(self):
        return self._buffer

    def get_points(self, ixs):
        return [self._buffer[i] for i in ixs]

    def __getitem__(self, item):
        return recgetter(self._buffer, item)

    def __setitem__(self, item, value):

        recsetter(self._buffer, item, value)

    def __len__(self):
        return self._buffer.__len__()

    def __contains__(self, item):
        return item in self._buffer

    def __iter__(self):
        return iter(self._buffer)

    def append(self, v):
        return self.add_item(v)

    def dumps(self):
        return gzip.compress(str(self._buffer).encode(), compresslevel=9)

    def loads(self, b: bytes):
        self._buffer = eval(gzip.decompress(b).decode())
        return self


default = GeometryBuffer(uuid="default")
