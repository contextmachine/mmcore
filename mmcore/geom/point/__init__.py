import string
import typing

import numpy as np
from mmcore.base import Delegate, APoints, AGroup
from mmcore.base.components import Component
from mmcore.base.models.gql import PointsMaterial
from mmcore.base.params import ParamGraph, pgraph
from mmcore.collections import DCLL
from mmcore.geom.materials import ColorRGB
from mmcore.geom.parametric import PlaneLinear, Linear
from mmcore.geom.parametric.algorithms import ray_triangle_intersection


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

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __call__(self, **kwargs):
        super().__call__(**kwargs)
        self.__repr3d__()
        return self

    def __repr3d__(self):
        self._repr3d = APoints(uuid=self.uuid,
                               name=self.name,
                               geometry=[self.x, self.y, self.z],
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
    __exclude__ = ["points_keys"]

    def __new__(cls, points=(), *args, **kwargs):
        node = super().__new__(cls, *args, **dict(zip(string.ascii_lowercase[:len(points) + 1], points)),
                               points_keys=list(string.ascii_lowercase[:len(points)]), **kwargs)

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
                    yield line_plane_collision(plane, Linear.from_two_points(self.points[a], self.points[b])).tolist()

    def plane_split(self, plane):
        sideA, sideB = self.divide_vertices_from_plane(plane)
        if (sideB == []) or (sideA == []):
            return [self.points[i] for i in sideA], [self.points[i] for i in sideB]

        else:
            if len(sideA) == 2:
                A0 = self.points[sideA[0]]
                A1 = self.points[sideA[1]]
                B0 = self.points[sideB[0]]
                AB1, AB2 = line_plane_collision(plane, Linear.from_two_points(A0, B0)).tolist(), line_plane_collision(
                    plane, Linear.from_two_points(A1, B0)).tolist()
                return [A0, A1, AB2, AB1], [B0, AB1, AB2]
            else:
                A0 = self.points[sideA[0]]
                B0 = self.points[sideB[0]]
                B1 = self.points[sideB[1]]
                BA1, BA2 = line_plane_collision(plane, Linear.from_two_points(A0, B0)).tolist(), line_plane_collision(
                    plane, Linear.from_two_points(A0, B1)).tolist()

                return [A0, BA1, BA2], [B0, B1, BA2, BA1],

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
