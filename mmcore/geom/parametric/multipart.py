import numpy as np
import typing

import dataclasses
from scipy.spatial.distance import euclidean
from mmcore.collections import DoublyLinkedList, DNode
from mmcore.geom.parametric import ParametricObject, PlaneLinear, Linear
from mmcore.geom.vectors import unit


class GeomDNode(DNode):

    @property
    def plane(self):
        return PlaneLinear.from_tree_pt(self.data, self.prev.data, self.next.data)

    def crc(self):
        vec1= unit(np.cross(self.dprev, self.plane.normal) * self.fillet)
        vec2 = unit(np.cross(self.dnext, self.plane.normal) * self.fillet)
        return self.data + vec1,self.data+vec2

    @property
    def data_local(self):
        self.plane.point_at(self.data)

    @property
    def prev_distance(self):
        if self.prev is None:
            return 0.0
        return euclidean(self.data, self.prev.data)

    @property
    def next_distance(self):
        if self.next is None:
            return 0.0
        return euclidean(self.data, self.next.data)

    @property
    def dnext(self):
        return unit(self.next.data - self.data)

    @property
    def dprev(self):
        """
        unit(self.data - self.prev.data)
        @return:
        """
        return unit( self.data-self.prev.data )

    @property
    def dnext_local(self):
        return unit(self.next.data_local - self.data_local)

    @property
    def dprev_local(self):
        """
        unit(self.data - self.prev.data)
        @return:
        """
        return unit(self.prev.data_loca - self.data_local)
    @property
    def deriv(self):
        return np.dot(self.dnext, self.dprev)

    def tabs(self):
        if self.prev is None:
            return 0

        else:
            return self.prev.tabs() + self.prev_distance

    @property
    def fillet(self):
        return self._fillet

    @fillet.setter
    def fillet(self, v):
        self._fillet = v


class GeomDLL(DoublyLinkedList):
    __node_type__ = GeomDNode

    def evaluate(self, t):

        for i in range(self.count):
            node = self.get(i)
            if node.tabs() > t:
                tr = t - node.prev.tabs()
                return node.data + node.dprev * tr


@dataclasses.dataclass
class PolyLine(ParametricObject):
    pts: dataclasses.InitVar[list[tuple[float, float, float]]]
    points: typing.Optional[DoublyLinkedList] = None

    def __post_init__(self, pts):
        self.points = GeomDLL(pts)

    def evaluate(self, t): ...

    def length(self): ...
