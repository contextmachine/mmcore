from OCC.Core import gp
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCC.Core.gp import gp_Pnt

from mmcore.collections import DCLL
from mmcore.geom.point import BUFFERS
from mmcore.geom.shapes.base import OccShape, ShapeDCLL

defbuf = BUFFERS["default"]


def make_occ_face(self):
    return BRepBuilderAPI_MakeFace(dcll_to_occ_wire(self)).Face()


def make_occ_shape(self):
    _occ_shape = self._occ_shape = OccShape(BRepBuilderAPI_MakeFace(make_occ_face(self)).Shape())
    return _occ_shape


def dcll_to_occ_wire(points: DCLL):
    mkw = BRepBuilderAPI_MakeWire()

    node = points.head
    for pt in range(len(points)):
        mkw.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(*node.data), gp_Pnt(*node.next.data)).Edge())
        node = node.next

    return mkw.Wire()


class OccPoints:
    def __init__(self, pts):
        self._i = -1
        self.ixs = [defbuf.append(pt) for pt in pts]

    def __iter__(self):
        return self

    def __next__(self):
        self._i += 1
        if len(self.ixs) == self._i:
            raise StopIteration

        return gp.gp_Pnt(*defbuf[self._i])

    def todcll(self):
        return ShapeDCLL.from_list([defbuf[j] for j in self.ixs])

    def towire(self):
        return dcll_to_occ_wire(self.todcll())

    def toshape(self):
        return make_occ_shape(self.todcll())

    def toface(self):
        return make_occ_face(self.todcll())
