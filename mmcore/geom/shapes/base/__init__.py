import dataclasses
import typing

import earcut
import numpy as np
from earcut import earcut

from mmcore.base import AMesh
from mmcore.base.geom import MeshData
from mmcore.collections import DCLL
from mmcore.geom.parametric import Linear, PlaneLinear, line_plane_intersection, ray_triangle_intersection
from mmcore.geom.tess import simple_tessellate
from mmcore.geom.transform import TransformV2
from mmcore.geom.vectors import V3

HAS_OCC = True
try:
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_GTransform,
    )
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Common
    from OCC.Core.gp import (
        gp_Pnt2d,
        gp_Circ2d,
        gp_Ax2d,
        gp_Dir2d,
        gp_Pnt,
        gp_Pln,
        gp_Vec,
        gp_OX,
        gp_Trsf,
        gp_GTrsf,
    )
    from OCC.Core.BRepLib import breplib_BuildCurves3d
    from OCC.Core.BRepOffset import BRepOffset_Skin
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakePrism
    from OCC.Display.SimpleGui import init_display
    from OCC.Core.GCE2d import GCE2d_MakeLine
    from OCC.Core.Geom import Geom_Plane
    from OCC.Core.Geom2d import Geom2d_Circle
    from OCC.Core.GeomAbs import GeomAbs_Arc
    from OCC.Core.TopTools import TopTools_ListOfShape
    from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape, TopoDS_Solid
    from OCC.Extend.TopologyUtils import TopologyExplorer
except Exception as err:
    print("OCC is not provide")
    HAS_OCC = False

USE_OCC = True


class Tri:
    """A 3d triangle"""

    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    def map(self, f):
        return Tri(f(self.v1), f(self.v2), f(self.v3))


class Quad:
    """A 3d quadrilateral (polygon with 4 vertices)"""

    def __init__(self, v1, v2, v3, v4):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.v4 = v4

    def map(self, f):
        return Quad(f(self.v1), f(self.v2), f(self.v3), f(self.v4))

    def swap(self, swap=True):
        if swap:
            return Quad(self.v4, self.v3, self.v2, self.v1)
        else:
            return Quad(self.v1, self.v2, self.v3, self.v4)


class SimpleMesh:
    """A collection of vertices, and faces between those vertices."""

    def __init__(self, verts=None, faces=None):
        self.verts = verts or []
        self.faces = faces or []

    def extend(self, other):
        l = len(self.verts)
        f = lambda v: v + l
        self.verts.extend(other.verts)
        self.faces.extend(face.map(f) for face in other.faces)

    def __add__(self, other):
        r = SimpleMesh()
        r.extend(self)
        r.extend(other)
        return r

    def translate(self, offset):
        new_verts = [V3(v.x + offset.x, v.y + offset.y, v.z + offset.z) for v in self.verts]
        return SimpleMesh(new_verts, self.faces)


point_table = []
patch_table = []


class VertexNode:
    table = point_table

    def __init__(self, data=None):
        super().__init__()

        self._data = None
        self.data = data
        self.prev = self
        self.next = self

    @property
    def data(self):
        return self.table[self._data]

    @data.setter
    def data(self, v):
        if isinstance(v, int):
            self._data = v
        else:
            if v not in self.table:
                self.table.append(v)

            self._data = self.table.index(v)

    def transform(self, m: TransformV2):
        ...


from scipy.spatial.distance import euclidean

EPS = 6


class ShapeDCLL(DCLL[VertexNode]):
    def isplanar(self):
        l = []
        for pt in list(self):
            l.append(round(euclidean(pt, self.plane.closest_point(pt)), EPS) == 0.0)
        return all(l)

    @property
    def plane(self):
        return PlaneLinear.from_tree_pt(self.head.data, self.head.next.data, self.head.next.next.data)


@dataclasses.dataclass
class OccShape:
    shape: typing.Any
    uuid: typing.Optional[str] = None
    color: tuple = (150, 150, 150)
    tess: typing.Optional[AMesh] = None

    def tessellate(self):
        self.tess = simple_tessellate(self.shape, uuid=self.uuid, color=self.color)
        return self.tess


@dataclasses.dataclass
class OccPrism(OccShape):
    prism: typing.Optional[typing.Any] = None

    def __post_init__(self):
        self.prism = self.shape

        self.shape = self.shape.Shape()


class Triangle:
    table = point_table
    width = None
    _occ = None

    def triangulate(self):
        pts = list(self.points)
        res = earcut.flatten([pts])
        tess = earcut.earcut(res['vertices'], res['holes'], res['dimensions'])

        self.mesh_data = MeshData(pts, indices=np.array(tess).reshape((len(tess) // 3, 3)).tolist())
        return self.mesh_data

    @classmethod
    def from_indices(cls, ixs):
        return cls(*[cls.table[pt] for pt in ixs])
    @property
    def uuid(self):
        return self._uuid
    @uuid.setter
    def uuid(self,v):
        self._uuid=v
        self._repr3d.uuid=self._uuid
    def __init__(self, *pts, uuid=None):
        object.__init__(self)
        self._uuid=uuid if uuid is not None else str(id(self))
        self.ixs = []
        for v in pts:
            if v not in point_table:
                point_table.append(v)

            self.ixs.append(self.table.index(v))

        self.points = ShapeDCLL.from_list(self.ixs)
        self.triangulate()



    def get_point(self, i):
        return self.table[i]

    def get_points(self):
        return [self.table[i] for i in self.ixs]

    def __getitem__(self, item):

        return self.get_point(self.ixs[item])

    @property
    def plane(self):
        return PlaneLinear.from_tree_pt(*self.get_points()[:3])

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

    def plane_split(self, plane, return_lists=False):
        sideA, sideB = self.divide_vertices_from_plane(plane)
        if (sideB == []) or (sideA == []):
            return [self.points[i] for i in sideA], [self.points[i] for i in sideB]

        la = ShapeDCLL.from_list([self.points[i] for i in sideA])
        lb = ShapeDCLL.from_list([self.points[i] for i in sideB])

        for i in sideA:
            for j in sideB:
                res = line_plane_intersection(plane, Linear.from_two_points(self.points[i], self.points[j])).tolist()
                lb.insert_end(res)
                la.insert_begin(res)
        if not return_lists:
            if len(la) > len(lb):
                sa, sb = Quad(*list(la)), Triangle(*list(lb))
            else:
                sa, sb = Triangle(*list(la)), Quad(*list(lb))
            return sa, sb
        return la, lb

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

    @property
    def normal(self):
        return self.plane.normal

    @property
    def centroid(self):
        return np.mean(np.array(self.points), axis=0).tolist()

    @property
    def v1(self):
        return self.points[0]

    @property
    def v2(self):
        return self.points[1]

    @property
    def v3(self):
        return self.points[2]

    @v1.setter
    def v1(self, v):
        self.points[0] = v

    @v2.setter
    def v2(self, v):
        self.points[1] = v

    @v3.setter
    def v3(self, v):
        self.points[2] = v

    def make_occ_prizm(self):
        if HAS_OCC:
            self._occ_prism = OccPrism(
                BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(self.make_occ_wire()).Face(),
                                      gp_Vec(*self.plane.normal * self.width)),
                uuid=f'{id(self)}')
            return self._occ_prism

    def make_occ_wire(self):
        if HAS_OCC:
            mkw = BRepBuilderAPI_MakeWire()
            self._occ_wire = mkw
            node = self.points.head
            for pt in range(len(self.points)):
                print(node.data, node.next.data)
                mkw.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(*node.data), gp_Pnt(*node.next.data)).Edge())
                node = node.next

            return mkw.Wire()

    def make_occ_face(self):
        if HAS_OCC:
            return BRepBuilderAPI_MakeFace(self.make_occ_wire()).Face()

    def make_occ_shape(self):
        if HAS_OCC:
            self._occ_shape = OccShape(BRepBuilderAPI_MakeFace(self.make_occ_face()).Shape())
            return self._occ_shape

    def __repr3d__(self):
        """
        if HAS_OCC:
            if USE_OCC:
                if self.width is not None:
                    print('0')
                    self._repr3d = self.make_occ_prizm().tessellate()
                else:
                    print('0.1')
                    self._repr3d = self.make_occ_shape().tessellate()
            else:
                print('1.1')
                self.triangulate()
                self._repr3d = self.mesh_data.to_mesh(uuid=f'{id(self)}')
        else:"""

        self.triangulate()
        self._repr3d = self.mesh_data.to_mesh(uuid=f'{id(self)}')
        return self._repr3d

    def root(self):
        return self._repr3d.root()

    def cut(self, cutter):

        return OccShape(
            BRepAlgoAPI_Cut(self.make_occ_prizm().prism.Shape(), cutter.make_occ_prizm().prism.Shape()).Shape())

    def intersection(self, cutter):

        return OccShape(
            BRepAlgoAPI_Common(self.make_occ_prizm().prism.Shape(), cutter.make_occ_prizm().prism.Shape()).Shape())
    @property
    def mesh(self):
        if hasattr(self,"_repr3d"):
            if isinstance(self._repr3d,AMesh):
                return self._repr3d
        else:
            return self.__repr3d__()
    def tomesh(self):
        return self.__repr3d__()

class Quad(Triangle):
    """A 3d quadrilateral (polygon with 4 vertices)"""

    table = point_table

    def __init__(self, /, *pts, **kwargs):
        super().__init__(*pts,**kwargs)

    def isplanar(self):
        return self.points.isplanar()

    def plane_split(self, plane, return_lists=False):
        a, b = super().plane_split(plane, return_lists=True)
        if return_lists:
            return Loop(*a), Loop(*b)
        else:
            return a, b

    @property
    def v4(self):
        return self.points[3]

    @v4.setter
    def v4(self, v):
        self.points[3] = v

    def _tri(self):
        return [Triangle(self.v1, self.v2, self.v3), Triangle(self.v2, self.v3, self.v4)]

    def map(self, f):
        return Quad(f(self.v1), f(self.v2), f(self.v3), f(self.v4))

    def swap(self, swap=True):
        if swap:
            return Quad(self.v4, self.v3, self.v2, self.v1)
        else:
            return Quad(self.v1, self.v2, self.v3, self.v4)


class Loop(Quad):
    @classmethod
    def from_loop(cls, poly):
        return cls.from_indices(poly.ixs)


class OccLikeLoop(Loop):
    _width = 28.0

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, v):
        self._width = v
        self.__repr3d__()
