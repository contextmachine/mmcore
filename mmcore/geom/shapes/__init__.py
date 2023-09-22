import dataclasses
import typing
import uuid
from collections import namedtuple

import earcut
import more_itertools
import numpy as np
import shapely
from earcut import earcut
from more_itertools import flatten
from shapely import MultiPolygon, Polygon

from mmcore.base import AMesh
from mmcore.base.delegate import class_bind_delegate_method, delegate_method
from mmcore.base.geom import MeshData
from mmcore.base.models.gql import MeshPhongMaterial
from mmcore.collections import DCLL
from mmcore.geom.materials import ColorRGB
from mmcore.geom.parametric import PlaneLinear, point_to_plane_distance, project_point_onto_plane
from mmcore.geom.transform import Transform, WorldXY


def to_list_req(obj):
    if not isinstance(obj, str):
        if hasattr(obj, "coords"):
            return to_list_req(obj.coords)
        else:
            try:

                return [to_list_req(o) for o in list(obj)]
            except Exception:
                return obj
    else:
        return obj


@class_bind_delegate_method
def bind_poly_to_shape(self, other, delegate=None):
    return self.__class__(boundary=list(delegate.boundary.coords), holes=list(delegate.interiors), color=self.color,
                          h=self.h)


@delegate_method
def delegate_shape_operator(self, item, m):
    if isinstance(item, self._ref.__class__):
        return item
    elif isinstance(item, self.__class__):
        return item._ref
    elif isinstance(item, (np.ndarray, list, tuple)):
        return self._ref.__class__(item)
    elif hasattr(item, m.__name__):
        return item
    else:
        raise ValueError(f"{m.__name__.capitalize()} operation unknown in {item.__class__} objects")


# @Delegate(delegate=Polygon)
@dataclasses.dataclass
class LegacyShape:
    boundary: list[list[float, float, float]]
    holes: typing.Optional[list[list[list[float, float, float]]]] = None
    color: ColorRGB = ColorRGB(150, 150, 150).decimal
    uuid: typing.Optional[str] = None
    h: typing.Any = None

    def __post_init__(self):
        if len(self.boundary[0]) == 3:
            self.boundary = (np.array(self.boundary)[..., :2]).tolist()

        if not self.uuid:
            self.uuid = uuid.uuid4().hex
        if self.h is None:
            self.h = 0
        if self.holes is None:

            self.holes = []
        else:
            if len(self.holes[0][0]) == 3:
                self.holes = (np.array(self.holes)[..., :2]).tolist()

        self._ref = Polygon(shell=self.boundary, holes=self.holes)

    @property
    def mesh(self):
        return AMesh(uuid=self.uuid + "-mesh",
                     geometry=self.mesh_data.create_buffer(),
                     material=MeshPhongMaterial(color=self.color),
                     name="Shape Mesh")

    def earcut_poly(self):

        data = earcut.flatten([self.boundary] + self.holes)
        res = earcut.earcut(data['vertices'], data['holes'], data['dimensions'])
        return np.array(res).reshape((len(res) // 3, 3))

    def to3d_mesh_pts(self):
        # print(self.boundary)
        rrr = np.array(list(flatten([self.boundary] + self.holes)))
        return np.c_[rrr, np.ones((rrr.shape[0], 1)) * self.h]

    def to3d_mesh_holes(self):
        l = []
        for hole in self.holes:
            rrr = np.array(hole)
            l.append(np.c_[rrr, np.zeros((rrr.shape[0], 1))].tolist())
        return l

    def to3d_mesh_bnd(self):
        rrr = np.array(self.boundary)
        return np.c_[rrr, np.zeros((rrr.shape[0], 1))].tolist()

    @property
    def mesh_data(self):
        _mesh_data = MeshData(self.to3d_mesh_pts(), indices=self.earcut_poly())
        # _mesh_data.calc_normals()
        return _mesh_data

    @delegate_shape_operator.bind
    def __contains__(self, delegate, item):
        return shapely.contains(self, delegate, item)

    def contains(self, other):
        """
        __contains__ alias
        @param other:
        @return:
        """
        return self.__contains__(other)

    @property
    def exterior(self):
        return list(self._ref.exterior.coords)

    def offset(self, distance,

               cap_style='flat',
               join_style='mitre',
               mitre_limit=1000,
               single_sided=True,
               inplace=False,
               **kwargs):

        """
        @param distance,
        @param quad_segs = 16,
        @param cap_style = "round",
        @param join_style = "round",
        @param mitre_limit = 5.0,
        @param single_sided = False
        @return: Shape
        """
        res = self._ref.buffer(distance, cap_style=cap_style,
                               join_style=join_style,
                               mitre_limit=mitre_limit,
                               single_sided=single_sided, **kwargs)

        bounds, holes = list(res.exterior.coords), [list(i.coords) for i in list(res.interiors)]
        if inplace:
            self.boundary = bounds
            self.holes = holes
            return self
        else:
            return LegacyShape(bounds, holes, color=self.color, h=self.h)

    @property
    def interior(self):
        return to_list_req(self._ref.interiors)

    @delegate_shape_operator.bind
    def within(self, delegate, item):
        return shapely.within(delegate, item)

    @delegate_shape_operator.bind
    def intersects(self, delegate, item):
        return shapely.intersects(delegate, item)

    @delegate_shape_operator.bind
    def contains_properly(self, delegate, item):
        return shapely.contains_properly(delegate, item)

    def evaluate(self, t):
        return np.asarray(self._ref.interpolate(t, normalized=True), dtype=float)

    def evaluate_distance(self, d):
        return np.asarray(self._ref.interpolate(d, normalized=False), dtype=float)

    def __add__(self, item):
        return self.union(item)

    @delegate_shape_operator.bind
    def __sub__(self, delegate, item):
        # print(delegate, item)
        res = shapely.difference(delegate, item)
        if isinstance(res, MultiPolygon):
            shapes = []
            for i in res.geoms:
                shapes.append(LegacyShape(boundary=list(i.exterior.coords),
                                          holes=to_list_req(i.interiors),
                                          color=self.color,
                                          h=self.h))
            return shapes
        else:
            return LegacyShape(boundary=list(res.exterior.coords),
                               holes=to_list_req(res.interiors),
                               color=self.color,
                               h=self.h)

    def __isub__(self, item):
        res = self.difference(item)
        self.boundary = list(res.exterior.coords)
        self.holes = to_list_req(res.interiors)

    @bind_poly_to_shape
    @delegate_shape_operator.bind
    def difference(self, delegate, item):
        return shapely.difference(delegate, item)

    @bind_poly_to_shape
    @delegate_shape_operator.bind
    def union(self, delegate, other):
        return shapely.union(delegate, other)

    def __iadd__(self, item):
        res = self.union(item)

        self.boundary = list(res.exterior.coords)
        self.holes = to_list_req(res.interiors)

    def intersection(self, other):
        res = shapely.intersection(self._ref, other._ref)
        if isinstance(res, MultiPolygon):
            shapes = []
            for i in res.geoms:
                shapes.append(LegacyShape(boundary=list(i.exterior.coords),
                                          holes=to_list_req(i.interiors),
                                          color=self.color,
                                          h=self.h))
            return shapes
        else:
            return LegacyShape(boundary=list(res.exterior.coords),
                               holes=to_list_req(res.interiors),
                               color=self.color,
                               h=self.h)

    def is_empty(self):
        return self.boundary == []


TransformContext = namedtuple("TransformContext", ["obj", "to_world", "from_world"])


class TrxMng:
    def __init__(self, obj):
        self._obj = obj

    def __enter__(self):
        self.to_world = self._obj.plane.transform_to_other(WorldXY)
        self.from_world = self._obj.plane.transform_from_other(WorldXY)
        self._obj.transform(self.to_world)
        return TransformContext(self._obj, self.to_world, self.from_world)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._obj.transform(self.from_world)


@dataclasses.dataclass
class Boundary:
    bnds: dataclasses.InitVar[typing.Optional[list[tuple[float, float, float]]]]
    hls: dataclasses.InitVar[typing.Optional[list[tuple[float, float, float]]]] = None
    matrix: Transform = Transform()
    uuid: typing.Optional[str] = None

    color = (0, 0, 0)

    def __post_init__(self, bnds, hls=None):
        self._plane = PlaneLinear(origin=np.array(WorldXY.origin), xaxis=np.array(WorldXY.xaxis),
                                  yaxis=np.array(WorldXY.yaxis))
        if self.uuid is None:
            self.uuid = uuid.uuid4().hex
        self._holes = [] if hls is None else hls
        self._boundary = bnds

    @property
    def boundary(self):
        return np.array(self._boundary @ self.matrix).tolist()

    @property
    def holes(self):
        l = []
        for hole in self._holes:
            l.append(np.array(self._holes @ self.matrix).tolist())
        return l

    def transform(self, m):
        self.matrix = self.matrix @ m

    def to_shape(self):
        if self._holes is not None:
            holes = (np.array(self._holes)[..., :2]).tolist()
            return LegacyShape((np.array(self._boundary)[..., :2]).tolist(), holes=holes)
        return LegacyShape((np.array(self._boundary)[..., :2]).tolist())

    def to_mesh(self, *args, **kwargs):

        msh = self.to_shape().mesh_data.to_mesh(uuid=self.uuid, *args, **kwargs)
        msh @ self.plane.transform_from_other(WorldXY)
        # msh.wires = ALine(uuid=self.uuid + "-wire", geometry=self.boundary, material=LineBasicMaterial(color=ColorRGB(*self.color).decimal))
        return msh

    def transform_as_new(self, t):
        obj = Boundary(bnds=self.boundary, hls=self.holes)
        obj.transform(t)
        return obj

    @property
    def plane(self):
        return PlaneLinear(self._plane.origin.tolist() @ self.matrix,
                           normal=self._plane.normal.tolist() @ self.matrix,
                           xaxis=self._plane.xaxis.tolist() @ self.matrix
                           )

    def __copy__(self):
        return Boundary(self._boundary, self._holes, matrix=self.matrix)


def earcut_poly(boundary, holes):
    data = earcut.flatten([boundary] + holes)
    res = earcut.earcut(data['vertices'], data['holes'], data['dimensions'])
    return np.array(res).reshape((len(res) // 3, 3))


import uuid as _uuid
from mmcore.geom.shapes import base as shpb

__breps__ = dict()


class Brep:
    vertices = []
    edges = []
    edge_nodes = []
    loops = []
    faces = []

    def __new__(cls, uuid=None):
        self = super().__new__(cls)
        if uuid is None:
            uuid = _uuid.uuid4().hex
        __breps__[uuid] = self
        self.uuid = uuid
        return self

    def add_points(self, pts):
        ixs = []
        for pt in pts:
            if pt not in self.vertices:
                self.vertices.append(pt)

            ixs.append(self.vertices.index(pt))

        return ixs

    def tess(self):
        ixs = []
        for face in self.faces:
            ixs.extend(more_itertools.chunked(face.tess(), 3))
        return MeshData(vertices=self.vertices, indices=ixs)

    def add_loop_from_pts(self, ptss):
        ixs = []
        for pts in more_itertools.pairwise(ptss):
            edge = Edge(list(pts), brep=self)
            ixs.append(edge._i)
        lp = Loop(edges=ixs, brep=self)
        return lp._i

    def add_loop_from_edges(self, edges):

        lp = Loop(edges=edges, brep=self)
        return lp._i

    def add_face(self, boundary, holes=[]):
        f = Face(boundary, holes, brep=self)
        return f._i


class BrepComponent:
    __table_name__ = None

    def __init__(self, brep=None):
        super().__init__()

        self._brep = brep.uuid if isinstance(brep, Brep) else brep
        self._i = len(getattr(self.brep, self.__table_name__))
        getattr(self.brep, self.__table_name__).append(self)

    @property
    def brep(self):
        return __breps__[self._brep]


class Edge(BrepComponent):
    verts: list[int]
    __table_name__ = "edges"

    def __init__(self, verts, brep):
        super().__init__(brep=brep)
        self.verts = self.brep.add_points(verts)

    def points(self):
        return [self.brep.vertices[i] for i in self.verts]


from more_itertools import chunked


def earcut_poly2(boundary, holes):
    data = earcut.flatten([boundary] + holes)
    res = earcut.earcut(data['vertices'], data['holes'], data['dimensions'])
    return data, res


class Shape:
    boundary: list[list[float]]
    holes: typing.Optional[list[list[list[float]]]] = None
    earcut = None

    def __init__(self, boundary, holes=None):
        super().__init__()
        self.cache = dict()
        self.boundary = boundary
        self.holes = holes

        self.tessellate()

    def shapely_polygon(self):
        hs = hash(repr((self.boundary, self.holes)))
        if hs not in self.cache.keys():
            self.cache[hs] = shapely.Polygon(shell=self.boundary, holes=self.holes)
        return self.cache[hs]

    def tessellate(self):
        self.earcut = Earcut(boundary=self.boundary, holes=self.holes, solve=True)

    @property
    def mesh_data(self):
        return self.earcut.mesh_data

    @property
    def buffer(self):
        return self.earcut.buffer


def offset_with_plane(bounds, plane, distance=1.0):
    styles = dict(join_style=shapely.BufferJoinStyle.mitre)
    pts = []
    for pt in list(shapely.Polygon([plane.in_plane_coords(pt).tolist() for pt in bounds]).buffer(distance=distance,
                                                                                                 **styles).boundary.coords):
        if len(pt) == 2:
            point = plane.point_at(list(pt + (0,))).tolist()
        else:
            point = plane.point_at(list(pt)).tolist()
        if point not in pts:
            pts.append(point)
    return pts


def plane_dist(plane, point):
    project_point_onto_plane(np.array(point), np.array(plane.origin), point)


def is_coplanar(points, eps=1e-6):
    if len(points) <= 3:
        return True, PlaneLinear.from_tree_pt(points[0], points[-1], points[1])

    ll = DCLL.from_list(points)
    node = ll.head
    plane = PlaneLinear.from_tree_pt(node.data, node.prev.data, node.next.data)
    res = True
    node = node.next
    for i in range(len(points) - 2):
        node = node.next
        d = point_to_plane_distance(np.array(node.data), np.array(plane.origin), np.array(plane.normal))
        if abs(d) > eps:
            res = False
            break
        else:
            continue

    return res, plane


def offset(bounds, distance=1.0):
    styles = dict(join_style=shapely.BufferJoinStyle.mitre)
    if all(pt[2] == 0.0 for pt in bounds):
        print("on 0")
        pts = list(shapely.Polygon(bounds).buffer(distance, **styles).boundary.coords)

        return [list(_pt + (0,)) for _pt in pts]
    else:
        res, plane = is_coplanar(bounds)
        if res:

            return offset_with_plane(bounds, plane, distance=distance)
        else:

            ll = DCLL.from_list(bounds)
            node = ll.head.prev
            pts = []
            for i in range(len(bounds)):
                node = node.next
                plane = PlaneLinear.from_tree_pt(node.data, node.prev.data, node.next.data)
                pts.append(offset_with_plane(bounds, plane, distance=distance)[i])

            return pts

class Earcut:
    arguments = None
    result = None
    mesh_data = None
    buffer = None

    def __init__(self, boundary, holes=None, solve=True):
        super().__init__()
        self.boundary = boundary
        if not holes:
            holes = None
        self.holes = holes
        if solve:
            self.solve()

    def solve(self):
        if self.holes is None:
            self.arguments = earcut.flatten([self.boundary])
        else:
            self.arguments = earcut.flatten([self.boundary] + self.holes)
        self.result = earcut.earcut(self.arguments['vertices'], self.arguments['holes'], self.arguments['dimensions'])
        self.mesh_data = MeshData(vertices=list(chunked(self.arguments['vertices'], 3)),
                                  indices=list(chunked(self.result, 3)))
        self.buffer = self.mesh_data.create_buffer()


class Loop(BrepComponent):
    brep: Brep
    __table_name__ = "loops"

    def __init__(self, edges, brep=None):
        super().__init__(brep=brep)
        self._edges = edges

    def edges(self):
        return [self.brep.edges[ed] for ed in self._edges]

    def vertices(self):
        pts = []
        for edge in self.edges():

            for pt in edge.verts:
                if pt not in pts:
                    pts.append(pt)
        return pts

    def points(self):
        return [self.brep.vertices[i] for i in self.vertices()]

    def shp(self):
        return shpb.Loop(self.points())


class Face(BrepComponent):
    _boundary: int
    _holes: list[int]
    brep: Brep
    __table_name__ = "faces"

    def __init__(self, boundary, holes, brep):
        super().__init__(brep=brep)
        self._holes = holes
        self._boundary = boundary

    def loops(self):
        return [self._boundary] + self._holes

    def vertices(self):
        vvs = []
        [vvs.extend(self.brep.loops[loop].vertices()) for loop in self.loops()]
        return vvs

    def edges(self):
        ee = []
        for l in self.loops():
            ee.extend(self.brep.loops[l]._edges)
        return ee

    def tess(self):
        res, faces = earcut_poly2(self.boundary.points(), [h.points() for h in self.holes])
        return (np.array(self.vertices())[faces]).tolist()

    @property
    def boundary(self):
        return self.brep.loops[self._boundary]

    @property
    def holes(self):
        return [self.brep.loops[hole] for hole in self._holes]
