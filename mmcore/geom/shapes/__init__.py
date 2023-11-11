import dataclasses
import typing
import uuid
from collections import namedtuple

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
from mmcore.geom import vectors
from mmcore.geom.materials import ColorRGB
from mmcore.geom.parametric import PlaneLinear, point_to_plane_distance, project_point_onto_plane
from mmcore.geom.parametric.algorithms import centroid
from mmcore.geom.shapes.shape import ShapeInterface, extrude_shape, tess_extrusion
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


def polygon_area_3d(polygon):
    polygon = np.array(polygon)
    a = polygon[-1]
    b = polygon[0]
    o = centroid(polygon)
    oa = a - o
    ob = b - o
    n0 = np.cross(oa, ob)
    area = 0.5 * vectors.norm(n0)
    for i in range(0, len(polygon) - 1):
        oa = ob
        b = polygon[i + 1]
        ob = b - o
        n = np.cross(oa, ob)
        if np.dot(n, n0) > 0:
            area += 0.5 * vectors.norm(n)
        else:
            area -= 0.5 * vectors.norm(n)
    return area


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


from mmcore.geom.shapes import base as shpb

__breps__ = dict()

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

    @classmethod
    def from_shape_interface(cls, obj: ShapeInterface):
        return cls(obj.bounds, holes=obj.holes)

    @classmethod
    def from_shape_interface_dict(cls, **kwargs):
        return cls(kwargs.get('bounds'), holes=kwargs.get('holes'))


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


def area_with_plane(bounds, plane):
    return shapely.Polygon([plane.in_plane_coords(pt).tolist() for pt in bounds]).area


def area3d(points):
    plane = PlaneLinear.from_tree_pt(points[1], points[0], points[2])
    return area_with_plane(points, plane)


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
        self.attributes = dict()
        self.indices = None
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
        self.attributes['position'] = list(chunked(self.arguments['vertices'], 3))
        self.indices = list(chunked(self.result, 3))
        self.mesh_data = MeshData(vertices=self.attributes['position'],
                                  indices=self.indices)
        # self.buffer = self.mesh_data.create_buffer()


class ShapeExtrusion:
    def __init__(self, shape, h):
        self.profile = shape
        self.h = h

    @property
    def shape1(self):
        return self.profile

    @property
    def shape2(self):
        return extrude_shape(self.profile, self.h)

    def tess(self):
        return tess_extrusion(self.shape1, self.shape2)

    @property
    def meshdata(self):
        pos, ixs = self.tess()

        return MeshData(vertices=pos, indices=ixs)

    def to_mesh(self, color=(100, 100, 100), **kwargs):
        col = ColorRGB(*color).decimal
        mesh = self.meshdata.to_mesh(color=col, **kwargs)
        mesh.__setattr__("cap1",
                         Shape(self.shape1.bounds, holes=self.shape1.holes).mesh_data.to_mesh(uuid=mesh.uuid + 'cap1',
                                                                                              name=mesh.name + 'cap1',
                                                                                              color=col))
        mesh.__setattr__("cap2", Shape(self.shape2.bounds,
                                       holes=self.shape2.holes).mesh_data.to_mesh(uuid=mesh.uuid + 'cap2',
                                                                                  name=mesh.name + 'cap2', color=col))
        return mesh


def polygon_area(poly, cast_to_numpy=True) -> float:
    """
    Реализация формула площади Гаусса с использованием numpy. Данная реализация превосходит по скорости (примерно в 10x)
    альтернативы из пакетов shapely и compas, однако в oтличае от shapely не поддерживает сложные полигоны. Бенчмарки приведены ниже.

    Если полигон имеет вид [p0,p1,p2,..,pn], передать необходимо [p0,p1,p2,..,pn,p0].
    Это приведение не делается автоматически, в том числе для того,
    чтобы избежать проверки на то является ли poly массивом numpy или списком, а также просто ради здравого смысла.
    Вы также можете избежать создания нового numpy массива если зададите cast_to_numpy=False. Это в целом благоприятно
    влияет на производительность, особенно на больших наборах.

    Parameters
    ----------
    poly :

    Returns
    -------
    float value of area

    Benchmarks
    -------
    >>> import shapely
    >>> import compas.geometry as cg
    >>> import time
    >>> def bench(N):
    ...     dat1=[(-180606.0, 23079.0, 0.0), (-181468.0, 59713.0, 0.0), (-173710.0, 59713.0, 0.0), (-173710.0, 49369.0, 0.0),
    ...      (-177589.0, 49585.0, 0.0), (-177589.0, 40965.0, 0.0), (-168753.0, 40965.0, 0.0), (-168969.0, 48076.0, 0.0),
    ...      (-164012.0, 60791.0, 0.0), (-151082.0, 61222.0, 0.0), (-156254.0, 17907.0, 0.0),(-180606.0, 23079.0, 0.0)]
    ...     dat = np.array(dat1+[dat1[0]])
    ...     s = time.time()
    ...     for i in range(N):
    ...         pgn = polygon_area(dat, cast_to_numpy=False)
    ...     print(f'[mmcore](without casting) {N} items at: ', divmod(time.time() - s, 60))
    ...     s = time.time()
    ...     for i in range(N):
    ...         pgn = polygon_area(dat)
    ...     print(f'[mmcore](with casting) {N} items at: ', divmod(time.time() - s, 60))
    ...
    ...     s=time.time()
    ...     for i in range(N):
    ...         pgn = shapely.area(shapely.Polygon(dat1))
    ...
    ...     print(f'[shapely] {N} items at: ',divmod(time.time()-s,60))
    ...     s = time.time()
    ...     for i in range(N):
    ...         pgn = cg.Polygon(
    ...             dat1)
    ...         pgn.area
    ...     print(f'[compas] {N} items at: ', divmod(time.time() - s, 60))
    ...
    >>> bench(10_000)
[mmcore](without casting) 10000 items at:  (0.0, 0.02608180046081543)
[mmcore](with casting) 10000 items at:  (0.0, 0.019220829010009766)
[shapely] 10000 items at:  (0.0, 0.1269359588623047)
[compas] 10000 items at:  (0.0, 0.19669032096862793)
    >>> bench(100_000)
[mmcore](without casting) 100000 items at:  (0.0, 0.16399192810058594)
[mmcore](with casting) 100000 items at:  (0.0, 0.1786212921142578)
[shapely] 100000 items at:  (0.0, 1.268747091293335)
[compas] 100000 items at:  (0.0, 1.969149112701416)
    >>> bench(1_000_000)
[mmcore](without casting) 1000000 items at:  (0.0, 1.5401110649108887)
[mmcore](with casting) 1000000 items at:  (0.0, 1.7677040100097656)
[shapely] 1000000 items at:  (0.0, 12.567844152450562)
[compas] 1000000 items at:  (0.0, 21.297150135040283)
    """

    if cast_to_numpy:
        poly = np.array(poly)
    length = poly.shape[0] - 1
    return np.abs(np.dot(poly[:length, 1], poly[1:, 0]) - np.dot(poly[:length, 0], poly[1:, 1])) / 2


polygon_area_vectorized = np.vectorize(polygon_area,
                                       doc="polygon_area обернутая в np.vectorize, что делает ее более подходящей для работы с массивами.\n\n" + polygon_area.__doc__,
                                       otypes=[float],
                                       excluded=['cast_to_numpy'],
                                       signature='(i,n)->()')


def shape_area(shp: ShapeInterface):
    result = polygon_area(np.array(shp.bounds + [shp.bounds[0]]), cast_to_numpy=False)
    if shp.holes:
        for hole in shp.holes:
            result -= polygon_area(np.array(hole + [hole[0]]), cast_to_numpy=False)
    return result
