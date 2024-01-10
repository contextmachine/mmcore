import functools
import typing
from collections import namedtuple
from dataclasses import dataclass

import more_itertools
import numpy as np
import shapely
from earcut import earcut
from shapely.geometry import mapping

from mmcore.base.geom import MeshData
from mmcore.geom import vectors
from mmcore.geom.mesh import MeshTuple, create_mesh_tuple
from mmcore.geom.parametric import PlaneLinear
from mmcore.geom.parametric import algorithms
from mmcore.geom.shapes.area import polygon_area

SplitResult = namedtuple("SplitResult", ['shapes', 'mask'])

from mmcore.base import points_traverse


@points_traverse
def split3d_vec(point, plane=None):
    if plane:
        return algorithms.ray_plane_intersection(np.array(point), np.array([0.0, 0.0, 1.0]), plane).tolist()
    else:
        return point


def points_to_shapes(points):
    if isinstance(points[0][0], float):
        return [ShapeInterface(points[0], holes=points[1:])]
    else:
        return [ShapeInterface(crd[0], holes=crd[1:]) for crd in points]


def poly_to_shapes(poly: typing.Union[shapely.Polygon, shapely.MultiPolygon]):
    crds = list(mapping(poly)['coordinates'])
    if isinstance(poly, shapely.MultiPolygon):
        return [ShapeInterface(crd[0], holes=crd[1:]) for crd in crds]
    else:
        return [ShapeInterface(crds[0], holes=crds[1:])]


def poly_to_points(poly: shapely.Polygon):
    return list(mapping(poly)['coordinates'])


def split2d(poly1, cont):
    if poly1.intersects(cont):

        if poly1.within(cont):
            # print("inside")

            return SplitResult(poly_to_shapes(poly1), 0)


        else:

            # print("intersects")
            poly3 = poly1.intersection(cont)

            return SplitResult(poly_to_shapes(poly3), 1)

    else:
        return SplitResult(poly_to_shapes(poly1), 2)


def extend(*clss):
    def wrapper(func):
        for cls in clss:
            setattr(cls, func.__name__, func)
        return func

    return wrapper


class HashList(list):
    def __hash__(self):
        return hash(self.__repr__())


@dataclass
class ShapeInterface:
    bounds: list[list[float]]
    holes: typing.Optional[list[list[list[float]]]] = None
    _poly = None
    split_result: typing.Optional[SplitResult] = None

    def to_poly(self):
        if self._poly is None:
            self._poly = shapely.Polygon(self.bounds, self.holes)
        return self._poly

    def to_world(self, plane=None):
        if plane is not None:
            bounds = [plane.in_plane_coords(pt) for pt in self.bounds]
            if self.holes:
                holes = [[plane.point_at(pt) for pt in hole] for hole in self.holes]
                return self.__class__(bounds=bounds, holes=holes)
            return self.__class__(bounds=bounds)

        return self

    def __iter__(self):
        if self.holes:
            return iter([self.bounds] + self.holes)
        else:
            return iter([self.bounds])

    def __getitem__(self, item):
        _l = len(item)
        if _l > 1:

            return self.get_by_tuple(item)
        else:
            if item == 0:
                return self.bounds
            else:
                return self.holes[item - 1]

    def get_by_pointer(self, item: 'ShapeVertexPointer'):
        return self.get_by_tuple(item)

    def get_by_tuple(self, item: tuple[int, int]):
        a, b = item
        return list(self)[a][b]

    def from_world(self, plane=None):
        if plane is not None:
            bounds = [plane.in_plane_coords(pt) for pt in self.bounds]
            holes = [[plane.in_plane_coords(pt) for pt in hole] for hole in self.holes]
            return ShapeInterface(bounds, holes=holes)
        return self

    def split(self, cont: 'Contour'):

        self.split_result = split2d(self.to_poly(), cont.poly)
        return self.split_result

    def __hash__(self):
        return hash(repr((self.bounds, self.holes)))

    def __eq__(self, other):
        return all([self.bounds == other.bounds, self.holes == other.holes])

    def astuple(self):
        return self.bounds, self.holes

    def area(self):
        return shape_area(self)

    def tessellate(self):
        return shape_earcut(self)

    def to_mesh(self) -> MeshTuple:
        res = self.tessellate()
        return create_mesh_tuple(attributes=dict(position=np.array(res.position).flatten()),
                                 indices=np.array(res.indices,
                                                  dtype=int).flatten())

def shape_area(shp: ShapeInterface):
    result = polygon_area(np.array(shp.bounds + [shp.bounds[0]]), cast_to_numpy=False)
    if shp.holes:
        for hole in shp.holes:
            result -= polygon_area(np.array(hole + [hole[0]]), cast_to_numpy=False)
    return result


@dataclass
class ContourShape(ShapeInterface):
    ...


def contour_from_dict(data: dict):
    plane = data.get('plane', None)
    if plane:
        plane = PlaneLinear(**plane)

    return Contour([ContourShape(**shp) for shp in data['shapes']], plane=plane)


@dataclass
class Contour:
    shapes: list[ContourShape]
    plane: typing.Optional[PlaneLinear] = None

    def __post_init__(self):

        self.shapes = [shape.to_world(self.plane) for shape in self.shapes]
        if len(self.shapes) == 1:
            self.poly = self.shapes[0].to_poly()
        else:
            self.poly = shapely.multipolygons(np.array(list(shape.to_poly() for shape in self.shapes)))

        # print(self.poly)

    def __eq__(self, other):

        return self.poly == other.poly

    @property
    def has_local_plane(self):
        return self.plane is not None


def move_shape(shp: ShapeInterface, xyz):
    if shp.holes is not None:
        return ShapeInterface((np.array(shp.bounds) + np.array(xyz)).tolist(),
                              holes=(np.array(shp.holes) + np.array(xyz)).tolist())
    else:
        return ShapeInterface((np.array(shp.bounds) + np.array(xyz)).tolist())


@functools.lru_cache(None)
def points_at_local(plane: PlaneLinear, points: HashList):
    return HashList(plane.point_at(point) for point in points)


@functools.lru_cache(None)
def points_at_global(plane: PlaneLinear, points: HashList):
    return HashList(plane.in_plane_coords(point) for point in points)


@functools.lru_cache(None)
def shape_at_local(plane: PlaneLinear, shape: ShapeInterface):
    return ShapeInterface([plane.point_at(point) for point in shape.bounds],
                          [[plane.point_at(point) for point in points] for points in shape.holes])


@functools.lru_cache(None)
def shape_at_global(plane: PlaneLinear, shape: ShapeInterface):
    if shape.holes:
        return ShapeInterface([plane.point_at(point) for point in shape.bounds],
                              [[plane.point_at(point) for point in points] for points in shape.holes])

    else:
        return ShapeInterface([plane.point_at(point) for point in shape.bounds])


_worldxy = PlaneLinear(origin=[0, 0, 0], xaxis=[1, 0, 0], yaxis=[0, 1, 0])

ShapeVertexPointer = namedtuple("ShapeVertexPointer", ['loop_index', 'point_index'])


def chamfer(shape: ShapeInterface, value: float, indices: list[ShapeVertexPointer]):
    ...


def chamfer_pts(prev_point, origin, next_point, value: float):
    pts = np.array([prev_point, origin, next_point])
    return np.array(
        [pts[0], pts[1] + value * vectors.unit(pts[0] - pts[1]), pts[1] + value * vectors.unit(pts[2] - pts[1]),
         pts[2]])





@functools.lru_cache(None)
def gen_tess(vln: int, start: int = 0):
    faces = []
    for i, j in zip(more_itertools.pairwise(range(vln)), more_itertools.pairwise(range(vln))):
        ll = list(i + j)

        for k in range(4):
            if ll[k] == vln:
                ll[k] = 0
        ui, vi, uj, vj = ll
        faces.extend(((start + ui, start + vj + vln, start + uj + vln), (start + vj + vln, start + ui, start + vi)))
    return faces


@functools.lru_cache(None)
def gen_tess2d(u: int, v: int, start: int = 0):
    faces = []
    for vv in range(v):
        faces.extend(gen_tess(u, start + vv))
    return faces


@functools.lru_cache(None)
def extrude_shape(shape1, h: float):
    return move_shape(shape1, [0, 0, h])


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

    def meshdata(self):
        pos, ixs = self.tess()

        return MeshData(vertices=pos, indices=ixs).merge2(ecut(self.shape1))


@functools.lru_cache(None)
def tess_extrusion(shape1: ShapeInterface, shape2: ShapeInterface):
    cnts = list(shape1)
    cnts2 = list(shape2)
    start = 0
    faces = []
    position = []

    for cnt, cnt1 in zip(cnts, cnts2):
        l = len(cnt) + 1
        faces.extend(gen_tess(l, start=start))
        c = np.array(cnt).tolist()
        c1 = np.array(cnt1).tolist()
        c.append(c[0])
        c1.append(c1[0])
        position.extend(c + c1)

        start += (l * 2)

    return position, faces


def ecut(self):
    if self.holes is None:
        arguments = earcut.flatten([self.boundary])
    else:
        arguments = earcut.flatten([self.boundary] + self.holes)

    return MeshData(vertices=arguments['vertices'],
                    indices=earcut.earcut(arguments['vertices'], arguments['holes'], arguments['dimensions']))


ShapeEarcutResult = namedtuple("ShapeEarcutResult", ['position', 'indices', 'arguments'])



def shape_earcut(self: ShapeInterface) -> ShapeEarcutResult:
    if self.holes is None:
        arguments = earcut.flatten([self.bounds])
        return ShapeEarcutResult(arguments['vertices'], earcut.earcut(arguments['vertices'],
                                                                      None,

                                                                      arguments['dimensions']), arguments)
    elif len(self.holes) == 0:
        arguments = earcut.flatten([self.bounds])
        return ShapeEarcutResult(arguments['vertices'], earcut.earcut(arguments['vertices'],
                                                                      None,
                                                                      arguments['dimensions']), arguments)
    else:
        arguments = earcut.flatten([self.bounds] + self.holes)
        return ShapeEarcutResult(arguments['vertices'], earcut.earcut(arguments['vertices'],
                                                                      arguments['holes'],
                                                                      arguments['dimensions']), arguments)

def bounds_holes_earcut(bounds, holes) -> ShapeEarcutResult:


        arguments = earcut.flatten([bounds] + holes)
        return ShapeEarcutResult(arguments['vertices'], earcut.earcut(arguments['vertices'],
                                                                      arguments['holes'],
                                                                      arguments['dimensions']), arguments)


def bounds_earcut(bounds) -> ShapeEarcutResult:
    arguments = earcut.flatten([bounds])
    return ShapeEarcutResult(arguments['vertices'], earcut.earcut(arguments['vertices'],
                                                                  arguments['holes'],
                                                                  arguments['dimensions']), arguments)


def loops_earcut(loops) -> ShapeEarcutResult:
    arguments = earcut.flatten(loops)
    return ShapeEarcutResult(arguments['vertices'],
                             earcut.earcut(arguments['vertices'], arguments['holes'], arguments['dimensions']),
                             arguments
                             )
