import typing
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import mapping

from mmcore.geom.parametric import PlaneLinear
from mmcore.geom.parametric import algorithms

SplitResult = namedtuple("SplitResult", ['shapes', 'mask'])

from mmcore.base import points_traverse


@points_traverse
def split3d_vec(point, plane=None):
    if plane:
        return algorithms.ray_plane_intersection(np.array(point), np.array([0.0, 0.0, 1.0]), plane).tolist()
    else:
        return point


def poly_to_shapes(poly: typing.Union[shapely.Polygon, shapely.MultiPolygon]):
    crds = list(mapping(poly)['coordinates'])
    if isinstance(poly, shapely.MultiPolygon):
        return [ShapeInterface(crd[0], holes=crd[1:]) for crd in crds]
    else:
        return [ShapeInterface(crds[0], holes=crds[1:])]


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
            bounds = [plane.point_at(pt) for pt in self.bounds]
            holes = [[plane.point_at(pt) for pt in hole] for hole in self.holes]
            return ShapeInterface(bounds, holes=holes)

        return self

    def from_world(self, plane=None):
        if plane is not None:
            bounds = [plane.in_plane_coords(pt) for pt in self.bounds]
            holes = [[plane.in_plane_coords(pt) for pt in hole] for hole in self.holes]
            return ShapeInterface(bounds, holes=holes)
        return self

    def split(self, cont: 'Contour'):

        self.split_result = split2d(self.to_poly(), cont.poly)
        return self.split_result


@dataclass
class ContourShape(ShapeInterface):

    def to_world(self, plane=None):
        if plane is not None:
            bounds = [plane.in_plane_coords(pt) for pt in self.bounds]
            holes = [[plane.in_plane_coords(pt) for pt in hole] for hole in self.holes]
            return ContourShape(bounds, holes=holes)
        return self


@dataclass
class Contour:
    shapes: list[ContourShape]
    plane: typing.Optional[PlaneLinear] = None

    def __post_init__(self):

        self.shapes = [shape.to_world(self.plane) for shape in self.shapes]
        if len(self.shapes) == 1:
            self.poly = self.shapes[0].to_poly()
        else:
            self.poly = shapely.multipolygons(shape.to_poly() for shape in self.shapes)

        # print(self.poly)

    def __eq__(self, other):

        return self.poly == other.poly

    @property
    def has_local_plane(self):
        return self.plane is not None
