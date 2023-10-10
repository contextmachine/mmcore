import typing
from dataclasses import dataclass

import shapely


@dataclass
class ShapeInterface:
    bounds: list[list[float]]
    holes: typing.Optional[list[list[list[float]]]] = None
    _poly = None

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
