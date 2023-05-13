import numpy as np
import shapely
from mmcore.collections import DCLL
from mmcore.base.geom import LineObject, LineBasicMaterial, GqlLine
from mmcore.geom.utils import area_2d, triangle_unit_normal
from mmcore.geom.parametric.sketch import Linear, Polyline


def from_list(seq):
    lst = DCLL()
    for s in seq:
        lst.append(s)

    return lst


class Polygon(LineObject):
    _holes = DCLL()
    _points = DCLL()

    @classmethod
    def from_shapely(cls, poly):
        return cls(points=np.array(poly.boundary.coords.xy).T)

    @property
    def is_holes_containing(self):
        return (self._holes is None) or (self._holes == [])

    @property
    def holes(self):

        return self._holes

    @property
    def to_parametric_polyline(self):
        return Polyline.from_points(self.points)

    @holes.setter
    def holes(self, v):
        if v is not None:
            if not (v == []):
                self._holes = DCLL()
                for hole in v:
                    self._holes.append(Polygon(points=hole))
                    self.solve_geometry()

    @property
    def points(self):
        return list(self._points)

    @points.setter
    def points(self, v):
        if v is not None:
            if not (v == []):
                self._points = from_list(list(v))

                if not (list(v)[0] == list(v)[-1]):
                    self._points.append(list(v)[0])

        self.solve_geometry()

    @property
    def poly_shapely(self):

        return shapely.Polygon(shell=self.points)

    @property
    def area(self):

        return area_2d(self.points)

    @property
    def centroid(self):

        return self.poly_shapely.centroid

    def __add__(self, other):

        return self.from_shapely(shapely.union(self.poly_shapely, other.poly_shapely))

    def __sub__(self, other):

        return self.from_shapely(shapely.difference(self.poly_shapely, other.poly_shapely))

    def __matmul__(self, other):

        return self.from_shapely(shapely.intersection(self.poly_shapely, other.poly_shapely))

    def crosses(self, other):

        return shapely.crosses(self.poly_shapely, other.poly_shapely)

    def within(self, other):

        return shapely.within(self.poly_shapely, other.poly_shapely)

    def overlaps(self, other):

        return shapely.overlaps(self.poly_shapely, other.poly_shapely)

    def intersects(self, other):

        return shapely.intersects(self.poly_shapely, other.poly_shapely)

    def disjoint(self, other):

        return shapely.disjoint(self.poly_shapely, other.poly_shapely)

    def contains(self, other):

        return shapely.contains(self.poly_shapely, other.poly_shapely)

    def __contains__(self, item):
        return self.contains(other=item)
    @property
    def normal(self):
        a= np.asarray(self.points[1]) - np.asarray(self.points[0])
        b=np.asarray(self.points[2]) - np.asarray(self.points[0])
        return np.cross(a,b)