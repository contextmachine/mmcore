import numpy as np
import shapely

from mmcore.base import ALine
from mmcore.collections import DCLL

from mmcore.geom.utils import area_2d, triangle_unit_normal
from mmcore.geom.parametric.sketch import Linear, Polyline


def from_list(seq):
    lst = DCLL()
    for s in seq:
        lst.append(s)

    return lst


class Polygon:
    _holes = []
    _points = DCLL()
    def __init__(self,**kw):
        super().__init__()
        self.__call__(**kw)
    def __call__(self,  **kwargs):
        for k,v in kwargs.items():
            setattr(self, k,v)
        return self
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
        self._holes=v


    @property
    def poly_shapely(self):

        return shapely.Polygon(list(self.points), [list(l) for l in self.holes])

    @property
    def area(self):

        return area_2d(self.points)
    def append(self, pt):
        self._points.append(pt)
    def append_hole(self, hole):
        self._holes.append(hole)
    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, v):
        self._points=DCLL()
        for i in v:
            self._points.append(i)

    @property
    def centroid(self):

        return np.asarray(self.poly_shapely.centroid.xy).flatten()

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


def check(crv, mask):
    return mask.contains(crv.poly_shapely), mask.intersects(crv.poly_shapely), mask.within(crv.poly_shapely)

