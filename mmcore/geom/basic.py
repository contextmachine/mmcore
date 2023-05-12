import json
from operator import add, sub

import strawberry

from mmcore.base.geom import GeometryObject, LineObject, PointsObject
import numpy as np
import operator

from functools import total_ordering, singledispatch


class Point(PointsObject):
    @property
    def origin(self):
        return self.points[0]

    @origin.setter
    def origin(self, v):
        if hasattr(v, "tolist"):
            v = v.tolist()
        self._points[0] = list(v)

    @property
    def xyz(self):
        """
        Point.origin alias
        @return:
        """
        return self.origin()

    @xyz.setter
    def xyz(self, v):
        """
        Point.origin alias
        @return:
        """
        self.origin = v

    def __array__(self):
        return np.array(self.origin, dtype=float)

    @classmethod
    def from_rhino(cls, rhino_point):
        return cls(points=[[rhino_point.X, rhino_point.Y, rhino_point.Z]])

    @classmethod
    def from_tuple(cls, tuple_point: tuple[float, float, float]):
        return cls(points=[list(tuple_point)])

    def __add__(self, other):
        return self.__class__(points=[list(map(lambda x: add(*x), zip(self.origin, other.origin)))])

    def __sub__(self, other):
        return self.__class__(points=[list(map(lambda x: sub(*x), zip(self.origin, other.origin)))])

    def __mul__(self, other):
        return self.__class__(points=[list(map(lambda x: x * other, self.origin))])

    def __truediv__(self, other):
        return self.__class__(points=[list(map(lambda x: x / other, self.origin))])

    def __iter__(self):
        return iter(self.origin)

    def __len__(self):
        return len(self.origin)

    def __getitem__(self, item: int):
        return self.origin[item]

    def __setitem__(self, item: int, v: float):
        self.points[0][item] = v

    @property
    def x(self):
        return self.points[0][0]

    @property
    def y(self):
        return self.points[0][1]

    @property
    def z(self):
        return self.points[0][2]

    @x.setter
    def x(self, v):
        self.points[0][0] = v

    @y.setter
    def y(self, v):
        self.points[0][1] = v

    @z.setter
    def z(self, v):
        self.points[0][2] = v

    def ToJSON(self):
        return json.dumps(strawberry.asdict(self()))

    @property
    def properties(self):
        dct = super(GeometryObject, self).properties
        dct.update(dict(
            x=self.x,
            y=self.y,
            z=self.z,
            color=self.color.to_dict()

        ))
        return dct


class NamedPoint(Point):
    @property
    def properties(self):
        dct = super(NamedPoint, self).properties
        dct.update(dict(
            name=self.name

        ))
        return dct


class PointArray(PointsObject):
    def __getitem__(self, item):
        return self.points[item]

    def __setitem__(self, item, v):
        self.points[item] = v

    def __len__(self):
        return len(self.points)
