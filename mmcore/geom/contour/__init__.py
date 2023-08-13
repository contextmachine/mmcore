import abc
import math

import numpy as np

from mmcore.geom.parametric import CurveCurveIntersect, IntersectFail
from mmcore.geom.parametric import Linear, NurbsCurve, PlaneLinear


class AbstractLoop:
    edges_table = []

    def __init__(self, edges_table=None):
        if edges_table is None:
            edges_table = []
        self.edges_table = edges_table

    @abc.abstractmethod
    def segment(self, node, next_node):
        ...

    def get_segment(self, item: int):
        node = self.get_node(item)
        return self.segment(node, node.next)


class PolylineContour(AbstractContour):

    @abc.abstractmethod
    def segment(self, node, next_node):
        return Segment(NurbsCurve([node.data, next_node.data], degree=1))


def list_from_rhino(geom):
    return [geom.X, geom.Y, geom.Z]


def vector(plane):
    vec = plane.xaxis * (math.e * 10 ** 6)
    origin = plane.origin
    inters_crv = Linear(*origin, *vec)

    return inters_crv


class Segment:

    def __init__(self, crv, option='cut', name=None, offset=0, **kwargs):
        self.crv = crv
        self.option = option
        self.offset = offset

        self.name = name
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)

    def get_intersections(self, vector):
        ...


class MarkedVector:

    @property
    def start_point(self):
        return self._start_point

    @start_point.setter
    def start_point(self, a):
        self._start_point = a

    @property
    def start_type(self):
        return self._start_type

    @start_type.setter
    def start_type(self, a):
        self._start_type = a

    @property
    def end_point(self):
        return self._end_point

    @end_point.setter
    def end_point(self, a):
        self._end_point = a

    @property
    def end_type(self):
        return self._end_type

    @end_type.setter
    def end_type(self, a):
        self._end_type = a


class Contour:
    max_height = 30000
    triangle = 600

    def __init__(self, crv, seg, option, st_pl, name=None):

        self.crv = self.to_nurbs(crv)
        self.name = name

        # все отрезки кривой контура
        self.seg = [Segment(s, option=o, name=name) for s, o in zip(seg, option)]

        # начальный вектор
        self.start_plane = PlaneLinear(origin=list_from_rhino(st_pl.Origin),
                                       xaxis=list_from_rhino(st_pl.XAxis),
                                       yaxis=list_from_rhino(st_pl.YAxis))

    @property
    def start_height(self):
        return self.start_plane.origin[2]

    def shift_vectors(self):

        # h = self.start_height
        h = 0
        shift_vectors = []

        while True:

            if h <= self.max_height:

                origin = np.asarray(self.start_plane.origin) + np.array([0, 0, h])
                plane = PlaneLinear(origin=origin, xaxis=self.start_plane.xaxis, yaxis=self.start_plane.yaxis)

                vec = vector(plane)
                shift_vectors.append(vec)

                h += self.triangle

            else:
                break

        return shift_vectors

    @property
    def intersections(self):

        self._intersections = []
        shift_vectors = self.shift_vectors()

        for inters_vec in shift_vectors:

            points = []
            vec = MarkedVector()

            for i in self.seg:

                inters = CurveCurveIntersect(i.crv, inters_vec)
                inters = inters(tolerance=1)

                if isinstance(inters, IntersectFail):
                    pass

                else:

                    if len(points) == 0:
                        vec.start_point = inters.pt.tolist()
                        vec.start_type = i.option
                        points.append(inters)
                    elif len(points) == 1:
                        vec.end_point = inters.pt.tolist()
                        vec.end_type = i.option
                        points.append(inters)

            if len(points) == 0:
                self._intersections.append(points)
            else:
                self._intersections.append(vec)

        return self._intersections

    def to_nurbs(self, curve):

        points = [[i.X, i.Y, i.Z] for i in curve]
        line = NurbsCurve(points, degree=1)
        return line


if len(segment) != 4:
    opt = ["spec", "cut", "spec", "cut", "cut", "cut", "spec", "cut"]
else:
    opt = ["spec", "cut", "spec", "cut"]

print(len(segment))

cntr = Contour(curve, segment, opt, start_plane, name="W_1")

cc = cntr.intersections
# print(cc)

l = []
ll = []
for i in cc:
    try:
        p = rg.Line(rg.Point3d(*i.start_point), rg.Point3d(*i.end_point))
        t = [i.start_type, i.end_type]
        l.append(p)
        ll.append(t)
    except:
        pass

c = th.list_to_tree(l)
b = repr(ll)
