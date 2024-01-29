"""
PolyCurve Example:
    >>> # Creating PolyCurve with points
    >>> from mmcore.geom.polycurve import PolyCurve
    >>> pts=np.array([(-22.047791358681653, -0.8324885498102903, 0.0),
    ...                (-9.3805108456226147, -28.660718210796471, 0.0),
    ...                (22.846252925543098, -27.408177802862003, 0.0),
    ...                (15.166676249569946, 2.5182225098112045, 0.0)])
    >>> poly=PolyCurve(pts)
    >>> poly
    Out: PolyCurve[LineNode](length=4) at 00000000000
    >>> # Inserting a single corner
    >>> pt=np.array((24.707457249539218, -8.0614698399460814, 0.0))
    >>> poly.insert_corner(pt)
    >>> poly.corners
    Out:
    array([[-22.04779136,  -0.83248855,   0.        ],
           [ -9.38051085, -28.66071821,   0.        ],
           [ 22.84625293, -27.4081778 ,   0.        ],
           [ 24.70745725,  -8.06146984,   0.        ],
           [ 15.16667625,   2.51822251,   0.        ]])

    >>> # Inserting a multiple corners
    >>> for i in [(-8.7294742050094118, 9.9843044974317401, 0.0), (-31.500187542229803, -6.5544241369704661, 0.0),
    ...           (-21.792672908993747, -25.849607543773036, 0.0), (33.695960118022263, -10.149799927057904, 0.0),
    ...           (19.793840396350852, -36.875426633374516, 0.0), (7.3298709907144382, 9.6247669184230027, 0.0),
    ...           (26.625054397516983, -0.44228529382181825, 0.0), (-34.376488174299752, -18.778701823267753, 0.0),
    ...           (0.85819456855706555, 14.418601305206250, 0.0)]:
    ...      poly.insert_corner(i)
    >>> poly
    Out: PolyCurve[LineNode](length=14) at 0x16f3633d0
    >>>  poly.corners
    Out:
    array([[-22.04779136,  -0.83248855,   0.        ],
           [ -9.38051085, -28.66071821,   0.        ],
           [ 22.84625293, -27.4081778 ,   0.        ],
           [ 33.69596012, -10.14979993,   0.        ],
           [ 19.7938404 , -36.87542663,   0.        ],
           [ 24.70745725,  -8.06146984,   0.        ],
           [-21.79267291, -25.84960754,   0.        ],
           [-34.37648817, -18.77870182,   0.        ],
           [-31.50018754,  -6.55442414,   0.        ],
           [ -8.72947421,   9.9843045 ,   0.        ],
           [  0.85819457,  14.41860131,   0.        ],
           [  7.32987099,   9.62476692,   0.        ],
           [ 26.6250544 ,  -0.44228529,   0.        ],
           [ 15.16667625,   2.51822251,   0.        ]])

    >>> # Evaluate
    >>> poly(np.linspace(0,4,10))
    Out:
    array([[-22.04779136,  -0.83248855,   0.        ],
           [-16.41788891, -13.20059062,   0.        ],
           [-10.78798646, -25.56869269,   0.        ],
           [  1.36174374, -28.24320474,   0.        ],
           [ 15.68474987, -27.68652012,   0.        ],
           [ 12.92649163, -27.06182886,   0.        ],
           [ -6.91303096, -26.36913096,   0.        ],
           [-23.19087461, -25.06395135,   0.        ],
           [-28.78368139, -21.92132659,   0.        ],
           [-34.37648817, -18.77870182,   0.        ]])

    >>> points=poly.evaluate_node(np.linspace(0,4,10))
    PointsOnCurveCollection(length=10)
    >>> np.array(points)
Out:
array([[-22.04779136,  -0.83248855,   0.        ],
       [-16.41788891, -13.20059062,   0.        ],
       [-10.78798646, -25.56869269,   0.        ],
       [  1.36174374, -28.24320474,   0.        ],
       [ 15.68474987, -27.68652012,   0.        ],
       [ 25.25729897, -23.57298272,   0.        ],
       [ 30.07939105, -15.90259255,   0.        ],
       [ 32.15128015, -13.11931401,   0.        ],
       [ 25.97256027, -24.99737032,   0.        ],
       [ 19.7938404 , -36.87542663,   0.        ]])

>>> poly.set_corner(0,[-15.,  0.,   0.        ])
>>> np.array(pts) # strong linking
array([[-15.        ,   0.        ,   0.        ],
       [ -9.37009755, -12.36810207,   0.        ],
       [ -3.7401951 , -24.73620414,   0.        ],
       [  1.36174374, -28.24320474,   0.        ],
       [ 15.68474987, -27.68652012,   0.        ],
       [ 25.25729897, -23.57298272,   0.        ],
       [ 30.07939105, -15.90259255,   0.        ],
       [ 32.15128015, -13.11931401,   0.        ],
       [ 25.97256027, -24.99737032,   0.        ],
       [ 19.7938404 , -36.87542663,   0.        ]])


"""

import functools
import itertools

import numpy as np
from scipy.spatial import KDTree

from mmcore.func import vectorize
from mmcore.geom.extrusion import polyline_to_lines
from mmcore.geom.intersections.predicates import intersects_segments
from mmcore.geom.line import Line
from mmcore.geom.line.cdll_line import LineCDLL, LineNode
from mmcore.geom.polygon import Polygon
from mmcore.geom.rectangle import Rectangle


@functools.lru_cache(maxsize=None)
def _cached_mbr(self: 'PolyCurve'):
    corns = self.corners
    return Rectangle.from_mbr(corns, closest_origin=corns[0])


class PolyCurve(LineCDLL):
    """
    PolyCurve Class

    This class represents a circular doubly linked list (CDLL) of line segments that form a polyline.

    Attributes:
        nodetype (LineNode): The type of node used in the CDLL.

    Methods:
        __init__(self, pts=None): Initializes a PolyCurve object.
        from_points(cls, pts): Creates a new PolyCurve object from a list of points.
        solve_kd(self): Solves the KDTree for the corners of the polyline.
        insert_corner(self, value, index=None): Inserts a corner at a specified index or at the nearest index.
        corners(self): Returns an array of the corners of the polyline.
        corners(self, corners): Sets the corners of the polyline.
    

    """
    nodetype = LineNode

    def __init__(self, segments=None):

        super().__init__()

        self._kd = None
        if segments is not None:

            for i in segments:
                self.append(i)

    def __hash__(self):
        return hash(tuple(hash(i) for i in self.gen_intersects()))

    @property
    def mbr(self):
        return _cached_mbr(self)

    @classmethod
    def from_points(cls, pts):
        lcdll = cls()

        lines = polyline_to_lines(np.array(pts, float))
        for line in lines:

            lcdll.append(Line.from_ends(*line))

        return lcdll

    def insert_corner(self, value, index=None):
        """
        Inserts a corner into the KDTree.

        :param value: The value to be inserted as a corner.
        :type value: numpy array or list
        :param index: The index where the corner should be inserted. If not specified, the method calculates the
        appropriate index based on the KDTree.
        :type index: int or None
        :return: None
        :rtype: None
        """
        value = np.array(value, float)
        if index is None:
            kd = self.solve_kd()
            dists, indices = kd.query(value, 2)
            indices = np.sort(np.abs(indices))
            delta = indices[0] - indices[1]
            if delta == 1:
                index = indices[0]

            else:

                index = indices[-1]
        super().insert_corner(value, index)

    def remove_corner(self, index):
        node = self.get_node(index)
        node.previous.end = node.end
        self.remove_by_index(index)

    @property
    def corners(self):

        return np.array([list(i) for i in self.gen_intersects()])

    @corners.setter
    def corners(self, corners):
        for i, (corner, node) in enumerate(itertools.zip_longest(corners, self.iter_nodes(), fillvalue=None)):
            print(corner, node)
            if corner is None:
                for j in range(i, self.count):
                    print(j, self.count)
                    self.remove_corner(j)
                break

            elif node is None:

                self.insert_corner(corner)
            elif np.allclose(corner, node.start):
                pass
            else:
                node.start = np.array(corner)
                node.previous.end = np.array(corner)

    def outside_point(self):
        mbr = self.mbr
        corner = self.corners
        return mbr.corners[0] + mbr.corners[0] - np.average(corner, axis=0)

    @vectorize(excluded=[0], signature='(n)->()')
    def point_inside(self, pt):
        res = np.sum(
                np.array(intersects_segments(np.array([pt, self.outside_point()]), polyline_to_lines(self.corners)),
                         int
                         )
                )

        return bool(res % 2)

    @classmethod
    def from_polygon(cls, poly: Polygon):
        if np.allclose(poly.corners[0], poly.corners[-1]):
            return cls.from_points(poly.corners[:-1])
        else:
            return cls.from_points(poly.corners[:-1])


from mmcore.geom.shapes.shape import loops_earcut
from typing import Any


class PolyCurveShape:
    loops: list[PolyCurve]

    def __init__(self, loops=()):
        self.loops = []
        for loop in loops:
            if isinstance(loop, PolyCurveShape):
                self.loops.extend(loop.loops)
            else:
                self.loops.append(loop if isinstance(loop, PolyCurve) else PolyCurve.from_points(np.array(loop, float)))

    @property
    def holes(self):
        return self.loops[1:]

    @property
    def boundary(self):
        return self.loops[0]

    @boundary.setter
    def boundary(self, v):
        self.loops[0] = v

    @holes.setter
    def holes(self, v):
        self.loops[1:] = v

    def add_hole(self, v):
        self.loops.append(v if isinstance(v, PolyCurve) else PolyCurve.from_points(np.array(v, float)))

    def earcut(self):
        return loops_earcut([loop.corners.tolist() for loop in self.loops])

    def __hash__(self):
        return hash(tuple(hash(loop) for loop in self.loops))

    def set_corners(self, i, corners: np.ndarray[Any, np.dtype[float]]):
        self.loops[i].corners = corners

    def __iter__(self):
        return iter(self.loops)
