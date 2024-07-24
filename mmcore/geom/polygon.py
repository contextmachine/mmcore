from typing import Any
from mmcore.numeric.aabb import aabb, aabb_overlap, point_in_aabb
from mmcore.geom.vec import *
import numpy as np
from earcut import earcut
from mmcore.numeric.vectors import scalar_dot


def ecut(self) -> tuple:
    if self.holes is None:
        arguments = earcut.flatten([self.boundary])
    else:
        arguments = earcut.flatten([self.boundary] + self.holes)

    return arguments['vertices'], earcut.earcut(arguments['vertices'], arguments['holes'],
                                                arguments['dimensions']), arguments


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
    return np.abs(scalar_dot(poly[:length, 1], poly[1:, 0]) - scalar_dot(poly[:length, 0], poly[1:, 1])) / 2


polygon_area_vectorized = np.vectorize(polygon_area,
                                       doc="polygon_area обернутая в np.vectorize, что делает ее более подходящей для работы с массивами.\n\n" + polygon_area.__doc__,
                                       otypes=[float],
                                       excluded=['cast_to_numpy'],
                                       signature='(i,n)->()')


def to_polygon_area(boundary: np.ndarray):
    """
    Добавляет первую точку контура в конец чтобы соответствовать требованиям polygon_area
    :param boundary:
    :type boundary:
    :return:
    :rtype:
    """
    return np.array([*boundary, boundary[0]], dtype=boundary.dtype)


def clip(subjectPolygon, clipPolygon):
    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]
    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]
        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
    return (outputList)


class Vertex:
    """Node in a circular doubly linked list.

    This class is almost exactly as described in the paper by Günther/Greiner.
    """

    def __init__(self, vertex, alpha=0.0, intersect=False, entry=True, checked=False):
        if isinstance(vertex, Vertex):
            vertex = (vertex.x, vertex.y, vertex.z)  # checked = True
        self._array = np.zeros(3, float)
        self._array[:len(vertex)] = vertex

        # point coordinates of the vertex
        self.next = None  # reference to the next vertex of the polygon
        self.previous = None  # reference to the previousious vertex of the polygon
        self.neighbour = None  # reference to the corresponding intersection vertex in the other polygon
        self.entry = entry  # True if intersection is an entry point, False if exit
        self.alpha = alpha  # intersection point's relative distance from previousious vertex
        self.intersect = intersect  # True if vertex is an intersection
        self.checked = checked  # True if the vertex has been checked (last phase)

    @property
    def x(self):
        return self._array[0]

    @property
    def y(self):
        return self._array[1]

    @property
    def z(self):
        return self._array[2]

    @x.setter
    def x(self, v):
        self._array[0] = v

    @y.setter
    def y(self, v):
        self._array[1] = v

    @z.setter
    def z(self, v):
        self._array[2] = v

    def __array__(self, dtype=float):
        return np.array(self._array, dtype=dtype)

    def __copy__(self):
        return Vertex(self._array)

    def isInside(self, poly):
        """Test if a vertex lies inside a polygon (odd-even rule).

        This function calculates the "winding" number for a point, which
        represents the number of times a ray emitted from the point to
        infinity intersects any edge of the polygon.

        An even winding number means the point lies OUTSIDE the polygon;
        an odd number means it lies INSIDE it.
        """
        winding_number = 0
        infinity = Vertex((1000000, self.y))
        for q in poly.__iter__():
            if not q.intersect and intersect(self, infinity, q, poly.next(q.next)):
                winding_number += 1

        return (winding_number % 2) != 0

    def setChecked(self):
        self.checked = True
        if self.neighbour and not self.neighbour.checked:
            self.neighbour.setChecked()

    def __repr__(self):
        """String representation of the vertex for debugging purposes."""
        return 'Vertex({}, {}, {})'.format(self.x, self.y, self.z)

    def __sub__(self, other):
        return self._array - np.array(other)

    def __add__(self, other):
        return self._array + np.array(other)

    @property
    def dot(self):
        return dot3pt(np.array(self.previous), np.array(self), np.array(self.next))

    @property
    def angle(self):
        return angle3pt(np.array(self.previous), np.array(self), np.array(self.next))

    def to_polycurve(self):
        return


class Polygon:
    """Manages a circular doubly linked list of Vertex objects that represents a polygon."""

    head = None

    def __init__(self, pts=None):
        if pts is not None:
            for s in np.array(pts, float):
                self.add(Vertex(s))

    def add(self, vertex):
        """Add a vertex object to the polygon (vertex is added at the 'end' of the list")."""
        if not self.head:
            self.head = vertex
            self.head.next = vertex
            self.head.previous = vertex
        else:
            next = self.head
            previous = next.previous
            next.previous = vertex
            vertex.next = next
            vertex.previous = previous
            previous.next = vertex

    def insert(self, vertex, start, end):
        """Insert and sort a vertex between a specified pair of vertices.

        This function inserts a vertex (most likely an intersection point)
        between two other vertices (start and end). These other vertices
        cannot be intersections (that is, they must be actual vertices of
        the original polygon). If there are multiple intersection points
        between the two vertices, then the new vertex is inserted based on
        its alpha value.
        """
        curr = start
        while curr != end and curr.alpha < vertex.alpha:
            curr = curr.next

        vertex.next = curr
        previous = curr.previous
        vertex.previous = previous
        previous.next = vertex
        curr.previous = vertex

    def next(self, v):
        """Return the next non intersecting vertex after the one specified."""
        c = v
        while c.intersect:
            c = c.next
        return c

    @property
    def nextPoly(self):
        """Return the next polygon (pointed by the head vertex)."""
        return self.head.nextPoly

    @property
    def head_intersect(self):
        """Return the head unchecked intersection point in the polygon."""
        for v in self.__iter__():
            if v.intersect and not v.checked:
                break
        return v

    @property
    def points(self):
        """Return the polygon's points as a list of tuples (ordered coordinates pair)."""
        p = []
        for v in self.__iter__():
            p.append((v.x, v.y))
        return p

    @property
    def corners(self):
        return np.array(list(self))

    def unprocessed(self):
        """Check if any unchecked intersections remain in the polygon."""
        for v in self.__iter__():
            if v.intersect and not v.checked:
                return True
        return False

    def union(self, clip):
        return self.clip(clip, False, False)

    def intersection(self, clip):
        return self.clip(clip, True, True)

    def difference(self, clip):
        return self.clip(clip, False, True)

    def __or__(self, other):
        if isinstance(other, Polygon):

            return self.__copy__().intersection(other.__copy__())
        elif isinstance(other, (list, tuple)):
            return [self.__or__(i) for i in other]
        else:
            raise TypeError(f"Unexpected type {type(other)}")

    def __ror__(self, other):
        if isinstance(other, Polygon):

            return other.__copy__().intersection(self.__copy__())
        elif isinstance(other, (list, tuple)):
            return [self.__ror__(i) for i in other]
        else:
            raise TypeError(f"Unexpected type {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Polygon):

            return self.__copy__().difference(other.__copy__())
        elif isinstance(other, (list, tuple)):
            return [self.__sub__(i) for i in other]
        else:
            raise TypeError(f"Unexpected type {type(other)}")

    def __rsub__(self, other):
        if isinstance(other, Polygon):

            return other.__copy__().difference(self.__copy__())
        elif isinstance(other, (list, tuple)):
            return [self.__rsub__(i) for i in other]
        else:
            raise TypeError(f"Unexpected type {type(other)}")

    def __add__(self, other):
        if isinstance(other, Polygon):

            return self.__copy__().union(other.__copy__())
        elif isinstance(other, (list, tuple)):
            return [self.__add__(i) for i in other]
        else:
            raise TypeError(f"Unexpected type {type(other)}")

    def __radd__(self, other):
        if isinstance(other, Polygon):

            return other.__copy__().union(self.__copy__())
        elif isinstance(other, (list, tuple)):
            return [self.__radd__(i) for i in other]
        else:
            raise TypeError(f"Unexpected type {type(other)}")

    @property
    def dots(self):
        crn = np.array([(v.previous, v, v.next) for v in self.__iter__()])
        return dot3pt(crn[..., 0, :], crn[..., 1, :], crn[..., 2, :])

    @property
    def angles(self):
        crn = np.array([(v.previous, v, v.next) for v in self.__iter__()])
        return angle3pt(crn[..., 0, :], crn[..., 1, :], crn[..., 2, :])

    def refine(self):
        node = self.head
        while True:
            if node.next is self.head:
                break
            node = node.next
            if np.allclose(np.abs(node.dot), 1.):
                node.previous.next = node.next
                node.next.previous = node.previous
                node = node.next

    def clip(self, clip, s_entry, c_entry):
        """Clip this polygon using another one as a clipper.

        This is where the algorithm is executed. It allows you to make
        a UNION, INTERSECT or DIFFERENCE operation between two polygons.

        Given two polygons A, B the following operations may be performed:

        A|B ... A OR B  (Union of A and B)
        A&B ... A AND B (Intersection of A and B)


        The entry records store the direction the algorithm should take when
        it arrives at that entry point in an intersection. Depending on the
        operation requested, the direction is set as follows for entry points
        (f=forward, b=backward; exit points are always set to the opposite):

              Entry
              A   B
              -----
        A|B   b   b
        A&B   f   f


        f = True, b = False when stored in the entry record
        """
        # phase one - find intersections
        for s in self.__iter__():  # for each vertex Si of subject polygon do
            if not s.intersect:
                for c in clip.__iter__():  # for each vertex Cj of clip polygon do
                    if not c.intersect:
                        try:

                            i, alphaS, alphaC = self.handle_intersection(s, c, clip)
                            iS = Vertex(i, alphaS, intersect=True, entry=False)
                            iC = Vertex(i, alphaC, intersect=True, entry=False)
                            iS.neighbour = iC
                            iC.neighbour = iS

                            self.insert(iS, s, self.next(s.next))
                            clip.insert(iC, c, clip.next(c.next))
                        except TypeError:
                            pass  # this simply means intersect() returned None

        # phase two - identify entry/exit points
        s_entry ^= self.head.isInside(clip)
        for s in self.__iter__():
            if s.intersect:
                s.entry = s_entry
                s_entry = not s_entry

        c_entry ^= clip.head.isInside(self)
        for c in clip.__iter__():
            if c.intersect:
                c.entry = c_entry
                c_entry = not c_entry

        # phase three - construct a list of clipped polygons
        list = []
        while self.unprocessed():
            current = self.head_intersect
            clipped = Polygon()
            clipped.add(Vertex(current))
            while True:
                current.setChecked()
                if current.entry:
                    while True:
                        current = current.next
                        clipped.add(Vertex(current))
                        if current.intersect:
                            break
                else:
                    while True:
                        current = current.previous
                        clipped.add(Vertex(current))
                        if current.intersect:
                            break

                current = current.neighbour
                if current.checked:
                    break

            list.append(clipped)

        if not list:
            list.append(self)

        return list

    def handle_intersection(self, s_node, clip_node, clip):
        i, alphaS, alphaC = intersect(s_node, self.next(s_node.next), clip_node, clip.next(clip_node.next))
        return i, alphaS, alphaC

    def __copy__(self):
        return self.__class__(np.copy(self.corners))

    def __repr__(self):
        """String representation of the polygon for debugging purposes."""

        return "Polygon({})".format(self.corners)

    def __iter__(self):
        """Iterator generator for this doubly linked list."""
        s = self.head
        while True:
            yield s
            s = s.next
            if s == self.head:
                return

    @property
    def area(self):
        return polygon_area(to_polygon_area(self.corners))


def intersect(s1, s2, c1, c2):
    """Test the intersection between two lines (two pairs of coordinates for two points).

    Return the coordinates for the intersection and the subject and clipper alphas if the test passes.

    Algorithm based on: http://paulbourke.net/geometry/lineline2d/
    """

    den = (c2.y - c1.y) * (s2.x - s1.x) - (c2.x - c1.x) * (s2.y - s1.y)

    if not den:
        return None

    us = ((c2.x - c1.x) * (s1.y - c1.y) - (c2.y - c1.y) * (s1.x - c1.x)) / den
    uc = ((s2.x - s1.x) * (s1.y - c1.y) - (s2.y - s1.y) * (s1.x - c1.x)) / den

    if (us == 0 or us == 1) and (0 <= uc <= 1) or (uc == 0 or uc == 1) and (0 <= us <= 1):
        print("whoops! degenerate case!")
        return None

    elif (0 < us < 1) and (0 < uc < 1):
        x = s1.x + us * (s2.x - s1.x)
        y = s1.y + us * (s2.y - s1.y)
        return (x, y), us, uc

    return None


def find_origin(subject, clipper):
    """Find the center coordinate for the given points."""
    x, y = [], []

    for s in subject:
        x.append(s[0])
        y.append(s[1])

    for c in clipper:
        x.append(c[0])
        y.append(c[1])

    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    width = x_max - x_min
    height = y_max - y_min

    return -x_max / 2, -y_max / 2, -(1.5 * width + 1.5 * height) / 2


def clip_polygon(subject, clipper, operation='difference'):
    """Higher level function for clipping two polygons (from a list of points)."""
    Subject = Polygon()
    Clipper = Polygon()

    for s in subject:
        Subject.add(Vertex(s))

    for c in clipper:
        Clipper.add(Vertex(c))

    clipped = Clipper.difference(Subject) if operation == 'reversed-diff' else Subject.__getattribute__(operation)(
        Clipper
    )

    return clipped


class PolygonCollection(list[Polygon]):
    def __init__(self, polygons=()):
        super(PolygonCollection, self).__init__(polygons)

    def tolist(self):
        return [poly.corners.tolist() for poly in self]

    @property
    def areas(self):
        return np.array([poly.area for poly in self])

    @property
    def area(self):
        return np.sum(self.areas)

    def _union(self):

        lst = [*self]
        i = 0
        lst2 = []
        while True:
            if i == len(self):
                break
            i += 1
            lst2.append(lst.pop(0))

            for i, p in enumerate(list(lst)):
                res = p.union(lst2[-1])
                if len(res) == 1:
                    lst[i] = res[0]
                    del lst2[-1]
                    break

            return lst, lst2


def ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray[Any, np.dtype[np.bool_]]:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def intersects_segments(ab: np.ndarray, cd: np.ndarray, ab_bbox=None, cd_bbox=None) -> bool:
    if ab_bbox is None:
        ab_bbox = aabb(ab)
    if cd_bbox is None:
        cd_bbox = aabb(cd)
    if not aabb_overlap(ab_bbox, cd_bbox):
        return False
    a, b = ab
    c, d = cd
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def segments_by_loop(loop, start_index=0):
    return np.array([(i, (i + 1) % len(loop)) for i in range(len(loop))], dtype=np.int32) + start_index


def point_in_polygon(points: np.ndarray, point: np.ndarray, segments=None, poly_aabb=None):
    if poly_aabb is None:
        poly_aabb = aabb(points)

    if not point_in_aabb(poly_aabb, point):
        return False
    if np.allclose(points[0], points[-1]):
        points = points[1:]
    if segments is None:
        segments = segments_by_loop(points)
    segment = np.array([point, point + poly_aabb[1] - poly_aabb[0]])

    bbox_segm = aabb(segment)
    cnt = 0
    for pts in points[segments]:
        if intersects_segments(pts, segment, cd_bbox=bbox_segm):
            cnt += 1
    return cnt % 2 == 1


import numpy as np


def is_point_in_polygon(point, polygon):
    """
    Determines if a point is inside an arbitrary polygon using the Ray-Casting algorithm.

    Parameters:
    point (tuple): The point to check, represented as (x, y).
    polygon (list): List of tuples representing the vertices of the polygon in (x, y) format.

    Returns:
    bool: True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


class BoundingBox:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def intersects_ray(self, x, y):
        # Check if the ray intersects the bounding box
        return y >= self.y_min and y <= self.y_max and x <= self.x_max

    def union(self, other):
        # Create a bounding box that encompasses both self and other
        return BoundingBox(
            min(self.x_min, other.x_min),
            max(self.x_max, other.x_max),
            min(self.y_min, other.y_min),
            max(self.y_max, other.y_max)
        )


class BVHNode:
    def __init__(self, bounding_box, left=None, right=None, edge=None):
        self.bounding_box = bounding_box
        self.left = left
        self.right = right
        self.edge = edge  # None for non-leaf nodes


def polygon_build_bvh(edges):
    def build_recursive(objects):
        if len(objects) == 1:
            return BVHNode(objects[0][0], edge=objects[0][1])

        # Sort edges by the center of their bounding boxes
        objects.sort(key=lambda obj: (obj[0].x_min + obj[0].x_max) / 2)

        mid = len(objects) // 2
        left = build_recursive(objects[:mid])
        right = build_recursive(objects[mid:])

        bounding_box = left.bounding_box.union(right.bounding_box)
        return BVHNode(bounding_box, left, right)

    bounding_boxes = [(BoundingBox(min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)), (x1, y1, x2, y2))
                      for (x1, y1), (x2, y2) in edges]
    return build_recursive(bounding_boxes)


def is_point_in_polygon_bvh(bvh_root, point):
    """

    :param point:
    :param bvh_root:
    :return:

    Example
    ------
    >>> polygon = [[6.2473630632829034, -4.6501623869364659], [4.5977790726622780, -7.9013992636043762], [5.6974884793053437, -8.4180460064619140], [6.4988720101896780, -7.0222670191270575], [5.9886117822374212, -6.7365615003726074], [6.4174018118259593, -6.0076576582442573], [7.3332739860138512, -6.4369235139392975], [6.0892830654991226, -8.6939253124958800], [7.130368038925603, -9.3282139115414324], [8.8187717638415339, -5.7738251567346959]]
    >>> edges = [(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))]
    >>> bvh_root = polygon_build_bvh(edges)

    >>> points = (6.2697549308428240, -7.0629610641788423),(6.6607986921973108, -6.4609212083145193)

    >>> result=[is_point_in_polygon_bvh( bvh_root,point) for point in points]
    >>> print(result)  # Output: [True, False]
    [True, False]
    """
    x, y = point
    intersections = 0

    def intersect_ray(node):
        nonlocal intersections
        if node is None or not node.bounding_box.intersects_ray(x, y):
            return

        if node.edge is not None:
            x1, y1, x2, y2 = node.edge
            if y > min(y1, y2) and y <= max(y1, y2) and x <= max(x1, x2):
                if y1 != y2:
                    xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    if x <= xinters:
                        intersections += 1
            return

        intersect_ray(node.left)
        intersect_ray(node.right)

    intersect_ray(bvh_root)
    return intersections % 2 == 1
