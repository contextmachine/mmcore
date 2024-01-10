import numpy as np

from mmcore.ds.cdll import CDLL
from mmcore.geom.shapes.area import polygon_area, to_polygon_area
from mmcore.geom.vec import *


class Vertex(object):
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


class Polygon(object):
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


from mmcore.geom.intersections.predicates import intersects_segments


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
