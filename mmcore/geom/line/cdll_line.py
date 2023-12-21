import functools
import itertools
from typing import Any

from mmcore.ds.cdll import CDLL, Node
from mmcore.geom.interfaces import ArrayInterface

from mmcore.geom.line import Line
from mmcore.geom.vec import *


class LineNode(Node):
    def __init__(self, data):
        super().__init__(data)

    def __repr__(self):
        return f'{self.__class__}({self.start}, {self.end})'

    @property
    def start(self):
        return self.data.start

    @property
    def end(self):
        return self.data.end

    @end.setter
    def end(self, v):
        self.data.end = v

    @start.setter
    def start(self, v):
        self.data.start = v

    @property
    def direction(self):
        return self.end - self.start

    @property
    def unit(self):
        return unit(self.direction)

    @property
    def length(self):
        return norm(self.direction)

    @vectorize(excluded=[0], signature='()->(i)')
    def __call__(self, t):
        return self.start + self.direction * t

    @vectorize(excluded=[0], signature='(i)->()')
    def closest_parametr(self, point):
        vec = np.array(point) - self.start
        return np.dot(self.unit, vec / self.length)

    def closest_point(self, point):
        return self(self.closest_parametr(point))

    def closest_distance(self, point):
        return dist(point, self(self.closest_parametr(point)))

    def intersect_lstsq(self, other: 'LineNode'):
        (x1, y1, z1), (x2, y2, z2) = self.start, self.end
        (x3, y3, z3), (x4, y4, z4) = other.start, other.end

        A = np.array([[x2 - x1, x4 - x3], [y2 - y1, y4 - y3]])
        b = np.array([x3 - x1, y3 - y1])

        return np.append(np.linalg.lstsq(A, b), z1)

    def intersect(self, other: 'LineNode'):
        (x1, y1, z1), (x2, y2, z2) = self.start, self.end
        (x3, y3, z3), (x4, y4, z4) = other.start, other.end

        A = np.array([[x2 - x1, x4 - x3], [y2 - y1, y4 - y3]])
        b = np.array([x3 - x1, y3 - y1])

        return np.append(np.linalg.solve(A, b), z1)

    def offset(self, dists: 'float|np.ndarray'):
        if np.isscalar(dists):
            dists = np.zeros(2, float) + dists
        print(dists)
        return LineOffset(dists, offset_previous=self)


class LineOffset(LineNode):
    def __init__(self, dists, offset_previous: LineNode = None):
        super().__init__(dists)
        self._offset_previous = offset_previous
        self.data = dists

    @property
    def offset_previous(self):
        return self._offset_previous

    @offset_previous.setter
    def offset_previous(self, v: LineNode):
        self._offset_previous = v

    @property
    def length(self):
        return dist(IntersectionPoint((self.previous, self)).p, IntersectionPoint((self, self.next)).p)

    def offset_pts(self):
        return IntersectionPoint((self.previous.offset_previous, self)), IntersectionPoint(
                (self, self.next.offset_previous)
                )

    @property
    def start(self):
        return self.offset_previous.start + self.offset_unit_direction * self.distance[0]

    @property
    def end(self):
        return self.offset_previous.end + self.offset_unit_direction * self.distance[1]

    @property
    def distance(self):
        return self.data

    @property
    def offset_unit_direction(self):
        return cross(self.offset_previous.unit, [0., 0., 1.])


class PointsOnCurveCollection(ArrayInterface):
    def __init__(self, t, owner):
        self.t = t
        self.owner = owner

    def __array__(self, dtype=float):
        return np.array(self.owner.evaluate(self.t), dtype=dtype)

    def __iter__(self):
        return iter(self.owner.evaluate(self.t))

    def __getitem__(self, item):
        return PointOnCurve(item, self)

    def __repr__(self):
        return f'{self.__class__.__name__}(length={len(self.t)})'


class PointOnCurve(ArrayInterface):
    def __init__(self, ixs, owner):
        self.ixs = ixs
        self.owner = owner

    @property
    def p(self):
        return self.owner.owner(self.owner.t[self.ixs])

    @property
    def t(self):
        return self.owner.t[self.ixs]

    @t.setter
    def t(self, v):
        self.owner.t[self.ixs] = v

    def __array__(self, dtype=float):
        return np.array(self.p, dtype=dtype)

    def __iter__(self):
        return iter(self.p)

    def __getitem__(self, item):
        a = self.__array__()
        return a[item]

    def __repr__(self):
        return (f'{self.__class__.__name__}(t{self.ixs} = {np.round(self.t, 4)}, p{self.ixs} '
                f'={np.round(self.p, 4)})')


class IntersectionPoint(Node, ArrayInterface):
    def __init__(self, lines: tuple[LineNode, LineNode]):
        super().__init__(lines)

    @property
    def line1(self):
        return self.data[0]

    @property
    def line2(self):
        return self.data[1]

    @line2.setter
    def line2(self, v):
        self.data[1] = v

    @line1.setter
    def line1(self, v):
        self.data[0] = v

    @functools.lru_cache(maxsize=None)
    def solve(self):
        self.t1, t2, _ = self.line1.intersect(self.line2)
        self.t2 = -t2
        return self.t1, self.t2

    def __getitem__(self, item):
        return self.p[item]

    def __hash__(self):
        return hash((hash(self.line1), hash(self.line2)))

    def __len__(self):
        return 3

    @property
    def p(self):
        t1, t2 = self.solve()
        return self.line1(t1)

    @property
    def bounded(self):
        return np.array([(1 >= i >= 0) for i in self.solve()], bool)

    def __array__(self, dtype=float):
        return np.array(self.p, dtype=dtype)

    def __iter__(self):
        return iter(self.p)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.line1} x {self.line2})'


class LineCDLL(CDLL):
    """
    Class LineCDLL

    A class representing a doubly linked circular list of line segments

    Properties:
    - nodetype (class attribute): The class of the nodes in the list
    - lengths: Returns an array of the lengths of all line segments in the list
    - starts: Returns an array of the starting points of all line segments in the list
    - ends: Returns an array of the ending points of all line segments in the list
    - angles: Returns an array of the angles between each line segment and its successor
    - dots: Returns an array of the dot products between each line segment and its successor
    - units: Returns an array of the unit vectors of each line segment
    - dot_matrix: Returns the dot product matrix of the unit vectors of each line segment
    - dist_matrix: Returns an array of the closest distances between the line segments and their midpoints
    - orient_dots: Returns an array of the dot products between each line segment's unit vector and [0., 1., 0.]
    - corners: Returns an array of the corner points from intersecting lines

    Methods:
    - from_cdll(cls, cdll: CDLL): Returns an instance of LineCDLL converted from a CDLL
    - close(): Closes the circular list by connecting the last node with the first node
    - evaluate(t): Returns the position of the line at time t
    - evaluate_node(t): Returns the position of the line at time t as a PointsOnCurveCollection object
    - offset(dists): Returns an LineCDLL instance offset by the given distance(s)
    - get_intersects(): Returns a CDLL instance containing all intersection points between line segments
    - gen_intersects(): Yields intersection points between line segments

    """
    nodetype = LineNode

    def __repr__(self):
        return f'{self.__class__.__name__}[{self.nodetype.__name__}](length={self.count}) at {hex(id(self))}'

    @property
    def lengths(self):
        """
        Returns an array of lengths for each node in the graph.

        :return: An array of lengths.
        :rtype: numpy.ndarray
        """
        return np.array([i.length for i in self.iter_nodes()])

    @property
    def starts(self):
        """
        Returns the starting positions of nodes as a NumPy array.

        :return: An array of starting positions.
        :rtype: numpy.ndarray
        """
        return np.array([i.start for i in self.iter_nodes()])

    @property
    def ends(self):
        return np.array([i.end for i in self.iter_nodes()])

    @property
    def angles(self):
        return np.array([angle(i.unit, i.next.unit) for i in self.iter_nodes()])

    @property
    def dots(self):
        return np.array([dot(i.unit, i.next.unit) for i in self.iter_nodes()])

    @property
    def units(self):
        return np.array([i.unit for i in self.iter_nodes()])

    @classmethod
    def from_cdll(cls, cdll: CDLL):
        lcdll = cls()

        lcdll.count = cdll.count

        temp = cdll.head.next
        while (temp != cdll.head):
            lcdll.append((temp.previous, temp))
            temp = temp.next

        return lcdll

    def close(self):
        node = self.nodetype((self.head.previous.data[1], self.head.data[0]))
        node.previous = self.head.previous
        self.head.previous.next = node
        self.head.previous = node
        node.next = self.head

    @property
    def dot_matrix(self):
        return self.units.dot(self.units.T)

    @property
    def dist_matrix(self):
        return np.array([i.closest_distance(self.starts + (self.starts - self.ends) / 2) for i in self.iter_nodes()])

    @property
    def orient_dots(self):
        return np.array([dot(i.unit, [0., 1., 0.]) for i in self.iter_nodes()])

    @vectorize(excluded=[0], signature='()->(i)')
    def evaluate(self, t):
        """
        Evaluate the value of the function at time 't'.

        :param t: The time at which to evaluate the function.
        :type t: float
        :return: The evaluated value of the function.
        :rtype: int
        """
        d, m = np.divmod(t, 1)

        return self.get(int(np.mod(d, self.count)))(m)

    def __call__(self, t):
        """
        .. method:: __call__(t)

            This method is used to call the `evaluate` method.

            :param t: The input parameter for the `evaluate` method.
            :type t: Any
            :return: The result of the `evaluate` method.
            :rtype: Any

        """
        return self.evaluate(t)

    def evaluate_node(self, t: 'float|np.ndarray[Any, np.dtype[float]]') -> PointsOnCurveCollection:
        """
        :param t: The parameter value at which to evaluate the node.
        :type t: float

        :return: The collection of points on the curve obtained by evaluating the node at the given parameter value.
        :rtype: PointsOnCurveCollection.
        """
        return PointsOnCurveCollection(t, self)

    @property
    def corners(self):

        return np.array([list(i) for i in self.gen_intersects()])

    @corners.setter
    def corners(self, corners):
        for corner, node in itertools.zip_longest(corners, self.iter_nodes(), fillvalue=None):
            if corner is None:
                break
            elif node is None:
                self.append_corner(corner)
            else:
                node.start = np.array(corner)
                node.previous.end = np.array(corner)

    def append_corner(self, value: 'np.ndarray | list[float] | tuple[float,float,float]'):
        """
        :param value: The value to append to the corner.
        :type value:  ndarray | List[float] | tuple[float,float,float]

        :return: None
        :rtype: None

        """
        self.append(Line(np.array(value, float), np.array([1.0, 0.0, 0.0])))
        node = self.head.previous
        node.end = self.head.start
        node.previous.end = node.start

    def set_corner(self, index: int, value: 'np.ndarray | list[float] | tuple[float,float,float]'):
        node = self.get_node(index)
        node.start = np.array(value)
        node.previous.end = node.start

    def insert_corner(self, value: 'np.ndarray | list[float] | tuple[float,float,float]', index: int):
        """


        :param value: The value to be inserted as the corner of the line.
        :type value: ndarray | List[float] | tuple[float,float,float]
        :param index: The index at which the corner should be inserted.
        :type index: int
        :return: None
        :rtype: None

        """
        self.insert(Line(np.array(value, float), np.array([1.0, 0.0, 0.0])), index)
        node = self.get_node(index)
        node.end = node.next.start
        node.previous.end = node.start

    def offset(self, dists):
        """
        :param dists: The distances to offset each node by. Can be a single value or an array-like object.
        :type dists: Union[Number, Sequence[Number]]
        :return: A new LineCDLL object with each node offset by the corresponding distance.
        :rtype: LineCDLL

        This method takes a LineCDLL object and offsets each node by the corresponding value in the `dists`
        parameter. If `dists` is a single value, all nodes will be offset by that distance
        *. If `dists` is an array-like object, each node will be offset by the corresponding value in the array.

        Example usage:
        ```python
        line = LineCDLL()
        line.append_node(Node(0))
        line.append_node(Node(1))
        line.append_node(Node(2))

        offsets = [1, 2, 3]

        result = line.offset(offsets)
        ```
        In this example, the `offsets` array specifies the distances to offset each node. The resulting `result`
        LineCDLL object will have nodes at positions (1, 2, 3), (3, 4, 5), and (6, 7
        *, 8), respectively.
        """
        lst = LineCDLL()
        for node, offset_dist in itertools.zip_longest(self.iter_nodes(), np.atleast_1d(dists),
                                                       fillvalue=0. if not np.isscalar(dists) else dists
                                                       ):
            lst.append_node(node.offset(offset_dist))

        return lst

    def get_intersects(self):
        """
        Returns a list of intersection points.

        :return: The list of intersection points.
        :rtype: CDLL
        """
        lst = CDLL()
        for node in self.iter_nodes():
            lst.append_node(IntersectionPoint((node.previous, node)))

        return lst

    def gen_intersects(self):
        """
        Returns a generator that yields IntersectionPoint objects for each node.

        :return: A generator that yields IntersectionPoint objects.
        :rtype: Generator[IntersectionPoint, None, None]
        """
        for node in self.iter_nodes():
            yield IntersectionPoint((node.previous, node))
