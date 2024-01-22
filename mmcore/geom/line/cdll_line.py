import functools
import itertools
from typing import Any, Protocol, Type, TypeVar
import numpy as np

from mmcore.geom.extrusion import polyline_to_lines

T = TypeVar("T")


from scipy.spatial import KDTree

from mmcore.ds.cdll import CDLL, Node
from mmcore.geom.interfaces import ArrayInterface

from mmcore.geom.line import Line
from mmcore.geom.vec import *


class LineNode(Node):
    """A class representing a node on a line.

    Inherits from Node class.

    Attributes:
        _next_offset: A LineOffset object representing the next offset of the line node.
        start: A tuple representing the start point of the line node.
        end: A tuple representing the end point of the line node.
    """

    def __init__(self, data: Line):
        self._next_offset = None
        super().__init__(data)

    @vectorize(excluded=[0, 2], signature="(i)->(i)")
    def closest_point(self, pt) -> np.ndarray[Any, np.dtype[float]]:
        t = self.closest_parameter(pt)

        return self.evaluate(t)

    @vectorize(excluded=[0], signature="(i)->()")
    def closest_parameter(
        self, pt: "tuple|list|np.ndarray"
    ) -> np.ndarray[Any, np.dtype[float]]:
        """
        :param pt: The point to which the closest parameter is to be found.
        :type pt: list, tuple or numpy array

        :return: The closest parameter value along the line to the given point.
        :rtype: float
        """
        vec = np.array(pt) - self.start

        return np.dot(self.unit, vec / norm(self.direction))

    @vectorize(excluded=[0], signature="(i)->()")
    def closest_distance(
        self, pt: "tuple|list|np.ndarray[3, float]"
    ) -> np.ndarray[Any, np.dtype[float]]:
        """
        :param pt: The point for which the closest distance needs to be calculated.
        :type pt: list, tuple or numpy array

        :return: The closest distance between the given point `pt` and the curve.
        :rtype: float

        """
        pt = np.array(pt)
        t = self.closest_parameter(pt)
        pt2 = self.evaluate(t)
        return dist(pt2, pt)

    @vectorize(excluded=[0], signature="()->(i)")
    def evaluate_distance(self, t: float):
        return self.start + (self.unit * t)

    def closest_point_node(self, pt):
        return ClosestPointsOnCurveCollection(pt, self)

    @vectorize(excluded=[0], signature="()->(i)")
    def evaluate(self, t):
        return self.start + self.direction * t

    def __hash__(self):
        return hash(self.data)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __repr__(self):
        return f"{self.__class__}({self.start}, {self.end})"

    @property
    def start(self):
        return self.data.unbounded_intersect(self.previous.data)

    @property
    def end(self):
        return self.data.unbounded_intersect(self.next.data)

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

    @property
    def start_angle(self):
        return angle(self.data.unit, -self.previous.data.unit)

    @property
    def end_angle(self):
        return angle(-self.data.unit, self.next.data.unit)

    @property
    def start_angle_dot(self):
        return dot(self.data.unit, self.previous.data.unit)

    @property
    def end_angle_dot(self):
        return dot(self.data.unit, self.next.data.unit)

    def __call__(self, t):
        return self.evaluate(t)

    def intersect_lstsq(self, other: "LineNode"):
        (x1, y1, z1), (x2, y2, z2) = self.start, self.end
        (x3, y3, z3), (x4, y4, z4) = other.start, other.end

        A = np.array([[x2 - x1, x4 - x3], [y2 - y1, y4 - y3]])
        b = np.array([x3 - x1, y3 - y1])

        return np.append(np.linalg.lstsq(A, b), z1)

    def intersect(self, other: "LineNode"):
        (x1, y1, z1), (x2, y2, z2) = self.start, self.end
        (x3, y3, z3), (x4, y4, z4) = other.start, other.end

        A = np.array([[x2 - x1, x4 - x3], [y2 - y1, y4 - y3]])
        b = np.array([x3 - x1, y3 - y1])

        return np.append(np.linalg.solve(A, b), z1)

    def offset(self, dists: "float|np.ndarray"):
        if np.isscalar(dists):
            dists = np.zeros(2, float) + dists
        v = cross(self.data.unit, [0.0, 0.0, 1.0])

        return self.__class__(
            Line.from_ends(self.start + v * dists[0], self.end + v * dists[1])
        )


class LineOffset(LineNode):
    """
    A class representing a line segment with an offset from a previous line node.

    Attributes:
        _offset_previous (LineNode): The previous line node.
        _distance (numpy.ndarray): The distance between the start and end points.

    """

    _distance = None

    def __init__(self, dists, offset_previous: LineNode = None):
        super().__init__(offset_previous.data)
        self._offset_previous = offset_previous
        self.distance = dists

    def __iter__(self):
        return iter((self.start, self.end))

    def __hash__(self):
        return hash((hash(self._offset_previous), hash(self.distance.tobytes())))

    @property
    def offset_previous(self):
        return self._offset_previous

    @offset_previous.setter
    def offset_previous(self, v: LineNode):
        self._offset_previous = v

    @property
    def length(self):
        return dist(
            IntersectionPoint((self.previous, self)).p,
            IntersectionPoint((self, self.next)).p,
        )

    def offset_pts(self):
        return IntersectionPoint(
            (self.previous.offset_previous, self)
        ), IntersectionPoint((self, self.next.offset_previous))

    @property
    def start(self):
        return (
            self.offset_previous.start + self.offset_unit_direction * self.distance[0]
        )

    @property
    def end(self):
        return self.offset_previous.end + self.offset_unit_direction * self.distance[1]

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, val):
        if np.isscalar(val):
            value = np.zeros(2) + val
        else:
            value = np.array(val)

        self._distance = value

    @property
    def offset_unit_direction(self):
        return cross(self.offset_previous.unit, [0.0, 0.0, 1.0])


from mmcore.geom.tolerance import hash_ndarray_float, HashNdArrayMethod

P = TypeVar("P")


class ExecutableNodeProtocol(Protocol[P, T]):
    """A protocol for executable nodes.

    This protocol defines the behavior of executable nodes in a graph.

    Args:
        P: The type of the inputs for the node.
        T: The type of the output of the node.

    Attributes:

        owner (Any): The owner of the node.

    """

    owner: "Any"

    def __hash__(self) -> int:
        ...

    def solve(self) -> T:
        ...

    @property
    def output(self) -> T:
        return self.solve()


class PointsOnCurveCollection(
    ExecutableNodeProtocol[
        np.ndarray[Any, np.dtype[float]], np.ndarray[Any, np.dtype[float]]
    ],
    ArrayInterface,
):
    """
    :class: `PointsOnCurveCollection`

    A class that represents a collection of points on a curve.

    :param params: An array of parameters representing the points on the curve.
    :type params: numpy.ndarray

    :param owner: The owner object of the collection.
    :type owner: Any

    Properties:
        - `t`: The array of parameters representing the points on the curve.

    Methods:
        - `solve()`: Solves the curve equation and returns an array of points.
        - `__array__(dtype: Type[T] = float) -> numpy.ndarray`: Converts the collection to a numpy array.
        - `__hash__()`: Returns the hash value of the collection.
        - `__iter__()`: Returns an iterator over the points on the curve collection.
        - `__getitem__(item) -> PointOnCurve`: Returns the point on the curve at the specified index.
        - `__repr__()`: Returns a string representation of the PointsOnCurveCollection object.

    Note:
        - This class implements the ExecutableNodeProtocol and ArrayInterface protocols.
    """

    def __init__(self, t: np.ndarray[Any, np.dtype[float]], owner):
        self._t = np.array(t, float)
        self.owner = owner
        self._output = None

    @property
    def params(self):
        return {"t": self._t}

    @property
    def p(self):
        return self._output

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, v):
        self._t = np.array(np.atleast_1d(v), float)

    def solve(self) -> np.ndarray[Any, np.dtype[float]]:
        self._output = self.owner.evaluate(self.t)
        return self._output

    def __array__(self, dtype: Type[T] = float) -> np.ndarray[Any, np.dtype[T]]:
        return np.array(self.solve(), dtype=dtype)

    def __hash__(self):
        return hash(
            (
                hash(self.owner),
                hash_ndarray_float(self.t, method=HashNdArrayMethod.full),
            )
        )

    def __iter__(self):
        return iter(self.solve())

    def __getitem__(self, item) -> "PointOnCurve":
        return PointOnCurve(item, self)

    def __repr__(self):
        return f"{self.__class__.__name__}(length={len(np.atleast_1d(self.t))})"

    @property
    def output(self) -> np.ndarray[Any, np.dtype[float]]:
        return self._output


class PointOnCurve(ArrayInterface):
    """A class representing a point on a curve.

    This class is used to represent a point on a curve. It provides properties and methods to access and manipulate
    the coordinates of the point.

    Attributes:
        ixs (int): The index of the point on the curve.
        owner (PointsOnCurveCollection): The parent collection that contains the point.

    """

    ixs: int
    owner: "PointsOnCurveCollection"

    def __init__(self, ixs: int, owner: PointsOnCurveCollection):
        self.ixs = ixs
        self.owner = owner

    @property
    def params(self):
        return {"i": self.ixs}

    @property
    def p(self) -> np.ndarray[Any, np.dtype[float]]:
        return self.owner.owner(self.t)

    @property
    def t(self) -> float:
        return float(self.owner.t[self.ixs])

    @t.setter
    def t(self, v: float):
        self.owner.t[self.ixs] = v

    def __array__(self, dtype=float):
        return np.array(self.p, dtype=dtype)

    def __iter__(self):
        return iter(self.p)

    def __getitem__(self, item):
        a = self.__array__()
        return a[item]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(t{self.ixs} = {np.round(self.t, 4)}, p{self.ixs} "
            f"={np.round(self.p, 4)})"
        )


class ClosestPointsOnCurveCollection(PointsOnCurveCollection):
    """
    Class ClosestPointsOnCurveCollection

    Class representing a collection of closest points on a curve.
    Inherits from PointsOnCurveCollection.

    Attributes:
        closest_points (list): List of closest points on the curve
        _t (float): Parameter value of the closest point

    Methods:
        __init__(closest_points, owner)
            Initialize the ClosestPointsOnCurveCollection object

            Parameters:
                closest_points (list): List of closest points on the curve
                owner (Line): Line object representing the curve

            Returns:
                None

        solve()
            Solve for the closest point on the curve

            Parameters:
                None

            Returns:
                None

        __setitem__(key, value)
            Set a value in closest_points using key

            Parameters:
                key: Position of the value in closest_points list
                value: Value to set in closest_points list

            Returns:
                None

        __getitem__(item)
            Get a ClosestPointOnCurve object from closest_points at the given position

            Parameters:
                item: Position of the ClosestPointOnCurve object in closest_points list

            Returns:
                ClosestPointOnCurve: ClosestPointOnCurve object

        t()
            Get the parameter value of the closest point

            Parameters:
                None

            Returns:
                float: Parameter value of the closest point
    """

    def __init__(self, closest_points, owner: Line):
        self.closest_points = closest_points

        super().__init__(owner.closest_parameter(np.array(self.closest_points)), owner)

    @property
    def params(self):
        return {"closest_points": self.closest_points} | super().params

    def solve(self):
        self._t = self.owner.closest_parameter(np.array(self.closest_points))
        return self._t

    def __setitem__(self, key, value):
        self.closest_points.__setitem__(key, value)

    def __getitem__(self, item):
        return ClosestPointOnCurve(item, self)

    @property
    def t(self):
        return self._t

    def __array__(self, dtype: Type[T] = float) -> np.ndarray[Any, np.dtype[T]]:
        return np.array(self.owner(self.solve()), dtype=dtype)

    def __iter__(self):
        for i in range(len(self.closest_points)):
            yield self[i]


class ClosestPointOnCurve(PointOnCurve):
    """
    Represents a closest point on a curve.

    This class extends the `PointOnCurve` class and adds additional functionality to calculate the closest point on a curve.

    Args:
        ixs (int): The index of the closest point on the curve.
        owner (ClosestPointsOnCurveCollection): The parent collection that manages the closest points.

    Attributes:
        ixs (int): The index of the closest point on the curve.
        owner (ClosestPointsOnCurveCollection): The parent collection that manages the closest points.

    Properties:
        p: Property representing the closest point on the curve.
        t: Property representing the parameter value of the closest point on the curve.

    """

    def __init__(self, ixs, owner: ClosestPointsOnCurveCollection):
        super().__init__(ixs, owner)

    @property
    def closest_point(self):
        return self.owner.closest_points[self.ixs]

    @closest_point.setter
    def closest_point(self, v):
        self.owner.closest_points[self.ixs] = v

    @property
    def p(self):
        return self.owner.owner(self.t)

    @property
    def t(self):
        return self.owner.solve()[self.ixs]

    @t.setter
    def t(self, v):
        raise AttributeError("t readonly")


class IntersectionPoint(Node, ArrayInterface):
    """

    :class:`IntersectionPoint` class represents the intersection point between two lines.

    .. attribute:: data
       :type: tuple[Node|LineNode|LineOffset, Node|LineNode|LineOffset]
       :readonly:

       A tuple containing the two lines that intersect.

    .. method:: __init__(lines: 'tuple[Node|LineNode|LineOffset, Node|LineNode|LineOffset]') -> None

       Initializes a new instance of the :class:`IntersectionPoint` class.
       It takes a tuple of two lines as input.

    .. property:: line1
       :type: Node|LineNode|LineOffset
       :readonly:

       Represents the first line of the intersection point.

    .. property:: line2
       :type: Node|LineNode|LineOffset
       :readonly:

       Represents the second line of the intersection point.

    .. method:: line2.setter

       Setter method for the second line.

    .. method:: line1.setter

       Setter method for the first line.

    .. method:: solve() -> Tuple[float, float]

       Solves for the intersection point between the two lines.
       Returns a tuple of two floats representing the parameters of the intersection point on the first line.

    .. method:: __getitem__(item) -> Any

       Retrieves the value of the intersection point at the given index.
       Returns the value at the specified index.

    .. method:: __hash__() -> int

       Calculates the hash value of the intersection point.
       Returns an integer representing the hash value.

    .. method:: __len__() -> int

       Returns the length of the intersection point, which is always 3.

    .. property:: p
       :type: Any
       :readonly:

       Returns the actual coordinates of the intersection point.
       Returns the coordinates as a tuple of two floats.

    .. property:: bounded
       :type: numpy.ndarray[bool]
       :readonly:

       Checks if the intersection point is within the bounds of both lines.
       Returns a numpy array of boolean values indicating if the intersection point is within bounds.

    .. method:: __array__(dtype=float) -> numpy.ndarray

       Converts the intersection point to a numpy array of specified data type.
       Returns a numpy array representing the intersection point.

    .. method:: __iter__() -> Iterator

       Returns an iterator for the intersection point.
       Returns an"""

    data: "tuple[Node|LineNode|LineOffset, Node|LineNode|LineOffset]"

    def __init__(
        self, lines: "tuple[Node|LineNode|LineOffset, Node|LineNode|LineOffset]"
    ):
        super().__init__([lines[0], lines[1]])
        self.prev_curve_node = lines[0]
        self.next_curve_node = lines[1]

        self.solve()

    @property
    def angle(self):
        return angle(self.line1.unit, self.line2.unit)

    @property
    def dot(self):
        return dot(self.line1.unit, self.line2.unit)

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
        return f"{self.__class__.__name__}({self.line1} x {self.line2})"


@functools.lru_cache(maxsize=None)
def _cached_solve_kd(self: "LineCDLL"):
    """
    :param self: The instance of the LineCDLL class.
    :type self: LineCDLL
    :return: The KDTree object created using the corners of the LineCDLL instance.
    :rtype: KDTree
    """
    return KDTree(self.corners)


from mmcore.geom.polygon import Polygon
class LineCDLL(CDLL):
    """
    :class:`LineCDLL`

    Subclass of :class:`CDLL` with node type :class:`LineNode`.

    .. method:: __repr__()

        Return a string representation of the object.

        :return: The string representation of the object.
        :rtype: str

    .. method:: closest_parametr(point: np.ndarray)

        Find the closest parameter on the line to a given point.

        :param point: The point to find the closest parameter to.
        :type point: numpy.ndarray
        :return: The closest parameter on the line to the given point.
        :rtype: float

    .. attribute:: lengths

        An array of lengths for each node in the graph.

        :type: numpy.ndarray

    .. attribute:: starts

        The starting positions of nodes as a NumPy array.

        :type: numpy.ndarray

    .. attribute:: ends

        The ending positions of nodes as a NumPy array.

        :type: numpy.ndarray

    .. attribute:: angles

        An array of angles for each node in the graph.

        :type: numpy.ndarray

    .. attribute:: dots

        An array of dot products for each node in the graph.

        :type: numpy.ndarray

    .. attribute:: units

        An array of unit vectors for each node in the graph.

        :type: numpy.ndarray

    .. method:: from_cdll(cdll: CDLL)

        Create a new LineCDLL from an existing CDLL.

        :param cdll: The CDLL to create LineCDLL from.
        :type cdll: CDLL
        :return: The new LineCDLL object.
        :rtype: LineCDLL

    .. method:: close()

        Close the line by connecting the last node to the first node.

    .. attribute:: dot_matrix

        The dot product matrix of the units of each node.

        :type: numpy.ndarray

    .. attribute:: dist_matrix

        An array of closest distances for each node in the graph.

        :type: numpy.ndarray

    .. attribute:: orient_dots

        An array of dot products between the unit vectors of each node and the vector [0., 1., 0.].

        :type: numpy.ndarray

    .. method:: evaluate(t)

        Evaluate the value of the function at time 't'.

        :param t: The time at"""

    nodetype = LineNode

    @classmethod
    def from_points(cls, pts):
        lcdll = cls()

        lines = polyline_to_lines(np.array(pts, float))
        for line in lines:

            lcdll.append(Line.from_ends(*line))

        return lcdll
    def __repr__(self):
        return f"{self.__class__.__name__}[{self.nodetype.__name__}](length={self.count}) at {hex(id(self))}"

    def __hash__(self):
        return hash(tuple(hash(l) for l in self))

    def closest_point_node(self, point):
        return self.closest_segment(point).closest_point_node(point)

    def closest_segment(self, point: np.ndarray) -> LineNode:
        return sorted(self.iter_nodes(), key=lambda node: node.closest_distance(point))[
            0
        ]

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
        return np.array([angle(i.previous.unit, i.unit) for i in self.iter_nodes()])

    @property
    def dots(self):
        return np.array([dot(i.previous.unit, i.unit) for i in self.iter_nodes()])

    @property
    def units(self):
        return np.array([i.unit for i in self.iter_nodes()])

    @classmethod
    def from_cdll(cls, cdll: CDLL):
        lcdll = cls()

        lcdll.count = cdll.count

        temp = cdll.head.next
        while temp != cdll.head:
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
        return np.array(
            [
                i.closest_distance(self.starts + (self.starts - self.ends) / 2)
                for i in self.iter_nodes()
            ]
        )

    @property
    def orient_dots(self):
        return np.array([dot(i.unit, [0.0, 1.0, 0.0]) for i in self.iter_nodes()])

    @vectorize(excluded=[0], signature="()->(i)")
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

    def evaluate_node(
        self, t: "float|np.ndarray[Any, np.dtype[float]]"
    ) -> PointsOnCurveCollection:
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

    def solve_kd(self):
        return _cached_solve_kd(self)

    @corners.setter
    def corners(self, corners):

        for i, (corner, node) in enumerate(itertools.zip_longest(
            corners, self.iter_nodes(), fillvalue=None
                )
                ):
            if corner is None:
                self.remove(node.data)

            elif node is None:

                self.append_corner(corner)
            else:
                self.set_corner(i, corner)


    def append_corner(
        self, value: "np.ndarray | list[float] | tuple[float,float,float]"
    ):
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

    def set_corner(
        self, index: int, value: "np.ndarray | list[float] | tuple[float,float,float]"
    ):
        node = self.get_node(index)
        node.start = np.array(value)
        node.previous.end = np.array(value)

    def insert_corner(
        self, value: "np.ndarray | list[float] | tuple[float,float,float]", index: int
    ):
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
        node.data.end = node.next.data.start
        node.previous.data.end = node.data.start

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
        lst = self.__class__()
        for node, offset_dist in itertools.zip_longest(
            self.iter_nodes(),
            np.atleast_1d(dists),
            fillvalue=0.0 if not np.isscalar(dists) else dists,
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

    def difference(self, other: 'LineCDLL'):
        return [self.__class__.from_polygon(p) for p in self.to_polygon().difference(other.to_polygon())]

    def union(self, other: 'LineCDLL'):
        return [self.__class__.from_polygon(p) for p in self.to_polygon().union(other.to_polygon())]

    def intersection(self, other: 'LineCDLL'):
        return [self.__class__.from_polygon(p) for p in self.to_polygon().intersection(other.to_polygon())]

    def to_polygon(self):
        return Polygon(self.corners)

    @classmethod
    def from_polygon(cls, poly: Polygon):
        return cls(seq=poly.corners[:-1])

    def update_from_polygon(self, poly: Polygon):
        self.corners = poly.corners[:-1]

    def split_by_corners(self, start: int, end: int):

        node = self.get_node(start)
        end_node = self.get_node(end)
        pts = []
        while True:
            if node is end_node:
                pts.append(node.start)
                break
            else:
                pts.append(node.start)
                node = node.next
        self.__class__.f
