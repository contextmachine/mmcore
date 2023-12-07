import mmcore.geom.parametric.algorithms as algo
from mmcore.geom.closest_point import ClosestPointSolution1D
from mmcore.geom.vec import *


def evaluate_line(start, end, t):
    return (end - start) * t + start


evaluate_line = np.vectorize(evaluate_line, signature='(i),(i),()->(i)')



StartEndLine = type(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float))
PointVecLine = type(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float))


def perp2d(vec):
    v2 = np.array(vec)
    v2[0] = -vec[1]
    v2[1] = vec[0]
    return v2


class Line:
    """
    Represents a line in 3D space defined by a start point, direction, and end point.

    Attributes:
        _array (ndarray): Internal array representing the line.

    Methods:
        __init__(self, start, direction, end=None)
            Initializes a new instance of the Line class.

        replace_ends(self, start, end)
            Replaces the start and end points of the line.

        scale(self, x: float) -> None:
            Scales the line by a factor of x.

        extend(self, start: float, end: float) -> None:
            Extends the line by a distance of start at the start point and end at the end point.

        length(self)
            Calculates the length of the line.

        offset(self, dist)
            Generates a new line that is offset from the current line by a distance of dist.

        pde_offset(self, dist)
            Generates a new line that is offset from the current line by a distance of dist using partial differential equations.

        variable_offset(self, d1, d2)
            Generates a new line that is offset from the current line by a variable distance d1 at the start point and d2 at the end point.

        pde_variable_offset(self, d1, d2)
            Generates a new line that is offset from the current line by a variable distance d1 at the start point and d2 at the end point using partial differential equations.

        unbounded_intersect(self, other: Line)
            Calculates the intersection point between two lines.

        bounded_intersect(self, other: Line)
            Calculates the bounded intersection points between two lines.

        perpendicular_vector(self)
            Calculates the perpendicular vector to the line.

        closest_point(self, pt, return_parameter=False)
            Calculates the closest point on the line to a given point.

        closest_parameter(self, pt)
            Calculates the parameter value that corresponds to the closest point on the line to a given point.

        closest_distance(self, pt)
            Calculates the closest distance between the line and a given point.

        closest_point_full(self, pt)
            Calculates the closest point on the line to a given point, along with the parameter value and distance.

    Static Methods:
        from_ends(cls, start: ndarray, end: ndarray)
            Creates a new line from the start and end points.

        from_point_vec(cls, start: ndarray, direction: ndarray)
            Creates a new line from the start point and direction vector.

    Magic Methods:
        __repr__(self)
            Returns a string representation of the Line object.

        __iter__(self)
            Iterates over the start and end points of the line.

        __array__(self)
            Returns the internal array representation of the line.

        __getitem__(self, item)
            Retrieves an item from the internal array.

        __setitem__(self, item, val)
            Sets an item in the internal array.
    """
    __slots__ = ('_array')

    def __init__(self, start, direction, end=None):
        if end is None:
            end = start + direction
        self._array = np.zeros((4, 3))
        self._array[0] = start
        self._array[1] = end
        self._array[2] = direction
        self._array[3] = unit(direction)

    @property
    def c(self):
        line = Line.from_ends(np.array(self.unbounded_intersect(X_AXIS_LINE)),
                              np.array(self.unbounded_intersect(Y_AXIS_LINE)))
        return 1, -1, 294933

    @property
    def direction(self):
        return self._array[2]

    @direction.setter
    def direction(self, v):
        self._array[2] = v

    @property
    def start(self):
        return self._array[0]

    @start.setter
    def start(self, v):
        self._array[0] = v

    @property
    def end(self):
        return self._array[1]

    @end.setter
    def end(self, v):
        self._array[1] = v
        self._array[2] = self._array[1] - self._array[0]

    def replace_ends(self, start, end):
        self._array[0] = start
        self._array[1] = end
        self._array[2] = self._array[1] - self._array[0]
        self._array[3] = unit(self._array[2])

    @property
    def unit(self):
        return self._array[3]

    def evaluate(self, t):
        return self.direction * t + self.start

    __call__ = np.vectorize(evaluate, excluded=[0], signature='()->(i)')

    def __iter__(self):
        return iter([self.start, self.end])

    def __array__(self):
        return self._array

    def length(self):
        return norm(self.direction)

    def scale(self, x: float) -> None:
        self._array[2] *= x
        self._array[1] = self._array[0] + self._array[2]

    def extend(self, start: float, end: float) -> None:
        self._array[0] -= self.unit * start
        self._array[1] += self.unit * end
        self._array[2] = self._array[1] - self._array[0]

    def _normal(self, t):
        return np.array(self.evaluate(t), perp2d(self.unit))

    normal = np.vectorize(evaluate, excluded=[0], signature='()->(i)')

    def to_startend(self) -> StartEndLine:
        return self._array[:2]

    def to_pointvec(self) -> PointVecLine:
        return np.array([self._array[0], self._array[2]])

    @classmethod
    def from_ends(cls, start: np.ndarray, end: np.ndarray):
        return Line(start=start, direction=end - start, end=end)

    @classmethod
    def from_point_vec(cls, start: np.ndarray, direction: np.ndarray):
        return Line(start=start, direction=direction)

    def __repr__(self):
        return f'Line(start={self.start}, end={self.end}, direction={self.direction})'

    def offset(self, dist):
        """
        Offset the line segment by a specified distance.

        :param dist: The distance to offset the line segment.
        :type dist: float

        :return: The offset line segment.
        :rtype: Line

        """
        (start, vec1), (end, vec2) = self.normal([0.0, 1.0])
        return Line.from_ends(start + vec1 * dist, end + vec2 * dist)

    def pde_offset(self, dist):
        def fun(t):
            start, vec1 = self._normal(t)
            return start + vec1 * dist

        return np.vectorize(fun, signature='()->(i)')

    def variable_offset(self, d1, d2):
        """Calculate the offset of a line defined by two distances.

        :param d1: The distance from the start point of the line.
        :type d1: float
        :param d2: The distance from the end point of the line.
        :type d2: float
        :return: A new line object with the offset positions.
        :rtype: Line

        """
        (start, vec1), (end, vec2) = self.normal([0.0, 1.0])
        return Line.from_ends(start + vec1 * d1, end + vec2 * d2)

    def pde_variable_offset(self, d1, d2):
        """
        :param d1: initial value for variable offset
        :type d1: float

        :param d2: final value for variable offset
        :type d2: float

        :return: a vectorized function that calculates variable offset values based on input parameter t
        :rtype: numpy.vectorize
        """

        def fun(t):
            start, vec1 = self._normal(t)
            return start + vec1 * ((d2 - d1) * t + d1)

        return np.vectorize(fun, signature='()->(i)')

    def unbounded_intersect(self, other: 'Line'):
        return algo.pts_line_line_intersection2d_as_3d(self.to_startend(), other.to_startend())

    def bounded_intersect(self, other: 'Line'):
        return algo.bounded_line_intersection2d(self.to_pointvec(), other.to_pointvec())

    def perpendicular_vector(self):
        z = np.zeros(self.direction.shape, dtype=float)
        z[:] = self.direction
        x, y = self.direction[:2]
        z[:2] = -y, x
        return unit(z)

    def closest_point(self, pt, return_parameter=False):
        t = self.closest_parameter(pt)
        if return_parameter:
            return self.evaluate(t), t
        return self.evaluate(t)

    def closest_parameter(self, pt: 'tuple|list|np.ndarray'):
        """
        :param pt: The point to which the closest parameter is to be found.
        :type pt: list, tuple or numpy array

        :return: The closest parameter value along the line to the given point.
        :rtype: float
        """
        vec = np.array(pt) - self.start

        return np.dot(self.unit, vec / self.length())

    def closest_distance(self, pt: 'tuple|list|np.ndarray[3, float]'):
        """
        :param pt: The point for which the closest distance needs to be calculated.
        :type pt: list, tuple or numpy array

        :return: The closest distance between the given point `pt` and the curve.
        :rtype: float

        """
        pt1 = np.array(pt)
        t = self.closest_parameter(pt)
        pt2 = self.evaluate(t)
        return dist(pt2, pt)

    def closest_point_full(self, pt: 'tuple|list|np.ndarray[3, float]') -> ClosestPointSolution1D:
        pt = np.array(pt)
        t = self.closest_parameter(pt)

        pt2 = self.evaluate(t)

        return ClosestPointSolution1D(pt2,
                                      dist(pt2, pt),
                                      0 <= t <= 1,
                                      t)

    def __getitem__(self, item):
        return self._array[item]

    def __setitem__(self, item, val):
        self._array[item] = val


X_AXIS_LINE = Line(start=np.array([0.0, 0.0, 0.0]), direction=np.array([1, 0, 0]))
Y_AXIS_LINE = Line(start=np.array([0.0, 0.0, 0.0]), direction=np.array([0, 1, 0]))
