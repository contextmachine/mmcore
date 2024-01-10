from collections import deque

from typing import Any, Iterable

from scipy.linalg import solve

from mmcore.geom.parametric import algorithms as algo
from mmcore.geom.closest_point import ClosestPointSolution1D
from mmcore.geom.plane import Plane
from mmcore.geom.proto import object_from_bytes, object_to_buffer, object_to_bytes
from mmcore.geom.vec import *
from mmcore.geom.vec import dist, unit


def closest_parameter(start, end, pt) -> float:
    """
    Calculate the closest point on a line segment to a given point.

    :param start: The starting point of the line segment.
    :type start: tuple(float)
    :param end: The ending point of the line segment.
    :type end: tuple(float)
    :param pt: The point to which the closest point needs to be calculated.
    :type pt: tuple(float)
    :return: The closest parameter (t) on the line segment to the given point.
    :rtype: float
    """
    line = start, end
    vec = np.array(pt) - line[0]

    return np.dot(unit(line[1] - line[0]), vec / dist(line[0], line[1]))


closest_parameter = np.vectorize(closest_parameter, signature='(i),(i),(i)->()', doc=closest_parameter.__doc__)


def closest_point(starts, ends, pts):
    """
    Finds the closest point on a line segment to a given set of points.

    :param starts: The starting points of the line segments.
    :type starts: list of tuples
    :param ends: The ending points of the line segments.
    :type ends: list of tuples
    :param pts: The points for which to find the closest points on the line segments.
    :type pts: list of tuples
    :return: The closest points on the line segments to the given points.
    :rtype: list of tuples
    """
    return evaluate_line(starts, ends, closest_parameter(starts, ends, pts))


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


from mmcore.geom.tolerance import hash_ndarray_float
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



        a,      x0,        z1
    x0  x2-x1    x1          0
    y0  y2-y1    y1
    z0  z2-z1    z1

    """
    __slots__ = ('_array')
    _shape = (3, 3)
    _dtype = float

    def __init__(self, start=None, direction=None, end=None):
        if end is None:
            end = start + direction
        self._array = np.zeros((self._shape[0] + 1, self._shape[1]), self._dtype)

        self._array[0] = start
        self._array[1] = end
        self._array[2] = direction
        self._array[3] = unit(direction)

    def __hash__(self):

        return hash((hash_ndarray_float(self.start), hash_ndarray_float(self.end)))
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
        self.solve()

    def solve(self):

        self._array[2] = self._array[1] - self._array[0]
        self._array[3] = unit(self._array[2])

    @property
    def unit(self):
        return self._array[3]

    @vectorize(excluded=[0], signature='()->(i)')
    def evaluate(self, t):
        return self.direction * t + self.start

    __call__ = evaluate

    def __array__(self, dtype=None):
        return np.array(self._array, dtype) if dtype is None else self._array


    def length(self):
        return norm(self.direction)

    def scale(self, x: float) -> None:
        self._array[2] *= x
        self._array[1] = self._array[0] + self._array[2]
        self.solve()

    def extend(self, start: float, end: float) -> None:
        self._array[0] -= self.unit * start
        self._array[1] += self.unit * end
        self.solve()

    @vectorize(excluded=[0], signature='()->(j,i)')
    def _normal(self, t):
        return np.array([self.evaluate(t), perp2d(self.unit)])

    # normal = evaluate

    def to_startend(self) -> StartEndLine:
        return self._array[:2]

    def to_pointvec(self) -> PointVecLine:
        return np.array([self._array[0], self._array[2]])

    @classmethod
    def from_ends(cls, start: np.ndarray, end: np.ndarray):

        if not all([isinstance(start, np.ndarray) or isinstance(end, np.ndarray)]):
            start, end = np.array([start, end], dtype=cls._dtype)
        return cls(start=start, direction=end - start, end=end)

    @classmethod
    def from_point_vec(cls, start: np.ndarray, direction: np.ndarray):
        if not all([isinstance(start, np.ndarray) or isinstance(direction, np.ndarray)]):
            start, end = np.array([start, direction], dtype=cls._dtype)
        return cls(start=start, direction=direction)

    def __repr__(self):
        return f'Line(start={self.start}, end={self.end}, direction={self.direction})'

    def offset(self, d):
        """
        Offset the line segment by a specified distance.

        :param dist: The distance to offset the line segment.
        :type dist: float

        :return: The offset line segment.
        :rtype: Line

        """
        start, end = self.evaluate(np.array([0, 1]))
        if np.isscalar(d):
            vec1 = vec2 = perp2d(self.unit) * d
        else:
            p = perp2d(self.unit)
            vec1 = p * d[0]
            vec2 = p * d[1]

        return Line.from_ends(self.start + vec1, end + vec2)

    def pde_offset(self, d):
        def fun(t):
            start, vec1 = self._normal(t)
            return start + vec1 * d

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
        (start, vec1), (end, vec2) = self._normal([0.0, 1.0])
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

        (x1, y1, z1), (x2, y2, z2) = self.start, self.end
        (x3, y3, z3), (x4, y4, z4) = other.start, other.end

        A = np.array([[x2 - x1, x4 - x3], [y2 - y1, y4 - y3]])
        b = np.array([x3 - x1, y3 - y1])

        return np.append(solve(A, b), z1)


    def intersect(self, other: 'Line'):

        (x1, y1), (x2, y2) = self.start[:2], self.end[:2]
        (x3, y3), (x4, y4) = other.start[:2], other.end[:2]

        A = np.array([[x2 - x1, x4 - x3], [y2 - y1, y4 - y3]])
        b = np.array([x3 - x1, y3 - y1])

        return solve(A, b)[0]
    def perpendicular_vector(self):
        z = np.zeros(self.direction.shape, dtype=float)
        z[:] = self.direction
        x, y = self.direction[:2]
        z[:2] = -y, x
        return unit(z)

    @vectorize(excluded=[0, 2], signature='(i)->(i)')
    def closest_point(self, pt)->np.ndarray[Any, np.dtype[float]]:
        t = self.closest_parameter(pt)


        return self.evaluate(t)

    @vectorize(excluded=[0], signature='(i)->()')
    def closest_parameter(self, pt: 'tuple|list|np.ndarray')->np.ndarray[Any, np.dtype[float]]:
        """
        :param pt: The point to which the closest parameter is to be found.
        :type pt: list, tuple or numpy array

        :return: The closest parameter value along the line to the given point.
        :rtype: float
        """
        vec = np.array(pt) - self.start
        np.array(pt) - self.start
        return np.dot(self.unit, vec / self.length())

    @vectorize(excluded=[0], signature='(i)->()')
    def closest_distance(self, pt: 'tuple|list|np.ndarray[3, float]')->np.ndarray[Any, np.dtype[float]]:
        """
        :param pt: The point for which the closest distance needs to be calculated.
        :type pt: list, tuple or numpy array

        :return: The closest distance between the given point `pt` and the curve.
        :rtype: float

        """
        pt1 = np.array(pt)
        t = self.closest_parameter(pt1)
        pt2 = self.evaluate(t)
        return dist(pt2, pt1)

    def closest_point_full(self, pt: 'tuple|list|np.ndarray[3, float]') -> ClosestPointSolution1D:
        pt = np.array(pt)
        t = self.closest_parameter(pt)

        pt2 = self(t)

        return ClosestPointSolution1D(pt2,
                                      dist(pt2, pt),
                                      0 <= t or t <= 1,
                                      t)

    @vectorize(excluded=[0], signature='()->(i)')
    def evaluate_distance(self, t: float):
        return self.start + (self.unit * t)

    def __getitem__(self, item):
        return self._array[item]

    def __setitem__(self, item, val):
        self._array[item] = val

    def plane_intersection(self, plane: Plane, epsilon=1e-3):

        ndotu = dot(unit(plane.normal), self.unit)
        print(ndotu, epsilon)
        if abs(ndotu) < epsilon:
            return None, None, None

        w = self.start - plane.origin
        si = -np.array(plane.normal).dot(w) / ndotu
        Psi = w + si * self.direction + plane.origin

        return w, si, Psi

    @classmethod
    def from_buffer(cls,
                    btc: bytearray,
                    dtype=None,
                    **kwargs):
        start, end, direction = object_from_bytes(cls, btc, dtype, **kwargs)
        return cls(start=start, end=end, direction=direction)

    def to_bytes(self, **kwargs) -> bytes:

        return object_to_bytes(self, **kwargs)

    def to_buffer(self, buffer: bytearray, **kwargs) -> dict:
        return object_to_buffer(self, buffer=buffer, dtype=self._dtype, **kwargs)

    def __iter__(self):
        return iter((self.start, self.end))

    @property
    def perp(self):
        return cross([0, 0, 1], self.unit)


X_AXIS_LINE = Line(start=np.array([0.0, 0.0, 0.0]), direction=np.array([1, 0, 0]))
Y_AXIS_LINE = Line(start=np.array([0.0, 0.0, 0.0]), direction=np.array([0, 1, 0]))


def lines_from_ends(points: Iterable):

    for point in points:
        yield Line.from_ends(*point)


def grid_intersection(u_lines, v_lines):
    first, second = u_lines, v_lines
    res = np.zeros((3, len(first), len(second)))
    res1 = np.zeros((len(first), len(second), 3))
    res2 = np.zeros((len(second), len(first), 3))

    for i, f in enumerate(first):
        for j, s in enumerate(second):
            res[:, i, j] = f.bounded_intersect(s)
            res1[i, j] = f.evaluate(res[0, i, j])
            res2[j, i] = s.evaluate(np.abs(res[1, i, j]))

    return res, res1, res2


def permutate_intersection(lines, arr, pts, aj):
    u = v = len(lines)
    res2 = np.zeros((u, v), int)

    for i in range(u):
        for j in range(v):
            # print(aj,flush=True,end='\r')
            if i != j and (res2[i, j] != 1):
                t = lines[i].bounded_intersect(lines[j]).tolist()

                if not np.all(np.isnan(t)) and (0 <= t[0] <= 1):
                    aj[i, j] = 1.0
                    print(t)

                    arr[:, i, j] = t

                    pts[i, j] = lines[i].evaluate(arr[0, i, j])
                    # res2[j, i] = lines[j].evaluate(np.abs(res[1, j, i]))

                res2[i, j] = 1

    return aj


def permutate_intersection_pts(lines):
    u = v = len(lines)
    res2 = np.zeros((u, v), int)


def line_angles(lns):
    d2 = deque(lns)
    d3 = deque(lns)
    d3.rotate(1)
    rot_to(d2, d3)
    return angle(*list(zip(*[(d2[i].unit, d3[i].unit) for i in range(len(lns))]
                           ))), d2, d3


def rot_to(d1, d2, i=0):
    if i > len(d1):
        return d1, d2
    else:
        if np.isclose(angle(d1[0].unit, d2[0].unit), np.pi / 2):
            return d1, d2
        else:
            d1.rotate(-1)
            d2.rotate(-1)
            return rot_to(d1, d2, i + 1)


