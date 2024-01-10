import functools
import itertools
from typing import Any
from scipy.spatial import ConvexHull
import numpy as np

from mmcore.func import vectorize
from mmcore.geom.line import evaluate_line
from mmcore.geom.plane import WXY, world_to_local, local_to_world


class BorderConvexHull(ConvexHull):
    def __init__(self, arr, *args, **kwargs):
        self._pts = arr
        if np.allclose(arr[..., -1], 0):

            arr = arr[..., :-1]
        super().__init__(arr, *args, **kwargs)

    _pts = None,

    @property
    def pts(self):
        return self._pts

    @property
    def bounds(self):
        return self._pts[self.vertices]

    def __repr__(self):
        return f'ConvexHull({self.bounds})'


def convex_hull_2d(pts):
    if len(pts.shape) > 2:
        return np.array([convex_hull_2d(grp) for grp in pts], object)
    else:
        return pts[ConvexHull(np.copy(pts)[..., :2]).vertices]

@vectorize(signature='(i,j),()->(j)')
def evaluate_polyline(corners, t: float) -> np.ndarray[Any, np.dtype[float]]:
    """
    Evaluate the position on a polyline at a given parametric value.

    :param corners: The corners of the polyline.
    :type corners: list of tuples of int
    :param t: The parametric value.
    :type t: float
    :return: The evaluated position on the polyline.
    :rtype: tuple of int
    """
    n, m = np.divmod(t, 1)

    lines = polyline_to_lines(corners)

    return evaluate_line(lines[int(np.mod(n, len(lines))), 0], lines[int(np.mod(n, len(lines))), 1], m)


@vectorize(signature='(i,j)->(i,k,j)')
def polyline_to_lines(pln_points):
    """
    Convert a polyline to a set of connected lines.

    :param pln_points: The points of the polyline.
    :type pln_points: numpy.ndarray

    :return: The lines as a numpy array.
    :rtype: numpy.ndarray
    """
    return np.stack([pln_points, np.roll(pln_points, -1, axis=0)], axis=1)


@vectorize(signature='(i,j),(),()->(k,j)')
def trim_polyline(pln, t1, t2):
    """
    Trim a polyline based on given start and end parameters.

    :param pln: The polyline to be trimmed.
    :type pln: numpy array of shape (k, j) where k is the number of vertices and j is the number of dimensions.
    :param t1: The start parameter value for trimming.
    :type t1: float
    :param t2: The end parameter value for trimming.
    :type t2: float
    :return: The trimmed polyline.
    :rtype: numpy array of shape (k', j) where k' is the number of vertices in the trimmed polyline.
    """
    n, m = np.divmod([t1, t2], 1)
    n = np.array(n, dtype=int)
    line = polyline_to_lines(pln)[n]
    p1, p2 = evaluate_line(line[0], line[1], m)
    return np.concatenate([[p1], pln[n[0] + 1:n[1] + 1], [p2]])


@vectorize(signature='(i,j),(u),(u,j)->(k,j)')
def insert_polyline_points(pln, ixs, pts):
    """
    :param pln: A NumPy array representing a polyline.
    :type pln: numpy.ndarray

    :param ixs: A NumPy array containing the indices at which to insert the new points.
    :type ixs: numpy.ndarray

    :param pts: A NumPy array containing the new points to be inserted into the polyline.
    :type pts: numpy.ndarray

    :return: A new NumPy array resulting from the insertion of the new points into the polyline.
    :rtype: numpy.ndarray
    """
    return np.insert(pln, ixs, pts, axis=0)


@vectorize(signature='(i,j),(),()->(k,j),(u,j)')
def split_closed_polyline(pln, t1, t2):
    """
    :param pln: A closed polyline represented as a numpy array of 2D points.
    :type pln: numpy.ndarray
    :param t1: The parameter value between 0 and 1 indicating where to split the polyline.
    :type t1: float
    :param t2: The parameter value between 0 and 1 indicating where to split the polyline.
    :type t2: float
    :return: A tuple containing two numpy arrays representing the split polyline.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    n, m = np.divmod([t1, t2], 1)

    n = np.array(n, dtype=int)
    line = polyline_to_lines(pln)[n]
    p1, p2 = evaluate_line(line[0], line[1], m)

    pln = np.insert(pln, [n[0] + 1, n[1] + 1], [p1, p2], axis=0)
    return split_closed_polyline_by_points(pln, int(n[0] + 1), int(n[1] + 2))


def split_closed_polyline_by_points(pln, i, j):
    """
    Split a closed polyline into two parts by specifying start and end points.

    :param pln: The original closed polyline.
    :type pln: numpy.ndarray

    :param i: The index of the starting point.
    :type i: int

    :param j: The index of the ending point.
    :type j: int

    :return: Two parts of the polyline, split by the specified points.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    return np.roll(pln, -i, axis=0)[:(j - i) + 1], np.roll(pln, -j, axis=0)[:pln.shape[0] - (j - i) + 1]


def split_polyline_by_point(pln, i):
    """
    Split a polyline by a given point index.

    :param pln: List representing the polyline.
    :type pln: list
    :param i: Index of the point to split at.
    :type i: int
    :return: Tuple containing the splitted parts of the polyline.
    :rtype: tuple
    """
    return pln[i:], pln[:i + 1]

def bounds_comparsion(t:np.ndarray[(2,), np.dtype[float]], bounds=(0.,1.), rtol=1e-5, atol=1e-5, equal_nan=False)->bool:
    return np.all(np.logical_or(np.isclose(t, bounds[0], rtol, atol, equal_nan=equal_nan),
                                np.isclose(t, bounds[1], rtol, atol, equal_nan=equal_nan)))
@vectorize(excluded='bounded', signature='(j,i),(j,i)->(j)')
def _line_line_intersection(self: np.ndarray[(2, 3), float], other: np.ndarray[(2, 3), float]):
    (x1, y1, z1), (x2, y2, z2) = self[0], self[1]
    (x3, y3, z3), (x4, y4, z4) = other[0], other[1]
    A = np.array([[x2 - x1, x4 - x3], [y2 - y1, y4 - y3]])
    b = np.array([x3 - x1, y3 - y1])
    if np.linalg.det(A) != 0.0:
        x1, x2 = np.linalg.solve(A, b).flatten()
        return np.array([x1, -x2])
    else:
        return np.array([np.nan, np.nan])


def polyline_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
        Find the intersection points between two polylines.

        :param a: The first polyline.
        :type a: ndarray of shape (num_points, 2)

        :param b: The second polyline.
        :type b: ndarray of shape (num_points, 2)

        :return: An array of intersection points.
        :rtype: ndarray of shape (num_intersections, 2)
    """
    pln, pln2 = polyline_to_lines(a), polyline_to_lines(b)
    l = []
    for i in range(len(pln)):
        res = _line_line_intersection(pln[i], pln2)
        correct = res[
            np.logical_and(np.logical_and(res.T[0] >= -0, res.T[0] <= 1), np.logical_and(res.T[1] >= -0, res.T[1] <= 1)
                    )].flatten()
        if len(correct) > 0:
            l.append(evaluate_line(*pln[i], correct[0]))
    return np.array(l)



def split_polyline(pln, tss):
    """
    .. function:: split_polyline(pln, tss)

        This function splits a polyline into multiple segments based on a list of t values.

        :param pln: The input polyline.
        :type pln: list of tuples

        :param tss: List of t values to split the polyline.
        :type tss: list of floats

        :return: A list of segmented polylines.
        :rtype: list of lists

        .. note:: The input polyline is represented as a list of tuples, where each tuple is a point with x and y
        coordinates. The t values represent the positions along the polyline where
    * the split should occur.

        .. code-block:: python

            pln = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
            tss = [0.3, 0.6]

            result = split_polyline(pln, tss)

        This will split the polyline into three segments: [(0, 0), (0.5, 0.5), (1, 1)], [(1, 1), (2, 2), (2.5, 2.5)],
        and [(2.5, 2.5), (3, 3), (4, 4), (5, 5)].
    """
    n, m = np.divmod(tss, 1)

    n = np.array(n, dtype=int)

    pts = evaluate_line(polyline_to_lines(pln)[n], m)

    _ = list(zip(np.append(np.arange(len(pln)), tss).tolist(), np.append(pln, pts, axis=0).tolist()))
    _.sort(key=lambda x: x[0])
    aa, bb = zip(*_)
    bb = np.array(bb)

    def gen():
        for p, i in itertools.pairwise(tss):
            _i = aa.index(i)
            _p = aa.index(p)
            yield bb[_p:_i + 1].tolist()

    return list(gen())


from scipy.spatial import ConvexHull, KDTree

from mmcore.geom.curves import ParametricPlanarCurve





class PolyLine(ParametricPlanarCurve):
    """
    PolyLine class represents a polyline in a 2D space.

    Usage:
        corners (list or ndarray): List of 2D points representing the corners of the polyline.
        plane (optional, default=WXY): Plane in which the polyline lies.

    Methods:
        evaluate(t):
            Evaluates the polyline at a given parameter value(s) `t` and returns the corresponding 2D points as an
            ndarray.

            Args:
                t (float or ndarray): Parameter value(s) at which to evaluate the polyline.

            Returns:
                ndarray: Array of evaluated 2D points.

        __call__(t):
            Calls the polyline at a given parameter value(s) `t` and returns the corresponding 2D points transformed
            to the world coordinate system.

            Args:
                t (float or ndarray): Parameter value(s) at which to call the polyline.

            Returns:
                ndarray: Array of called 2D points in the world coordinate system.

        chamfer(value):
            Creates a new polyline by adding chamfers to the corners of the current polyline.

            Args:
                value (float or ndarray): Chamfer distance(s) to be added to each corner. Can be a scalar or an ndarray.

            Returns:
                PolyLine: New polyline object with chamfers added to the corners.

        __len__():
            Returns the number of corners in the polyline.

    Attributes:
        corners (ndarray): Array of 2D points representing the corners of the polyline.
        plane (Plane): Plane in which the polyline lies.
    """
    def __new__(cls, corners, plane=WXY):
        self = super().__new__(cls)
        self.plane = plane
        self.corners = np.array(corners)

        return self

    def evaluate(self, t) -> np.ndarray[Any, np.dtype[float]]:
        return evaluate_polyline(self.corners, t)

    def __call__(self, t) -> np.ndarray:
        return local_to_world(evaluate_polyline(self.corners, t), self.plane)

    def chamfer(self, value: 'float|np.ndarray'):
        if np.isscalar(value):
            value = np.zeros(len(self), float) + value

        dists = np.array([value, 1 - value]) + np.tile(np.arange(len(self)), (2, 1))
        res = self(dists.T).flatten()
        lr = len(res)

        return PolyLine(res.reshape((lr // 3, 3)), plane=self.plane)

    def __len__(self):
        return len(self.corners)
