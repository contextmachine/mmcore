import itertools
import numpy as np

from mmcore.func import vectorize
from mmcore.geom.line.cdll_line import LineCDLL, LineNode
from mmcore.geom.line import evaluate_line, Line
from mmcore.geom.plane import WXY, world_to_local, local_to_world



@vectorize(signature='(i,j),()->(j)')
def evaluate_polyline(corners, t: float):
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


def polyline_intersection(a, b):
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


from scipy.spatial import KDTree

from mmcore.geom.curves import ParametricPlanarCurve


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
    Example:
    >>> # Creating PolyCurve with points
    >>> from mmcore.geom.polyline import PolyCurve
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






















poly.insert_corner(pt)
poly.corners
    """
    nodetype = LineNode

    def __init__(self, pts=None):
        super().__init__()
        self._kd = None
        if pts is not None:
            lines = polyline_to_lines(np.array(pts, float))
            for line in lines:
                self.append((Line.from_ends(*line)))

    @classmethod
    def from_points(cls, pts):
        lcdll = cls()

        lines = polyline_to_lines(np.array(pts, float))
        for line in lines:

            lcdll.append(Line.from_ends(*line))

        return lcdll

    def solve_kd(self):
        return KDTree(self.corners)

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

    @property
    def corners(self):

        return np.array([list(i) for i in self.gen_intersects()])

    @corners.setter
    def corners(self, corners):
        for corner, node in itertools.zip_longest(corners, self.iter_nodes(), fillvalue=None):

            if corner is None:
                break
            elif node is None:

                self.insert_corner(corner)
            elif np.allclose(corner, node.start):
                pass
            else:
                node.start = np.array(corner)
                node.previous.end = np.array(corner)


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

    def evaluate(self, t) -> np.ndarray:
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
