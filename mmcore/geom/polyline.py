import itertools
import numpy as np

from mmcore.func import vectorize
from mmcore.geom.line.cdll_line import LCDLL, LineNode
from mmcore.geom.line import evaluate_line, Line
from mmcore.geom.plane import WXY, world_to_local, local_to_world



@vectorize(signature='(i,j),()->(j)')
def evaluate_polyline(corners, t: float):
    n, m = np.divmod(t, 1)

    lines = polyline_to_lines(corners)

    return evaluate_line(lines[int(np.mod(n, len(lines))), 0], lines[int(np.mod(n, len(lines))), 1], m)


@vectorize(signature='(i,j)->(i,k,j)')
def polyline_to_lines(pln_points):
    return np.stack([pln_points, np.roll(pln_points, -1, axis=0)], axis=1)


@vectorize(signature='(i,j),(),()->(k,j)')
def trim_polyline(pln, t1, t2):
    n, m = np.divmod([t1, t2], 1)
    n = np.array(n, dtype=int)
    p1, p2 = evaluate_line(polyline_to_lines(pln)[n], m)
    return np.concatenate([[p1], pln[n[0] + 1:n[1] + 1], [p2]])


@vectorize(signature='(i,j),(u),(u,j)->(k,j)')
def insert_polyline_points(pln, ixs, pts):
    return np.insert(pln, ixs, pts, axis=0)


@vectorize(signature='(i,j),(),()->(k,j),(u,j)')
def split_closed_polyline(pln, t1, t2):
    n, m = np.divmod([t1, t2], 1)

    n = np.array(n, dtype=int)
    p1, p2 = evaluate_line(polyline_to_lines(pln)[n], m)

    pln = np.insert(pln, [n[0] + 1, n[1] + 1], [p1, p2], axis=0)
    return split_closed_polyline_by_points(pln, n[0] + 1, n[1] + 2)


def split_closed_polyline_by_points(pln, i, j):
    return np.roll(pln, -i, axis=0)[:(j - i) + 1], np.roll(pln, -j, axis=0)[:pln.shape[0] - (j - i) + 1]


def split_polyline_by_point(pln, i):
    return pln[i:], pln[:i + 1]


def split_polyline(pln, tss):
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


from mmcore.geom.curves import ParametricPlanarCurve


class CDLLPolyLine(LCDLL):
    nodetype = LineNode

    def __init__(self, pts=None):
        super().__init__()
        if pts is not None:
            lines = polyline_to_lines(np.array(pts, float))
            for line in lines:
                self.append((Line.from_ends(*line)))

    @classmethod
    def from_points(cls, pts):
        lcdll = cls()

        lines = polyline_to_lines(np.array(pts, float))
        for line in lines:

            lcdll.append((Line.from_ends(*line)))

        return lcdll
class PolyLine(ParametricPlanarCurve):

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
