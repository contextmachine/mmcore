
import math
from typing import Callable

import numpy as np

import sys

from mmcore.geom.implicit.tree.collapse2d import Point, Shape, Min, Max, Tree, Root, Empty, Full, Leaf, Cell, Side

sys.setrecursionlimit(100000)


def abs_cmp(a, b, op):
    res = op(abs(a), abs(b))
    return a if abs(a) == res else b


def nd_max(a, b):
    a_cond, b_cond = isinstance(a, np.ndarray), isinstance(b, np.ndarray)
    if a_cond and b_cond:
        aa, ba = np.abs(a), np.abs(b)
        a[aa < ba] = b[aa < ba]
        return a
    elif a_cond:
        aa, ba = np.abs(a), abs(b)
        a[aa < ba] = b
        return a
    elif b_cond:
        aa, ba = abs(a), np.abs(b)
        b[ba < aa] = a
        return b
    else:

        return abs_cmp(a, b, max)


def circle(center: Point, r: float) -> Shape:
    def _circle(p: Point) -> float:
        if isinstance(p, np.ndarray) and p.ndim >= 1:
            return (
                math.sqrt(
                    np.sum(
                        np.array(center).reshape(
                            (2, *np.ones((len(p.shape) - 1,), dtype=int))
                        )
                        - p
                    )
                    ** 2
                )
                - r
            )
        else:
            x0, y0 = center
            x, y = p

            return math.sqrt((x0 - x) ** 2 + (y0 - y) ** 2) - r

    return _circle


def left(x0: float) -> Shape:
    def _left(p: Point) -> float:
        x, _ = p
        return x - x0

    return _left


def right(x0: float) -> Shape:
    def _right(p: Point) -> float:
        x, _ = p
        return x0 - x

    return _right


def lower(y0: float) -> Shape:
    def _lower(p: Point) -> float:
        _, y = p
        return y - y0

    return _lower


def upper(y0: float) -> Shape:
    def _upper(p: Point) -> float:
        _, y = p
        return y0 - y

    return _upper


def union(a: Shape, b: Shape) -> Shape:
    def _union(p: Point) -> float:
        return opUnion(a(p), b(p))

    return _union


def intersection(a: Shape, b: Shape) -> Shape:
    def _intersection(p: Point) -> float:
        return opIntersection(a(p), b(p))

    return _intersection


def inv(a: Shape) -> Shape:
    def _inv(p: Point) -> float:
        return -1 * (a(p))

    return _inv


def substract(a: Shape, b: Shape) -> Shape:
    def wrap(p: Point) -> float:
        return opSubtraction(b(p), a(p))

    return wrap


def rectangle(min: Min, max: Max) -> Shape:
    xmin, ymin = min
    xmax, ymax = max
    return intersection(
        right(xmin), intersection(left(xmax), intersection(upper(ymin), lower(ymax)))
    )


def hi() -> Shape:
    h = test4()

    i = union(rectangle((0.75, 0.1), (0.9, 0.55)), circle((0.825, 0.75), 0.1))

    return union(h, i)


def test1():
    return union(circle((0.35, 0.35), 0.25), rectangle((0.1, 0.1), (0.6, 0.35)))


def test2():
    return union(circle((0.35, 0.35), 0.1), rectangle((0.25, 0.05), (0.45, 0.35)))


def test3():
    return substract(test1(), test2())


def test4():
    return union(test3(), rectangle((0.1, 0.1), (0.25, 0.9)))


def opUnion(d1, d2):
    return min(d1, d2)


def opSubtraction(d1, d2):
    return max(-d1, d2)


def opIntersection(d1, d2):
    return max(d1, d2)


def opXor(d1, d2):
    return max(min(d1, d2), -max(d1, d2))


def trav(
    tree: Tree,
    cb: Callable[[Tree, Tree], None],
    fltr: Callable[[Tree, Tree], bool] = lambda prnt, ch: isinstance(ch, Leaf),
) -> None:

    for ch in tree.children:
        if fltr(tree, ch):
            cb(tree, ch)
        else:
            trav(ch, cb)
def foldTree(tree: Tree, quad: Cell, root, empty, full, leaf) -> float:
    if isinstance(tree, Root):
        (xmin, ymin), (xmax, ymax) = quad
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2

        return (
            root(quad)
            + foldTree(tree.a, ((xmin, ymin), (xmid, ymid)), root, empty, full, leaf)
            + foldTree(tree.b, ((xmid, ymin), (xmax, ymid)), root, empty, full, leaf)
            + foldTree(tree.c, ((xmin, ymid), (xmid, ymax)), root, empty, full, leaf)
            + foldTree(tree.d, ((xmid, ymid), (xmax, ymax)), root, empty, full, leaf)
        )

    elif isinstance(tree, Empty):
        return empty(quad)
    elif isinstance(tree, Full):
        return full(quad)
    elif isinstance(tree, Leaf):
        return leaf(quad)


def index(shape: Shape, cell: Cell) -> int:
    (xmin, ymin), (xmax, ymax) = cell
    pts = [(x, y) for y in [ymin, ymax] for x in [xmin, xmax]]

    res = sum(2 ** (3 - i) if shape(pt) < 0 else 0 for i, pt in enumerate(pts))

    print(pts, res)
    return res


def edges(shape: Shape, cell: Cell) -> list[tuple[Side, Side]]:
    return lut[index(shape, cell)]


def pt(shape: Shape, cell: Cell, side: Side) -> Point:
    (xmin, ymin), (xmax, ymax) = cell

    return {
        Side.Left: lambda: zero(shape, (xmin, ymin), (xmin, ymax)),
        Side.Right: lambda: zero(shape, (xmax, ymin), (xmax, ymax)),
        Side.Lower: lambda: zero(shape, (xmin, ymin), (xmax, ymin)),
        Side.Upper: lambda: zero(shape, (xmin, ymax), (xmax, ymax)),
    }[side]()


def zero(shape: Shape, a: Point, b: Point) -> Point:
    ax, ay = a
    bx, by = b

    if shape(a) >= 0:
        return zero(shape, b, a)
    else:

        def zero_(f: float, step: float, i: int) -> Point:
            if i == 0:
                return pos(f)
            else:
                if shape(pos(f)) < 0:
                    return zero_(f + step, step / 2, i - 1)
                else:
                    return zero_(f - step, step / 2, i - 1)

        def pos(f: float) -> Point:
            x = ax * (1 - f) + bx * f
            y = ay * (1 - f) + by * f
            return x, y

        return zero_(0.5, 0.25, 10)


def contours(shape, cell):
    # Acquire edges from current cell
    edges_ = edges(shape, cell)
    # Calculate points for each edge and store in a list
    contours_ = [(pt(shape, cell, a), pt(shape, cell, b)) for a, b in edges_]
    # Return the final list of tuples representing contours
    return contours_


def merge(shape: Shape, tree: Tree) -> Tree:
    def _merge(a: Tree, b: Tree, c: Tree, d: Tree) -> Tree:
        if (
            isinstance(a, Leaf)
            and isinstance(b, Leaf)
            and isinstance(c, Leaf)
            and isinstance(d, Leaf)
        ):
            min, _ = a.data
            _, max = d.data
            _, i = a.data
            q, r = b.data
            s, t = c.data

            scores = [score(shape, (min, max), p) for p in [i, q, r, s, t]]

            if all(s < 0.001 for s in scores):
                return Leaf((min, max))
            else:
                return Root(a, b, c, d)
        else:
            return Root(a, b, c, d)

    if isinstance(tree, Root):
        a = merge(shape, tree.a)
        b = merge(shape, tree.b)
        c = merge(shape, tree.c)
        d = merge(shape, tree.d)
        return _merge(a, b, c, d)
    else:
        return tree


def interpolate(shape: Shape, cell: Cell, p: Point) -> float:
    (xmin, ymin), (xmax, ymax) = cell
    x, y = p

    dx = (x - xmin) / (xmax - xmin)
    dy = (y - ymin) / (ymax - ymin)

    ab = shape((xmin, ymin)) * (1 - dx) + shape((xmax, ymin)) * dx
    cd = shape((xmin, ymax)) * (1 - dx) + shape((xmax, ymax)) * dx

    return ab * (1 - dy) + cd * dy


def score(shape: Shape, cell: Cell, pt: Point) -> float:
    return abs(interpolate(shape, cell, pt) - shape(pt))


def deriv(shape: Shape, p: Point) -> Point:
    x, y = p
    epsilon = 0.001

    dx = shape((x + epsilon, y)) - shape((x - epsilon, y))
    dy = shape((x, y + epsilon)) - shape((x, y - epsilon))

    length = math.sqrt(dx**2 + dy**2)
    return dx / length, dy / length


def feature(shape, cell):
    pts_ = sum(contours(shape, cell), ())
    if len(pts_) >= 2:
        from_tuple = lambda x: np.array([x[0], x[1]])
        pts = list(map(from_tuple, pts_))
        nms = list(map(from_tuple, [deriv(shape, pt) for pt in pts]))
        center = sum(pts) / len(pts)

        a = np.stack(nms)
        b = np.array([(pt - center).dot(nm) for pt, nm in zip(pts, nms)]).reshape(-1, 1)

        p = center + np.linalg.lstsq(a, b, rcond=None)[0].ravel()
        return p[0], p[1]
    else:
        return None


Edge = tuple[Point, Point]


def dc(shape: Shape, tree: Tree) -> list[Edge]:
    return faceProc(shape, tree)


def faceProc(shape, t):
    # TODO Черти что тут пока происходит
    if isinstance(t, Root):
        a, b, c, d = t.a, t.b, t.c, t.d
        return (
            sum(map(lambda x: faceProc(shape, x), [a, b, c, d]), [])
            + edgeProcH(shape, a, b)
            + edgeProcH(shape, c, d)
            + edgeProcV(shape, a, c)
            + edgeProcV(shape, b, d)
        )
    else:
        return []


def edgeProcH(shape, tree1, tree2):
    if isinstance(tree1, Leaf) and isinstance(tree2, Leaf):
        return [(feature(shape, tree1.data), feature(shape, tree2.data))]
    elif isinstance(tree1, Leaf) and isinstance(tree2, Root):
        a, _, c, _ = tree2.a, tree2.b, tree2.c, tree2.d
        return edgeProcH(shape, tree1, a) + edgeProcH(shape, tree1, c)
    elif isinstance(tree1, Root) and isinstance(tree2, Leaf):
        _, b, _, d = tree1.a, tree1.b, tree1.c, tree1.d
        return edgeProcH(shape, b, tree2) + edgeProcH(shape, d, tree2)
    elif isinstance(tree1, Root) and isinstance(tree2, Root):
        _, b, _, d = tree1.a, tree1.b, tree1.c, tree1.d
        a, _, c, _ = tree2.a, tree2.b, tree2.c, tree2.d
        return edgeProcH(shape, b, a) + edgeProcH(shape, d, c)
    else:
        return []


def edgeProcV(shape, tree1, tree2):
    if isinstance(tree1, Leaf) and isinstance(tree2, Leaf):
        return [(feature(shape, tree1.data), feature(shape, tree2.data))]
    elif isinstance(tree1, Leaf) and isinstance(tree2, Root):
        _, _, c, d = tree2.a, tree2.b, tree2.c, tree2.d
        return edgeProcV(shape, tree1, c) + edgeProcV(shape, tree1, d)
    elif isinstance(tree1, Root) and isinstance(tree2, Leaf):
        a, b, _, _ = tree1.a, tree1.b, tree1.c, tree1.d
        return edgeProcV(shape, a, tree2) + edgeProcV(shape, b, tree2)
    elif isinstance(tree1, Root) and isinstance(tree2, Root):
        a, b, _, _ = tree1.a, tree1.b, tree1.c, tree1.d
        _, _, c, d = tree2.a, tree2.b, tree2.c, tree2.d
        return edgeProcV(shape, a, c) + edgeProcV(shape, b, d)
    else:
        return []


lut = [
    [],  # 0000
    [(Side.Upper, Side.Right)],  # 000d
    [(Side.Left, Side.Upper)],  # 00c0
    [(Side.Left, Side.Right)],  # 00cd
    [(Side.Right, Side.Lower)],  # 0b00
    [(Side.Upper, Side.Lower)],  # 0b0d
    [(Side.Right, Side.Lower), (Side.Left, Side.Upper)],  # 0bc0
    [(Side.Left, Side.Lower)],  # 0bcd
    [(Side.Lower, Side.Left)],  # a000
    [(Side.Lower, Side.Left), (Side.Upper, Side.Right)],  # a00d
    [(Side.Lower, Side.Upper)],  # a0c0
    [(Side.Lower, Side.Right)],  # a0cd
    [(Side.Right, Side.Left)],  # ab00
    [(Side.Upper, Side.Left)],  # ab0d
    [(Side.Right, Side.Upper)],  # abc0
    [],
]  # abcd
