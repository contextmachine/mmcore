# mmcore/geom/implicit/dc.py
import sys
from typing import Callable

import numpy as np
from .cbuild_tree3d import build_tree3d,Full,Empty,Tree,Root,Leaf

sys.setrecursionlimit(100000)
Point = tuple[float, float, float]
Shape = Callable[[Point], float]

Min = Point
Max = Point


Cell = tuple[Min, Max]

"""
class Tree:
    def __init__(self):
        pass

    @property
    def children(self):
        return ()






class Root(Tree):
    __match_args__ = tuple("a,b,c,d,e,f,g,h".split(","))
    __slots__ = ("a", "b", "c", "d", "e", "f", "g", "h",)
    def __init__(self, a, b, c, d, e, f, g, h):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.h = h

    @property
    def children(self):
        return self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h


class Empty(Tree):
    __slots__ = ("node",)
    def __init__(self, node: Tree = None):
        super().__init__()
        self.node = node

    @property
    def children(self):
        return (self.node,)



class Full(Tree):
    __slots__ = ("node",)
    def __init__(self, node: Tree = None):
        super().__init__()
        self.node = node

    @property
    def children(self):
        return (self.node,)



class Leaf(Tree):
    __match_args__ = ("data",)
    __slots__ = ("data",)

    def __init__(self, data):
        super().__init__()
        self.data = data

    @property
    def children(self):
        return ()


def build_tree3d(min: Min, max: Max, i: int) -> Tree:
    if i == 0:

        return Leaf((min, max))

    xmin, ymin, zmin = min
    xmax, ymax, zmax = max
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    zmid = (zmin + zmax) / 2

    a = build_tree3d((xmin, ymin, zmin), (xmid, ymid, zmid), i - 1)
    b = build_tree3d((xmid, ymin, zmin), (xmax, ymid, zmid), i - 1)
    c = build_tree3d((xmin, ymid, zmin), (xmid, ymax, zmid), i - 1)
    d = build_tree3d((xmid, ymid, zmin), (xmax, ymax, zmid), i - 1)
    e = build_tree3d((xmin, ymin, zmid), (xmid, ymid, zmax), i - 1)
    f = build_tree3d((xmid, ymin, zmid), (xmax, ymid, zmax), i - 1)
    g = build_tree3d((xmin, ymid, zmid), (xmid, ymax, zmax), i - 1)
    h = build_tree3d((xmid, ymid, zmid), (xmax, ymax, zmax), i - 1)


    return Root(a, b, c, d, e, f, g, h)
"""


def collapse3d(shape: Shape, tree: Tree) -> Tree:
    """

    :param shape:
    :param tree:
    :return:

    Example
    ----



    """

    def _collapse(leaf: Leaf) -> Tree:
        (xmin, ymin, zmin), (xmax, ymax, zmax) = leaf.data
        values = [shape(np.array((x, y, z))) for x in [xmin, xmax] for y in [ymin, ymax] for z in [zmin, zmax]]

        if all(v < 0 for v in values):
            return Full(leaf)
        elif all(v >= 0 for v in values):
            return Empty(leaf)
        else:
            return leaf

    def _collapse_root(root: Root) -> Tree:
        a = collapse3d(shape, root.a)
        b = collapse3d(shape, root.b)
        c = collapse3d(shape, root.c)
        d = collapse3d(shape, root.d)
        e = collapse3d(shape, root.e)
        f = collapse3d(shape, root.f)
        g = collapse3d(shape, root.g)
        h = collapse3d(shape, root.h)

        if (
            isinstance(a, Empty)
            and isinstance(b, Empty)
            and isinstance(c, Empty)
            and isinstance(d, Empty)
            and isinstance(e, Empty)
            and isinstance(f, Empty)
            and isinstance(g, Empty)
            and isinstance(h, Empty)
        ):
            return Empty(root)
        elif (
            isinstance(a, Full)
            and isinstance(b, Full)
            and isinstance(c, Full)
            and isinstance(d, Full)
            and isinstance(e, Full)
            and isinstance(f, Full)
            and isinstance(g, Full)
            and isinstance(h, Full)
        ):
            return Full(root)
        else:


            return Root(a, b, c, d, e, f, g, h)

    if isinstance(tree, Leaf):
        return _collapse(tree)
    elif isinstance(tree, Root):
        return _collapse_root(tree)
    else:
        return tree

def collapse_adaptive(shape: Shape, tree: Leaf, min_size=1.) -> Root:
    bt=build_tree3d(tree.data[0],tree.data[1], 3)
    for child in bt.children:
        child




__all__ = ['build_tree3d','collapse3d']