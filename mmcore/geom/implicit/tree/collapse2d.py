from enum import Enum, auto
from typing import Callable


Point = tuple[float, float]
Shape = Callable[[Point], float]
Min = Point
Max = Point


class Tree:
    def __init__(self):
        pass

    @property
    def children(self):
        return ()

    def reduce(self): ...


class Root(Tree):
    __match_args__ = tuple("a,b,c,d".split(","))

    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    @property
    def children(self):
        return self.a, self.b, self.c, self.d

    def reduce(self):
        return Root(self.a.reduce(), self.b.reduce(), self.c.reduce(), self.d.reduce())


class Empty(Tree):
    def __init__(self, node: Tree = None):
        self.node = node

    @property
    def children(self):
        return (self.node,)

    def reduce(self):
        return self.node.reduce()


class Full(Tree):
    def __init__(self, node: Tree = None):
        self.node = node

    @property
    def children(self):
        return (self.node,)

    def reduce(self):

        return self.node.reduce()


class Leaf(Tree):
    __match_args__ = ("data",)

    def __init__(self, data):
        self.data = data

    @property
    def children(self):
        return ()

    def reduce(self):
        return self


Cell = tuple[Min, Max]


def build_tree2d(min: Min, max: Max, i: int) -> Tree:
    if i == 0:
        return Leaf((min, max))

    xmin, ymin = min
    xmax, ymax = max
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2

    a = build_tree2d((xmin, ymin), (xmid, ymid), i - 1)
    b = build_tree2d((xmid, ymin), (xmax, ymid), i - 1)
    c = build_tree2d((xmin, ymid), (xmid, ymax), i - 1)
    d = build_tree2d((xmid, ymid), (xmax, ymax), i - 1)

    return Root(a, b, c, d)


def collapse2d(shape: Shape, tree: Tree) -> Tree:
    """

    :param shape:
    :param tree:
    :return:

    Example
    ----

    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.patches as patches
    >>> from mmcore.geom.implicit.dc import *

    >>> def drawLeaf(color, cell):
    ...     (xmin, ymin), (xmax, ymax) = cell
    ...     rect = patches.Rectangle((xmin, ymin), xmax - xmin - 0.005, ymax - ymin - 0.005,
    ...                              linewidth=1, edgecolor=color, facecolor='none')
    ...     return rect

    >>> def draw(res):
    ...     fig, ax = plt.subplots()
    ...
    ...     def draw_inner(tree:Root, quad):
    ...         if isinstance(tree, Empty):
    ...             return
    ...         elif isinstance(tree, Full):
    ...             ax.add_patch(drawLeaf('black', quad))
    ...         elif isinstance(tree, Leaf):
    ...             ax.add_patch(drawLeaf('lightgreen', quad))
    ...         else:
    ...             (xmin, ymin), (xmax, ymax) = quad
    ...             xmid = (xmin + xmax) / 2
    ...             ymid = (ymin + ymax) / 2
    ...
    ...             draw_inner(tree.a, ((xmin, ymin), (xmid, ymid)))
    ...             draw_inner(tree.b, ((xmid, ymin), (xmax, ymid)))
    ...             draw_inner(tree.c, ((xmin, ymid), (xmid, ymax)))
    ...             draw_inner(tree.d, ((xmid, ymid), (xmax, ymax)))
    ...     _tree=build_tree2d((0, 0), (1, 1), res) # Построение дерева, res - резолюшн, сколько раз будет делиться QuadTree
    ...     _tree = collapse2d(hi(), _tree)   # collapse, перестраивает дерево под шейп. hi - sdf шейп

    ...     draw_inner(tree, ((0, 0), (1, 1)))
    ...     ax.set_aspect('equal')
    ...     ax.axis('off')
    ...     plt.tight_layout()
    ...     plt.show()
    ...

    >>> for res in range(1, 6):
    ...     draw(res)

    """

    def _collapse(leaf: Leaf) -> Tree:
        (xmin, ymin), (xmax, ymax) = leaf.data
        values = [shape((x, y)) for x in [xmin, xmax] for y in [ymin, ymax]]

        if all(v < 0 for v in values):
            return Full(leaf)
        elif all(v >= 0 for v in values):
            return Empty(leaf)
        else:
            return leaf

    def _collapse_root(root: Root) -> Tree:
        a = collapse2d(shape, root.a)
        b = collapse2d(shape, root.b)
        c = collapse2d(shape, root.c)
        d = collapse2d(shape, root.d)

        if (
            isinstance(a, Empty)
            and isinstance(b, Empty)
            and isinstance(c, Empty)
            and isinstance(d, Empty)
        ):
            return Empty(root)
        elif (
            isinstance(a, Full)
            and isinstance(b, Full)
            and isinstance(c, Full)
            and isinstance(d, Full)
        ):
            return Full(root)
        else:
            return Root(a, b, c, d)

    if isinstance(tree, Leaf):
        return _collapse(tree)
    elif isinstance(tree, Root):
        return _collapse_root(tree)
    else:
        return tree


class Side(Enum):
    Upper = auto()
    Lower = auto()
    Left = auto()
    Right = auto()


__all__ = ['build_tree2d','collapse2d']