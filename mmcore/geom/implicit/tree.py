from __future__ import annotations

from mmcore.geom.implicit.dc import buildTree, collapse, Empty, Full, Leaf
from mmcore.geom.implicit.implicit import Implicit2D


class ImplicitTree2D:
    def __init__(self, shape: Implicit2D, depth=3):
        self.full = []
        self.empty = []
        self.border = []
        self.depth = depth
        self.shape = shape
        self._tree = None
        self._collapsed_tree = None

    def build_tree(self):
        minb, maxb = self.shape.bounds()
        self._tree = buildTree(minb, maxb, self.depth)

    def collapse(self):
        self._collapsed_tree = collapse(self.shape.implicit, self._tree)

    def clear(self):
        self.full.clear()
        self.empty.clear()
        self.border.clear()

    def build(self, depth=None):
        self.clear()
        if depth is not None:
            self.depth = depth
        self.build_tree()
        self.collapse()
        self.build_node(self._collapsed_tree, self.shape.bounds())

    def build_node(self, tree, quad):
        if isinstance(tree, Empty):
            return self.empty.append(quad)
        elif isinstance(tree, Full):
            self.full.append(quad)
        elif isinstance(tree, Leaf):
            self.border.append(quad)

        else:
            (xmin, ymin), (xmax, ymax) = quad
            xmid = (xmin + xmax) / 2
            ymid = (ymin + ymax) / 2

            self.build_node(tree.a, ((xmin, ymin), (xmid, ymid)))
            self.build_node(tree.b, ((xmid, ymin), (xmax, ymid)))
            self.build_node(tree.c, ((xmin, ymid), (xmid, ymax)))
            self.build_node(tree.d, ((xmid, ymid), (xmax, ymax)))
