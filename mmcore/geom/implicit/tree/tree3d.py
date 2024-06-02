from __future__ import annotations

from mmcore.geom.implicit.tree.collapse3d import Empty, Full, Leaf, build_tree3d, collapse3d



class ImplicitTree3D:
    def __init__(self, shape: 'Implicit3D', depth=3, bounds=None):
        self.full = []
        self.empty = []
        self.border = []
        self.depth = depth
        self.shape = shape
        self._tree = None
        self._collapsed_tree = None
        self.bounds = getattr(self.shape,"bounds", lambda :( (-5,-5,-5),(5,5,5.)))() if bounds is None else bounds


    def build_tree(self):
        minb, maxb = self.bounds
        self._tree = build_tree3d(minb, maxb, self.depth)

    def collapse(self):
        self._collapsed_tree = collapse3d(getattr(self.shape, 'implicit', self.shape), self._tree)

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
        self.build_node(self._collapsed_tree, self.bounds)

    def build_node(self, tree, quad):
        if isinstance(tree, Empty):
            return self.empty.append(quad)
        elif isinstance(tree, Full):
            self.full.append(quad)
        elif isinstance(tree, Leaf):
            self.border.append(quad)

        else:
            (xmin, ymin,zmin), (xmax, ymax,zmax) = quad
            xmid = (xmin + xmax) / 2
            ymid = (ymin + ymax) / 2
            zmid = (zmin + zmax) / 2
       
            self.build_node(tree.a, ((xmin, ymin, zmin), (xmid, ymid, zmid)))
            self.build_node(tree.b, ((xmid, ymin, zmin), (xmax, ymid, zmid)))
            self.build_node(tree.c, ((xmin, ymid, zmin), (xmid, ymax, zmid)))
            self.build_node(tree.d, ((xmid, ymid, zmin), (xmax, ymax, zmid)))
            self.build_node(tree.e, ((xmin, ymin, zmid), (xmid, ymid, zmax)))
            self.build_node(tree.f, ((xmid, ymin, zmid), (xmax, ymid, zmax)))
            self.build_node(tree.g, ((xmin, ymid, zmid), (xmid, ymax, zmax)))
            self.build_node(tree.h, ((xmid, ymid, zmid), (xmax, ymax, zmax)))

