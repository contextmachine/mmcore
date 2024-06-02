from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from mmcore.geom.implicit.tree.collapse2d import Empty, Full, Leaf, build_tree2d, collapse2d


class ImplicitTree2D:
    def __init__(self, shape: 'Implicit2D', depth=3, bounds=None):
        self.full = []
        self.empty = []
        self.border = []
        self.depth = depth
        self.shape = shape
        self._tree = None
        self._collapsed_tree = None
        self.bounds = getattr(self.shape, "bounds", lambda: ((-5., -5.), (5., 5.)))() if bounds is None else bounds

    def build_tree(self):
        minb, maxb = self.bounds
        self._tree = build_tree2d(minb, maxb, self.depth)

    def collapse(self):
        self._collapsed_tree = collapse2d(getattr(self.shape, 'implicit', self.shape), self._tree)

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


def _find_feature(sdfs, node, node_norm):
    def f(x):
        return max(abs(sdfs[0].implicit(x)), abs(sdfs[1].implicit(x)))

    x0 = np.average(node, axis=0)
    if node_norm < abs(sdfs[0].implicit(x0)):
        return False, None
    if node_norm < abs(sdfs[1].implicit(x0)):
        return False, None
    else:
        res = minimize(
            f, x0=x0, bounds=((node[0][0], node[1][0]), (node[0][1], node[1][1]))
        )

        return np.isclose(res.fun, 0), res


def implicit_find_features(sdfs, nodes, rtol=None, atol=None):
    node = nodes[0]
    node_norm = np.linalg.norm([node[1][0] - node[0][0], node[1][1] - node[0][1]])
    kws = dict(

    )
    if atol:
        kws['atol'] = atol
    if rtol:
        kws['rtol'] = atol
    last = None
    for node in nodes:
        success, res = _find_feature(sdfs, node, node_norm)

        if success:

            if last is not None and np.allclose(last.x, res.x, **kws):
                continue
            last = res
            yield res.x
