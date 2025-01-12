import numpy as np
from .utils import *
import sys

sys.setrecursionlimit(100000)
STEP = (0.6, 0.6, 0.6)

MULTIPLY = 6

transform_sine = lambda pts, p1, p2: np.sin(pts * p1) * p2


class OCNode:
    __slots__ = ["min", "max", "parent", "child", "dims", "center", "_lamp"]

    def __init__(self, min_max, parent=None):
        self.min, self.max = min_max[0], min_max[1]

        self.parent = parent
        self.child = []
        self.dims = round(min_max[1][0] - min_max[0][0], 4), round(min_max[1][1] - min_max[0][1], 4), round(
            min_max[1][2] - min_max[0][2], 4)
        self.center = self.min[0] + self.dims[0] / 2, self.min[1] + self.dims[1] / 2, self.min[2] + self.dims[2] / 2

        self._lamp = None

    @property
    def lamp(self):
        return self._lamp

    @lamp.setter
    def lamp(self, v):
        if v:
            #x_shift = transform_sine(v[1], MULTIPLY, X_COEFFICIENT)
            #y_shift = transform_sine(v[0], MULTIPLY, Y_COEFFICIENT)
            #z_shift = transform_sine(v[0], MULTIPLY, Z_COEFFICIENT)

            #x,y,z = [v[0]+x_shift, v[1]+y_shift, v[2]+z_shift]
            x, y, z = [v[0], v[1], v[2]]
            try:
                self._lamp = [[xx, yy, zz] for xx, yy, zz in zip(x.tolist(), y.tolist(), z.tolist())]
            except:
                self._lamp = [x, y, z]
        else:
            self._lamp = v

    def create_children(self, count=2):
        self._lamp = None

        if count % 2 == 0:
            for i in divide_xyz(self.min, self.max, self.center):
                node = OCNode(i, parent=self)
                node.lamp = node.center
                self.child.append(node)

    def overlap(self, min_max) -> bool:

        d1x = min_max[0][0] - self.max[0]
        d1y = min_max[0][1] - self.max[1]
        d1z = min_max[0][2] - self.max[2]

        d2x = self.min[0] - min_max[1][0]
        d2y = self.min[1] - min_max[1][1]
        d2z = self.min[2] - min_max[1][2]

        if d1x > 0.0 or d1y > 0.0 or d1z > 0.0:
            return False

        if d2x > 0.0 or d2y > 0.0 or d2z > 0.0:
            return False

        return True

    def in_bbox(self, min_max) -> bool:
        low, high = min_max[0], min_max[1]
        return (low[0] <= self.min[0]
                and low[1] <= self.min[1]
                and low[2] <= self.min[2]
                and high[0] >= self.max[0]
                and high[1] >= self.max[1]
                and high[2] >= self.max[2])

    def contains_bbox(self, min_max) -> bool:
        low, high = min_max[0], min_max[1]
        return (self.min[0] <= low[0]
                and self.min[1] <= low[1]
                and self.min[2] <= low[2]
                and self.max[0] >= high[0]
                and self.max[1] >= high[1]
                and self.max[2] >= high[2])

    def contains_point(self, point: tuple[float, float, float]):
        return self.max[0] >= point[0] >= self.min[0] and self.max[1] >= point[1] >= self.min[1] and self.max[2] >= \
            point[2] >= self.min[2]

    def __contains__(self, obj: tuple[float, float, float]
                                | tuple[tuple[float, float, float],
    tuple[float, float, float]]) -> bool:

        if isinstance(obj[0], (float, int)):
            return self.contains_point(obj)
        elif isinstance(obj[0], (tuple, list, np.ndarray)):
            return self.contains_bbox(obj)

    def to_intervals(self):
        return zip(self.min, self.max)

    def __repr__(self):
        s = ', '.join(f'{start} to {end}' for start, end in self.to_intervals())
        return f'{self.__class__.__name__}({s})'


def closest_node(head: OCNode, point: tuple[float, float, float]) -> OCNode:
    if point in head:
        if len(head.child) > 0:
            for child in head.child:
                res = closest_node(child, point)
                if res is not None:
                    return res

        else:
            return head




def traverse(head: OCNode):
    if head:
        if head.lamp:
            head.create_children()
        else:
            [traverse(i) for i in head.child]

def create_octree(root, target=STEP):
    # target - resulting bbox dims

    if all((root.dims[0] <= target[0], root.dims[1] <= target[1], root.dims[2] <= target[2])):
        root.lamp = root.center
        return
    else:
        arr = find_division_option(root.min, root.max, root.center, root.dims, target)
        for i in arr:
            node = OCNode(i, parent=root)
            root.child.append(node)
            create_octree(node, target)




if __name__ == '__main__':
    import numpy as np

    import time

    s = time.time()
    head = OCNode([[0, 0, 0], [38.4, 38.4, 38.4]])
    create_octree(head, (0.6, 0.6, 0.6))
    print(divmod(time.time() - s, 60))
