import numpy as np
from more_itertools import flatten

from mmcore.base import ALine
from mmcore.geom.parametric import Linear

[(-208372.87900099999, -14759.245091000001, 0.0), (-188554.34088413313, -15118.908494768657, 0.0),
 (-186299.52990600001, -31025.609608259147, 0.0), (-213065.70753399999, -35086.371292000003, 0.0)]

A = [(0.0, 0.0, 0.0), (-228.69999999999527, 0.0, -7.7444497653168538e-16),
     (-28.699999999994645, 600.0, 3.3107987928753852e-12), (1.1368683772161603e-13, 600.0, 0.0), (0.0, 0.0, 0.0)]
A_M = [(1.7053025658242404e-13, 600.0, 5.1939252519446172e-12), (28.700000000002149, 600.0, 1.3321254830884726e-12),
       (228.70000000000255, 0.0, 5.1939252519446172e-12), (0.0, 0.0, 5.1939252519446172e-12),
       (1.7053025658242404e-13, 600.0, 5.1939252519446172e-12)]
B = [(1.1121695195015491e-05, 599.99996062170385, -2.9409505979572262e-11),
     (228.70001112169768, 599.99996062170385, 1.3605471925188766e-12),
     (28.700011121697059, -3.9378296094128018e-05, -2.9413058693251062e-11),
     (1.1121695081328653e-05, -3.9378296094128018e-05, -2.9409505979572262e-11),
     (1.1121695195015491e-05, 599.99996062170385, -2.9409505979572262e-11)]
B_M = [(6.2576873460784554e-06, -5.4368865676224232e-07, -4.7683897719252855e-08),
       (-28.699993742307356, -4.9807249524747021e-07, -4.3683159211595397e-08),
       (-228.69999278865993, 599.99999981981011, 2.6503379044656584e-08),
       (7.2113353439817729e-06, 599.99999945631134, -5.3506888836804093e-09),
       (6.2576873460784554e-06, -5.4368865676224232e-07, -4.7683897719252855e-08)]
Base = [(-199.99997631578185, 599.99997641116977, 6.8553159354678147e-13),
        (200.00001112168843, 599.99996062170487, 1.9787193726301638e-12),
        (-6.2812550822854973e-06, -1.5789464896442951e-05, 6.7501555699773911e-13),
        (-199.99997631578185, 599.99997641116977, 6.8553159354678147e-13)]
Base_M = [(199.99997003452677, 7.799365334903996e-06, -3.410609191441363e-15),
          (-200.00001740295068, 2.3588830288190366e-05, -6.8553163552113754e-13), (1.1368683772161603e-13, 600.0, 0.0),
          (199.99997003452677, 7.799365334903996e-06, -3.410609191441363e-15)]


class Panel:
    def __init__(self, boundary=None, *args, **kwargs):
        super().__init__()
        self._plane = PlaneLinear([0, 0, 0], normal=[0, 0, 1])
        self.matrix = Transform()
        self.__call__(boundary, **kwargs)

    @property
    def plane(self):

        return PlaneLinear(self._plane.origin @ self.matrix,
                           normal=self._plane.normal.tolist() @ self.matrix,
                           xaxis=self._plane.xaxis.tolist() @ self.matrix
                           )

    @plane.setter
    def plane(self, v):
        self.matrix = v.transform_from_other(
            self._plane
        )

    def __call__(self, boundary=None, **kwargs):
        if boundary is not None:
            self.boundaries = [boundary]
        for k in kwargs.keys():
            if kwargs[k] is not None:
                setattr(self, k, kwargs[k])

        return self

    @property
    def boundaries(self):
        bnds = []
        for bound in self._boundaries:
            bnds.append(list(map(lambda x: (x @ self.matrix).tolist(), bound)))
        return bnds

    @boundaries.setter
    def boundaries(self, v):

        self._boundaries = v

    def centroid(self):

        *flat_bounds, = flatten(self.boundaries)
        return np.mean(flat_bounds, axis=0)

    def is_compound(self):
        return not len(self.boundaries) == 1

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, v):
        self._matrix = v

    def transform(self, m):
        self._matrix = self._matrix @ m

    def explode(self):
        l = []
        for bound in self.boundaries:
            l.append(Panel(boundary=bound))
        return l


from mmcore.geom.transform import Transform


class Triang:
    geom = A

    def __init__(self, plane, pos=None):
        self.next = None
        self.prev = None
        self.pos = pos

        self.plane = plane

        self.shape = Panel(boundary=self.geom)
        self.shape.transform(self.transformation)

    @property
    def transformation(self):
        tr = Transform.from_world_to_plane(self.plane)
        return tr

    @property
    def prev(self):
        return self._prev

    @prev.setter
    def prev(self, s):
        self._prev = s


class Base_cl(Triang):
    geom = Base
    left = 400 / 2

    def __init__(self, plane=None, pos=None, leftover=None):
        super().__init__(plane, pos=pos)

        if leftover > 200:
            self.next = BaseM_cl
        else:
            self.next = AM_cl


class BaseM_cl(Triang):
    geom = Base_M
    left = 400 / 2

    def __init__(self, plane=None, pos=None, leftover=None):
        super().__init__(plane, pos=pos)

        if leftover > 200:
            self.next = Base_cl
        else:
            self.next = B_cl


class A_cl(Triang):
    geom = A
    left = 228.7

    def __init__(self, plane=None, pos=None, leftover=None):
        super().__init__(plane, pos=pos)

        self.next = Base_cl
        self.prev = AM_cl


class AM_cl(Triang):
    geom = A_M
    left = 228.7

    def __init__(self, plane=None, pos=None, leftover=None):
        super().__init__(plane, pos=pos)

        self.next = A_cl
        self.prev = Base_cl


class B_cl(Triang):
    geom = B
    left = 228.7

    def __init__(self, plane=None, pos=None, leftover=None):
        super().__init__(plane, pos=pos)

        self.next = BM_cl
        self.prev = BaseM_cl


class BM_cl(Triang):
    geom = B_M
    left = 228.7

    def __init__(self, plane=None, pos=None, leftover=None):
        super().__init__(plane, pos=pos)

        self.next = BaseM_cl
        self.prev = B_cl


from mmcore.collections import DCLL

from mmcore.geom.parametric import PlaneLinear
from mmcore.geom.vectors import unit


class PolyLineWallIterator:
    def __init__(self, obj):
        self._obj = obj
        self._i = -1

    class Item:
        def __init__(self, parent: 'PolyLineWall', i):
            self.parent = parent
            self.i = i

        def evaluate(self, t):
            u, v = t
            return self.parent.evaluate((u + self.i, v))

        def plane_at(self, t):
            u, v = t
            return self.parent.plane_at((u + self.i, v))

        def size(self):
            return self.parent.lines()[self.i].length, self.parent.high

    def __next__(self):
        self._i += 1
        if self._i > len(self._obj) - 1:
            raise StopIteration

        return self.Item(self._obj, int(self._i))


class PolyLineWall:
    def __init__(self, points, high=600):
        super().__init__()
        self.points = DCLL()

        self.high = high
        for point in points:
            self.points.append(point)

    def evaluate(self, t):
        """

        @param t: tuple[float, float] uv - like. U=0.3 place in first segment, U=1.3 place in second segment, etc...
        @return:
        """
        u, v = t
        seg, U = divmod(u, 1)
        if int(seg) == len(self.points):
            seg = 0
        node = self.points.get_node(int(seg))
        point = Linear.from_two_points(node.data, node.next.data).evaluate(U)
        point[-1] += v * self.high
        return point.tolist()

    def plane_at(self, t):
        u, v = t
        seg, U = divmod(u, 1)
        if int(seg) == len(self.points):
            seg = 0
        node = self.points.get_node(int(seg))
        ln = Linear.from_two_points(node.data, node.next.data)

        return PlaneLinear(origin=ln.evaluate(U), xaxis=unit(ln.direction), yaxis=[0, 0, 1])

    def lines(self):
        node = self.points.head
        lns = []
        for i in range(len(self.points)):
            lns.append(Linear.from_two_points(node.data, node.next.data))
        return lns

    def __getitem__(self, item):
        return PolyLineWallIterator.Item(self, item)

    def __len__(self):
        return len(self.points)

    def __iter__(self):

        return PolyLineWallIterator(self)


class Position:
    opt = (A_cl, AM_cl, Base_cl, BaseM_cl, B_cl, BM_cl)

    def __init__(self, surf: PolyLineWall):
        self.state = 0

        self.surf = surf
        self.origin = [(0, 0), (0, 0), (0, 0), (0, 0)]
        self.init_frame = [i.plane_at(o) for i, o in zip(self.surf, self.origin)]

        self.geoms = [[A_cl(self.init_frame[0], pos=0)], [], [], []]

        self.leftover = [i.size()[0] for i in self.surf]

    def __call__(self):

        for i in range(4):
            o = 0
            while True:

                if self.leftover[self.state] < 0:

                    try:
                        prev = self.geoms[self.state][-1].next

                        self.state += 1
                        self.leftover[self.state] -= prev.left
                        shape = prev(self.init_frame[self.state], pos=0, leftover=self.leftover[self.state])

                        self.geoms[self.state].append(shape)
                        break

                    except IndexError:

                        if self.geoms[0][0].prev != self.geoms[3][-1].next:
                            follow = self.geoms[0][0].prev

                            shape = follow(self.geoms[3][-2].plane, self.geoms[3][-2].pos, leftover=0)
                            # del self.geoms[-1][-1]
                            # del self.geoms[-1][-1]

                            # self.geoms[3].append(shape)

                        break

                prev = self.geoms[self.state][-1]

                print(self.leftover[self.state], type(prev), prev.left, o, self.state)

                self.leftover[self.state] -= prev.left
                o += 1

                self.next()

    def next(self):
        prev = self.geoms[self.state][-1]
        follow = prev.next

        point = self.move_point(prev.left)
        new_frame = self.surf[self.state].plane_at(point)

        shape = follow(new_frame, prev.pos + 1, leftover=self.leftover[self.state])

        self.geoms[self.state].append(shape)

    def move_point(self, dist):

        U = self.origin[self.state]
        size = self.surf[self.state].size()[0]

        ll = size / dist
        norm = 1 / ll
        U = U[0] + norm, U[1]

        pt = self.surf[self.state].evaluate(U)
        print(U, dist, pt)
        self.origin[self.state] = U

        return U


v = [(-208372.879001, -14759.245091, 0.0),
     (-188554.34088413313, -15118.908494768657, 0.0),
     (-186299.529906, -31025.609608259147, 0.0),
     (-213065.707534, -35086.371292, 0.0)]
pw = PolyLineWall(v)

b = Position(pw)
b()

a = [[ALine(geometry=i.shape.boundaries[0]) for i in b.geoms[0]],
     [ALine(geometry=i.shape.boundaries[0]) for i in b.geoms[1]],
     [ALine(geometry=i.shape.boundaries[0]) for i in b.geoms[2]],
     [ALine(geometry=i.shape.boundaries[0]) for i in b.geoms[3]]]
from mmcore.geom.shapes.base import OccLikeLoop


def bounds(outer=[[-241245.93137, -10699.151687, 0],
                  [-126531.615261, -6806.179119, 0],
                  [-143306.718695, -27002.024251, 0],
                  [-229852.594394, -39899.110783, 0]
                  ],

           # x =  np.array([(3829.7653620639830, -259.05847478062663, 0.0), (-18.839782761661354, 1.2743875736564974, 0.0), (152.61152519640734, 2853.5259909564056, 0.0), (4196.9531817063826, 3178.7879245086524, 0.0)]).tolist()
           inner=np.array([[-213065.707534, -35086.371292, 0],
                           [-208372.861742, -14759.249076, 0],
                           [-187527.064191, -14063.649196, 0],
                           [-186302.555933, -31077.040224, 0],
                           ]).tolist()):
    inner_border = OccLikeLoop(*inner)
    outer_border = OccLikeLoop(*outer)
    inner_border.width = 32000
    outer_border.width = 24000
    return inner_border, outer_border
