# importlib.reload( mmcore.geom.parametric )
# importlib.reload( mmcore.geom.transform )
import numpy as np
from more_itertools import flatten
from scipy.spatial.distance import euclidean

from mmcore.base import AGroup, A
from mmcore.base.sharedstate import serve
from mmcore.collections.basic import DoublyLinkedList
from mmcore.geom.parametric import Linear
# importlib.reload(mmcore)
from mmcore.geom.parametric import PlaneLinear
from mmcore.geom.shapes import Shape
from mmcore.geom.transform import Transform, Plane, YZ_TO_XY, WorldXY
from mmcore.geom.vectors import unit

pts = [[-241245.93137, -10699.151687, 18315.472853],
       [-126531.615261, -6806.179119, -2283.526903],
       [-229852.594394, -39899.110783, 22587.502373],
       [-143306.718695, -27002.024251, 4106.366197]]
panel_boundary = [[-600.0, 0.0], [0.0, -200.0], [0.0, 200.0], [-600.0, 0.0]]
pnl = Shape(boundary=panel_boundary).offset(-2.3).mesh
pnl @ YZ_TO_XY


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

    def __copy__(self):
        if not self.is_compound():
            return Panel(boundary=self._boundaries[0], matrix=self.matrix)
        else:
            p = Panel(boundary=self._boundaries[0], matrix=self.matrix)
            for bnd in self._boundaries:
                p._boundaries.append(bnd)
            return p

    def toshape(self):
        m = self.plane.transform_to_other(WorldXY)
        Shape()
grp = AGroup(name="Panels")


def tri_transform_gen(j, line, next_line, flip=1, step=400):
    # grp = AGroup(name="row", uuid=uuid.uuid4().hex)
    s = step * np.dot(unit(line.direction), unit(next_line.direction))
    i = 0
    for xx, xy in zip(line.divide_distance_planes(step), next_line.divide_distance_planes(s)):
        try:
            point2 = xy.origin
            y = unit(point2 - xx.origin)
            v2 = np.cross(unit(flip * xx.normal), unit(y))

            pln = Plane(origin=xx.origin, xaxis=unit(y), yaxis=v2, normal=unit(flip * xx.normal))
            # tr = Transform()
            # tr.rotate(axis=[1, 0, 0], angle=-np.pi / 2)
            # panel @ tr
            # d, i = KD.query(x.origin + (y * 0.3))
            # print(x.origin, x.origin + (y * 0.3), d, i, ldt[i])
            # panel.properties |= ldt[i].__dict__
            pd = {-1: "1", 1: "2"}
            oo = A(uuid=f"{j}-{i}-{pd[flip]}", name="Panel")

            oo @ Transform.from_world_to_plane(pln)
            i += 1
            oo.mesh = pnl

            grp.add(oo)

        except:
            pass


def f2(i, node):
    if node.next is None:
        pass
    elif node.prev is None:
        pass




    else:
        if i % 2 != 0:
            ln = node.data.extend(200, 1200)
            lnn = node.next.data.extend(200, 1200)
            lnp = node.prev.data.extend(200, 1200)
        else:
            ln = node.data.extend(0, 1200)
            lnn = node.next.data.extend(0, 1200)
            lnp = node.prev.data.extend(0, 1200)

        tri_transform_gen(i, ln, lnn, flip=-1)
        tri_transform_gen(i, ln, lnp, flip=1)


def hyp(pts=pts, step=600):
    mn = np.mean(np.array(pts), axis=0)
    # print(mn)
    X, Y, Z, U = np.array(pts)
    V1 = lambda t: (Y - X) * t + X
    V2 = U - Z
    V22 = lambda t: Z - V1(t)
    ED = lambda t: euclidean(V1(t), Z)
    D = lambda t: np.dot(unit(V2), unit(V22(t)))
    TT = lambda t: Z - unit(V2) * D(t) * ED(t)
    # print("\n\n".join(str(i) for i in [t,Y-X,(Y-X)*t,V1(t),V2,V22(t),ED(t),D(t),TT(t)]))
    b = V1(0).tolist()
    a = TT(0).tolist()

    c = V1(1).tolist()

    d = TT(1).tolist()
    cnt = euclidean(a, d) / step

    pstep = 1 / cnt

    dll = DoublyLinkedList()
    for i in range(int(cnt) + 1):
        print(pstep * i, i)
        _b = V1(pstep * i).tolist()
        _a = TT(pstep * i).tolist()

        dll.append(Linear.from_two_points(_a, _b))
    # print(dll)
    node = dll.head
    for i in range(dll.count):
        # print(i, node)
        f2(i, node)

        node = node.next


hyp()
grp.scale(0.001, 0.001, 0.001)
grp.rotate(-np.pi / 2, (1, 0, 0))
grp.dump('pppp.json')
serve.start()
