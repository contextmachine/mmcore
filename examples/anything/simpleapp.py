import multiprocessing as mmp
from operator import attrgetter

import multiprocess as mp
from more_itertools import flatten

from mmcore.base import A as AA
from mmcore.base import ALine
from mmcore.collections import DoublyLinkedList
from mmcore.geom.parametric import HypPar4ptGrid, Linear, PlaneLinear
from mmcore.geom.parametric.algorithms import ClosestPoint
from mmcore.geom.transform import Transform

mmp.set_start_method = "swapn"
mp.set_start_method = "swapn"

from localparams import *


class LnPointDescr:

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, instance, owner):
        if instance:

            return np.array(
                [instance.line.start, instance.line.end], dtype=float).tolist()
        else:
            return self

    def __set__(self, instance, value):
        instance.line = Linear.from_two_points(value[0], value[1])


class Axis(ALine):
    line: Linear
    points = LnPointDescr()
    part: str = "SW"
    subtype: str = "P-1"
    offset: float

    def __new__(cls, line, index=0, *args, subtype=None, part='SW', **kwargs):
        return super().__new__(cls,line=line,
                               uuid=uuid.uuid4(),
                               *args,
                               subtype=subtype,
                               part=part,
                               **kwargs)
    @property
    def properties(self):
        prp = super().properties
        prp |= {
            "length": self.line.length,
            "part": self.part,
            "subtype": self.subtype,
            "offset": self.offset,

        }
        return prp

    def __call__(self, *args, properties=None, **kwargs):
        data = super().__call__(*args, **kwargs)
        if properties:
            data["userData"]["properties"] |= dict(zip(properties, attrgetter(*properties)(self)))
        return data


def lay_to_lines(lay, part=None, name=None, high=None, color=None, **kwargs):
    grp = AGroup(name=name, uuid=uuid.uuid4().hex)
    for i, l in enumerate(lay):
        grp.add(Axis(line=l, part=part, index=i, subtype=name, offset=high, color=color, **kwargs))
    return grp


class SubSystem:
    A, B, C, D = A, B, C, D
    uuid=uuid.uuid4().hex
    part: str = "SW"
    layer_state = [
        dict(step=0.6,
             side="D",
             color=(70, 10, 240),
             name="initial"
             ),
        dict(high=0.3,
             step=0.6,
             color=(259, 49, 10),
             name="layer-1"
             ),
        dict(high=0.2,
             step=3,
             color=(259, 49, 10),
             name="layer-2"
             ),
        dict(
            high=0.5,
            step=1.5,
            color=(25, 229, 100),
            name="layer-3"
        )
    ]


    def __call__(self, *args, **kwargs):

        self.hp = HypPar4ptGrid(self.A, self.B, self.C, self.D)
        self.grpr = []
        _a, _b, _c, _d = self.layer_state
        self.initial = self.first(**_a)
        self.lay1 = self.layer(self.initial, **_b)

        self.lay2 = self.layer(self.lay1, **_c)
        self.lay3 = self.layer(self.lay2, **_d)
        dll = DoublyLinkedList()
        for i in self.initial:
            dll.append(i)
        self.trx=mp_calc(dll, step=PANELS_STEP)


        return self.trx
    def offset_hyp(self, h=2):
        hyp = self.hp
        A1, B1, C1, D1 = hyp.evaluate((0, 0)), hyp.evaluate((1, 0)), hyp.evaluate((1, 1)), hyp.evaluate((0, 1))
        hpp = []
        for item in [A1, B1, C1, D1]:
            hpp.append(np.array(item.point) + np.array(item.normal) * h)
        return HypPar4ptGrid(*hpp)

    def first(self, step=0.6, side="D", color=(70, 10, 240), name=None):
        hyp = self.hp
        current = hyp.parallel_side_grid(step, side=side)

        return current

    def layer(self, prev, high, step, color=(70, 10, 240), name="layer"):
        hyp = self.hp
        prev.sort(key=lambda x: x.length)
        nxt = prev[-1]
        hp_next = self.offset_hyp(high)
        current = hp_next.parallel_vec_grid(step, nxt)
        return current


def hyp_transform(line, next_line, step=0.4):
    for x in line.divide_distance_planes(step):
        point2 = np.array(ClosestPoint(x.origin, next_line)(x0=[0.5], bounds=[(0, 1)]).pt[0]).flatten()
        yield Transform.from_world_to_plane(PlaneLinear(origin=x.origin,
                                                        xaxis=x.xaxis,
                                                        yaxis=point2 - x.origin))



panels = AGroup(name="Panels", uuid=uuid.uuid4().hex)
class Panel(AA):
    panel: AMesh
    #triangle: ALine
    _uuid=None
    arch_type: str = "A"
    engineering_type: str = "A"
    extra: bool = True
    part: str = "SW"
    _children={'f9f81ea7ac0c4839a7f103e9b8dd4159'}
    child_keys={'panel'}

    #@property
    #def center(self):
    #    return np.mean(self.triangle_pts, axis=0)

    #@OwnerTransform
    #def triangle_pts(self):
    #    return self.triangle.points
    def __new__(cls, *args, **kwargs):
        inst = super().__new__(cls, *args, **kwargs)
        inst.idict[inst.uuid]["panel"] = PANEL_UUID

        return inst

    @property
    def uuid(self):
        if self._uuid is None:
            self._uuid=uuid.uuid4().hex
        return self._uuid
    @property
    def name(self):
        return f"{self.arch_type}-{self.engineering_type}",

    @property
    def properties(self):
        return {
            "arch_type": self.arch_type,
            "engineering_type": self.engineering_type,
            "extra": self.extra,
            "part": self.part,
            "priority": 1,
            "name": self.name

        }


def f(node, grp, obj):
    # *r,=range(1,len(dl)-2)
    if node.next is None:
        pass

    elif node.prev is not None:


        lx, rx = hyp_transform(node.data, node.next.data), hyp_transform(node.data, node.prev.data)



def ff(dl, grp, obj, step=0.4):
    for i in range(1, len(dl)):
        item = dl.get(i)
        g = AGroup(name="Panel-Pair")
        yield f(item, g, )


def no_mp(dl):
    return list(ff(dl))


def mp_calc(ll, step=0.4):
    def f2(i):

        node = ll.get(i)
        # *r,=range(1,len(dl)-1)
        if node.next is None:
            pass
        elif node.prev is not None:
            *z,=zip(hyp_transform(node.data,node.next.data),hyp_transform(node.data,node.prev.data))
            return z



    with mp.Pool(20) as p:
        return [Panel(matrix=np.asarray(f.__array__(), dtype=float).T.flatten().tolist()) for f in flatten(flatten(p.map(f2, range(1, len(ll)))))]

