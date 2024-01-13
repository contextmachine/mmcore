import itertools
import weakref

import numpy as np

from mmcore.base.ecs.components import component
from mmcore.geom.line import Line
from mmcore.geom.vec import cross


@component()
class Outputs:
    ...


class ParamIterator:
    def __init__(self, param):
        self._p = weakref.WeakValueDictionary({"p": param})
        self._end = False

    def __next__(self):
        if not self._end:
            self._end = False
            return self._p.get("p")
        else:
            raise StopIteration()


class Param:
    def __init__(self, value=None, table=None):
        self._value = value
        self._table = table
        self._use_table = table is not None

    @property
    def value(self):
        if self._use_table:
            return self._table[self._value]
        else:
            return self._value

    @value.setter
    def value(self, val):
        if self._use_table:
            self._table[self._value] = val
        else:
            self._value = val

    def __iter__(self):
        return self._value


@component()
class ReferenceLineInputs:
    start: Param = None
    end: Param = None


@component()
class ReferenceLineOutputs:
    line: Param


@component()
class PointInputs:
    t: Param = None
    line: Param = None


@component()
class IntersectionInputs:
    line1: Param = None
    line2: Param = None


@component()
class PointOutputs:
    xyz: Param = None


@component()
class IntersectionOutputs:
    t1: Param = None
    t2: Param = None


class ElementInterface:
    _inputs = None
    _outputs = None

    def __init__(self):
        self.scheduled = []

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def __hash__(self):
        return hash(tuple(hash(v.value) for v in self._inputs.values()))

    def execute(self):
        for i in self.scheduled:
            i.execute()


class PointOnLine(ElementInterface):
    inputs: PointInputs
    outputs: PointOutputs

    def __init__(self, t: Param, line: Param):
        super().__init__()
        self._inputs = PointInputs(t=t, line=line)
        self._outputs = PointOutputs(xyz=Param(value=None))

    def __hash__(self):
        return hash((self._inputs.line.value, self._inputs.t.value))

    @property
    def xyz(self):
        self.execute()
        return self._outputs.xyz.value

    def execute(self):
        self._outputs.xyz.value = self.inputs.line.value(self.inputs.t.value)
        super().execute()

    def __repr__(self):
        return (f"{self.__class__.__name__}(inputs={self._inputs},outputs={self._outputs})")


class Intersection(ElementInterface):
    def __init__(self, line1: Param, line2: Param):
        super().__init__()
        self._inputs = IntersectionInputs(line1=line1, line2=line2)
        self._outputs = IntersectionOutputs(t1=Param(value=None), t2=Param(value=None))

    def __hash__(self):
        return hash((self._inputs.line1.value, self._inputs.line1.value))

    @property
    def line1(self):
        return self._inputs.line1

    def execute(self):
        t1, t2 = self._inputs.line1.value.bounded_intersect(self._inputs.line2.value)[:2] * np.array([1, -1])
        self._outputs.t1.value = t1
        self._outputs.t2.value = t2
        super().execute()

    @property
    def is_bounded_first(self):
        return 0 <= self.t1 <= 1

    @property
    def is_bounded_second(self):
        return 0 <= self.t2[1] <= 1

    @property
    def success(self):
        return self.is_bounded

    @property
    def is_bounded(self):
        return self.is_bounded_first and self.is_bounded_second

    @property
    def t1(self):
        return self._outputs.t1.value

    @property
    def t2(self):
        return self._outputs.t2.value


class ReferenceLine(ElementInterface):
    def __init__(self, start: Param, end: Param):
        super().__init__()
        self._inputs = ReferenceLineInputs(start=start, end=end)
        self._outputs = ReferenceLineOutputs(line=Param(value=Line.from_ends(start.value, end.value))
                )

    def length(self):
        return self._outputs.line.value.length()

    def execute(self):
        self._outputs.line.value.start = self._inputs.start.value
        self._outputs.line.value.end = self._inputs.end.value
        self._outputs.line.value.solve()
        super().execute()


@component()
class OffsetLineInputs:
    distance: Param = None
    line: Param = None


@component()
class OffsetLineOutputs:
    line: Param = None


class OffsetLine(ElementInterface):
    def __init__(self, distance: Param, line: Param):
        super().__init__()
        self._inputs = OffsetLineInputs(distance, line)
        self._outputs = ReferenceLineOutputs(line=Param())

    def execute(self):
        self._outputs.line.value = self._inputs.line.value.offset(self._inputs.distance.value
                )
        super().execute()


class LineOffset(Line):
    _shape = (3, 3)
    _dtype = float

    def __init__(self, ln: Line, d=0.0):
        self._self_array = np.zeros((self._shape[0] + 1, self._shape[1]), float)
        self._self_array = []
        self.d = d
        self._owner = ln

        self.solve()

    def solve(self):
        self._self_array[:] = self._array

    def __hash__(self):
        return hash((self._owner.__hash__(), self.d))

    @property
    def perp(self):
        return cross([0, 0, 1], self._owner.unit)

    @property
    def start(self):
        return self._owner.start + self.perp * self.d

    @property
    def direction(self):
        return self.end - self.start

    @property
    def end(self):
        return self._owner.end + self.perp * self.d

    @property
    def _array(self):
        return np.array([self._owner._array[0] + self.d * self.perp, *self._owner._array[1:]]
                )

    @_array.setter
    def _array(self, v):
        self._owner._array[:] = v


def clust(lns, ds=(-17.0, -14.0, -17.0, -14.0)):
    prms = []
    for l, d in zip(lns, ds):
        prms.append(Param(LineOffset(l, d)))
    prms2 = []
    for p1, p2 in itertools.pairwise(prms):
        i1 = Intersection(p1, p2)
        i1.execute()
        p22 = PointOnLine(i1.outputs.t1, i1.inputs.line1)
        i1.scheduled.append(p22)
        p22.execute()
        prms2.append(p22)

    return prms, prms2


def case():
    """
    r = Rectangle(100, 150)

    from mmcore.geom.mesh.shape_mesh import mesh_from_bounds

    def place_boxes(a, d1, d2, boxes):
        for i, aa, in enumerate(a):
            dd = dot(d1[i].unit, d2[i].unit)
            h = 20.0 if np.isclose(aa, np.pi / 2) else 0.1

            if aa < np.pi / 2:

                b1 = Box(25 + 14, 17, h=h, xaxis=d1[i].unit, origin=d1[i].start)
                b2 = Box(18, 14, h=h, xaxis=-d2[i].unit, origin=d2[i].end + d1[i].unit * 14 - d2[i].unit * dd * 14)
            elif aa >= np.pi / 2:
                b1 = Box(25, 17, h=h, xaxis=d1[i].unit, origin=d1[i].start + (14 * d1[i].unit))
                b2 = Box(18, 14, h=h, xaxis=-d2[i].unit, origin=d2[i].end + d1[i].unit * 14)

            boxes.append([b1, b2]
                         )
    r = Rectangle(100, 150)
    corners = r.corners
    corners[-1] += np.array([10., 0., 0.])
    lns = [Line.from_ends(*l) for l in polyline_to_lines(corners)]

    a, b, c = line_angles(lns)
    boxes = []
    place_boxes(a, b, c, boxes)
    from mmcore.common.viewer import ViewerBaseGroup
    poly = mesh_from_bounds(corners.tolist()).amesh()
    vgg = ViewerBaseGroup((), 'ygtp')
    vgg.add(poly)
    for bx in boxes:
        aa, bb = bx
        vgg.add(aa.to_mesh())
        vgg.add(
            bb.to_mesh())

    vgg.dump('iii.json')"""
    ...


@component()
class Inputs:
    ...


def offsets_from_lines(lns, dists=(-17.0, -14.0, -17.0, -14.0)):
    prms = []
    for l, d in zip(lns, dists):
        prms.append(Param(LineOffset(l, d)))
    return prms


def intersect_from_offsets(offs):
    prms = []
    prms0 = []
    for i, p1 in enumerate(offs):
        print(i)
        i = i + 1
        if i >= len(offs):
            i = 0

        print(i)
        p2 = offs[i]
        i1 = Intersection(p1, p2)
        i1.execute()
        prms0.append(i1)
        p22 = PointOnLine(i1.outputs.t1, i1.inputs.line1)
        i1.scheduled.append(p22)
        p22.execute()
        prms.append(p22)

    return prms0, prms


def lines_from_ipts(pts):
    prms = []
    for i, p1 in enumerate(pts):
        print(i)
        i = i + 1
        if i >= len(pts):
            i = 0

        print(i)
        p2 = pts[i]
        rl = ReferenceLine(p1.outputs.xyz, p2.outputs.xyz)
        p1.scheduled.append(rl)
        p2.scheduled.append(rl)
        rl.execute()
        prms.append(rl)
    return prms


def lines_to_box(lls, maxes=(44, 18, 44, 18)):
    prms = []
    for i, p1 in enumerate(lls):
        a, b = divmod(p1.length(), maxes[i])
        proezdi = b // 13
        ppp = []

        ppp.append(6)
        while proezdi > 0:
            ppp.append(10)
            ppp.append(maxes[i])
            ppp.append(10)
            ppp.append(6)
            proezdi -= 1

        return
    return prms
