import typing

from more_itertools import flatten

from mmcore.base import ALine, A, APoints, AGroup
from mmcore.base.geom import MeshData
# This Api provides an extremely flexible way of updating data. You can pass any part of the parameter dictionary
# structure, the parameters will be updated recursively and only the part of the graph affected by the change
# will be recalculated.
from mmcore.base.params import ParamGraphNode, param_graph_node, param_graph_node_native
from mmcore.collections import DCLL
from mmcore.geom.csg import CSG
from mmcore.geom.parametric import Linear
from mmcore.geom.parametric.nurbs import NurbsCurve, NormalPoint
from mmcore.geom.parametric.pipe import Pipe
from mmcore.geom.vectors import unit

a = ParamGraphNode(dict(x=1.0, y=2.0, z=3.0), name="A")
b = ParamGraphNode(dict(x=-1.0, y=-2.0, z=-3.0), name="B")
c = ParamGraphNode(dict(x=10.0, y=20.0, z=30.0), name="С")
d = ParamGraphNode(dict(x=-11.0, y=12.0, z=13.0), name="D")

from mmcore.geom.materials import ColorRGB

# render_lines.todict(no_attrs=True) will return the complete dictionary of parameters affecting the system.


# I use json.dumps(..., indent=3) to visually print out the whole dictionary, I could also use something like pprint,
# but it"s important to show that the parameters are parsed to the scolar simplest data types.
# We can decompose a system of any complexity into a parameter tree with prime, numbers, strings, boolean values, etc.


class Component:

    def __new__(cls, *args, name=None, uuid=None, **params):
        self = super().__new__(cls)

        if uuid is None:
            self.uuid = _uuid.uuid4().hex
        self.uuid = uuid
        self.name = name

        dct = dict(zip(list(cls.__annotations__.keys())[:len(args)], args))
        params |= dct

        print(params)

        for k, v in params.items():
            if v is not None:
                self.__dict__[k] = v
        prms = dict()
        for k in params.keys():
            if not k.startswith("_"):
                prms[k] = params[k]
        node = ParamGraphNode(prms, uuid=uuid, name=self.name, resolver=self)
        self.param_node = node

        node.solve()
        return node

    def __call__(self, **params):
        for k, p in params.items():
            if p is not None:
                setattr(self, k, p)
        return self

    @property
    def endpoint(self):
        return f"params/node/{self.param_node.uuid}"


import numpy as np
import time

from mmcore.base.sharedstate import serve

col = ColorRGB(70, 70, 70).decimal
col2 = ColorRGB(157, 75, 75).decimal

from mmcore.base.models.gql import PointsMaterial, LineBasicMaterial


@param_graph_node_native
def spiral(radius=10, high=10, pitches=12):
    def evaluate(t):
        return np.cos(t * pitches) * radius * np.cos(t), np.sin(t * pitches) * radius * np.cos(t), (t) * high

    return NurbsCurve([evaluate(t) for t in np.linspace(0, 1, 8 * pitches)], dimension=3)

class ControlPoint(Component):
    x: float = 0
    y: float = 0
    z: float = 0
    size: float = 0.5
    color: tuple = (157, 75, 75)

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __call__(self, **kwargs):
        super().__call__(**kwargs)
        self.__repr3d__()
        return self

    def __repr3d__(self):
        self._repr3d = APoints(uuid=self.uuid,
                               name=self.name,
                               geometry=[self.x, self.y, self.z],
                               material=PointsMaterial(color=ColorRGB(*self.color).decimal, size=self.size),
                               _endpoint=self.endpoint,
                               controls=self.param_node.todict())
        return self._repr3d

    def __iter__(self):
        return iter([self.x, self.y, self.z])


def line_from_points(start, end):
    return Linear.from_two_points(list(start.values()), list(end.values()))


@param_graph_node(params=dict(start=a, end=b))
def first_line_from_points(start, end):
    return line_from_points(start, end)


@param_graph_node(params=dict(start=b, end=c))
def second_line_from_points(start, end):
    return line_from_points(start, end)


@param_graph_node(params=dict(line=first_line_from_points, t=0.2))
def first_line_eval(line, t):
    return line.evaluate(t)


@param_graph_node(params=dict(line=second_line_from_points, t=0.8))
def second_line_eval(line, t):
    return line.evaluate(t)


@param_graph_node(params=dict(start=first_line_eval, end=second_line_eval))
def result_line_from_points(start, end):
    return Linear.from_two_points(start, end)


@param_graph_node(dict(line=result_line_from_points,
                       num=100,
                       uuid="test_polyline",
                       color=(157, 75, 75),
                       secondary_color=(157, 75, 75),
                       sleep=False, sleep_time=0.01)
                  )
def bezier_polyline(line, num, uuid, color, secondary_color, sleep, sleep_time):
    points = []
    for i in np.linspace([0, 1], [1, 0], num=num):
        if sleep:
            time.sleep(sleep_time)

        # render_lines(result={
        #    "line": {
        #        "start": {
        #           "t": i[0]
        #       },
        #       "end": {
        #           "t": i[1]
        #       }
        #   },
        #   "color": (20, int(i[0] * 255), int(i[1] * 255))
        # },
        #   secondary_color=secondary_color
        # )
        points.append(line.evaluate(i[0]).tolist())

    return ALine(geometry=points, name="Bezier Curve", uuid=uuid,
                 material=LineBasicMaterial(color=ColorRGB(*color).decimal))


import uuid as _uuid

evlbz = lambda x: lambda a, b, c: lambda t: x(**{
    'start': {
        'line': {
            'start': {
                'x': a.x, 'y': a.y, 'z': a.z
            },
            'end': {
                'x': b.x, 'y': b.y, 'z': b.z
            }
        },
        't': t},
    'end': {
        'line': {
            'start': {
                'x': b.x, 'y': b.y, 'z': b.z
            },
            'end': {'x': c.x, 'y': c.y, 'z': c.z}},
        't': t}})
evlbz2 = evlbz(result_line_from_points)
import string


class ControlPointList(Component):

    def __new__(cls, points=(), *args, **kwargs):
        node = super().__new__(cls, *args, **dict(zip(string.ascii_lowercase[:len(points) + 1], points)),
                               _points_keys=list(string.ascii_lowercase[:len(points)]), **kwargs)

        return node

    def __array__(self):
        return np.array([list(i) for i in self.points], dtype=float)

    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return self.points.__len__()

    def __getitem__(self, item):
        return self.points.__getitem__(item)

    @property
    def points(self):
        lst = []
        for k in self._points_keys:
            lst.append(getattr(self, k))
        return lst

    def __repr3d__(self):
        self._repr3d = AGroup(seq=self.points, uuid=self.uuid, name=self.name)

        return self._repr3d


# cgraph = ComponentGraph()
cpt = ControlPointList(points=[ControlPoint(x=0, y=0, z=0, name="PointA", uuid="pointa"),
                               ControlPoint(x=10, y=0, z=10, name="PointB", uuid="pointb"),
                               ControlPoint(x=5, y=10, z=5, name="PointC", uuid="pointc")])


class Bezier(Component):
    points = cpt

    num = 100

    def __new__(cls, name=None,
                uuid="test_bezier",
                points=cpt,
                num=100,
                color=(157, 75, 75),
                secondary_color=(100, 100, 100),
                sleep=False,
                sleep_time=0.01,
                us=(0, 1),
                vs=(1, 0)):
        node = super().__new__(cls, name=name,

                               uuid=uuid,
                               points=points,
                               num=num,
                               color=color,
                               secondary_color=secondary_color,

                               )
        node.resolver.sleep = sleep
        node.resolver.sleep_time = sleep_time
        node.resolver.vs = vs
        node.resolver.us = us
        return node

    def __call__(self, **params):
        """
        line=result_line_from_points,
        num=100,
        uuid="test-polyline",
        color=(150, 150, 40),
        secondary_color=(200, 200, 40),
        sleep=False,
        sleep_time=0.01
        us,vs=[0, 1], [1, 0]
        @param params:
        @return:
        """
        super().__call__(**params)
        self._bz = evlbz2(*self.points)
        self.__repr3d__()

        # print(params)

        return self

    def __repr3d__(self):
        item = A(uuid=self.uuid,
                 _endpoint=f"params/node/{self.param_node.uuid}",
                 controls=self.param_node.todict()
                 )
        item.bezier_controls = ALine(_endpoint=f"params/node/{self.param_node.uuid}",
                                     controls=self.param_node.todict(), uuid=self.uuid + "_bezier_controls",
                                     geometry=np.array(self.points).tolist(),
                                     material=LineBasicMaterial(color=ColorRGB(*self.secondary_color).decimal))
        item.bezier_controls.control_points = APoints(_endpoint=f"params/node/{self.param_node.uuid}",
                                                      controls=self.param_node.todict(),
                                                      uuid=self.uuid + "_control_points",
                                                      geometry=np.array(self.points).tolist(),
                                                      material=PointsMaterial(color=ColorRGB(*self.color).decimal,
                                                                              size=0.3))

        pts = []
        for n in np.linspace(0, 1, self.num):
            pts.append(self._bz(n).evaluate(n).tolist())

        item.bezier_curve = ALine(_endpoint=f"params/node/{self.param_node.uuid}",
                                  controls=self.param_node.todict(), uuid=self.uuid + "_bezier_curve", geometry=pts,
                                  material=LineBasicMaterial(color=ColorRGB(*self.color).decimal))
        self._repr3d = item

    def tan(self, t):
        return NormalPoint(self.evaluate(t).tolist(), unit(self._bz(t).direction).tolist())

    def root(self):
        return self._repr3d.root()

    def evaluate(self, t):
        return self._bz(t).evaluate(t)


class Loft(Component):

    profiles: list
    path: typing.Any = spiral
    color: tuple = (200, 10, 10)

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def tesselate(self):

        lcpts = list(flatten(self.cpts))
        indices = []
        node = self.cpts.head
        normals = []
        polys = []
        for i in range(len(self.cpts)):
            nodeV = node.data.head
            nodeV2 = node.next.data.head

            for j in range(len(node.next.data)):
                d1 = np.array(nodeV2.data) - np.array(nodeV.data)
                d2 = np.array(nodeV2.next.data) - np.array(nodeV.data)
                norm = np.cross(unit(d1), unit(d2))
                normals.append(norm)

                a, b, c = [nodeV.data, nodeV2.data, nodeV.next.data]
                d, e, f = [nodeV2.next.data, nodeV.next.data, nodeV2.data]
                from mmcore.geom.csg import BspPolygon, BspVertex
                indices.extend([lcpts.index(i) for i in [a, b, c, d, e, f]])
                polys.append(BspPolygon([BspVertex(pos=a), BspVertex(pos=b), BspVertex(pos=c)]))
                polys.append(BspPolygon([BspVertex(pos=d), BspVertex(pos=e), BspVertex(pos=f)]))
                nodeV = nodeV.next
                nodeV2 = nodeV2.next

            node = node.next

        self._mesh_data = MeshData(vertices=lcpts, indices=np.array(indices).reshape((len(indices) // 3, 3)),
                                   normals=normals)
        self._csg = CSG.fromPolygons(polys)

    def solve_ll(self, bzz, prf, prf2):
        # xxx = bzz(points=self.path.points)
        pp = Pipe(bzz, prf)
        pp2 = Pipe(bzz, prf2)

        self.cpts = DCLL()
        for p in np.linspace(1, 0, 100):
            pl = DCLL()
            for pt in pp.evaluate_profile(p).control_points:
                pl.append(pt)
            self.cpts.append(pl)
        for p2 in np.linspace(0, 1, 100):
            pl = DCLL()
            for pt in pp2.evaluate_profile(p2).control_points:
                pl.append(pt)
            self.cpts.append(pl)
        return self.cpts

    def __call__(self, **kwargs):
        super().__call__(**kwargs)

        self.solve_ll(self.path, NurbsCurve(self.profiles[0], degree=1), NurbsCurve(self.profiles[1], degree=1))
        self.tesselate()
        self._repr3d = self._mesh_data.to_mesh(_endpoint=f"params/node/{self.param_node.uuid}",
                                               controls=self.param_node.todict(), uuid=self.uuid,
                                               color=ColorRGB(*self.color).decimal, opacity=0.5)
        self._repr3d.path = self.path
        return self


# prf=NurbsCurve(np.array(shapely.Polygon(prf.control_points).buffer(5).boundary.coords).tolist())
# prf2=NurbsCurve(np.array(shapely.Polygon(prf2.control_points).buffer(1).boundary.coords).tolist())

bz = Bezier()

p1 = np.array([[-23, 40, 0], [23, 40, 0],
               [25, 38, 0],
               [25, -38, 0],
               [23, -40, 0],
               [-23, -40, 0],
               [-25, -38, 0],
               [-25, 38, 0],
               [-23, 40, 0]]) * 0.05
profile_outer = p1.tolist()

p2 = np.array([[-21.0, 36.5, 0],
               [21, 36.5, 0],
               [21.5, 36, 0],
               [21.5, -36, 0],
               [21, -36.5, 0],
               [-21, -36.5, 0],
               [-21.5, -36, 0],
               [-21.5, 36, 0],
               [-21.0, 36.5, 0]]) * 0.05

path = bz()
profile_inner = p2.tolist()
profile_outer.reverse()
profile_inner.reverse()

# srf1 = Loft(name="test_loft_b", path=spiral, profiles=(profile_outer, profile_inner), uuid="test_loft_a")
srf2 = Loft(name="test_loft_a", path=bz, profiles=(profile_outer, profile_inner), uuid="test_loft_b")
serve.start()

# mesh=parametric_mesh_bruteforce(pp,uv=[50, 9] ).to_mesh()
# mesh2=parametric_mesh_bruteforce(pp2,uv=[50, 9] ).to_mesh()
# mesh3=parametric_mesh_bruteforce(srf,uv=[320, 32]).to_mesh(color=col)
# mesh3.dump("mmm.json")
# s=Shape(boundary=prf.control_points, holes=[ prf2.control_points] )
# s2=Shape(boundary=pp.evaluate_profile(1).control_points, holes=[ pp2.evaluate_profile(1).control_points])
# grp=AGroup(name="pipe")
# grp.add(mesh)
# grp.add(mesh2)
#
#
# grp.dump("pp.json")
