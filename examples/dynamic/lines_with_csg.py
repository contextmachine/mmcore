import time
import json
import typing
from abc import abstractmethod

import geomdl.operations
import numpy as np
import shapely
from more_itertools import flatten

from mmcore.base import ALine, AGroup
from mmcore.base.geom import MeshData
from mmcore.base.models.gql import LineBasicMaterial
from mmcore.collections import DoublyLinkedList, DCLL
from mmcore.geom.csg import CSG
from mmcore.geom.parametric import Linear, WorldXY

# This Api provides an extremely flexible way of updating data. You can pass any part of the parameter dictionary
# structure, the parameters will be updated recursively and only the part of the graph affected by the change
# will be recalculated.
from mmcore.base.params import ParamGraphNode, param_graph_node, param_graph_node_native, TermParamGraphNode
from mmcore.geom.shapes import Shape
from mmcore.geom.tess import parametric_mesh_bruteforce
from mmcore.geom.vectors import unit

a = ParamGraphNode(dict(x=1.0, y=2.0, z=3.0), name="A")
b = ParamGraphNode(dict(x=-1.0, y=-2.0, z=-3.0), name="B")
c = ParamGraphNode(dict(x=10.0, y=20.0, z=30.0), name="С")
d = ParamGraphNode(dict(x=-11.0, y=12.0, z=13.0), name="D")


def line_from_points(start, end):
    return Linear.from_two_points(list(start.values()), list(end.values()))


@param_graph_node(params=dict(start=a, end=b))
def first_line_from_points(start, end):
    return line_from_points(start, end)


@param_graph_node_native
def first_line_from_points2(start=a, end=b):
    return line_from_points(start, end)


@param_graph_node(params=dict(start=c, end=d))
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


from mmcore.geom.materials import ColorRGB


def render_line(line, uuid, color):
    return ALine(uuid=uuid,
                 geometry=[line.tolist(), line.end.tolist()],
                 material=LineBasicMaterial(color=ColorRGB(*color).decimal))


# render_lines.todict(no_attrs=True) will return the complete dictionary of parameters affecting the system.


# I use json.dumps(..., indent=3) to visually print out the whole dictionary, I could also use something like pprint,
# but it"s important to show that the parameters are parsed to the scolar simplest data types.
# We can decompose a system of any complexity into a parameter tree with prime, numbers, strings, boolean values, etc.
# For example:
# {
#    "result": {
#       "line": {
#          "start": {
#             "line": {
#                "start": {
#                   "x": 1.0,
#                   "y": 2.0,
#                   "z": 3.0
#                },
#                "end": {
#                   "x": -1.0,
#                   "y": -2.0,
#                   "z": -3.0
#                }
#             },
#             "t": 0.2
#          },
#          "end": {
#             "line": {
#                "start": {
#                   "x": 10.0,
#                   "y": 20.0,
#                   "z": 30.0
#                },
#                "end": {
#                   "x": -11.0,
#                   "y": 12.0,
#                   "z": 13.0
#                }
#             },
#             "t": 0.8
#          }
#       },
#       "uuid": "test_line",
#       "color": [
#          20,
#          200,
#          60
#       ]
#    },
#    "uuid": "result_group"
# }


# You can discard any part of the dictionary from the "back", the most important thing you cannot do
# is lose keys from the "front".
# For example:
# ----------------------------------------------------------------------------------------------------------------------
#  We"ll use the dictionary from the example above, so what can we pass to the function in this case?
#     Wrong:
#     {
#       "result": {
#           "start":{
#               "t":0.1
#               },
#           "end":{
#               "t":0.9
#               }
#             }
#           }
#
#    Correct:
#     {
#       "result":{
#         "line": {
#             "start": {
#                 "t": 0.5
#             },
#             "end": {
#                 "t": 0.5
#             }
#           }
#         }
#       }
# def changes1():
#    render_lines(result={
#        "line": {
#            "start": {
#                "t": 0.5
#            },
#            "end": {
#                "t": 0.5
#            }
#        }
#    }
#    )
#

ptA, ptB, ptC, ptD = (-8.323991, 6.70421, -8.323991), (-8.323991, -6.70421, 8.323991), (8.323991, 6.70421, 8.323991), (
    8.323991, -6.70421, -8.323991)

from mmcore.base.registry import AGraph


# lass ComponentGraph(AGraph['...,Component']):
#   def set_relay(self, node: 'Component', name: str, v: typing.Any):
#       ...


def changes2(a=ptA,
             b=ptB,
             c=ptC,
             d=ptD):
    ...


@param_graph_node(dict(line=result_line_from_points,
                       num=100,
                       uuid="test-polyline",
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


# cgraph = ComponentGraph()


class Component:

    def __new__(cls, *args, name=None, uuid=None, **params):
        self = super().__new__(cls)

        if uuid is None:
            self.uuid = _uuid.uuid4().hex
        self.uuid = uuid
        self.name = name
        self.params = dict()
        return ParamGraphNode(params, uuid=uuid, name=self.name, resolver=self)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__call__(**kwargs)

    @abstractmethod
    def __call__(self, **params):
        ...


import numpy as np
import time


class Line(Component):
    def __new__(cls, *args, start=first_line_eval, end=second_line_eval, **kwargs):
        return super().__new__(cls, *args, start=start, end=end, **kwargs)

    def __call__(self, start=None, end=None):
        self.line = Linear.from_two_points(start, end)
        return self


# left = Bezier(name="left", uuid="left-besier-curve", num=100, us=[0, 1], vs=[1, 0], sleep=False)
# right = Bezier(name="right", uuid="right-besier-curve", num=100, us=[1, 0], vs=[0, 1], sleep=False)


# class Animate(Component):
#    def __new__(cls, name="Animate", uuid="---", curves=(left, right)):
#        return super().__new__(cls, name=name, uuid=uuid, curves=curves)
#
#    def __call__(self, curves=()):
#        grp = AGroup(name="animate", uuid="animate")
#        [grp.add(crv().render()) for crv in curves]
#        return grp
#

from mmcore.base.sharedstate import serve

# animate = Animate(curves=[right, left])
evlbz = lambda x: lambda a, b, c: lambda t: x(**{
    'start': {
        'line': {
            'start': {
                'x': a[0], 'y': a[1], 'z': a[2]
            },
            'end': {
                'x': b[0], 'y': b[1], 'z': b[2]
            }
        },
        't': t},
    'end': {
        'line': {
            'start': {
                'x': b[0], 'y': b[1], 'z': b[2]
            },
            'end': {'x': c[0], 'y': c[1], 'z': c[2]}},
        't': t}})

# evlbz = lambda x, t: x(**{'start': {'t': t}, 'end': {'t': t}}).evaluate(t)
evlbz2 = evlbz(result_line_from_points)
evlbz3 = evlbz2((0, 0, 0), (10, 0, 10), (5, 10, 5))

serve.start()
col = ColorRGB(70, 70, 70).decimal
col2 = ColorRGB(157, 75, 75).decimal

class Bezier(Component):
    def __new__(cls, name=None,
                uuid="test-bezier",
                points=((0, 0, 0), (10, 0, 10), (5, 10, 5)),
                num=100,
                color=(157, 75, 75),
                secondary_color=(100, 100, 100),
                sleep=False,
                sleep_time=0.01,
                us=(0, 1),
                vs=(1, 0)):
        return super().__new__(cls, name=name,

                               uuid=uuid,
                               points=points,
                               num=num,
                               color=color,
                               secondary_color=secondary_color,
                               sleep=sleep,
                               sleep_time=sleep_time,
                               us=us,
                               vs=vs
                               )

    def setpts(self, a, b, c):
        self.points=a,b,c
        self._bz = evlbz2(a, b, c)

    def __call__(self, points=None, **params):
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
        if points is not None:
            self.setpts(*points)
        self.params |= params
        # print(params)
        if params.get("line") is not None:
            self.line = params.get("line")
        self.params = params



        return self
    def render(self):
        self.disp = ALine(uuid=self.uuid + "-curve", geometry=[self.evaluate(t) for t in np.linspace(0, 1, 50)],
                          material=LineBasicMaterial(color=col2))

        self.disp2 = ALine(uuid=self.uuid + "-axis", geometry=[self.points], material=LineBasicMaterial(opacity=0.4,
                                                                                                   color=col))

    def tan(self, t):
        return NormalPoint(self.evaluate(t), self._bz(t).direction)

    def root(self):
        return self.disp.root()

    def evaluate(self, t):

        return self._bz(t).evaluate(t)


bz = Bezier()

from mmcore.geom.parametric.pipe import Pipe
from mmcore.geom.parametric.nurbs import NurbsCurve, NurbsLoft, NormalPoint, NurbsSurface, NurbsSurfaceGeometry

prfd1 = np.array([[-23, 40, 0], [23, 40, 0],
                  [25, 38, 0],
                  [25, -38, 0],
                  [23, -40, 0],
                  [-23, -40, 0],
                  [-25, -38, 0],
                  [-25, 38, 0],
                  [-23, 40, 0]]) * 0.05
lprfd1 = prfd1.tolist()
lprfd1.reverse()
dat = np.array([[-21.0, 36.5, 0],
                [21, 36.5, 0],
                [21.5, 36, 0],
                [21.5, -36, 0],
                [21, -36.5, 0],
                [-21, -36.5, 0],
                [-21.5, -36, 0],
                [-21.5, 36, 0],
                [-21.0, 36.5, 0]]) * 0.05





xxx = bz(points=(np.array(bz.todict()["points"]) * 2).tolist())
ldat = dat.tolist()
ldat.reverse()


class Loft(Component):
    def __new__(cls, *args, path=bz, section=(lprfd1, ldat), color=(157, 75, 75), **kwargs):
        return super().__new__(cls, *args, path=path, section=section,color=color,**kwargs)

    def tesselate(self):

        lcpts = list(flatten(self.cpts))
        indices = []
        node = self.cpts.head
        normals = []
        polys = []
        for i in range(len( self.cpts)):
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

    def solve_ll(self,bzz , prf,prf2):
        xxx=bzz(points=(np.array(bz.todict()["points"]) * 2).tolist())
        pp = Pipe(xxx, prf)
        pp2 = Pipe(xxx, prf2)

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


    def __call__(self, path, section, **kwargs):
        self.params|=kwargs
        self.solve_ll(path, NurbsCurve(section[0], degree=1),NurbsCurve(section[1], degree=1))
        self.tesselate()
        self.mesh=self._mesh_data.to_mesh(uuid=self.uuid+"-mesh",color=self.params.get("color"))
        return self



# prf=NurbsCurve(np.array(shapely.Polygon(prf.control_points).buffer(5).boundary.coords).tolist())
# prf2=NurbsCurve(np.array(shapely.Polygon(prf2.control_points).buffer(1).boundary.coords).tolist())



srf =Loft(uuid="test_loft")


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
