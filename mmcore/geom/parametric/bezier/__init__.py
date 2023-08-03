from mmcore.base import A, ALine, APoints
from mmcore.base.components import GeometryComponent
from mmcore.base.models.gql import LineBasicMaterial, PointsMaterial
from mmcore.base.params import ParamGraphNode, param_graph_node, param_graph_node_native
from mmcore.geom.parametric import Linear, NurbsCurve
from mmcore.geom.parametric.base import NormalPoint
from mmcore.geom.point import ControlPointList
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


import numpy as np

col = ColorRGB(70, 70, 70).decimal
col2 = ColorRGB(157, 75, 75).decimal


@param_graph_node_native
def spiral(radius=10, high=10, pitches=12):
    def evaluate(t):
        return np.cos(t * pitches) * radius * np.cos(t), np.sin(t * pitches) * radius * np.cos(t), (t) * high

    return NurbsCurve([evaluate(t) for t in np.linspace(0, 1, 8 * pitches)], dimension=3)


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


# cpt = ControlPointList(points=[ControlPoint(x=0, y=0, z=0, name="PointA", uuid="pointa"),
#                               ControlPoint(x=10, y=0, z=10, name="PointB", uuid="pointb"),
#                               ControlPoint(x=5, y=10, z=5, name="PointC", uuid="pointc")])


class Bezier(GeometryComponent):
    points: ControlPointList
    pts = []
    num = 100
    color = (157, 75, 75)
    secondary_color = (100, 100, 100)
    __exclude__ = ["vs", "us", "sleep_time", "pts"]
    num = 100

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
        self.pts = []
        for n in np.linspace(0, 1, self.num):
            self.pts.append(self._bz(n).evaluate(n).tolist())
        self.__repr3d__()

        # print(params)

        return self

    def solve(self):
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



        item.bezier_curve = ALine(_endpoint=f"params/node/{self.param_node.uuid}",
                                  controls=self.param_node.todict(), uuid=self.uuid + "_bezier_curve",
                                  geometry=self.pts,
                                  material=LineBasicMaterial(color=ColorRGB(*self.color).decimal))
        return item

    def tan(self, t):
        return NormalPoint(self.evaluate(t).tolist(), unit(self._bz(t).direction).tolist())

    def evaluate(self, t):
        return self._bz(t).evaluate(t)

    def to_nurbs(self):
        return NurbsCurve(np.array(self.points).tolist(), degree=2)

    @property
    def length(self):
        return self.to_nurbs().length
