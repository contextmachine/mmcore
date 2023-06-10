
import time
import json

from mmcore.base import ALine, AGroup
from mmcore.base.models.gql import LineBasicMaterial
from mmcore.geom.parametric import Linear

# This Api provides an extremely flexible way of updating data. You can pass any part of the parameter dictionary
# structure, the parameters will be updated recursively and only the part of the graph affected by the change
# will be recalculated.
from mmcore.base.params import ParamGraphNode, param_graph_node


a = ParamGraphNode(dict(x=1.0, y=2.0, z=3.0), name="A")
b = ParamGraphNode(dict(x=-1.0, y=-2.0, z=-3.0), name="B")
c = ParamGraphNode(dict(x=10.0, y=20.0, z=30.0), name="С")
d = ParamGraphNode(dict(x=-11.0, y=12.0, z=13.0), name="D")


def line_from_points(start, end):
    return Linear.from_two_points(list(start.values()), list(end.values()))


@param_graph_node(params=dict(start=a, end=b))
def first_line_from_points(start, end):
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
                 geometry=[line.start.tolist(), line.end.tolist()],
                 material=LineBasicMaterial(color=ColorRGB(*color).decimal))


colors = [(20, 200, 60),
          (100, 100, 100),
          (100, 100, 100)
          ]
render_nodes = []
for i, line in enumerate([result_line_from_points, first_line_from_points, second_line_from_points]):
    render_nodes.append(ParamGraphNode(dict(line=line,
                                            uuid=f"line_{i}",
                                            color=colors[i]),
                                       name=f"render_line_{i}",
                                       resolver=lambda prm: render_line(**prm)))


@param_graph_node(params=dict(result=render_nodes[0], uuid="result_group", secondary_color=(100, 100, 100)))
def render_lines(result, uuid, secondary_color):
    grp = AGroup(name="Lines Group", uuid=uuid)

    grp.add(result)
    for node in render_nodes[1:]:
        grp.add(node(color=secondary_color))
    return grp


# render_lines.todict(no_attrs=True) will return the complete dictionary of parameters affecting the system.
print(json.dumps(render_lines.todict(no_attrs=True), indent=3))


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
def changes1():
    render_lines(result={
        "line": {
            "start": {
                "t": 0.5
            },
            "end": {
                "t": 0.5
            }
        }
    }
    )
ptA,ptB,ptC,ptD=(-8.323991, 6.70421, -8.323991) ,(-8.323991, -6.70421, 8.323991) ,(8.323991, 6.70421, 8.323991 ) ,(8.323991, -6.70421, -8.323991)


def changes2(a=ptA,
             b=ptB,
             c=ptC,
             d=ptD):

    render_lines(result={
        "line": {
            "start": {
                "line": {
                    "start": {"x": a[0], "y": a[1], "z": a[2]},
                    "end":  {"x": b[0], "y": b[1], "z": b[2]}
                },
                "t": 0.5
            },
            "end": {
                "line": {"start":  {"x": c[0], "y": c[1], "z": c[2]},
                         "end":  {"x": d[0], "y": d[1], "z": d[2]}},
                "t": 0.2
            }
        }
    }
    )
@param_graph_node(dict(points=(), uuid="test-polyline", color=(150,150,40)))
def polyline(points, uuid, color):
    return ALine(geometry=points, uuid=uuid, material=LineBasicMaterial(color=ColorRGB(*color).decimal))

def animate():
    import numpy as np
    import time
    points=[]
    for i in np.linspace([0, 1], [1, 0], num=100):
        time.sleep(0.01)

        render_lines(result={
            "line": {
                "start": {
                    "t": i[0]
                },
                "end": {
                    "t": i[1]
                }
            },
            "color": (20, int(i[0] * 255), int(i[1] * 255))
        }, secondary_color=(200, 200, 40)

        )
        points.append(result_line_from_points().evaluate(i[0]).tolist())
        polyline(points=points,uuid="polyline" ,color=(200,200,40))
        
    for j in np.linspace([1, 0], [0, 1], num=100):
        time.sleep(0.01)
        render_lines(result={
            "line": {
                "start": {
                    "t": j[0]
                },
                "end": {
                    "t": j[1]
                }
            },
            "color": (20, int(j[1] * 255), int(j[0] * 255))
        },
         secondary_color=(200,200,40) )
        points.append(result_line_from_points().evaluate(j[1]).tolist())
        polyline(points=points,uuid="polyline" ,color=(150,150,40))
    points.append(result_line_from_points().evaluate(0).tolist())
    polyline(points=points)
    

from mmcore.base.sharedstate import serve
serve.start()

changes2()
time.sleep(1)
animate()