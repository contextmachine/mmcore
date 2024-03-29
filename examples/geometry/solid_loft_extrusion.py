# This Api provides an extremely flexible way of updating data. You can pass any part of the parameter dictionary
# structure, the parameters will be updated recursively and only the part of the graph affected by the change
# will be recalculated.
from mmcore.base.params import ParamGraphNode, param_graph_node_native
from mmcore.geom.parametric import Linear
from mmcore.geom.parametric.nurbs import NurbsCurve
from mmcore.geom.parametric.sweep import Sweep

a = ParamGraphNode(dict(x=1000.0, y=2000.0, z=3000.0), name="A")
b = ParamGraphNode(dict(x=-1000.0, y=-2000.0, z=-3000.0), name="B")
c = ParamGraphNode(dict(x=10000.0, y=20000.0, z=30000.0), name="С")
d = ParamGraphNode(dict(x=-11000.0, y=12000.0, z=13000.0), name="D")

from mmcore.geom.materials import ColorRGB


# render_lines.todict(no_attrs=True) will return the complete dictionary of parameters affecting the system.


# I use json.dumps(..., indent=3) to visually print out the whole dictionary, I could also use something like pprint,
# but it"s important to show that the parameters are parsed to the scolar simplest data types.
# We can decompose a system of any complexity into a parameter tree with prime, numbers, strings, boolean values, etc.


@param_graph_node_native
def spiral(radius=2000, high=2000, pitches=12, start=[0, 0, 1]):
    def evaluate(t):
        return np.array(
            [np.cos(t * pitches) * radius * np.cos(t), np.sin(t * pitches) * radius * np.cos(t), (t) * high])

    return NurbsCurve([(np.array(start) + evaluate(t)).tolist() for t in np.linspace(0, 1, 8 * pitches)], dimension=3)


def line_from_points(start, end):
    return Linear.from_two_points(list(start.values()), list(end.values()))


import numpy as np

from mmcore.base.sharedstate import serve, debug_properties

col = ColorRGB(70, 70, 70).decimal
col2 = ColorRGB(157, 75, 75).decimal

# cgraph = ComponentGraph()
# cpt = ControlPointList(points=[ControlPoint(x=0, y=0, z=0, name="pointa", uuid="pointa"),
#                               ControlPoint(x=10, y=0, z=10, name="pointb", uuid="pointb"),
#                               ControlPoint(x=5, y=10, z=5, name="pointc", uuid="pointc")])




# prf=NurbsCurve(np.array(shapely.Polygon(prf.control_points).buffer(5).boundary.coords).tolist())
# prf2=NurbsCurve(np.array(shapely.Polygon(prf2.control_points).buffer(1).boundary.coords).tolist())

#bz = Bezier(cpt)

p1 = np.array([[-23, 40, 0], [23, 40, 0],
               [25, 38, 0],
               [25, -38, 0],
               [23, -40, 0],
               [-23, -40, 0],
               [-25, -38, 0],
               [-25, 38, 0],
               [-23, 40, 0]]) * 10
profile_outer = p1.tolist()

p2 = np.array([[-21.0, 36.5, 0.0],
               [21, 36.5, 0],
               [21.5, 36, 0],
               [21.5, -36, 0],
               [21, -36.5, 0],
               [-21, -36.5, 0],
               [-21.5, -36, 0],
               [-21.5, 36, 0],
               [-21.0, 36.5, 0]]) * 10

#path = bz()
profile_inner = p2.tolist()
profile_outer.reverse()  # Don't mind me I'm just a fool and wrote down the original profile coordinates clockwise
profile_inner.reverse()  # Don't mind me I'm just a fool and wrote down the original profile coordinates clockwise

srf1 = Sweep(name="sweep_a", uuid="test_loft_a", path=spiral, profiles=(profile_outer, profile_inner))
#srf2 = Sweep(name="sweep_b", path=bz, profiles=(profile_outer, profile_inner), uuid="test_loft_b")


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

debug_properties["target"] = "fetch/test_loft_a"
serve.start()

