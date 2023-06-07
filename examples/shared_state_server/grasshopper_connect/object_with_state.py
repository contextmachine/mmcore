from mmcore.geom.parametric.nurbs import NurbsCurve
from mmcore.base.sharedstate import serve
import numpy as np

serve.start()  # Server run in non-blocked thread on http://localhost:7711

initial = [[0, 0, 0], [1, 2, -1], [4, 4, 1], [-3, 4, 2], [-6, 4, 2]]
# It dynamically updates when you will create a new objects or change exist
nc = NurbsCurve(initial)

print(f"Starting with initial control points value: {nc.control_points}")

@serve.resolver
def nurbs_resolver(**kws):
    n = NurbsCurve(kws.get('x'))
    return n.evaluate(np.linspace(0, 1, 100)).tolist()
@serve.resolver
def nurbs_state_resolver(**kws):
    nc.control_points = kws.get("x")
    nc.resolve()
    print(f"Some remote changes. New control points value: {nc.control_points}")
    return nc.evaluate(np.linspace(0, 1, 100)).tolist()
