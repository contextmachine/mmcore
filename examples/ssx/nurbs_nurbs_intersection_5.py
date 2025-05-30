"""
 
 This example demonstrates the application of SSX to objects of different scales. We are using a large flat surface and a tiny cylinder.
 
 It may seem simple, but it is actually not the most trivial case.
 
"""
import time
import numpy as np
from mmcore.geom._nurbs_eval import _tuple_to_nurbs, _nurbs_to_tuple, NURBSSurfaceTuple
from mmcore.construction import cylinder_surface_2pt
from mmcore.numeric.intersection.ssx import ssx

# Creating intersection objects
st1 = cylinder_surface_2pt(np.array([0.0, 0.0, -3.0]), np.array([0.0, 0.0, 3.0]), radius=1.0)
st2 = NURBSSurfaceTuple(
    order_u=2,
    order_v=2,
    knot_u=np.array([0.0, 0.0, 150.0, 150.0]),
    knot_v=np.array([0.0, 0.0, 150.0, 150.0]),
    control_points=np.array([[[-75.0, -75.0, 1.0], [-75.0, 75.0, 1.0]], [[75.0, -75.0, 1.0], [75.0, 75.0, 1.0]]]),
    weights=np.array([[1.0, 1.0], [1.0, 1.0]]),
)

s1 = _tuple_to_nurbs(st1)
s2 = _tuple_to_nurbs(st2)

# Perform SSX
start_time = time.time()
result = ssx(s1, s2, tol=1e-7, spt=0.001)

print(f"intersection computed at: {time.time() - start_time} sec.")

# Printing and rendering results
print(f"\n({s1} X \n\t{s2}):")

for i, (spatial, uv1, uv2) in enumerate(result[0]):
    print(f"\t{i + 1}. {spatial}, {uv1}, {uv2}")
    cpts = (spatial.control_points).tolist()
    cpts_repr = repr(cpts)
    # if len(cpts)>4:
    #    cpts_repr=f'[{cpts[1]}, {cpts[2]}, ... , {cpts[-2]}, {cpts[-1]}]'
    print(f"\t\tcontrol points: {cpts_repr}")
    print(f"\t\tdegree: {spatial.degree}")


try:
    from mmcore.renderer.renderer3dv2 import CADRenderer, Camera

    print(dir(Camera))
    centr = np.average(s1.control_points_flat, axis=0)
    renderer = CADRenderer(camera=Camera(zoom=50.0, near=1.))

    renderer.add_nurbs_surface(s1, color=(1.0, 1.0, 1.0))
    renderer.add_nurbs_surface(s2, color=(1.0, 1.0, 1.0))

    for crv, uv1, uv2 in result[0]:
        renderer.add_nurbs_curve(crv, color=(0.0, 1.0, 0.5))

    renderer.run()

except ModuleNotFoundError as err:
    print("mmcore.renderer is not installed, skip preview.")
except ImportError as err:
    print("mmcore.renderer is not installed, skip preview.")
except Exception as err:
    raise err
