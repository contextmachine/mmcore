"""
 
 This example demonstrates the intersection of two NURBS spheres.
"""
import time

from mmcore.construction import cylinder_surface_2pt
from mmcore.geom._nurbs_eval import _tuple_to_nurbs
from mmcore.geom._nurbs_transform import transform_nurbs

from mmcore.numeric.intersection.ssx import ssx

# Creating intersection objects
import numpy as np

start = np.array([0.533136, -2.144876, -1])
end = np.array([2.294869, -0.144876, 0.683482])

surface = cylinder_surface_2pt(start, end, 2.0)
# Curve example
T = np.array([
    [0.0, -1.0, 0.0, 2.0],  # rotate 90° about z and translate by (2,0,0)
    [1.0, 0.0, 0.0, -1.0],
    [0.0, 0.0, 1.0, 0.5],
    [0.0, 0.0, 0.0, 1.0],
])

# Surface example

surface2 = transform_nurbs(surface, T)  # surface is a NURBSSurfaceTuple

s1 = _tuple_to_nurbs(surface)
s2 = _tuple_to_nurbs(surface2)

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
    
    from mmcore.extras.renderer import CADRenderer, Camera
    def draw_ssx(s1,s2, result, renderer=None):
        
        renderer = renderer if renderer is not None else CADRenderer(camera=Camera(zoom=50.0, near=1.))
        renderer.add_nurbs_surface(s1, color=(1.0, 1.0, 1.0))
        renderer.add_nurbs_surface(s2, color=(1.0, 1.0, 1.0), )
        
        for crv, uv1, uv2 in result[0]:
            renderer.add_nurbs_curve(crv, color=(0.0, 1.0, 0.5))
        return renderer

    renderer=draw_ssx(s1, s2, result)

    renderer.run()

except ModuleNotFoundError as err:
    print("mmcore.renderer is not installed, skip preview.")
except ImportError as err:
    print("mmcore.renderer is not installed, skip preview.")
except Exception as err:
    raise err
