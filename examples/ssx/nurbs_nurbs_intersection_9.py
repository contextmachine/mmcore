"""
 
 This example demonstrates the intersection of two NURBS spheres.
"""
import time

from mmcore.construction import cylinder_surface_2pt
from mmcore.geom._nurbs_eval import _tuple_to_nurbs, NURBSSurfaceTuple
from mmcore.geom._nurbs_transform import transform_nurbs
from mmcore.geom.bvh.lbvh import AABB
from mmcore.geom.nurbs import NURBSSurface

from mmcore.numeric.intersection.ssx import ssx

# Creating intersection objects
import numpy as np

start = np.array([0.533136, -2.144876, -1])
end = np.array([2.294869, -0.144876, 0.683482])

s1 = cylinder_surface_2pt(start, end, 2.0)
# Curve example
T = np.array([
    [0.0, -1.0, 0.0, 2.0],  # rotate 90° about z and translate by (2,0,0)
    [1.0, 0.0, 0.0, -1.0],
    [0.0, 0.0, 1.0, 0.5],
    [0.0, 0.0, 0.0, 1.0],
])

# Surface example

s2 = transform_nurbs(s1, T)  # surface is a NURBSSurfaceTuple


from mmcore.numeric.intersection.ssx import nurbs_ssx
s=time.time()
result=nurbs_ssx(s1,s2,atol=1e-3)


print(f'intersection computed at: {time.time() - s} sec.')


print(f'\n({s1} X \n\t{s2}):')

for i, branch in enumerate(result[0]):
            print(f'\t{i + 1}. {branch}')
            cpts=(branch.curve_xyz.control_points).tolist()
            cpts_repr = repr(cpts)
            if len(cpts)>4:
                cpts_repr=f'[{cpts[1]}, {cpts[2]}, ... , {cpts[-2]}, {cpts[-1]}]'
            print(f'\t\tcontrol points: {cpts_repr}')
            print(f'\t\tdegree: {branch.curve_xyz.degree}')


RENDER=True
try:
    if RENDER:
        from mmcore.geom.bvh.lbvh import AABB
        from mmcore.geom._nurbs_eval import _tuple_to_nurbs, NURBSSurfaceTuple, _nurbs_to_tuple
        from mmcore.extras.renderer.renderer3d import Viewer, OrbitCamera

        def draw_ssx(s1: NURBSSurfaceTuple, s2: NURBSSurfaceTuple, result, renderer=None):
            bb = AABB.from_points(s1.control_points.reshape(-1, 3)).merge(AABB.from_points(s2.control_points.reshape(-1, 3)))
            renderer = renderer if renderer is not None else Viewer(camera=OrbitCamera(target=bb.centroid(), distance=50.0, near=1.0))
            renderer.add_nurbs_surface(s1, color=(1.0, 1.0, 1.0, 1.0))
            renderer.add_nurbs_surface(
                s2,
                color=(1.0, 1.0, 1.0, 1.0),
            )

            for branch in result[0]:
                renderer.add_nurbs_curve(branch.curve_xyz, color=(0.0, 1.0, 0.5,1.))
            for p in result[1]:
                renderer.add_point3d(p.xyz, color=(0.0, 1.0, 0.5,1.), size_px=12)

            return renderer

        renderer = draw_ssx(s1, s2, result)

        renderer.run()
except ModuleNotFoundError as err:
        print("mmcore.renderer is not installed, skip preview.")
except ImportError as err:
        print("mmcore.renderer is not installed, skip preview.")
except Exception as err:
        raise err
