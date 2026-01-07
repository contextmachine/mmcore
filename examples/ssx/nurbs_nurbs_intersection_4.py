import time

from mmcore.construction import cylinder_surface_2pt,torus

x, y, v, u, z = [
    [[12.359112840551504, -7.5948049557495425, 0.0], [2.656625109045951, 1.2155741170561933, 0.0]],
                 [[7.14384241216015, -6.934735074711716, -0.1073366304415263],
                  [7.0788761013028365, 4.016931402130641, 0.8727530304189204]],
    
                 [[8.072688942425103, -2.3061831591019826, 0.2615779273274319],
                  [7.173685617288537, -3.4427234423361512, 0.4324928834164773],
                  [7.683972288682133, -2.74630545102506, 0.07413871667321925],
                  [7.088944240699163, -4.61458155002528, -0.22460509818398067],
                  [7.304629277158477, -3.9462033818505433, 0.8955725109783643],
                  [7.304629277158477, -3.3362864951018985, 0.8955725109783643],
                  [7.304629277158477, -2.477065729786164, 0.7989970582016114],
                  [7.304629277158477, -2.0988672326949933, 0.7989970582016114]], 0.72648, 1.0]
from mmcore.geom._nurbs_eval import _tuple_to_nurbs
import numpy as np

s1=cylinder_surface_2pt(*np.array(x),radius=u)
s2=cylinder_surface_2pt(*np.array(y),radius=z)


from mmcore.numeric.intersection.ssx import ssx
import logging
logging.basicConfig(level=logging.DEBUG)
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
        from mmcore.construction import nurbs_curve

        def draw_ssx(s1: NURBSSurfaceTuple, s2: NURBSSurfaceTuple, result, renderer=None):
            bb = AABB.from_points(s1.control_points.reshape(-1, 3)).merge(AABB.from_points(s2.control_points.reshape(-1, 3)))
            renderer = renderer if renderer is not None else Viewer(camera=OrbitCamera(target=bb.centroid(), distance=150.0, near=1.0))
            renderer.add_nurbs_surface(s1)
            renderer.add_nurbs_surface(
                s2,
            )

            for branch in result[0]:
                renderer.add_nurbs_curve(branch.curve_xyz, color=(0.0, 1.0, 0.5, 1.0))
                for p in branch.curve_xyz.control_points:
                    renderer.add_point3d(p, color=(0.0, 1.0, 0.5, 0.4), size_px=8)
                renderer.add_nurbs_curve(nurbs_curve(branch.curve_xyz.control_points, 1), color=(0.0, 1.0, 0.5, 0.7))
                # renderer.add_point3d(branch.curve_xyz.end(), color=(0.0, 1.0, 0.5, 1.0), size_px=6)
            for p in result[1]:
                renderer.add_point3d(p.xyz, color=(0.0, 1.0, 0.5, 1.0), size_px=12)

            return renderer

        renderer = draw_ssx(s1, s2, result)

        renderer.run()
except ModuleNotFoundError as err:
        print("mmcore.renderer is not installed, skip preview.")
except ImportError as err:
        print("mmcore.renderer is not installed, skip preview.")
except Exception as err:
        raise err
