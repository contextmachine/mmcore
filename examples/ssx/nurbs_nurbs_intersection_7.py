"""
 
 This example demonstrates the intersection of two NURBS spheres.
"""
import time
from mmcore.geom._nurbs_eval import _tuple_to_nurbs

from mmcore.numeric.intersection.ssx import ssx

# Creating intersection objects
import numpy as np
from mmcore.geom._nurbs_eval import NURBSSurfaceTuple


s1 = NURBSSurfaceTuple(
    order_u=3,
    order_v=3,
    knot_u=np.array([ 0.        ,  0.        ,  0.        , 17.27875959, 17.27875959,
           34.55751919, 34.55751919, 51.83627878, 51.83627878, 69.11503838,
           69.11503838, 69.11503838]),
    knot_v=np.array([-17.27875959, -17.27875959, -17.27875959,  -0.        ,
            -0.        ,  17.27875959,  17.27875959,  17.27875959]),
    control_points=np.array([[[  6.,   0., -11.],
            [ 17.,   0., -11.],
            [ 17.,   0.,  -0.],
            [ 17.,   0.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [ 17.,  11., -11.],
            [ 17.,  11.,  -0.],
            [ 17.,  11.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [  6.,  11., -11.],
            [  6.,  11.,  -0.],
            [  6.,  11.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [ -5.,  11., -11.],
            [ -5.,  11.,  -0.],
            [ -5.,  11.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [ -5.,   0., -11.],
            [ -5.,   0.,  -0.],
            [ -5.,   0.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [ -5., -11., -11.],
            [ -5., -11.,  -0.],
            [ -5., -11.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [  6., -11., -11.],
            [  6., -11.,  -0.],
            [  6., -11.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [ 17., -11., -11.],
            [ 17., -11.,  -0.],
            [ 17., -11.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [ 17.,   0., -11.],
            [ 17.,   0.,  -0.],
            [ 17.,   0.,  11.],
            [  6.,   0.,  11.]]]),
    weights=np.array([[1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ]])
)




s2 = NURBSSurfaceTuple(
    order_u=3,
    order_v=3,
    knot_u=np.array([ 0.        ,  0.        ,  0.        , 17.27875959, 17.27875959,
           34.55751919, 34.55751919, 51.83627878, 51.83627878, 69.11503838,
           69.11503838, 69.11503838]),
    knot_v=np.array([-17.27875959, -17.27875959, -17.27875959,  -0.        ,
            -0.        ,  17.27875959,  17.27875959,  17.27875959]),
    control_points=np.array([[[ 11.,   0., -13.],
            [ 22.,   0., -13.],
            [ 22.,   0.,  -2.],
            [ 22.,   0.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [ 22.,  11., -13.],
            [ 22.,  11.,  -2.],
            [ 22.,  11.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [ 11.,  11., -13.],
            [ 11.,  11.,  -2.],
            [ 11.,  11.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [  0.,  11., -13.],
            [  0.,  11.,  -2.],
            [  0.,  11.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [  0.,   0., -13.],
            [  0.,   0.,  -2.],
            [  0.,   0.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [  0., -11., -13.],
            [  0., -11.,  -2.],
            [  0., -11.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [ 11., -11., -13.],
            [ 11., -11.,  -2.],
            [ 11., -11.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [ 22., -11., -13.],
            [ 22., -11.,  -2.],
            [ 22., -11.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [ 22.,   0., -13.],
            [ 22.,   0.,  -2.],
            [ 22.,   0.,   9.],
            [ 11.,   0.,   9.]]]),
    weights=np.array([[1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ]])
)

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
