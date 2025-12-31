import time

import numpy as np

from mmcore.geom._nurbs_eval import _nurbs_to_tuple
from mmcore.geom.nurbs import NURBSSurface
from mmcore.numeric.intersection.ssx import nurbs_ssx
pts1 = np.array(
    [
        [-25.0, -25.0, -10.0],
        [-25.0, -15.0, -5.0],
        [-25.0, -5.0, 0.0],
        [-25.0, 5.0, 0.0],
        [-25.0, 15.0, -5.0],
        [-25.0, 25.0, -10.0],
        [-15.0, -25.0, -8.0],
        [-15.0, -15.0, -4.0],
        [-15.0, -5.0, -4.0],
        [-15.0, 5.0, -4.0],
        [-15.0, 15.0, -4.0],
        [-15.0, 25.0, -8.0],
        [-5.0, -25.0, -5.0],
        [-5.0, -15.0, -3.0],
        [-5.0, -5.0, -8.0],
        [-5.0, 5.0, -8.0],
        [-5.0, 15.0, -3.0],
        [-5.0, 25.0, -5.0],
        [5.0, -25.0, -3.0],
        [5.0, -15.0, -2.0],
        [5.0, -5.0, -8.0],
        [5.0, 5.0, -8.0],
        [5.0, 15.0, -2.0],
        [5.0, 25.0, -3.0],
        [15.0, -25.0, -8.0],
        [15.0, -15.0, -4.0],
        [15.0, -5.0, -4.0],
        [15.0, 5.0, -4.0],
        [15.0, 15.0, -4.0],
        [15.0, 25.0, -8.0],
        [25.0, -25.0, -10.0],
        [25.0, -15.0, -5.0],
        [25.0, -5.0, 2.0],
        [25.0, 5.0, 2.0],
        [25.0, 15.0, -5.0],
        [25.0, 25.0, -10.0],
    ]
)
pts1 = pts1.reshape((6, len(pts1) // 6, 3))
pts2 = np.array(
    [
        [25.0, 14.774795467423544, 5.5476189978794661],
        [25.0, 10.618169208735296, -15.132510312735601],
        [25.0, 1.8288992061686002, -13.545426491756078],
        [25.0, 9.8715747661086723, 14.261864686419623],
        [25.0, -15.0, 5.0],
        [25.0, -25.0, 5.0],
        [15.0, 25.0, 1.8481369394623908],
        [15.0, 15.0, 5.0],
        [15.0, 5.0, -1.4589623860307768],
        [15.0, -5.0, -1.9177595746260625],
        [15.0, -15.0, -30.948650572598954],
        [15.0, -25.0, 5.0],
        [5.0, 25.0, 5.0],
        [5.0, 15.0, -29.589097491066767],
        [3.8028908181980938, 5.0, 5.0],
        [5.0, -5.0, 5.0],
        [5.0, -15.0, 5.0],
        [5.0, -25.0, 5.0],
        [-5.0, 25.0, 5.0],
        [-5.0, 15.0, 5.0],
        [-5.0, 5.0, 5.0],
        [-5.0, -5.0, -27.394523521151221],
        [-5.0, -15.0, 5.0],
        [-5.0, -25.0, 5.0],
        [-15.0, 25.0, 5.0],
        [-15.0, 15.0, -23.968082282285287],
        [-15.0, 5.0, 5.0],
        [-15.0, -5.0, 5.0],
        [-15.0, -15.0, -18.334465891060319],
        [-15.0, -25.0, 5.0],
        [-25.0, 25.0, 5.0],
        [-25.0, 15.0, 14.302789083068138],
        [-25.0, 5.0, 5.0],
        [-25.0, -5.0, 5.0],
        [-25.0, -15.0, 5.0],
        [-25.0, -25.0, 5.0],
    ]
)


pts2 = pts2.reshape((6, len(pts2) // 6, 3))
s21 = NURBSSurface(pts1, (3, 3))
s22 = NURBSSurface(pts2, (3, 3))
s=time.time()
s1,s2=_nurbs_to_tuple(s21),_nurbs_to_tuple(s22)
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
            renderer = renderer if renderer is not None else Viewer(camera=OrbitCamera(target=bb.centroid(), distance=150.0))
            renderer.add_nurbs_surface(s1)
            renderer.add_nurbs_surface(
                s2,

            )

            for branch in result[0]:
                renderer.add_nurbs_curve(branch.curve_xyz, color=(0.0, 1.0, 0.5, 1.0))
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
