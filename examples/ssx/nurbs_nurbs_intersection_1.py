import time

import numpy as np

from mmcore.geom.nurbs import NURBSSurface
from mmcore.numeric.intersection.ssx import ssx
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
result=ssx(s21,s22,0.001)



print(f'intersection computed at: {time.time() - s} sec.')


print(f'\n({s21} X \n\t{s22}):')

for i, (spatial, uv1, uv2) in enumerate(result):
        print(f'\t{i + 1}. {spatial}, {uv1}, {uv2}')
        cpts=(spatial.control_points).tolist()
        cpts_repr = repr(cpts)
        if len(cpts)>4:
            cpts_repr=f'[{cpts[1]}, {cpts[2]}, ... , {cpts[-2]}, {cpts[-1]}]'
        print(f'\t\tcontrol points: {cpts_repr}')
        print(f'\t\tdegree: {spatial.degree}')
with open("ssx1.txt",'w') as tf:
    for i, (spatial, uv1, uv2) in enumerate(result):
            cpts = (spatial.control_points).tolist()
            tf.write(repr(cpts))

try:
    from mmcore.renderer.renderer3dv2 import CADRenderer,Camera

    print(dir(Camera))
    centr=np.average(s21.control_points_flat, axis=0)
    renderer=CADRenderer(camera=Camera( zoom=75.
        )
    )

    renderer.add_nurbs_surface(s21,color=(1.,1.,1.))
    renderer.add_nurbs_surface(s22,color=(1.,1.,1.))

    for (crv,uv1,uv2) in result:
        renderer.add_nurbs_curve(crv, color=(0.,1.,0.5))


    renderer.run()
except ModuleNotFoundError as err:
    print("mmcore.renderer is not installed, skip preview.")
except ImportError as err:
    print("mmcore.renderer is not installed, skip preview.")
except Exception as err:
    raise err
