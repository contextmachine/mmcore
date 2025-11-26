import numpy as np


from mmcore.construction import nurbs_curve,NURBSCurveTuple
from mmcore.numeric.intersection.ccx import nurbs_ccx
from mmcore.geom._nurbs_knots import split_curve_multiple,trim_curve
curve1:NURBSCurveTuple = nurbs_curve(np.array(
        [
            [-19.77608536, 23.10065701, 0.0],
            [-14.86834768, 28.69713066, 0.0],
            [-5.8568525, 25.12677787, 0.0],
            [-12.62581769, 15.26478654, 0.0],
        ]
    ))
curve2:NURBSCurveTuple = nurbs_curve(np.array(
        [
            [-22.0315362, 18.75969713, 0.0],
            [-19.42270945, 28.2502867, 0.0],
            [-8.46791623, 27.56878356, 0.0],
            [-10.43007782, 19.78973126, 0.0],
        ]
    ))

isolated,overlaps=nurbs_ccx(curve1,curve2)
print(overlaps)

overs=[]

crvs=[]
def construct_overlap_representation(curve1_i,curve2_i, over,curves, crv_intervals_map):
    overlap_curves = []
    curve1=curves[curve1_i]


    t0, s0 = over['uv_path'][0]
    t1, s1 = over['uv_path'][-1]



    overlap_curves.append(trim_curve(curve1, t0, t1))

    if np.isclose(t0,curve1.interval()[0]):
        ...
    else:
        crvs.append(trim_curve(curve1,curve1.interval()[0],t0 ))
for o in overlaps:
    t0,s0=o['uv_path'][0]
    t1,s1=o['uv_path'][-1]
    overs.append(trim_curve(curve1, t0, t1))
    if np.isclose(t0,curve1.interval()[0]):
        ...
    else:
        crvs.append(trim_curve(curve1,curve1.interval()[0],t0 ))

    if np.isclose(t1,curve1.interval()[1]):
        ...
    else:
        crvs.append(trim_curve(curve1,t1,curve1.interval()[1] ))
    if np.isclose(s0, curve2.interval()[0]):
        ...
    else:
        crvs.append(trim_curve(curve2, curve2.interval()[0], s0))

    if np.isclose(s1, curve2.interval()[1]):
        ...
    else:
        crvs.append(trim_curve(curve2, s1, curve1.interval()[1]))

from mmcore.numeric.intersection.ccx import nurbs_ccx,nurbs_ccx_multiple
from mmcore.extras.renderer.renderer3d import Viewer,OrbitCamera




viewer=Viewer(camera=OrbitCamera())
for crv in crvs:
    viewer.add(crv,color=(0.7, 0.9, 1.0, 1.0))
for overlap_segm,overlap in zip(overs,overlaps):
    viewer.add(overlap_segm,color=(0.0, 1.0, 0.5, 1.0))
    viewer.add(overlap['xyz_path'][0],color=(0.0, 1.0, 0.5, 1.0),size_px=12)
    viewer.add(overlap['xyz_path'][-1], color=(0.0, 1.0, 0.5, 1.0), size_px=12)
viewer.run()