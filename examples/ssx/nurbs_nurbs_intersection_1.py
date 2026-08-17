import pickle
import time
from pathlib import Path

import numpy as np

from mmcore.construction import nurbs_curve
from mmcore.nurbs._nurbs_eval import _nurbs_to_tuple
from mmcore.nurbs._nurbs_knots import normalize_knots_surface_inplace
from mmcore.nurbs._core import NURBSSurface
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
normalize_knots_surface_inplace(s1)
normalize_knots_surface_inplace(s2)
import logging
from examples.ssx.common_helpers import parse_args, save_pkl, draw_ssx, VIEWER_INSTALLED, CurveMaterial, ControlNetMaterial, PointMaterial
args = parse_args()
logging.basicConfig(level=getattr(logging, args.loglevel, logging.INFO))
from mmcore.numeric.intersection.ssx import nurbs_ssx

s = time.time()
result = nurbs_ssx(s1, s2, atol=args.atol)

print(f"intersection computed at: {time.time() - s} sec.")
print(len(result['branches']), "branch(s)")
print(len(result['points']), "pts(s)")

if args.save_pkl or args.pkl_path is not None:
    path = save_pkl(s1, s2, result, fp=args.pkl_path)
    print(path.absolute().as_posix())

RENDER = args.viewer and VIEWER_INSTALLED

if RENDER:
    inter_curves_mat = CurveMaterial(
        (0.0, 1.0, 0.5, 1.0),
        show_control_net=args.show_inter_cpts,
        control_net_material=ControlNetMaterial((0.0, 1.0, 0.5, 0.7), control_point_material=PointMaterial((0.0, 1.0, 0.5, 0.4), size=8)),
    )

    viewer = draw_ssx(s1, s2, result, intersection_curves_material=inter_curves_mat)

    viewer.run()
