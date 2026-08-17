"""
 
 This example demonstrates the intersection of two NURBS spheres.
"""
import time

from mmcore.construction import cylinder_surface_2pt
from mmcore.geom._nurbs_eval import _tuple_to_nurbs, NURBSSurfaceTuple
from mmcore.geom._nurbs_transform import transform_nurbs
from mmcore.numeric.bvh.lbvh import AABB
from mmcore.geom.nurbs import NURBSSurface
from mmcore.construction import nurbs_surface

# Creating intersection objects
import numpy as np


s1=nurbs_surface(np.array([[[0.0, 0.0, 10.0], [5.0, 5.0, 10.0], [5.0, 10.0, 10.0], [0.0, 15.0, 10.0]], [[5.0, 0.0, 0.0], [10.0, 5.0, 0.0], [10.0, 10.0, 0.0], [5.0, 15.0, 0.0]], [[10.0, 0.0, 10.0], [15.0, 5.0, 10.0], [15.0, 10.0, 10.0], [10.0, 15.0, 10.0]]]))
s2=nurbs_surface(np.array([[[0.0, 0.0, 0.0], [5.0, 5.0, 0.0], [5.0, 10.0, 0.0], [0.0, 15.0, 0.0]], [[5.0, 0.0, 10.0], [10.0, 5.0, 10.0], [10.0, 10.0, 10.0], [5.0, 15.0, 10.0]], [[10.0, 0.0, 0.0], [15.0, 5.0, 0.0], [15.0, 10.0, 0.0], [10.0, 15.0, 0.0]]]))



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
