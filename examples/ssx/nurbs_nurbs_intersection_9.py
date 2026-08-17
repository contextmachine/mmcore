"""
 
 This example demonstrates the intersection of two NURBS spheres.
"""
import time

from mmcore.construction import cylinder_surface_2pt
from mmcore.nurbs._nurbs_eval import _tuple_to_nurbs, NURBSSurfaceTuple
from mmcore.nurbs._nurbs_transform import transform_nurbs
from mmcore.numeric.bvh.lbvh import AABB
from mmcore.nurbs._core import NURBSSurface


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
