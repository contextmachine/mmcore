"""
 
 This example demonstrates the intersection of a flat surface and a cylindrical rational surface at an angle.
 
"""
import time
import numpy as np
from mmcore.geom._nurbs_eval import _tuple_to_nurbs, NURBSSurfaceTuple
from mmcore.construction import cylinder_surface_2pt

# Creating intersection objects
s1 = cylinder_surface_2pt(np.array([40, 40, -10]), np.array([42.5, 42.5, 10.0]), radius=50.0)
s2 = NURBSSurfaceTuple(
    order_u=2,
    order_v=2,
    knot_u=np.array([0.0, 0.0, 150.0, 150.0]),
    knot_v=np.array([0.0, 0.0, 150.0, 150.0]),
    control_points=np.array([[[-75.0, -75.0, 1.0], [-75.0, 75.0, 1.0]], [[75.0, -75.0, 1.0], [75.0, 75.0, 1.0]]]),
    weights=np.array([[1.0, 1.0], [1.0, 1.0]]),
)



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
