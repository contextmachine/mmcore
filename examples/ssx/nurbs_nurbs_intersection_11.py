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

import numpy as np
from mmcore.geom._nurbs_eval import NURBSSurfaceTuple


s1 = NURBSSurfaceTuple(
    order_u=4,
    order_v=4,
    knot_u=np.array([   0.00000000,    0.00000000,    0.00000000,    0.00000000,
            584.56714755, 1169.13429511, 1169.13429511, 1169.13429511,
           1169.13429511]),
    knot_v=np.array([  0.00000000,   0.00000000,   0.00000000,   0.00000000,
           675.00000000, 675.00000000, 675.00000000, 675.00000000]),
    control_points=np.array([[[2488.56941006, 1499.08843254,    0.00000000],
            [2394.60779838, 1603.70965174,    0.00000000],
            [2394.60779838, 1675.09483487,    0.00000000],
            [2488.56941006, 1779.71605406,    0.00000000]],

           [[2582.53102174, 1394.46721335,    0.00000000],
            [2550.34379877, 1515.98031620,  152.12435146],
            [2550.34379877, 1762.82417041,  152.12435146],
            [2582.53102174, 1884.33727326,    0.00000000]],

           [[2754.10779838, 1284.40224330,    0.00000000],
            [2754.10779838, 1466.26757810,  257.50000000],
            [2754.10779838, 1812.53690851,  257.50000000],
            [2754.10779838, 1994.40224330,    0.00000000]],

           [[2925.68457502, 1394.46721335,    0.00000000],
            [2957.87179798, 1515.98031620,  152.12435146],
            [2957.87179798, 1762.82417041,  152.12435146],
            [2925.68457502, 1884.33727326,    0.00000000]],

           [[3019.64618670, 1499.08843254,    0.00000000],
            [3113.60779838, 1603.70965174,    0.00000000],
            [3113.60779838, 1675.09483487,    0.00000000],
            [3019.64618670, 1779.71605406,    0.00000000]]]),
    weights=np.array([[1.00000000, 1.00000000, 1.00000000, 1.00000000],
           [1.00000000, 1.00000000, 1.00000000, 1.00000000],
           [1.00000000, 1.00000000, 1.00000000, 1.00000000],
           [1.00000000, 1.00000000, 1.00000000, 1.00000000],
           [1.00000000, 1.00000000, 1.00000000, 1.00000000]])
)
import numpy as np
from mmcore.geom._nurbs_eval import NURBSSurfaceTuple


s2 = NURBSSurfaceTuple(
    order_u=2,
    order_v=2,
    knot_u=np.array([  0.00000000,   0.00000000, 854.10934693, 854.10934693]),
    knot_v=np.array([   0.00000000,    0.00000000, 1119.84031048, 1119.84031048]),
    control_points=np.array([[[2372.10906381, 1470.47565841,  170.90189109],
            [2614.15055984, 2054.83790061,  228.66096126]],

           [[2953.02855072, 1248.46099847,  -17.28737813],
            [3195.07004676, 1832.82324067,   40.47169204]]]),
    weights=np.array([[1.00000000, 1.00000000],
           [1.00000000, 1.00000000]])
)


s1,s2=s2,s1
import logging
from examples.ssx.common_helpers import parse_args, save_pkl, draw_ssx, VIEWER_INSTALLED, CurveMaterial, ControlNetMaterial, PointMaterial
args = parse_args()
logging.basicConfig(level=getattr(logging, args.loglevel, logging.INFO))
from mmcore.numeric.intersection.ssx import nurbs_ssx

s = time.time()
result = nurbs_ssx(s1, s2, atol=0.001)

print(f"intersection computed at: {time.time() - s} sec.")
print(len(result['branches']), "branch(s)")
print(len(result['points']), "pts(s)")

if args.save_pkl or args.pkl_path is not None:
    path = save_pkl(s1, s2, result, fp=args.pkl_path)
    print(path.absolute().as_posix())

RENDER = args.viewer

if RENDER:
    inter_curves_mat = CurveMaterial(
        (0.0, 1.0, 0.5, 1.0),
        show_control_net=args.show_inter_cpts,
        control_net_material=ControlNetMaterial((0.0, 1.0, 0.5, 0.7), control_point_material=PointMaterial((0.0, 1.0, 0.5, 0.4), size=8)),
    )

    viewer = draw_ssx(s1, s2, result, intersection_curves_material=inter_curves_mat)

    viewer.run()
