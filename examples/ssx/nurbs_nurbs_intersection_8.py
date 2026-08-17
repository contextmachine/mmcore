"""
 
 This example demonstrates the intersection of two NURBS surfaces with partial overlaps.
"""
import time
from mmcore.geom._nurbs_eval import _tuple_to_nurbs


# Creating intersection objects
import numpy as np
from mmcore.geom._nurbs_eval import NURBSSurfaceTuple


s1 = NURBSSurfaceTuple(
    order_u=2,
    order_v=2,
    knot_u=np.array([  0.        ,   0.        , 256.50009777, 256.50009777]),
    knot_v=np.array([  0.        ,   0.        , 259.71657438, 259.71657438]),
    control_points=np.array([[[-128.25004889, -129.85828719,   67.43742325],
            [-128.25004889,  129.85828719,    0.        ]],

           [[ 128.25004889,  -46.98266257,    0.        ],
            [ 128.25004889,  129.85828719,    0.        ]]]),
    weights=np.array([[1., 1.],
           [1., 1.]])
)


s2 = NURBSSurfaceTuple(
    order_u=2,
    order_v=2,
    knot_u=np.array([  0.        ,   0.        , 256.50009777, 256.50009777]),
    knot_v=np.array([  0.        ,   0.        , 259.71657438, 259.71657438]),
    control_points=np.array([[[-128.25004889, -129.85828719,    0.        ],
            [-128.25004889,  129.85828719,    0.        ]],

           [[ 128.25004889, -129.85828719,    0.        ],
            [ 128.25004889,  129.85828719,    0.        ]]]),
    weights=np.array([[1., 1.],
           [1., 1.]])
)



import logging
from examples.ssx.common_helpers import parse_args, save_pkl, draw_ssx, VIEWER_INSTALLED, CurveMaterial, ControlNetMaterial, PointMaterial,surface_material
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
    surface_material.show_wires = False
    inter_curves_mat = CurveMaterial(
        (0.0, 1.0, 0.5, 1.0),
        show_control_net=args.show_inter_cpts,
        control_net_material=ControlNetMaterial((0.0, 1.0, 0.5, 0.7), control_point_material=PointMaterial((0.0, 1.0, 0.5, 0.4), size=8)),
    )

    viewer = draw_ssx(s1, s2, result, surf1_material=surface_material,surf2_material=surface_material,intersection_curves_material=inter_curves_mat)

    viewer.run()
