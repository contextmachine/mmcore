import time

from mmcore.construction import cylinder_surface_2pt


x, y, v, u, z = [
    [[12.359112840551504, -7.5948049557495425, 0.0], [2.656625109045951, 1.2155741170561933, 0.0]],
    [[7.14384241216015, -6.934735074711716, -0.1073366304415263], [7.0788761013028365, 4.016931402130641, 0.8727530304189204]],
    [
        [8.072688942425103, -2.3061831591019826, 0.2615779273274319],
        [7.173685617288537, -3.4427234423361512, 0.4324928834164773],
        [7.683972288682133, -2.74630545102506, 0.07413871667321925],
        [7.088944240699163, -4.61458155002528, -0.22460509818398067],
        [7.304629277158477, -3.9462033818505433, 0.8955725109783643],
        [7.304629277158477, -3.3362864951018985, 0.8955725109783643],
        [7.304629277158477, -2.477065729786164, 0.7989970582016114],
        [7.304629277158477, -2.0988672326949933, 0.7989970582016114],
    ],
    0.72648,
    1.0,
]

import numpy as np

try:
    import rich

    print = rich.print
except ImportError:
    pass
s1 = cylinder_surface_2pt(*np.array(x), radius=u)
s2 = cylinder_surface_2pt(*np.array(y), radius=z)


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
