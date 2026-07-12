import numpy as np
from mmcore.geom._nurbs_eval import NURBSCurveTuple, NURBSSurfaceTuple, _tuple_to_nurbs, evaluate_nurbs_curve

import time


import rich

from mmcore.geom._nurbs_knots import trim_curve
from mmcore.numeric import evaluate_curvature_vec
from mmcore.numeric.approx import adaptive_curve_sampler
from mmcore.numeric.intersection.csx import nurbs_csx_v2
from mmcore.numeric.intersection.csx._ncsx4 import  nurbs_csx
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    ssx_params = parser.add_argument_group(title="CSX Parameters")
    ssx_params.add_argument("--atol", type=float, default=1e-3)
    ssx_params.add_argument("--angle_tol", type=float, default=0.052)

    general_params = parser.add_argument_group(title="General")
    general_params.add_argument('--viewer', action='store_true')

    return parser.parse_args()


args = parse_args()

curve = NURBSCurveTuple(
    order=3,
    knot=np.array([  0.        ,   0.        ,   0.        ,  91.83300275,
           148.16516477, 206.24102425, 296.33032955, 296.33032955,
           296.33032955]),
    control_points=np.array([[   0.64617093, -137.        ,  -24.        ],
           [   0.38629333,  -87.41587386,    0.        ],
           [  35.63100531,  -21.05150855,    0.        ],
           [  41.52768692,   23.7340877 ,    0.        ],
           [  20.09543746,   74.05471129,    0.        ],
           [  20.64617093,  101.40810621,   16.        ]]),
    weights=np.array([1., 1., 1., 1., 1., 1.])
)

surface = NURBSSurfaceTuple(
    order_u=4,
    order_v=3,
    knot_u=np.array([ 0.        ,  0.        ,  0.        ,  0.        , 40.22437072,
           40.22437072, 40.22437072, 40.22437072]),
    knot_v=np.array([  0.        ,   0.        ,   0.        , 153.56627554,
           307.13255109, 307.13255109, 307.13255109]),
    control_points=np.array([[[ -49.35382907, -137.        ,    0.        ],
            [  30.89785834,  -57.        ,    0.        ],
            [  46.17096231,   59.        ,    0.        ],
            [ -20.35382907,   97.40810621,    0.        ]],

           [[ -50.02049574, -137.        ,   11.        ],
            [  23.23119168,  -57.        ,   11.        ],
            [  38.50429565,   59.        ,   11.        ],
            [ -21.02049574,   97.40810621,   11.        ]],

           [[ -50.68716241, -137.        ,   22.        ],
            [  15.56452501,  -57.        ,   22.        ],
            [  30.83762898,   59.        ,   22.        ],
            [ -21.68716241,   97.40810621,   22.        ]],

           [[ -51.35382907, -137.        ,   33.        ],
            [   7.89785834,  -57.        ,   33.        ],
            [  23.17096231,   59.        ,   33.        ],
            [ -22.35382907,   97.40810621,   33.        ]]]),
    weights=np.array([[1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.]])
)

#s = time.time()
#result = nurbs_csx(_tuple_to_nurbs(curve), _tuple_to_nurbs(surface))
#print(f"CSX v1 performed at: {time.time()-s} secs.")
#over=[]
#isol=[]
#print(result)
#for tp,item,uv in result:
#    if tp =='overlap':
#        over.append(item)
#    else:
#        isol.append(item)
#print('isolated:')
#rich.print(isol)
#print('overlaps:')
#rich.print(over)

s = time.time()
isolated,overlaps = nurbs_csx(curve, surface, tol=args.atol)
print(f"CSX v4 performed at: {time.time()-s} secs.")


print('isolated:')

if isolated is not None:
    rich.print(isolated)
print('overlaps:')
if overlaps is not None:
    rich.print(overlaps)
RENDERER=False
if args.viewer:
    try:
        from mmcore.extras.renderer.renderer3d import Viewer,OrbitCamera
        viewer=Viewer(camera=OrbitCamera(near=1,far=1e+9))
        primary_color=(*(np.array([250, 102, 166])/255).tolist(),1)
        srf = viewer.add_nurbs_surface(surface, color=(0.7,0.7,0.7,1),surface_color=(0.5, 0.5, 0.9, 0.05),)

        if isolated is not None:
            uvs = []
            for pt in isolated:
                viewer.add(pt['point'], color=(0.0, 1.0, 0.5, 1.0), size_px=13)

        if overlaps is not None:

            for overlap in overlaps:
                t0,t1=overlap['t_range']

                viewer.add(evaluate_nurbs_curve(curve, t0,d_order=0)['C'], color=(0.0, 1.0, 0.5, 1.0), size_px=6)
                viewer.add(evaluate_nurbs_curve(curve, t1,d_order=0)['C'], color=(0.0, 1.0, 0.5, 1.0), size_px=6)

            for o in overlaps:

                t0 = o["t_range"][0]
                t1 = o["t_range"][-1]

                pts=np.linspace(t0,t1,800)
                for t in pts:

                    evl=evaluate_nurbs_curve(curve,t,d_order=0)
                    viewer.add_point3d(evl['C'],color=(0.0, 1.0, 0.5, 1.0), size_px=3)
        viewer.run()


    except ModuleNotFoundError as err:
        print("mmcore.renderer is not installed, skip preview.")
    except ImportError as err:
        print("mmcore.renderer is not installed, skip preview.")
    except Exception as err:
        raise err
