import argparse
import time
from pathlib import Path

import numpy as np
import rich


from mmcore.geom._nurbs_eval import NURBSCurveTuple, NURBSSurfaceTuple, _tuple_to_nurbs, evaluate_nurbs_curve
from mmcore.geom.nurbs_iso import extract_isocurve
from mmcore.numeric.closest_point import nurbs_surface_closest_point
from mmcore.numeric.intersection.csx import nurbs_csx_v2, nurbs_csx
import logging
logging.basicConfig(level=logging.DEBUG)
curve = NURBSCurveTuple(
    order=4,
    knot=np.array(
        [
            -2.67615298,
            -2.67615298,
            -2.67615298,
            -2.67615298,
            0.0,
            0.0,
            0.0,
            3.12101814,
            3.12101814,
            3.12101814,
            6.88039589,
            6.88039589,
            6.88039589,
            6.88039589,
        ]
    ),
    control_points=np.array(
        [
            [-48.0003111, 64.08408847, 0.0],
            [-48.89236209, 64.08408847, 0.0],
            [-49.78441309, 64.08408847, 0.0],
            [-50.67646408, 64.08408847, 0.0],
            [-51.1718386, 64.99891638, 0.0],
            [-51.66721312, 65.91374429, 0.0],
            [-52.16258764, 66.82857221, 0.0],
            [-52.58835156, 67.61484744, 0.0],
            [-53.36295339, 69.04533557, 0.0],
            [-58.19051474, 67.75179441, 0.0],
        ]
    ),
    weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
)

surface = NURBSSurfaceTuple(
    order_u=4,
    order_v=3,
    knot_u=np.array([ 0.        ,  0.        ,  0.        ,  0.        , 11.15682108,
           11.15682108, 11.15682108, 11.15682108]),
    knot_v=np.array([0.        , 0.        , 0.        , 1.57079633, 1.57079633,
           3.14159265, 3.14159265, 4.71238898, 4.71238898, 6.28318531,
           6.28318531, 6.28318531]),
    control_points=np.array([[[-50.55017856,  62.5128825 ,   0.93183021],
            [-51.36958836,  62.06917638,   1.29472477],
            [-51.6887016 ,  61.89637824,   0.36289455],
            [-52.00781484,  61.72358009,  -0.56893565],
            [-51.18840503,  62.1672862 ,  -0.93183021],
            [-50.36899523,  62.61099232,  -1.29472477],
            [-50.04988199,  62.78379047,  -0.36289456],
            [-49.73076875,  62.95658861,   0.56893565],
            [-50.55017856,  62.5128825 ,   0.93183021]],

           [[-52.32101251,  65.78315229,   0.93183021],
            [-53.14042231,  65.33944617,   1.29472477],
            [-53.45953555,  65.16664803,   0.36289455],
            [-53.77864879,  64.99384988,  -0.56893566],
            [-52.95923899,  65.43755599,  -0.93183022],
            [-52.13982919,  65.88126211,  -1.29472477],
            [-51.82071595,  66.05406025,  -0.36289456],
            [-51.50160271,  66.2268584 ,   0.56893566],
            [-52.32101251,  65.78315229,   0.93183021]],

           [[-54.09184647,  69.05342208,   0.93183021],
            [-54.91125627,  68.60971596,   1.29472476],
            [-55.2303695 ,  68.43691782,   0.36289456],
            [-55.54948274,  68.26411967,  -0.56893566],
            [-54.73007294,  68.70782578,  -0.93183021],
            [-53.91066314,  69.1515319 ,  -1.29472477],
            [-53.5915499 ,  69.32433004,  -0.36289455],
            [-53.27243666,  69.49712819,   0.56893565],
            [-54.09184647,  69.05342208,   0.93183021]],

           [[-55.86268042,  72.32369187,   0.93183021],
            [-56.68209022,  71.87998575,   1.29472477],
            [-57.00120346,  71.70718761,   0.36289455],
            [-57.3203167 ,  71.53438946,  -0.56893565],
            [-56.50090689,  71.97809557,  -0.93183021],
            [-55.68149709,  72.42180169,  -1.29472477],
            [-55.36238385,  72.59459984,  -0.36289456],
            [-55.04327061,  72.76739798,   0.56893565],
            [-55.86268042,  72.32369187,   0.93183021]]]),
    weights=np.array([[1.        , 0.70710678, 1.        , 0.70710678, 1.        ,
            0.70710678, 1.        , 0.70710678, 1.        ],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ,
            0.70710678, 1.        , 0.70710678, 1.        ],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ,
            0.70710678, 1.        , 0.70710678, 1.        ],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ,
            0.70710678, 1.        , 0.70710678, 1.        ]])
)
def parse_args():
    parser = argparse.ArgumentParser()
    ssx_params=parser.add_argument_group(title="SSX Parameters")
    ssx_params.add_argument("--atol", type=float, default=1e-3)
    

    general_params=parser.add_argument_group(title="General")
    general_params.add_argument('--viewer', action='store_true')



    return parser.parse_args()
args = parse_args()

#s = time.time()
#result = nurbs_csx(_tuple_to_nurbs(curve), _tuple_to_nurbs(surface))
#print(f"CSX v1 performed at: {time.time()-s} secs.")
#overlaps=[]
#isol=[]
#print(result)
#for tp,item,uv in result:
#    if tp =='overlap':
#        overlaps.append(item)
#    else:
#        isol.append(item)
#print('isolated:')
#rich.print(isol)
#print('overlaps:')
#rich.print(overlaps)
s = time.time()
isolated,overlaps = nurbs_csx_v2(curve, surface, tol=args.atol,overlap_dist_tol=args.atol)
print(f"CSX v2 performed at: {time.time()-s} secs.")

#print('\n\n',result,'\n\n')
print('isolated:')
rich.print(isolated['point'].tolist())
print('overlaps:')
rich.print(overlaps['point'].tolist())

if args.viewer:
    try:

        from mmcore.extras.renderer.renderer3d import Viewer,OrbitCamera


        viewer=Viewer(camera=OrbitCamera(target=  surface.control_points.reshape(-1,3).mean(axis=0)))

        srf = viewer.add_nurbs_surface(surface, color=(0.7,0.7,0.7,1),surface_color=(0.5, 0.5, 0.9, 0.1), v_count=4)
        crv=viewer.add(curve, color=(0.9, 0.9, 0.9, 1.0))
        if isolated is not None:
            uvs=[]
            for pt in isolated['point']:


                viewer.add(pt, color=(0.0, 1.0, 0.5,1.0),size_px=6)


        if overlaps is not None:

            for start,end in overlaps['point']:
                viewer.add(start, color=(0.0, 1.0, 0.5, 1.0), size_px=6)
                viewer.add(end, color=(0.0, 1.0, 0.5, 1.0), size_px=6)

            for o in overlaps["t"]:

                t0 = o[0]
                t1 = o[-1]
                start = evaluate_nurbs_curve(curve, t0, d_order=0)
                end = evaluate_nurbs_curve(curve, t1, d_order=0)










                points=[]
                ders=[]
                offset=1

                pts=np.linspace(t0,t1,500)
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
