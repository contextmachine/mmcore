"""


"""
import argparse
import time

import rich

from mmcore.geom._nurbs_eval import _tuple_to_nurbs, _curve_interval, evaluate_nurbs_curve
from mmcore.geom._nurbs_knots import trim_curve
from mmcore.numeric.intersection.csx._ncsx4 import nurbs_csx
import logging
from mmcore.geom.nurbs_iso import extract_surface_boundaries_tuple
# Creating intersection objects
import numpy as np
from mmcore.geom._nurbs_eval import NURBSSurfaceTuple
def parse_args():
    parser = argparse.ArgumentParser()
    ssx_params = parser.add_argument_group(title="CSX Parameters")
    ssx_params.add_argument("--atol", type=float, default=1e-3)
    

    general_params = parser.add_argument_group(title="General")
    general_params.add_argument('--viewer', action='store_true')

    return parser.parse_args()


args = parse_args()


st1 = NURBSSurfaceTuple(
    order_u=2,
    order_v=2,
    knot_u=np.array([0.0, 0.0, 256.50009777, 256.50009777]),
    knot_v=np.array([0.0, 0.0, 259.71657438, 259.71657438]),
    control_points=np.array(
        [
            [[-128.25004889, -129.85828719, 67.43742325], [-128.25004889, 129.85828719, 0.0]],
            [[128.25004889, -46.98266257, 0.0], [128.25004889, 129.85828719, 0.0]],
        ]
    ),
    weights=np.array([[1.0, 1.0], [1.0, 1.0]]),
)


st2 = NURBSSurfaceTuple(order_u=2, order_v=2, knot_u=np.array([  0.        ,   0.        , 256.50009777, 256.50009777]), knot_v=np.array([  0.        ,   0.        , 259.71657438, 259.71657438]), control_points=np.array([[[-128.25004889, -129.85828719,  0.0],
        [-128.25004889,  129.85828719,    0.        ]],

       [[ 128.25004889, -129.85828719,    0.        ],
        [ 128.25004889,  129.85828719,    0.        ]]]), weights=np.array([[1., 1.],
       [1., 1.]]))
from mmcore.geom._nurbs_knots import join_curves
bnds=join_curves(extract_surface_boundaries_tuple(st1))
#s2 = _tuple_to_nurbs(st2)
#s1 = _tuple_to_nurbs(st1)
result=[]
# Perform SSX


for b in bnds:
    print(b)
    start_time = time.time()

    s = time.time()
    t0,t1=_curve_interval(b)
    #d=b.control_points[1]    - b.control_points[0]
    #d/=np.linalg.norm(d)
    #
    #b.control_points[1]+=d
    #b.control_points[0] -= d

 
 
    result.append(nurbs_csx(b, st2, tol=args.atol))
    print("CSX v4 performed at: ", time.time() - s, " secs.")
isolated,overlaps=[],[]
for i,o,_status in  result:
    if i is not None:
        isolated.extend(i)
    if o is not None:
        overlaps.extend(o)
rich.print('\nisolated:')
rich.print(isolated)
rich.print('\noverlaps:')
rich.print(overlaps)


if args.viewer:
    try:
        from mmcore.extras.renderer.renderer3d import Viewer,OrbitCamera

        viewer=Viewer(camera=OrbitCamera(distance=np.linalg.norm(st1.control_points.reshape(-1,3).mean(axis=0))**2,target=  st1.control_points.reshape(-1,3).mean(axis=0)))
        srf = viewer.add_nurbs_surface(st1, color=(0.7, 0.7, 0.7, 1),surface_color=(0.5, 0.5, 0.9, 0.1), v_count=4)
        srf2 = viewer.add_nurbs_surface(st2, color=(0.7, 0.7, 0.7, 1), surface_color=(0.5, 0.5, 0.9, 0.1), v_count=4)


        def render_result(result,curve,surface=None):
            if surface is not None:
                srf = viewer.add_nurbs_surface(surface, color=(0.7, 0.7, 0.7, 1), v_count=4)

            crv=  viewer.add(curve, color=(0.9, 0.9, 0.9, 1.0))
            isolated, overlaps ,_= result
            if isolated is not None:
                uvs=[]
                for pt in isolated:

                    viewer.add(pt['point'], color=(0.0, 1.0, 0.5,1.0),size_px=6)

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
        for (res,curve) in zip(result,bnds):
            render_result(res,curve)


        viewer.run()

    except ModuleNotFoundError as err:
        print("mmcore.renderer is not installed, skip preview.",err)
    except ImportError as err:
        print("mmcore.renderer is not installed, skip preview.",err)
    except Exception as err:
        raise err
