import numpy as np
import rich

from mmcore.nurbs._nurbs_eval import _nurbs_to_tuple, evaluate_nurbs_curve

from mmcore.numeric.intersection.csx import nurbs_csx

cpts = np.array(
    [
        [-9.1796875, 13.229166666666666, -4.5186767578125],
        [-9.1796875, 14.739583333333332, -4.49395751953125],
        [-9.1796875, 16.432291666666664, -4.580108642578125],
        [-9.1796875, 18.372395833333332, -4.8531036376953125],
    ]
)
spts = np.array(
    [
        [
            [-5.849180481790346, 18.372395833333336, -1.5018374203712104],
            [-5.858792686592141, 16.432291666666668, -2.719633841323509],
            [-5.871782152540512, 14.739583333333334, -3.1229032219131403],
            [-5.8852911971268185, 13.229166666666668, -3.116512598566417],
        ],
        [
            [-6.88688536134276, 18.372395833333336, -1.6832837287863824],
            [-6.894094514944105, 16.432291666666668, -2.9325012796112815],
            [-6.9038366144053835, 14.739583333333334, -3.3409109281989706],
            [-6.913968397845114, 13.229166666666668, -3.3217200441495054],
        ],
        [
            [-7.97766402100707, 18.372395833333336, -2.2658223298287345],
            [-7.983070886208079, 16.432291666666668, -3.617300740689732],
            [-7.990377460804037, 14.739583333333334, -4.0455099012702895],
            [-7.997976298383835, 13.229166666666668, -3.992029157974829],
        ],
        [
            [-9.1863730157553, 18.372395833333336, -2.7409113689805125],
            [-9.190428164656058, 16.432291666666668, -4.174381706229336],
            [-9.195908095603027, 14.739583333333334, -4.61538352236864],
            [-9.201607223787876, 13.229166666666668, -4.527011611303911],
        ],
    ]
)

import argparse
import time

import rich

from mmcore.nurbs._nurbs_eval import _tuple_to_nurbs, _curve_interval, evaluate_nurbs_curve
from mmcore.nurbs._nurbs_knots import trim_curve
from mmcore.numeric.intersection.csx import nurbs_csx
import logging
from mmcore.nurbs.nurbs_iso import extract_surface_boundaries_tuple
# Creating intersection objects
import numpy as np
from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
def parse_args():
    parser = argparse.ArgumentParser()
    ssx_params = parser.add_argument_group(title="CSX Parameters")
    ssx_params.add_argument("--atol", type=float, default=1e-3)


    general_params = parser.add_argument_group(title="General")
    general_params.add_argument('--viewer', action='store_true')

    return parser.parse_args()


args = parse_args()


from mmcore.nurbs._core import NURBSCurve, NURBSSurface


surface: NURBSSurfaceTuple =_nurbs_to_tuple( NURBSSurface(np.array(spts), (3, 3)))

curve = _nurbs_to_tuple(NURBSCurve(cpts))
# ress = new_intersection_candidates(surf, curve, u, v, t, np.array(surf.evaluate_v2(u, v)))

import time

s = time.time()

result = nurbs_csx(curve, surface,tol=args.atol)

print(f"CSX v4 performed at: {time.time()-s} secs.")
print('isolated:')
if result[0] is not None:
    rich.print(result[0])
print('overlaps:')
if result[1] is not None:
    rich.print(result[1])
isolated,overlaps=result[0],result[1]

if args.viewer:

    try:
        if args.viewer:
            from mmcore.extras.renderer.renderer3d import Viewer, OrbitCamera

            viewer = Viewer(camera=OrbitCamera(target=surface.control_points.reshape(-1, 3).mean(axis=0)))
            srf = viewer.add_nurbs_surface(surface, color=(0.7, 0.7, 0.7, 1.), surface_color=(0.5, 0.5, 0.9, 0.1),
                                           v_count=4)


            def render_result(result, curve, surface=None):
                if surface is not None:
                    srf = viewer.add_nurbs_surface(surface, color=(0.3, 0.3, 0.3, 0.05), v_count=4)

                crv = viewer.add(curve, color=(0.9, 0.9, 0.9, 1.0))
                isolated, overlaps,_ = result
                if isolated is not None:
                    uvs = []
                    for pt in isolated:
                        viewer.add(pt['point'], color=(0.0, 1.0, 0.5, 1.0), size_px=6)

                if overlaps is not None:

                    for overlap in overlaps:
                        t0, t1 = overlap['t_range']

                        viewer.add(evaluate_nurbs_curve(curve, t0, d_order=0)['C'], color=(0.0, 1.0, 0.5, 1.0),
                                   size_px=6)
                        viewer.add(evaluate_nurbs_curve(curve, t1, d_order=0)['C'], color=(0.0, 1.0, 0.5, 1.0),
                                   size_px=6)

                    for o in overlaps:

                        t0 = o["t_range"][0]
                        t1 = o["t_range"][-1]

                        pts = np.linspace(t0, t1, 800)
                        for t in pts:
                            evl = evaluate_nurbs_curve(curve, t, d_order=0)
                            viewer.add_point3d(evl['C'], color=(0.0, 1.0, 0.5, 1.0), size_px=3)


            render_result(result, curve)

            viewer.run()


    except ModuleNotFoundError as err:
        print("mmcore.renderer is not installed, skip preview.")
    except ImportError as err:
        print("mmcore.renderer is not installed, skip preview.")
    except Exception as err:
        raise err
