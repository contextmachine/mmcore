import numpy as np

from mmcore.geom._nurbs_eval import _nurbs_to_tuple
from mmcore.numeric.intersection.csx import nurbs_csx_v2

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
from mmcore.geom.nurbs import NURBSCurve, NURBSSurface

u, v, t = 0.9939461136471586, 0.995759608283125, 0.004240391716877873
surf = NURBSSurface(np.array(spts), (3, 3))

curve = NURBSCurve(cpts)
# ress = new_intersection_candidates(surf, curve, u, v, t, np.array(surf.evaluate_v2(u, v)))

import time

s = time.time()
result = nurbs_csx_v2(_nurbs_to_tuple(curve), _nurbs_to_tuple(surf))

print(f"CSX performed at: {time.time()-s} secs.")


try:
    from mmcore.extras.renderer import CADRenderer, Camera

    print(dir(Camera))

    centr = np.average(surf.control_points_flat, axis=0)
    renderer = CADRenderer(camera=Camera(zoom=50.0, near=0.1))
    renderer.add_nurbs_curve(
        curve,
        color=(0.0, 1.0, 0.5)
    )
    tess=renderer.add_nurbs_surface(surf, color=(0.9, 0.9, 0.9))

    for item in result:
        print(item.curve_eval["C"])
        renderer.add_point(item.curve_eval["C"], np.array((1.0, 0.5, 0.0)), 4)

    renderer.run()

except ModuleNotFoundError as err:
    print("mmcore.renderer is not installed, skip preview.")
except ImportError as err:
    print("mmcore.renderer is not installed, skip preview.")
except Exception as err:
    raise err
