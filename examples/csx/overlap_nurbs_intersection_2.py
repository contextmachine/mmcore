"""


"""

import time
import numpy as np
from mmcore.geom._nurbs_eval import _tuple_to_nurbs, _nurbs_to_tuple, NURBSSurfaceTuple
from mmcore.numeric.intersection.csx import nurbs_csx_v2
import logging
from mmcore.numeric.intersection.ssx import ssx
from mmcore.geom.nurbs_iso import extract_surface_boundaries_tuple
# Creating intersection objects
import numpy as np
from mmcore.geom._nurbs_eval import NURBSSurfaceTuple


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


st2 = NURBSSurfaceTuple(order_u=2, order_v=2, knot_u=np.array([  0.        ,   0.        , 256.50009777, 256.50009777]), knot_v=np.array([  0.        ,   0.        , 259.71657438, 259.71657438]), control_points=np.array([[[-128.25004889, -129.85828719,  -22.57323617],
        [-128.25004889,  129.85828719,    0.        ]],

       [[ 128.25004889, -129.85828719,    0.        ],
        [ 128.25004889,  129.85828719,    0.        ]]]), weights=np.array([[1., 1.],
       [1., 1.]]))

bnds=extract_surface_boundaries_tuple(st2)
s2 = _tuple_to_nurbs(st2)
s1 = _tuple_to_nurbs(st1)
result=[]
# Perform SSX
logging.basicConfig(level=logging.DEBUG)




for b in extract_surface_boundaries_tuple(st2):
        start_time = time.time()

        s = time.time()
        result += nurbs_csx_v2(b, st1)

try:
    from mmcore.renderer.renderer3dv2 import CADRenderer, Camera

    print(dir(Camera))

    centr = np.average(s1.control_points_flat, axis=0)
    renderer = CADRenderer(camera=Camera(zoom=50.0, near=0.1))
    for b in bnds:
        renderer.add_nurbs_curve(
            _tuple_to_nurbs(b),
            color=(0.0, 1.0, 0.5)
        )
    tess=renderer.add_nurbs_surface(s1, color=(0.9, 0.9, 0.9))

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
