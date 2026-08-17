import numpy as np

from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
from mmcore.numeric._bez_closest_point import nurbs_surface_closest_points

import numpy as np
from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple


val = NURBSSurfaceTuple(
    order_u=3,
    order_v=3,
    knot_u=np.array([0.        , 0.        , 0.        , 1.57079633, 1.57079633,
           3.14159265, 3.14159265, 4.71238898, 4.71238898, 6.28318531,
           6.28318531, 6.28318531]),
    knot_v=np.array([    0.        ,     0.        ,     0.        , 18229.95928377,
           18229.95928377, 18229.95928377]),
    control_points=np.array([[[ 29270.15914894,  74315.94083939,      0.        ],
            [ 27859.34559903,  74747.56166201,   5584.09680622],
            [ 20288.94274113,  55045.69586343,  11168.19361245]],

           [[ 29270.15914894,  74315.94083939,      0.        ],
            [ 22519.55619347,  76381.20368081,   4108.73535657],
            [  9609.36393002,  58312.97990104,   8217.47071313]],

           [[ 29270.15914894,  74315.94083939,      0.        ],
            [ 23930.36974339,  75949.58285819,  -1475.36144966],
            [ 12430.99102985,  57449.7382558 ,  -2950.72289931]],

           [[ 29270.15914894,  74315.94083939,      0.        ],
            [ 25341.1832933 ,  75517.96203557,  -7059.45825588],
            [ 15252.61812968,  56586.49661056, -14118.91651176]],

           [[ 29270.15914894,  74315.94083939,      0.        ],
            [ 30680.97269886,  73884.32001677,  -5584.09680622],
            [ 25932.19694078,  53319.21257296, -11168.19361245]],

           [[ 29270.15914894,  74315.94083939,      0.        ],
            [ 36020.76210441,  72250.67799797,  -4108.73535657],
            [ 36611.77575189,  50051.92853536,  -8217.47071313]],

           [[ 29270.15914894,  74315.94083939,      0.        ],
            [ 34609.9485545 ,  72682.29882059,   1475.36144966],
            [ 33790.14865207,  50915.1701806 ,   2950.72289931]],

           [[ 29270.15914894,  74315.94083939,      0.        ],
            [ 33199.13500458,  73113.91964321,   7059.45825588],
            [ 30968.52155224,  51778.41182583,  14118.91651176]],

           [[ 29270.15914894,  74315.94083939,      0.        ],
            [ 27859.34559903,  74747.56166201,   5584.09680622],
            [ 20288.94274113,  55045.69586343,  11168.19361245]]]),
    weights=np.array([[1.        , 1.        , 1.        ],
           [0.70710678, 0.70710678, 0.70710678],
           [1.        , 1.        , 1.        ],
           [0.70710678, 0.70710678, 0.70710678],
           [1.        , 1.        , 1.        ],
           [0.70710678, 0.70710678, 0.70710678],
           [1.        , 1.        , 1.        ],
           [0.70710678, 0.70710678, 0.70710678],
           [1.        , 1.        , 1.        ]])
)


query = np.array([25317.249206, 61395.297031, 0.0])  # on the rotation axis

if __name__ == "__main__":
    import time

    R, H = 3.0, 4.0
    cone = val


    t0 = time.perf_counter()
    res = nurbs_surface_closest_points(cone, query, atol=1e-3)
    dt = time.perf_counter() - t0


    for e in res:
        if e["kind"] == "degenerate_curve":
            d = np.linalg.norm(e["points"] - query[None, :], axis=1)
            print(f"  kind={e['kind']}  closed={e['closed']}  "
                  f"n_points={len(e['uv'])}  distance={e['distance']}  "
                  f"(spread {d.max() - d.min():.2e})")
            print(f"  ring in UV: u in [{e['uv'][:, 0].min()}, {e['uv'][:, 0].max()}], "
                  f"v ~ {e['uv'][:, 1].mean():.4f}")
            if "circle" in e:
                # Every equidistant curve lies on the sphere of radius d_min
                # about the query; this one is planar, hence an EXACT circle.
                c = e["circle"]
                print(f"  certified circle: center={c['center']}  "
                      f"radius={c['radius']}  normal={c['normal']}  "
                      f"arc={np.degrees(c['arc_angle'])} deg")
        else:
            print(f"  kind={e['kind']}  distance={e['distance']} "
                  f"(u={e['u']}, v={e['v']})")
    print("-> the answer is a CLOSED equidistant ring, returned as one curve entity.")
