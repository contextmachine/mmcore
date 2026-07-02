import numpy as np

from mmcore.geom._nurbs_eval import NURBSSurfaceTuple
from mmcore.numeric._bez_closest_point import nurbs_surface_closest_points


val = NURBSSurfaceTuple(
    order_u=3,
    order_v=2,
    knot_u=np.array([    0.        ,     0.        ,     0.        ,  9393.01092241,
            9393.01092241, 18786.02184483, 18786.02184483, 28179.03276724,
           28179.03276724, 37572.04368966, 37572.04368966, 37572.04368966]),
    knot_v=np.array([    0.        ,     0.        , 12268.57175443, 12268.57175443]),
    control_points=np.array([[[20949.04757432, 62731.69387923, -4094.18138263],
            [26488.26531008, 55368.39245528,  5974.30602132]],

           [[20949.04757432, 62731.69387923, -4094.18138263],
            [29209.23350522, 51895.67836871,  1937.68420983]],

           [[20949.04757432, 62731.69387923, -4094.18138263],
            [32406.41903248, 56586.18609625,    57.55957111]],

           [[20949.04757432, 62731.69387923, -4094.18138263],
            [35603.60455973, 61276.6938238 , -1822.56506761]],

           [[20949.04757432, 62731.69387923, -4094.18138263],
            [32882.6363646 , 64749.40791036,  2214.05674388]],

           [[20949.04757432, 62731.69387923, -4094.18138263],
            [30161.66816947, 68222.12199693,  6250.67855537]],

           [[20949.04757432, 62731.69387923, -4094.18138263],
            [26964.48264221, 63531.61426939,  8130.80319409]],

           [[20949.04757432, 62731.69387923, -4094.18138263],
            [23767.29711495, 58841.10654184, 10010.92783281]],

           [[20949.04757432, 62731.69387923, -4094.18138263],
            [26488.26531008, 55368.39245528,  5974.30602132]]]),
    weights=np.array([[1.        , 1.        ],
           [0.70710678, 0.70710678],
           [1.        , 1.        ],
           [0.70710678, 0.70710678],
           [1.        , 1.        ],
           [0.70710678, 0.70710678],
           [1.        , 1.        ],
           [0.70710678, 0.70710678],
           [1.        , 1.        ]])
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
