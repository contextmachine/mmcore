from functools import lru_cache

import numpy as np

from mmcore.geom.polygon import point_in_polygon
from mmcore.geom.surfaces import CurveOnSurface
from mmcore.topo.mesh.triangle import triangulate
from mmcore.topo.mesh.triangle.tri import segments_by_loop

def tessellate_curve_on_surface(crv: CurveOnSurface, u_count=25, v_count=25, boundary_count=100):
    plgn = crv.curve(np.linspace(*tuple(crv.interval()), boundary_count))

    @lru_cache(maxsize=None)
    def pinp(u, v):
        return point_in_polygon(plgn, [u, v, 0.0])

    mask = []
    pts = []
    uu = np.linspace(0.0, 1.0, u_count)
    vv = np.linspace(0.0, 1.0, v_count)

    for i in range(u_count):
        for j in range(v_count):
            u = uu[i]
            v = vv[j]
            r = pinp(u, v)

            mask.append(r)
            pts.append((u, v))
    edges = []
    boundary = segments_by_loop(plgn, start_index=0)
    edges.extend(boundary)
    trires = triangulate(
        dict(
            vertices=np.asarray(
                [*plgn[..., :2], *np.array(pts)[np.array(mask, dtype=bool)]],
                dtype=float,
            ),
            segments=np.array(edges, dtype=np.int32),
        )
    )
    trires["position"] = crv.surf(trires["vertices"])
    return trires
