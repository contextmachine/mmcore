from typing import Iterable
import numpy as np
from mmcore.numeric.routines import uvs
from mmcore.geom.polygon import is_point_in_polygon_bvh, polygon_build_bvh
from mmcore.geom.surfaces import CurveOnSurface, Surface
from mmcore.topo.mesh.triangle import triangulate, convex_hull
from mmcore.topo.mesh.triangle.tri import segments_by_loop


def tessellate_curve_on_surface(crv: CurveOnSurface, u_count=25, v_count=25, boundary_count=100):
    plgn = polygon = crv.curve(np.linspace(*tuple(crv.interval()), boundary_count))[..., :2]

    edges = [(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))]
    bvh_root = polygon_build_bvh(edges)

    mask = []
    pts = []
    uu = np.linspace(0.0, 1.0, u_count)
    vv = np.linspace(0.0, 1.0, v_count)

    for i in range(u_count):
        for j in range(v_count):
            u = uu[i]
            v = vv[j]
            point = (u, v)
            r = is_point_in_polygon_bvh(bvh_root, point)

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


def _process_trim(trim: CurveOnSurface, boundary_count=100):
    polygon = trim.curve(np.linspace(*tuple(trim.interval()), boundary_count))[..., :2]

    edges = np.array([(i, (i + 1) % len(polygon)) for i in range(len(polygon))], dtype=np.int32)
    #bvh_root = polygon_build_bvh(polygon[edges])
    return polygon, edges,  #bvh_root


def _is_close_0(a, tol=1e-3):
    return abs(a) <= tol


def match_edge_cases(polygon, bounds, tol=1e-3):
    if np.allclose(polygon[-1], polygon[0]):
        return polygon

    ((u_min, v_min), (u_max, v_max)) = bounds
    if ((_is_close_0(polygon[0][0] - u_min, tol) and _is_close_0(polygon[-1][0] - u_min, tol))
            or (_is_close_0(polygon[0][0] - u_max, tol) and _is_close_0(polygon[-1][0] - u_max, tol))
            or (_is_close_0(polygon[0][1] - v_min, tol) and _is_close_0(polygon[-1][1] - v_min, tol))
            or (_is_close_0(polygon[0][1] - v_max, tol) and _is_close_0(polygon[-1][1] - v_max, tol))
    ):
        return polygon
    np.array([polygon[0],
              polygon[-1]])


def tessellate_surface(surface: Surface, trims: Iterable[CurveOnSurface] = (), u_count=25, v_count=25,
                       boundary_count=100):
    """
    :param surface: The surface to be tessellated.
    :param trims: The collection of trims to be included in the tessellation.
    :param u_count: The number of divisions in the u direction.
    :param v_count: The number of divisions in the v direction.
    :param boundary_count: The number of divisions in the boundary.

    :return: The tessellated surface.


    """
    uv_interval = ((u_min, v_min), (u_max, v_max)) = surface.interval()

    boundary = np.array([*np.linspace((u_min, v_min), (u_max, v_min), u_count),
                         *np.linspace((u_max, v_min), (u_max, v_max), v_count),
                         *np.linspace((u_max, v_max), (u_min, v_max), u_count),
                         *np.linspace((u_min, v_max), (u_min, v_min), v_count)]
                        )
    u_step = (u_max - u_min) / u_count
    v_step = (v_max - v_min) / v_count
    tess_uv = ((u_min + u_step, v_min + v_step), (u_max - u_step, v_max - v_step))
    boundary_edges = np.array([(i, (i + 1) % len(boundary)) for i in range(len(boundary))], dtype=np.int32)
    #boundary_bvh_root = polygon_build_bvh(boundary[boundary_edges])
    tessellation_params = dict(vertices=[*boundary], segments=[*boundary_edges])
    _max = len(boundary)

    for trim in trims:
        polygon, edges = _process_trim(trim, boundary_count)

        tessellation_params['segments'].extend(edges + _max)
        tessellation_params['vertices'].extend(polygon)
        _max += len(edges)
    uv = uvs(u_count - 1, v_count - 1, *tess_uv)
    tessellation_params['vertices'].extend(uv)
    tessellation_params['vertices'] = np.array(tessellation_params['vertices'], dtype=float)
    tessellation_params['segments'] = np.array(tessellation_params['segments'], dtype=np.int32)
    tessellation = triangulate(
        tessellation_params
    )

    tessellation["position"] = surface(tessellation["vertices"])
    return tessellation
