from typing import Collection

import numpy as np

from mmcore.geom.polygon import is_point_in_polygon_bvh, polygon_build_bvh
from mmcore.numeric.routines import uvs
from mmcore.topo.mesh.triangle import triangulate
from mmcore.topo.mesh.triangle.tri import segments_by_loop


def tessellate_curve_on_surface(crv: 'CurveOnSurface', u_count=25, v_count=25, boundary_count=100):
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


def _process_trim(trim: 'CurveOnSurface', boundary_count=100):
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


def calculate_uv_ratio(surf: 'Surface'):
    (u_min, u_max), (v_min, v_max) = surf.interval()

    crv1 = surf.isoline_u((u_min + u_max) / 2)
    crv2 = surf.isoline_v((v_min + v_max) / 2)
    l1 = crv1.evaluate_length(crv1.interval())
    l2 = crv2.evaluate_length(crv2.interval())
    return l1 / l2, l1, l2


def tessellate_surface(surface: 'Surface',
                       trims: Collection['CurveOnSurface'] = (),
                       u_count: int = None, v_count: int = None,
                       boundary_count: int = 100, calculate_density: bool = False):
    """
    :param surface: The surface to be tessellated.
    :param trims: Collection of curves on the surface to be included in the tessellation.
    :param u_count: Optional. Number of divisions in the u direction of the surface. If not provided, default value is 25.
    :param v_count: Optional. Number of divisions in the v direction of the surface. If not provided, default value is 25.
    :param boundary_count: Optional. Number of divisions in the boundary of the surface. Defaults to 100.
    :param calculate_density: Optional. If True, calculates the density of the divisions based on the length of the trims. Defaults to False.
    :return: The tessellation of the surface as a dictionary with vertices, segments, position, and other properties.

    """

    trims_density = [boundary_count] * len(trims)
    if calculate_density:
        ratio, lu, lv = calculate_uv_ratio(surface)

        if u_count is not None and v_count is None:
            u_count = 25
            v_count = int(ratio * u_count)
        elif u_count is None:
            u_count = int((1 / ratio) * v_count)
        elif v_count is None:
            v_count = int(ratio * u_count)
        else:
            pass
        for i, trim in enumerate(trims):
            l = trim.evaluate_length(trim.interval())
            trims_density[i] = int(((lu / l) * u_count) / 4)
    else:
        if u_count is not None and v_count is None:
            u_count = 25
            v_count = 25
        elif u_count is None:
            u_count = v_count

        elif v_count is None:
            v_count = u_count

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

    for i, trim in enumerate(trims):
        polygon, edges = _process_trim(trim, trims_density[i])

        tessellation_params['segments'].extend(edges + _max)
        tessellation_params['vertices'].extend(polygon)
        _max += len(edges)
    uv = uvs(u_count - 1, v_count - 1, *tess_uv)

    tessellation_params['vertices'].extend(uv)
    vxs = np.array(tessellation_params['vertices'], dtype=float)
    if calculate_density:
        vxs[..., 1] *= ratio
    tessellation_params['vertices'] = vxs
    tessellation_params['segments'] = np.array(tessellation_params['segments'], dtype=np.int32)
    if calculate_density:
        tessellation = triangulate(
            tessellation_params, opts='q'
        )
    else:
        tessellation = triangulate(
            tessellation_params
        )
    vxs = np.array(tessellation["vertices"])
    if calculate_density:
        vxs[..., 1] /= ratio
    tessellation["vertices"] = vxs
    tessellation["position"] = surface(vxs)
    return tessellation


def as_polygons(triangulate_result):
    """
    The small debug helper
    :param triangulate_result:
    :return:
    """
    return triangulate_result['position'][triangulate_result['triangles']]
from mmcore.geom.bvh import Object3D,BoundingBox,build_bvh,PTriangle
def as_bvh(triangulate_result):
    uvs=triangulate_result['vertices'][triangulate_result['triangles']]
    pos=triangulate_result['position'][triangulate_result['triangles']]

    return build_bvh([PTriangle(pos[i],uvs[i]) for i in range(len(uvs))])



