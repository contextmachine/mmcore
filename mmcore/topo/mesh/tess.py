from __future__ import annotations
from typing import Collection,TypedDict

import numpy as np
from ._classes import Tessellation,tess_to_mesh
from mmcore.geom.nurbs import NURBSSurface, decompose_surface
from mmcore.geom.polygon import is_point_in_polygon_bvh, polygon_build_bvh
from mmcore.numeric.algorithms.adaptive_polyline import adaptive_polyline
from mmcore.numeric.routines import uvs
from mmcore.topo.mesh.triangle import triangulate
from mmcore.topo.mesh.triangle.tri import segments_by_loop
from mmcore.geom.nurbs_iso import extract_surface_boundaries

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


def _process_trim(trim: 'CurveOnSurface',tol=1e-2):

    pts,prms=adaptive_polyline(trim, tol=tol)
    uv_pts=np.array(trim.curve.evaluate_multi(prms))[...,:2]




    edges = np.array([(i, (i + 1) % len(uv_pts)) for i in range(len(uv_pts))], dtype=np.int32)
    #bvh_root = polygon_build_bvh(polygon[edges])
    return uv_pts, edges,  #bvh_root


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


def tess_boundaries(surface, tol=1e-3):
    (u_min, u_max), (v_min, v_max) = surface.interval()

    crvs = extract_surface_boundaries(surface)


    plns = [adaptive_polyline(crv, tol)[1] for crv in crvs]
    boundary = []
    boundary_pts_count=[[None,None],[None,None]]
    for i, p in enumerate(plns):
        _uv = np.zeros(p.shape + (2,))

        if i < 2:
            boundary_pts_count[1][i]=len(p)
            _uv[..., 0] = [u_min, u_max][i]
            _uv[..., 1] = p
        else:
            boundary_pts_count[0][i - 2] = len(p)
            _uv[..., 1] = [v_min, v_max][i - 2]
            _uv[..., 0] = p
        boundary.extend(_uv)

    boundary = np.array(boundary)
    boundary_edges = np.array([(i, (i + 1) % len(boundary)) for i in range(len(boundary))], dtype=np.int32)
    return boundary, boundary_edges,boundary_pts_count

def tessellate_surface(surface: NURBSSurface,
                       trims: Collection['CurveOnSurface'] = (),

                        tol=1e-3):
    """
    :param surface: The surface to be tessellated.
    :param trims: Collection of curves on the surface to be included in the tessellation.
    :param u_count: Optional. Number of divisions in the u direction of the surface. If not provided, default value is 25.
    :param v_count: Optional. Number of divisions in the v direction of the surface. If not provided, default value is 25.
    :param boundary_count: Optional. Number of divisions in the boundary of the surface. Defaults to 100.
    :return: The tessellation of the surface as a dictionary with vertices, segments, position, and other properties.

    """



    (u_min ,u_max), (v_min ,v_max)=surface.interval()
    boundary,boundary_edges,boundary_pts_count=tess_boundaries(surface,tol)
    u_count=max(boundary_pts_count[0])
    v_count = max(boundary_pts_count[1])
    print(u_count,v_count)
    u_step=(u_max-u_min)/u_count
    v_step = (v_max - v_min) / v_count

    tess_uv = (u_min + u_step, u_max - u_step), (v_min + v_step,v_max - v_step)

    #boundary_bvh_root = polygon_build_bvh(boundary[boundary_edges])
    tessellation_params = dict(vertices=[*boundary], segments=[*boundary_edges])
    _max = len(boundary)

    for i, trim in enumerate(trims):

        polyline, edges = _process_trim(trim, tol=tol)

        tessellation_params['segments'].extend(edges + _max)
        tessellation_params['vertices'].extend(polyline)
        _max += len(edges)
    uv = uvs(u_count - 1, v_count - 1, *tess_uv)


    tessellation_params['vertices'].extend(uv)
    vxs = np.array(tessellation_params['vertices'], dtype=float)

    tessellation_params['vertices'] = vxs
    tessellation_params['segments'] = np.array(tessellation_params['segments'], dtype=np.int32)

    tessellation = triangulate(
            tessellation_params
        )
    vxs = np.array(tessellation["vertices"])

    tessellation["vertices"] = vxs
    tessellation["position"] = surface.evaluate_multi(vxs)
    return tessellation
from .fuse import fuse_meshes
def surface_to_mesh(surface: NURBSSurface,tol=1e-3):
    return fuse_meshes([tess_to_mesh(tessellate_surface(s,tol=tol) )for s in decompose_surface(surface)])[0]

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

