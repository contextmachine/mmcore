from __future__ import annotations
from typing import Collection,TypedDict

import numpy as np
from ._classes import Tessellation, Mesh, tess_to_mesh
from mmcore.geom.nurbs import NURBSSurface, decompose_surface
from mmcore.geom.polygon import is_point_in_polygon_bvh, polygon_build_bvh
from mmcore.numeric.algorithms.adaptive_polyline import adaptive_polyline
from mmcore.numeric.routines import uvs
from mmcore.topo.mesh.triangle import triangulate
from mmcore.topo.mesh.triangle.tri import segments_by_loop
from mmcore.geom.nurbs_iso import extract_surface_boundaries
from ...geom._nurbs_eval import to_homogeneous_2d
from ...numeric.approx import adaptive_bern_sampler_2d


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
    #print(u_count,v_count)
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
    uv = np.array(adaptive_bern_sampler_2d(to_homogeneous_2d(surface.control_points,surface.weights),interval=surface.interval(),rational=True)).reshape((-1, 2))


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
def surface_to_mesh(surface: NURBSSurface,tol=1e-2):
    return fuse_meshes([tess_to_mesh(tessellate_surface(s,tol=tol) )for s in decompose_surface(surface)])[0]

def tessellate_brep_face(brep, face_id: int, tol: float = 1e-2) -> Mesh:
    """Tessellate a single BRep face into a triangle mesh.

    Walks the face's loops, samples pcurves to get UV polylines,
    generates adaptive interior UV points, triangulates via constrained
    Delaunay, and evaluates the surface at all vertices.

    Parameters
    ----------
    brep : BRep
        The BRep containing the face.
    face_id : int
        ID of the face to tessellate.
    tol : float
        Tessellation tolerance (controls adaptive sampling density).

    Returns
    -------
    Mesh
        dict with 'position' (N,3) float32 and 'faces' (T,3) int32.
    """
    from mmcore.geom._nurbs_eval import (
        evaluate_nurbs_surface,
        evaluate_nurbs_curve_array,
        to_homogeneous_2d,
        NURBSSurfaceTuple,
    )

    face = brep.F[face_id]
    if face.surf is None:
        raise ValueError(f"Face {face_id} has no surface geometry")
    srf = brep.G_SRF[face.surf]

    # --- collect UV boundary polyline from the outer loop's pcurves ---
    def _sample_loop_pcurves(loop_id, n_samples=50):
        """Sample all pcurves in a loop, returning a single UV polyline."""
        uv_pts = []
        sampled_edges = set()
        for he_id in brep._loop_halfedges(loop_id):
            he = brep.HE[he_id]
            if he.pcurve is None:
                continue
            # Digon vs seam: both have two HEs sharing the same edge,
            # but in a digon both HEs are in the SAME loop (self-overlapping),
            # while in a seam they're in the same loop but with different
            # UV paths (e.g. u=0 and u=1 on a cylinder).
            # Key distinction: seam twins are in the same loop (after weld),
            # digon twins are also in the same loop — but a digon loop has
            # exactly 2 HEs, while a seam loop has more.
            twin = brep.HE[he.twin]
            if he.edge in sampled_edges:
                # Count HEs in this loop to distinguish digon (2) from seam (>2)
                loop_size = sum(1 for _ in brep._loop_halfedges(he.loop))
                if loop_size <= 2:
                    continue  # digon: skip duplicate
                # seam edge in a larger loop: sample both UV paths
            sampled_edges.add(he.edge)

            pcurve = brep.G_PCRV[he.pcurve]
            t0, t1 = pcurve.interval()
            ts = np.linspace(t0, t1, n_samples)
            pts_2d = np.array(
                [evaluate_nurbs_curve_array(pcurve, float(t), d_order=0)[0, :2] for t in ts],
                dtype=float,
            )
            # Skip the first point of each segment after the first
            # to avoid duplicate vertices at curve joints
            if len(uv_pts) > 0:
                pts_2d = pts_2d[1:]
            uv_pts.extend(pts_2d)
        return np.array(uv_pts, dtype=float) if uv_pts else np.empty((0, 2))

    if face.outer is None:
        return Mesh(position=np.empty((0, 3), dtype=np.float32),
                    faces=np.empty((0, 3), dtype=np.int32))
    outer_uv = _sample_loop_pcurves(face.outer)
    if len(outer_uv) < 3:
        return Mesh(position=np.empty((0, 3), dtype=np.float32),
                    faces=np.empty((0, 3), dtype=np.int32))

    # Close the loop if not already closed
    if np.linalg.norm(outer_uv[0] - outer_uv[-1]) > tol * 0.1:
        outer_uv = np.vstack([outer_uv, outer_uv[0:1]])

    # Build boundary segments
    n_outer = len(outer_uv)
    outer_segments = np.array(
        [(i, (i + 1) % n_outer) for i in range(n_outer)], dtype=np.int32
    )

    tess_params = dict(
        vertices=list(outer_uv),
        segments=list(outer_segments),
    )
    vertex_offset = n_outer

    # --- inner loops (holes) ---
    hole_points = []
    for inner_loop_id in face.inners:
        inner_uv = _sample_loop_pcurves(inner_loop_id)
        if len(inner_uv) < 3:
            continue
        if np.linalg.norm(inner_uv[0] - inner_uv[-1]) > tol * 0.1:
            inner_uv = np.vstack([inner_uv, inner_uv[0:1]])

        n_inner = len(inner_uv)
        inner_segments = np.array(
            [(vertex_offset + i, vertex_offset + (i + 1) % n_inner)
             for i in range(n_inner)],
            dtype=np.int32,
        )
        tess_params['vertices'].extend(inner_uv)
        tess_params['segments'].extend(inner_segments)
        vertex_offset += n_inner

        # Hole point: centroid of the inner loop (must be inside the hole)
        hole_points.append(inner_uv.mean(axis=0))

    # --- interior UV grid points ---
    # The 'p' (PSLG) triangulation flag ensures Triangle only creates
    # triangles inside the boundary, so no inset margin is needed.
    uv_min = outer_uv.min(axis=0)
    uv_max = outer_uv.max(axis=0)
    uv_span = uv_max - uv_min

    # Exclude the boundary itself (one grid step in from each edge)
    n_boundary = len(outer_uv)
    n_interior_u = max(3, int(n_boundary * 0.3))
    n_interior_v = max(3, int(n_boundary * 0.3))
    inner_min = uv_min + uv_span / (n_interior_u + 1)
    inner_max = uv_max - uv_span / (n_interior_v + 1)

    if np.all(inner_max > inner_min):
        uu = np.linspace(inner_min[0], inner_max[0], n_interior_u)
        vv = np.linspace(inner_min[1], inner_max[1], n_interior_v)
        interior_uv = np.array([(u, v) for u in uu for v in vv], dtype=float)
        tess_params['vertices'].extend(interior_uv)

    # --- triangulate ---
    tess_params['vertices'] = np.array(tess_params['vertices'], dtype=float)
    tess_params['segments'] = np.array(tess_params['segments'], dtype=np.int32)
    if hole_points:
        tess_params['holes'] = np.array(hole_points, dtype=float)

    tess_result = triangulate(tess_params, opts='p')

    # --- evaluate surface at all UV vertices → 3D positions ---
    uv_verts = np.array(tess_result['vertices'], dtype=float)
    try:
        # Fast path: convert to Cython NURBSSurface for batch evaluation
        from mmcore.geom._nurbs_eval import _tuple_to_nurbs
        surf_cy = _tuple_to_nurbs(srf)
        positions = np.ascontiguousarray(
            surf_cy.evaluate_multi(uv_verts), dtype=np.float32
        )
    except Exception:
        # Fallback: point-by-point evaluation
        positions = np.array(
            [evaluate_nurbs_surface(srf, float(uv[0]), float(uv[1]), d_order=0)['S']
             for uv in uv_verts],
            dtype=np.float32,
        )

    return Mesh(
        position=positions,
        faces=np.array(tess_result['triangles'], dtype=np.int32),
    )


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

