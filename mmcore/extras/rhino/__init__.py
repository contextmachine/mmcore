from __future__ import annotations
IN_RHINO = False

from mmcore.geom._nurbs_eval import _tuple_to_nurbs, _nurbs_to_tuple, NURBSCurveTuple, to_homogeneous_1d,NURBSSurfaceTuple,to_homogeneous_2d, \
    from_homogeneous_2d
try:
    import Rhino.Geometry as rg
    IN_RHINO = True
except ImportError:
    import rhino3dm as rg
import numpy as np
def rhcurve_to_nt(x:rg.Curve)->NURBSCurveTuple:
    pts = []
    w = []
    nx = x.ToNurbsCurve()
    for pt in nx.Points:
        pts.append([pt.X / pt.Weight, pt.Y / pt.Weight, pt.Z / pt.Weight])
        w.append(pt.Weight)

    knots = list(nx.Knots)
    knots = [knots[0]] + knots + [knots[-1]]
    order = nx.Order
    return NURBSCurveTuple(order, np.array(knots), np.array(pts), np.array(w))


def nt_to_rhcurve(nt: NURBSCurveTuple)->rg.NurbsCurve:
    # rg.NurbsCurve(dimension: int, rational: bool, order: int, pointCount: int)
    crv = rg.NurbsCurve(3, True, nt.order, nt.control_points.shape[0])
    cpts = to_homogeneous_1d(nt.control_points, nt.weights)
    knots = nt.knot[1:][:-1]
    for i in range(len(crv.Knots)):
        crv.Knots[i] = knots[i]
    for i in range(nt.control_points.shape[0]):
        crv.Points.SetPoint(i, rg.Point3d(*cpts[i][:3]))

        crv.Points.SetWeight(i, cpts[i][-1])
    return crv


def is_opennurbs_style_knots(degree, knots):
    _, cnt = np.unique(knots, return_counts=True)

    return cnt[0] == degree


def create_nt_surf(
    deg_u: int,
    deg_v: int,
    knots_u: list[float],
    knots_v: list[float],
    control_points: list[list[rg.Point3d]],
    weights: list[list[float]] = None,
):
    # rg.NurbsCurve(dimension: int, rational: bool, order: int, pointCount: int)
    u_count, v_count = len(control_points), len(control_points[0])
    cpts = np.zeros((u_count, v_count, 3))

    surf = NURBSSurfaceTuple(
        deg_u + 1,
        deg_v + 1,
        np.asarray([knots_u[0]] + knots_u + [knots_u[-1]] if is_opennurbs_style_knots(deg_u, knots_u) else knots_u,dtype=float),
        np.asarray(
        [knots_v[0]] + knots_v + [knots_v[-1]] if is_opennurbs_style_knots(deg_v, knots_v) else knots_v,dtype=float),
        cpts,
        np.asarray(weights),
    )

    for i in range(u_count):
        for j in range(v_count):

            surf.control_points[i, j, :] = tuple(control_points[i][j])

    return surf


def nt_to_rhsurf(nt:NURBSSurfaceTuple)->rg.NurbsSurface:
    b = nt
    knots_u = b.knot_u
    knots_v = b.knot_v
    knots_u = knots_u[1:][:-1]
    knots_v = knots_v[1:][:-1]
    cpts = to_homogeneous_2d(b.control_points, b.weights)
    v_count = len(cpts[0])
    u_count = len(cpts)
    order_u = b.order_u
    order_v = b.order_v
    rational = True

    crv = rg.NurbsSurface.Create(3, rational, order_u, order_v, u_count, v_count)
    if IN_RHINO:
        for i in range(crv.KnotsU.Count):

            crv.KnotsU[i] = knots_u[i]

        for i in range(crv.KnotsV.Count):

            crv.KnotsV[i] = knots_v[i]
    else:
        for i in range(len(crv.KnotsU)):

            crv.KnotsU[i] = knots_u[i]

        for i in range(len(crv.KnotsV)):

            crv.KnotsV[i] = knots_v[i]
    for i in range(crv.Points.CountU):
        for j in range(crv.Points.CountV):
            if IN_RHINO:
                crv.Points.SetPoint(i, j, rg.Point3d(*cpts[i][j][:3]))
            else:
                print('crv')
                crv.Points[i][j] = rg.Point3d(*cpts[i][j][:3])

            if rational:
                crv.Points.SetWeight(i, j, cpts[i][j][-1])

    return crv

def rhbrep_to_brep(rh_brep) -> 'BRep':
    """Convert a Rhino Brep (RhinoCommon or rhino3dm) to our BRep.

    Directly populates the BRep entity dictionaries from Rhino's flat-array
    structure, bypassing Euler operators. This is a data conversion, not a
    topological construction.

    Parameters
    ----------
    rh_brep : Rhino.Geometry.Brep or rhino3dm.Brep
        The source Rhino BRep.

    Returns
    -------
    BRep
        The converted mmcore BRep with geometry (G_CRV, G_PCRV, G_SRF)
        populated from the Rhino curves and surfaces.
    """
    from mmcore.topo.brep import BRep, Vertex, Edge, HalfEdge, Loop, Face, Shell

    brep = BRep()

    # RhinoCommon collections use .Count, not len()
    def _count(collection):
        return collection.Count if hasattr(collection, 'Count') else len(collection)

    # --- Step 1: Convert geometry arrays ---
    # Map Rhino surface index → our G_SRF id
    srf_map = {}  # rh_surface_index → brep G_SRF id
    for i in range(_count(rh_brep.Surfaces)):
        srf = rhsurf_to_nt(rh_brep.Surfaces[i])
        srf_map[i] = brep.new_surface(srf)

    # Map Rhino 3D curve index → our G_CRV id
    crv3d_map = {}  # rh_curve3d_index → brep G_CRV id
    for i in range(_count(rh_brep.Curves3D)):
        crv = rhcurve_to_nt(rh_brep.Curves3D[i])
        crv3d_map[i] = brep.new_curve(crv)

    # Map Rhino 2D curve index → our G_PCRV id
    crv2d_map = {}  # rh_curve2d_index → brep G_PCRV id
    for i in range(_count(rh_brep.Curves2D)):
        crv = rhcurve_to_nt(rh_brep.Curves2D[i])
        crv2d_map[i] = brep.new_pcurve(crv)

    # --- Step 2: Convert vertices ---
    vtx_map = {}  # rh_vertex_index → brep vertex id
    for i in range(_count(rh_brep.Vertices)):
        rh_v = rh_brep.Vertices[i]
        pt = rh_v.Location
        v = brep.new_vertex((pt.X, pt.Y, pt.Z), tol=rh_v.VertexTolerance if hasattr(rh_v, 'VertexTolerance') else 1e-6)
        vtx_map[i] = v.id

    # --- Step 3: Convert edges ---
    edge_map = {}  # rh_edge_index → brep edge id
    for i in range(_count(rh_brep.Edges)):
        rh_e = rh_brep.Edges[i]
        v_start = vtx_map[rh_e.StartVertex.VertexIndex]
        v_end = vtx_map[rh_e.EndVertex.VertexIndex]

        # Edge curve geometry
        crv_idx = rh_e.EdgeCurveIndex
        geom_id = crv3d_map.get(crv_idx) if crv_idx >= 0 else None

        # Edge parameter range from domain
        dom = rh_e.Domain
        param = (dom.T0, dom.T1) if hasattr(dom, 'T0') else (dom[0], dom[1])

        e = brep.new_edge(v_start, v_end, geom=geom_id, param=param)
        edge_map[i] = e.id

    # --- Step 4: Convert trims → half-edges ---
    # We need to do this in two passes:
    # Pass 1: create all HEs (without next/prev/twin wiring)
    # Pass 2: wire next/prev within loops, and twins via edge sharing
    trim_map = {}  # rh_trim_index → brep he id
    edge_to_hes = {}  # brep edge id → list of brep he ids (for twin wiring)

    for i in range(_count(rh_brep.Trims)):
        rh_t = rh_brep.Trims[i]

        # Determine edge (singular trims have no edge)
        rh_edge = rh_t.Edge
        if rh_edge is not None:
            edge_id = edge_map[rh_edge.EdgeIndex]
        else:
            # Singular trim — create a degenerate edge if needed
            # For now, create an edge with same start/end vertex
            sv = vtx_map[rh_t.StartVertex.VertexIndex]
            edge_id = brep.new_edge(sv, sv).id

        # Face and loop (will be set properly in loop/face passes)
        # For now, store None — we'll fill in during loop construction
        face_id = None
        loop_id = None

        # Orientation: in Rhino, IsReversed means trim direction is opposite to edge
        orient = not rh_t.IsReversed

        # PCurve
        pcurve_idx = rh_t.TrimCurveIndex
        pcurve_id = crv2d_map.get(pcurve_idx) if pcurve_idx >= 0 else None

        # Vertex at head of half-edge (destination vertex)
        # In our convention: he.vert = head vertex
        # If orient=True (same as edge): head = edge.v_end
        # If orient=False (reversed): head = edge.v_start
        edge_obj = brep.E[edge_id]
        vert_id = edge_obj.v_end if orient else edge_obj.v_start

        he = brep.new_halfedge(
            edge=edge_id,
            face=face_id,
            loop=loop_id,
            vert=vert_id,
            orient=orient,
            pcurve=pcurve_id,
        )
        trim_map[i] = he.id

        # Track HEs per edge for twin wiring
        edge_to_hes.setdefault(edge_id, []).append(he.id)

    # --- Step 5: Wire twins ---
    for edge_id, he_ids in edge_to_hes.items():
        if len(he_ids) == 2:
            he_a, he_b = he_ids
            brep.HE[he_a].twin = he_b
            brep.HE[he_b].twin = he_a
            brep.E[edge_id].he = he_a
        elif len(he_ids) == 1:
            # Boundary edge or singular — twin points to self (or we create a virtual twin)
            he_a = he_ids[0]
            brep.HE[he_a].twin = he_a  # self-twin for boundary
            brep.E[edge_id].he = he_a

    # --- Step 6: Convert loops and wire next/prev ---
    loop_map = {}  # rh_loop_index → brep loop id
    for i in range(_count(rh_brep.Loops)):
        rh_loop = rh_brep.Loops[i]

        # Collect this loop's trim indices
        trim_indices = []
        for j in range(_count(rh_loop.Trims)):
            rh_t = rh_loop.Trims[j]
            trim_indices.append(rh_t.TrimIndex)

        if not trim_indices:
            continue

        # Determine loop type
        loop_type = rh_loop.LoopType
        is_outer = (str(loop_type) == 'BrepLoopType.Outer' or
                    str(loop_type) == 'Outer' or
                    (hasattr(loop_type, 'value') and loop_type.value == 0) or
                    loop_type == 0)

        # Create the loop
        first_he_id = trim_map[trim_indices[0]]
        lp = brep.new_loop(face=None, he=first_he_id, is_outer=is_outer)
        loop_map[i] = lp.id

        # Wire next/prev chain and assign loop/face
        he_ids_in_loop = [trim_map[ti] for ti in trim_indices]
        n = len(he_ids_in_loop)
        for k in range(n):
            he = brep.HE[he_ids_in_loop[k]]
            he.next = he_ids_in_loop[(k + 1) % n]
            he.prev = he_ids_in_loop[(k - 1) % n]
            he.loop = lp.id

    # --- Step 7: Convert faces ---
    face_map = {}  # rh_face_index → brep face id
    for i in range(_count(rh_brep.Faces)):
        rh_face = rh_brep.Faces[i]

        # Surface geometry
        surf_idx = rh_face.SurfaceIndex if hasattr(rh_face, 'SurfaceIndex') else i
        surf_id = srf_map.get(surf_idx)

        # Same sense
        same_sense = not rh_face.OrientationIsReversed

        # Find outer and inner loops for this face
        outer_loop_id = None
        inner_loop_ids = []

        for j in range(len(rh_brep.Loops)):
            rh_loop = rh_brep.Loops[j]
            if rh_loop.Face.FaceIndex == i:
                our_loop_id = loop_map.get(j)
                if our_loop_id is None:
                    continue
                lp = brep.L[our_loop_id]
                if lp.is_outer:
                    outer_loop_id = our_loop_id
                else:
                    inner_loop_ids.append(our_loop_id)

        f = brep.new_face(
            outer=outer_loop_id,
            inners=inner_loop_ids,
            same_sense=same_sense,
            surf=surf_id,
        )
        face_map[i] = f.id

        # Assign face to all loops and HEs
        for lid in ([outer_loop_id] if outer_loop_id else []) + inner_loop_ids:
            brep.L[lid].face = f.id
            for he_id in brep._loop_halfedges(lid):
                brep.HE[he_id].face = f.id

    # --- Step 8: Create shell ---
    face_ids = list(face_map.values())
    if face_ids:
        shell = brep.new_shell(faces=face_ids)

    return brep


def rhsurf_to_nt(x:rg.Surface)->NURBSSurfaceTuple:
    nx = x.ToNurbsSurface()
    pts = [(pt.X, pt.Y, pt.Z, pt.Weight) for pt in nx.Points]
    ku = list(nx.KnotsU)
    kv = list(nx.KnotsV)
    ku = [ku[0]] + ku + [ku[-1]]
    kv = [kv[0]] + kv + [kv[-1]]
    ou = nx.OrderU
    ov = nx.OrderV
    cpt, ws = from_homogeneous_2d(np.array(pts).reshape((nx.Points.CountU, nx.Points.CountV, 4)))
    return NURBSSurfaceTuple(ou, ov, knot_u=np.array(ku), knot_v=np.array(kv), control_points=cpt, weights=ws)
