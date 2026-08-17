"""Build a closed cylinder BRep with caps and display in the viewer.

Topology: 3 faces, 2 edges (circles), 2 vertices (seam points).
Each circle edge is shared between the cylindrical face and a cap face.
"""
import pickle

import numpy as np
from mmcore.construction._cylinder import cylinder_surface
from mmcore.nurbs._nurbs_construct import circle
from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple, NURBSCurveTuple, evaluate_nurbs_curve
from mmcore.nurbs.nurbs_iso import extract_isocurve
from mmcore.nurbs._nurbs_knots import reverse_curve
from mmcore.numeric.closest_point import nurbs_curve_closest_point
from mmcore.topo.brep import BRep


def _make_planar_disk_surface(center, normal, radius, xaxis=None):
    """Create a bilinear NURBS surface for a planar disk.

    The surface maps the unit square to a square region in the plane
    centered at `center`. The square extends from -radius to +radius
    in both in-plane directions. The disk trim comes from the BRep loop.
    """
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    if xaxis is not None:
        xaxis = np.asarray(xaxis, dtype=float)
        xaxis = xaxis / np.linalg.norm(xaxis)
    else:
        # Choose xaxis perpendicular to normal
        ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        xaxis = np.cross(normal, ref)
        xaxis = xaxis / np.linalg.norm(xaxis)
    yaxis = np.cross(normal, xaxis)

    # 4 corners of the square patch
    r = radius * 1.5  # slightly larger than the circle to ensure trim is interior
    c00 = center - r * xaxis - r * yaxis
    c10 = center + r * xaxis - r * yaxis
    c01 = center - r * xaxis + r * yaxis
    c11 = center + r * xaxis + r * yaxis

    control_points = np.array([[c00, c01], [c10, c11]])
    weights = np.ones((2, 2))

    return NURBSSurfaceTuple(
        order_u=2, order_v=2,
        knot_u=np.array([0.0, 0.0, 1.0, 1.0]),
        knot_v=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=control_points,
        weights=weights,
    )


def _project_curve_to_plane(curve_3d, center, xaxis, yaxis):
    """Project a 3D NURBS curve onto a plane, returning a 2D pcurve.

    For planar geometry, this is an exact affine transformation of the
    control points — no iterative projection needed. Each 3D control
    point P is mapped to 2D as (dot(P-center, xaxis), dot(P-center, yaxis)).
    Weights are preserved.
    """
    center = np.asarray(center, dtype=float)
    xaxis = np.asarray(xaxis, dtype=float)
    yaxis = np.asarray(yaxis, dtype=float)

    pts_3d = curve_3d.control_points  # (N, 3)
    rel = pts_3d - center
    pts_2d = np.column_stack([rel @ xaxis, rel @ yaxis])

    return NURBSCurveTuple(
        order=curve_3d.order,
        knot=curve_3d.knot.copy(),
        control_points=pts_2d,
        weights=curve_3d.weights.copy(),
    )


def _add_cap_face(brep, cap_srf, circle_crv, center, xaxis, yaxis):
    """Add a planar cap face bounded by a single circle.

    Uses make_face_from_surface for topology, then replaces the
    marched pcurves with analytically projected ones (exact for planes).
    """
    face, shell, loop, verts, edges = brep.make_face_from_surface(
        cap_srf, boundary_curves=[circle_crv]
    )

    # Replace marched pcurves with analytic projections
    # The cap face has one edge with two half-edges (the digon from N=1)
    edge = edges[0]
    he_fwd = brep.HE[edge.he]
    he_rev = brep.HE[he_fwd.twin]

    # Compute the surface's local coordinate mapping
    # Our bilinear patch maps: S(u,v) = c00 + u*(c10-c00) + v*(c01-c00)
    # So UV = inv_affine(P - c00)
    # But it's easier to just project the 3D circle control points to UV
    # using the plane's local axes, then rescale to the surface's UV domain
    r = cap_srf.control_points[1, 0] - cap_srf.control_points[0, 0]  # u-direction in 3D
    u_extent = np.linalg.norm(r)
    r_v = cap_srf.control_points[0, 1] - cap_srf.control_points[0, 0]
    v_extent = np.linalg.norm(r_v)

    origin_3d = cap_srf.control_points[0, 0]  # corner of the bilinear patch
    u_axis = r / u_extent
    v_axis = r_v / v_extent

    for he in [he_fwd, he_rev]:
        crv_3d = brep.G_CRV[edge.geom]
        # Determine the curve direction for this half-edge
        if he.orient:
            crv_to_project = crv_3d
        else:
            crv_to_project = reverse_curve(crv_3d)

        # Project control points to UV
        pts_3d = crv_to_project.control_points
        rel = pts_3d - origin_3d
        u_coords = (rel @ u_axis) / u_extent
        v_coords = (rel @ v_axis) / v_extent
        pts_2d = np.column_stack([u_coords, v_coords])

        pcurve = NURBSCurveTuple(
            order=crv_to_project.order,
            knot=crv_to_project.knot.copy(),
            control_points=pts_2d,
            weights=crv_to_project.weights.copy(),
        )

        # Remove old pcurve if any, store new one
        if he.pcurve is not None and he.pcurve in brep.G_PCRV:
            del brep.G_PCRV[he.pcurve]
        he.pcurve = brep.new_pcurve(pcurve)

    return face, shell, loop, verts, edges


def make_cylinder_brep(radius=5.0, height=10.0, analytic_caps=True):
    """Build a closed cylinder BRep: cylindrical face + top cap + bottom cap.

    Parameters
    ----------
    analytic_caps : bool
        If True, use analytic affine projection for cap pcurves (fast).
        If False, use the default compute_pcurve marching path (tests the
        general pipeline).

    Returns (brep, faces_dict) where faces_dict has keys 'cyl', 'top', 'bottom'.
    """
    brep = BRep()

    # --- surfaces ---
    cyl_srf = cylinder_surface(radius=radius, height=height)
    bot_srf = _make_planar_disk_surface(
        center=[0, 0, 0], normal=[0, 0, -1], radius=radius
    )
    top_srf = _make_planar_disk_surface(
        center=[0, 0, height], normal=[0, 0, 1], radius=radius
    )

    # --- Build the cylindrical face ---
    cyl_face, cyl_shell, cyl_loop, cyl_verts, cyl_edges = brep.make_face_from_surface(cyl_srf)
    print(f"After cylinder face: {brep.summary()}")
    assert brep.validate() == [], brep.validate()

    # --- edge curves: extract circles from the cylinder boundaries ---
    bottom_circle = extract_isocurve(cyl_srf, 0.0, direction='v')
    top_circle = extract_isocurve(cyl_srf, 1.0, direction='v')

    # Cap circles are reversed (outward-facing normal requires opposite winding)
    bot_circle_rev = reverse_curve(bottom_circle)
    top_circle_rev = reverse_curve(top_circle)

    if analytic_caps:
        # --- analytic path: affine projection for cap pcurves ---
        bot_face, bot_shell, bot_loop, bot_verts, bot_edges = _add_cap_face(
            brep, bot_srf, bot_circle_rev,
            center=np.array([0, 0, 0]),
            xaxis=np.array([1, 0, 0]),
            yaxis=np.array([0, 1, 0]),
        )
        print(f"After bottom cap (analytic): {brep.summary()}")

        top_face, top_shell, top_loop, top_verts, top_edges = _add_cap_face(
            brep, top_srf, top_circle_rev,
            center=np.array([0, 0, height]),
            xaxis=np.array([1, 0, 0]),
            yaxis=np.array([0, 1, 0]),
        )
        print(f"After top cap (analytic): {brep.summary()}")
    else:
        # --- default path: marching predictor-corrector for cap pcurves ---
        bot_face, bot_shell, bot_loop, bot_verts, bot_edges = brep.make_face_from_surface(
            bot_srf, boundary_curves=[bot_circle_rev]
        )
        print(f"After bottom cap (marched): {brep.summary()}")

        top_face, top_shell, top_loop, top_verts, top_edges = brep.make_face_from_surface(
            top_srf, boundary_curves=[top_circle_rev]
        )
        print(f"After top cap (marched): {brep.summary()}")

    errors = brep.validate()
    assert errors == [], errors

    return brep, {
        'cyl': cyl_face,
        'top': top_face,
        'bottom': bot_face,
    }

def make_cylinder_brep_v2(radius=5.0, height=10.0):
    """Build a closed cylinder BRep: cylindrical face + top cap + bottom cap.

    Parameters
    ----------
    analytic_caps : bool
        If True, use analytic affine projection for cap pcurves (fast).
        If False, use the default compute_pcurve marching path (tests the
        general pipeline).

    Returns (brep, faces_dict) where faces_dict has keys 'cyl', 'top', 'bottom'.
    """
    brep = BRep()

    # --- surfaces ---
    cyl_srf = cylinder_surface(radius=radius, height=height)

    # --- Build the cylindrical face ---
    cyl_face, cyl_shell, cyl_loop, cyl_verts, cyl_edges = brep.make_face_from_surface(cyl_srf,auto_close=True)


    print(f"After cylinder face: {brep.summary()}")
    assert brep.validate() == [], brep.validate()
    #brep.weld_edges(cyl_edges[1].id, cyl_edges[3].id)
    #print(f"After welding edges: {brep.summary()}")
    #assert brep.validate() == [], brep.validate()
    brep.cap_planar_openings(1e-3)
    print(f"After cap: {brep.summary()}")
    assert brep.validate() == [], brep.validate()
    errors = brep.validate()
    assert errors == [], errors

    return brep

if __name__ == "__main__":
    #brep, faces = make_cylinder_brep(radius=5.0, height=10.0,analytic_caps=False)
    #
    #print(f"\nFinal BRep: {brep.summary()}")
    #print(f"G_CRV={len(brep.G_CRV)}, G_PCRV={len(brep.G_PCRV)}, G_SRF={len(brep.G_SRF)}")
    #print(f"Faces with surf: {sum(1 for f in brep.F.values() if f.surf is not None)}")
    #
    #errors = brep.validate()
    #print(f"Validate: {len(errors)} errors")
    #if errors:
    #    for e in errors[:10]:
    #        print(f"  {e}")
    #
    ## Tessellate and check
    #from mmcore.topo.mesh.tess import tessellate_brep_face
    #for name, face in faces.items():
    #    if face.surf is None:
    #        continue
    #    mesh = tessellate_brep_face(brep, face.id, tol=0.5)
    #    print(f"  {name}: {len(mesh['position'])} verts, {len(mesh['faces'])} tris")
    #print(mesh)
    # Launch viewer
    from mmcore.extras.renderer.renderer3d import Viewer, OrbitCamera


    brep2=make_cylinder_brep_v2(radius=5.0, height=10.0)
    from mmcore.compat.step.step_writer import StepWriter

    writer=StepWriter()
    writer.add_brep(brep2)
    from pathlib import Path
    path=Path('cylinder.stp').absolute()
    with open(path, 'w') as f:
        writer.write(f)
    print('step file saved at:', path)

    cam = OrbitCamera(target=(0, 0, 5), distance=30, ortho_half_height=-15)
    viewer = Viewer(width=1200, height=800, camera=cam)
    viewer.add_brep(
        brep2,
        edge_color=(1.0, 1.0, 1.0, 1.0),
        surface_color=(0.4/4, 0.6/4, 0.9/4, 0.5),
        tol=0.5,
    )
    viewer.run()
