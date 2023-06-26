from mmcore.geom.shapes.base import Quad, Tri


def mesh_to_obj(f, mesh):
    s = ""
    """Export to .obj  mesh format"""
    if hasattr(mesh, "vertices"):
        for v in mesh.vertices:
            s += "v {} {} {}\n".format(v.x, v.y, v.z)
    elif hasattr(mesh, "verts"):
        for v in mesh.verts:
            s += "v {} {} {}\n".format(v.x, v.y, v.z)
    for face in mesh.faces:
        if isinstance(face, Quad):
            s += "f {} {} {} {}\n".format(face.v1, face.v2, face.v3, face.v4)
        if isinstance(face, Tri):
            s += "f {} {} {}\n".format(face.v1, face.v2, face.v3)
    return s
