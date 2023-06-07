#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
from __future__ import annotations

import scipy.spatial.distance
import uuid
from collections import namedtuple
from enum import Enum

import numpy as np


from scipy.spatial import distance as spdist

from mmcore.geom.vectors import unit


class ThreeTypes(str, Enum):
    Object3d = "Object"
    Group = "Group"
    Mesh = "Mesh"
    BufferGeometry = "BufferGeometry"





from scipy.spatial import distance
import math


def invdist(a, b):

    d=distance.euclidean(a, b)
    if d != 0:
        return 1 / d



def degtorad(deg):
    return deg * (math.pi / 180)


def radtodeg(rad):
    return rad / (math.pi / 180)


def area_2d(pts):
    """

    @param pts: array like [[ 1., 2.],[ 0., 0.],[ -1.33, 0.], ...],
        or collections of objects with __array__ method (np.asarray compatibility).


    @return:
    """
    lines = np.hstack([np.asarray(pts), np.roll(pts, -1, axis=0)])
    area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in lines))
    return area


def triangle_normal(p1, p2, p3):
    Ax, Ay, Az = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    Bx, By, Bz = p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]

    Nx = Ay * Bz - Az * By
    Ny = Az * Bx - Ax * Bz
    Nz = Ax * By - Ay * Bx
    return Nx, Ny, Nz


def triangle_unit_normal(p1, p2, p3):
    Ax, Ay, Az = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    Bx, By, Bz = p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]

    Nx = Ay * Bz - Az * By
    Ny = Az * Bx - Ax * Bz
    Nz = Ax * By - Ay * Bx
    return unit(np.array([Nx, Ny, Nz]))


def get_mesh_uv(msh): return [(i.X, i.Y) for i in list(msh.TextureCoordinates)]


def get_np_mesh_uv(msh) -> np.ndarray: return np.asarray(get_mesh_uv(msh))


def get_mesh_vertices(msh) -> list[tuple]: return [(i.X, i.Y, i.Z) for i in list(msh.Vertices)]


def get_np_mesh_vertices(msh) -> np.ndarray: return np.asarray(get_mesh_vertices(msh))


def get_mesh_normals(msh): return [(i.X, i.Y, i.Z) for i in list(msh.Normals)]


def get_np_mesh_normals(msh) -> np.ndarray: return np.asarray(get_mesh_normals(msh))


def create_buffer(indices, verts, normals, uv, uid=None):
    return {
        "uuid": uuid.uuid4().__str__() if uid is None else uid,
        "type": "BufferGeometry",
        "data": {
            "attributes": {
                "normal": {
                    "array": np.asarray(normals, dtype=float).flatten().tolist(),
                    "itemSize": 3,
                    "type": "Float32Array",
                    "normalized": False
                },
                "position": {
                    "array": np.asarray(verts, dtype=float).flatten().tolist(),
                    "itemSize": 3,
                    "type": "Float32Array",
                    "normalized": False
                },
                "uv": {
                    'itemSize': 2,
                    "array": np.asarray(uv, dtype=float).flatten().tolist(),
                    "type": "Float32Array",
                    "normalized": False
                }
            },
            "index": dict(type='Uint16Array',
                          array=np.asarray(indices, dtype=int).flatten().tolist())
        }
    }


def mesh_decomposition(msh):
    f = msh.Faces
    f.ConvertQuadsToTriangles()
    vertss = get_mesh_vertices(msh)
    normals = get_mesh_normals(msh)
    llst = []
    vv = []
    for i in range(f.TriangleCount):
        pts = f.GetFaceVertices(i)
        lst = []
        for i in range(3):
            pt = pts[i + 1]
            vv.append((pt.X, pt.Y, pt.Z))
            try:
                lst.append(vertss.index((pt.X, pt.Y, pt.Z)))
            except ValueError as err:
                vrt = list(range(len(vertss)))
                vrt.sort(key=lambda x: spdist.cosine([pt.X, pt.Y, pt.Z], vertss[x]))
                lst.append(vrt[0])

        llst.append(lst)

    return llst, vertss, normals


def rhino_mesh_to_topology(input_mesh):
    indices, verts, normals = mesh_decomposition(input_mesh)
    uv = get_np_mesh_uv(input_mesh)
    return CommonMeshTopology(indices, verts, normals, uv)


def mesh_to_buffer(input_mesh) -> dict:
    indices, verts, normals = mesh_decomposition(input_mesh)
    uv = get_np_mesh_uv(input_mesh)

    return create_buffer(indices, verts, normals, uv)


def mesh_to_buffer_geometry(input_mesh) -> dict:
    (indices, position, normals), uv = mesh_decomposition(input_mesh), get_np_mesh_uv(input_mesh)
    return dict(indices=indices, position=position, normals=normals, uv=uv)


def mesh_to_buffer_mesh(mesh,

                        material,
                        name="TestMesh",

                        cast_shadow=True,
                        receive_shadow=True,
                        layers=1,
                        matrix=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)):
    geom = mesh_to_buffer(mesh)
    return {
        "metadata": {
            "version": 4.5,
            "type": "Object",
            "generator": "Object3D.toJSON"
        },
        "geometries": [geom],
        "materials": [material.data],
        "object": {
            "uuid": uuid.uuid4().__str__(),
            "type": "Mesh",
            "name": name,
            "castShadow": cast_shadow,
            "receiveShadow": receive_shadow,
            "layers": layers,
            "matrix": matrix,
            "geometry": geom["uuid"],
            "material": material.uuid
        }

    }


def obj_notation_from_mesh(name, geometry_uuid, material_uuid, userdata={},
                           matrix=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), uid=None):
    if uid is None:
        uid = uuid.uuid4().__str__()
    return {
        'uuid': uid,
        'type': 'Mesh',
        'name': name,
        'castShadow': True,
        'receiveShadow': True,
        'userData': userdata,
        'layers': 1,
        'matrix': matrix,
        'geometry': geometry_uuid,
        'material': material_uuid
    }


def group_notation_from_mesh(name, userdata={},
                             matrix=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), children=[], uid=None):
    if uid is None:
        uid = uuid.uuid4().__str__()
    return {
        'uuid': uid,
        'type': 'Group',
        'name': name,
        'castShadow': True,
        'receiveShadow': True,
        'userData': userdata,
        'layers': 1,
        'matrix': matrix,
        'children': children
    }


def tree_obj_from_mesh_obj(obj):
    return obj_notation_from_mesh(obj.name, geometry_uuid=obj.geometry, material_uuid=obj.material,
                                  userdata=obj.userData, matrix=obj.matrix)


CommonMeshTopology = namedtuple("CommonMeshTopology", ["indices", "vertices", "normals", "uv"])
