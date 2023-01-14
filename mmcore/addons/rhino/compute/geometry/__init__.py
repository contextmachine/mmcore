import itertools
import uuid
from typing import Any

import numpy as np
import rhino3dm
import rhino3dm as rh

MESHING_PARAMS = rh.MeshingParameters.FastRenderMesh
MESHING_PARAMS.ComputeCurvature = True
MESHING_PARAMS.Tolerance = 0.01

from mmcore.addons.rhino.compute import Mesh, Surface

from mmcore.addons.rhino import Tuple3dVec, create_buffer, get_mesh_normals, get_mesh_vertices, get_np_mesh_uv


def surf_to_buffer_geometry(surface: rhino3dm.Surface | rhino3dm.NurbsSurface,
                            meshing_params: rhino3dm.MeshingParameters = MESHING_PARAMS) -> dict:
    mesh = Mesh.CreateFromSurface1(surface, meshing_params)
    indices, position, normals = get_triangle_mesh_indices(mesh, surface)
    uv = get_np_mesh_uv(mesh)
    return dict(indices=indices, position=position, normals=normals, uv=uv)


def surf_to_buffer(surface: rhino3dm.Surface | rhino3dm.NurbsSurface,
                   meshing_params: rhino3dm.MeshingParameters = MESHING_PARAMS) -> dict:
    return create_buffer(*surf_to_buffer_geometry(surface, meshing_params=meshing_params).values())


def get_triangle_mesh_indices(msh, srf):
    f = msh.Faces
    f.ConvertQuadsToTriangles()
    vertss = get_mesh_vertices(msh)
    normals = get_mesh_normals(msh)
    ll = []
    vv = []
    for i in range(f.TriangleCount):
        pts = f.GetFaceVertices(i)
        l = []

        # print(pts, verts)

        for i in range(3):
            pt = pts[i + 1]
            vv.append((pt.X, pt.Y, pt.Z))
            try:
                l.append(vertss.index((pt.X, pt.Y, pt.Z)))
            except ValueError as err:
                vertss.append((pt.X, pt.Y, pt.Z))

                l.append(vertss.index((pt.X, pt.Y, pt.Z)))
                nrm = srf.NormalAt(*Surface.ClosestPoint(srf, rh.Point3d(pt.X, pt.Y, pt.Z))[1:])
                normals.append(Tuple3dVec((nrm.X, nrm.Y, nrm.Z)))
            except Exception as err:
                raise err
        ll.append(l)

    return ll, vertss, normals


def surface_closest_normals(srf, points):
    """
    It is too sloooow :(
    """

    uvs = Surface.ClosestPoint(list(itertools.repeat(srf, len(points))), [rh.Point3d(*pt) for pt in points],
                               multiple=True)
    return [srf.NormalAt(*uv) for uv in uvs]


def surf_to_buffer_mesh(surface,
                        meshing_params,
                        name="TestSurface",
                        cast_shadow=True,
                        receive_shadow=True,
                        layers=1,
                        material=None,
                        matrix=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)):
    """
    Low level 3dm to Threejs convertion function
    """

    material.uuid = uuid.uuid4().__str__()
    geom = surf_to_buffer(surface, meshing_params)
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


def get1_triangle_mesh_indices(msh, srf):
    f = msh.Faces
    f.ConvertQuadsToTriangles()
    vertss = get_mesh_vertices(msh)
    normals = get_mesh_normals(msh)

    ll = []
    vv = []
    for i in range(f.TriangleCount):
        pts = f.GetFaceVertices(i)
        l = []

        # print(pts, verts)

        for i in range(3):
            pt = pts[i + 1]
            vv.append((pt.X, pt.Y, pt.Z))
            try:
                l.append(vertss.index((pt.X, pt.Y, pt.Z)))
            except ValueError as err:
                vertss.append((pt.X, pt.Y, pt.Z))

                l.append(vertss.index((pt.X, pt.Y, pt.Z)))
                nrm = srf.NormalAt(*Surface.ClosestPoint(srf, rh.Point3d(pt.X, pt.Y, pt.Z))[1:])
                normals.append(Tuple3dVec((nrm.X, nrm.Y, nrm.Z)))
            except Exception as err:
                raise err
        ll.append(l)

    return ll, vertss, normals


class GeomBack:
    _mesh_parameters = {
        'TextureRange': 2,
        'JaggedSeams': False,
        'RefineGrid': True,
        'SimplePlanes': True,
        'ComputeCurvature': True,
        'ClosedObjectPostProcess': False,
        'GridMinCount': 0,
        'GridMaxCount': 0,
        'GridAngle': 0.0,
        'GridAspectRatio': 1.0,
        'GridAmplification': 0.0,
        'Tolerance': 0.001,
        'MinimumTolerance': 0.0,
        'RelativeTolerance': 1.0,
        'MinimumEdgeLength': 0.0001,
        'MaximumEdgeLength': 0.0,
        'RefineAngle': 0.0
        }

    @property
    def mesh_parameters(self):
        return self._mesh_parameters

    @mesh_parameters.setter
    def mesh_parameters(self, value):
        self._mesh_parameters = value

    def mesh(self, instance):
        return Mesh.CreateFromSurface1(instance.delegate, rh.MeshingParameters.Decode(self.mesh_parameters))

    def __get__(self, instance, owner) -> Any:
        mesh = self.mesh(instance)
        index, position, normal = get_triangle_mesh_indices(mesh, instance.delegate)

        return {
            "attributes": {

                "normal": {
                    "array": np.asarray(normal).flatten().tolist(),
                    "itemSize": 3,
                    "type": "Float32Array",
                    "normalized": False
                    },
                "position": {
                    "array": np.asarray(position, dtype=float).flatten().tolist(),
                    "itemSize": 3,
                    "type": "Float32Array",
                    "normalized": False
                    },
                "uv": {
                    'itemSize': 2,
                    "array": np.asarray(get_np_mesh_uv(mesh), dtype=float).flatten().tolist(),
                    "type": "Float32Array",
                    "normalized": False
                    }
                },
            "index": dict(type='Uint16Array',
                          array=np.asarray(index).flatten().tolist())
            }
