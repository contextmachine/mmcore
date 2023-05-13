import typing
import uuid as _uuid

import numpy as np
from scipy.spatial import distance as spdist
from typing_extensions import runtime_checkable

import mmcore.base
from mmcore.base.geom.utils import create_buffer_from_dict, parse_attribute


@runtime_checkable
class BufferGeometryBuilder(typing.Protocol):
    choices = {
        '000': (mmcore.base.gql_models.Data1, mmcore.base.gql_models.Attributes1),
        '100': (mmcore.base.gql_models.Data1, mmcore.base.gql_models.Attributes3),
        '110': (mmcore.base.gql_models.Data1, mmcore.base.gql_models.Attributes2),
        '111': (mmcore.base.gql_models.Data, mmcore.base.gql_models.Attributes2)
    }

    def create_buffer(self, uuid: typing.Optional[str] = None) -> mmcore.base.models.gql.BufferGeometry: ...


class MeshBufferGeometryBuilder(BufferGeometryBuilder):
    _uv = None
    _vertices = None
    _normals = None
    _indices = None

    @classmethod
    def from_three(cls, dct: dict) -> 'MeshBufferGeometryBuilder':
        buff = create_buffer_from_dict(dct)
        return cls(indices=parse_attribute(buff.data.index),
                   vertices=parse_attribute(buff.data.attributes.position),
                   normals=parse_attribute(buff.data.attributes.normal),
                   uv=parse_attribute(buff.data.attributes.uv),
                   uuid=buff.uuid
                   )

    def __init__(self, vertices=None, indices=None, normals=None, uv=None, uuid=None):
        self._uuid = uuid

        self._vertices, self._indices, self._normals, self._uv = vertices, indices, normals, uv

    @property
    def uuid(self):
        return self._uuid

    @property
    def uv(self):
        return self._uv

    @property
    def normals(self):
        return self._normals

    @property
    def vertices(self):
        return self._vertices

    @property
    def indices(self):
        return self._indices

    def create_buffer(self, uuid: str = None) -> mmcore.base.models.gql.BufferGeometry:
        selector = ['0', '0', '0']  # norm,uv,index
        attributes = {
            "position": mmcore.base.models.gql.Position(**{
                "array": np.asarray(self.vertices, dtype=float).flatten().tolist(),
                "itemSize": 3,
                "type": "Float32Array",
                "normalized": False
            }),
        }
        data = {}

        if self.normals is not None:
            selector[0] = '1'
            attributes["normal"] = mmcore.base.models.gql.Normal(**{
                "array": np.asarray(self.normals, dtype=float).flatten().tolist(),
                "itemSize": 3,
                "type": "Float32Array",
                "normalized": False
            })

        if self.uv is not None:
            selector[1] = '1'
            attributes["uv"] = mmcore.base.models.gql.Uv(**{
                'itemSize': 2,
                "array": np.array(self.uv, dtype=float).flatten().tolist(),
                "type": "Float32Array",
                "normalized": False
            })

        if self.indices is not None:
            print(self.indices)

            selector[2] = '1'
            data["index"] = mmcore.base.models.gql.Index(**dict(type='Uint16Array',
                                                                array=np.asarray(self.indices,
                                                                                 dtype=int).flatten().tolist()))
        _data, _attribs = self.choices["".join(selector)]
        data['attributes'] = _attribs(**attributes)


        # print(selector)
        return mmcore.base.gql_models.BufferGeometry(**{
            "type": "BufferGeometry",
            "data": _data(**data)
        })


class PointsBufferGeometryBuilder(BufferGeometryBuilder):
    _points = None
    _colors = None
    choices = {
        '0': (mmcore.base.gql_models.Data1, mmcore.base.gql_models.Attributes1),
        '1': (mmcore.base.gql_models.Data1, mmcore.base.gql_models.Attributes4),
    }

    def __init__(self, points=None, colors=None):

        self.points = points
        self.colors = colors

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, v):
        self._colors = v

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, v):
        self._points = v

    def create_buffer(self, uuid: typing.Optional[str] = None) -> mmcore.base.models.gql.BufferGeometry:

        selector = ['0']  # norm,uv,index
        attributes = {
            "position": mmcore.base.models.gql.Position(**{
                "array": np.asarray(self.points, dtype=float).flatten().tolist(),
                "itemSize": 3,
                "type": "Float32Array",
                "normalized": False
            }),
        }
        data = {}
        if self.colors is not None:
            selector[0] = '1'
            attributes["colors"] = mmcore.base.models.gql.Uv(**{
                'itemSize': 1,
                "array": np.array(self.colors, dtype=float).flatten().tolist(),
                "type": 'Uint16Array',
                "normalized": False
            })

        _data, _attribs = self.choices["".join(selector)]
        data['attributes'] = _attribs(**attributes)

        # print(selector)
        return mmcore.base.gql_models.BufferGeometry(**{

            "type": "BufferGeometry",
            "data": _data(**data)
        })


class RhinoMeshBufferGeometryBuilder(MeshBufferGeometryBuilder):
    def __init__(self, rhino_mesh, **kwargs):

        self._data = rhino_mesh
        super().__init__(**kwargs)

    def calc_uv(self):
        return [(i.X, i.Y) for i in list(self._data.TextureCoordinates)]

    def calc_vertices(self):
        return np.asarray(self.get_mesh_vertices(self._data))

    def calc_normals(self):
        return [(i.X, i.Y, i.Z) for i in list(self._data.Normals)]

    def mesh_decomposition(self):
        f = self._data.Faces
        f.ConvertQuadsToTriangles()
        self._verices = self.calc_vertices()
        self._normals = self.calc_normals()
        self._indices = []
        vv = []
        for i in range(f.TriangleCount):
            pts = f.GetFaceVertices(i)
            lst = []
            for i in range(3):
                pt = pts[i + 1]
                vv.append((pt.X, pt.Y, pt.Z))
                try:
                    lst.append(self._verices.index((pt.X, pt.Y, pt.Z)))
                except ValueError as err:
                    vrt = list(range(len(self._verices)))
                    vrt.sort(key=lambda x: spdist.cosine([pt.X, pt.Y, pt.Z], self._verices[x]))
                    lst.append(vrt[0])

            self._indices.append(lst)

        return self._indices, self._verices, self._normals

    def create_buffer(self) -> mmcore.base.models.gql.BufferGeometry:

        self.mesh_decomposition()
        self.calc_uv()
        return super().create_buffer()


class RhinoBrepBufferGeometryBuilder(RhinoMeshBufferGeometryBuilder):

    def __init__(self, brep, uuid: str):
        try:
            import Rhino.Geometry as rg


        except ImportError:

            import rhino3dm as rg

        mesh = rg.Mesh()
        [mesh.Append(l) for l in list(rg.Mesh.CreateFromBrep(brep, rg.MeshingParameters.FastRenderMesh))]
        super().__init__(mesh, uuid=uuid)
