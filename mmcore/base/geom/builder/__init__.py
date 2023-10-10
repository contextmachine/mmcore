import dataclasses
import typing

import more_itertools
import numpy as np
from scipy.spatial import distance as spdist
from typing_extensions import runtime_checkable

import mmcore.base.models.gql as gql
from mmcore.base.geom.utils import parse_attribute
from mmcore.base import create_buffer_from_dict

buffer_geometry_itemsize_map = {
    "position": 3,
    "normal": 3,
    "index": 3,
    "color": 3,
    'uv': 2
}

T = typing.TypeVar("T")
S = typing.TypeVar("S")
@runtime_checkable
class BufferGeometryBuilder(typing.Protocol):
    choices = {
        '000': (gql.Data1, gql.Attributes1),
        '001': (gql.Data, gql.Attributes1),
        '100': (gql.Data1, gql.Attributes3),
        '101': (gql.Data, gql.Attributes3),
        '110': (gql.Data1, gql.Attributes2),
        '011': (gql.Data, gql.Attributes2),
        '111': (gql.Data, gql.Attributes2)
    }

    def create_buffer(self, uuid: typing.Optional[str] = None) -> gql.BufferGeometry: ...


class MeshBufferGeometryBuilder(BufferGeometryBuilder):
    _uv = None
    _vertices = None
    _normals = None
    _indices = None

    @classmethod
    def from_three(cls, dct: dict) -> 'MeshBufferGeometryBuilder':
        buff = create_buffer_from_dict(dct)
        return cls(indices=parse_attribute(buff.data.index),
                   vertices=parse_attribute(gql.attributes.position),
                   normals=parse_attribute(gql.attributes.normal),
                   uv=parse_attribute(gql.attributes.uv),
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

    def create_buffer(self, uuid: str = None) -> gql.BufferGeometry:

        selector = ['0', '0', '0']  # norm,uv,index
        attributes = {
            "position": gql.Position(**{
                "array": np.asarray(self.vertices, dtype=float).flatten().tolist(),
                "itemSize": 3,
                "type": "Float32Array",
                "normalized": False
            }),
        }
        data = {}

        if self.normals is not None:
            selector[0] = '1'
            attributes["normal"] = gql.Normal(**{
                "array": np.asarray(self.normals, dtype=float).flatten().tolist(),
                "itemSize": 3,
                "type": "Float32Array",
                "normalized": False
            })

        if self.uv is not None:
            selector[1] = '1'
            attributes["uv"] = gql.Uv(**{
                'itemSize': 2,
                "array": np.array(self.uv, dtype=float).flatten().tolist(),
                "type": "Float32Array",
                "normalized": False
            })

        if self.indices is not None:
            #print(self.indices)

            selector[2] = '1'

            indices=list(more_itertools.flatten(self.indices))
            data["index"] = gql.Index(**dict(type='Uint16Array',
                                                                array=np.array(indices,
                                                                               dtype=int).flatten().tolist()))
        _data, _attribs = self.choices["".join(selector)]
        data['attributes'] = _attribs(**attributes)


        # ##print(selector)
        return gql.BufferGeometry(**{
            "type": "BufferGeometry",
            "data": _data(**data)
        })


class PointsBufferGeometryBuilder(BufferGeometryBuilder):
    _points = None
    _colors = None
    choices = {
        '0': (gql.Data1, gql.Attributes1),
        '1': (gql.Data1, gql.Attributes4),
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

    def create_buffer(self, uuid: typing.Optional[str] = None) -> gql.BufferGeometry:

        selector = ['0']  # norm,uv,index
        attributes = {
            "position": gql.Position(**{
                "array": np.asarray(self.points, dtype=float).flatten().tolist(),
                "itemSize": 3,
                "type": "Float32Array",
                "normalized": False
            }),
        }
        data = {}
        if self.colors is not None:
            selector[0] = '1'
            attributes["colors"] = gql.Uv(**{
                'itemSize': 1,
                "array": np.array(self.colors, dtype=float).flatten().tolist(),
                "type": 'Uint16Array',
                "normalized": False
            })

        _data, _attribs = self.choices["".join(selector)]
        data['attributes'] = _attribs(**attributes)

        # ##print(selector)
        return gql.BufferGeometry(**{

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

    def create_buffer(self) -> gql.BufferGeometry:

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



class ConvertorProtocol(typing.Protocol[S, T]):
    type_map: typing.Optional[dict]

    def convert(self, obj: S) -> T: ...


class Convertor(typing.Generic[S, T]):
    target: typing.Type[T]
    source: S
    type_map: dict = None

    def __init__(self, source: S, type_map=None, **kwargs):
        super().__init__()
        if type_map is None:
            if self.type_map is None:
                self.type_map = dict()
        else:
            self.type_map = type_map
        self.source = source
        self.__dict__ |= kwargs

    def convert(self) -> T:
        ...




class DictToAnyConvertor(Convertor[dict, typing.Any]):
    source: dict
    target: typing.Any
    def __init__(self, source: S, target, **kwargs):
        super().__init__(source, **kwargs)
        self.target=target
    def convert(self) -> typing.Any:
        dct = dict()
        for k, v in self.source.items():
            dct[self.type_map[k]] = v
        return self.target(**dct)

class DictToAnyMeshDataConvertor(Convertor[dict, typing.Any]):
    source: dict
    target: typing.Any
    def __init__(self, source: S, target, **kwargs):
        super().__init__(source, **kwargs)
        self.target=target
    def convert(self) -> typing.Any:
        dct = dict()
        for k, v in self.source.items():
            flt=np.array(v).flatten()
            #print("VVV", k, v)
            if k in buffer_geometry_itemsize_map.keys():
                dct[self.type_map[k]] = flt.reshape((flt.shape[0]//3,3)).tolist()
            else:
                dct[self.type_map[k]]=v
        return self.target(**dct)
class DataclassToDictConvertor(Convertor[typing.Any, dict]):

    def convert(self) -> dict:
        return dataclasses.asdict(self.source)

