import dataclasses
import json
import typing
import uuid
from collections import deque
from operator import itemgetter

import more_itertools
import numpy as np
import mmcore.base.models.gql as gql_models
import mmcore.base.registry
from mmcore.base.basic import AMesh
from mmcore.base.geom.builder import Convertor, DataclassToDictConvertor, DictToAnyConvertor, \
    DictToAnyMeshDataConvertor, MeshBufferGeometryBuilder, RhinoBrepBufferGeometryBuilder, \
    RhinoMeshBufferGeometryBuilder
from mmcore.base.geom.utils import parse_attribute, points_traverse
from mmcore.node import node_eval

MeshPhongMaterial, create_buffer_from_dict = gql_models.MeshPhongMaterial, gql_models.create_buffer_from_dict
MODE = {"children": "parents"}
from mmcore.geom.materials import ColorRGB
from mmcore.base.registry import matdict

DEFAULTCOLOR = 9868950  # 150, 150, 150
buffer_geometry_type_map = {
    "position": "vertices",
    "normal": "normals",
    "index": "indices",
    "color": "colors",
    'uv': "uv",
    'uuid': 'uuid'
}
buffer_geometry_itemsize_map = {
    "position": 3,
    "normal": 3,
    "index": 3,
    "color": 3,
    'uv': 2
}


def _buff_attr_checker(builder, name, attributes, cls):
    value = getattr(builder, name)
    if value is not None:
        attributes["normal"] = cls(**{
            "array": np.asarray(value, dtype=float).flatten().tolist(),
            "type": "Float32Array",
            "normalized": False
        })


T = typing.TypeVar("T")
S = typing.TypeVar("S")

matdict["MeshPhongMaterial"] = gql_models.MeshPhongMaterial(color=ColorRGB(50, 50, 50).decimal,
                                                            type=gql_models.Materials.MeshPhongMaterial)
matdict["PointsMaterial"] = gql_models.PointsMaterial(color=ColorRGB(50, 50, 50).decimal)
matdict["LineBasicMaterial"] = gql_models.LineBasicMaterial(color=ColorRGB(50, 50, 50).decimal,
                                                            type=gql_models.Materials.LineBasicMaterial)

from mmcore.geom.vectors import triangle_normal


class BufferGeometryToMeshDataConvertor(Convertor):
    type_map = buffer_geometry_type_map
    source: gql_models.BufferGeometry

    def __init__(self, source, **kwargs):
        super().__init__(source, **kwargs)

    def convert(self) -> 'MeshData':
        dct = dict()
        dct["uuid"] = self.source.uuid

        if hasattr(self.source.data, "index"):
            dct["index"] = self.source.data.index.array

        for k, v in DataclassToDictConvertor(self.source.data.attributes).convert().items():
            dct[k] = v['array']

        return DictToAnyMeshDataConvertor(dct, MeshData, type_map=self.type_map).convert()


class BufferGeometryDictToMeshDataConvertor(BufferGeometryToMeshDataConvertor):
    def convert(self) -> 'MeshData':
        return BufferGeometryToMeshDataConvertor(create_buffer_from_dict(self.source)).convert()


@dataclasses.dataclass
class MeshData:
    vertices: typing.Union[list, tuple, np.ndarray] = None
    normals: typing.Optional[typing.Union[list, tuple, np.ndarray]] = None
    indices: typing.Optional[typing.Union[list, tuple, np.ndarray]] = None
    uv: typing.Optional[typing.Union[list, tuple, np.ndarray]] = None
    uuid: typing.Optional[str] = None
    _buff = None

    def calc_indices(self):
        d = deque(range(len(self.vertices)))
        d1 = d.copy()
        d2 = d.copy()
        d2.rotate(-1)
        d.rotate(1)
        *l, = zip(d1, d, d2)
        self.indices = np.array(l, dtype=int).tolist()

    def calc_normals(self):
        self.normals = []
        for face in self.faces:
            self.normals.append(triangle_normal(*face))

    def __post_init__(self):
        self._buf = None
        if self.uuid is None:
            self.uuid = uuid.uuid4().hex
        if self.indices is not None:
            f = np.array(self.indices).flatten()

            self.indices = np.array(self.indices).flatten().reshape((len(f) // 3, 3))

    def asdict(self):
        dct = {}
        for k, v in dataclasses.asdict(self).items():
            if v is not None:
                dct[k] = v
        return dct

    def create_buffer(self) -> gql_models.BufferGeometry:
        if self._buf is None:
            self._buf = MeshBufferGeometryBuilder(**self.asdict()).create_buffer()
        return self._buf

    def get_face(self, item):
        if (self.indices is not None) and (self.indices != ()):
            return itemgetter(*self.indices[item])(self.vertices)

    @property
    def faces(self):

        if (self.indices is not None) and (self.indices != ()):
            l = []
            for i in range(len(self.indices)):
                l.append(self.get_face(i))
            return l

    def translate(self, v):
        self.vertices = np.array(self.vertices) + np.array(v)

    def merge(self, other: 'MeshData'):
        count = np.array(self.indices).max()

        dct = dict(vertices=np.array(self.vertices).tolist() + np.array(other.vertices).tolist())
        if ("indices" in self.__dict__.keys()) and ("indices" in other.__dict__.keys()):
            dct["indices"] = np.array(self.indices).tolist() + (np.array(other.indices, dtype=int) + count).tolist()

        for k in self.__dict__.keys():
            if (k in other.__dict__) and (k != "indices") and (k != "uuid") and (k != "vertices"):
                if all([not isinstance(self.__dict__[k], np.ndarray),
                        not isinstance(other.__dict__[k], np.ndarray),
                        self.__dict__[k] is not None,
                        other.__dict__[k] is not None]):

                    dct[k] = self.__dict__[k] + other.__dict__[k]
                elif isinstance(self.__dict__[k], np.ndarray) and not isinstance(other.__dict__[k], np.ndarray):
                    dct[k] = self.__dict__[k].tolist() + other.__dict__[k]
                elif isinstance(other.__dict__[k], np.ndarray) and not isinstance(self.__dict__[k], np.ndarray):
                    dct[k] = self.__dict__[k] + other.__dict__[k].tolist()
                elif isinstance(other.__dict__[k], np.ndarray) and isinstance(self.__dict__[k], np.ndarray):
                    dct[k] = self.__dict__[k].tolist() + other.__dict__[k].tolist()
                else:
                    pass
        dct["uuid"] = uuid.uuid4().hex

        return MeshData(**dct)

    def merge2(self, other: 'MeshData'):
        ixs = np.array(self.indices, dtype=int)
        ixsf = ixs.flatten()
        mx = np.max(ixsf)
        keys = ["vertices",
                "normals",

                "uv"]

        kws = dict()
        for k in keys:
            val = getattr(self, k)
            if val:
                kws[k] = np.array(val).tolist() + np.array(val).tolist()

        return MeshData(**kws, indices=list(
            more_itertools.chunked(ixsf.tolist() + (np.array(other.indices, dtype=int).flatten() + mx).tolist(), 3)))

    def to_mesh(self, color=None, flatShading=True, opacity=1.0, cls=AMesh, name="Mesh", **kwargs):
        return cls(name=name, geometry=self.create_buffer(),
                     material=MeshPhongMaterial(color=DEFAULTCOLOR if color is None else color, opacity=opacity,
                                                flatShading=flatShading), **kwargs)

    @classmethod
    def from_buffer_geometry(cls, geom: typing.Union[gql_models.BufferGeometry, dict]) -> 'MeshData':

        return cls.from_buffer(geom)

    @classmethod
    def from_buffer(cls, geom: typing.Union[gql_models.BufferGeometry, dict]) -> 'MeshData':
        if isinstance(geom, dict):
            return BufferGeometryDictToMeshDataConvertor(geom).convert()
        return BufferGeometryToMeshDataConvertor(geom).convert()

    def to_buffer(self) -> gql_models.BufferGeometry:
        return self.create_buffer()


@node_eval
def pointsMaterial(color):
    line = f"makePointMaterial({color.decimal})"
    # language=JavaScript
    return '''const THREE = require("three");
                      function makePointMaterial( color) {
                            const mat = new THREE.PointsMaterial({color: color})
                            console.log(JSON.stringify(mat.toJSON()));
                      }; ''' + line


@node_eval
def lineMaterial(color, width, opacity=1.):
    line = f"makePointMaterial({color.decimal}, {width}, {opacity}, {json.dumps(opacity < 1.)})"
    # language=JavaScript
    return '''const THREE = require("three");
                      function makePointMaterial( color, width, opacity, transparent) {
                            const mat = new THREE.LineBasicMaterial({color: color, linewidth:width, opacity:opacity, transparent:transparent})
                            console.log(JSON.stringify(mat.toJSON()));
                      }; ''' + line
