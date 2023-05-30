from collections import deque

import shapely
from operator import itemgetter

import uuid

import abc
import dataclasses
import json

import typing

import uuid as muuid
import uuid as _uuid
import numpy as np
import mmcore.base.models.gql
from mmcore.base.basic import Object3D, Group, AMesh
from mmcore.base.geom.builder import MeshBufferGeometryBuilder, RhinoMeshBufferGeometryBuilder, \
    RhinoBrepBufferGeometryBuilder, DictToAnyConvertor, DataclassToDictConvertor, Convertor
from mmcore.base.geom.utils import create_buffer_from_dict, parse_attribute
from mmcore.base.models.gql import GqlGeometry, GqlLine, GqlPoints, MeshPhongMaterial, Material, PointsMaterial, \
    LineBasicMaterial
from mmcore.base.utils import getitemattr

MODE = {"children": "parents"}
from mmcore.geom.materials import ColorRGB
from mmcore.base.registry import geomdict, matdict
from mmcore.collections.multi_description import ElementSequence

DEFAULTCOLOR = 9868950  # 150, 150, 150
buffer_geometry_type_map = {
    "position": "vertices",
    "normal": "normals",
    "index": "indices",
    "color": "colors",
    'uv': "uv",
    'uuid': 'uuid'
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

matdict["MeshPhongMaterial"] = mmcore.base.models.gql.MeshPhongMaterial(color=ColorRGB(50, 50, 50).decimal,
                                                                        type=mmcore.base.models.gql.Materials.MeshPhongMaterial)
matdict["PointsMaterial"] = mmcore.base.models.gql.PointsMaterial(color=ColorRGB(50, 50, 50).decimal)
matdict["LineBasicMaterial"] = mmcore.base.models.gql.LineBasicMaterial(color=ColorRGB(50, 50, 50).decimal,
                                                                        type=mmcore.base.models.gql.Materials.LineBasicMaterial)


class GeometryObject(Object3D):
    """
    GeometryObject.
    @note It should be used to implement three-dimensional representations of objects.  Note that geometry
    has no children's field. The Three JS Api does not explicitly state that they exist, however, in most cases,
    attempting to nest objects of geometric types will cause the nested object to lose its display. In short,
    we do not recommend doing so.
    """
    bind_class = GqlGeometry

    material_type = mmcore.base.models.gql.BaseMaterial
    _material: typing.Optional[str] = "MeshPhongMaterial"
    _geometry: typing.Optional[str] = None
    _color: typing.Optional[ColorRGB] = ColorRGB(128, 128, 128)
    _children = ()

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, v):
        if isinstance(v, dict):
            self._color = ColorRGB(*list(v.values()))
        elif isinstance(v, list) or isinstance(v, tuple):
            self._color = ColorRGB(*v)
        elif isinstance(v, ColorRGB):
            self._color = v
        else:
            raise ValueError("\n\n\t\t:(\n\n")
        mat = self.material_type(color=self._color.decimal)
        self._material = mat.uuid
        self.material = mat

    def __call__(self, *args, **kwargs):
        s = super().__call__(*args, **kwargs)
        self.solve_geometry()
        return s

    @property
    def geometry(self) -> mmcore.base.models.gql.BufferGeometry:
        """
        Geometry property
        @note You can override this property at your discretion (e.g. to implement dynamic material).
        The only conditions are:
            [1] self._geometry : returns the string uuid
            [2] self.geometry  : strawberry.type e.g mmcore.base.models.BufferGeometry

        @return: mmcore.base.models.BufferGeometry

        """

        return geomdict.get(self._geometry)

    @geometry.setter
    def geometry(self, v):
        """
        Geometry property
        @param v: You can set this property with dict or BufferGeometry object, or uuid (object
        with this uuid will exist .
        @note You can also override this property at your discretion (e.g. to implement dynamic material).
        The only conditions are:
            [1] self._geometry : returns the string uuid
            [2] self.geometry  : strawberry.type e.g mmcore.base.geoms.BufferGeometry

        @return: None

        """
        try:
            del geomdict[self._geometry]
        except KeyError:
            pass

        if isinstance(v, str):
            self._geometry = v
        elif isinstance(v, dict):

            self._geometry = getitemattr("uuid")(v)
            geomdict[self._geometry] = create_buffer_from_dict(v)

        else:

            self._geometry = v.uuid
            # print(f"Geometry set event: {self.name} <- {self._geometry}")
            geomdict[self._geometry] = v

    @property
    def material(self) -> mmcore.base.models.gql.Material:
        """
        Material property
        @note If you use mmcore.geom.materials api , you must pass <material object>.data . You can also override
        this property at your discretion (e.g. to implement dynamic material). The only conditions are:
            [1] self._material : returns the string uuid
            [2] self.material : strawberry.type e.g mmcore.base.models.Material
        @return: mmcore.base.models.Material

        """
        mat = matdict.get(self._material)
        mat.color = self.color.decimal
        return mat

    @material.setter
    def material(self, v):
        """
        Material property
        @param v: You can set this property with dict or models.Material object, or uuid (object
        with this uuid will exist .
        @note If you use mmcore.geom.materials api , you must pass <material object>.data . You can also override
        this property at your discretion (e.g. to implement dynamic material). The only conditions are:
            [1] self._material : returns the string uuid
            [2] self.material : strawberry.type e.g mmcore.base.models.Material

        @return: None

        """
        es = ElementSequence(list(matdict.values()))

        if isinstance(v, dict):
            if 'metadata' in v:
                v.pop('metadata')

            m = self.material_type(**v)
            self._material = m.uuid
            matdict[self._material] = m

        else:

            self._material = v.uuid
            matdict[self._material] = v

    def threejs_type(self):
        return "Geometry"

    @property
    def threejs_repr(self):
        rpr = super().threejs_repr
        rpr |= {"geometry": self._geometry,
                "material": self._material,
                "castShadow": True,
                "receiveShadow": True}
        return rpr

    @classmethod
    def from_three(cls, obj, geom, material, **kwargs) -> 'GeometryObject':

        inst = cls.from_three(obj)
        inst.geometry = MeshBufferGeometryBuilder.from_three(geom)
        inst.material = cls.material_type(**material)
        return inst

    @abc.abstractmethod
    def solve_geometry(self):
        ...


from mmcore.geom.vectors import triangle_normal


class BufferGeometryToMeshDataConvertor(Convertor):
    type_map = buffer_geometry_type_map
    source: mmcore.base.models.gql.BufferGeometry

    def __init__(self, source, **kwargs):
        super().__init__(source, **kwargs)

    def convert(self) -> 'MeshData':
        dct = dict()
        dct["uuid"] = self.source.uuid

        if hasattr(self.source.data, "index"):
            dct["index"] = self.source.data.index.array

        for k, v in DataclassToDictConvertor(self.source.data.attributes).convert().items():
            dct[k] = v['array']

        return DictToAnyConvertor(dct, MeshData, type_map=self.type_map).convert()


class BufferGeometryDictToMeshDataConvertor(BufferGeometryToMeshDataConvertor):
    def convert(self) -> 'MeshData':
        return BufferGeometryToMeshDataConvertor(create_buffer_from_dict(self.source)).convert()


@dataclasses.dataclass
class MeshData:
    vertices: typing.Union[list, tuple]
    normals: typing.Optional[typing.Union[list, tuple]] = None
    indices: typing.Optional[typing.Union[list, tuple]] = None
    uv: typing.Optional[typing.Union[list, tuple]] = None
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
        for a, b, c in self.indices[:3]:
            self.normals.append(
                triangle_normal(np.array(self.vertices[a]), np.array(self.vertices[b]), np.array(self.vertices[c])))

    def __post_init__(self):
        self._buf = None
        if self.uuid is None:
            self.uuid = uuid.uuid4().hex

    def asdict(self):
        dct = {}
        for k, v in dataclasses.asdict(self).items():
            if v is not None:
                dct[k] = v
        return dct

    def create_buffer(self) -> mmcore.base.models.gql.BufferGeometry:
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
        dct = dict()
        if ("indices" in self.__dict__.keys()) and ("indices" in other.__dict__.keys()):
            dct["indices"] = self.indices + (np.array(other.indices, dtype=int) + count).tolist()

        for k in self.__dict__.keys():
            if (k in other.__dict__) and (k != "indices") and (k != "uuid"):
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

    def to_mesh(self):
        return AMesh(geometry=self.create_buffer(), material=MeshPhongMaterial(color=DEFAULTCOLOR))

    @classmethod
    def from_buffer_geometry(cls, geom: typing.Union[mmcore.base.models.gql.BufferGeometry, dict]) -> 'MeshData':
        if isinstance(geom, dict):
            return BufferGeometryDictToMeshDataConvertor(geom).convert()
        return BufferGeometryToMeshDataConvertor(geom).convert()


class MeshObject(GeometryObject):
    mesh: typing.Union[MeshData, dict] = MeshData([])
    __match_args__ = "mesh"

    def solve_geometry(self):

        if isinstance(self.mesh, MeshData):

            self.geometry = MeshBufferGeometryBuilder(**self.mesh.asdict()).create_buffer()

        elif isinstance(self.mesh, dict):
            self.geometry = MeshBufferGeometryBuilder(**self.mesh).create_buffer()
        elif isinstance(self.mesh, mmcore.base.models.gql.BufferGeometry):
            self.geometry = self.mesh

            # TODO Добавить property для всех обязательных аттрибутов меши

    geometry_type = MeshBufferGeometryBuilder
    castShadow: bool = True
    receiveShadow: bool = True
    material_type = MeshPhongMaterial

    @property
    def threejs_type(self):
        return "Mesh"

    @classmethod
    def from_rhino(cls, name, mesh, uuid=None):
        if uuid is None:
            uuid = muuid.uuid4().__str__()
        inst = cls(name=name)
        inst._uuid = uuid
        inst.geometry = RhinoMeshBufferGeometryBuilder(mesh).create_buffer()
        inst._rhino = mesh

        return inst


class RhinoBrepObject(MeshObject):
    material_type = MeshPhongMaterial
    brep: typing.Any

    def solve_geometry(self):
        self.geometry = RhinoBrepBufferGeometryBuilder(self.brep).create_buffer()


class RhinoMeshObject(MeshObject):
    def solve_geometry(self):
        self._geometry = self.uuid + "-geom"
        self.geometry = RhinoMeshBufferGeometryBuilder(self.mesh).create_buffer()


from mmcore.node import node_eval


class PointsObject(GeometryObject):
    _material: typing.Optional[str] = "PointsMaterial"
    _geometry: typing.Optional[str]
    __match_args__ = "points",
    material_type = PointsMaterial

    bind_class = GqlPoints
    name: str = "PointsObject"
    _color = ColorRGB(128, 128, 128)
    _points = None

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, v):
        self._points = v
        self.solve_geometry()

    @property
    def threejs_type(self):
        return "Points"

    def append(self, v):
        self._points.append(v)
        self.solve_geometry()

    def solve_geometry(self):
        try:
            del geomdict[self._geometry]

        except KeyError:
            pass
        self.geometry = mmcore.base.models.gql.BufferGeometryObject(**{
            'data': mmcore.base.models.gql.Data1(
                **{'attributes': mmcore.base.models.gql.Attributes1(
                    **{'position': mmcore.base.models.gql.Position(
                        **{'itemSize': 3,
                           'type': 'Float32Array',
                           'array': np.array(
                               self.points).flatten().tolist(),
                           'normalized': False})})})})

    @property
    def properties(self):
        return dict(points=self.points)


class SharedGeometryObject(GeometryObject):
    target: GeometryObject
    width: int = 2
    bind_class = GqlLine
    material_type = LineBasicMaterial
    _material: typing.Optional[str] = 'LineBasicMaterial'

    @property
    def geometry(self):
        return self.target.geometry

    @geometry.setter
    def geometry(self, v):
        self.target(**v)
        self.target.solve_geometry()


class LineObject(PointsObject):
    width: int = 2
    bind_class = GqlLine
    material_type = LineBasicMaterial
    _material: typing.Optional[str] = 'LineBasicMaterial'
    _geometry: typing.Optional[str] = None

    @property
    def threejs_type(self):
        return "Line"

    def solve_geometry(self):
        super().solve_geometry()


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


def hyp(arr):
    d = arr[1].reshape((3, 1)) + ((arr[0] - arr[1]).reshape((3, 1)) * np.stack(
        [np.linspace(0, 1, num=10), np.linspace(0, 1, num=10), np.linspace(0, 1, num=10)]))
    c = arr[2].reshape((3, 1)) + ((arr[1] - arr[2]).reshape((3, 1)) * np.stack(
        [np.linspace(0, 1, num=10), np.linspace(0, 1, num=10), np.linspace(0, 1, num=10)]))
    f = arr[2].reshape((3, 1)) + ((arr[3] - arr[2]).reshape((3, 1)) * np.stack(
        [np.linspace(0, 1, num=10), np.linspace(0, 1, num=10), np.linspace(0, 1, num=10)]))
    g = arr[3].reshape((3, 1)) + ((arr[0] - arr[3]).reshape((3, 1)) * np.stack(
        [np.linspace(0, 1, num=10), np.linspace(0, 1, num=10), np.linspace(0, 1, num=10)]))

    grp = Group(name="grp")

    lns2 = list(zip(g.T, c.T))
    lns = list(zip(f.T, d.T))
    for i, (lna, lnb) in enumerate(lns):
        grp.add(LineObject(name=f"1-{i}", points=(lna.tolist(), lnb.tolist())))
    for i, (lna, lnb) in enumerate(lns2):
        grp.add(LineObject(name=f"2-{i}", points=(lna.tolist(), lnb.tolist())))
