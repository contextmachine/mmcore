import abc
import dataclasses
import json

import typing
import typing_extensions

import uuid as muuid
import uuid as _uuid
import numpy as np
import scipy.spatial.distance as spdist
import strawberry
from typing_extensions import runtime_checkable
import mmcore.base.models.gql
from mmcore.base.basic import Object3D, Group
from mmcore.base.geom.utils import create_buffer_from_dict, parse_attribute
from mmcore.base.models import Point
from mmcore.base.models.gql import GqlGeometry, GqlLine, GqlPoints, MeshPhongMaterial, Material, PointsMaterial, \
    LineBasicMaterial
from mmcore.base.utils import getitemattr
from mmcore.geom.materials import MeshPhysicalMetallic

MODE = {"children": "parents"}
from mmcore.geom.materials import ColorRGB
from mmcore.base.registry import geomdict, matdict
from mmcore.collections.multi_description import ElementSequence


@runtime_checkable
class BufferGeometryBuilder(typing.Protocol):
    choices = {
        '000': (mmcore.base.gql_models.Data1, mmcore.base.gql_models.Attributes1),
        '100': (mmcore.base.gql_models.Data1, mmcore.base.gql_models.Attributes3),
        '110': (mmcore.base.gql_models.Data1, mmcore.base.gql_models.Attributes2),
        '111': (mmcore.base.gql_models.Data, mmcore.base.gql_models.Attributes2)
    }

    def create_buffer(self, uuid: typing.Optional[str] = None) -> mmcore.base.models.gql.BufferGeometry: ...


def _buff_attr_checker(builder, name, attributes, cls):
    value = getattr(builder, name)
    if value is not None:
        attributes["normal"] = cls(**{
            "array": np.asarray(value, dtype=float).flatten().tolist(),
            "type": "Float32Array",
            "normalized": False
        })


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
        if uuid is None:
            uuid = self.uuid

        # print(selector)
        return mmcore.base.gql_models.BufferGeometry(**{
            "uuid": uuid,
            "type": "BufferGeometry",
            "data": _data(**data)
        })


T = typing.TypeVar("T")


class GeometryObject(Object3D):
    """
    GeometryObject.
    @note It should be used to implement three-dimensional representations of objects.  Note that geometry
    has no children's field. The Three JS Api does not explicitly state that they exist, however, in most cases,
    attempting to nest objects of geometric types will cause the nested object to lose its display. In short,
    we do not recommend doing so.
    """
    bind_class = GqlGeometry
    material_type = Material
    _material: typing.Optional[str] = None
    _geometry: typing.Optional[str] = None
    color: typing.Optional[ColorRGB] = None
    _children = ()

    def __call__(self, *args, geometry=None, material=None, **kwargs):
        super().__call__(self, *args, **kwargs)
        if self.uuid is None:
            self.uuid = _uuid.uuid4().__str__()
        if (material is None) and (self.color is not None):
            self.material = self.material_type(color=self.color.decimal)
        elif material is not None:
            self.material = material
        if geometry is not None:
            self.geometry = geometry

        return self.get_child_three()

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
        self.solve_geometry()
        return geomdict[self._geometry]

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
        if isinstance(v, str):
            self._geometry = v
        elif isinstance(v, dict):

            self._geometry = getitemattr("uuid")(v)
            geomdict[self._geometry] = create_buffer_from_dict(v)

        else:

            self._geometry = v.uuid
            print(f"Geometry set event: {self.name} <- {self._geometry}")
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
        return matdict[self._material]

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

        if isinstance(v, str):
            material = matdict[v]
        elif isinstance(v, dict):
            if 'metadata' in v:
                v.pop('metadata')
            material = self.material_type(**v)
        else:

            material = v
        if material.color in es["color"]:
            self._material = list(matdict.keys())[es["color"].index(material.color)]
        else:

            matdict[material.uuid] = material
            self._material = material.uuid

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


class PointObject(GeometryObject):
    _xyz: list[float]
    __match_args__ = "xyz",

    @property
    def threejs_type(self):
        return "Points"

    @property
    def xyz(self):
        return self._xyz

    @xyz.setter
    def xyz(self, v):
        self._xyz = v

    @property
    def properties(self):
        dct = super().properties
        dct |= {
            "xyz": self.xyz

        }
        return dct


class Mesh(GeometryObject):
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


class Brep(Mesh):
    material_type = MeshPhongMaterial

    @classmethod
    def from_rhino(cls, name, brep, uuid=None):
        if uuid is None:
            uuid = muuid.uuid4().__str__()
        inst = cls(name=name)
        inst._uuid = uuid
        inst.geometry = RhinoBrepBufferGeometry(brep).create_buffer()
        inst._rhino = brep
        return inst


class Line(GeometryObject):
    castShadow: bool = True
    receiveShadow: bool = True
    material_type = LineBasicMaterial

    @property
    def threejs_type(self):
        return "Line"


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
        if uuid is None:
            uuid = _uuid.uuid4().__str__()

        # print(selector)
        return mmcore.base.gql_models.BufferGeometry(**{
            "uuid": uuid,
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


class RhinoBrepBufferGeometry(RhinoMeshBufferGeometryBuilder):

    def __init__(self, brep, uuid: str):
        try:
            import Rhino.Geometry as rg


        except ImportError:

            import rhino3dm as rg

        mesh = rg.Mesh()
        [mesh.Append(l) for l in list(rg.Mesh.CreateFromBrep(brep, rg.MeshingParameters.FastRenderMesh))]
        super().__init__(mesh, uuid=uuid)


from mmcore.node import node_eval


class PointsObject(GeometryObject):
    _material: typing.Optional[str] = None
    _geometry: typing.Optional[str] = None
    __match_args__ = "points",
    material_type = PointsMaterial
    bind_class = GqlPoints
    name: str = "PointsObject"
    _color = ColorRGB(0.5, 0.5, 0.5)
    _points = None

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, v):
        self._color = v

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
        self._geometry = self.uuid + "-geom"
        geomdict[self._geometry] = mmcore.base.models.gql.BufferGeometry(**{
            'uuid': self._geometry,
            'type': 'BufferGeometry',
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
        dct=super(GeometryObject, self).properties

        dct.update(dict(points=self._points, color=self._color))
        return dct
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


class LineObject(PointsObject):
    __match_args__ = "name", "color"

    width: int = 2
    bind_class = GqlLine
    material_type = LineBasicMaterial
    _material: typing.Optional[str] = None
    _geometry: typing.Optional[str] = None

    @property
    def threejs_type(self):
        return "Line"


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
