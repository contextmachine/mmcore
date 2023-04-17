import dataclasses
import json
import typing
import uuid
import uuid as muuid
from abc import ABCMeta

import numpy as np
import scipy.spatial.distance as spdist
from OCC.Core.Tesselator import ShapeTesselator

import mmcore.base.models.gql
from mmcore.base.basic import geomdict, matdict, Group
from mmcore.base.geom.utils import create_buffer_from_dict, parse_attribute
from mmcore.base.models import Point
from mmcore.base.models.gql import GqlGeometry, GqlLine, GqlPoints
from mmcore.base.utils import getitemattr
from mmcore.geom.materials import MeshPhysicalMetallic

MODE = {"children": "parents"}
from mmcore.geom.materials import ColorRGB

from mmcore.collections.multi_description import ElementSequence



class MeshBufferGeometry:
    @classmethod
    def from_three(cls, dct: dict) -> 'MeshBufferGeometry':
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

    def create_buffer(self) -> mmcore.base.models.gql.BufferGeometry:
        return mmcore.base.models.gql.BufferGeometry(**{
            "uuid": self.uuid,
            "type": "BufferGeometry",
            "data": mmcore.base.models.gql.Data(**{
                "attributes": mmcore.base.models.gql.Attributes(**{

                    "normal": mmcore.base.models.gql.Normal(**{
                        "array": np.asarray(self.normals, dtype=float).flatten().tolist(),
                        "itemSize": 3,
                        "type": "Float32Array",
                        "normalized": False
                    }),
                    "position": mmcore.base.models.gql.Position(**{
                        "array": np.asarray(self.vertices, dtype=float).flatten().tolist(),
                        "itemSize": 3,
                        "type": "Float32Array",
                        "normalized": False
                    }),
                    "uv": mmcore.base.models.gql.Uv(**{
                        'itemSize': 2,
                        "array": np.asarray(self.uv, dtype=float).flatten().tolist(),
                        "type": "Float32Array",
                        "normalized": False

                    }),

                }),
                "index": mmcore.base.models.gql.Index(**dict(type='Uint16Array',
                                                             array=np.asarray(self.indices, dtype=int).flatten().tolist()))
            })
        })


T = typing.TypeVar("T")


class GeometryObject(Group):
    """
    GeometryObject.
    @note It should be used to implement three-dimensional representations of objects.  Note that geometry
    has no children's field. The Three JS Api does not explicitly state that they exist, however, in most cases,
    attempting to nest objects of geometric types will cause the nested object to lose its display. In short,
    we do not recommend doing so.
    """
    bind_class = GqlGeometry
    material_class = MeshPhysicalMetallic
    _material: str | None = None
    _geometry: str | None = None

    def commit(self):
        ...

    def solve(self):
        return None

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

        if isinstance(v, str):
            self._material = v
        elif isinstance(v, dict):

            self._material = getitemattr("uuid")(v)

            if 'metadata' in v:
                v.pop('metadata')

            matdict[self._material] = v

        else:

            self._material = v.uuid
            matdict[v.uuid] = v

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
        inst.geometry = MeshBufferGeometry.from_three(geom)
        inst.material = cls.material_class(**material)
        return inst


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
    geometry_type = MeshBufferGeometry
    castShadow: bool = True
    receiveShadow: bool = True

    @property
    def threejs_type(self):
        return "Mesh"

    @classmethod
    def from_rhino(cls, name, mesh, uuid=None):
        if uuid is None:
            uuid = muuid.uuid4().__str__()
        inst = cls(name=name)
        inst._uuid = uuid
        inst.geometry = RhinoMeshBufferGeometry(mesh).create_buffer()
        inst._rhino = mesh
        return inst

    @classmethod
    def from_rhino(cls, name, mesh, uuid=None):
        if uuid is None:
            uuid = muuid.uuid4()
        inst = cls(name=name)
        inst._uuid = uuid
        inst.geometry = RhinoMeshBufferGeometry(mesh).create_buffer()
        inst._rhino = mesh
        return inst

def to_camel_case(name:str):
    """
    Ключевая особенность, при преобразовании имени начинающиегося с подчеркивания, подчеркивание будет сохранено.

        foo_bar -> FooBar
        _foo_bar -> _FooBar
    @param name: str
    @return: str
    """
    if not name.startswith("_"):
        return "".join(nm.capitalize() for nm in name.split("_"))

    else:
        return "_"+"".join(nm.capitalize() for nm in name.split("_"))

class Brep(Mesh):

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

    @property
    def threejs_type(self):
        return "Line"

from mmcore.base.models import gql
class MeshRh(Mesh):
    def __call__(self, rhino_mesh, *args, **kwargs):
        self._rhino_mesh = rhino_mesh
        super().__call__(*args, **kwargs)

    def solve(self):
        return RhinoMeshBufferGeometry(self._rhino_mesh).create_buffer()

def generate_edges_material(uid, color, linewidth):
    return gql.LineBasicMaterial(**{"uuid": uid, "type": "LineBasicMaterial", "color": color.decimal, "vertexColors": True, "depthFunc": 3,
            "depthTest": True, "depthWrite": True, "colorWrite": True, "stencilWrite": False, "stencilWriteMask": 255,
            "stencilFunc": 519, "stencilRef": 0, "stencilFuncMask": 255, "stencilFail": 7680, "stencilZFail": 7680,
            "stencilZPass": 7680, "linewidth": linewidth, "toneMapped": False})




def export_edgedata_to_json(edge_hash, point_set):
    """Export a set of points to a LineSegment buffergeometry"""
    # first build the array of point coordinates
    # edges are built as follows:
    # points_coordinates  =[P0x, P0y, P0z, P1x, P1y, P1z, P2x, P2y, etc.]

    points_coordinates = []
    for point in point_set:
        for coord in point:
            points_coordinates.append(coord)
    # then build the dictionary exported to json
    edges_data = {
        "uuid": edge_hash,
        "type": "BufferGeometry",
        "data": {
            "attributes": {
                "position": {
                    "itemSize": 3,
                    "type": "Float32Array",
                    "array": points_coordinates,
                }
            }
        },
    }
    return edges_data

def generate_material(self):
    vv = list(matdict.values())
    if len(vv) > 0:
        print(vv)
        es = ElementSequence(vv)
        print(self.color, es["color"])
        if self.color.decimal in es["color"]:
            i = es["color"].index(self.color.decimal)
            print(i)
            vvv = es._seq[i]
            print(vvv)
            self.mesh._material = vvv.uuid
        else:
            self.mesh.material = mmcore.base.models.gql.MeshPhongMaterial(name=f"{'MeshPhongMaterial'} {self._name}",
                                                                          color=self.color.decimal)

    else:
        self.mesh.material = mmcore.base.models.gql.MeshPhongMaterial(name=f"{'MeshPhongMaterial'} {self._name}",
                                                                      color=self.color.decimal)

class Tessellate(metaclass=ABCMeta):
    def __init__(self, shape, name, color):
        super().__init__()
        self.mesh = Mesh(name=name)
        self.tess = ShapeTesselator(shape)
        self._name = name
        self.color = color
        self.generate_material()

    def tessellate(self, compute_edges=False, mesh_quality=1.0, parallel=True):

        self.tess.Compute(compute_edges=compute_edges,
                          mesh_quality=mesh_quality,
                          parallel=parallel)

        _uuid = uuid.uuid4().__str__()

        self.mesh.geometry = create_buffer_from_dict(

            json.loads(self.tess.ExportShapeToThreejsJSONString(_uuid)))

        if compute_edges:

            # export each edge to a single json
            # get number of edges
            nbr_edges = self.tess.ObjGetEdgeCount()

            grp= Group(name="edges")
            self.mesh.edges = grp
            for i_edge in range(nbr_edges):

                # after that, the file can be appended
                str_to_write = ""
                edge_point_set = []
                nbr_vertices = self.tess.ObjEdgeGetVertexCount(i_edge)
                for i_vert in range(nbr_vertices):
                    edge_point_set.append(self.tess.GetEdgeVertex(i_edge, i_vert))
                # write to file
                ln = Line(name=f"edge-{i_edge}")
                ln.geometry = export_edgedata_to_json(uuid.uuid4().__str__(), edge_point_set)
                ln.material = generate_edges_material(uuid.uuid4().__str__(), color=ColorRGB(0, 0, 0), linewidth=1.0)
                grp.add(ln)

        return self.mesh

    def generate_material(self):
        vv = list(matdict.values())
        if len(vv) > 0:
            print(vv)
            es = ElementSequence(vv)
            print(self.color, es["color"])
            if self.color.decimal in es["color"]:
                i = es["color"].index(self.color.decimal)
                print(i)
                vvv = es._seq[i]
                print(vvv)
                self.mesh._material = vvv.uuid

            else:
                self.mesh.material = mmcore.base.models.gql.MeshPhongMaterial(name=f"{'MeshPhongMaterial'} {self._name}",
                                                                              color=self.color.decimal)

        else:
            self.mesh.material = mmcore.base.models.gql.MeshPhongMaterial(name=f"{'MeshPhongMaterial'} {self._name}",
                                                                          color=self.color.decimal)


class TessellateIfc(Tessellate):
    def __init__(self, shape):

        self._shape = shape
        super().__init__(shape.geometry, color=ColorRGB(*shape.styles[0][:-1]), name=shape.data.name)

    def tessellate(self, compute_edges=False, mesh_quality=1.0, parallel=True):
        return super().tessellate( compute_edges=False, mesh_quality=1.0, parallel=True)
class RhinoMeshBufferGeometry(MeshBufferGeometry):
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


class RhinoBrepBufferGeometry(RhinoMeshBufferGeometry):

    def __init__(self, brep, uuid: str):
        try:
            import rhino3dm as rg
        except ImportError:
            from mmcore.addons import ModuleResolver
            with ModuleResolver():
                import Rhino.Geometry as rg
            import Rhino.Geometry as rg
        mesh = rg.Mesh()
        [mesh.Append(l) for l in list(rg.Mesh.CreateFromBrep(brep, rg.MeshingParameters.FastRenderMesh))]
        super().__init__(mesh, uuid)


from mmcore.node import node_eval


class PointsObject(GeometryObject):
    __match_args__ = "points",
    material_class = mmcore.base.models.gql.PointsMaterial
    bind_class = GqlPoints
    points: list[Point]
    _points = None
    name: str = "PointsObject"
    _color = ColorRGB(0.5, 0.5, 0.5)

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        dct = self.solve()
        dct.pop("metadata")
        self.geometry = create_buffer_from_dict(dct)
        dct = pointsMaterial(self.color)
        dct.pop("metadata")
        es = ElementSequence(list(matdict.values()))
        if dct["color"] in es["color"]:
            self._material = list(matdict.keys())[es["color"].index(dct["color"])]
        else:
            self.material = mmcore.base.models.gql.PointsMaterial(**dct)

        return self

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

    @property
    def threejs_type(self):
        return "Points"

    @node_eval
    def solve(self):

        return mmcore.base.models.gql.BufferGeometry({'metadata': mmcore.base.models.gql.Metadata(),
                                      'uuid': muuid.uuid4().__str__(),
                                      'type': 'BufferGeometry',
                                      'data': mmcore.base.models.gql.Data1(**{'attributes': mmcore.base.models.gql.Attributes1(
                                          **{'position': mmcore.base.models.gql.Position(**{'itemSize': 3,
                                                                            'type': 'Float32Array',
                                                                            'array': np.asarray(
                                                                                self.points).flatten().tolist(),
                                                                            'normalized': False})})})})


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
def lineMaterial(color, width):
    line = f"makePointMaterial({color.decimal}, {width})"
    # language=JavaScript
    return '''const THREE = require("three");
                      function makePointMaterial( color, width) {
                            const mat = new THREE.LineBasicMaterial({color: color, linewidth:width})
                            console.log(JSON.stringify(mat.toJSON()));
                      }; ''' + line


class LineObject(GeometryObject):
    __match_args__ = "name", "color"
    points: list[dict[str, float]] = []
    color: ColorRGB = ColorRGB(0.7, 0.7, 0.7)
    width: int = 2
    bind_class = GqlLine

    def __call__(self, **kwargs):

        super().__call__(**kwargs)

        dct = self.solve()
        dct.pop("metadata")
        self.geometry = create_buffer_from_dict(dct)
        dct = lineMaterial(self.color, self.width)
        dct.pop("metadata")
        es = ElementSequence(list(matdict.values()))

        if dct["color"] in es["color"]:
            self._material = list(matdict.keys())[es["color"].index(dct["color"])]
        else:
            self.material = mmcore.base.models.gql.LineBasicMaterial(**dct)

        return self

    @property
    def threejs_type(self):
        return "Line"

    def solve(self):
        return {'metadata': {'version': 4.5,
                             'type': 'BufferGeometry',
                             'generator': 'BufferGeometry.toJSON'},
                'uuid': 'e3508e43-0ee5-4659-a4f0-0548d8f931fd',
                'type': 'BufferGeometry',
                'data': {'attributes': {'position': {'itemSize': 3,
                                                     'type': 'Float32Array',
                                                     'array': np.asarray(self.points).flatten().tolist(),
                                                     'normalized': False}}}}


