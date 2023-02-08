import uuid
from typing import Any, Optional

import numpy as np
import pydantic
from mmcore.addons.rhino import obj_notation_from_mesh
from mmcore.baseitems import Matchable
from mmcore.collection.multi_description import ElementSequence
from mmcore.geom.base import MmGeometry
from mmcore.geom.materials import Material

Number = int | float


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


class BufferMaterial(pydantic.BaseModel):
    ...


class BufferBndSphere(pydantic.BaseModel):
    center: list[float]
    radius: float


class BufferAttribute(pydantic.BaseModel):
    itemSize: int = 3,
    type: str = "Float32Array",
    array: list[float]
    normalized: bool = False


class BufferMetaData(pydantic.BaseModel):
    type: str = "BufferGeometry"
    generator: str = "mmcore"
    version: str = '4.5'


class BufferAttributes(pydantic.BaseModel):
    position: BufferAttribute
    normal: BufferAttribute


class BufferData(pydantic.BaseModel):
    attributes: BufferAttributes
    boundingSphere: BufferBndSphere
    type: str = "BufferGeometry"


class GeometryPrimitive(pydantic.BaseModel):
    type: str
    uuid: str = uuid.uuid4().__str__()


class CapsuleGeometry(GeometryPrimitive):
    type: str = "CapsuleGeometry"
    radius: Number = 1
    height: Number = 1
    capSegments: int = 4
    radialSegments: int = 8


class BufferGeometry(pydantic.BaseModel):
    data: BufferData
    type: str = "BufferGeometry"


class BufferObjectField(pydantic.BaseModel):
    uuid: str = uuid.uuid4().__str__()
    type: str = "Mesh"
    name: Optional[str]
    castShadow: bool = True
    receiveShadow: bool = True
    layers: int = 1
    matrix: list[float | int] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    geometry: str
    material: str
    children: Optional[list['BufferObjectField']] = None


class BufferObject3d(pydantic.BaseModel):
    geometries: list[GeometryPrimitive]
    materials: list[BufferMaterial]
    object: BufferObjectField
    metadata: BufferMetaData = BufferMetaData(type="Object")
    userData: Any = {}


class BufferGroup(pydantic.BaseModel):
    uuid: str
    name: str
    type: str = "Group"
    castShadow: bool = True
    receiveShadow: bool = True
    layers: int = 1,
    matrix: list[float | int] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    children: list[BufferObjectField] | list = []
    userData: Any = {}


from mmcore.baseitems.descriptors import GroupUserData


class Group(ElementSequence, Matchable):
    _name = None
    _children = []
    __match_args__ = "_seq", "name", "matrix"
    userData = GroupUserData()

    def __repr__(self):
        return super(ElementSequence, self).__repr__()

    @property
    def name(self):
        if self._name is None:
            return f"Group {self.uuid}"
        else:
            return self.name

    _matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @property
    def object(self):
        return BufferGroup(**{
            'uuid': self.uuid,
            'type': 'Group',
            'name': self.name,
            'castShadow': True,
            'receiveShadow': True,
            'userData': self.userData,
            'layers': 1,
            'matrix': self.matrix,
            'children': self.children
            })

    @property
    def children(self):
        return


def create_root(self):
    return {
        "metadata": {
            "version": 4.5,
            "type": "Object",
            "generator": "Object3D.toJSON"
            },
        "geometries": [],
        "materials": [],
        "object": self.obj
        }


from mmcore.utils import redis_tools


class Scene(redis_tools.RC):
    ...


class BufferObjectRoot(ElementSequence, Matchable):
    _geometries = []
    _materials = []
    __match_args__ = 'children', 'name', 'matrix'

    def __init__(self, children=(), name="Group", matrix=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), **kwargs):
        for child in children:
            Matchable.__init__(self, name, matrix, **kwargs)
        super(ElementSequence, self).__init__(children)

    _name = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    _matrix = None

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @property
    def material_sequence(self):
        return ElementSequence(self.materials)

    @property
    def geometries(self):
        # [mesh_to_buffer(obj) for obj in self["mesh"]]
        return self._geometries

    @geometries.setter
    def geometries(self, v):
        self._geometries = v

    @property
    def materials(self):
        # self.materials = list(map(lambda x: MeshPhongFlatShading(color=ColorRGB(*eval(x)[:-1])), list(self.color_counter.keys())))

        return self._materials

    @materials.setter
    def materials(self, v):
        self._materials = v

    @property
    def root(self):
        return {
            "metadata": {
                "version": 4.5,
                "type": "Object",
                "generator": "Object3D.toJSON"
                },
            "geometries": self.geometries,
            "materials": self.materials,
            "object": self.object
            }

    def __call__(self, *args, **kwargs):
        return self.object(*args, **kwargs)

    @property
    def object(self):
        return group_notation_from_mesh(self.name, userdata=self.userData,
                                        matrix=self.matrix, uid=self.uuid)


class BufferObject(Matchable):
    __match_args__ = "root", "name"

    _area = 0
    _matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    _material = None
    _geometry = None
    _root = None

    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)

    @property
    def object(self):
        return obj_notation_from_mesh(self.name, self.geometry, self.material, userdata=self.userData,
                                      matrix=self.matrix, uid=self.uuid)

    @property
    def geometry(self):
        return self._geometry

    @property
    def material(self):
        return self._material

    @property
    def root(self):
        return self._root.root

    @root.setter
    def root(self, v):
        self._root = v

    @material.setter
    def material(self, v):
        if isinstance(v, str):
            if v in ElementSequence(self._root.materials)["uuid"]:
                self._material = v
            else:
                raise KeyError(v)
        elif isinstance(v, Material):
            self._material = v.uuid
            if not self._root.materials:
                self._root.materials.append(v)

            elif v.uuid in ElementSequence(self.root['materials'])["uuid"]:
                self._root.materials[(ElementSequence(self._root.materials)["uuid"]).index(v.uuid)] = v

            else:
                self.root['materials'].append(v)
        elif isinstance(v, int):
            self._material = self._root.materials[v]['uuid']

    @geometry.setter
    def geometry(self, v):
        if isinstance(v, str):
            if v in ElementSequence(self._root.geometries)["uuid"]:
                self._geometry = v
            else:
                raise KeyError(v)
        elif isinstance(v, MmGeometry):
            self._geometry = v.uuid
            if not self._root.geometries:
                self._root.geometries.append(v)

            elif v.uuid in ElementSequence(self._root.geometries)["uuid"]:
                self._root.geometries[(ElementSequence(self.root['geometries'])["uuid"]).index(v.uuid)] = v

            else:
                self._root.geometries.append(v)
        elif isinstance(v, int):
            self._geometry = self._root.geometries[v]['uuid']

    @property
    def matrix(self):
        return np.asarray(self._matrix).reshape((4, 4)).T.flatten().tolist()

    @matrix.setter
    def matrix(self, v):
        self._matrix = v
