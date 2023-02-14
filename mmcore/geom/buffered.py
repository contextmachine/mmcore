import uuid
from enum import Enum
from typing import Any, Optional

import pydantic

from mmcore.addons.rhino import obj_notation_from_mesh
from mmcore.baseitems import Matchable
from mmcore.baseitems.descriptors import UserData
from mmcore.collection.multi_description import ElementSequence
from mmcore.geom.base import MmGeometry
from mmcore.gql.client import GQLQuery

Number = int | float


class ThreeJSTypes(str, Enum):
    Object3d = "Object3d"
    Mesh = "Mesh"
    Group = "Group"
    Bone = "Bone"
    Line = "Line"
    Points = "Points"
    Sprite = "Sprite"
    BufferGeometry = "BufferGeometry"
    CapsuleGeometry = "CapsuleGeometry"


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


from mmcore.geom.materials import Material


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


class GeometryPrimitive(pydantic.BaseModel):
    type: str
    uuid: str = uuid.uuid4().__str__()


class CapsuleGeometry(pydantic.BaseModel):
    type: ThreeJSTypes = ThreeJSTypes.CapsuleGeometry
    radius: Number = 1
    height: Number = 1
    capSegments: int = 4
    radialSegments: int = 8


class BufferGeometryDictionary(dict):
    def __init__(self, req):
        dct = {
            "data": {
                "attributes": dict((a, req[a]) for a in ["position", "normal", "uv"]),
                "index": req["index"]
            },
            "type": req["type"],
            "uuid": req["uuid"]
        }

        dict.__init__(self, **dct)


class BufferGeometry(pydantic.BaseModel):
    data: dict
    type: ThreeJSTypes = ThreeJSTypes.BufferGeometry
    uuid: pydantic.UUID4

    def __init__(self, **data):
        if not data.get("uuid"):
            data['uuid'] = uuid.uuid4()
        super().__init__(**data)

    @classmethod
    def from_request(cls):
        client = GQLQuery()
        inst = BufferGeometryDictionary(**client.run_query())
        inst.gql_client = client
        return inst


class Root:
    geometries = []
    materials = []

    def __init__(self, obj=None):
        super().__init__()
        self.obj = obj

    def append_geometries(self, val):
        self.geometries.append(val)

    def remove_geometries(self, val):
        self.geometries.remove(val)

    def append_materials(self, val):
        self.materials.append(val)

    def remove_materials(self, val):
        self.materials.remove(val)

    def get_root_dict(self):
        return create_root_descriptor(self)

    def __get__(self, key, value):
        return self

    def __set__(self, instance, value):
        self.obj = value


class BufferObject(Matchable):
    type: ThreeJSTypes = ThreeJSTypes.Mesh
    uuid: pydantic.UUID4
    userData: dict = UserData()
    castShadow: bool = True
    receiveShadow: bool = True
    layers: int = 1
    matrix: list[float | int] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    _name: Optional[str] = None
    _geometry: Optional[pydantic.UUID4] = None
    _material: Optional[pydantic.UUID4] = None
    root = Root()

    def __init__(self, *args, **data):
        super().__init__(self, *args, **data)
        if not data.get("uuid"):
            data['uuid'] = uuid.uuid4()

    @property
    def object(self):
        return obj_notation_from_mesh(self.name, self.geometry, self.material, userdata=self.userData,
                                      matrix=self.matrix, uid=self.uuid)

    @property
    def name(self):
        if self._name is None:
            return f"Group {self.uuid}"
        else:
            return self.name

    @name.setter
    def name(self, v):
        self._name = v

    @property
    def material(self):
        return self._material

    @property
    def geometry(self):
        return self._geometry

    @material.setter
    def material(self, v):

        if isinstance(v, str):
            if v in ElementSequence(self.root.materials)["uuid"]:
                self.material = v
            else:
                raise KeyError(v)
        elif isinstance(v, Material):
            self._material = v.uuid
            if not self.root.materials:
                self.root.materials.append(v)

            elif v.uuid in ElementSequence(self.root.materials)["uuid"]:
                self.root.materials[(ElementSequence(self.root.materials)["uuid"]).index(v.uuid)] = v

            else:
                self.root.append_materials(v)
        elif isinstance(v, int):
            self._material = self.root.materials[v]['uuid']

    @geometry.setter
    def geometry(self, v):
        if isinstance(v, str):
            if v in ElementSequence(self.root.geometries)["uuid"]:
                self._geometry = v
            else:
                raise KeyError(v)
        elif isinstance(v, MmGeometry):
            self._geometry = v.uuid
            if not self.root.geometries:
                self.root.geometries.append(v)

            elif v.uuid in ElementSequence(self.root.geometries)["uuid"]:
                self.root.geometries[(ElementSequence(self.root.geometries)["uuid"]).index(v.uuid)] = v

            else:
                self.root.geometries.append(v)
        elif isinstance(v, int):
            self._geometry = self.root.geometries[v]['uuid']

    def to_dict(self):
        return Matchable.to_dict(self)

    def json(self):
        return self.toJSON()

    def dict(self):
        return self.to_dict()


from mmcore.baseitems.descriptors import GroupUserData


# TODO: Понять что не так с ```name```
class BufferGroup(ElementSequence, BufferObject):
    userData: Any = GroupUserData()
    children: list[BufferObject] | list = []
    type: ThreeJSTypes = ThreeJSTypes.Group

    def __init__(self, children, *args, **kwargs):
        super(ElementSequence, self).__init__(children)
        BufferObject.__init__(self, *args, **kwargs)


"""
class Group(ElementSequence, Matchable):
    _matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
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

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @property
    def object(self):
        return

    @property
    def children(self):
        return []

"""


def create_root(self):
    return {
        "metadata": {
            "version": 4.5,
            "type": "Object",
            "generator": "Object3D.toJSON"
        },
        "geometries": [],
        "materials": [],
        "object": self.to_dict()
    }


def create_root_descriptor(self):
    return {
        "metadata": {
            "version": 4.5,
            "type": "Object",
            "generator": "Object3D.toJSON"
        },
        "geometries": self.geometries,
        "materials": self.materials,
        "object": self.obj.to_dict()
    }


from mmcore.utils import redis_tools


class Scene(redis_tools.RC):
    ...


def assign_root(root, obj):
    obj.root = root


class BufferObjectRoot(BufferGroup):
    _geometries = []
    _materials = []
    __match_args__ = 'children',

    def __init__(self, children=(), **kwargs):
        super().__init__(**kwargs)

        super(ElementSequence, self).__init__(children)
        self.traverse_children(callback=assign_root)

    root = None

    def traverse_children(self, callback=lambda x: x):
        def trav(data):
            if data.get("children"):
                return trav(data["children"])
            else:
                return callback(data)

        return trav
