import uuid
from enum import Enum
from typing import Any

import pydantic

from mmcore.baseitems import Matchable
from mmcore.baseitems.descriptors import UserData
from mmcore.gql.client import geometry_query

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
    query = geometry_query()

    def __init__(self, **data):
        if not data.get("uuid"):
            data['uuid'] = uuid.uuid4()
        super().__init__(**data)

    @classmethod
    def from_query(cls):
        inst = cls(**BufferGeometryDictionary(
            **cls.query("position", "normal", "uv", "type", "uuid", "index").run_query().json()))

        return inst


class Root:
    geometries: list = []
    materials: list = []

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


class ThreeJSObject(UserData):
    ...

    def item_model(self, name: str, value: Any):

        if name == "userdata":
            return {"userData": value}
        elif value is None:
            pass
        else:
            return {name: value}


class _BufferObject(Matchable):
    type: str | ThreeJSTypes = ThreeJSTypes.Mesh
    castShadow: bool = True
    receiveShadow: bool = True
    layers: int = 1
    matrix: list[float | int] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    name = None

    # object = ThreeJSObject() #"name", 'castShadow', 'layers', 'receiveShadow', 'type', "userdata", "matrix"

    geometry: str = None
    material: str = None
    object = UserData(
        "name",
        'castShadow',
        'layers',
        'receiveShadow',
        'type', "userdata", "matrix", "geometry", "material")


def __init__(self, *args, **data):
    super().__init__(self, *args, **data)

    if not data.get("uuid"):
        data['uuid'] = uuid.uuid4()


class BufferObject(_BufferObject):
    """
    Example:
        >>> class B(BufferObject):
        ...     __match_args__="name", "area", "subtype", "tag"
        ...     userdata = UserData(*__match_args__)
        >>> c=B(1,2,34,5)
        >>> c.object
    {'name': 1,
     'castShadow': True,
     'layers': 1,
     'receiveShadow': True,
     'type': 'Mesh',
     'userdata': {'name': 1, 'area': 2, 'subtype': 34, 'tag': 5},
     'matrix': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}
        >>> c.geometry = "999"
        >>> c.object
    {'name': 1,
     'castShadow': True,
     'layers': 1,
     'receiveShadow': True,
     'type': 'Mesh',
     'userdata': {'name': 1, 'area': 2, 'subtype': 34, 'tag': 5},
     'matrix': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
     'geometry': '999'}


    """
    ...


"""

# TODO: Понять что не так с ```name```
class BufferGroup(ElementSequence, BufferObject):
    userData: Any = GroupUserData()
    children: list[BufferObject] | list = []
    type: ThreeJSTypes = ThreeJSTypes.Group

    def __init__(self, children, *args, **kwargs):
        super(ElementSequence, self).__init__(children)
        BufferObject.__init__(self, *args, **kwargs)


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
        return obj_notation_from_mesh(self.name, self.geometry, self.material, userdata=self.userdata,
                                      matrix=self.matrix, uid=self.uuid)

    @property
    def children(self):
        return []
      @material.setter
    def material(self, v):

        if isinstance(v, str):
            if v in ElementSequence(self.root.materials)["uuid"]:
                self.material = v
            else:
                raise KeyError(v)
        elif isinstance(v, MeshPhongMaterial):
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


