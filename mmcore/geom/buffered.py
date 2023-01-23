import uuid
import weakref
from typing import Any, Optional

import numpy as np
import pydantic

from mmcore.addons.rhino import obj_notation_from_mesh
from mmcore.baseitems import Matchable
from mmcore.collection.multi_description import ElementSequence
from mmcore.geom.base import MmGeometry
from mmcore.geom.materials import Material

Number = int | float


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


class BufferObjectRoot:
    _geometries = []
    _materials = []
    __match_args__ = '_obj',

    def __init__(self, obj):
        super().__init__()
        self._obj = obj
        self._obj._root = weakref.proxy(self)

    @property
    def geometries(self):
        return self._geometries

    @geometries.setter
    def geometries(self, v):
        self._geometries = v

    @property
    def materials(self):
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
            "object": self._obj.object
            }

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)


class BufferObject(Matchable):
    __match_args__ = "name",
    userData = {}
    _area = 0
    _matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    _material = None
    _geometry = None
    _root = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
