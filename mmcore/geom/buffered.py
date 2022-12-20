import uuid
from typing import Any, Optional

import pydantic

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
    type: str= "CapsuleGeometry"
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

