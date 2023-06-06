
"""
import functools
import hashlib
import typing
import json, ujson
from pydantic import BaseModel, BaseConfig
from pydantic import config


class ThreeJsModel(BaseModel):

    def dict(self, *args, exclude_none=True, **kwargs):
        return super().dict(*args, exclude_none=exclude_none, **kwargs)

    def json(self, *args, exclude_none=True, **kwargs):
        return super().json(*args, exclude_none=exclude_none, **kwargs)


class BufferGeometryAttribute(ThreeJsModel):
    array: typing.Union[list[float], list[int]]
    type: str
    itemSize: typing.Optional[int]


class BufferGeometryAttributes(ThreeJsModel):
    position: BufferGeometryAttribute
    normal: typing.Optional[BufferGeometryAttribute]
    uv: typing.Optional[BufferGeometryAttribute]
    color: typing.Optional[BufferGeometryAttribute]


class BufferGeometryData(ThreeJsModel):
    attributes: BufferGeometryAttributes
    index: typing.Optional[BufferGeometryAttribute]


class BufferGeometry(ThreeJsModel):
    uuid: str
    data: BufferGeometryData
    type: str = 'BufferGeometry'

    class Config(BaseConfig):
        ...

    def __post_init__(self):
        self.uuid = hashlib.sha256(ujson.dumps(self.data.attributes.position.array)).hexdigest()
"""