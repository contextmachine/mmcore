import json
import typing
from typing import Any

import pydantic
from mmcore.addons import ModuleResolver
with ModuleResolver() as rsl:
    import rhino3dm
import rhino3dm

from mmcore.addons import rhino
from pydantic import ConstrainedStr


class PropertyBaseModel(pydantic.BaseModel):
    """
    Workaround for serializing properties with pydantic until
    https://github.com/samuelcolvin/pydantic/issues/935
    is solved
    """

    def __repr_args__(self) -> 'ReprArgs':
        """
        Returns the attributes to show in __str__, __repr__, and __pretty__ this is generally overridden.

        Can either return:
        * name - value pairs, e.g.: `[('foo_name', 'foo'), ('bar_name', ['b', 'a', 'r'])]`
        * or, just values, e.g.: `[(None, 'foo'), (None, ['b', 'a', 'r'])]`
        """
        attrs = ((s, getattr(self, s)) for s in self.dict().keys())
        return [(a, v) for a, v in attrs if v is not None]

    @classmethod
    def get_properties(cls):
        return [prop for prop in dir(cls) if
                isinstance(getattr(cls, prop), property) and prop not in ("__values__", "fields")]

    def dict(
        self,
        *,
        include: typing.Any = None,
        exclude: typing.Union['AbstractSetIntStr', 'MappingIntStrAny'] = None,
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        ) -> dict[str, Any]:
        attribs = super().dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none
            )
        props = self.get_properties()
        # Include and exclude properties
        if include:
            props = [prop for prop in props if prop in include]
        if exclude:
            props = [prop for prop in props if prop not in exclude]

        # Update the attribute dict with the properties
        if props:
            attribs.update({prop: getattr(self, prop) for prop in props})

        return attribs

    def json(self, *args, **kwargs):

        return \
            json.dumps(self.dict(), *args, **kwargs)

class SnakeCaseName(ConstrainedStr):
    strip_whitespace = True
    to_upper = False
    to_lower = True
    min_length: int | None = None
    max_length: int | None = None
    curtail_length: int | None = None
    regex: int | None = None
    strict = False


class RhinoVersion(int):
    def __new__(cls, v):
        return 70


class Archive3dm(pydantic.BaseModel):
    opennurbs: int
    version: int
    archive3dm: RhinoVersion
    data: str

    @property
    def api_type(self):
        return "Rhino.Geometry.GeometryBase"

    @classmethod
    def from_3dm(cls, data3dm) -> 'Archive3dm':
        return cls(**rhino.RhinoEncoder().default(data3dm))

    def to_3dm(self) -> dict:
        return rhino.RhinoDecoder().decode(self.data)

    def __repr__(self):
        ss = super().__repr__().split("data")
        return ss[0] + "data: ... ')"

    def __str__(self):
        ss = super().__str__().split("data")
        return ss[0] + "data: ... ')"


class InnerTreeItem(pydantic.BaseModel):
    type: str | bytes
    data: str | bytes


class NormInnerItem(pydantic.BaseModel):
    type: str
    data: dict[str, Any]

    def transform_json(self):
        dct = self.dict()
        return {
            "type": self.type,
            "data": json.dumps(dct["data"])
        }


class ComputeJson(NormInnerItem):
    type: str
    data: list[Archive3dm]

    def transform_json(self):
        return super().transform_json()


class DataTreeParam(pydantic.BaseModel):
    ParamName: str
    InnerTree: dict[str, list[InnerTreeItem]]


class ComputeRequest(pydantic.BaseModel):
    pointer: str
    values: list[DataTreeParam]


class ComputeResponse(pydantic.BaseModel):
    data: dict | list
