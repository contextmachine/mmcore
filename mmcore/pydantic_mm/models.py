from typing import Any

import pydantic
from pydantic import ConstrainedStr
import rhino3dm
from ..conversions import rhino


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

    @classmethod
    def from_3dm(cls, data3dm) -> 'Archive3dm':
        return cls(**rhino.RhinoEncoder().default(data3dm))

    def to_3dm(self) -> dict | rhino3dm.CommonObject | Any:
        return rhino.RhinoDecoder().decode(self.data)

    def __repr__(self):
        ss = super().__repr__().split("data")
        return ss[0] + "data: ... ')"

    def __str__(self):
        ss = super().__str__().split("data")
        return ss[0] + "data: ... ')"


class InnerTreeItem(pydantic.BaseModel):
    type: str
    data: str


class DataTreeParam(pydantic.BaseModel):
    ParamName: str
    InnerTree: dict[str, list[InnerTreeItem]]


class ComputeRequest(pydantic.BaseModel):
    pointer: str
    values: list[DataTreeParam]