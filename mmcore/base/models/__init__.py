from __future__ import annotations

import dataclasses
import uuid



from mmcore.base.models.gql import BaseMaterial


@dataclasses.dataclass
class Point:
    x: float
    y: float
    z: float

    @classmethod
    def from_array(cls, arr):
        return cls(*arr)

    @property
    def xyz(self):
        return [self.x, self.y, self.z]




@dataclasses.dataclass
class NamedPoint(Point):
    name: str = "Foo"
    uuid: str = ""

    def __post_init__(self):
        self.uuid = uuid.uuid4().__str__()


