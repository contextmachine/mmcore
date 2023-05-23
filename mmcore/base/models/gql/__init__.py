from __future__ import annotations

import copy
import dataclasses
import sys
import typing
import uuid
import hashlib

import numpy as np
import strawberry
from strawberry.scalars import JSON

from mmcore.base.utils import getitemattr
from mmcore.base.registry import objdict

hashlib.sha224()


class ChildrenDesc:

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __init__(self, default=None):
        self._default = default

    def __get__(self, instance, owner):
        if instance is None:
            return self._default
        else:
            try:
                l = []
                for uid in objdict[instance.uuid]._children:
                    l.append(objdict[uid].bind_class(**objdict[uid].threejs_repr))
                return l
            except KeyError:
                pass

    def __set__(self, instance, value):

        if isinstance(value, (set, list, tuple)):
            instance._children = set()
            for val in value:
                if isinstance(val, str):
                    instance._children.add(val)
                else:
                    instance._children.add(getitemattr("uuid")(val))
        else:

            pass


@strawberry.type
class Metadata:
    version: float = 4.5
    type: str = "Object"
    generator: str = "Object3D.toJSON"


@strawberry.type
class BufferAttribute:
    type: str
    array: list[float]


class ItemSize:
    def __init__(self, size=1):
        self._size = size

    def __set_name__(self, owner, name):
        owner.__item_size__ = lambda slf: self._size
        self._name = "_" + name

    def __get__(self, instance, owner):
        if instance is None:

            return self._size
        else:
            return getattr(instance, self._name, __default=self._size)


@strawberry.type
class Position(BufferAttribute):
    type: str = "Float32Array"
    array: list[float]
    itemSize: int = ItemSize(3)
    normalized: bool = False

    def item_size(self) -> int:
        return 3


@strawberry.type
class Color(Position):
    itemSize: int = ItemSize(3)


@strawberry.type
class Normal(Position):
    itemSize: int = ItemSize(3)


@strawberry.type
class Uv(Position):
    itemSize: int = ItemSize(2)


@strawberry.type
class Attributes1:
    position: Position


@strawberry.type
class Attributes2(Attributes1):
    position: Position
    normal: typing.Union[Normal, None] = None
    uv: typing.Union[Uv, None] = None


@strawberry.type
class Attributes(Attributes1):
    position: Position
    color: typing.Union[Color, None] = None
    normal: typing.Union[Normal, None] = None
    uv: typing.Union[Uv, None] = None


@strawberry.type
class Attributes3(Attributes1):
    position: Position
    normal: typing.Union[Normal, None] = None


@strawberry.type
class Attributes4(Attributes1):
    position: Position
    colors: typing.Union[Color, None] = None


@strawberry.type
class BoundingSphere:
    center: list[float]
    radius: float


@strawberry.type
class Index(BufferAttribute):
    type: str
    array: list[int]
    itemSize: strawberry.Private[int] = ItemSize(1)


@strawberry.type
class Data:
    attributes: typing.Union[Attributes, Attributes1, Attributes2, Attributes3]
    index: typing.Union[Index, None] = None


@strawberry.type
class Data1:
    attributes: typing.Union[Attributes, Attributes1, Attributes2, Attributes3]


@strawberry.type
class PositionOnlyData:
    attributes: Attributes1


@strawberry.type
class Data3:
    attributes: Attributes3


import ujson


class HashUUID:
    def __init__(self, default=None):
        self._default = default

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner) -> str:
        if self._default is None:
            if instance:
                self._default= instance.sha().hexdigest()

        return self._default

    def __set__(self, instance, value):
        return DeprecationWarning(f"UUID set of {instance} is deprecated and not supported!")


@dataclasses.dataclass
class BufferGeometryObject:
    data: typing.Union[Data, Data1]
    type: str = "BufferGeometry"
    uuid: typing.Optional[str ]= None
    def __post_init__(self):
        print("create")
        self.uuid = self.sha().hexdigest()


    def sha(self):
        return hashlib.sha1(ujson.dumps(self.data.attributes.position.array).encode())

    def __hash__(self):
        return int(self.uuid, 16)


@strawberry.type
class BufferGeometry(BufferGeometryObject):
    data: typing.Union[Data, Data1]
    type: str = "BufferGeometry"


@strawberry.type
class SphereBufferGeometry(BufferGeometry):
    radius: typing.Optional[float] = None
    detail: typing.Optional[int] = None


@strawberry.type
class PointsBufferGeometry(BufferGeometry):
    data: PositionOnlyData


@strawberry.type
class LineBufferGeometry(PointsBufferGeometry):
    data: PositionOnlyData


@strawberry.type
class LinkItem:
    name: str


@strawberry.type
class GqlUserData:
    properties: typing.Optional[JSON] = None
    gui: list[GqlChart] | None = None
    params: typing.Optional[JSON] = None


@strawberry.type
class LineBufferGeometry(BufferGeometry):
    type = "LineGeometry"


@strawberry.type
class GqlBaseObject:
    name: str
    uuid: str
    userData: GqlUserData
    matrix: list[float] = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    layers: int = 1
    type: str = "Object3D"
    castShadow: bool = True
    receiveShadow: bool = True


@strawberry.type
class Camera:
    uuid: str
    type: str
    layers: int
    fov: int
    zoom: int
    near: float
    far: int
    focus: int
    aspect: int
    filmGauge: int
    filmOffset: int


@strawberry.type
class Shadow:
    camera: Camera


@strawberry.type
class GqlGeometry(GqlBaseObject):
    geometry: typing.Union[str, None] = None
    material: typing.Union[str, None] = None
    children: typing.Optional[list[GQLObject3DUnion]] = ChildrenDesc()


@strawberry.type
class GqlObject3D(GqlBaseObject):
    name: str
    uuid: str
    userData: GqlUserData
    matrix: list[float] = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    layers: int = 1
    type: str = "Object3D"
    castShadow: bool = True
    receiveShadow: bool = True
    children: typing.Optional[list[GQLObject3DUnion]] = ChildrenDesc()


@strawberry.type
class GqlGroup(GqlObject3D):
    type: str = "Group"
    matrix: list[float] = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    children: list[GQLObject3DUnion] = ChildrenDesc()


@strawberry.type
class GqlMesh(GqlGeometry):
    type: str = "Mesh"
    children: typing.Optional[list[GQLObject3DUnion]] = ChildrenDesc(default=None)


@strawberry.type
class GqlLine(GqlGeometry):
    type: str = "Line"
    children: typing.Optional[list[GQLObject3DUnion]] = ChildrenDesc(default=None)


@strawberry.type
class GqlPoints(GqlGeometry):
    type: str = "Points"
    children: typing.Optional[list[GQLObject3DUnion]] = ChildrenDesc(default=None)


import zlib

GQLGeometryUnion = strawberry.union("GQLObject3DUnion", types=[GqlGeometry, GqlMesh, GqlLine, GqlPoints])
GQLObject3DUnion = strawberry.union("GQLObject3DUnion",
                                    types=[GqlObject3D, GqlGroup, GqlGeometry, GqlMesh, GqlLine, GqlPoints])


class UidSha256:
    def __init__(self, default='', concrete_override: str = None):
        self.concrete_override = concrete_override
        self._default = default

    def __get__(self, inst, own):
        if inst:
            if self.concrete_override is not None:
                return self.concrete_override

            else:
                return hashlib.sha256(self.to_bytes(inst)).hexdigest()
        else:
            return self._default

    def __set_name__(self, owner, name):
        self.name = name

    def to_bytes(self, inst):

        d = dict(inst.__dict__)
        if 'uuid' in d.keys():
            del d['uuid']
        if self.name in d.keys():
            del d[self.name]
        s = []
        for k, v in d.items():
            if hasattr(v, self.name):

                s.append(f'{k}:{getattr(v, self.name)}')
            elif v.__class__.__name__ in ['str', 'int', 'bytes', 'float', 'bool']:
                s.append(f'{k}:{v}')
            elif isinstance(v, str):
                s.append(f'{k}:{v}')
            elif isinstance(v, typing.Iterable):
                s.append(f'{k}:{",".join([self.to_bytes(vv) for vv in v])}')
            elif isinstance(v, dict):
                s.append(",".join([f'{k}={self.to_bytes(v)}' for k, v in v.items()]).encode())
            else:
                s.append(f'{k}:{v.__repr__()}')
        return ','.join(s).encode()

    def __set__(self, instance, value):
        ...
        # self.concrete_override=value


import pprint


def compare_content(self, other, verbose=False):
    d = dict(self.__dict__)
    d2 = dict(other.__dict__)

    r = []
    ks = []
    vls = []
    for k in d.keys():
        if not (k == 'uuid') and not (k == 'hash'):
            r.append(d.get(k) == d2.get(k))
            ks.append(k)
            vls.append((d.get(k), d2.get(k)))
    if verbose:
        pprint.pprint(dict(zip(ks, zip(r, vls))))
    return dict(zip(ks, r))


@dataclasses.dataclass
class BaseMaterial:
    color: int
    type: typing.Optional[str] = None

    opacity: float = 1.0
    transparent: bool = False
    uuid: typing.Optional[str] = ""


    def __post_init__(self):
        if self.type is None:
            self.type = self.__class__.__name__
        if self.uuid is None:
            self.uuid = str(self.color)+self.__class__.__name__.lower()
        if self.opacity < 1.0:
            self.transparent = True

    def __eq__(self, other):

        return self.color==other.color

    def __hash__(self):
        return self.color


from enum import Enum


@strawberry.enum
class Materials(str, Enum):
    PointsMaterial = 'PointsMaterial'
    LineBasicMaterial = 'LineBasicMaterial'
    MeshPhongMaterial = 'MeshPhongMaterial'


@strawberry.type
class Material(BaseMaterial):
    color: int
    type: Materials
    uuid: typing.Optional[str] = ""
    reflectivity: typing.Optional[float] = None
    refractionRatio: typing.Optional[float] = None
    depthFunc: int
    depthTest: bool
    depthWrite: bool
    colorWrite: bool
    stencilWrite: bool
    stencilWriteMask: int
    stencilFunc: int
    stencilRef: int
    stencilFuncMask: int
    stencilFail: int
    stencilZFail: int
    stencilZPass: int
    wireframe: typing.Optional[bool] = None
    vertexColors: typing.Optional[bool] = None
    toneMapped: typing.Optional[bool] = None
    emissive: typing.Optional[int] = None
    specular: typing.Optional[int] = None
    shininess: typing.Optional[int] = None
    side: int = 2
    flatShading: typing.Optional[bool] = None
    thickness: typing.Optional[float] = None




@strawberry.type
class LineBasicMaterial(BaseMaterial):
    uuid: typing.Optional[str] = ""
    type: Materials = Materials.LineBasicMaterial
    color: int = 16724838

    depthFunc: int = 3
    depthTest: bool = True
    depthWrite: bool = True
    colorWrite: bool = True
    stencilWrite: bool = False
    stencilWriteMask: int = 255
    stencilFunc: int = 519
    stencilRef: int = 0
    stencilFuncMask: int = 255
    stencilFail: int = 7680
    stencilZFail: int = 7680
    stencilZPass: int = 7680
    linewidth: float = 2.0
    opacity: float = 1.0
    transparent: bool = False



@strawberry.type
class PointsMaterial(BaseMaterial):
    uuid: typing.Optional[str] = ""
    type: Materials = Materials.PointsMaterial
    color: int = 11672217
    size: float = 1
    sizeAttenuation: bool = True
    depthFunc: int = 1
    depthTest: bool = True
    depthWrite: bool = True
    colorWrite: bool = True
    stencilWrite: bool = False
    stencilWriteMask: int = 255
    stencilFunc: int = 519
    stencilRef: int = 0
    stencilFuncMask: int = 255
    stencilFail: int = 7680
    stencilZFail: int = 7680
    stencilZPass: int = 7680



from mmcore.geom.materials import ColorRGB


@strawberry.type
class MeshPhongMaterial(BaseMaterial):
    name: str = "Default"
    color: int
    type: Materials = Materials.MeshPhongMaterial
    emissive: int = 0
    specular: int = 1118481
    shininess: int = 30
    reflectivity: float = 1.2
    refractionRatio: float = 0.98
    side: int = 2
    depthFunc: int = 3
    depthTest: bool = True
    depthWrite: bool = True
    colorWrite: bool = True
    stencilWrite: bool = False
    stencilWriteMask: int = 255
    stencilFunc: int = 519
    stencilRef: int = 0
    stencilFuncMask: int = 255
    stencilFail: int = 7680
    stencilZFail: int = 7680
    stencilZPass: int = 7680
    flatShading: bool = True
    uuid: typing.Optional[str] = ""


@strawberry.input
class ColorInput:
    r: int
    g: int
    b: int

    @property
    def color(self):
        return ColorRGB(self.r, self.g, self.b)


@strawberry.input
class MaterialInput:
    color: ColorInput
    type: Materials = Materials.MeshPhongMaterial
    name: str = "Default"

    @property
    def material(self) -> Materials:
        return getattr(sys.modules.get("__main__"), self.type)(color=self.color, name=self.name)


@strawberry.type
class GqlChart:
    key: str
    id: str = "chart"
    name: str = "Chart"
    colors: str = "default"
    require: tuple[str] = ('linechart', 'piechart')

    def __post_init__(self):
        self.name = self.key.capitalize() + " " + self.name
        self.id = self.name.lower().replace(" ", "_") + "_" + "_".join(self.require)

    def to_dict(self):
        return strawberry.asdict(self)


AnyMaterial = strawberry.union("AnyMaterial", (MeshPhongMaterial,
                                               PointsMaterial,
                                               LineBasicMaterial,
                                               Material
                                               ),
                               description="All materials in one Union Type")

AnyObject3D = strawberry.union("AnyObject3D", (GqlObject3D,
                                               GqlGroup,
                                               GqlGeometry,
                                               GqlMesh,
                                               GqlLine,
                                               GqlPoints,
                                               ),
                               description="All objects in one Union Type")
