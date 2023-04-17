from __future__ import annotations

import sys
import typing
import uuid

import strawberry
from strawberry.scalars import JSON

from mmcore.base.utils import getitemattr
from mmcore.base.utils import objdict


class ChildrenDesc:

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __init__(self, default=None):
        self._default = default

    def __get__(self, instance, owner):
        if instance is None:
            return self._default
        else:
            l = []
            for uid in objdict[instance.uuid]._children:
                l.append(objdict[uid].bind_class(**objdict[uid].threejs_repr))
            return l

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
    type: str
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
    color: typing.Union[Color,None]=None
    normal: typing.Union[Normal,None]=None
    uv: typing.Union[Uv,None]=None


@strawberry.type
class Attributes3(Attributes1):
    position: Position
    normal: typing.Union[Normal,None]=None


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
class BufferGeometry:
    uuid: str
    type: str
    data: typing.Union[Data, Data1] = None


@strawberry.type
class SphereBufferGeometry(BufferGeometry):
    radius: float | None = None
    detail: int | None = None


@strawberry.type
class LinkItem:
    name: str


@strawberry.type
class GqlUserData:
    properties: JSON | None = None
    gui: list[GqlChart] | None = None
    params: JSON | None = None


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
    name: str
    uuid: str
    geometry: str
    material: str
    type: str = "Geometry"
    children: 'list[typing.Union[GqlGeometry, None]]' = ChildrenDesc()

strawberry.auto
@strawberry.type
class GqlObject3D(GqlBaseObject):
    name: str
    uuid: str
    userData: GqlUserData
    matrix: list[float] = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    layers: int = 1
    type: str = "Object3D"
    castShadow: bool = True
    receiveShadow: bool = True
    children:  'list[typing.Union[ GqlObject3D, GqlGroup, GqlGeometry, None]]' = ChildrenDesc()




@strawberry.type
class GqlGroup(GqlObject3D):
    type: str = "Group"
    matrix: list[float] = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    children: 'list[typing.Union[ GqlObject3D, GqlGroup, GqlGeometry, None]]' = ChildrenDesc()





@strawberry.type
class GqlMesh(GqlGeometry):
    type: str = "Mesh"


@strawberry.type
class GqlLine(GqlGeometry):
    type: str = "Line"


@strawberry.type
class GqlPoints(GqlGeometry):
    type: str = "Points"


@strawberry.type
class BaseMaterial:
    uuid: str
    color: int
    type: str


@strawberry.type
class Material(BaseMaterial):
    uuid: str
    type: str
    color: int
    reflectivity: float | None = None
    refractionRatio: float | None = None
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
    wireframe: bool | None = None
    vertexColors: bool | None = None
    toneMapped: bool | None = None
    emissive: int | None = None
    specular: int | None = None
    shininess: int | None = None
    side: int | None = None
    flatShading: bool | None = None
    thickness: float | None = None


@strawberry.type
class LineBasicMaterial(BaseMaterial):
    uuid: str
    type: str = "LineBasicMaterial"
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


@strawberry.type
class PointsMaterial:
    uuid: str = 'a8494fb8-3b8a-420a-ab45-bb47f60f1eb6'
    type: str = 'PointsMaterial'
    color: int = 11672217
    size: int = 1
    sizeAttenuation: bool = True
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


from enum import Enum

from mmcore.geom.materials import ColorRGB


@strawberry.type
class MeshPhongMaterial(BaseMaterial):
    name: str = "Default"
    color: int
    type: str = "MeshPhongMaterial"
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
    uuid: str | None = None

    def __post_init__(self):
        self.uuid = uuid.uuid4().__str__()


@strawberry.enum
class Materials(Enum):
    PointsMaterial = 'PointsMaterial'
    LineBasicMaterial = 'LineBasicMaterial'
    MeshPhongMaterial = 'MeshPhongMaterial'


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


        return getattr(sys.modules.get("__main__"),self.type)(color=self.color, name=self.name)


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
