from __future__ import annotations

import uuid as _uuid
import dataclasses
import hashlib
import pprint
import typing
from enum import Enum

import numpy as np
import strawberry
from strawberry.scalars import JSON

import mmcore
from mmcore.base.registry import ageomdict, objdict
from mmcore.base.utils import getitemattr
from mmcore.geom.materials import ColorRGB


# noinspection PyProtectedMember
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
                lst = []
                for uid in objdict[instance.uuid]._children:
                    lst.append(objdict[uid].bind_class(**objdict[uid].threejs_repr))
                return lst
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
        return len(self.array)


@strawberry.type
class Color(Position):
    type: str = "Float32Array"
    array: list[float]
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


@dataclasses.dataclass
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
    attributes: typing.Union[
        Attributes, Attributes1, Attributes2, Attributes3, Attributes4
    ]
    index: typing.Union[Index, None] = None


@strawberry.type
class Data2:
    attributes: typing.Union[
        Attributes, Attributes1, Attributes2, Attributes3, Attributes4
    ]
    index: typing.Union[Index, None] = None
    boundingSphere: typing.Union[BoundingSphere, None] = None


@strawberry.type
class Data1:
    attributes: typing.Union[Attributes, Attributes1, Attributes2, Attributes3]


@strawberry.type
class Data12:
    attributes: typing.Union[Attributes, Attributes1, Attributes2, Attributes3]
    boundingSphere: typing.Union[BoundingSphere, None] = None


@strawberry.type
class PositionOnlyData:
    attributes: Attributes1


@strawberry.type
class Data3:
    attributes: Attributes3


class HashUUID:
    def __init__(self, default=None):
        self._default = default

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner) -> str:
        if self._default is None:
            if instance:
                self._default = instance.sha().hexdigest()

        return self._default

    def __set__(self, instance, value):
        return DeprecationWarning(
            f"UUID set of {instance} is deprecated and not supported!"
        )


@dataclasses.dataclass
class BufferGeometryObject:
    data: typing.Union[Data, Data1]
    type: str = "BufferGeometry"
    uuid: typing.Optional[str] = None

    def __post_init__(self):
        ##print("create")
        if self.uuid is None:
            self.uuid = hex(self.sha())
        ageomdict[self.uuid] = self

    def sha(self):
        if isinstance(self.data, dict):
            return hash(repr(self.data["attributes"]["position"]["array"]))
        return hash(repr(self.data.attributes.position.array))

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
    matrix: list[float] = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
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


GQLGeometryUnion = strawberry.union(
    "GQLObject3DUnion", types=[GqlGeometry, GqlMesh, GqlLine, GqlPoints]
)
GQLObject3DUnion = strawberry.union(
    "GQLObject3DUnion",
    types=[GqlObject3D, GqlGroup, GqlGeometry, GqlMesh, GqlLine, GqlPoints],
)


class UidSha256:
    def __init__(self, default="", concrete_override: str = None):
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
        if "uuid" in d.keys():
            del d["uuid"]
        if self.name in d.keys():
            del d[self.name]
        s = []
        for k, v in d.items():
            if hasattr(v, self.name):
                s.append(f"{k}:{getattr(v, self.name)}")
            elif v.__class__.__name__ in ["str", "int", "bytes", "float", "bool"]:
                s.append(f"{k}:{v}")
            elif isinstance(v, str):
                s.append(f"{k}:{v}")
            elif isinstance(v, typing.Iterable):
                s.append(f'{k}:{",".join([self.to_bytes(vv).decode() for vv in v])}')
            elif isinstance(v, dict):
                s.append(
                    ",".join([f"{k}={self.to_bytes(v)}" for k, v in v.items()]).encode()
                )
            else:
                s.append(f"{k}:{v.__repr__()}")
        return ",".join(s).encode()

    def __set__(self, instance, value):
        ...
        # self.concrete_override=value


def compare_content(self, other, verbose=False):
    d = dict(self.__dict__)
    d2 = dict(other.__dict__)

    r = []
    ks = []
    vls = []
    for k in d.keys():
        if not (k == "uuid") and not (k == "hash"):
            r.append(d.get(k) == d2.get(k))
            ks.append(k)
            vls.append((d.get(k), d2.get(k)))
    if verbose:
        pprint.pprint(dict(zip(ks, zip(r, vls))))
    return dict(zip(ks, r))


def create_material_uuid(self: BaseMaterial, postfix=None):
    if postfix is None:
        return "-".join(
            [
                hex(self.color),
                str(round(self.opacity, 1)).replace(".", "_"),
                self.__class__.__name__.lower(),
            ]
        )
    else:
        return "-".join(
            [
                hex(self.color),
                str(round(self.opacity, 1)).replace(".", "_"),
                f"{postfix}".lower(),
            ]
        )


@dataclasses.dataclass
class BaseMaterial:
    color: int
    type: typing.Optional[str] = None

    opacity: float = 1.0
    side: int = 2
    transparent: bool = False
    uuid: typing.Optional[str] = None

    def __post_init__(self):
        if self.type is None:
            self.type = self.__class__.__name__
        if self.uuid is None:
            self.uuid = (
                hex(self.color)
                + str(round(self.opacity, 1)).replace(".", "_")
                + self.__class__.__name__.lower()
            )
        if self.opacity < 1.0:
            self.transparent = True

    def __eq__(self, other):
        return self.color == other.color

    def __hash__(self):
        return hash(self.uuid)


@strawberry.enum
class Materials(str, Enum):
    PointsMaterial = "PointsMaterial"
    LineBasicMaterial = "LineBasicMaterial"
    MeshPhongMaterial = "MeshPhongMaterial"
    MeshBasicMaterial = "MeshBasicMaterial"
    MeshStandardMaterial = "MeshStandardMaterial"


@strawberry.type
class MeshBasicMaterial(BaseMaterial):
    uuid: typing.Optional[str] = None
    type: Materials = Materials.MeshBasicMaterial
    name: str = "DefaultBasicMaterial"
    color: int = 0
    reflectivity: float = 1.0
    refractionRatio: float = 0.98

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


@strawberry.type
class LineBasicMaterial(BaseMaterial):
    uuid: typing.Optional[str] = None
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
    uuid: typing.Optional[str] = None
    type: Materials = Materials.PointsMaterial
    color: int = 11672217
    size: float = 0.1
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


@strawberry.type
class MeshPhongMaterial(BaseMaterial):
    name: str = "Default"
    color: int
    vertexColors: bool = False
    type: Materials = Materials.MeshPhongMaterial

    emissive: int = 0
    specular: int = 1118481
    shininess: int = 30
    reflectivity: float = 0.5
    refractionRatio: float = 0.5
    side: int = 2
    depthFunc: int = 2
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
    uuid: typing.Optional[str] = None


@dataclasses.dataclass
class MeshStandardMaterial(BaseMaterial):
    name: str = "Default"
    color: int
    vertexColors: bool = False
    type: Materials = Materials.MeshStandardMaterial
    side: int = 2
    emissive: int = 0
    roughness: float = 0.68
    metalness: float = 0.46
    envMapIntensity: float = 1
    blendColor: float = 0
    flatShading: bool = True
    uuid: typing.Optional[str] = None


@strawberry.type
class MeshStandardVertexMaterial(BaseMaterial):
    name: str = "VertexStandardDefault"
    color: int = 15461355
    vertexColors: bool = True
    type: Materials = Materials.MeshStandardMaterial
    side: int = 2
    emissive: int = 0
    roughness: float = 0.68
    metalness: float = 0.46
    envMapIntensity: float = 1
    blendColor: float = 0
    flatShading: bool = True
    uuid: typing.Optional[str] = None


@strawberry.type
class GqlChart:
    key: str
    id: str = "chart"
    name: str = "Chart"
    colors: str = "default"
    require: tuple[str] = ("linechart", "piechart")

    def __post_init__(self):
        self.name = self.key.capitalize() + " " + self.name
        self.id = self.name.lower().replace(" ", "_") + "_" + "_".join(self.require)

    def todict(self):
        return strawberry.asdict(self)


AnyMaterial = strawberry.union(
    "AnyMaterial",
    (
        MeshPhongMaterial,
        PointsMaterial,
        LineBasicMaterial,
    ),
    description="All materials in one Union Type",
)

AnyObject3D = strawberry.union(
    "AnyObject3D",
    (
        GqlObject3D,
        GqlGroup,
        GqlGeometry,
        GqlMesh,
        GqlLine,
        GqlPoints,
    ),
    description="All objects in one Union Type",
)
mapattrs = {"normal": Normal, "position": Position, "uv": Uv}
attributes = []


def geom_attributes_from_dict(att):
    ddct = {}
    for k, v in att.items():
        ddct[k] = mapattrs[k](**v)
    if len(ddct.keys()) == 1:
        return Attributes1(**ddct)
    elif len(ddct.keys()) == 3:
        return Attributes2(**ddct)
    elif len(ddct.keys()) == 2:
        return Attributes3(**ddct)
    else:
        return Attributes(**ddct)


def create_buffer_from_dict(kwargs) -> BufferGeometry:
    if "index" in kwargs["data"]:
        dct = BufferGeometry(
            **{
                "uuid": kwargs.get("uuid"),
                "type": kwargs.get("type"),
                "data": Data(
                    **{
                        "attributes": geom_attributes_from_dict(
                            kwargs["data"]["attributes"]
                        ),
                        "index": Index(**kwargs.get("data").get("index")),
                    }
                ),
            }
        )

    else:
        dct = BufferGeometry(
            **{
                "uuid": kwargs.get("uuid"),
                "type": kwargs.get("type"),
                "data": Data1(
                    **{
                        "attributes": geom_attributes_from_dict(
                            kwargs["data"]["attributes"]
                        ),
                    }
                ),
            }
        )

    return dct


def create_full_buffer(
    uuid,
    position=None,
    uv=None,
    index=None,
    normal=None,
    color: typing.Optional[list[float]] = None,
    threejs_type="BufferGeometry",
):
    attra = dict(position=create_buffer_position(position))
    if color is not None:
        attra["color"] = create_float32_buffer(color)
    if normal is not None:
        attra["normal"] = create_float32_buffer(normal)
    if uv is not None:
        attra["uv"] = create_buffer_uv(uv)
    if index is not None:
        ixs = create_buffer_index(index)
        return BufferGeometry(
            **{
                "uuid": uuid,
                "type": threejs_type,
                "data": {"attributes": attra, "index": ixs},
            }
        )
    else:
        return BufferGeometry(
            **{"uuid": uuid, "type": threejs_type, "data": {"attributes": attra}}
        )


def create_buffer_from_occ(kwargs) -> BufferGeometry:
    return BufferGeometry(
        **{
            "uuid": kwargs.get("uuid"),
            "type": kwargs.get("type"),
            "data": Data(
                **{
                    "attributes": Attributes(
                        **{
                            "normal": Normal(
                                **kwargs.get("data").get("attributes").get("normal")
                            ),
                            "position": Position(
                                **kwargs.get("data").get("attributes").get("position")
                            ),
                        }
                    )
                }
            ),
        }
    )


def create_buffer_position(array, normalized=False):
    return create_float32_buffer(array, normalized=normalized)


def create_float32_buffer(array, normalized=False):
    return {
        "array": array,
        "type": "Float32Array",
        "itemSize": 3,
        "normalized": normalized,
    }


METERIAL_TYPE_MAP = dict(
    MeshBasicMaterial=MeshBasicMaterial,
    MeshPhongMaterial=MeshPhongMaterial,
    PointsMaterial=PointsMaterial,
    LineBasicMaterial=LineBasicMaterial,
)


def create_material(
    store,
    uuid=None,
    color=(100, 100, 100),
    transparent=False,
    opacity: float = 1.0,
    vertexColors=False,
    material_type="MeshPhongMaterial",
    **kwargs,
):
    if uuid is None:
        uuid = _uuid.uuid4().hex
    m = METERIAL_TYPE_MAP[material_type](
        uuid=uuid,
        color=ColorRGB(*color).decimal,
        vertexColors=vertexColors,
        transparent=transparent,
        opacity=opacity,
        **kwargs,
    )
    store[uuid] = m
    return m


def update_material_color(store: dict, uuid, color: tuple[int, int, int]):
    store[uuid].color = ColorRGB(color).decimal


def update_material_opacity(store: dict, uuid, transparent=True, opacity: float = 1.0):
    mat = store[uuid]
    mat.transparent = transparent
    mat.opacity = opacity


def create_buffer_index(array):
    return {"type": "Uint16Array", "array": array}


def create_buffer_color(array, normalized=False):
    return {
        "array": array,
        "type": "Float32Array",
        "itemSize": 3,
        "normalized": normalized,
    }


def create_buffer_uv(array, normalized=False):
    return {
        "array": array,
        "type": "Float32Array",
        "itemSize": 2,
        "normalized": normalized,
    }


def create_buffer_normals(array, normalized=False):
    return {
        "array": array,
        "type": "Float32Array",
        "itemSize": 3,
        "normalized": normalized,
    }


from typing import TypedDict, Union


class BufferAttr(TypedDict):
    array: list[Union[float, int]]
    normalized: typing.Optional[bool]


def buffer_attr(array, normalized=None):
    if normalized is not None:
        return {"array": array, "normalized": normalized}
    else:
        return {"array": array}


def create_shape_buffer(
    uuid,
    position,
    index,
    color: typing.Optional[list[float]] = None,
    threejs_type="BufferGeometry",
) -> BufferGeometry:
    if color:
        return BufferGeometry(
            **{
                "uuid": uuid,
                "type": threejs_type,
                "data": {
                    "attributes": {
                        "position": create_buffer_position(position),
                        "color": create_buffer_color(color),
                    },
                    "index": create_buffer_index(index),
                },
            }
        )
    return BufferGeometry(
        **{
            "uuid": uuid,
            "type": threejs_type,
            "data": {
                "attributes": {"position": create_buffer_position(position)},
                "index": create_buffer_index(index),
            },
        }
    )


def update_shape_buffer(
    store: dict,
    uuid,
    position=None,
    index=None,
    color: typing.Optional[list[float]] = None,
) -> None:
    buf = store[uuid]
    for name, attr in (("position", position), ("color", color)):
        if attr is not None:
            buf.data["attributes"][name] = create_float32_buffer(array=attr)
    if index is not None:
        buf.index = create_buffer_index(index)


def create_uvlike_buffer(
    store: dict,
    uuid,
    position=None,
    uv=None,
    normal=None,
    color: typing.Optional[list[float]] = None,
    threejs_type="BufferGeometry",
) -> None:
    attra = dict(position=create_buffer_position(position))
    if color is not None:
        attra["color"] = create_float32_buffer(color)
    if normal is not None:
        attra["normal"] = create_float32_buffer(normal)
    if uv is not None:
        attra["uv"] = create_buffer_uv(uv)

    return BufferGeometry(
        **{"uuid": uuid, "type": threejs_type, "data": {"attributes": attra}}
    )


def create_full_buffer(
    uuid,
    position=None,
    uv=None,
    index=None,
    normal=None,
    color: typing.Optional[list[float]] = None,
    threejs_type="BufferGeometry",
):
    attra = dict(position=create_buffer_position(position))
    if color is not None:
        attra["color"] = create_float32_buffer(color)
    if normal is not None:
        attra["normal"] = create_float32_buffer(normal)
    if uv is not None:
        attra["uv"] = create_buffer_uv(uv)
    if index is not None:
        ixs = create_buffer_index(index)
        return BufferGeometry(
            **{
                "uuid": uuid,
                "type": threejs_type,
                "data": {"attributes": attra, "index": ixs},
            }
        )
    else:
        return BufferGeometry(
            **{"uuid": uuid, "type": threejs_type, "data": {"attributes": attra}}
        )


def create_buffer(uuid, normals, vertices, uv, indices) -> BufferGeometry:
    return BufferGeometry(
        **{
            "uuid": uuid,
            "type": "BufferGeometry",
            "data": Data(
                **{
                    "attributes": Attributes(
                        **{
                            "normal": Normal(
                                **{
                                    "array": np.asarray(normals, dtype=float)
                                    .flatten()
                                    .tolist(),
                                    "itemSize": 3,
                                    "type": "Float32Array",
                                    "normalized": False,
                                }
                            ),
                            "position": Position(
                                **{
                                    "array": np.asarray(vertices, dtype=float)
                                    .flatten()
                                    .tolist(),
                                    "itemSize": 3,
                                    "type": "Float32Array",
                                    "normalized": False,
                                }
                            ),
                            "uv": Uv(
                                **{
                                    "itemSize": 2,
                                    "array": np.asarray(uv, dtype=float)
                                    .flatten()
                                    .tolist(),
                                    "type": "Float32Array",
                                    "normalized": False,
                                }
                            ),
                        }
                    ),
                    "index": Index(
                        **dict(
                            type="Uint16Array",
                            array=np.asarray(indices, dtype=int).flatten().tolist(),
                        )
                    ),
                }
            ),
        }
    )
