import base64
import functools
import typing
from dataclasses import asdict, dataclass, field

import sys
import ujson

from mmcore.base import amatdict
from mmcore.compat.gltf.consts import BUFFER_DEFAULT_HEADER, DEFAULT_ASSET, GLTFBufferDecoded
from mmcore.geom.materials import ColorRGB

ScalarType = (str, float, int, bool)
JSONType = typing.Dict[str, typing.Any]
GLTFExtras = JSONType


def todict_minify(self):
    def gen(obj):
        for k in obj.__slots__:
            v = getattr(obj, k)
            if v is not None:
                yield k, v

    return dict(gen(self))


def todict_nested(self, base_cls):
    def gen(obj):
        if isinstance(obj, (list, tuple)):
            if len(obj) > 0:
                return [gen(o) for o in obj]
        elif isinstance(obj, base_cls):
            dct = dict()
            for k in obj.__slots__:
                v = getattr(obj, k)
                val = gen(v)
                if val is not None:
                    dct[k] = val
            return dct
        else:

            return obj

    return gen(self)

def dataclass_compat(cls=None, /, *, init=True, repr=True, eq=True, order=False,
                     unsafe_hash=False, frozen=False, match_args=True,
                     kw_only=False, slots=False, weakref_slot=False):
    if sys.version_info.minor >= 9:
        return dataclass(cls, init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, frozen=frozen)
    return dataclass(cls, init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, frozen=frozen,
                     kw_only=kw_only, match_args=match_args, slots=slots, weakref_slot=weakref_slot)

# @label('gltf', 'compat')
class GLTFComponent:
    __slots__ = ()

    def accept(self, visitor):
        lookup = "visit_" + type(self).__qualname__.replace(".", "_").lower()
        return getattr(visitor, lookup)(self)

    @classmethod
    def from_gltf(cls, gltf: dict):
        return cls(**gltf)

    """
    def todict(self):
        def gen(obj):
            if isinstance(obj, list):
                if len(obj) > 0:
                    return [gen(o) for o in obj]
            elif isinstance(obj, GLTFComponent):
                dct = dict()
                for k in obj.__slots__:
                    v = getattr(obj, k)
                    if v is not None:
                        dct[k] = gen(v)
                return dct
            else:
                return obj

        return gen(self)"""


class GLTFRequiredComponent(GLTFComponent):
    __slots__ = ()


# @label('gltf', 'compat')
@dataclass
class GLTFExtension(GLTFComponent):
    ...


# @label('gltf', 'compat')
@dataclass(slots=True)
class GLTFBufferView(GLTFComponent):
    byteOffset: int
    byteLength: int
    target: int
    byteStride: typing.Optional[int] = None
    buffer: int = 0
    name: typing.Optional[str] = None

    def todict(self):
        return todict_minify(self)


__encoded_buffs__ = dict()


@functools.lru_cache(None)
def decode_buffer(buff: str):
    headers, bts = buff.split(',')
    return GLTFBufferDecoded(headers, base64.b64decode(bts))


# @label('gltf', 'compat')
@dataclass(slots=True, unsafe_hash=True)
class GLTFBuffer(GLTFComponent):
    uri: str
    byteLength: int

    def decode(self):
        if id(self) not in __encoded_buffs__:
            __encoded_buffs__[id(self)] = decode_buffer(self.uri)

        return __encoded_buffs__[id(self)]

    @staticmethod
    def encode(bts: typing.Union[bytes, bytearray]):
        return f"{BUFFER_DEFAULT_HEADER},{base64.b64encode(bts).decode()}"

    @classmethod
    def from_bytes(cls, bts: typing.Union[bytes, bytearray]):

        return GLTFBuffer(byteLength=len(bts), uri=cls.encode(bts))

    def todict(self):
        for k in self.__slots__:
            if not k.startswith('_'):
                self._encoded

        return asdict(self)

    def __post_init__(self):
        if not self.uri.startswith(BUFFER_DEFAULT_HEADER):
            with open(self.uri, 'rb') as f:
                self.uri = ",".join([BUFFER_DEFAULT_HEADER, base64.b64encode(f.read()).decode()])

    @classmethod
    def from_gltf(cls, gltf: dict):

        if not gltf['uri'].startswith(BUFFER_DEFAULT_HEADER):
            with open(gltf['uri'], 'rb') as f:
                return cls.from_bytes(f.read())


        else:
            return GLTFBuffer(**gltf)


# @label('gltf', 'compat')
@dataclass(slots=True, unsafe_hash=True)
class GLTFAccessor(GLTFComponent):
    """
    https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#schema-reference-accessor
    """

    componentType: int
    count: int
    type: str
    bufferView: typing.Optional[int] = None
    max: typing.Optional[list[typing.Union[int, float]]] = None
    min: typing.Optional[list[typing.Union[int, float]]] = None
    byteOffset: typing.Optional[int] = None
    normalized: bool = False
    name: typing.Optional[str] = None
    extensions: typing.Optional[typing.Any] = None
    extras: typing.Optional[typing.Any] = None

    def todict(self):
        return todict_minify(self)


from mmcore.base.models.gql import MeshStandardMaterial
# @label('gltf', 'compat')
@dataclass(slots=True)
class GLTFPbrMetallicRoughness(GLTFComponent):
    baseColorFactor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallicFactor: float = 0.5
    roughnessFactor: float = 0.5
    baseColorTexture: typing.Optional[dict] = None
    metallicRoughnessTexture: typing.Optional[dict] = None
    extensions: typing.Optional[GLTFExtension] = None
    extras: typing.Optional[GLTFExtras] = None

    def __hash__(self):
        return hash(self.baseColorFactor + (self.metallicFactor, self.roughnessFactor))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def todict(self):
        return todict_minify(self)

    def to_three(self):
        mat = MeshStandardMaterial(color=ColorRGB(*self.baseColorFactor[:-1]).decimal,
                                   roughness=self.roughnessFactor,
                                   metalness=self.metallicFactor)
        amatdict[mat.uuid] = mat
        return mat



@dataclass(slots=True)
class GLTFMaterial(GLTFComponent):
    name: typing.Optional[str] = None
    extensions: typing.Optional[GLTFExtension] = None
    extras: typing.Optional[GLTFExtras] = None
    emissiveFactor: tuple[float] = (0, 0, 0)

    doubleSided: bool = False
    pbrMetallicRoughness: GLTFPbrMetallicRoughness = field(default_factory=GLTFPbrMetallicRoughness)
    normalTexture: typing.Optional[dict] = None
    occlusionTexture: typing.Optional[dict] = None
    emissiveTexture: typing.Optional[dict] = None

    def todict(self):
        return todict_nested(self, GLTFComponent)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return hash(self) == hash(other)


# @label('gltf', 'compat')
@dataclass(slots=True, unsafe_hash=True)
class GLTFPrimitive(GLTFComponent):
    mode: int = 4
    attributes: dict = field(default_factory=dict)
    indices: typing.Optional[int] = None
    material: typing.Optional[int] = None
    targets: typing.Optional[list[dict]] = None
    extensions: typing.Optional[GLTFExtension] = None
    extras: typing.Optional[GLTFExtras] = None

    def to_gltf(self):
        def gen():
            for k, v in asdict(self).items():
                if v is not None:
                    yield k, v

        return dict(gen())

    def todict(self):

        return todict_minify(self)


# @label('gltf', 'compat')
@dataclass(slots=True, unsafe_hash=True)
class GLTFMesh(GLTFComponent):
    primitives: list[GLTFPrimitive]
    name: typing.Optional[str] = None
    weights: typing.Optional[list[float]] = None
    extensions: typing.Optional[GLTFExtension] = None
    extras: typing.Optional[GLTFExtras] = None

    @classmethod
    def from_gltf(cls, gltf: dict):
        prims = gltf.pop('primitives')

        return cls(primitives=[GLTFPrimitive(**p) for p in prims], **gltf)

    def todict(self):
        return todict_nested(self, GLTFComponent)


# @label('gltf', 'compat')
@dataclass(slots=True, unsafe_hash=True)
class GLTFNode(GLTFComponent):
    children: typing.Optional[list[int]] = None
    mesh: typing.Optional[int] = None
    matrix: list[float] = field(default_factory=lambda: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    name: typing.Optional[str] = None
    weights: typing.Optional[list[float]] = None
    extensions: typing.Optional[GLTFExtension] = None
    extras: typing.Optional[GLTFExtras] = None
    translation: typing.Optional[list] = None
    rotation: typing.Optional[list] = None
    scale: typing.Optional[list] = None

    def todict(self):
        return todict_nested(self, GLTFComponent)


# @label('gltf', 'compat')
@dataclass(slots=True, unsafe_hash=True)
class GLTFScene(GLTFComponent):
    nodes: list[int]
    name: typing.Optional[str] = None
    extensions: typing.Optional[GLTFExtension] = None
    extras: typing.Optional[GLTFExtras] = None

    def todict(self):
        return todict_minify(self)


@dataclass(slots=True, unsafe_hash=True)
class GLTFDocument(GLTFComponent):
    scenes: list[GLTFScene] = field(default_factory=lambda: [GLTFScene([])])
    scene: int = 0
    nodes: list = field(default_factory=list)
    bufferViews: list[GLTFBufferView] = field(default_factory=list)
    buffers: list[GLTFBuffer] = field(default_factory=list)
    accessors: list[GLTFAccessor] = field(default_factory=list)
    materials: list = field(default_factory=list)
    meshes: list[GLTFMesh] = field(default_factory=list)
    extensionsUsed: typing.Optional[list[str]] = None
    extensionsRequired: typing.Optional[list[str]] = None
    extensions: typing.Optional[list] = None
    extras: typing.Optional[GLTFExtras] = None
    asset: dict = field(default_factory=lambda: dict(DEFAULT_ASSET))

    @classmethod
    def from_gltf(cls, gltf: dict):
        return cls(
            **{k: [DOCUMENT_MAP.get(k).from_gltf(item) for item in v] if k in DOCUMENT_MAP else v for k, v in
               gltf.items()})

    def todict(self):
        return todict_nested(self, GLTFComponent)

    def dump(self, fp):
        with open(fp, 'w') as f:
            ujson.dump(self.todict(), f)

DOCUMENT_MAP = dict(
    scenes=GLTFScene,
    nodes=GLTFNode,
    bufferViews=GLTFBufferView,
    buffers=GLTFBuffer,
    accessors=GLTFAccessor,
    materials=GLTFMaterial,
    meshes=GLTFMesh

)
RELATIVE_FIELDS_MAP = dict(
    GLTFScene=(),
    GLTFNode=('children', 'mesh'),
    GLTFBufferView=(),
    GLTFBuffer=(),
    GLTFAccessor=('bufferView'),
    GLTFMaterial=(),
    GLTFMesh=(),
    GLTFPrimitive=("indices", "indices")

)
