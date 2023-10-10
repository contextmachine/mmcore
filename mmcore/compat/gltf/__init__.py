"""Grasshopper Script"""
import abc
import base64
import math
import operator
from enum import Enum
from functools import reduce

a = "Hello Python 3 in Grasshopper!"
print(a)

from dataclasses import dataclass, asdict
import typing
import struct
from collections import OrderedDict, namedtuple

from more_itertools import chunked
from dataclasses import dataclass, asdict, InitVar, field
import typing
import struct
from collections import namedtuple

from more_itertools import chunked

_components = dict(ixs=dict(count=3, size=1, dtype='h', byteOffset=0, byteLength=6),
                   pos=dict(count=3, size=3, dtype='f', byteOffset=8, byteLength=36))

from uuid import uuid4

docs_table = dict()
entities_table = dict()
components_table = dict()
DEFAULT_UUID_FACTORY = lambda: uuid4().hex


@dataclass
class GLTFEntity:
    uuid: str = field(default_factory=DEFAULT_UUID_FACTORY)

    def __post_init__(self):
        if self.uuid not in entities_table:
            entities_table[self.uuid] = dict()

    @property
    def components(self) -> dict:
        return entities_table[self.uuid]

    def add_component(self, name: str, component: typing.Type['GLTFComponent']):
        entities_table[self.uuid][name] = component


__typreg__ = dict()

GLTFType = namedtuple('GLTFType', ['name', 'format', 'size'])


class GLTFTypeEnum(GLTFType, Enum):
    VEC2 = GLTFType('VEC2', format='f', size=2)
    VEC3 = GLTFType('VEC3', format='f', size=3)
    SCALAR = GLTFType("SCALAR", format="h", size=1)


VVVV = dict(VEC2=GLTFType('VEC2', format='f', size=2),
            VEC3=GLTFType('VEC3', format='f', size=3),
            SCALAR=GLTFType("SCALAR", format="h", size=1))


@dataclass
class GLTFBufferView:
    buffer: int
    byteOffset: int
    byteLength: int
    target: int
    byteStride: typing.Optional[int] = None


GLTFBufferDecoded = namedtuple('GLTFBufferDecoded', ['headers', 'buffer'])
comp_table = dict()
comp_props = dict()
comp_tps = dict()
comp_docs = dict()
_STORES = comp_props, comp_docs


class GLTFComponent:
    ...


@dataclass
class GLTFBuffer(GLTFComponent):
    byteLength: int
    uri: str

    def decode(self):
        headers, btc = self.uri.split(',')
        return GLTFBufferDecoded(headers, base64.b64decode(btc))


@dataclass
class GLTFAccessor(GLTFComponent):
    bufferView: int
    componentType: int
    count: int
    max: list[typing.Union[int, float]]
    min: list[typing.Union[int, float]]
    type: str


GLTFAttribute = namedtuple('GLTFAttribute', ['name', 'value'])


class GLTFAttributeEnum(str, Enum):
    POSITION = 'POSITION'
    NORMAL = 'NORMAL'
    TEXCOORD_0 = 'TEXCOORD_0'


@dataclass
class GLTFPrimitive:
    mode: int
    attributes: dict = field(default_factory=dict)
    indices: typing.Optional[int] = None
    material: typing.Optional[int] = None

    def to_gltf(self):
        def gen():
            for k, v in asdict(self).items():
                if v is not None:
                    yield k, v

        return dict(gen())

    def get_attributes(self, doc):
        for k, v in self.attributes.items():
            yield k, buffer_format(doc.accessors[v], doc)

    def get_indices(self, doc):

        return buffer_format(doc.accessors[self.indices], doc)

    def gen_mesh_dict(self, doc):
        for k, v in self.get_attributes(doc):
            yield k, v
        if self.indices is not None:
            yield 'indices', self.get_indices(doc)


class MeshPartIterator:
    def __init__(self, mesh, doc):
        self.mesh = mesh
        self.doc = doc

    def __iter__(self):
        return (dict(p.gen_mesh_dict(self.doc)) for p in self.mesh.primitives)


@dataclass
class GLTFMesh(GLTFComponent):
    primitives: list[typing.Union[GLTFPrimitive, dict]]

    def __post_init__(self):
        prm = []
        for p in self.primitives:
            if isinstance(p, dict):
                prm.append(GLTFPrimitive(**p))
            else:
                prm.append(p)
        self.primitives = prm


def mesh_iterator(doc):
    for item in doc.meshes:
        yield list(MeshPartIterator(item, doc))


@dataclass
class GLTFDocument:
    scenes: list = field(default_factory=list)
    scene: int = 0
    nodes: list = field(default_factory=list)
    bufferViews: list[GLTFBufferView] = field(default_factory=list)
    buffers: list[GLTFBuffer] = field(default_factory=list)
    accessors: list[GLTFAccessor] = field(default_factory=list)
    materials: list = field(default_factory=list)
    meshes: list[GLTFMesh] = field(default_factory=list)
    asset: dict = field(default_factory=dict)


def create_doc(data):
    return GLTFDocument(bufferViews=[GLTFBufferView(**accessor) for accessor in data['bufferViews']],
                        buffers=[GLTFBuffer(**accessor) for accessor in data['buffers']],
                        accessors=[GLTFAccessor(**accessor) for accessor in data['accessors']],
                        meshes=[GLTFMesh(**accessor) for accessor in data['meshes']])


def gltfff(buff='AAABAAIAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAACAPwAAAAA=',
           components: list[tuple[GLTFAccessor, GLTFBufferView]] = None):
    bts = base64.b64decode(buff)
    for assessor, bufferview in components:
        dtype = VVVV[assessor.type]
        *res, = chunked(
            struct.unpack(f"{assessor.count * dtype.size}{dtype.format}",
                          bts[bufferview.byteOffset:bufferview.byteOffset + bufferview.byteLength]), dtype.size)
        yield res


def buffer_format(accessor: GLTFAccessor, bufferView: GLTFBufferView) -> str:
    if bufferView.byteStride is None:
        bufferView.byteStride = accessor.count * accessor.type.size
    return f"{bufferView.byteStride}{accessor.type.format}"


def read_buffer_item(buffer: bytes, accessor: GLTFAccessor, buffer_view: GLTFBufferView):
    return chunked(
        struct.unpack(
            buffer_format(accessor, buffer_view),
            buffer[buffer_view.byteOffset: buffer_view.byteOffset + buffer_view.byteLength]
        ),
        accessor.type.size
    )


def read_buffer(doc: GLTFDocument):
    buffers = [buffer.decode()[1] for buffer in doc.buffers]
    for view, accessor in zip(doc.bufferViews, doc.accessors):
        read_buffer_item(buffers[view.buffer], accessor, view)
        yield list()


def meshdict(doc):
    for mesh in doc.meshes:
        for primitive in mesh.primitives:
            primitive
    GLTFMesh()


with open('/Users/andrewastakhov/Downloads/scene-5.gltf') as f:
    import json

    gltfdata = json.load(f)
doc = create_doc(gltfdata)
h, bb = doc.buffers[0].uri.split(",")
data = list(gltfff(bb, list(zip(doc.accessors, doc.bufferViews))))
