"""Grasshopper Script"""
import base64

from mmcore.base.table import Table, TableProxy

a = "Hello Python 3 in Grasshopper!"
print(a)

from dataclasses import dataclass, asdict, field
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

__typreg__ = dict()


def finalize_gltf_buffer(data: tuple, typ_size: int):
    if typ_size == 1:
        return data
    else:
        return chunked(data, typ_size)


meshPrimitiveMode = [
    {
        "const": 0,
        "description": "POINTS",
        "type": "integer"
    },
    {
        "const": 1,
        "description": "LINES",
        "type": "integer"
    },
    {
        "const": 2,
        "description": "LINE_LOOP",
        "type": "integer"
    },
    {
        "const": 3,
        "description": "LINE_STRIP",
        "type": "integer"
    },
    {
        "const": 4,
        "description": "TRIANGLES",
        "type": "integer"
    },
    {
        "const": 5,
        "description": "TRIANGLE_STRIP",
        "type": "integer"
    },
    {
        "const": 6,
        "description": "TRIANGLE_FAN",
        "type": "integer"
    }
]
gltfTypes = [
    {
        "const": "SCALAR",
        "size": 1,

    },
    {
        "const": "VEC2",
        "size": 2,

    },
    {
        "const": "VEC3",
        "size": 3
    },
    {
        "const": "VEC4",
        "size": 4
    },
    {
        "const": "MAT2",
        "size": 2 * 2
    },
    {
        "const": "MAT3",
        "size": 3 * 3
    },
    {
        "const": "MAT4",
        "size": 4 * 4
    }

]
componentTypes = [
    {
        "const": 5120,
        "description": "BYTE",
        "type": "integer"
    },
    {
        "const": 5121,
        "description": "UNSIGNED_BYTE",
        "type": "integer"
    },
    {
        "const": 5122,
        "description": "SHORT",
        "type": "integer"
    },
    {
        "const": 5123,
        "description": "UNSIGNED_SHORT",
        "type": "integer"
    },
    {
        "const": 5125,
        "description": "UNSIGNED_INT",
        "type": "integer"
    },
    {
        "const": 5126,
        "description": "FLOAT",
        "type": "integer"
    }
]

componentTypesTable = Table("const",
                            rows=componentTypes,
                            name='gltf_component_types_table',
                            schema={
                                "const": int,
                                "description": str,
                                "type": str
                            }
                            )
meshPrimitiveModeTable = Table("const",
                               rows=meshPrimitiveMode,
                               name='gltf_mesh_primitive_mode_table',
                               schema={
                                   "const": int,
                                   "description": str,
                                   "type": str
                               }
                               )
componentTypeCodeTable = Table.from_lists(pk='gltf',
                                          rows=[('b', 1, componentTypesTable[5120], "Int8Array"),
                                                ('B', 1, componentTypesTable[5121], "Uint8Array"),
                                                ('h', 2, componentTypesTable[5122], "Int16Array"),
                                                ('H', 2, componentTypesTable[5123], "Uint16Array"),
                                                ('L', 4, componentTypesTable[5125], "Uint32Array"),
                                                ('f', 4, componentTypesTable[5126], "Float32Array")],
                                          name='gltf_typecode_table',
                                          schema={
                                              'typecode': str,
                                              'size': int,
                                              'gltf': TableProxy,
                                              'js': str
                                          }
                                          )

TYPE_TABLE = Table('const',
                   rows=gltfTypes,
                   schema=dict(const=str, size=int),
                   name='gltf_target_type_table')


# componentTypeTable = {comp_type['const']: GLTFComponentType(**comp_type) for comp_type in componentTypes}


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
    """
    https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#schema-reference-accessor
    """

    bufferView: int
    componentType: int
    count: int
    max: list[typing.Union[int, float]]
    min: list[typing.Union[int, float]]
    type: str
    bufferOffset: int = 0
    normalized: bool = False
    name: typing.Optional[str] = None
    extensions: typing.Optional[typing.Any] = None
    extras: typing.Optional[typing.Any] = None

GLTFAttribute = namedtuple('GLTFAttribute', ['name', 'value'])


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
    return GLTFDocument(bufferViews=[GLTFBufferView(**buffview) for buffview in data['bufferViews']],
                        buffers=[GLTFBuffer(**buffer) for buffer in data['buffers']],
                        accessors=[GLTFAccessor(**accessor) for accessor in data['accessors']],
                        meshes=[GLTFMesh(**mesh) for mesh in data['meshes']])


def gltf_buffer_to_data(buffers: list[str],
                        assessors: list[GLTFAccessor],
                        bufferViews: list[GLTFBufferView],
                        type_table: Table = TYPE_TABLE,
                        component_type_table: Table = componentTypeCodeTable):
    bts_list = [base64.b64decode(buffer) for buffer in buffers]

    for assessor, bufferview in zip(assessors, bufferViews):
        dtype = type_table[assessor.type]
        cmp_type = component_type_table[assessor.componentType]
        bts = bts_list[bufferview.buffer]
        yield finalize_gltf_buffer(
            struct.unpack(f"{assessor.count * dtype['size']}{cmp_type['typecode']}",
                          bts[bufferview.byteOffset:bufferview.byteOffset + bufferview.byteLength]), dtype['size'])


def buffer_format(accessor: GLTFAccessor, bufferView: GLTFBufferView) -> str:
    if bufferView.byteStride is None:
        bufferView.byteStride = accessor.count * accessor.type.size
    return f"{bufferView.byteStride}{accessor.type.format}"


def read_buffer_item(buffer: list[bytes], accessor: GLTFAccessor, buffer_view: GLTFBufferView):
    return chunked(
        struct.unpack(
            buffer_format(accessor, buffer_view),
            buffer[buffer_view.byteOffset: buffer_view.byteOffset + buffer_view.byteLength]
        ),
        accessor.type.size
    )


def gltf_decode_buffer(buffers):
    _prefixes = []
    _buffers = []
    for buffer in buffers:
        pref, buff = buffer.uri.split(",")
        _buffers.append(buff)
        _prefixes.append(pref)
    return _prefixes, _buffers


EFFECTIVE_BYTE_STRIDE = 4

MESH_PRIM_KEYS = 'POSITION', 'NORMAL', 'TANGENT', 'TEXCOORD_n', 'COLOR_n', 'JOINTS_n', 'WEIGHTS_n', '_SPECIFIC'
attrmap = dict(

    POSITION='vertices',
    NORMAL='normals',
    TANGENT='tangent',
    TEXCOORD='uv',
    COLOR='colors',
    JOINTS='joints',
    WEIGHTS='WEIGHTS'.lower()

)


def create_mesh_prim_data(prim_keys=MESH_PRIM_KEYS, name_mapping=attrmap):
    if name_mapping is None:
        name_mapping = dict()
    for pkey in prim_keys:
        # row=dict.fromkeys(schema.keys(), None)
        row = {'gltf_name': pkey, 'mmcore_name': None, 'collection': False, 'specific': False}
        if pkey.startswith('_'):
            row['specific'] = True
        spkey = pkey.split('_')
        if all([len(spkey) > 1, spkey[-1] == 'n']):
            row['collection'] = True
            row['gltf_name'] = spkey[0]

        row['mmcore_name'] = name_mapping.get(spkey[0], pkey.lower())
        yield row


meshPrimitiveAttrTable = Table(
    'gltf_name',
    rows=list(create_mesh_prim_data(prim_keys=MESH_PRIM_KEYS)),
    name="meshPrimitiveAttrTable",
    schema={'gltf_name': str, 'mmcore_name': str, 'collection': bool, 'specific': bool}
)


class Indices:
    indices: int


class Attributes:
    indices: int


def gltf_mesh_primitive_table(primitives, buffer_components, attribute_table: Table = meshPrimitiveAttrTable):
    for primitive in primitives:
        data = dict()

        for k, v in primitive.attributes.items():
            split_k = k.split('_')
            key = attribute_table[split_k[0]]

            data[key['mmcore_name']] = list(buffer_components[v])
        if primitive.indices is not None:
            data['indices'] = list(buffer_components[primitive.indices])

        yield data
