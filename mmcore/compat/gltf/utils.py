import operator
import struct
from functools import reduce

import numpy as np
from more_itertools import chunked

from mmcore.base.geom import MeshData
from mmcore.base.table import Table
from mmcore.compat.gltf.components import *
from mmcore.compat.gltf.components import GLTFAccessor
from mmcore.compat.gltf.consts import MESH_PRIM_KEYS, TYPE_TABLE, _create_mesh_prim_data, attrmap, \
    attrmap2, componentTypeCodeTable, \
    meshPrimitiveAttrTable
from mmcore.compat.gltf.consts import typeTargetsMap


def finalize_gltf_buffer(data: tuple, typ_size: int):
    if typ_size == 1:
        return data
    else:
        return chunked(data, typ_size)


def finalize_gltf_buffer_np(data: np.ndarray, typ_size: int):
    if typ_size == 1:
        return data
    else:
        _l = len(data)
        return data.reshape((_l // typ_size, typ_size))
def gltf_decode_buffer(buffers):
    _prefixes = []
    _buffers = []
    for buffer in buffers:
        pref, buff = buffer.uri.split(",")
        _buffers.append(buff)
        _prefixes.append(pref)
    return _prefixes, _buffers


def array_flatlen(arr):
    if isinstance(arr, (list, tuple)):
        arr = np.array(arr)
    return int(functools.reduce(operator.mul, arr.shape))


def byte_stride(type: str, componentType: int):
    return TYPE_TABLE[type]['size'] * componentTypeCodeTable[componentType]['size']


def struct_fmt(data, dtype: int):
    return f"{array_flatlen(data)}{componentTypeCodeTable[dtype]['typecode']}"


def packer(buffer, data, dtype=5126, offset=0):
    res = np.array(data, dtype=componentTypeCodeTable[dtype]['numpy']).flatten()
    fmt = struct_fmt(data, dtype)
    struct.pack_into(fmt, buffer, offset, *res)


def pack(data, dtype=5126):
    res = np.array(data, dtype=componentTypeCodeTable[dtype]['numpy']).flatten()

    return res.tobytes()


def addBufferView(arr, buffer: bytearray, gltf_type="VEC3", dtype=5126, offset=0, name=None):
    flatview = np.array(arr, dtype=componentTypeCodeTable[dtype]['numpy']).flatten()

    struct.pack_into(struct_fmt(flatview, dtype), buffer, offset, flatview)
    return {
        "buffer": 0,
        "byteLength": len(flatview) * componentTypeCodeTable[dtype]['size'],
        "byteOffset": offset,
        "byteStride": byte_stride(gltf_type, dtype),
        "name": name,
        "target": typeTargetsMap[gltf_type]
    }


def appendBufferView(arr, buffer: bytearray, gltf_type="VEC3", dtype=5126, name=None, use_stride=False):
    flatview = np.array(arr, dtype=componentTypeCodeTable[dtype]['numpy']).flatten()

    res = {
        "buffer": 0,
        "byteLength": len(flatview) * componentTypeCodeTable[dtype]['size'],
        "byteOffset": len(buffer),

        "name": name,
        "target": typeTargetsMap[gltf_type][0]
    }
    if all([use_stride, (gltf_type != 'SCALAR')]):
        res["byteStride"] = byte_stride(gltf_type, dtype)
    buffer.extend(flatview.tobytes())
    return res


def appendBufferView2(view, length, buffer: bytearray, gltf_type="VEC3", dtype=5126, name=None, use_stride=False):
    # flatview = np.array(arr, dtype=componentTypeCodeTable[dtype]['numpy']).flatten()

    res = {
        "buffer": 0,
        "byteLength": length,
        "byteOffset": len(buffer),

        "name": name,
        "target": typeTargetsMap[gltf_type][0]
    }
    if all([use_stride, (gltf_type != 'SCALAR')]):
        res["byteStride"] = byte_stride(gltf_type, dtype)
    buffer.extend(view)
    return res



# componentTypeTable = {comp_type['const']: GLTFComponentType(**comp_type) for comp_type in componentTypes}


def create_doc(data):
    return GLTFDocument(bufferViews=[GLTFBufferView(**buffview) for buffview in data['bufferViews']],
                        buffers=[GLTFBuffer(**buffer) for buffer in data['buffers']],
                        accessors=[GLTFAccessor(**accessor) for accessor in data['accessors']],
                        meshes=[GLTFMesh.from_gltf(mesh) for mesh in data['meshes']])


def component_to_buf():
    ...


def gltf_buffer_to_data(buffers: list[str],
                        accessors: list[GLTFAccessor],
                        bufferViews: list[GLTFBufferView],
                        type_table: Table = TYPE_TABLE,
                        component_type_table: Table = componentTypeCodeTable):
    bts_list = [base64.b64decode(buffer) for buffer in buffers]

    for accessor, bufferview in zip(accessors, bufferViews):
        dtype = type_table[accessor.type]
        cmp_type = component_type_table[accessor.componentType]
        bts = bts_list[bufferview.buffer]
        try:
            yield finalize_gltf_buffer(
                np.frombuffer(bts[bufferview.byteOffset:bufferview.byteOffset + bufferview.byteLength],
                              dtype=cmp_type['numpy']), dtype['size'])
            # ), dtype['size'])

        except struct.error as e:
            print(accessor, bufferview, cmp_type, dtype, f"{accessor.count * dtype['size']}{cmp_type['typecode']}",
                  sep='\n')
            # raise e


def accessor_to_data(accessor: GLTFAccessor, doc: GLTFDocument):
    if hasattr(accessor, 'bufferView'):
        bufferView = doc.bufferViews[accessor.bufferView]
        dtype = TYPE_TABLE[accessor.type]
        arr = buffer_to_array(accessor, bufferView, doc.buffers[bufferView.buffer])
        return finalize_gltf_buffer_np(arr, dtype['size'])
    else:
        raise "Accessor without buffview"


def extract_accessors_data(doc: GLTFDocument):
    for accessor in doc.accessors:
        yield accessor_to_data(accessor, doc)


from mmcore.geom.mesh import MeshTuple, union_mesh


def remap_keys(dct, keymap, default='MISSING_KEY'):
    return {keymap.get(k, default): dct[k] for k in dct.keys()}


def mesh_attrs(dct, keymap, accessors_data):
    return {keymap[k]: accessors_data[dct[k]].tolist() for k in dct.keys()}


def extract_meshes(doc: GLTFDocument, attr_mapping=attrmap2):

    *data, = extract_accessors_data(doc)
    for mesh in doc.meshes:
        prims = []

        for mp in mesh.primitives:
            dct = mp.todict()

            # dct['attributes']=remap_keys(dct['attributes'],keymap=attr_mapping)
            dct['attributes'] = mesh_attrs(dct['attributes'], keymap=attr_mapping, accessors_data=data)
            if 'indices' in dct:
                dct['indices'] = data[dct['indices']]

            prims.append(MeshTuple(**dct, extras=dct.get('extras', None)))

        yield union_mesh(prims, extras=mesh.extras)



def extract_meshes_dict(doc: GLTFDocument, accessors, attr_mapping=attrmap2, **kwargs):
    data = accessors
    for mesh in doc.meshes:
        prims = []
        for mp in mesh.primitives:
            dct = mp.todict()

            # dct['attributes']=remap_keys(dct['attributes'],keymap=attr_mapping)
            dct['attributes'] = mesh_attrs(dct['attributes'], keymap=attr_mapping, accessors_data=data)
            if 'indices' in dct:
                dct['indices'] = data[dct['indices']]

            prims.append(dct)

        yield dict(primitives=prims, **kwargs)


class ExNode:
    def __init__(self, children=(), mesh=None, name=None, table=None, **kwargs):
        self._children = children
        self._mesh = mesh
        self._name = name
        self._table = table
        self.kwargs = kwargs

    @property
    def mesh(self):
        return self._table['meshes'][self._mesh]

    @mesh.setter
    def mesh(self, v):
        self._table['meshes'] = v

    @property
    def children(self):
        return [self._table['nodes'][ch] for ch in self._children]


def extract_nodes(doc: GLTFDocument):
    *accessors, = extract_accessors_data(doc)
    *meshes, = extract_meshes_dict(doc, accessors)
    tab = {
        'accessors': accessors,
        'nodes': [],
        'meshes': meshes
    }
    for node in doc.nodes:
        tab['nodes'].append(node.todict())
    return tab


def buffer_to_array(accessor: GLTFAccessor, bufferView: GLTFBufferView, buffer: GLTFBuffer):
    head, bts = buffer.decode()
    cmp_type = componentTypeCodeTable[accessor.componentType]
    return np.frombuffer(bts[bufferView.byteOffset:bufferView.byteOffset + bufferView.byteLength],
                         dtype=cmp_type['numpy'])


def fmt(count, dtype: int, componentType: str):
    f"{count * TYPE_TABLE[dtype]['size']}{componentTypeCodeTable[componentType]['typecode']}"


def accessor_type(accessor: GLTFAccessor, type_table: Table = TYPE_TABLE):
    dtype = type_table[accessor.type]
    size = accessor.count * dtype['size']


def buffer_format(accessor: GLTFAccessor, bufferView: GLTFBufferView, type_table: Table = TYPE_TABLE) -> str:
    if bufferView.byteStride is None:
        bufferView.byteStride = accessor.count * accessor.type.size
    return f"{bufferView.byteStride}{accessor.type.format}"


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


gltfdocs = dict()
mesh_data_attrs = Table(
    'gltf_name',
    rows=list(_create_mesh_prim_data(prim_keys=MESH_PRIM_KEYS, name_mapping=attrmap)),
    name="meshPrimitiveAttrTable",
    schema={'gltf_name': str, 'mmcore_name': str, 'collection': bool, 'specific': bool}
)
mmattrs = Table(
    'gltf_name',
    rows=list(_create_mesh_prim_data(prim_keys=MESH_PRIM_KEYS, name_mapping=attrmap2)),
    name="meshPrimitiveAttrTable",
    schema={'gltf_name': str, 'mmcore_name': str, 'collection': bool, 'specific': bool}
)


def buffer_to_mesh_dicts(doc, attribute_table: Table = mmattrs):
    h, bb = gltf_decode_buffer(doc.buffers)
    buff_data = list(gltf_buffer_to_data(bb, doc.accessors, doc.bufferViews))
    for mesh in doc.meshes:
        yield list(gltf_mesh_primitive_table(mesh.primitives, buff_data, attribute_table))


def buffer_to_meshdata(doc):
    h, bb = gltf_decode_buffer(doc.buffers)
    buff_data = list(gltf_buffer_to_data(bb, doc.accessors, doc.bufferViews))
    for mesh in doc.meshes:
        yield reduce(MeshData.merge2,
                     [MeshData(**m) for m in gltf_mesh_primitive_table(mesh.primitives, buff_data, mesh_data_attrs)])
