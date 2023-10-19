"""Grasshopper Script"""
import base64
from functools import reduce

from mmcore.base.geom import MeshData
from mmcore.base.table import Table
from mmcore.compat.gltf.components import *
from mmcore.compat.gltf.consts import GLTFBufferDecoded, MESH_PRIM_KEYS, TYPE_TABLE, attrmap, componentTypeCodeTable, \
    meshPrimitiveAttrTable
from mmcore.compat.gltf.utils import finalize_gltf_buffer, gltf_decode_buffer

a = "Hello Python 3 in Grasshopper!"
print(a)

import struct


# componentTypeTable = {comp_type['const']: GLTFComponentType(**comp_type) for comp_type in componentTypes}


def create_doc(data):
    return GLTFDocument(bufferViews=[GLTFBufferView(**buffview) for buffview in data['bufferViews']],
                        buffers=[GLTFBuffer(**buffer) for buffer in data['buffers']],
                        accessors=[GLTFAccessor(**accessor) for accessor in data['accessors']],
                        meshes=[GLTFMesh.from_gltf(mesh) for mesh in data['meshes']])


def component_to_buf():
    ...


import io


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
        yield finalize_gltf_buffer(
            struct.unpack(f"{accessor.count * dtype['size']}{cmp_type['typecode']}",
                          bts[bufferview.byteOffset:bufferview.byteOffset + bufferview.byteLength]), dtype['size'])


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


from mmcore.compat.gltf.consts import componentAttrMap, attrTable

gltfdocs = dict()


def buffer_to_mesh_dicts(mesh, doc):
    h, bb = gltf_decode_buffer(doc.buffers)
    buff_data = list(gltf_buffer_to_data(bb, doc.accessors, doc.bufferViews))

    yield list(gltf_mesh_primitive_table(mesh.primitives, buff_data))


def buffer_to_meshes(doc):
    h, bb = gltf_decode_buffer(doc.buffers)
    buff_data = list(gltf_buffer_to_data(bb, doc.accessors, doc.bufferViews))
    for mesh in doc.meshes:
        yield reduce(MeshData.merge2, [MeshData(**m) for m in gltf_mesh_primitive_table(mesh.primitives, buff_data)])
