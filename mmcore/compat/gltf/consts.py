from collections import namedtuple
from enum import Enum

import numpy as np

from mmcore import __version__
from mmcore.base.table import Index, Table, TableProxy


class MediaTypes(str, Enum):
    JSON = "model/gltf+json"
    GLB = "model/gltf-binary"


ALIGN = 4
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
BUFFER_DEFAULT_HEADER = "data:application/octet-stream;base64"
MESH_PRIM_KEYS = 'POSITION', 'NORMAL', 'TANGENT', 'TEXCOORD_n', 'COLOR_0', 'JOINTS_n', 'WEIGHTS_n', '_SPECIFIC', '_OBJECTID'
DEFAULT_ASSET = {
    "generator": f"mmcore@{__version__()}",
    "version": "2.0"

}


def _create_mesh_prim_data(prim_keys, name_mapping):
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


GLTFBufferDecoded = namedtuple('GLTFBufferDecoded', ['headers', 'buffer'])
GLTFAttribute = namedtuple('GLTFAttribute', ['name', 'value'])

typeTable = Table('const',
                  rows=[
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

                  ],
                  schema=dict(const=str, size=int),
                  name='gltf_target_type_table')

gltfTypeNames = list(typeTable.keys())

componentTypesTable = Table("const",
                            rows=[
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
                            ],
                            name='gltf_component_types_table',
                            schema={
                                "const": int,
                                "description": str,
                                "type": str
                            }
                            )

meshPrimitiveModeTable = Table("const",
                               rows=[
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
                               ],
                               name='gltf_mesh_primitive_mode_table',
                               schema={
                                   "const": int,
                                   "description": str,
                                   "type": str
                               }
                               )

componentTypeCodeTable = Table.from_lists(pk='gltf',
                                          rows=[('b', 1, componentTypesTable[5120], "Int8Array", np.int8, int),
                                                ('B', 1, componentTypesTable[5121], "Uint8Array", np.uint8, int),
                                                ('h', 2, componentTypesTable[5122], "Int16Array", np.int16, int),
                                                ('H', 2, componentTypesTable[5123], "Uint16Array", np.uint16, int),
                                                ('L', 4, componentTypesTable[5125], "Uint32Array", np.uint32, int),
                                                ('f', 4, componentTypesTable[5126], "Float32Array", np.float32, float)],
                                          name='gltf_typecode_table',
                                          schema={
                                              'typecode': str,
                                              'size': int,
                                              'gltf': TableProxy,
                                              'js': str,
                                              'numpy': type,
                                              'py': type
                                          }
                                          )
print(componentTypeCodeTable.keys())
componentTypeCodeTableNp = Table.from_lists(pk='numpy',
                                            rows=[('b', 1, componentTypesTable[5120], "Int8Array", np.int8, int),
                                                  ('B', 1, componentTypesTable[5121], "Uint8Array", np.uint8, int),
                                                  ('h', 2, componentTypesTable[5122], "Int16Array", np.int16, int),
                                                  ('H', 2, componentTypesTable[5123], "Uint16Array", np.uint16, int),
                                                  ('L', 4, componentTypesTable[5125], "Uint32Array", np.uint32, int),
                                                  ('f', 4, componentTypesTable[5126], "Float32Array", np.float32,
                                                   float)],
                                            name='numpy_typecode_table',
                                            schema={
                                                'typecode': str,
                                                'size': int,
                                                'gltf': TableProxy,
                                                'js': str,
                                                'numpy': type,
                                                'py': type
                                            }
                                            )

TYPE_TABLE = typeTable

attrmap = dict(
    POSITION='vertices',
    NORMAL='normals',
    TANGENT='tangent',
    TEXCOORD='uv',
    COLOR_0='colors',
    JOINTS='joints',
    WEIGHTS='weights',
    _OBJECTID='_objectid'
)

attrmap2 = dict(
    POSITION='position',
    NORMAL='normal',
    TANGENT='tangent',
    TEXCOORD='uv',
    COLOR_0='color',
    JOINTS='joints',
    WEIGHTS='weights',
    _OBJECTID='_objectid'
)
attrTable = Table.from_lists(pk="mmcore", rows=[
    ["position", 'vertices', 'POSITION', 'VEC3'],
    ["normal", 'normals', 'NORMAL', 'VEC3'],
    ["tangent", None, 'TANGENT', 'VEC3'],
    ["uv", 'uv', 'TEXCOORD_0', 'VEC2'],
    ["color", None, 'COLOR_0', 'VEC3'],
    ["joints", None, 'JOINTS_0', 'SCALAR'],
    ["weights", None, 'WEIGHTS_0', 'SCALAR'],
    ["_objectid", "_objectid", "_OBJECTID", "SCALAR"]
], schema=dict(mmcore=str, meshdata=str, gltf=str, gltf_type=str))

attrmap_ext = dict(
    position='POSITION',
    normal='NORMAL',
    tangent='TANGENT',
    uv='TEXCOORD',
    color='COLOR_0',
    joints='JOINTS',
    weights='WEIGHTS',
    _objectid="_OBJECTID"
)
typeAttrTable = dict(
    POSITION='vertices',
    NORMAL='normals',
    TANGENT='tangent',
    TEXCOORD='uv',
    COLOR_0='colors',
    JOINTS='joints',
    WEIGHTS='weights',
    _OBJECTID='_objectid'
)

typeAttrTable3 = dict(
    POSITION='vertices',
    NORMAL='normals',
    TANGENT='tangent',
    TEXCOORD='uv',
    COLOR_0='colors',
    JOINTS='joints',
    WEIGHTS='weights',
    _OBJECTID='_objectid'
)
typeTargetsMap = Index.fromdict({
    'SCALAR': ARRAY_BUFFER,
    'VEC2': ARRAY_BUFFER,
    'VEC3': ARRAY_BUFFER,
    'VEC4': ARRAY_BUFFER,
    'MAT2': ARRAY_BUFFER,
    'MAT3': ARRAY_BUFFER,
    'MAT4': ARRAY_BUFFER
})
componentAttrMap = Index.fromdict({
    attrTable["position"]: componentTypeCodeTable[5126],
    attrTable["normal"]: componentTypeCodeTable[5126],
    attrTable["tangent"]: componentTypeCodeTable[5126],
    attrTable["uv"]: componentTypeCodeTable[5126],
    attrTable["color"]: componentTypeCodeTable[5126],
    attrTable["joints"]: componentTypeCodeTable[5126],
    attrTable["weights"]: componentTypeCodeTable[5126],
    attrTable["_objectid"]: componentTypesTable[5123]

})
meshPrimitiveAttrTable = Table(
    'gltf_name',
    rows=list(_create_mesh_prim_data(prim_keys=MESH_PRIM_KEYS, name_mapping=attrmap)),
    name="meshPrimitiveAttrTable",
    schema={'gltf_name': str, 'mmcore_name': str, 'collection': bool, 'specific': bool}
)
