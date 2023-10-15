from mmcore.base import AGroup
from mmcore.base.geom import MeshData
from mmcore.compat.gltf import create_doc, gltf_buffer_to_data, gltf_decode_buffer, gltf_mesh_primitive_table

with open('scene-7.gltf') as f:
    import json

    gltfdata = json.load(f)
doc = create_doc(gltfdata)
h, bb = gltf_decode_buffer(doc.buffers)
buff_data = list(gltf_buffer_to_data(bb, doc.accessors, doc.bufferViews))
mds = []
for mesh in doc.meshes:
    *meshprimtable, = gltf_mesh_primitive_table(mesh.primitives, buff_data)

    mds.extend([MeshData(**m) for m in meshprimtable])

group = AGroup(uuid='test_gltf')
for md in mds:
    group.add(md.to_mesh())

group.dump('testgltf.json')
