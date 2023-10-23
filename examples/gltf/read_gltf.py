import os

import sys

from mmcore.base import AGroup
from mmcore.compat.gltf import GLTFDocument, extract_meshes, \
    extract_meshes_dict

sys.path.extend(f"{os.getcwd()}/examples/data")
with open('Угол стены.gltf') as f:
    import json

    gltfdata = json.load(f)

doc = GLTFDocument.from_gltf(gltfdata)
import time

s = time.time()
*meshes1, = extract_meshes(doc)
print(divmod(time.time() - s, 60))
s = time.time()

*meshes2, = extract_meshes_dict(doc)
print(divmod(time.time() - s, 60))
group = AGroup(uuid='test_gltf2')
