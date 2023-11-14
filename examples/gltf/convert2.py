import numpy as np

from mmcore.compat.gltf.convert import create_union_mesh_node, create_scene_from_meshes, DEFAULT_MATERIAL_COMPONENT, \
    SceneNode, asscene, Mesh, MeshPart
import time
import requests
import ujson
from mmcore.geom.mesh import union_mesh, mesh_from_shapes
from mmcore.geom.shapes import ShapeInterface


def _gen(shapes, names, mask, tags):
    cols = {'A-0': (0.5, 0.5, 0.5)}
    for i, t in enumerate(shapes):
        if mask[i] != 2:
            name = names[i]
            if name in tags:
                tag = tags[name]
                if tag not in cols:
                    # print(tag)

                    cols[tag] = tuple(np.random.random(3))
                col = cols[tag]


            else:
                col = (0.5, 0.5, 0.5)

            for tt in t:
                if len(tt) > 3:

                    if tt[0] == tt[-1]:
                        tt = tt[:-1]
                    yield ShapeInterface(tt), col
                elif len(tt) == 3:

                    yield ShapeInterface(tt), col


def case2(parts=["w1", "w2", "w3", "w4", 'l2', 'f1', 'f2', 'f3', 'f5', 'sl1', 'sl3', 'sl1b', 'sl2b', 'sl3b', 'sl4b'],
          random_mat=False):
    # print(parts, "random mterial:", random_mat)
    # s = time.time()

    # with open('test_data.json', 'r') as f:
    #    _ = ujson.load(f)
    #    coldata = _['coldata']
    #    ress = _['ress']
    resp = requests.post("https://viewer.contextmachine.online/cxm/api/v2/mfb_contour_server/sw/contours-merged",
                         json=dict(names=parts))
    resp2 = requests.get("https://viewer.contextmachine.online/cxm/api/v2/mfb_sw_w/stats"
                         )
    resp3 = requests.get("https://viewer.contextmachine.online/cxm/api/v2/mfb_sw_l2/stats"
                         )
    resp4 = requests.get("https://viewer.contextmachine.online/cxm/api/v2/mfb_sw_f/stats"
                         )
    coldata = {rr['name']: rr['tag'] for rr in resp2.json() + resp3.json() + resp4.json()}
    mats = {"A-0": DEFAULT_MATERIAL_COMPONENT}
    cols = {'A-0': (0.5, 0.5, 0.5)}
    ress = resp.json()
    # print("request", divmod(time.time() - s, 60))
    parts = []
    s = time.time()

    shps, cols = list(zip(*_gen(ress['shapes'], ress['names'], ress['mask'], coldata)))
    print(shps[:3], cols[:3])
    m = union_mesh(mesh_from_shapes(shps, cols), ks=['position', 'color'])

    print("creating parts", divmod(time.time() - s, 60))
    s = time.time()

    node_test = create_union_mesh_node(m, name='test_mesh')

    print("create nodes", divmod(time.time() - s, 60))
    s = time.time()

    scene2 = asscene(node_test)
    print("create scene", divmod(time.time() - s, 60))
    s = time.time()
    scene_dct = scene2.todict()
    print("create dict", divmod(time.time() - s, 60))
    s = time.time()
    with open('testsw3.gltf', 'w') as f:
        ujson.dump(scene_dct, f, indent=2)
    print("dump json", divmod(time.time() - s, 60))


# case2()
def create_union(
        parts=["w1", "w2", "w3", "w4", 'l2', 'f1', 'f2', 'f3', 'f5', 'sl1', 'sl3', 'sl1b', 'sl2b', 'sl3b', 'sl4b'],
        random_mat=False):
    # print(parts, "random mterial:", random_mat)
    # s = time.time()

    # with open('test_data.json', 'r') as f:
    #    _ = ujson.load(f)
    #    coldata = _['coldata']
    #    ress = _['ress']
    resp = requests.post("https://viewer.contextmachine.online/cxm/api/v2/mfb_contour_server/sw/contours-merged",
                         json=dict(names=parts))
    resp2 = requests.get("https://viewer.contextmachine.online/cxm/api/v2/mfb_sw_w/stats"
                         )
    resp3 = requests.get("https://viewer.contextmachine.online/cxm/api/v2/mfb_sw_l2/stats"
                         )
    resp4 = requests.get("https://viewer.contextmachine.online/cxm/api/v2/mfb_sw_f/stats"
                         )
    coldata = {rr['name']: rr['tag'] for rr in resp2.json() + resp3.json() + resp4.json()}

    ress = resp.json()
    # print("request", divmod(time.time() - s, 60))

    shps, cols = list(zip(*_gen(ress['shapes'], ress['names'], ress['mask'], coldata)))

    return union_mesh(mesh_from_shapes(shps, cols), ks=['position', 'color'])
