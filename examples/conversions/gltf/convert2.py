import numpy as np
import requests
import time
import ujson

from mmcore.base import AGroup
from mmcore.compat.gltf.convert import (DEFAULT_MATERIAL_COMPONENT, asscene, create_union_mesh_node,
    )
from mmcore.geom.mesh import (build_mesh_with_buffer, union_mesh, union_mesh2, union_mesh_simple, vertexMaterial,
    )
from mmcore.geom.mesh.shape_mesh import mesh_from_shapes
from mmcore.geom.shapes import ShapeInterface


def check_last(pts):
    if np.allclose(pts[0], pts[-1]):
        return check_last(pts[:-1])
    else:
        return pts


def _gen(shapes, names, mask, tags, stats_data):
    cols = {"A-0": (0.5, 0.5, 0.5)}
    for i, t in enumerate(shapes):
        name = names[i]

        if (mask[i] != 2) and (name in stats_data):
            if name in tags:
                tag = tags[name]
                if tag not in cols:
                    # print(tag)

                    cols[tag] = tuple(np.random.random(3))
                col = cols[tag]

            else:
                col = (0.5, 0.5, 0.5)

            for tt in t:
                yield ShapeInterface(check_last(tt)), col, stats_data[name]


def case2(parts=["w1", "w2", "w3", "w4", "l2"], random_mat=False):
    # print(parts, "random mterial:", random_mat)
    s = time.time()

    # with open('test_data.json', 'r') as f:
    #    _ = ujson.load(f)
    #    coldata = _['coldata']
    #    ress = _['ress']
    resp = requests.post("https://viewer.contextmachine.online/cxm/api/v2/mfb_contour_server/sw/contours-merged",
            json=dict(names=parts), )
    resp2 = requests.get("https://viewer.contextmachine.online/cxm/api/v2/mfb_sw_w/stats"
            )
    resp3 = requests.get("https://viewer.contextmachine.online/cxm/api/v2/mfb_sw_l2/stats"
            )
    coldata = {rr["name"]: rr["tag"] for rr in resp2.json() + resp3.json()}
    mats = {"A-0": DEFAULT_MATERIAL_COMPONENT}
    cols = {"A-0": (0.5, 0.5, 0.5)}
    ress = resp.json()
    print("request", divmod(time.time() - s, 60))
    parts = []
    s = time.time()

    shps, cols = list(zip(*_gen(ress["shapes"], ress["names"], ress["mask"], coldata)))
    print(shps[:3], cols[:3])
    m = union_mesh_old(mesh_from_shapes(shps, cols), ks=["position", "color"])

    print("creating parts", divmod(time.time() - s, 60))
    s = time.time()

    node_test = create_union_mesh_node(m, name="test_mesh")

    print("create nodes", divmod(time.time() - s, 60))
    s = time.time()

    scene2 = asscene(node_test)
    print("create scene", divmod(time.time() - s, 60))
    s = time.time()
    scene_dct = scene2.todict()
    print("create dict", divmod(time.time() - s, 60))
    s = time.time()
    with open("testsw3.gltf", "w") as f:
        ujson.dump(scene_dct, f, indent=2)
    print("dump json", divmod(time.time() - s, 60))


# case2()
def create_union(parts=["w1", "w2", "w3", "w4", "w0", "l2"], random_mat=False):
    # print(parts, "random mterial:", random_mat)
    s = time.time()

    # with open('test_data.json', 'r') as f:
    #    _ = ujson.load(f)
    #    coldata = _['coldata']
    #    ress = _['ress']
    resp = requests.post("https://viewer.contextmachine.online/cxm/api/v2/mfb_contour_server/ne/contours-merged",
            json=dict(names=parts), )
    resp2 = requests.get("https://viewer.contextmachine.online/cxm/api/v2/mfb_ne_w/stats"
            )
    resp3 = requests.get("https://viewer.contextmachine.online/cxm/api/v2/mfb_ne_l2/stats"
            )

    stats = resp2.json() + resp3.json()

    coldata = {rr["name"]: rr["tag"] for rr in stats}
    print()
    ress = resp.json()
    print("request", divmod(time.time() - s, 60))

    shps, cols = list(zip(*_gen(ress["shapes"], ress["names"], ress["mask"], coldata)))
    (*meshes,) = mesh_from_shapes(shps, cols, stats)
    s = time.time()
    RES = union_mesh_old(meshes, ks=["position", "color"])
    print("union", divmod(time.time() - s, 60))
    return RES


containers = ["mfb_ne_w", "mfb_ne_l2", "mfb_sw_w", "mfb_sw_l2", "mfb_sw_f"]
stats_scopes = ["mfb_ne_w", "mfb_ne_l2", "mfb_sw_w", "mfb_sw_l2", "mfb_sw_f"]
blocks = ["sw", "ne"]


def get_parts(conts):
    for c in conts:
        yield from requests.get(f"https://viewer.contextmachine.online/cxm/api/v2/{c}/zone-scopes"
                ).json()


def get_stats(conts):
    for c in conts:
        yield from requests.get(f"https://viewer.contextmachine.online/cxm/api/v2/{c}/stats"
                ).json()


def create_union2(parts=None, random_mat=False):
    # print(parts, "random mterial:", random_mat)
    if parts is None:
        parts = ["w1", "w2", "w3", "w4", "l2"]
    s = time.time()

    # with open('test_data.json', 'r') as f:
    #    _ = ujson.load(f)
    #    coldata = _['coldata']
    #    ress = _['ress']

    containers_ne = ["mfb_ne_w", "mfb_ne_l2"]
    # r1 = requests.post(f"https://viewer.contextmachine.online/cxm/api/v2/mfb_contour_server/sw/contours-merged",
    #                   json=dict(names=list(get_parts(containers_sw)))).json()
    r2 = requests.post(f"https://viewer.contextmachine.online/cxm/api/v2/mfb_contour_server/ne/contours-merged",
            json=dict(names=list(get_parts(containers_ne))), ).json()
    # for k in list(r1.keys()):
    #    r1[k].extend(r2[k])
    resp = r2

    (*stats,) = get_stats(containers_ne)
    coldata = {rr["name"]: rr["tag"] for rr in stats}
    stats_data = {rr["name"]: rr for rr in stats}
    print()
    ress = resp
    print("request", divmod(time.time() - s, 60))

    shps, cols, prop = list(zip(*_gen(ress["shapes"], ress["names"], ress["mask"], coldata, stats_data))
            )
    (*meshes,) = mesh_from_shapes(shps, cols, prop)
    s = time.time()
    RES = union_mesh(meshes)
    print("union", divmod(time.time() - s, 60))
    return RES, meshes, shps


m1, m2, shps = create_union2()

s = time.time()

amesh = build_mesh_with_buffer(m1, "test-union-mesh", material=vertexMaterial)
amesh.scale(0.001, 0.001, 0.001)
amesh.translate((84, -85, 0))
amesh.dump("testsw.json")
print("create dict", divmod(time.time() - s, 60))
node_test = create_union_mesh_node(m1)
s = time.time()

scene2 = asscene(node_test)
print("create scene", divmod(time.time() - s, 60))
s = time.time()
scene_dct = scene2.todict()
with open("testsw.gltf", "w") as f:
    ujson.dump(scene_dct, f)

print("dump json", divmod(time.time() - s, 60))
