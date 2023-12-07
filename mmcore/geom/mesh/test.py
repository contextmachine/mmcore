import numpy as np
import time

from mmcore.geom.mesh import union
from mmcore.geom.mesh.union import union_mesh
from mmcore.geom.rectangle import Rectangle, to_mesh


def test_union_mesh():
    r1 = Rectangle(10, 20)
    r2 = Rectangle(40, 10, xaxis=np.array([0., 1., 0.]), normal=np.array([0., 0., 1.]))

    r2.translate([1., 1., 1.])
    meshes = [to_mesh(r1), to_mesh(r2)]
    s = time.time()
    a = union_mesh(meshes)
    print(divmod(time.time() - s, 60))
    s = time.time()
    b = union.union_mesh(meshes)
    print(divmod(time.time() - s, 60))
    rr = []
    for k in a.attributes.keys():
        if not np.allclose(a.attributes[k], b.attributes[k]):
            rr.append((k, a.attributes[k], b.attributes[k]))
        if not np.allclose(a.indices, b.indices):
            rr.append(('indices', a.indices, b.indices))

    if len(rr) > 0:
        print("FAIL")
        print(rr)
    else:
        print("Done")


if __name__ == '__main__':
    test_union_mesh()
