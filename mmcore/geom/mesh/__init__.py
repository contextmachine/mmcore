from mmcore.geom.mesh.union import *
from mmcore.geom.mesh.consts import *
from mmcore.geom.mesh.compat import *

def gen_indices_and_extras(meshes, ks):
    max_index = -1

    for j, m in enumerate(meshes):

        if m.indices is not None:

            ixs = m.indices + max_index + 1
            face_cnt = len(m.indices) // 3
            max_index = np.max(ixs)

            yield *tuple(m.attributes[k] for k in ks), ixs, np.repeat(j, face_cnt)
        else:
            try:
                yield *tuple(m.attributes[k] for k in ks), None, None
            except Exception as err:
                print(m, err)


def union_mesh_old(meshes, ks=('position',)):
    *zz, = zip(*gen_indices_and_extras(meshes, ks))
    try:
        if zz[-2][0] is not None:
            return create_mesh_tuple({ks[j]: np.concatenate(k) for j, k in enumerate(zz[:len(ks)])},
                                     np.concatenate(zz[-2]),
                                     extras=dict(
                                         parts=np.concatenate(zz[-1])))
        else:
            return MeshTuple({ks[j]: np.concatenate(k) for j, k in enumerate(zz[:len(ks)])},
                             None,
                             extras={})
    except IndexError:
        return MeshTuple({ks[j]: np.concatenate(k) for j, k in enumerate(zz[:len(ks)])},
                         None,
                         extras={})


def mesh_comparison(a: MeshTuple, b: MeshTuple) -> tuple[bool, list]:
    """
    Compare two mesh tuples.

    :param a: First mesh to compare.
    :type a: MeshTuple
    :param b: Second mesh to compare.
    :type b: MeshTuple
    :return: A tuple containing a boolean indicating whether the meshes are equal and a list of differences.
    :rtype: tuple[bool, list]
    """
    rr = []
    for k in a.attributes.keys():
        if not np.allclose(a.attributes[k], b.attributes[k]):
            rr.append((k, a.attributes[k], b.attributes[k]))
    if not np.allclose(a.indices, b.indices):
        rr.append(('indices', a.indices, b.indices))

    if len(rr) > 0:
        return False, rr


    else:
        return True, []
