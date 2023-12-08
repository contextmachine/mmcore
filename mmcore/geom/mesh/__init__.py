from mmcore.geom.mesh.compat import *
from mmcore.geom.mesh.consts import *
from mmcore.geom.mesh.union import *


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
