from mmcore.geom.mesh.compat import *
from mmcore.geom.mesh.consts import *
from mmcore.geom.mesh.union import *

from mmcore.geom.mesh.mesh_tuple import extract_material
import uuid as _uuid
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


'''
def extract_material_from_color(color, flatShading=True):
    col = ColorRGB(*np.average(
        color.reshape(
            (len(color) // 3, 3)
        ),
        axis=0
    )).decimal

    return amatdict.get(f'{col}-mesh',
                        MeshPhongMaterial(uuid=f'{col}-mesh', color=col, side=2, flatShading=flatShading))


def build_cmesh_with_buffer(msh: dict,
                            uuid=None,
                            name: str = "Mesh",
                            material=simpleMaterial,
                            props=None, controls=None,
                            **kwargs):
    """
    Builds a mesh with buffer.

    :param mesh: The mesh object.
    :type mesh: <class 'Mesh'>
    :param uuid: The UUID of the mesh. Defaults to None.
    :type uuid: Optional[str]
    :param name: The name of the mesh. Defaults to "Mesh".
    :type name: str
    :param material: The material of the mesh. Defaults to simpleMaterial.
    :type material: <class 'Material'>
    :param props: The properties of the mesh. Defaults to None.
    :type props: Optional[dict]
    :param controls: The controls of the mesh. Defaults to None.
    :type controls: Optional[Any]
    :param kwargs: Additional keyword arguments.
    :type kwargs: Any
    :return: The mesh with buffer.
    :rtype: <class 'AMesh'>
    """

    if uuid is None:
        uuid = _uuid.uuid4().hex
    index = None if msh['indices'] is None else msh['indices'].tolist()
    if props is None:
        props = {}

    m = AMesh(uuid=uuid,
              name=name,
              geometry=create_mesh_buffer(uuid + 'geom',
                                          **{k: attr.tolist() for k, attr in msh['attributes'].items()},
                                          index=index.tolist() if isinstance(index, np.ndarray) else index
                                          ),
              material=material,
              properties=props,
              controls=controls,
              **kwargs)

    return m


def amesh(self: Mesh, uuid=None, name="Mesh", material=None, flatShading=True, props=dict(), controls=dict()):
    if material is None:
        material = extract_material_from_color(self.get_color(), flatShading=flatShading)
    return build_cmesh_with_buffer(self.asdict(), uuid=uuid, name=name, material=material, props=props,
                                   controls=controls)
'''
