from collections import Counter, namedtuple
from functools import lru_cache

import numpy as np

from mmcore.base import amatdict
from mmcore.base.models.gql import MeshPhongMaterial
from mmcore.geom.materials import ColorRGB
from mmcore.geom.mesh.compat import build_mesh_with_buffer
from mmcore.geom.mesh.consts import DEFAULT_COLOR, MESH_OBJECT_ATTRIBUTE_NAME

_MeshTuple = namedtuple('MeshTuple', ['attributes', 'indices', 'extras'], defaults=[dict(), None, dict()])


class MeshTuple(_MeshTuple):
    """
    The `MeshTuple` class extends the `_MeshTuple` class and provides a way to represent a mesh with various attributes.

    Methods:
        - `__hash__(self)`
            - Computes the hash value of the position attribute converted to bytes.
            - Returns the computed hash value.

        - `amesh(self, uuid=None, name="Mesh", material=None, flatShading=True, props=dict(), controls=dict())`
            - Creates a mesh with the given parameters.
            - Parameters:
                - `uuid` (optional): The unique identifier for the mesh.
                - `name` (optional): The name of the mesh.
                - `material` (optional): The material for the mesh.
                - `flatShading` (optional): Flag indicating whether to use flat shading.
                - `props` (optional): Dictionary of additional properties for the mesh.
                - `controls` (optional): Dictionary of controls for the mesh.
            - Returns:
                - The created mesh.
    """
    def __hash__(self):
        return hash(self.attributes['position'].tobytes())

    def amesh(self, uuid=None, name="Mesh", material=None, flatShading=True, props=dict(), controls=dict()):
        if material is None:
            material = extract_material(self, flatShading=flatShading)
        return build_mesh_with_buffer(self, uuid=uuid, name=name, material=material, props=props, controls=controls)


def apply(mesh, data):
    for name, val in data.items():
        if name == 'attributes':
            for k, v in val.items():
                apply_attribute(mesh, k, v)
        elif name == 'extras':
            pass
        else:

            obj = getattr(mesh, name)
            obj[:] = val


def is_union(mesh):
    return MESH_OBJECT_ATTRIBUTE_NAME in mesh.attributes


def explode(mesh: MeshTuple):
    if MESH_OBJECT_ATTRIBUTE_NAME in mesh.attributes:
        objects = mesh.attributes[MESH_OBJECT_ATTRIBUTE_NAME]
        names = list(mesh.attributes.keys())
        names.remove(MESH_OBJECT_ATTRIBUTE_NAME)
        oc = Counter(objects)

        obj = 0

        for k, i in enumerate(objects):
            if i > obj:
                cnt = oc[i]
                yield MeshTuple({name: mesh.attributes[name][k:k + cnt * 3] for name in names},
                                None, mesh.extras['children'][i])
                obj += 1
    else:
        raise ValueError('Mesh is not a union!')


def create_mesh_tuple(attributes, indices=None, color=DEFAULT_COLOR, extras: dict = None):
    """

    :param attributes: The attributes of the mesh.
    :type attributes: Any

    :param indices: The indices of the mesh. (Optional)
    :type indices: list or None

    :param color: The color of the mesh. (Default: DEFAULT_COLOR)
    :type color: Any

    :param extras: The extra data for the mesh. (Optional)
    :type extras: dict or None

    :return: The created mesh tuple.
    :rtype: MeshTuple

    """
    if extras is None:
        extras = dict()
    if indices is not None:
        m = MeshTuple(attributes, np.array(indices, dtype=int), extras)
    else:
        m = MeshTuple(attributes, None, extras)
    colorize_mesh(m, color)
    return m


def extract_material(mesh: MeshTuple, flatShading=True):
    col = ColorRGB(*np.average(
        mesh.attributes['color'].reshape(
            (len(mesh.attributes['color']) // 3, 3)
        ),
        axis=0
    )).decimal

    return amatdict.get(f'{col}-mesh',
                        MeshPhongMaterial(uuid=f'{col}-mesh', color=col, side=2, flatShading=flatShading))


def union_extras(mesh: MeshTuple, other: MeshTuple):
    """
    :param mesh: The main mesh tuple
    :type mesh: MeshTuple
    :param other: The other mesh tuple to be combined with the main mesh
    :type other: MeshTuple
    :return: None
    :rtype: None

    This function combines the extras of two mesh tuples by updating the 'children' key in the extras dictionary of the main mesh tuple. The behavior of the combination depends on whether
    * the mesh tuples are unioned or not.

    - If both mesh tuples are unioned (is_union(mesh) = 1 and is_union(other) = 1), the 'children' lists of the extras dictionaries are merged.
    - If only the main mesh tuple is unioned (is_union(mesh) = 1 and is_union(other) = 0), the 'children' list of the extras dictionary of the main mesh tuple is appended with the 'children
    *' list from the extras dictionary of the other mesh tuple.
    - If only the other mesh tuple is unioned (is_union(mesh) = 0 and is_union(other) = 1), the 'children' key in the extras dictionary of the main mesh tuple is updated with a new list
    * containing a copy of the extras dictionary of the main mesh tuple and the extras dictionary of the other mesh tuple.
    - If neither mesh tuple is unioned (is_union(mesh) = 0 and is_union(other) = 0), the 'children' key in the extras dictionary of the main mesh tuple is updated with a new list containing
    * a copy of the extras dictionary of the main mesh tuple and the extras dictionary of the other mesh tuple.

    Note: The is_union() function is not provided here but is assumed to be implemented separately.

    Example usage:
        mesh_tuple = MeshTuple()
        other_mesh_tuple = MeshTuple()
        union_extras(mesh_tuple, other_mesh_tuple)
    """
    def one():
        mesh.extras['children'] = [dict(**mesh.extras), other.extras]

    def two():
        mesh.extras['children'] = [mesh.extras] + other.extras['children']

    def three():
        mesh.extras['children'].extend(other.extras['children'])

    def four():
        mesh.extras['children'].append(other.extras['children'])

    cases = {
        (0, 0): one,
        (0, 1): two,
        (1, 1): three,
        (1, 0): four
    }
    cases[(is_union(mesh), is_union(other))]()


def apply_attribute(mesh, name, val):
    """
    Applies an attribute to a mesh object.

    :param mesh: The mesh object.
    :type mesh: Mesh
    :param name: The name of the attribute to apply.
    :type name: str
    :param val: The value to assign to the attribute.
    :type val: list or array
    :return: None
    :rtype: None
    """
    obj = getattr(mesh, name)
    obj.resize(len(val), refcheck=False)
    obj[:] = val


def colorize_mesh(mesh: MeshTuple, color: tuple[float, float, float]):
    """
    Set the color attribute of the mesh to the specified color.

    :param mesh: The mesh object.
    :type mesh: MeshTuple
    :param color: The color to assign to the mesh.
    :type color: tuple[float, float, float]
    :return: None
    :rtype: None
    """
    mesh.attributes['color'] = np.tile(color, len(mesh.attributes['position']) // 3)
