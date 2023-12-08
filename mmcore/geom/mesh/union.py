from mmcore.geom.mesh.consts import EXTRACT_MESH_ATTRS_PERFORMANCE_METHOD_LIMIT, MESH_OBJECT_ATTRIBUTE_NAME

from mmcore.geom.mesh.mesh_tuple import *


def union_mesh(meshes, extras=None):
    """
    Union multiple meshes into a single mesh.
    :param meshes: A list of mesh objects to be unioned.
    :type meshes: list
    :param extras: Additional attributes for the resulting mesh. (default is None)
    :type extras: dict
    :return: The unioned mesh object.
    :rtype: MeshTuple
    """
    extras = dict() if extras is None else extras
    attribute_names = _get_attribute_names(meshes)

    # Generate indices and extra attributes for all meshes
    indices_and_extras = list(zip(*gen_indices_and_extras2(meshes, names=attribute_names)))

    attributes = _combine_attributes(indices_and_extras, attribute_names)
    indices = _get_indices(indices_and_extras)
    children = _get_children(indices_and_extras)

    extras_with_children = {**extras, **dict(children=children)}

    return MeshTuple(attributes=attributes, indices=indices, extras=extras_with_children)


def _get_attribute_names(meshes):
    """Get attribute names from meshes, ensuring to include the MESH_OBJECT_ATTRIBUTE_NAME."""
    names = extract_mesh_attrs_union_keys(meshes)
    if MESH_OBJECT_ATTRIBUTE_NAME not in names:
        names = names + (MESH_OBJECT_ATTRIBUTE_NAME,)

    return names


def _combine_attributes(indices_and_extras, attribute_names):
    """
    Concatenate attributes from indices_and_extras based on attribute_names.
    This results in a dictionary mapping attribute names to concatenated attribute values.
    """
    return {
        attribute_names[i]: np.concatenate(values)
        for i, values in enumerate(indices_and_extras[:len(attribute_names)])
    }


def _get_indices(indices_and_extras):
    """Get concatenated indices from indices_and_extras or None if not present."""
    last_values = indices_and_extras[-2]
    return np.concatenate(last_values) if last_values[0] is not None else None


def _get_children(indices_and_extras):
    """Get children (last values) from indices_and_extras."""
    return indices_and_extras[-1]


def sum_meshes(a: MeshTuple, b: MeshTuple):
    return union_mesh([a, b])


MeshTuple.__add__ = sum_meshes


def extract_mesh_attrs_union_keys_with_counter(meshes):
    return sorted(list(Counter([tuple(mesh.attributes.keys()) for mesh in meshes]).keys()))[0]


def extract_mesh_attrs_union_keys_with_set(meshes):
    return set.intersection(*(set(mesh.attributes.keys()) for mesh in meshes))


def extract_mesh_attrs_union_keys(meshes):
    if len(meshes) <= EXTRACT_MESH_ATTRS_PERFORMANCE_METHOD_LIMIT:
        return tuple(extract_mesh_attrs_union_keys_with_set(meshes))
    return tuple(extract_mesh_attrs_union_keys_with_counter(meshes))


def gen_indices_and_extras2(meshes, names):
    """
    Generate indices and extras for a list of meshes.

    :param meshes: List of Mesh objects.
    :type meshes: List[Mesh]
    :param names: List of attribute names to extract from each Mesh object's attributes dictionary.
    :type names: List[str]
    :return: Generator that yields tuples containing extracted attribute values, indices, and extras for each Mesh object.
    :rtype: Generator[Tuple[Any, ...], None, None]
    """
    max_index = -1
    for j, m in enumerate(meshes):

        length = len(m.attributes['position'])
        m.attributes[MESH_OBJECT_ATTRIBUTE_NAME] = np.repeat(j, length // 3)

        if m.indices is not None:

            ixs = m.indices + max_index + 1

            max_index = np.max(ixs)

            yield *tuple(m.attributes[name] for name in names), ixs, m.extras
        else:
            yield *tuple(m.attributes[name] for name in names), None, m.extras


def union_mesh2(meshes, extras=None):
    if extras is None:
        extras = dict()
    names = extract_mesh_attrs_union_keys(meshes)
    if MESH_OBJECT_ATTRIBUTE_NAME not in names:
        names = names + (MESH_OBJECT_ATTRIBUTE_NAME,)
    *zz, = zip(*gen_indices_and_extras2(meshes,
                                        names=names))

    return MeshTuple(attributes={names[j]: np.concatenate(k) for j, k in enumerate(zz[:len(names)])},
                     indices=np.concatenate(zz[-2]) if zz[-2][0] is not None else None,
                     extras=extras | dict(children=zz[-1]))
