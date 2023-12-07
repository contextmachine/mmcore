import numpy as np

from mmcore.geom.mesh import MESH_OBJECT_ATTRIBUTE_NAME, MeshTuple, extract_mesh_attrs_union_keys, \
    gen_indices_and_extras2


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
