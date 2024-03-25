from mmcore.geom.mesh.consts import EXTRACT_MESH_ATTRS_PERFORMANCE_METHOD_LIMIT

from mmcore.geom.mesh.mesh_tuple import *


def union_mesh(meshes, extras=None, keys=None):
    """

    Unions multiple meshes into a single mesh (a process known as "mesh fusion"). This is especially
    useful when preparing large 3D models for rendering in a web browser as it requires only a single
    draw call.

    Parameters
    ----------
    meshes : list
        A list of Mesh objects that are to be unified into a single Mesh object.
    extras : dict, (optional)
        Additional attribute fields for the resulting Mesh object. If not specified, an empty dictionary,
        will be used as a default.

    Returns
    -------
    Mesh namedtuple
        The object consisting of combined attributes, indices and extras from input meshes.

    Performance Considerations
    --------------------------
    The main function `union_mesh` under the hood calls `gen_indices_and_extras2` and `_combine_attributes`
    functions. The `gen_indices_and_extras2` function traverses through each mesh's attributes resulting in
    time complexity of O(m*n), where m is the number of meshes and n is the attribute count of a mesh.
    The `_combine_attributes` function concatenates same attribute fields of all meshes, thus it has linear
    time complexity of O(n), where n is the count of attribute names.

    Consequently, the overall time complexity of the union operation of meshes is O(m*n),
    where m is the number of meshes and n is the count of attribute names in a mesh.

    Usage Example
    -------------

    >>> from mmcore.geom.mesh.union import union_mesh
    >>> from mmcore.compat.gltf.convert import  create_union_mesh_node, asscene
    >>> import time
    >>> import ujson
    >>> import pickle

    >>> # Load test data (a collection of 18140 Mesh objects)
    >>> with open("tests/data/test_union_gltf_data.pkl", 'rb') as f:
    >>>     meshes = pickle.load(f)

    >>> # Union the list of meshes into a single mesh
    >>> single_mesh = union_mesh(meshes)

    >>> # Create a GLTF SceneNode from the fused mesh
    >>> scene=asscene(create_union_mesh_node(single_mesh,"mesh_node"))
    >>>  with open('single_mesh.gltf', 'w') as f: # dump gltf file
    >>>     ujson.dump(scene.todict(), f)
    """

    extras = dict() if extras is None else extras
    attribute_names = keys if keys else _get_attribute_names(meshes)

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
    return sorted(list(Counter([tuple(mesh[0].keys()) for mesh in meshes]).keys()))[0]


def extract_mesh_attrs_union_keys_with_set(meshes):
    return set.intersection(*(set(mesh[0].keys()) for mesh in meshes))


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

        length = len(m[0]['position'])
        m[0][MESH_OBJECT_ATTRIBUTE_NAME] = np.repeat(j, length // 3)

        if m[1] is not None:

            ixs = m[1] + max_index + 1

            max_index = np.max(ixs)

            yield *tuple(m[0][name] for name in names), ixs, m[2]
        else:
            yield *tuple(m[0][name] for name in names), None, m[2]


def union_mesh2(meshes, extras=None, keys=None):
    if extras is None:
        extras = dict()
    names = keys if keys else extract_mesh_attrs_union_keys(meshes)
    if MESH_OBJECT_ATTRIBUTE_NAME not in names:
        names = names + (MESH_OBJECT_ATTRIBUTE_NAME,)
    *zz, = zip(*gen_indices_and_extras2(meshes,
                                        names=names))

    return MeshTuple(attributes={names[j]: np.concatenate(k) for j, k in enumerate(zz[:len(names)])},
                     indices=np.concatenate(zz[-2]) if zz[-2][0] is not None else None,
                     extras=extras | dict(children=zz[-1]))


def gen_indices_and_extras(meshes, ks):
    max_index = -1

    for j, m in enumerate(meshes):

        if m[1] is not None:

            ixs = m[1] + max_index + 1
            face_cnt = len(m[1]) // 3
            max_index = np.max(ixs)

            yield *tuple(m[0][k] for k in ks), ixs, np.repeat(j, face_cnt)
        else:
            try:
                yield *tuple(m[0][k] for k in ks), None, None
            except Exception as err:
                print(m, err)


def union_mesh_simple(meshes):

    attribute_names = _get_attribute_names(meshes)

    # Generate indices and extra attributes for all meshes
    indices_and_extras = list(zip(*gen_indices_and_extras2(meshes, names=attribute_names)))

    attributes = _combine_attributes(indices_and_extras, attribute_names)
    indices = _get_indices(indices_and_extras)

    return MeshTuple(attributes=attributes, indices=indices, extras={})
