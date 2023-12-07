
import typing
import uuid as _uuid
from collections import Counter, namedtuple
from functools import lru_cache

import numpy as np

from mmcore.base import AMesh
from mmcore.base.models.gql import BufferGeometry, MeshPhongMaterial, create_buffer_index, \
    create_buffer_position, \
    create_buffer_uv, \
    create_float32_buffer
from mmcore.base.registry import amatdict
from mmcore.geom.materials import ColorRGB

Vec3Union = tuple[float, float, float]
Vec2Union = tuple[float, float]
Vec4Union = tuple[float, float, float, float]
vertexMaterial = MeshPhongMaterial(uuid='vxmat', color=ColorRGB(200, 200, 200).decimal, vertexColors=True, side=2)
simpleMaterial = MeshPhongMaterial(uuid='vxmat', color=ColorRGB(200, 200, 200).decimal, side=2)



DEFAULT_COLOR = (0.5, 0.5, 0.5)

_MeshTuple = namedtuple('MeshTuple', ['attributes', 'indices', 'extras'], defaults=[dict(), None, dict()])


class MeshTuple(_MeshTuple):

    def __add__(self, other):
        a, b = [tuple(explode(mesh)) if is_union(mesh) else (mesh,) for mesh in (self, other)]

        return union_mesh(a + b)

    def __hash__(self):
        return hash(self.attributes['position'].tobytes())

    def amesh(self, uuid=None, name="Mesh", material=None, flatShading=True, props=dict(), controls=dict()):
        if material is None:
            material = extract_material(self, flatShading=flatShading)
        return build_mesh_with_buffer(self, uuid=uuid, name=name, material=material, props=props, controls=controls)

@lru_cache()
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
    obj = getattr(mesh, name)
    obj.resize(len(val), refcheck=False)
    obj[:] = val


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


def colorize_mesh(mesh: MeshTuple, color: tuple[float, float, float]):

    mesh.attributes['color'] = np.tile(color, len(mesh.attributes['position']) // 3)


def create_mesh_tuple(attributes, indices=None, color=DEFAULT_COLOR, extras: dict = None):
    if extras is None:
        extras = dict()
    if indices is not None:
        m = MeshTuple(attributes, np.array(indices, dtype=int), extras)
    else:
        m = MeshTuple(attributes, None, extras)
    colorize_mesh(m, color)
    return m


def sum_meshes(a: MeshTuple, b: MeshTuple):
    attributes = dict()
    for attr_name in a.attributes.keys():
        attributes[attr_name] = np.append(np.array(a.attributes[attr_name]), b.attributes[attr_name])
    if a.indices:

        return MeshTuple(attributes,
                         np.append(np.array(a.indices), np.array(b.indices) + (1 + np.max(a.indices))),
                         {'parts': np.append(a.extras['parts'], a.extras['parts'][-1] + 1 + b.extras['parts'])})
    else:
        return MeshTuple(attributes,
                         None,
                         {})




MESH_OBJECT_ATTRIBUTE_NAME = '_objectid'


def extract_mesh_attrs_union_keys_with_counter(meshes):
    return sorted(list(Counter([tuple(mesh.attributes.keys()) for mesh in meshes]).keys()))[0]


def extract_mesh_attrs_union_keys_with_set(meshes):
    return set.intersection(*(set(mesh.attributes.keys()) for mesh in meshes))


EXTRACT_MESH_ATTRS_PERFORMANCE_METHOD_LIMIT = 500


def extract_mesh_attrs_union_keys(meshes):
    if len(meshes) <= EXTRACT_MESH_ATTRS_PERFORMANCE_METHOD_LIMIT:
        return tuple(extract_mesh_attrs_union_keys_with_set(meshes))
    return tuple(extract_mesh_attrs_union_keys_with_counter(meshes))


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


def gen_indices_and_extras2(meshes, names):
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


def create_buffer_objectid(array):
    return {
        'type': 'Uint16Array',
        "itemSize": 1,
        "array": array
    }


def create_mesh_buffer(
        uuid,
        position=None,
        uv=None,
        index=None,
        normal=None,
        _objectid=None,
        color: typing.Optional[list[float]] = None, threejs_type="BufferGeometry"):
    attra = dict(position=create_buffer_position(position))
    if color is not None:
        attra['color'] = create_float32_buffer(color)
    if normal is not None:
        attra['normal'] = create_float32_buffer(normal)
    if uv is not None:
        attra['uv'] = create_buffer_uv(uv)

    if _objectid is not None:
        attra['_objectid'] = create_buffer_objectid(_objectid)
    if index is not None:
        ixs = create_buffer_index(index)
        return BufferGeometry(**{
            "uuid": uuid,
            "type": threejs_type,
            "data": {
                "attributes": attra,
                "index": ixs

            }
        })

    else:
        return BufferGeometry(**{
            "uuid": uuid,
            "type": threejs_type,
            "data": {
                "attributes": attra

            }
        })


def build_mesh_with_buffer(mesh: MeshTuple,
                           uuid=None,
                           name: str = "Mesh",
                           material=simpleMaterial,
                           props=dict(),
                           controls=dict(), **kwargs):
    if uuid is None:
        uuid = _uuid.uuid4().hex
    index = None if mesh.indices is None else mesh.indices.tolist()
    return AMesh(uuid=uuid,
                 name=name,
                 geometry=create_mesh_buffer(uuid + 'geom',
                                             **{k: attr.tolist() for k, attr in mesh.attributes.items()},
                                             index=index
                                             ),
                 material=material,
                 properties=mesh.extras | props,
                 controls=controls,
                 **kwargs)

    # for k,v in mesh.extras.items():

    #    a.add_userdata_item(k, v)
