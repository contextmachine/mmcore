from collections import Counter, namedtuple
from functools import lru_cache

import numpy as np

from mmcore.base import amatdict
from mmcore.base.models.gql import MeshPhongMaterial
from mmcore.geom.materials import ColorRGB
from mmcore.geom.mesh.consts import DEFAULT_COLOR, MESH_OBJECT_ATTRIBUTE_NAME
from mmcore.geom.mesh.compat import build_mesh_with_buffer

_MeshTuple = namedtuple('MeshTuple', ['attributes', 'indices', 'extras'], defaults=[dict(), None, dict()])


class MeshTuple(_MeshTuple):

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
    if extras is None:
        extras = dict()
    if indices is not None:
        m = MeshTuple(attributes, np.array(indices, dtype=int), extras)
    else:
        m = MeshTuple(attributes, None, extras)
    colorize_mesh(m, color)
    return m


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


def colorize_mesh(mesh: MeshTuple, color: tuple[float, float, float]):
    mesh.attributes['color'] = np.tile(color, len(mesh.attributes['position']) // 3)
