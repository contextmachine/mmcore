import typing
from collections import namedtuple

import numpy as np
from msgspec import Struct, field

from mmcore.base.geom import MeshData
from mmcore.geom.shapes.shape import shape_earcut

Vec3Union = tuple[float, float, float]
Vec2Union = tuple[float, float]
Vec4Union = tuple[float, float, float, float]


class StructComponent(Struct, tag=True):
    def todict(self):
        dct = dict()
        for k in self.__struct_fields__:
            v = getattr(self, k)
            if v is not None:
                dct[k] = v
        return dct


class MeshAttributes(Struct, tag=True):
    position: list[Vec3Union]
    normal: typing.Optional[list[Vec3Union]] = None
    uv: typing.Optional[list[Vec2Union]] = None
    color: typing.Optional[list[Vec3Union]] = None


    def __add__(self, other):
        dct = dict()
        for k in self.__struct_fields__:
            _a, _b = getattr(self, k, None), getattr(other, k, None)

            if _a and _b:
                dct[k] = _a + _b
        return MeshAttributes(**dct)

    def __iadd__(self, other):

        for k in self.__struct_fields__:
            _a, _b = getattr(self, k, None), getattr(other, k, None)

            if _a and _b:

                setattr(other, k, _a + _b)
            elif all([_b is None, _a is not None]):
                raise AttributeError("To merge attributes of different sets, use the standard __add__ . {k}")


class MeshPart(Struct, tag=True):
    attributes: MeshAttributes
    indices: typing.Optional[list[int]] = None
    material: typing.Optional[int] = None
    mode: int = 4
    extras: typing.Optional[dict] = field(default_factory=dict)
    mode: 4

    def __add__(self, other: 'MeshPart'):
        return MeshPart(attributes=self.attributes + other.attributes,
                        indices=np.concatenate([self.indices, (other.indices + np.max(self.indices))]),
                        material=self.material,
                        mode=self.mode,
                        extras=self.extras)

    def __iadd__(self, other: 'MeshPart'):
        self.attributes = self.attributes + other.attributes
        self.indices = np.concatenate([self.indices, (other.indices + np.max(self.indices))])


class Mesh(Struct, tag=True):
    primitives: typing.List[MeshPart] = []
    name: typing.Optional[str] = None
    extras: typing.Optional[dict] = field(default_factory=dict)

    def __add__(self, other: 'Mesh'):
        return Mesh(primitives=self.primitives + other.primitives, name=self.name, extras=self.extras)

    def __iadd__(self, other: 'Mesh'):
        self.primitives.extend(other.primitives)


class MeshAttrsD(typing.TypedDict):
    position: typing.Optional[list[Vec3Union]]
    normals: typing.Optional[list[Vec3Union]]
    uv: typing.Optional[list[Vec2Union]]
    colors: typing.Optional[list[Vec2Union]]


def add_mesh_attrs(self, other: dict[str, list]) -> 'MeshAttrsD':
    dct: MeshAttrsD = dict()
    for k in self.keys() & other.keys():

        a, b = self.get(k), other.get(k)
        if a and b:
            dct[k] = a + b
    return dct


def iadd_mesh_attrs(self, other: dict[str, list]):
    for k in self.keys() & other.keys():

        a, b = self.get(k), other.get(k)
        if a and b:
            self[k] = a + b
        elif all([a is not None, b is None]):
            raise KeyError("To merge attributes of different sets, use the standard __add__ . {k}")


class MeshPrimitiveD(typing.TypedDict):
    attributes: MeshAttrsD
    indices: typing.Optional[list[int]]
    material: typing.Optional[int]


def add_mesh_prims(self, other: 'MeshPrimitiveD'):
    o = MeshPrimitiveD(**self)
    if 'indices' in self.keys() and 'indices' in other.keys():

        o.update(attributes=add_mesh_attrs(self['attributes'], other['attributes']),
                 indices=np.concatenate([self['indices'], (other['indices'] + np.max(self['indices']))]))
    else:
        o.update(attributes=add_mesh_attrs(self['attributes'], other['attributes']))
    return o


def iadd_mesh_prims(self, other: 'MeshPrimitiveD'):
    iadd_mesh_attrs(self['attributes'], other['attributes'])

    if 'indices' in self.keys() and 'indices' in other.keys():
        self['indices'] = np.concatenate([self['indices'], (other['indices'] + np.max(self['indices']))])


class MeshD(typing.TypedDict):
    primitives: list[MeshPrimitiveD]
    name: typing.Optional[str]
    extras: typing.Optional[dict]


def add_meshes(self, other: 'MeshD'):
    return MeshD(primitives=self['primitives'] + other['primitives'])


def iadd_meshes(self, other: 'MeshD'):
    self['primitives'].extend(other['primitives'])


def mesh_prim_to_md(mesh_prim):
    return MeshData(vertices=mesh_prim['attributes'].get('position'), indices=mesh_prim.get('indices'),
                    normals=mesh_prim['attributes'].get('normal'), uv=mesh_prim['attributes'].get('uv'))


def mesh_to_md(mesh):
    p = mesh['primitives'][0]
    if len(mesh['primitives']) > 1:
        for pp in mesh['primitives'][1:]:
            iadd_mesh_prims(p, pp)
    return mesh_prim_to_md(p)


MeshTuple = namedtuple('MeshTuple', ['attributes', 'indices', 'extras'], defaults=[dict(), None, dict()])


def colorize_mesh(mesh: MeshTuple, color: tuple[float, float, float]):
    mesh.attributes['color'] = np.tile(color, len(mesh.attributes['position']) // 3)


def create_mesh_tuple(attributes, indices=None, color=(0.5, 0.5, 0.5)):
    if indices:
        m = MeshTuple(attributes, np.array(indices, dtype=int), {'parts': np.zeros(len(indices), dtype=int)})
    else:
        m = MeshTuple(attributes, None, {})
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

def mesh_from_shapes(shps, cols):
    for shp, c in zip(shps, cols):

        pos, ixs, _ = shape_earcut(shp)

        yield create_mesh_tuple(dict(position=pos), ixs, c)


def gen_indices_and_extras(meshes, ks):
    max_index = -1

    for j, m in enumerate(meshes):
        ixs = None
        if m.indices is not None:

            ixs = m.indices + max_index + 1
            face_cnt = len(m.indices) // 3
            max_index = np.max(ixs)

            yield *tuple(m.attributes[k] for k in ks), ixs, np.repeat(j, face_cnt)
        else:
            try:
                yield *tuple(m.attributes[k] for k in ks), ixs, None
            except Exception as err:
                print(m, err)

def union_mesh(meshes, ks=('position',)):
    *zz, = zip(*gen_indices_and_extras(meshes, ks))
    try:
        if zz[-2][0] is not None:
            return MeshTuple({ks[j]: np.concatenate(k) for j, k in enumerate(zz[:len(ks)])},
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