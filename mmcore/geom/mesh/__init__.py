import typing
import uuid
from collections import Counter, namedtuple

import numpy as np
from msgspec import Struct, field

from mmcore.base import AMesh
from mmcore.base.geom import MeshData
from mmcore.base.models.gql import BufferGeometry, MeshPhongMaterial, create_buffer_index, create_buffer_position, \
    create_buffer_uv, \
    create_float32_buffer
from mmcore.geom.materials import ColorRGB
from mmcore.geom.shapes.shape import shape_earcut
import typing
import uuid
from collections import Counter, namedtuple

import numpy as np
from msgspec import Struct, field

from mmcore.base import AMesh
from mmcore.base.geom import MeshData
from mmcore.base.models.gql import BufferGeometry, MeshPhongMaterial, create_buffer_index, create_buffer_position, \
    create_buffer_uv, \
    create_float32_buffer
from mmcore.geom.materials import ColorRGB
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


DEFAULT_COLOR = (0.5, 0.5, 0.5)

_MeshTuple = namedtuple('MeshTuple', ['attributes', 'indices', 'extras'], defaults=[dict(), None, dict()])


class MeshTuple(_MeshTuple):

    def __add__(self, other):
        a, b = [tuple(explode(mesh)) if is_union(mesh) else (mesh,) for mesh in (self, other)]

        return union_mesh2(a + b)


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


def mesh_from_shapes(shps, cols, stats):
    for shp, c, props in zip(shps, cols, stats):
        pos, ixs, _ = shape_earcut(shp)

        yield create_mesh_tuple(dict(position=pos), ixs, c, extras=dict(properties=props))


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


def union_mesh(meshes, ks=('position',)):
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


def union_mesh2(meshes, extras=None):
    if extras is None:
        extras = dict()
    names = extract_mesh_attrs_union_keys(meshes) + (MESH_OBJECT_ATTRIBUTE_NAME,)
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
        attra['_objectid'] = create_buffer_objectid(index)
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


vertexMaterial = MeshPhongMaterial(uuid='vxmat', color=ColorRGB(200, 200, 200).decimal, vertexColors=True, side=2)
simpleMaterial = MeshPhongMaterial(uuid='vxmat', color=ColorRGB(200, 200, 200).decimal, side=2)


def build_mesh_with_buffer(mesh: MeshTuple, name: str = "Mesh", material=simpleMaterial):
    uid = uuid.uuid4().hex
    index = None if mesh.indices is None else mesh.indices.tolist()
    a = AMesh(uuid=uid + 'mesh',
              name=name,
              geometry=create_mesh_buffer(uid, **{k: attr.tolist() for k, attr in mesh.attributes.items()},
                                          index=index))

    # for k,v in mesh.extras.items():

    #    a.add_userdata_item(k, v)
    return a
