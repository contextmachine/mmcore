import copy
import dataclasses
import hashlib
import itertools
import json
import operator
import typing
import uuid
from collections import namedtuple

import numpy as np
import strawberry
import ujson
from pyquaternion import Quaternion
from scipy.spatial.distance import euclidean
from strawberry.scalars import JSON

import mmcore.base.models.gql as gql_models
from mmcore import typegen
from mmcore.base.params import ParamGraphNode, TermParamGraphNode
from mmcore.base.registry import adict, ageomdict, amatdict, idict
from mmcore.collections.multi_description import ElementSequence
from mmcore.geom.vectors import unit

NAMESPACE_MMCOREBASE = uuid.UUID('5901d0eb-61fb-4e8c-8fd3-a7ed8c7b3981')
Link = namedtuple("Link", ["name", "parent", "child"])
LOG_UUIDS = False


class ExEncoder(json.JSONEncoder):
    def default(self, ob):

        def df(o):

            if isinstance(o, set):
                return list(o)
            elif hasattr(o, "_type_definition"):
                return strawberry.asdict(o)
            elif dataclasses.is_dataclass(o):
                return df(dataclasses.asdict(o))
            elif isinstance(o, np.int64):
                return int(o)
            else:
                return o

        try:
            return json.JSONEncoder.default(self, ob)
        except TypeError:
            return df(ob)


def hasitemattr(attr):
    def wrp(obj):
        if isinstance(obj, dict):
            return attr in obj.keys()
        else:
            if hasattr(obj, attr):
                a = getattr(obj, attr)
                if a is not None:

                    if hasattr(a, "__len__") and not isinstance(a, str):
                        if len(a) == 0:
                            return False
                        return True
                    return True
                return True
            return False

    return wrp


def graph_from_json(data):
    a = []
    for d in data:
        cls = globals()["type"]
        a.append(cls(uuid=d["uuid"], dump_dict=d))
    return a


class GeometrySet(set):

    def __contains__(self, item):
        # TODO: Это нужно переписать на np.allclose по буфферу, т.e. проверять идентичность геометрии честно
        uuids = ElementSequence(list(self))["uuid"]

        return item.uuid in uuids


class MaterialSet(set):
    def __contains__(self, item):
        colors = ElementSequence(list(self))["color"]
        # TODO: Это нужно расширить и проверять не только по цвету,
        #  для этого нужно реально понять какие параметры важные
        return item.color in colors


parenthashmap = dict()


class UUIDMissPermException(AttributeError):
    ...


ShaSub = namedtuple("ShaSub", ["int", "hex"])


class GroupIterator(typing.Iterator):
    def __init__(self, seq: typing.Iterable = ()):
        self._seq = iter(seq)

    def __next__(self):
        return self._seq.__next__()


def getattr_(obj):
    baseic = 'uuid', '_children', '_parents', 'properties'
    psw = 'uuid', 'children', 'parents', 'properties'
    getter = operator.attrgetter(*baseic)
    dct = dict(zip(psw, getter(obj)))
    if hasattr(obj, '_geometry'):
        dct['geometry'] = obj._geometry
        dct['material'] = obj._material
    return dct


ROOT_DOC = """"""


@strawberry.interface(description=ROOT_DOC)
class RootInterface:
    metadata: gql_models.Metadata
    materials: gql_models.AnyMaterial
    geometries: gql_models.BufferGeometry
    object: gql_models.AnyObject3D

    # shapes: typing.Optional[JSON] = None

    @strawberry.field
    def all(self) -> JSON:
        return strawberry.asdict(self)


def gqlcb(x):
    x.field


class Delegate:
    def __init__(self, delegate):
        self._delegate = delegate

    def __call__(self, owner):
        self._owner = owner

        def _getattr_(inst, item):

            if hasattr(self._owner, item):

                return self._owner.__getattribute__(inst, item)

            else:

                return getattr(inst._ref, item)

        self._owner.__getattr__ = _getattr_
        d = set(dir(self._delegate))
        d1 = set(dir(self._owner))
        d.update(d1)

        def dr(dlf):
            r = object.__dir__(dlf._ref)
            rr = set(object.__dir__(dlf))
            rr.update(set(r))
            return rr

        self._owner.__dir__ = dr

        return self._owner


def iscollection(item):
    return isinstance(item, typing.MutableSequence) or isinstance(item, (tuple, set, frozenset))


DEFAULT_MATRIX = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def remove_empty(data, remove_empty_collections=True):
    if isinstance(data, dict):
        d = {}
        for k, v in data.items():
            res = remove_empty(v)
            if res is not None or not (res == []):
                d[k] = res

        return d
    elif iscollection(data):

        if len(data) == [] and remove_empty_collections:
            return None

        else:
            l = []
            for i, v in enumerate(data):
                res = remove_empty(v)
                if res is not None:
                    l.append(res)
            return l
    else:
        return data


def sumdicts(*dicts):
    d = dict(dicts[0])
    for dct in dicts[1:]:
        d |= dct
    return d


class ChildSet(set):
    def __init__(self, instance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instance = instance

    def add(self, element) -> None:
        d = idict[self.instance.uuid]
        idict[self.instance.uuid][f"child_{len(d)}"] = element.uuid
        super().add(element)


@dataclasses.dataclass
class UserDataItem:
    def todict(self):
        def tod(obj):
            dct = dict()
            for k, v in dataclasses.asdict(obj).items():
                if hasattr(v, "todict"):
                    dct[k] = v.todict()
                elif dataclasses.is_dataclass(v):
                    dct[k] = tod(v)
                elif isinstance(v, (list, tuple)):
                    dct[k] = [tod(i) for i in v]
                else:
                    dct[k] = v

            return dct

        return tod(self)

    def asdict(self):
        return self.todict()


@dataclasses.dataclass
class GuiConnectConfig:
    address: str = "http://localhost:7711"
    api_prefix: str = "/"

    def update(self, dct):
        self.__dict__ |= dct


@dataclasses.dataclass
class GuiPost(UserDataItem):
    target: dataclasses.InitVar[str]
    config: dataclasses.InitVar[GuiConnectConfig]
    endpoint: typing.Optional[str] = None

    def __post_init__(self, target: 'A', config):
        self.endpoint = f'{config.address}{config.api_prefix}{target}'


@dataclasses.dataclass
class GuiItem(UserDataItem):
    """
    self.dct={"gui": [
            {
                "type": "controls",
                "data":
                    obj.properties
                ,
                "post": {
                    "endpoint": "https://api.contextmachine.online:7771",

                }
            }
        ]}
    """
    target: dataclasses.InitVar[str]

    type: str
    data: typing.Optional[dict[str, typing.Any]] = None
    post: typing.Optional[GuiPost] = None
    config: dataclasses.InitVar[GuiConnectConfig] = GuiConnectConfig()

    def __post_init__(self, target, config):
        if self.post is None:
            self.post = GuiPost(target=target, config=config)
        if self.data is None:
            self.data = {}


class Controls:
    def __init__(self, config=None, default=None):
        self.default = {}
        if isinstance(default, dict):
            self.default |= default
        self.config = GuiConnectConfig()
        if isinstance(config, dict):
            self.config.update(config)

    def __set_name__(self, owner, name):
        self._name = "_" + name
        setattr(owner, self._name, None)
        setattr(owner, '__gui_controls__', self)

    def __get__(self, instance, own):
        if instance is not None:
            return GuiItem(type="controls",
                           target=instance._endpoint,
                           config=self.config,
                           data=getattr(instance, self._name, self.default),

                           )
        else:
            return self

    def __set__(self, instance, value):
        setattr(instance, self._name, value)


class A:
    idict = idict
    args_keys = ["name"]
    _uuid: str = "no-uuid"
    name: str = "Object"
    _state_keys = {
        "uuid",
        "name",
        "matrix"
    }
    priority = 1.0
    controls = Controls()
    properties_keys = {
        "priority",
        "name"

    }
    _matrix = list(DEFAULT_MATRIX)
    _include_geometries = GeometrySet()
    _include_materials = MaterialSet()
    _properties = dict()
    _controls = {}
    _endpoint = "/"
    RENDER_BBOX = False

    def __getstate__(self):
        dct = dict(self.__dict__)

        dct |= {

            "_self_idict": copy.deepcopy(self.idict[self.uuid])

        }
        return dct

    def __setstate__(self, state):
        _self_idict = state.pop("_self_idict")
        _self_uuid = state["_uuid"]

        adict[_self_uuid] = self
        idict[_self_uuid] = _self_idict

        for k, v in state.items():
            setattr(self, k, v)


    def __new__(cls, *args, **kwargs):
        inst = object.__new__(cls)
        if kwargs.get("uuid") is None:

            _ouuid = uuid.uuid4().hex
        else:
            _ouuid = kwargs.pop("uuid")

        adict[_ouuid] = inst
        idict[_ouuid] = dict()
        inst._uuid = _ouuid
        inst.child_keys = set()
        inst._children = ChildSet(inst)
        if "_endpoint" not in kwargs.keys():
            kwargs["_endpoint"] = _ouuid
        inst.set_state(*args, **kwargs)

        return inst

    def traverse_child_three(self):
        """
        get_child_three() is a function that takes an object and returns a threejs_root object with all of its children.

        Parameters:
            self (object): The object to be processed.

        Returns:
            threejs_root: A threejs_root object with all of the object's children.

        This function takes an object and creates a deep copy of its threejs_repr. It then adds all of the object's children to the threejs_root object. Finally, it binds the class to the threejs_root object and returns it.
        @return:
        """
        self._include_materials = MaterialSet()
        self._include_geometries = GeometrySet()

        def childthree(obj):

            self._add_includes(obj)
            dct = obj()
            if len(idict[obj.uuid]) > 0:
                dct['children'] = []
                for child in idict[obj.uuid].values():
                    dct['children'].append(childthree(child))

                return dct
            else:
                # ##print(dct)
                if 'children' in dct:
                    if len(dct.get('children')) == 0:
                        del dct['children']
                return dct

        return childthree(self)

    @property
    def state_keys(self):
        return set(list(self.args_keys + list(self.child_keys) + list(self._state_keys)))

    @state_keys.setter
    def state_keys(self, v):
        self._state_keys = set(v)

    @property
    def properties(self):

        return self._properties

    @properties.setter
    def properties(self, v):
        self._properties = v

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, v):
        self._matrix = v

    @matrix.deleter
    def matrix(self):
        self._matrix = list(DEFAULT_MATRIX)

    def gql_input_type(self):
        ds = typegen.dict_schema.DictSchema(self.properties)
        ds.get_init_default_strawberry()

    @property
    def threejs_type(self):
        return "Object3D"

    @property
    def uuid(self) -> str:
        return self._uuid

    @uuid.setter
    def uuid(self, v):

        val = adict[self._uuid]
        adict[str(v)] = val
        del adict[self._uuid]
        self._uuid = str(v)

    @classmethod
    def get_object(cls, uuid: str):
        return adict[uuid]

    @property
    def children(self):
        return [adict[v] for v in idict[self.uuid].values()]

    def render(self):
        l = []
        for w in self.children:
            l.append(w())
            self._add_includes(w)

    def get_state(self):
        dct = {}
        for k in self.state_keys:
            val = self.__getattr__(k)
            if isinstance(val, A):
                dct[k] = val.get_state()
            else:
                dct[k] = val
        return dct

    def set_state(self, *args, **kwargs):
        _kwargs = dict(zip(self.args_keys, args[:len(self.args_keys)]))
        _kwargs |= kwargs

        for k, v in _kwargs.items():
            self.__setattr__(k, v)
        # self.traverse_child_three()

    def root(self, shapes=None):
        geometries = set()
        materials = set()
        obj = self(materials=materials, geometries=geometries)
        data = {
            "metadata": {
                "version": 4.5,
                "type": "Object",
                "generator": "Object3D.toJSON"
            },
            "object": obj,
            "geometries": [strawberry.asdict(ageomdict[uid]) for uid in geometries],
            "materials": [strawberry.asdict(amatdict[uid]) for uid in materials]}
        if shapes is not None:
            data["shapes"] = "shapes"
        return data

    def __call__(self, *args, callback=lambda x: x, geometries: set = None, materials: set = None, **kwargs):
        self.set_state(*args, **kwargs)
        data = {
            "name": self.name,
            "uuid": self.uuid,
            "type": self.threejs_type,
            "layers": 1,
            "matrix": self.matrix,
            "castShadow": True,
            "receiveShadow": True,

            "userData": {

                "properties": sumdicts({
                    "name": self.name
                },
                    self.properties,
                ),
                "gui": [
                    self.controls.todict()
                ]

            }
        }

        if geometries is not None:
            if hasattr(self, "_geometry"):
                geometries.add(self._geometry)
                materials.add(self._material)
        if idict.get(self.uuid):
            if len(idict.get(self.uuid)) > 0:
                data["children"] = [ch(geometries=geometries, materials=materials) for ch in self.children]

        return data

    _parents = set()

    def __getattr__(self, key):

        if key in idict[self.uuid].keys():

            return adict[idict[self.uuid][key]]
        elif key == "value":
            return self

        else:
            return object.__getattribute__(self, key)

    def set_state_attr(self, key, value):
        self.__setattr__(key, value)
        self._state_keys.add(key)

    def __setattr__(self, key, v):

        if isinstance(v, A):

            idict[self.uuid][key] = v.uuid
        elif isinstance(v, (ParamGraphNode, TermParamGraphNode)):
            print("SETSTATE", key, v, v())
            self.__setattr__(key, v())
        else:

            object.__setattr__(self, key, v)

    def __delattr__(self, key):

        if key in idict[self.uuid]:
            del idict[self.uuid][key]
        else:
            object.__delattr__(self, key)

    def _add_includes(self, obj):
        if hasattr(obj, "_geometry"):
            if obj._geometry is not None:
                self._include_geometries.add(obj._geometry)
                self._include_materials.add(obj._material)

    def matrix_to_square_form(self) -> np.ndarray:
        return np.array(self.matrix, dtype=float).reshape((4, 4)).T

    def transform(self, matrix):
        """
        Этот метод применяет трансформацию к уже существующей матрице,
        если вы просто хотите заменить матрицу трансформации используйте `self.matrix = <matrix>`.
        @param matrix:
        @return:
        """
        self.matrix = (
                self.matrix_to_square_form() @ np.array(matrix).reshape((4, 4))).T.flatten().tolist()

    def rotate(self, angle: float, axis: tuple[float, float, float] = (0, 0, 1)):
        """

        @param axis:
        @param angle: radians
        @return:
        """
        q = Quaternion(axis=unit(axis), angle=angle)
        self.transform(q.transformation_matrix)

    def translate(self, vector: tuple[float, float, float]):
        matrix = np.array([[1, 0, 0, vector[0]],
                           [0, 1, 0, vector[1]],
                           [0, 0, 1, vector[2]],
                           [0, 0, 0, 1]], dtype=float)
        self.transform(matrix)

    def scale(self, x: float = 1, y: float = 1, z: float = 1):
        matrix = np.array([[x, 0, 0, 0],
                           [0, y, 0, 0],
                           [0, 0, z, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.transform(matrix)

    def reset_transform(self):
        self.matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    def dispose(self):

        for k, obj, in adict.items():
            v = idict[k]
            if self.uuid in v.values():
                obj.__delattr__(list(v.keys())[list(v.values()).index(self.uuid)])
        del idict[self.uuid]
        adict.__delitem__(self.uuid)

    def dump(self, filename):
        with open(filename, "w") as f:
            ujson.dump(self.root(), f)

    def dispose_with_children(self):

        for child in self.children:
            adict[child.uuid].dispose_with_children()

        self.dispose()

    def props_update(self, uuids: list[str], props: dict):

        ...

class AGroup(A):

    def __new__(cls, seq=(), **kwargs):
        inst = super().__new__(cls, **kwargs)

        idict[inst.uuid]["__children__"] = set()
        if len(seq) > 0:
            idict[inst.uuid]["__children__"] = set([s.uuid for s in seq])

        return inst

    @property
    def threejs_type(self):
        return "Group"

    def add(self, obj):
        idict[self.uuid]["__children__"].add(obj.uuid)

    def update(self, items):
        set.update(idict[self.uuid]["__children__"], set(s.uuid for s in items))

    def union(self, items):
        set.union(idict[self.uuid]["__children__"], set(s.uuid for s in items))

    def __contains__(self, item):
        return set.__contains__(idict[self.uuid]["__children__"], item.uuid)

    def __iter__(self):
        return iter(idict[self.uuid]["__children__"])

    @property
    def children(self):
        return [adict[child] for child in idict[self.uuid]["__children__"]]


class RootForm(A):
    def __call__(self, res=None, *args, **kwargs):
        _ = A.__call__(self, *args, **kwargs)
        return {
            "metadata": {
                "version": 4.5,
                "type": "Object",
                "generator": "Object3D.toJSON"
            },
            "object": res,
            "geometries": list(self._include_geometries),
            "materials": list(self._include_materials)
        }


class AGeometryDescriptor:
    adict = dict()

    def __init__(self, default=None):
        super().__init__()
        if default is not None:
            ageomdict[default.uuid] = default
            self._default = default.uuid
        else:
            self._default = default

    def __set_name__(self, owner, name):
        if not (name == "geometry"):
            raise
        self._name = "_" + name

    def __get__(self, instance, owner):
        if instance is None:

            return ageomdict.get(self._default)
        else:
            geom = ageomdict.get(getattr(instance, self._name))
            if hasattr(instance, "boundingSphere"):
                setattr(geom.data, "boundingSphere", instance.boundingSphere)
            return geom

    def __set__(self, instance, value):

        ageomdict[value.uuid] = value
        setattr(instance, self._name, value.uuid)


from mmcore.geom.materials import ColorRGB


class ADynamicGeometryDescriptor:
    adict = dict()

    def __init__(self, resolver):
        self.resolver = resolver

    def __set_name__(self, owner, name):
        if not (name == "geometry"):
            raise
        self._name = "_" + name

    def __get__(self, instance, owner):
        if instance is None:

            return self
        else:
            return self.resolve(instance)

    def __set__(self, instance, value):
        self.resolver = value

    def resolve(self, instance):
        def wrap(*args, **kwargs):
            res = self.resolver(instance, *args, **kwargs)
            ageomdict[res.uuid] = res
            return res

        return wrap


class Domain:
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b

    @property
    def min(self):
        return self.a

    @property
    def max(self):
        return self.b

    @property
    def delta(self):
        return self.max - self.min

    def __getitem__(self, item):
        return [self.max, self.min][item]

    def __setitem__(self, item, v):
        [self.max, self.min][item] = v

    def __iter__(self):
        return iter([self.max, self.min, self.delta])

    def __array__(self):
        return np.array([self.max, self.min], dtype=float)

    def __repr__(self):
        return f"Domain({self.max} to  {self.min}, delta={self.delta})"

    def __sub__(self, other):
        return Domain(self.max - other.a, self.b - other.b)

    def __add__(self, other):
        return Domain(self.a + other.a, self.b + other.b)

    def __mul__(self, other):
        if isinstance(other, (int, float)):

            return Domain(self.a + other, self.b + other)
        else:
            return Domain(self.a + other.a, self.b + other.b)

    def __contains__(self, item):
        return self.a <= item <= self.b


@dataclasses.dataclass(unsafe_hash=True)
class Cube:
    u: typing.Union[tuple, Domain] = (0, 1)
    v: typing.Union[tuple, Domain] = (0, 1)
    h: typing.Union[tuple, Domain] = (0, 1)

    def __post_init__(self):
        l = []
        for i in [self.u, self.v, self.h]:
            if not isinstance(i, Domain):
                l.append(Domain(*i))

        self.u, self.v, self.h = l


class CubeTable:
    def __init__(self, proxy):
        self.proxy = proxy

    _table = {
        0: operator.attrgetter("u.min", "v.min", "h.min"),
        1: operator.attrgetter("u.min", "v.min", "h.max"),
        2: operator.attrgetter("u.max", "v.min", "h.max"),
        3: operator.attrgetter("u.max", "v.max", "h.max"),
        4: operator.attrgetter("u.min", "v.max", "h.max"),
        5: operator.attrgetter("u.min", "v.max", "h.min"),
        6: operator.attrgetter("u.max", "v.max", "h.min"),
        7: operator.attrgetter("u.max", "v.min", "h.min"),
    }

    _aj = {
        0: operator.itemgetter(*[1, 4]),
        1: operator.itemgetter(*[2, 5]),
        2: operator.itemgetter(*[3, 6]),
        3: operator.itemgetter(*[4, 0]),
        4: operator.itemgetter(*[5, 7]),
        5: operator.itemgetter(*[6, 1]),
        6: operator.itemgetter(*[7, 2]),
        7: operator.itemgetter(*[0, 4])}

    def __getitem__(self, item):
        return list(map(lambda x: x(self.proxy), self._aj[item](self._table)))

    def __iter__(self):
        def gen():
            for i in range(len(self._table)):
                for j in self[i]:
                    yield self._table[i](self.proxy), j

        return iter(gen())


class BBox:
    def __init__(self, points: np.ndarray, owner):

        super().__init__()
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        self.u, self.v, self.h = [Domain(*item) for item in
                                  zip(points.min(axis=0).tolist(), points.max(axis=0).tolist())]
        self.owner = owner
        if owner.RENDER_BBOX:
            self.owner.bbox = self.__repr3d__()

    def __array__(self):
        return np.array([self.u, self.v, self.h], dtype=float)

    @property
    def domains(self):
        return [self.u, self.v, self.h]

    @property
    def points(self):

        return [pt @ self.matrix for pt in [[self.u.min, self.v.min, self.h.min],
                                            [self.u.min, self.v.min, self.h.max],
                                            [self.u.min, self.v.max, self.h.max],

                                            [self.u.max, self.v.max, self.h.max],
                                            [self.u.max, self.v.max, self.h.min],
                                            [self.u.max, self.v.min, self.h.min],

                                            [self.u.min, self.v.min, self.h.min],
                                            [self.u.max, self.v.min, self.h.max],
                                            [self.u.max, self.v.max, self.h.max],
                                            [self.u.min, self.v.max, self.h.max]

                                            ]]

    @property
    def matrix(self):
        mx = self.owner.matrix
        if not hasattr(mx, "matrix"):
            mx = self.owner.matrix_to_square_form()

        return mx

    @matrix.setter
    def matrix(self, v):

        self._matrix = v

    def __iter__(self):
        return iter([self.u, self.v, self.h])

    def __contains__(self, item):
        return all(i in dmn for i, dmn in zip(item, self.domains))

    def inside(self, point):
        return point in self

    def __repr__(self):
        return f"BBox(u={self.u}, v={self.v}, h={self.h})"

    def __repr3d__(self):

        self._repr3d = ALine(name=f'BBox at {self.owner.uuid}', uuid=f'{self.owner.uuid}_bbox', geometry=self.points,
                             material=gql_models.LineBasicMaterial(color=ColorRGB(150, 150, 150).decimal))

        return self._repr3d


class BBoxDescriptor:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        arr = instance.geometry.data.attributes.position.array

        return BBox(np.array(instance.geometry.data.attributes.position.array, dtype=float).reshape((len(arr) // 3, 3)),
                    owner=instance)


class AMaterialDescriptor:
    adict = dict()

    def __init__(self, default=None):
        super().__init__()
        if default is not None:
            amatdict[default.uuid] = default
            self._default = default.uuid
        else:
            self._default = default

    def __set_name__(self, owner, name):
        if not (name == "material"):
            raise
        self._name = "_" + name

    def __get__(self, instance, owner):
        if instance is None:

            return self._default
        else:
            return amatdict.get(getattr(instance, self._name, self._default))

    def __set__(self, instance, value):

        amatdict[value.uuid] = value
        setattr(instance, self._name, value.uuid)


class AGeom(A):
    material_type = gql_models.BaseMaterial
    geometry = AGeometryDescriptor()
    material = AMaterialDescriptor()
    aabb = BBoxDescriptor()

    def __getstate__(self):
        dct = super().__getstate__()
        dct["geometry"] = self.geometry
        dct["material"] = self.material
        dct["_geometry"] = self._geometry
        dct["_material"] = self._material
        return dct

    def __setstate__(self, state):
        ageomdict[state["_geometry"]] = state["geometry"]
        amatdict[state["_material"]] = state["material"]
        super().__setstate__(state)

    @property
    def points(self):
        geom = ageomdict.get(self._geometry)
        return np.array(geom.data.attributes.position.array).reshape(
            (len(geom.data.attributes.position.array) // 3, 3)).tolist()

    @property
    def boundingSphere(self):

        x = np.array(self.points)[:, 0]
        y = np.array(self.points)[:, 1]
        z = np.array(self.points)[:, 2]
        corners = np.array([[np.min(x), np.min(y), np.min(z)],
                            [np.max(x), np.min(y), np.min(z)],
                            [np.max(x), np.max(y), np.min(z)],
                            [np.max(x), np.max(y), np.max(z)],
                            [np.min(x), np.max(y), np.max(z)],
                            [np.min(x), np.min(y), np.max(z)],
                            [np.min(x), np.max(y), np.min(z)],
                            [np.max(y), np.min(y), np.max(z)]]).tolist()
        dists = []
        for corn in corners:
            dists.append(euclidean(self.centroid, corn))
        dists.sort(reverse=True)
        rad = dists[0]

        return self.centroid, rad

    @property
    def threejs_type(self):
        return "Geometry"

    def __call__(self, *args, **kwargs):
        if kwargs.get('material') is None:
            if kwargs.get('color') is not None:
                self.color = kwargs.get('color').decimal
                kwargs['material'] = self.material_type(
                    color=kwargs.get('color').decimal if isinstance(kwargs.get('color'), int) else kwargs.get(
                        'color').decimal)

        res = super().__call__(*args, **kwargs)
        res |= {
            "geometry": self.geometry.uuid if self.geometry else None,
            "material": self._material if self._material else None,
        }
        return res


class AMesh(AGeom):
    material_type = gql_models.MeshPhongMaterial
    geometry = AGeometryDescriptor(default=None)
    material = AMaterialDescriptor(default=gql_models.MeshPhongMaterial(color=ColorRGB(120, 200, 40).decimal))

    @property
    def threejs_type(self):
        return "Mesh"


def position_hash(points):
    return hashlib.sha512(ujson.dumps(np.array(points).tolist()).encode()).hexdigest()


class APointsGeometryDescriptor(AGeometryDescriptor):
    def __get__(self, instance, owner):
        if instance is None:
            return self._default
        else:
            if hasattr(instance, self._name):
                return gql_models.BufferGeometryObject(**{
                    'data': gql_models.Data1(
                        **{'attributes': gql_models.Attributes1(
                            **{'position': gql_models.Position(
                                **{'itemSize': 3,
                                   'type': 'Float32Array',
                                   'array': np.array(
                                       instance.points).flatten().tolist(),
                                   'normalized': False})})})})




            else:
                return None

    def __set__(self, instance, value):

        instance.points = value

        uid = position_hash(value)
        setattr(instance, self._name, uid)
        ageomdict[uid] = gql_models.BufferGeometryObject(**{
            'data': gql_models.Data1(
                **{'attributes': gql_models.Attributes1(
                    **{'position': gql_models.Position(
                        **{'itemSize': 3,
                           'type': 'Float32Array',
                           'array': np.array(value).flatten().tolist(),
                           'normalized': False})})})})


class APoints(AGeom):
    material_type = gql_models.PointsMaterial
    geometry = APointsGeometryDescriptor(default=None)

    material = AMaterialDescriptor(default=gql_models.PointsMaterial(color=ColorRGB(120, 200, 40).decimal))
    _material = None
    _points = []
    kd = None

    def solve_kd(self):
        self.kd = KDTree(self.points)
        return self.kd

    @property
    def points(self):
        return list(self._points)

    @points.setter
    def points(self, v):
        if isinstance(v, np.ndarray):
            v = v.tolist()
        self._points = v

    @property
    def threejs_type(self):
        return "Points"


from scipy.spatial import KDTree


class APoint(APoints):

    def __new__(cls, x, y, z, *args, **kwargs):
        return super().__new__(cls, points=[x, y, z], *args, **kwargs)

    @property
    def x(self):
        return self.points[0]

    @property
    def y(self):
        return self.points[1]

    @property
    def z(self):
        return self.points[2]

    @z.setter
    def z(self, v):
        self.points[2] = v

    @y.setter
    def y(self, v):
        self.points[1] = v

    @x.setter
    def x(self, v):
        self.points[0] = v

    def distance(self, other: 'APoint'):
        return euclidean(self.points, other.points)


LineDefaultMaterial = gql_models.LineBasicMaterial(color=ColorRGB(120, 200, 40).decimal, uuid="line-12020040")
amatdict[LineDefaultMaterial.uuid] = LineDefaultMaterial


class ALine(APoints):
    material_type = gql_models.LineBasicMaterial
    geometry = APointsGeometryDescriptor(default=None)
    material = AMaterialDescriptor(
        default=LineDefaultMaterial)
    _material = 'line-12020040'

    @property
    def start(self):
        return self.points[0]

    @start.setter
    def start(self, value):
        self.points[0] = value

    @property
    def end(self):
        return self.points[-1]

    @end.setter
    def end(self, value):
        self.points[-1] = value

    @property
    def threejs_type(self):
        return "Line"


grp = AGroup(name="base_root", uuid="_")


class TestException(Exception): ...


class ALineDynamic(ALine):
    _colors = None
    _closed = False

    @property
    def closed(self):
        return self._closed

    @closed.setter
    def closed(self, v):
        self._closed = bool(v)

    @property
    def colors(self):

        return self._colors

    @colors.setter
    def colors(self, v):
        self._colors = v

    @property
    def geometry(self):
        if self.closed:
            pts = self.points + [self.points[0]]
        else:
            pts = self.points
        if self.colors is not None:
            if len(self.colors) > 0:
                return gql_models.BufferGeometryObject(**{
                    'uuid': self._geometry,
                    'data': gql_models.Data12(
                        **{'attributes': gql_models.Attributes4(
                            **{
                                'position': gql_models.Position(
                                    **{'itemSize': 3,
                                       'type': 'Float32Array',
                                       'array': np.array(pts).flatten().tolist(),
                                       'normalized': False
                                       }

                                ),
                                "colors": gql_models.Color(array=np.array(self.colors).flatten().tolist())
                            }

                        ),
                            "boundingSphere": gql_models.BoundingSphere(*self.boundingSphere)

                        }
                    )
                }
                                                       )

        return gql_models.BufferGeometryObject(**{
            'uuid': self._geometry,
            'data': gql_models.Data12(
                **{'attributes': gql_models.Attributes1(
                    **{
                        'position': gql_models.Position(
                            **{'itemSize': 3,
                               'type': 'Float32Array',
                               'array': np.array(pts).flatten().tolist(),
                               'normalized': False
                               }
                        )
                    }

                ),
                    "boundingSphere": gql_models.BoundingSphere(*self.boundingSphere)
                }
            )
        }
                                               )

    @property
    def _geometry(self):
        return "geom_" + str(self.uuid)

    @property
    def centroid(self):
        return np.average(np.array(self.points), axis=0).tolist()

    @property
    def boundingSphere(self):
        x = np.array(self.points)[:, 0]
        y = np.array(self.points)[:, 1]
        z = np.array(self.points)[:, 2]
        corners = np.array([[np.min(x), np.min(y), np.min(z)],
                            [np.max(x), np.min(y), np.min(z)],
                            [np.max(x), np.max(y), np.min(z)],
                            [np.max(x), np.max(y), np.max(z)],
                            [np.min(x), np.max(y), np.max(z)],
                            [np.min(x), np.min(y), np.max(z)],
                            [np.min(x), np.max(y), np.min(z)],
                            [np.max(y), np.min(y), np.max(z)]]).tolist()
        dists = []
        for corn in corners:
            dists.append(euclidean(self.centroid, corn))
        dists.sort(reverse=True)
        rad = dists[0]

        return self.centroid, rad


class APointsDynamic(ALineDynamic):
    _closed = False
    material_type = APoints.material_type


class ObjectThree:
    def __init__(self, root: A):
        self.root = root

    def get_obj(self):
        return adict[self.root]

    def all(self):
        return adict[self.root].root()

    def walk(self):
        three = {}
        obj = self.get_obj()
        if len(obj.child_keys) > 0:
            for k in obj.idict.keys():
                name, key = k
                if key == obj.uuid:
                    three[name] = self.__class__(obj.idict[(name, key)])
        return three

    def __getitem__(self, item):
        if item == "all":
            return self.all()
        else:
            return self.walk()[item]

    def keys(self):
        return self.walk().keys()

    def todict(self):

        dct = {"all": self.all()}
        for k, v in self.walk().items():
            if isinstance(v, ObjectThree):
                dct[k] = v.todict()
        return dct


class GqlObjectThree(ObjectThree):

    def __getitem__(self, item):
        if item != "all":

            return super().__getitem__(item)
        else:
            return super().__getitem__(item)


class Three:
    all: JSON


def new_three(origin: GqlObjectThree = None):
    attrs = dict(itertools.zip_longest(origin.keys(), ['GenericThree'], fillvalue='GenericThree'))
    define = f"""
    
@strawberry.type
class GenericThree:
    __annotations__ = {attrs}
    
    @property
    def origin(self):
        return self._origin
      
    @strawberry.field
    def all(self) -> JSON:
        return self.origin.all()
        
"""
    for k in origin.keys():
        define += f"""   
    @strawberry.field 
    def {k}(self)->'GenericThree':
        return new_three(self.origin["{k}"])"""

    cd = compile(define, "_i", "exec")

    exec(cd, globals(), locals())
    e = eval('GenericThree')()
    e._origin = origin
    return e
