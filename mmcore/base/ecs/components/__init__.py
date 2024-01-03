import copy
import inspect
import json
import queue

import itertools

components = dict()
components_ctors = dict()
_components_type_name_map = dict()
import functools
import uuid as _uuid
from dataclasses import is_dataclass, asdict
import numpy as np
import numpy
from numpy import ndarray, dtype

ndarray, dtype, numpy = ndarray, dtype, numpy
NS = _uuid.UUID("eecf16e3-726f-49e4-9fc3-73d22f8c81ff")


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return list(o.flat)
        else:
            return super().default(o)


class SoAField:
    """
        Copy:
        ----

    >>> from mmcore.geom.rectangle import Rectangle,to_mesh
    >>> import copy
    >>> r1=Rectangle(10,20)
    >>> a=copy.deepcopy(r1)
    >>> a
    Rectangle([[ 0.  0.  0.]
     [10.  0.  0.]
     [10. 20.  0.]
     [ 0. 20.  0.]])
    >>> a.ecs_rectangle
    RectangleComponent({'plane': PlaneComponent({'ref': array([[0., 0., 0.],
           [1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]]), 'origin': 0, 'xaxis': 1, 'yaxis': 2, 'zaxis': 3}) at 7e40fc49f42843ca8bfa29d6d8fd7511,
           'uv': UV({'u': Length({'value': 10}) at b0e3951fdd654dd296dd7e474128803b, 'v': Length({'value': 20}) at
           ad105e2814254394b551b26d238a9af2}) at 7487ff2dac4d413da67999d4d46b324b}) at a30d188f1bb1400888da10982088972b
    >>> r1.ecs_rectangle
    RectangleComponent({'plane': PlaneComponent({'ref': array([[0., 0., 0.],
           [1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]]), 'origin': 0, 'xaxis': 1, 'yaxis': 2, 'zaxis': 3}) at 1426df91b19f42459f35d04306c43538,
           'uv': UV({'u': Length({'value': 10}) at 424d0b8cd21d4034b734f756e94700be, 'v': Length({'value': 20}) at
           88797284c9984353bb0be9e295bd834c}) at 9bc79b28188c4096acf50ebfe78f8f87}) at 7955f77c6a094f3295ff8ce32072b81e


    """

    def __init__(self, default=None):
        self.default = default
        self.data = dict()
        self._parent = None

    @property
    def parent(self):
        return components.get(self._parent)

    @parent.setter
    def parent(self, v):
        if isinstance(v, SoA):
            self._parent = v.uuid
        else:
            self._parent = v

    def __getitem__(self, key):
        return self.data.get(key, self.default)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        self.data.__delitem__(key)

    def get(self, key, __default=None):
        return self.data.get(key, __default)

    def update(self, *args, **kwargs):
        self.data.update(*args, **kwargs)

    def __len__(self):
        return self.data.__len__()

    def __iter__(self):
        return self.data.__iter__()


def soa_parent(self):
    return components[self._parent]


def safe_get_field(self, key):
    if key not in self.parent.fields.keys():
        self.parent.add_field(key)
    return self.parent.fields[key]


class SoAItem:
    def __init__(self, uuid, parent_id):
        super().__init__()
        self._parent = parent_id
        self._uuid = uuid
        self._component_type_name = self.parent.type_name

    def __copy__(self):
        uu = _uuid.uuid4().hex
        self._parent[uu] = dict(self.items())
        return self._parent[uu]

    def __deepcopy__(self, memodict={}):
        return components_from_spec(
            copy.deepcopy(components_to_spec(self), memodict), return_root_only=True
        )

    def get_field(self, k):
        return {
            int: lambda x: self.parent.fields[list(self.parent.fields.keys())[x]],
            str: lambda x: safe_get_field(self, x),
        }[type(k)](k)

    @property
    def component_class(self):
        return components_ctors[self._component_type_name]

    @property
    def component_type(self):
        return self._component_type_name

    @property
    def uuid(self):
        return self._uuid

    def update(self, val):
        for key, v in val.items():
            self.get_field(key)[self.uuid] = v

    def clear(self):
        for key in self.keys():
            del self.parent.fields[key][self.uuid]

    def __setitem__(self, key, val):
        if isinstance(key, (str, int)):
            self.get_field(key)[self.uuid] = val
        else:
            if len(key) > 0:
                key1 = key[0]
                self.get_field(key1)[self.uuid].__setitem__(key[1:], val)
            elif len(key) == 1:
                self[key[0]] = val

    def __getitem__(self, key):
        if isinstance(key, (str, int)):
            return self.get_field(key)[self.uuid]
        else:
            if len(key) > 1:
                key1 = key[0]

                return self.get_field(key1)[self.uuid].__getitem__(key[1:])
            elif len(key) == 1:
                return self[key[0]]

    def get_index(self):
        (*ks,) = self.keys()

        vs = sorted(
            ([v] if np.isscalar(v) else v for v in self.values()),
            key=lambda x: len(x),
            reverse=True,
        )

        return np.ndindex((len(ks), len(vs[0])))

    def __len__(self):
        return len([*self.keys()])

    @property
    def parent(self):
        return soa_parent(self)

    def values(self):
        for field in soa_parent(self).fields.values():
            if field[self.uuid] is not None:
                yield field[self.uuid]

    def keys(self):
        for key, field in soa_parent(self).fields.items():
            if field[self.uuid] is not None:
                yield key

    def __iter__(self):
        for key, field in self.parent.fields.items():
            if field[self.uuid] is not None:
                yield key, field[self.uuid]

    def items(self):
        for key, field in self.parent.fields.items():
            if field[self.uuid] is not None:
                yield key, field[self.uuid]

    def __ror__(self, other):
        self.update(other)

    def __ior__(self, other):
        _ks = self.keys()
        for k, v in other.items():
            if k not in _ks:
                self[k] = v

    def __getattr__(self, item):
        if item.startswith("_"):
            return object.__getattribute__(self, item)

        elif item in soa_parent(self).fields.keys():
            return self[item]
        else:
            return super().__getattribute__(item)

    def __setattr__(self, item, val):
        if item.startswith("_"):
            object.__setattr__(self, item, val)

        elif item in soa_parent(self).fields.keys():
            self[item] = val

        else:
            self.parent.add_field(item, SoAField())
            self[item] = val

    def __dir__(self):
        return list(super().__dir__()) + list(self.keys())

    def __repr__(self):
        kws = dict(self)
        return f"{self.parent.type_name}({kws}) at <{self.uuid}>"

    def todict(self):
        return todict(self)


class SoArrayItem(SoAItem):
    def __init__(self, uuid, parent_id, arr_index):
        self._parent = parent_id
        self._arr_index = arr_index
        super().__init__(uuid, parent_id)

    @property
    def parent(self):
        return soa_parent(self)

    def __getitem__(self, item):
        return soa_parent(self)._arr[self._arr_index][soa_parent(self).by_index(item)]

    def __setitem__(self, item, v):
        if np.isscalar(item):
            soa_parent(self)._arr[self._arr_index][self.parent.by_index(item)] = v

        else:
            soa_parent(self)._arr[self._arr_index][item] = v

        # if v.shape[0]> self.parent.item_shape[0]:
        #    self.parent.item_shape=(v.shape[0], *self.parent.item_shape[1:])
        #    self.parent._arr.resize(self.parent._arr.shape[0],self.parent.item_shape)

        # self.parent.by_index(self._uuid, item[0])[item[1:]]=v

    def __getattr__(self, item):
        if item.startswith("_"):
            return object.__getattribute__(self, item)

        elif item in soa_parent(self).fields.keys():
            return self[item]
        else:
            return super().__getattribute__(item)

    def __setattr__(self, item, val):
        if item.startswith("_"):
            object.__setattr__(self, item, val)

        elif item in soa_parent(self).fields.keys():
            self[list(soa_parent(self).fields.keys()).index(item)] = val

        else:
            self.parent.add_field(item, SoAField())
            self[item] = val

    def __array__(self):
        return self.parent._arr[self._arr_index]

    def __repr__(self):
        p = soa_parent(self)
        a = (
            np.array_str(p._arr[self._arr_index], precision=4)
            .replace("\n", "")
            .replace(" ", ", ")
        )
        return f'{self.component_type}({", ".join(p.fields.keys())}, {a}) at <{self._uuid}>'


from dataclasses import dataclass


@dataclass
class SoAProps:
    allow_nulls: bool = False


entity_attrs_type_map = dict()


class SoA:
    def __init__(self, type_name: str, props: SoAProps = None, **fields):
        assert type_name
        self.type_name = type_name

        self.props = props if props else SoAProps()
        self.fields = dict()
        self.__items__ = dict()
        self._mmcore_uuid = _uuid.uuid5(NS, type_name)
        _components_type_name_map[type_name] = self._mmcore_uuid.hex
        components[self._mmcore_uuid.hex] = self
        entity_attrs_type_map[type_name] = self.__items__

        for k, v in fields.items():
            self.add_field(k, v)

    @property
    def uuid(self):
        return self._mmcore_uuid.hex

    def add_field(self, key, fld=None):
        if fld is None:
            fld = SoAField()
        fld.parent = self

        self.fields[key] = fld

    def remove_field(self, key):
        del self.fields[key]

    def __getitem__(self, uuid):
        return self.__items__[uuid]

    def __setitem__(self, uuid, v):
        if uuid not in self.__items__.keys():
            self.__items__[uuid] = SoAItem(uuid, self.uuid)

        self.__items__[uuid].update(v)

    def __contains__(self, item):
        return self.__items__.__contains__(item)

    def __repr__(self):
        return (
            f"SoA(type={self.type_name}, length={len(self.__items__)}) at {self.uuid}"
        )


class SoArray(SoA):
    def __init__(self, *args, item_shape=(3,), **kwargs):
        self.item_shape = item_shape
        self._arr = np.empty((128, len(kwargs), *item_shape), dtype=object)
        self.__arr_index__ = dict()
        super().__init__(*args, **kwargs)

    def add_field(self, key, fld):
        super().add_field(key, fld)

    def add_new_field(self, key, fld):
        self.resize(fields_count=1)
        super().add_field(key, fld)

    def __setitem__(self, uuid, v):
        v = np.array(v)
        if uuid not in self.__items__.keys():
            ixs = len(self.__items__)

            if ixs >= self._arr.shape[0]:
                self.resize(count=128)

            self._arr[ixs] = v
            self.__arr_index__[uuid] = ixs
            self.__items__[uuid] = SoArrayItem(
                uuid=uuid, parent_id=self.uuid, arr_index=ixs
            )

        self._arr[self.__arr_index__[uuid]] = v

    def resize(self, count=0, fields_count=0, item_shape=0):
        resize_shape = (count, fields_count, item_shape)
        new = np.array(resize_shape, dtype=int) + np.array(self._arr.shape, dtype=int)
        self.item_shape = tuple(new[2:])
        a, b = np.divmod(new[0], 128)
        if b > 0:
            a += 1
        self._arr.resize((a * 128, new[1], *self.item_shape), refcheck=False)

    def by_index(self, ixs):
        return {int: lambda x: x, str: lambda x: list(self.fields.keys()).index(x)}[
            type(ixs)
        ](ixs)

    def by_uuid(self, uuid, key, ixs=()):
        return self._arr[self.__arr_index__[uuid]][(self.by_index(key),) + ixs]


Component = SoAItem
ComponentType = SoA


def component(name=None, array_like=False, item_shape=()):
    """
    >>> import typing
    >>> from mmcore.base.tags import SoAField,component,todict
    >>> class ChildCountField(SoAField):
    ...     def __getitem__(self, pk):
    ...         return len(self.parent.fields['children'].get(pk, ()))

    >>> @component('test')
    ... class TestComponent:
    ...     tag:str="A"
    ...     mount:bool=False
    ...     children:typing.Any=None
    ...     children_count:int = ChildCountField(0)

    >>> tc=TestComponent("C")
    >>> tc2=TestComponent("D")
    >>> tc3=TestComponent("D")
    >>> tc4=TestComponent()
    >>> tc.children=[tc2,tc3]
    >>> tc2.children=[tc4]
    >>> todict(tc)
    {'tag': 'C',
     'mount': False,
     'children': [{'tag': 'D',
       'mount': False,
       'children': [{'tag': 'A', 'mount': False, 'children_count': 0}],
       'children_count': 1},
      {'tag': 'D', 'mount': False, 'children_count': 0}],
     'children_count': 2}
    """

    def wrapper(cls):
        nonlocal name
        if name is None:
            name = cls.__name__
        fields = dict()
        if hasattr(cls, "__annotations__"):
            for k, v in cls.__annotations__.items():
                if k in cls.__dict__:
                    val = cls.__dict__[k]
                    if isinstance(val, SoAField):
                        fields[k] = val

                    else:
                        fields[k] = SoAField(default=val)

                else:
                    fields[k] = SoAField(default=None)

        if array_like:
            _soa = SoArray(name, item_shape=item_shape, **fields)

            @functools.wraps(cls)
            def ctor(*args, uuid=None, **kwargs):
                if uuid is None:
                    uuid = _uuid.uuid4().hex
                if uuid in _soa.__items__:
                    return _soa[uuid]
                _soa[uuid] = args + tuple(kwargs.values())

                return _soa[uuid]

        else:
            _soa = SoA(name, **fields)

            @functools.wraps(cls)
            def ctor(*args, uuid=None, **kwargs):
                (*keys,) = fields.keys()
                if uuid is None:
                    uuid = _uuid.uuid4().hex
                if uuid in _soa.__items__:
                    return _soa[uuid]
                dct = dict(zip(keys[: len(args)], args))
                dct |= kwargs
                for fld in set.difference(set(fields), set(dct.keys())):
                    val = getattr(cls, fld)

                    if inspect.isfunction(getattr(cls, fld)):
                        dct[fld] = val()

                _soa[uuid] = dct
                return _soa[uuid]

        ctor.component_type = name
        _soa.__component_ctor__ = ctor
        components_ctors[name] = ctor

        ctor.__view_name__ = name
        return ctor

    return wrapper


def apply(obj, data):
    for k, v in data.items():
        if isinstance(v, dict):
            apply(getattr(obj, k), v)
        else:
            obj.__setitem__(k, v)


COMPONENT_TYPE_FIELD = "component_type"


def is_component(obj):
    return hasattr(obj, COMPONENT_TYPE_FIELD)


def todict(obj):
    if is_dataclass(obj):
        return asdict(obj)
    elif hasattr(obj, "items"):
        return {k: todict(v) for k, v in obj.items()}
    elif hasattr(obj, "todict"):
        return obj.todict()
    elif isinstance(obj, (list, tuple)):
        return [todict(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def request_component_type(name):
    return components[_components_type_name_map[name]]


def request_component_ctor(name):
    return components_ctors[name]


def request_component(name, uuid):
    return request_component_type(name)[uuid]


def default_value(comp, field: str):
    return request_component_type(comp.component_type).fields["data"].default()


def dumps(cmp, **kwargs):
    return json.dumps(todict(cmp), cls=NpEncoder, **kwargs)


def dump(cmp, f, **kwargs):
    return json.dump(todict(cmp), f, cls=NpEncoder, **kwargs)


def ref_to_fields(cmp, ref, fields):
    return {k: ref[cmp[k]] for k in fields}


from collections import defaultdict

CASTS = defaultdict(lambda: lambda tp, x: x)
from itertools import count

CASTS["ndarray"] = lambda tp, x: np.array(x)


def from_spec(spec, refs: list, comps_queue: queue.Queue, comps: list):
    if not isinstance(spec, dict):
        return spec
    sp = spec.get("spec")
    if sp:
        ref = sp.get("ref")
        kind = sp.get("kind")

        if kind == "LEAF":
            nm = sp.get("type")

            if "ndarray" in nm:
                shp, dtp = eval(nm).__args__
                return np.array(spec.get("value"), dtp).reshape(shp)
            else:
                return spec.get("value")

        elif kind == "COMP":
            fields = dict()
            for k, f in spec.get("fields").items():
                obj = from_spec(f, refs, comps_queue, comps)
                if inspect.isfunction(obj):
                    fields[k] = obj(len(comps), k)
                else:
                    fields[k] = obj

            if ref is not None:
                fields["ref"] = from_spec(refs[ref], refs, comps_queue, comps)

            _res = request_component_ctor(sp.get("type"))(**fields)

            comps.append(_res)
            return _res
        elif kind == "COMP_REF":
            val = spec.get("value")
            cb = lambda index, field: comps_queue.put(
                lambda: setattr(comps[index], field, val)
            )

            return cb

    else:
        return spec


from mmcore.func import extract_type


@functools.lru_cache()
def pretty_name(tp):
    if hasattr(tp, "__qualname__") and not hasattr(tp, "__args__"):
        return tp.__qualname__
    else:
        return repr(tp)


def to_spec(cmp, refs: list, cnt: itertools.count, comps: list):
    if not is_component(cmp):
        return {
            "value": cmp,
            "spec": {"kind": "LEAF", "type": pretty_name(extract_type(cmp))},
        }

    cnp_uuid_hash = hash(cmp.uuid)
    if cnp_uuid_hash in comps:
        cmp_index = comps.index(cnp_uuid_hash)

        return {
            "value": int(cmp_index),
            "spec": {"kind": "COMP_REF", "type": cmp.component_type},
        }
    else:
        comps.append(cnp_uuid_hash)
    (*fields,) = cmp.keys()

    if "ref" in fields:
        fields.remove("ref")
        ref = cmp.ref
        res = np.arange(len(refs))[np.in1d([id(i["value"]) for i in refs], [id(ref)])]

        if len(res) > 0:
            print(res)
            l = int(res[0])

        else:
            refs.append(to_spec(ref, refs, cnt, comps))
            l = next(cnt)
        return {
            "uuid": cmp.uuid,
            "fields": {k: to_spec(cmp[k], refs, cnt, comps) for k in fields},
            "spec": {"kind": "COMP", "type": cmp.component_type, "ref": l},
        }
    else:
        return {
            "uuid": cmp.uuid,
            "fields": {k: to_spec(cmp[k], refs, cnt, comps) for k in fields},
            "spec": {"kind": "COMP", "type": cmp.component_type},
        }


def components_to_spec(*cmp):
    cnt = count()
    refs = []
    comps = []
    specs = [to_spec(cm, refs, cnt, comps) for cm in cmp]
    return dict(spec=specs, refs=refs)


def components_from_spec(spec, return_root_only=False):
    refs = spec["refs"]

    comps = []
    que = queue.Queue()
    [from_spec(cm, refs, que, comps) for cm in spec["spec"]]
    while not que.empty():
        que.get()()
    if return_root_only:
        return comps[-1]
    return comps


ECS_CONTAINER_FIELD_NAME = "ecs_components"


class EcsContainer:
    def __init__(self, field_name="ecs_components"):
        self.field_name = field_name
        self._data = dict()

    def __set_name__(self, owner, name):
        if hasattr(owner, "ecs_map"):
            owner.ecs_map = {self.field_name: name}
        self._data[owner.__name__] = dict()
        self._name = name

    def __get__(self, instance, owner):
        if instance:
            return self._data[owner.__name__].get(id(instance), None)
        else:
            return self._data[owner.__name__]

    def new_range(self, ecs_map):
        return np.empty(len(ecs_map), dtype=object)

    def self_index(self, ecs_map):
        return ecs_map[self._name]

    def __set__(self, instance, value):
        self._data[instance.__class__.__name__][id(instance)] = value


class EcsProperty:
    def __init__(self, i=None, type=None):
        self.i = i
        self.typ = type

    def __set_name__(self, owner, name):
        if not hasattr(owner, "ecs_map"):
            owner.ecs_map = dict()
        if self.i is not None:
            if self.i in owner.ecs_map.values():
                raise IndexError(
                    f"component {self.i} is already defined: {owner.ecs_map}"
                )
        else:
            self.i = len(owner.ecs_map)
        owner.ecs_map[name] = dict(n=self.i, component_type=self.typ)

        self._name = name

    def __get__(self, instance, owner):
        if instance:
            return self.ecs_get(instance)
        else:
            return self

    def __set__(self, instance, value):
        self.ecs_set(instance, value)

    def ecs_get(self, instance):
        return instance.ecs_components[instance.ecs_map[self._name]["n"]]

    def ecs_set(self, instance, value):
        instance.ecs_components[instance.ecs_map[self._name]["n"]] = value


def deepcopy_components(cmps, memo={}, return_root_only=False):
    return components_from_spec(
        copy.deepcopy(components_to_spec(*cmps), memo),
        return_root_only=return_root_only,
    )


class EcsProto:
    ecs_map = {}

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        self.ecs_components = np.empty(len(cls.ecs_map), dtype=object)

        # print(cls.ecs_map)
        # print(self.ecs_components)
        self.__init__(*args, **kwargs)
        return self

    def ecs_component_types(self):
        return (
            (e["component_type"].component_type, e["n"]) for e in self.ecs_map.values()
        )
