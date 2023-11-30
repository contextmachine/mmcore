import inspect
import json

components = dict()
_components_type_name_map = dict()
import functools
import uuid as _uuid

import numpy as np

NS = _uuid.UUID('eecf16e3-726f-49e4-9fc3-73d22f8c81ff')


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return list(o.flat)
        else:
            return super().default(o)
class SoAField:
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


class SoAItem:
    def __init__(self, uuid, parent_id):
        super().__init__()
        self._parent = parent_id
        self._uuid = uuid
        self._component_type_name = self.parent.type_name

    def get_field(self, k):
        return {
            int: lambda x: self.parent.fields[list(self.parent.fields.keys())[x]],
            str: lambda x: self.parent.fields[x]
        }[type(k)](k)


    @property
    def component_type(self):
        return self._component_type_name

    @property
    def uuid(self):
        return self._uuid

    def update(self, val):
        for key, v in val.items():
            self.get_field(key)[self.uuid] = v

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
        *ks, = self.keys()

        vs = sorted(([v] if np.isscalar(v) else v for v in self.values()), key=lambda x: len(x), reverse=True)

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
        if item.startswith('_'):
            return object.__getattribute__(self, item)

        elif item in soa_parent(self).fields.keys():
            return self[item]
        else:
            return super().__getattribute__(item)

    def __setattr__(self, item, val):
        if item.startswith('_'):
            object.__setattr__(self, item, val)

        elif item in soa_parent(self).fields.keys():

            self[item] = val

        else:

            self.parent.add_field(item, SoAField())
            self[item] = val

    def __dir__(self):
        return list(super().__dir__()) + list(self.keys())

    def __repr__(self):
        return f'{self.parent.type_name}({dict(self)}) at {self.uuid}'


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
        if item.startswith('_'):
            return object.__getattribute__(self, item)

        elif item in soa_parent(self).fields.keys():
            return self[item]
        else:
            return super().__getattribute__(item)

    def __setattr__(self, item, val):

        if item.startswith('_'):
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
        a = np.array_str(p._arr[self._arr_index], precision=4).replace('\n', '').replace(" ", ", ")
        return f'{self.component_type}({", ".join(p.fields.keys())}, {a}) at {self._uuid}'

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

    def add_field(self, key, fld):
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
        return f'SoA(type={self.type_name}, length={len(self.__items__)}) at {self.uuid}'


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
            self.__items__[uuid] = SoArrayItem(uuid=uuid, parent_id=self.uuid, arr_index=ixs)

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

        return {
            int: lambda x: x,
            str: lambda x: list(self.fields.keys()).index(x)
        }[type(ixs)](ixs)

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
            name = cls.__name__.lower()
        fields = dict()

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

                *keys, = fields.keys()
                if uuid is None:
                    uuid = _uuid.uuid4().hex
                if uuid in _soa.__items__:
                    return _soa[uuid]
                dct = dict(zip(keys[:len(args)], args))
                dct |= kwargs
                for fld in set.difference(set(fields), set(dct.keys())):
                    val = getattr(cls, fld)

                    if inspect.isfunction(getattr(cls, fld)):
                        dct[fld] = val()

                _soa[uuid] = dct
                return _soa[uuid]


        ctor.component_type = name
        _soa.__component_ctor__ = ctor
        return ctor

    return wrapper


def apply(obj, data):
    for k, v in data.items():
        if isinstance(v, dict):
            apply(getattr(obj, k), v)
        else:
            setattr(obj, k, v)


def todict(obj):
    if hasattr(obj, 'items'):
        return {k: todict(v) for k, v in obj.items()}
    elif hasattr(obj, 'todict'):
        return obj.todict()
    elif isinstance(obj, (list, tuple)):
        return [todict(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def request_component_type(name):
    return components[_components_type_name_map[name]]


def request_component(name, uuid):
    return request_component_type(name)[uuid]


def default_value(comp, field: str):
    return request_component_type(comp.component_type).fields['data'].default()
