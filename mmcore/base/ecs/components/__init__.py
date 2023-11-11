components = dict()
_components_type_name_map = dict()
import functools
import uuid as _uuid

import numpy as np

NS = _uuid.UUID('eecf16e3-726f-49e4-9fc3-73d22f8c81ff')


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

    @property
    def component_type(self):
        return self._component_type_name

    @property
    def uuid(self):
        return self._uuid

    def update(self, val):
        for key, v in val.items():
            self.parent.fields[key][self.uuid] = v

    def __setitem__(self, key, val):
        self.parent.fields[key][self.uuid] = val

    def __getitem__(self, key):
        return self.parent.fields[key][self.uuid]

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


Component = SoAItem
ComponentType = SoA


def component(name=None):
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
                fields[k] = SoAField(default=v())

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
