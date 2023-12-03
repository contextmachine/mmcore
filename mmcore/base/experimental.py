import dataclasses
import json
import time
import types
import uuid as _uuid
from collections import namedtuple, OrderedDict
from types import FunctionType, LambdaType

import rich

object_table = dict()
component_table = dict()
geom_table = dict()
mat_table = dict()
tables = dict(component=component_table, geometry=geom_table, material=mat_table)

registry = {}


class MultiMethod(object):
    def __init__(self, name):
        self.name = name
        self.typemap = {}

    def __call__(self, *args):
        types = tuple(arg.__class__ for arg in args)  # a generator expression!
        function = self.typemap.get(types)
        if function is None:
            raise TypeError("no match")
        return function(*args)

    def register(self, types, function):
        if types in self.typemap:
            raise TypeError("duplicate registration")
        self.typemap[types] = function


def multimethod(*types):
    def register(function):
        function = getattr(function, "__lastreg__", function)
        name = function.__name__
        mm = registry.get(name)
        if mm is None:
            mm = registry[name] = MultiMethod(name)
        mm.register(types, function)
        mm.__lastreg__ = function
        return mm

    return register

class Object3D:
    def __new__(cls, uuid=None):
        if uuid is None:
            uuid = _uuid.uuid4().hex
        obj = super().__new__(cls)
        obj.uuid = uuid
        component_table[uuid] = dict()
        object_table[uuid] = obj
        return obj


class Group(Object3D):
    def __new__(cls, children=None, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        if children is not None:
            children = set(child.uuid for child in children)
        else:
            children = set()
        component_table[obj.uuid]["__children__"] = children
        return obj

    def add(self, obj):
        component_table[self.uuid]["__children__"].add(obj.uuid)

    def remove(self, obj):
        component_table[self.uuid]["__children__"].remove(obj.uuid)

    @property
    def children_uuids(self):
        return component_table[self.uuid]["__children__"]

    @property
    def children(self):
        return [object_table[uuid] for uuid in self.children_uuids]


class GeometryObject(Object3D):
    def __new__(cls, geometry=None, material=None, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        component_table[obj.uuid]["__material__"] = material
        component_table[obj.uuid]["__geometry__"] = geometry
        return obj

    @property
    def geometry_uuid(self):
        return component_table[self.uuid]['__geometry__']

    @property
    def geometry(self):
        return geom_table[self.geometry_uuid]


def get_all_geometry_test(grp: Group):
    geoms = []
    for uuid in grp.children_uuids:
        if "__children__" in component_table[uuid].keys():
            geoms.extend(get_all_geometry_test(object_table[uuid]))
        if "__geometry__" in component_table[uuid].keys():
            geoms.append(object_table[uuid])
    return geoms


from collections import deque


class NsInterface:
    __slots__ = ['get_uuid',
                 'get_keys',
                 'getter',
                 'setter',
                 'get_dct',
                 'get_ns',
                 'get_parent',
                 'get_children',
                 'child',
                 'tree',
                 'get_time'
                 ]

    def __iter__(self):
        return (getattr(self, k) for k in self.__slots__)

    def __init__(self, *args):
        for k, v in zip(self.__slots__, args):
            setattr(self, k, v)


__ns_slots__ = NsInterface.__slots__

__nsmap__ = dict()

__nsobjmap__ = dict()


def nstype(ns_interface: NsInterface = None):
    def _func_object(**attrs):
        _time = time.time()
        dct = attrs
        children = deque()
        uuid = _uuid.uuid4().hex

        def tree():
            _ns = dict()
            for k, v in get_ns().items():
                if isinstance(v, NsInterface):
                    _ns[k] = v.tree()
                else:
                    _ns[k] = v
            return _ns

        def get_time():
            return _time

        def get_keys():
            l = dct.keys()
            if ns_interface:
                return l | NsInterface(*ns_interface).get_keys()
            return l

        def getter(name):
            nonlocal dct

            if name not in dct:
                if ns_interface:
                    return NsInterface(*ns_interface).getter(name)
            return dct[name]

        def setter(name, val, overwrite=False):
            nonlocal dct
            if name in dct:
                if isinstance(dct[name], NsInterface):
                    if isinstance(val, dict):
                        if overwrite:
                            for k, v in val.items():
                                dct[name].setter(k, v)

                        else:
                            dct[name] = dct[name].child(**val)

                    else:
                        dct[name] = val
                else:
                    dct[name] = val
            else:
                dct[name] = val

        def get_dct():
            nonlocal dct
            return dct

        def get_uuid():

            return uuid

        def get_parent():

            return ns_interface

        def get_ns():

            return dict((k, getter(k)) for k in get_keys())

        def get_children():

            return children

        def child(**kws):

            obj = nstype(get_interface())
            _intrfs = obj(**kws)
            children.appendleft(_intrfs)

            return _intrfs

        def get_interface():

            if uuid not in __nsmap__:
                n = NsInterface(get_uuid,

                                get_keys,
                                getter,
                                setter,
                                get_dct,
                                get_ns,
                                get_parent,
                                get_children,
                                child,
                                tree,
                                get_time
                                )
                __nsmap__[uuid] = n
                return n

            return __nsmap__[uuid]

        return get_interface()

    return _func_object


__root__ = nstype()


def dict_eq(dct1, dct2):
    if len(dct1) != len(dct2):
        return False
    else:
        return len(dct1.items() & dct2.items()) == len(dct1)


def _tt(kwargs):
    for k, v in kwargs.items():
        if isinstance(v, NsObject):
            yield k, v.interface
        else:
            yield k, v


import typing
from rich import repr as _repr


@_repr.auto
@dataclasses.dataclass
class Version:
    uuid: str
    changes: dict
    islast: bool
    time: float
    time_iso: str = None

    def __post_init__(self):
        self.time_iso = datetime.datetime.fromtimestamp(self.time).isoformat()

    def get_object(self):
        return NsObject.from_uuid(self.uuid)


import datetime


@_repr.auto
@dataclasses.dataclass
class VersionNode:
    version: Version
    children: list = dataclasses.field(default_factory=list)

    def __iter__(self):
        return iter(dataclasses.asdict(self).items())

    def find_last(self):
        res = list(self.find_leafs())
        res.sort(key=lambda x: x.version.time, reverse=True)
        return res[0]

    def find_leafs(self, inverse=False):
        return VersionFinder(value=not inverse,
                             deep=True,
                             condition=lambda node, value: node.version.islast == value)(self)

    def find_changes(self, changes: dict):
        return VersionFinder(value=changes,
                             deep=True,
                             condition=lambda node, value: node.version.changes == value)(self)

    def find_uuid(self, uuid: str):
        return VersionFinder(value=uuid,
                             deep=True,
                             condition=lambda node, value: node.version.uuid == value)(self)

    def get_interface(self) -> NsInterface:
        return __nsmap__[self.uuid]

    def get_insobject(self):
        return NsObject(self.get_interface())


class VersionFinderCheck:
    node: VersionNode
    value: typing.Any


@dataclasses.dataclass
class VersionFinder:
    condition: '(node, value: VersionNode) -> bool'
    value: typing.Optional[typing.Any] = None
    deep: bool = False

    def find(self, node: VersionNode, value: typing.Any, deep: bool):
        target = []
        if self.condition(node, value):
            target.append(node)
        _i = 0
        while True:
            if (not deep) and (len(target) > 0):
                break
            elif _i >= len(node.children):
                break
            target.extend(self.find(node.children[_i], value, deep))
            _i += 1
        return target

    def __call__(self, node: VersionNode, value=None, deep=None):
        if value is None:
            value = self.value
        if deep is None:
            deep = self.deep
        return self.find(node, value, deep)


islast_filter = VersionFinder(condition=lambda node, value: node.version.islast == value,
                              value=True)
uuid_filter = VersionFinder(condition=lambda node, value: node.uuid == value, deep=True)

from rich.pretty import Pretty, pretty_repr


class NsObject:
    __slots__ = ['interface']

    def __new__(cls, interface=None, /, **kwargs):
        if interface is None:
            interface = __root__(**dict(_tt(kwargs)))

        elif len(kwargs) > 0:

            interface = interface.child(**dict(_tt(kwargs)))
        if interface.get_uuid() in __nsobjmap__:
            return __nsobjmap__[interface.get_uuid()]
        self = super().__new__(cls)
        self.interface = interface
        __nsobjmap__[interface.get_uuid()] = self
        return self

    @classmethod
    def from_uuid(cls, uuid):
        return cls(__nsmap__[uuid])

    def __repr__(self):
        pretty = pretty_repr(
            dict(cls=self.__class__.__qualname__, data=dict(self), version=dataclasses.asdict(self.version)))

        return pretty

    def __init__(self, interface=None, /, **kwargs):
        if interface is None:
            interface = __root__(**dict(_tt(kwargs)))
        elif len(kwargs) > 0:

            interface = interface.child(**dict(_tt(kwargs)))

        self.interface = interface

    def __dir__(self):
        return list(set(super().__dir__()) | self.interface.get_keys())

    def __getattr__(self, item):
        if item.startswith('_') or (item in self.__slots__):
            return super().__getattribute__(item)
        elif item in self.interface.get_keys():
            v = self.interface.getter(item)
            if isinstance(v, NsInterface):
                return self.__class__(v)
            return v

        else:
            return super().__getattribute__(item)

    def __setattr__(self, item, val):
        if item.startswith('_') or (item in self.__slots__):
            return super().__setattr__(item, val)


        else:
            if isinstance(val, self.__class__):
                self.interface.setter(item, val.interface)
            else:
                self.interface.setter(item, val)

    def __getitem__(self, item):
        return self.interface.getter(item)

    def __setitem__(self, item, val):

        if isinstance(val, self.__class__):
            self.interface.setter(item, val.interface)
        else:
            self.interface.setter(item, val)

    @property
    def islast(self):

        return len(self.interface.get_children()) == 0

    def get_last(self):
        if self.islast:
            return self
        else:
            return NsObject(self.interface.get_children()[0]).get_last()

    def version_tree(self):
        return VersionNode(
            version=self.version,
            children=[NsObject(ch).version_tree() \
                      for ch in self.interface.get_children()])

    def version_tree_dict(self) -> dict:
        return dataclasses.asdict(self.version_tree())

    @property
    def version(self) -> Version:
        return Version(
            uuid=self.interface.get_uuid(),
            changes=self.interface.get_dct(),
            islast=self.islast,
            time=self.interface.get_time()
        )

    def make_child(self, **kwargs):
        return NsObject(self.interface.child(**kwargs))

    def __iter__(self):
        return iter(self.interface.tree().items())


def test():
    r11 = NsObject(a=6, b=8)
    r21 = NsObject(x=6, y=8, z=6)
    r31 = r21.make_child(y=18, z=16)

    r41 = NsObject(start=r31, end=r21, tags=r11)

    r41.tags = {'a': 5, 'c': 7}
    r41.tags = {'a': 51, 'd': 11}
    r41.end = dict(z=99)
    rrrr = r11.make_child(ff=666)
    rich.print(r11.version_tree_dict())

    rich.print(r41.version_tree_dict())
    rich.print(r11.version_tree().find_leafs())
