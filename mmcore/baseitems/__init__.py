from __future__ import annotations, absolute_import

__all__ = ['Base', 'Versioned', 'Identifiable', 'Item', 'GeometryItem', 'DictableItem',
           'DataviewInterface', 'Dataview', 'DataviewDescriptor', 'Metadata', 'ReprData', 'GeomConversionMap',
           'GeomDataItem', "Matchable"]

#  Copyright (c) 2022. Computational Geometry,(int, bool, bytes, float, str) Digital Engineering and Optimizing your construction processe"

import base64
import gzip
import itertools
import json
import typing
import uuid
from abc import ABCMeta, abstractmethod
from datetime import datetime
from json import JSONEncoder
from typing import Callable, Generator, ItemsView, KeysView, Mapping, Sequence, Union

import numpy as np
import pydantic
from cxmdata import BasicTypes
from mmcore.baseitems.descriptors import DataDescriptor, Descriptor, NoDataDescriptor
from mmcore.collection.traversal import traverse
from typing_extensions import ParamSpec

collection_to_dict = traverse(lambda x: x.to_dict(), traverse_seq=True, traverse_dict=False)
T = typing.TypeVar('T')  # Any type.
KT = typing.TypeVar('KT')  # Key type.
VT = typing.TypeVar('VT')  # Value type.
T_co = typing.TypeVar('T_co', covariant=True)  # Any type covariant containers.
T_contra = typing.TypeVar('T_contra', contravariant=True)  # Ditto contravariant.
P = ParamSpec('P')


def strong_attr_items(data) -> ItemsView:
    if isinstance(data, Mapping):
        return data.items()
    elif isinstance(data, Sequence):
        return [strong_attr_items(d) for d in data]

    else:

        keys = {}
        for key in dir(data):
            if not key.startswith("_"):
                g = getattr(data, key)
                if not isinstance(g, property) and isinstance(g, Callable):
                    continue
                else:
                    keys[key] = getattr(data, key)

        return keys.items()


def strong_attr_names(data) -> KeysView:
    return dict(strong_attr_items(data)).keys()


def wrp(dt):
    x = {}
    while True:
        if hasattr(dt, 'to_dict'):
            return dt.to_dict()
        elif isinstance(dt, Sequence) and not isinstance(x, str):
            for i in dt:
                return wrp(dt)
        elif isinstance(dt, dict):
            for k, v in dt.items():
                wrp(dt)


class Now(str):
    def __new__(cls, *a, **kw):
        d = datetime.now()
        instance = str.__new__(cls, d.isoformat())
        instance.__datetime__ = d
        return instance


class MatchableType(type):
    def __new__(mcs, name, bases, dct, with_slots=True, **kws):
        try:
            assert dct["__match_args__"]
            if with_slots:
                dct["__slots__"] = dct["__match_args__"]
            return super().__new__(mcs, name, bases + (object,), dct, **kws)
        except KeyError as err:
            # print(err)
            raise


class BaseI:
    """
    Base Abstract class
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.__call__(**kwargs)

    def __call__(self, **kwargs):
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        return self


class Base(Callable):
    """
    Base Abstract class
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.__call__(**kwargs)

    def __call__(self, **kwargs):
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        return self


class Versioned(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _version(self):
        self.version = Now()

    def __eq__(self, other):
        return hex(self.version) == hex(other.version)

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        self._version()


class VersionedI(BaseI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _version(self):
        self.version = Now()

    def __eq__(self, other):
        return hex(self.version) == hex(other.version)

    def __call__(self, *args, **kwargs):
        super().__call__(**kwargs)
        self._version()


class IdentifiableI(VersionedI):
    def __init__(self, **kwargs):
        self._uuid = uuid.uuid4()
        super().__init__(**kwargs)

    @property
    def uid(self):
        return hex(id(self))

    @property
    def uuid(self):
        return str(self._uuid)

    @uuid.setter
    def uuid(self, v):
        self._uuid = v

    def __hash__(self):
        ...


class Identifiable(Versioned):
    def __init__(self, *args, **kwargs):
        self._uuid = uuid.uuid4()
        super().__init__(*args, **kwargs)

    @property
    def uid(self):
        return hex(id(self))

    @property
    def uuid(self):
        return str(self._uuid)

    @uuid.setter
    def uuid(self, v):
        self._uuid = v

    def __hash__(self):
        ...


class DataviewInterface(metaclass=ABCMeta):
    include: list[str] = []
    replace: dict[str, str] = dict()

    def __init__(self, **kwargs):
        super().__init__()
        for name, constrain in kwargs.items():
            if constrain is not None: setattr(self, name, constrain)

    @abstractmethod
    def __get_dict__(self, instance, owner):
        pass


class Dataview(DataviewInterface):
    include: list[str] = []
    replace: dict[str, str] = dict()

    def __get_dict__(self, instance, owner):
        get_dict = {}
        for k in self.include:
            get_dict[self.replace[k] if k in self.replace.keys() else k] = getattr(instance, k)
        return get_dict


class DataviewDescriptor(Dataview):
    include: list[str] = []
    replace: dict[str, str] = dict()

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner) -> dict:
        return self.__get_dict__(instance, owner)


class Metadata(DataviewDescriptor):
    include = ["uid", "uuid", "dtype", "version"]
    replace = {
        "_dtype": "dtype"
        }


class ReprData(DataviewDescriptor):
    include = []
    replace = {
        "_dtype": "dtype"
        }

    def __init__(self, *include, **kwargs):
        super().__init__(**kwargs)
        list_include = list(include)
        list_include.extend(self.include)
        self.include = list_include

    def __set_name__(self, owner, name):
        self.name = name


class ItemFormatter:
    _dtype = "ItemFormatter"
    __representation: ReprData
    representation = ReprData()

    @property
    def dtype(self):
        return self._dtype

    @property
    def format_spec(self):
        return self.representation

    def __format__(self, format_spec: dict = None):
        if format_spec is not None:
            self.representation |= format_spec
        return "{}({})".format(self.__class__.__name__,
                               "".join([f"{k}={v}, " for k, v in self.representation.items()])[:-1][:-1])

    def __str__(self):
        """
        Item Format str
        """

        return self.__format__()

    def __repr__(self):
        return f"< {self.__format__()} at {hex(id(self))} >"


class ItemEncoder(JSONEncoder):

    def default(self, o):
        try:
            if isinstance(o, pydantic.BaseModel):
                return o.dict()
            else:
                return o.to_dict()

        except:
            raise TypeError(f'Object of type {o.__class__.__name__} '
                            f'is not JSON serializable')


class Item(Identifiable, ItemFormatter):
    metadata = Metadata()
    representation = ReprData("version", "uid", "dtype")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)


class GeomConversionMap(DataviewDescriptor):
    include = ["to_rhino", "to_compas"]
    replace = {

        }


class GeometryItem(Item):
    data = GeomConversionMap()

    def to_rhino(self) -> Union[list[float], Generator]:
        ...

    def to_compas(self) -> Union[list[float], Generator]:
        ...


class DataItem(Item):
    """
    Check name

    MModelException: The class name does not match the descriptor signature!
    classname: B, input name: A

    """
    _exclude = {"exclude", "custom_fields", "default_fields"}
    dtype: str

    @property
    def metadata(self):
        return dict(metadata=dict(self.custom_fields))

    @property
    def custom_fields(self):
        _fields = []

        for k in self.__dict__.keys():
            if not (k in self.exclude) and (not hasattr(self.__class__, k)):
                _fields.append((k, getattr(self, k)))
        return _fields

    @property
    def default_fields(self):
        _fields = []
        for k in self.__dict__.keys():
            if not (k in self.exclude) and hasattr(self.__class__, k):
                _fields.append((k, getattr(self, k)))
        _fields.append(("metadata", dict()))
        return _fields

    @property
    def mro_items(self):
        l = []
        for base in self.__class__.__bases__:
            if issubclass(base, DataItem):
                l.append(base)
        return l

    @property
    def exclude(self):
        for base in self.mro_items:
            self._exclude.update(base._exclude)
        return self._exclude

    @exclude.setter
    def exclude(self, v):
        self._exclude.add(v)


class FieldItem(Item):
    fields = []
    exclude = ("fields", "base_fields", "custom_fields", "del_keys", "__array__", "uid")

    def __init__(self, *args, **kwargs):

        self.custom_fields = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.custom_fields = []
        self.base_fields = []
        super().__call__(*args, **kwargs)

        self.check_fields()

    def check_fields(self):
        self.__class__.fields = []
        self.custom_fields = []
        for k in self.__dict__.keys():
            if k in self.exclude:
                continue
            else:
                if hasattr(self.__class__, k):
                    self.__class__.fields.append(k)
                    self.base_fields.append(k)
                else:

                    self.custom_fields.append(k)


class DictableItem(FieldItem, ItemFormatter):
    fields = []

    representation = ReprData("uid", "version")
    exclude = ('args', 'kw', 'representation', 'aliases', "fields", "uid", "__array__", "_dtype")
    metadata = Metadata(include=["uuid", "dtype", "version", "custom_fields", "base_fields"])

    def __hash__(self):
        st = ""
        for k, v in self.to_dict().items():

            if not ((k in self.exclude) or (k == "metadata")):
                try:
                    iter(v)

                    if not isinstance(v, str):
                        # #print(f'hash iter {v}')
                        st.join([hex(int(n)) for n in np.asarray(np.ndarray(v) * 100, _dtype=int)])
                    else:
                        continue
                except:
                    # print(f'hash not iter {v}')
                    if isinstance(v, int) or isinstance(v, float):
                        st += hex(int(v * 100))
                    else:
                        continue

        return st

    def to_dict(self):
        st: dict = {}

        for k, v in self.__dict__.items():
            k = k[1:] if k[0] == "_" else k
            if k in self.exclude:
                continue

            else:

                try:
                    iter(v)
                    if isinstance(list(itertools.chain(v))[0], DictableItem):
                        dct = list(map(lambda x: x.to_dict(), v))
                    else:
                        dct = v
                except:
                    if isinstance(v, DictableItem):
                        dct = v.to_dict()
                    else:

                        dct = v

                if k in self.base_fields:

                    st |= {k: dct}
                else:
                    pass
        st["metadata"] = self.metadata
        return st

    def encode(self, **kwargs):
        return ItemEncoder(**kwargs).encode(self)

    def to_data(self):
        data = self.to_dict()
        data |= {"guid": self.uuid}
        return data

    def to_json(self, **kwargs):
        return self.encode(**kwargs)

    def b64encode(self):
        return base64.b64encode(self.gzip_encode())

    def gzip_encode(self):
        return gzip.compress(self.encode().encode(), compresslevel=9)

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)

    def to_compas(self):
        ...

    def to_occ(self):
        ...


class GeomDataItem(DictableItem, GeometryItem):
    exclude = ["to_rhino", "to_compas", "to_occ"]
    exclude.extend(DictableItem.exclude)

    def to_dict(self):
        dct = super().to_dict()
        dct["data"] = self.data["data"]
        return dct


# New Style Classes
# ----------------------------------------------------------------------------------------------------------------------
from mmcore.baseitems import descriptors
from types import prepare_class


class MMtype(type):
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        prepare_class(metacls, name, bases, kwds)


from mmcore.collection.basic import OrderedSet


class Follower:

    def __init_subclass__(cls, follow, include_parents_follow=False, **kwargs):
        cls.follow = OrderedSet(list(follow))
        if include_parents_follow:
            cls.follow.update(cls.__base__.follow)
        super().__init_subclass__(**kwargs)

    ...


class MatchingDecorator(Follower, follow=('__match_args__',)):
    def __init__(self, func: Callable[P, T]):
        self._func = func
        self.name = func.__name__

    def __call__(self, *args: P.args, **kwargs: P.kwargs):
        print(f'{self._func.__name__} was called with {kwargs}')
        return self._func(*args, **kwargs)


class Matchable(object):
    """
    New style baseclass version.
    Matchable can be initialized from *args if they are listed in the __matched_args__ field.
    Generic __repr__ use __match_args__ by default, but you can use __repr_ignore__ from change.

    """
    match_args: tuple[str] = ()
    __match_args__ = ()
    __match_args__ += match_args

    properties = descriptors.UserDataProperties()
    userdata = descriptors.UserData()

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._uuid = uuid.uuid4().__str__()
        self.__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if args:
            if len(self.__match_args__) + 1 < len(args):
                raise TypeError(
                    f"length self.__match_args__ = {len(self.__match_args__)} > length *args = {len(args)}, {args}")
            else:
                kwargs |= dict(zip(self.__match_args__[:len(args)], args))

        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:

        s = f"{self.__class__.__name__}("
        for k in self.__class__.__match_args__:
            s += f"{k}={getattr(self, k)}, "
        return s + ")"

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, value):
        self._uuid = uuid

    def _dump_keys(self) -> KeysView:

        return strong_attr_names(self)

    def to_dict(self):
        def wrp(dt):

            if type(dt) in BasicTypes:
                return dt
            elif hasattr(dt, "to_dict"):
                return dict(strong_attr_items(self))
            elif isinstance(dt, Mapping):
                dct = {}
                for k, v in dt.items():
                    dct[k] = wrp(v)
                return dct
            elif isinstance(dt, Sequence):
                return [wrp(d) for d in dt]
            else:
                pass

        return wrp(self)

    def toJSON(self):

        return json.dumps(self, default=self.__class__.to_dict,
                          sort_keys=True, indent=3)


class MatchableItem(Matchable, Base):
    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, value):
        raise AttributeError("You dont can set UUID from any object.")

    def __init__(self, *args, **kwargs):
        self._uuid = uuid.uuid4()
        Base.__init__(self, **kwargs)
        Matchable.__init__(self, *args)

    def __call__(self, *args, **kwargs):
        super(Matchable, self).__call__(*args)
        return Base.__call__(self, **kwargs)

    def __repr__(self):
        return Matchable.__repr__(self) + f" at {self.uuid}"


class FromKet():
    match_targets = "__match_args__",

    def __init__(self, func: Callable[P, T]):
        self._func = func
        self.name = func.__name__


class Trait:
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    def __get__(self, instance, owner):
        return instance.__dict__[self.key]

    def __set__(self, instance, value):
        if self.minimum < value < self.maximum:
            instance.__dict__[self.key] = value
        else:
            raise ValueError("value not in range")

    def __set_name__(self, owner, name):
        self.key = name


import weakref


class WeakAttribute:
    def __get__(self, instance, owner):
        return instance.__dict__[self.name]()

    def __set__(self, instance, value):
        instance.__dict__[self.name] = weakref.ref(value)

    # this is the new initializer:
    def __set_name__(self, owner, name):
        self.name = name


class TreeNode:
    parent = WeakAttribute()

    def __init__(self, parent):
        self.parent = parent
