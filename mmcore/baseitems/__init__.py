from __future__ import annotations, absolute_import

__all__ = ['Base', 'Versioned', 'Identifiable', 'Item', 'GeometryItem', 'DictableItem',
           'DataviewInterface', 'Dataview', 'DataviewDescriptor', 'Metadata', 'ReprData', 'GeomConversionMap',
           'GeomDataItem', "Matchable"]

#  Copyright (c) 2022. Computational Geometry,(int, bool, bytes, float, str) Digital Engineering and Optimizing your construction processe"

import base64
import gzip
import itertools
import os
import subprocess
import threading
import time
import typing
import uuid
from abc import ABCMeta, abstractmethod
from datetime import datetime
from json import JSONEncoder
from typing import Callable, Generator, ItemsView, KeysView, Mapping, Sequence, Union

import numpy as np
import pydantic
from typing_extensions import ParamSpec

from mmcore.baseitems.descriptors import DataDescriptor, Descriptor, NoDataDescriptor
from mmcore.collections.traversal import traverse

BasicTypes = []
collection_to_dict = traverse(lambda x: x.to_dict(), traverse_seq=True, traverse_dict=False)
T = typing.TypeVar('T')  # Any type.
KT = typing.TypeVar('KT')  # Key type.
VT = typing.TypeVar('VT')  #
# Value type.
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
from mmcore.collections import OrderedSet


class resolve:
    """
    class A:
        @resolve(with_meta="__new__")
        def ha(obj, ): ...
    """

    def __new__(cls, with_meta="__new__", **kwargs):
        inst = super().__new__(cls)
        inst.__dict__ |= kwargs
        inst.with_meta = with_meta

        def decore(func):
            inst.func = func
            return inst

        return decore

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class MMtype(type):
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        ns = dict(super().__prepare__(name, bases, **kwds))

        return ns

    def __new__(mcs, name, bases, attrs, **kwargs):
        cl = super().__new__(mcs, name, bases, attrs, **kwargs)

        for attr in attrs:
            if hasattr(attr, "with_meta") and attr.with_meta == "__name__":
                cl.__dict__[attr]()
        return cl


class MMItem(metaclass=MMtype):
    __match_args__ = ()

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

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join([f"{k}={getattr(self, k)}" for k in self.__match_args__])}) at {id(self)}'

    @classmethod
    def resolve_match_args(cls):
        match_args = OrderedSet([])
        for base in cls.mro():
            if hasattr(base, "__match_args__"):
                match_args.extend(base.__match_args__)
        cls.__match_args__ = tuple(match_args)


class Matchable(object):
    """
    New style baseclass version.
    Matchable can be initialized from *args if they are listed in the __matched_args__ field.
    Generic __repr__ use __match_args__ by default, but you can use __repr_ignore__ from change.

    """
    match_args: tuple[str] = ()
    __match_args__ = ()
    __match_args__ += match_args
    repr_args = "__match_args__"
    properties = descriptors.UserDataProperties()
    userdata = descriptors.UserData()
    repr_exclude = ()

    def __new__(cls, *args, **kwargs):
        cls.resolve_match_args()
        inst = super().__new__(cls)

        return inst

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

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join([f"{k}={getattr(self, k)}" for k in self.__match_args__])}) at {id(self)}'

    @classmethod
    def resolve_match_args(cls):
        match_args = OrderedSet([])
        for base in cls.mro():
            if hasattr(base, "__match_args__"):
                match_args.extend(base.__match_args__)
        cls.__match_args__ = tuple(match_args)

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, value):
        self._uuid = uuid.uuid4().__str__()

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
            elif hasattr(dt, "Encode"):
                return dt.Encode()
            else:
                pass

        return wrp(self)

    @classmethod
    def encode(cls, o):
        return cls.ToJSON(o)

    def ToJSON(self, **kwargs):
        return json.dumps(self.to_dict(), **kwargs)


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
        return Matchable.__repr__(self)


class Entity(Matchable):
    type: str = "Entity"

    json = descriptors.JsonView("userdata", )

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join([f"{k}={getattr(self, k)}" for k in self.__match_args__])}) at {self.uuid}'


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


class simpleproperty:
    def __init__(self, expression=None, default=None):
        super().__init__()
        self.expression = expression
        self.default = default

    def __set_name__(self, owner, name):
        setattr(owner, "_" + name, self.default)
        self.name = name

    def __set__(self, instance, val):
        setattr(instance, "_" + self.name, val)

    def __get__(self, instance, owner):
        if self.expression:
            return self.expression(instance, self.name, getattr(instance, "_" + self.name))
        else:

            return getattr(instance, "_" + self.name)


import socket, jinja2, json


def node_no_params(sock, cont):
    sock.send(cont)
    dat2 = sock.recv(9999)
    d2 = dat2[:-len(cont)]
    print(d2)
    dtt = d2.decode()
    print(dtt)
    return json.loads(dtt)


def node_with_params(tmp, je, jsn):
    tmpl = je.from_string(tmp).render(jsn=json.dumps(jsn)).encode()
    return ex(tmpl)


class NodeSocketResolver:
    def __init__(self, unix_addr="/tmp/cxm.sock",
                 node_code_path=__file__.replace("baseitems/__init__.py", "js/temp.js"), size=8096):
        self.unix_addr = unix_addr
        self.size = size
        self.traffic = 0
        self.node_code_path = node_code_path
        self.reload()

    def send(self, data):
        self.sock.send(data)
        dat = self.sock.recv(self.size)
        self.traffic += len(data)
        self.traffic += len(dat)
        return dat[:-len(data)]

    def execute(self, code):
        result = self.send(code)
        return json.loads(result.decode())

    @property
    def reload_command(self):

        return f"bash {os.getcwd()}/bin/node_serve.sh"

    @property
    def node_serve_command(self):
        return f"node {self.node_code_path}"

    def node_serve_thread(self):
        th = threading.Thread(
            target=lambda: subprocess.Popen(list(self.node_serve_command.split(" ")), subprocess.PIPE))

        return th

    def reload(self):
        self.traffic = 0
        if hasattr(self, "thread"):
            self.thread.join(60)
        proc1 = subprocess.Popen(list(self.reload_command.split(" ")))
        proc1.wait()
        self.thread = self.node_serve_thread()
        self.thread.start()
        time.sleep(3)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.unix_addr)


class Obj3D(Matchable):
    __match_args__ = "name", "param1"
    new_temp = '((require) => {' \
               'const newObject = new THREE.Object3D();' \
               'localResult =  newObject.toJSON();})'
    call_temp = c2 = '((require) => {' \
                     'const newObject = objLoader.parse({{ jsn }});' \
                     'localResult = newObject.toJSON();})'
    properties = descriptors.UserDataProperties("param1", "uuid")
    three_fields = {"name", "userData"}

    def __init__(self, *args, is_root=False,conn=None, **kwargs ):
        self._je = jinja2.Environment()
        self.conn=conn
        dct = self.conn.execute(self.new_temp.encode())
        self.three_fields.update(set(dct["object"].keys()))
        kwargs |= dct["object"]
        super().__init__(*args, **kwargs, is_root=is_root, metadata=dct["metadata"])
        self._uuid = dct["object"]["uuid"]

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)

        return self

    @property
    def object3d(self):
        if self.is_root:
            return dict(object=dict((k, getattr(self, k)) for k in self.three_fields), metadata=self.metadata)
        else:
            return dict(object=dict((k, getattr(self, k)) for k in self.three_fields))

    @property
    def object(self):

        return dict((k, getattr(self, k)) for k in self.three_fields)

    @property
    def userData(self):

        return self.userdata
