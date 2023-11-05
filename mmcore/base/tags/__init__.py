import dataclasses
import functools
import pickle
import uuid as _uuid
from collections import Counter, namedtuple

__databases__ = dict()
__items__ = dict()

import numpy as np


class TagDBItem:
    __slots__ = ["index", "dbid"]

    def __new__(cls, index, dbid):
        ix = str(index)

        if ix in __items__[dbid]:
            return __items__[dbid][ix]
        else:
            obj = super().__new__(cls)
            obj.index = ix
            obj.dbid = dbid

            __items__[dbid][ix] = obj
            return obj

    def __getstate__(self):
        return {"index": self.index, "dbid": self.dbid}

    def __setstate__(self, state):
        ix, dbid = state.get("index"), state.get("dbid")
        self.index = ix
        self.dbid = dbid
        __items__[dbid][ix] = self

    def todict(self):
        return dict(self.__iter__())

    def __deepcopy__(self, memodict={}):
        return self

    def __copy__(self):
        return self

    @property
    def __annotations__(self):
        return self.db.__annotations__

    @property
    def db(self) -> 'TagDB':
        return __databases__[self.dbid]

    def __getitem__(self, field):
        return self.db.get_column_item(field, self.index)

    def __setitem__(self, field, v):
        self.db.set_column_item(field, self.index, v)

    def __iter__(self):
        # yield "name", self.index
        for name in self.db.names:
            yield name, self.db.get_column_item(name, self.index)

    def set(self, item: dict):
        for k, v in dict(item).items():
            self.db.set_column_item(k, self.index, v)

    def get_as_type_instance(self):
        return self.db.make_dataclass()(**dict(self.__iter__()))

    def __ior__(self, other):
        for k, v in other.items():
            self.db.set_column_item(k, self.index, v)
        return self


class CustomTagDBItem:
    ...


TagDBOverrideEvent = namedtuple("TagDBOverrideEvent", ["field", "index", "old", "new", "timestamp"])


class TagDBIterator:
    def __init__(self, dbid):
        super().__init__()
        self._owner_uuid = dbid
        self._cursor = -1
        self._iter = iter(__items__[self._owner_uuid].values())

    def __iter__(self):
        return self

    def __next__(self):
        return dict(self._iter.__next__())


class TagDB:
    columns: dict
    defaults = dict()
    overrides = list()
    types = dict()

    # {'mfb_sw_panel_117_17_1':{
    #   'tag':345
    # }

    def __new__(cls, uuid=None, strong_types=False, resolve_types=True, conn=None):

        self = super().__new__(cls)
        if uuid is None:
            uuid = _uuid.uuid4().hex
        self.uuid = uuid

        __databases__[self.uuid] = self
        __items__[self.uuid] = dict()
        self.names = []
        self.columns = dict()
        self.overrides = list()
        self.defaults = dict()
        self.types = dict()
        self.strong_types = strong_types
        self.resolve_types = resolve_types
        self.conn = conn

        return self

    @classmethod
    def load(cls, uuid, conn=None):
        if conn is not None:
            print("Loaded from redis")
            data = pickle.loads(conn.get(f"mmcore:api:tagdb:dump:{uuid.replace('_', ':')}"))
            data.conn = conn
            return data

        else:
            with open(f"{os.getenv('HOME')}/.cxm/{uuid}.cache.pkl", "rb") as f:
                print("Loaded from file")
                return pickle.load(f)

    def __getstate__(self):
        dct = dict(self.__dict__)

        del dct["conn"]
        return dct

    def __setstate__(self, state):

        __databases__[state["uuid"]] = self
        __items__[state["uuid"]] = dict()

        for k, v in state.items():
            if not k == "conn":
                self.__dict__[k] = v

    def make_dataclass(self):
        import types
        dcls = dataclasses.make_dataclass(self.uuid, fields=list(self.get_fields()))
        setattr(types, self.uuid, dcls)
        return dcls

    def get_annotations(self):
        return dict((name, self.types.get(name)) for name in self.names)

    def get_fields(self):
        return [(name, self.types.get(name), dataclasses.field(default=self.defaults.get(name))) for name in self.names]

    def save(self):

        if self.conn is not None:
            self.conn.set(f"mmcore:api:tagdb:dump:{self.uuid.replace('_', ':')}", pickle.dumps(self))
        else:
            with open(f"{os.getenv('HOME')}/.cxm/{self.uuid}.cache.pkl", "wb") as f:
                pickle.dump(self, f)

    def __setitem__(self, key, value):
        item = TagDBItem(key, self.uuid)
        item.set(value)
        del item

    def __getitem__(self, item) -> TagDBItem:

        return TagDBItem(item, self.uuid)

    def get_column(self, k) -> dict:

        return self.columns[k]

    def set_column(self, k, v):
        if k not in self.names:
            self.add_column(k)
        self.columns[k] = v

    def set_column_item(self, field, item, v):
        if field not in self.columns.keys():
            self.add_column(field, default=None, column_type=type(v))

        self.columns[field][item] = self.types[field](v)

    def get_column_item(self, field, item):

        return self.columns[field].get(item, self.defaults[field])

    def update_column(self, field, value):
        for k, v in value.items():
            self.set_column_item(field, k, v)

    def add_column(self, name, default=None, column_type=None):
        if name not in self.names:
            self.names.append(name)
        self.columns[name] = dict()
        if column_type is None:
            if self.resolve_types:
                if default is not None:
                    column_type = type(default)
        else:
            if default is not None:
                if self.strong_types:
                    if type(default) != column_type:
                        raise TypeError(f"Column type: {column_type} != type of default value: {default}\n\t "
                                        f"To disable this error set 'strong_typing=False' for this db. ")
            else:
                if self.resolve_types:
                    default = column_type()

        self.types.__setitem__(name, column_type)
        self.defaults.__setitem__(name, default)

    def get_row(self, index):
        return TagDBItem(index, self.uuid)

    def get_column_counter(self, name):
        return Counter(self.get_column(name).values())

    def items(self):
        return __items__[self.uuid]

    def __iter__(self):
        return TagDBIterator(self.uuid)

    @property
    def item_names(self):
        return self.items().keys()


__soas__ = dict()


class SoAField:
    def __init__(self, default=None):
        self.default = default
        self.data = dict()
        self._parent = None

    @property
    def parent(self):
        return __soas__.get(self._parent)

    @parent.setter
    def parent(self, v):
        if isinstance(v, SoA):
            self._parent = v.name
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
    return __soas__[self._parent]


class SoAItem:

    def __init__(self, uuid, parent_id):
        super().__init__()
        self._parent = parent_id
        self._uuid = uuid

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
            return object.__getattribute__(self, '_parent')

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
        return f'component(name={self._parent} data={dict(self)})'


from dataclasses import dataclass


@dataclass
class SoAProps:
    allow_nulls: bool = False


class SoA:
    def __init__(self, name: str, props: SoAProps = None, **fields):
        assert name
        self.name = name

        self.props = props if props else SoAProps()
        self.fields = dict()
        self.__items__ = dict()
        __soas__[name] = self
        for k, v in fields.items():
            self.add_field(k, v)

    def add_field(self, key, fld):
        fld.parent = self
        self.fields[key] = fld

    def remove_field(self, key):
        del self.fields[key]

    def __getitem__(self, uuid):

        return self.__items__[uuid]

    def __setitem__(self, uuid, v):
        if uuid not in self.__items__.keys():
            self.__items__[uuid] = SoAItem(uuid, self.name)

        self.__items__[uuid].update(v)

    def __contains__(self, item):
        return self.__items__.__contains__(item)


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

        return ctor

    return wrapper


def todict(obj):
    if hasattr(obj, 'items'):
        return {k: todict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [todict(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def request_component(name, uuid):
    return __soas__[name][uuid]


import os
from collections import defaultdict
import threading as th
import queue

__jobs_io__ = dict(inputs=dict(), outputs=dict(), links=dict(), funcs=defaultdict(lambda x: x), funcs_link=dict(),
                   queue=queue.Queue())


def getlink(key):
    return __jobs_io__['links'].get(key, None)


def setlink(outkey, inpkey):
    __jobs_io__['links'][inpkey] = outkey
    __jobs_io__['queue'].put(inpkey)


def dellink(inpkey):
    del __jobs_io__['links'][inpkey]


def getout(key):
    return __jobs_io__['outputs'].get(key, None)


def exec_childs(key):
    for k, v in __jobs_io__['links'].items():
        if v == key:
            __jobs_io__['queue'].put(k)


def setout(key, v):
    __jobs_io__['outputs'][key] = v
    exec_childs(key)


def updout(key, dat):
    __jobs_io__['outputs'][key].update(dat)
    exec_childs(key)


def getinp(key):
    return __jobs_io__['inputs'].get(key, None)


def getlink_val(key):
    return getout(getlink(key))


class JobCmp:
    __cls_key__ = ''

    def __init__(self, key, data=None):
        self.key = key

        __jobs_io__[self.__cls_key__][self.key] = data if data is not None else dict()

    def __iter__(self):
        return iter(__jobs_io__[self.__cls_key__][self.key].items())


class JobInput(JobCmp):
    __cls_key__ = 'inputs'

    @property
    def data(self):
        return getlink_val(self.key)

    def set_link(self, key):
        setlink(outkey=key, inpkey=self.key)

    def del_link(self):
        dellink(self.key)


class JobOutput(JobCmp):
    __cls_key__ = 'outputs'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_link(self, key):
        setlink(outkey=self.key, inpkey=key)

    def del_link(self, key):
        dellink(key)

    @property
    def data(self):
        return getout(self.key)

    @data.setter
    def data(self, v):
        setout(self.key, v)

    def update(self, v):
        updout(self.key, v)


class JobNode:
    def __init__(self, key):
        self.key = key
        self._input = JobInput(self.key)
        self._output = JobOutput(self.key)

        __jobs_io__['funcs_link'][self.key] = 'default'

    @property
    def resolver(self):
        return __jobs_io__['funcs'][__jobs_io__['funcs_link'][self.key]]

    def execute(self):
        self._output.data = self.resolver(self._input.data)

    def bind(self, name=None):
        def wrp(fun):
            nonlocal name
            if name is None:
                name = fun.__name__

            __jobs_io__['funcs'][name] = fun
            __jobs_io__['funcs_link'][self.key] = name
            return fun

        return wrp


class UpdSystem:
    """
    supd=UpdSystem()
supd.start()


foon=JobNode('foo')

o3=JobOutput('foot', dict(a=9,b=1))
f1=JobInput('f1')

__jobs_io__['links'][f1.key]=foon.key
@foon.bind('add_node')
def addnode(ab):
    a,b=ab.values()
    return {'c':a+b}

foon._input.set_link(o3.key)


o3.update(dict(b=44))
    """

    def __init__(self):
        self.stop = False

    def loop(self):
        while True:
            if self.stop:
                print(" stopping")
                break
            else:
                q = __jobs_io__['queue']

                if not q.empty():
                    key = q.get()
                    print(key, getlink_val(key))
                    if key in __jobs_io__['funcs_link'].keys():

                        res = __jobs_io__['funcs'][__jobs_io__['funcs_link'][key]](getlink_val(key))
                        setout(key, res)
                        print(key, res, " done!")
                    else:
                        print(key, " pass!")

        print(" stop")

    def start(self):
        self._thread = th.Thread(target=self.loop, daemon=True)
        self._thread.start()
