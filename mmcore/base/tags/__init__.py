import dataclasses
import os
import pickle
import uuid as _uuid
from collections import Counter, namedtuple

__databases__ = dict()
__items__ = dict()

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
