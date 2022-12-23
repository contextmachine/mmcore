import os
import pickle
from abc import ABC
from typing import Any, ItemsView, KeysView, Mapping, TypeVar, ValuesView

import redis_om

KeyProxy = type(dict().keys())

REDIS_CONN = redis_om.get_redis_connection(url=os.getenv("REDIS_URL"), db=0)
ConnType = redis_om.connections.redis.Redis
T = TypeVar('T')  # Any type.
KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.


class SlimRedisMapping(Mapping[str, Any], ABC):
    __name__ = "SlimRedisMapping"
    __qualname__ = "SlimRedisMapping"

    def __init__(self, mk: str = "ug:test:", conn: ConnType = REDIS_CONN):
        Mapping.__init__(self, 1)
        self._keys = []

        self.mk = mk
        self.conn = conn

    def __getitem__(self, pk: str) -> Any:

        return pickle.loads(bytes.fromhex(self.conn.get(self.mk + pk)))

    def __setitem__(self, pk: str, item: Any) -> None:
        self.conn.set(self.mk + pk, pickle.dumps(item).hex())

    def keys(self) -> KeysView[str]:

        return KeysView._copy_with(self.conn.keys(self.mk + "*"))

    def __iter__(self):
        pass

    def __len__(self):
        pass

    def items(self) -> ItemsView:
        self._items = []
        for k in self._keys:
            self._items.append((k, self[k]))
        return ItemsView(self._items)

    def values(self) -> KeysView:
        def generate():
            for k in self._keys:
                yield self[k]

        return ValuesView(list(generate()))


class RC(dict):
    def __init__(self, mk="ug:test:", conn=None):
        dict.__init__(self)
        self._keys = []

        self.root_key = mk
        self.conn = conn

    def __getitem__(self, pk):

        return unpickle(self.conn.get(self.root_key + pk))

    def __setitem__(self, pk, item):
        if pk not in self._keys:
            self._keys.append(pk)
        self.conn.set(self.root_key + pk, topickle(item))

    def keys(self) -> KeysView:

        return KeysView(self._keys)

    def items(self) -> ItemsView:
        self._items = []
        for k in self._keys:
            self._items.append((k, self[k]))
        return ItemsView(self._items)

    def values(self) -> KeysView:
        def generate():
            for k in self._keys:
                yield self[k]

        return ValuesView(list(generate()))


def unpickle(get_result):
    return pickle.loads(bytes.fromhex(get_result))


def topickle(item):
    return pickle.dumps(item).hex()
