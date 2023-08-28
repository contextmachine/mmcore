import pickle
import typing

from mmcore.services.redis import connect
from mmcore.services.redis.stream import encode_dict_with_bytes

rconn = connect.get_cloud_connection()


class Hdict:
    pk: str = 'api:mmcore:hsets:'

    def __init__(self, key: str, pk: str = None):
        super().__init__()
        if pk is not None:
            self.pk = pk
        self.key = key

    @property
    def full_key(self):
        return self.pk + self.key

    def __setitem__(self, key, value):
        rconn.hset(self.full_key, key, json.dumps(value))

    def __getitem__(self, key):
        return encode_dict_with_bytes(rconn.hget(self.full_key, key))

    def __delitem__(self, key):
        self.hdel(key)

    def hdel(self, key) -> bool:
        return bool(rconn.hdel(self.full_key, key))

    def items(self):
        return zip(self.keys(), self.values())

    def values(self):
        return list(encode_dict_with_bytes(rconn.hgetall(self.full_key)))

    def keys(self):
        return list(encode_dict_with_bytes(rconn.hkeys(self.full_key)))

    def __contains__(self, value):
        return rconn.hexists(self.full_key, value)

    def __len__(self):
        return rconn.hlen(self.full_key)

    def scan(self, match=None, count=None):
        def wrapiter():
            cursor = "0"
            while cursor != 0:
                cursor, data = rconn.hscan(self.full_key, cursor=cursor, match=match, count=count)
                yield from encode_dict_with_bytes(data).items()

        return wrapiter()

    def __iter__(self):
        return self.scan()


import gzip, json


class CompressedHdict(Hdict):
    """
    Hdict that compress values to gzip automatically
    """
    pk: str = 'api:mmcore:compressed:hsets:'

    def __setitem__(self, key, value: dict):
        rconn.hset(self.full_key, key, gzip.compress(json.dumps(value).encode()))

    def __getitem__(self, key):
        return encode_dict_with_bytes(gzip.decompress(rconn.hget(self.full_key, key)))

    def items(self):
        return list(encode_dict_with_bytes([gzip.decompress(i) for i in rconn.hgetall(self.full_key)]))

    def keys(self):
        return list(encode_dict_with_bytes(rconn.hkeys(self.full_key)))


class ExtendedHdict(Hdict):
    def __init__(self, key: str, pk: str = None, enable_gzip=False):
        if pk is None:
            pk = 'api:hsets:'
        super().__init__(key, pk=pk)
        if self['enable_gzip'] is None:
            rconn.hset(self.full_key, "__enable_gzip__", int(enable_gzip))

    @property
    def enable_gzip(self):

        return bool(int(rconn.hget(self.full_key, "__enable_gzip__")))

    @enable_gzip.setter
    def enable_gzip(self, v):

        raise AttributeError("Impossible change compress type of a existing Hdict object")

    def __setitem__(self, key, value: dict):
        if self.enable_gzip:
            rconn.hset(self.full_key, key, gzip.compress(json.dumps(value).encode()))
        else:
            rconn.hset(self.full_key, key, json.dumps(value).encode())

    def __getitem__(self, key):
        if self.enable_gzip:
            return encode_dict_with_bytes(gzip.decompress(rconn.hget(self.full_key, key)))
        else:
            return encode_dict_with_bytes(rconn.hget(self.full_key, key))


class GeometryHdict(CompressedHdict):
    """
    CompressedHdict with geometry specific
    """
    pk: str = 'api:mmcore:geometry:hsets:'


class PickleHdict(Hdict):
    """
    CompressedHdict with geometry specific
    """
    pk: str = 'api:mmcore:pickle:hsets:'

    def __setitem__(self, key, value: typing.Any):
        rconn.hset(self.full_key, key, pickle.dumps(value))

    def __getitem__(self, key):
        return pickle.loads(rconn.hget(self.full_key, key))

    def items(self):
        return list([pickle.loads(i) for i in rconn.hgetall(self.full_key)])

    def keys(self):
        return list(encode_dict_with_bytes(rconn.hkeys(self.full_key)))


class BindedDict(dict):
    def __init__(self, binded, key):
        super().__init__()
        self._key = key
        self._binded = binded
        self.full_key = binded.pk + self._key

    @property
    def wrapped(self):
        return self._binded

    @property
    def wrapkey(self):
        return self._key

    def __getitem__(self, item):
        return self._binded[self._key, item]

    def __setitem__(self, k, item):
        self._binded[self._key, k] = item

    def __delitem__(self, key):
        rconn.hdel(key[0], *key[1:])

    def items(self):
        return zip(self.keys(), self.values())

    def values(self):
        return list(encode_dict_with_bytes(rconn.hgetall(self.full_key)))

    def keys(self):
        return list(encode_dict_with_bytes(rconn.hkeys(self.full_key)))

    def __contains__(self, value):
        return rconn.hexists(self.full_key, value)

    def __len__(self):
        return rconn.hlen(self.full_key)

    def scan(self, match=None, count=None):
        def wrapiter():
            cursor = "0"
            while cursor != 0:
                cursor, data = rconn.hscan(self.full_key, cursor=cursor, match=match, count=count)
                yield from encode_dict_with_bytes(data).items()

        return wrapiter()

    def __iter__(self):
        return self.scan()


class RedisHashDict(dict):
    pk: str = "api:mmcore:tables:"

    def __init__(self, pk=None):
        super().__init__()
        if pk is not None:
            self.pk = self.pk + pk + ":"

    def __repr__(self):
        return f"({self.__class__.__name__}({self.pk})"

    def full_key(self, v):
        return self.pk + v

    def __setitem__(self, key, value):
        self._set(*key, value)

    def __getitem__(self, key):
        return self._get(key)

    def _get(self, key):

        if isinstance(key, tuple) and (len(key) > 2):
            name, field = key[:2]

            res = encode_dict_with_bytes(rconn.hget(self.full_key(name), field))

            if isinstance(res, str):
                *keys, = self.keys()
                if res in keys:
                    print(res, key[1:])
                    return self.__getitem__((res,) + key[2:])
            else:
                return res
        elif len(key) == 2:
            return encode_dict_with_bytes(rconn.hget(self.full_key(key[0]), key[1]))
        elif len(key) == 1:

            return Hdict(key[0], pk=self.pk)

        else:
            return Hdict(key, pk=self.pk)

    def _set(self, key, field, value):

        rconn.hset(self.full_key(key), field, json.dumps(value).encode())

    def items(self):
        return zip(self.keys(), self.values())

    def values(self):
        return ((dict((hk, self[k, hk]) for hk in self.hkeys(k))) for k in self.keys())

    def hkeys(self, key):
        return encode_dict_with_bytes(rconn.hkeys(self.pk + key))

    def keys(self):
        return (k.replace(self.pk, "") for k in list(encode_dict_with_bytes(rconn.keys(self.pk + "*"))))
