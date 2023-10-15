import pickle
import typing
import uuid

from mmcore.base.links import clear_token, islink, make_link
from mmcore.services.redis import connect
from mmcore.services.redis.stream import encode_dict_with_bytes

rconn = connect.get_cloud_connection()


class Hdict:
    pk: str = 'api:mmcore:hsets:'

    def __init__(self, key: str, pk: str = None, conn=rconn):
        super().__init__()
        if pk is not None:
            self.pk = pk
        self.key = key
        self.rconn = conn

    @property
    def full_key(self):
        return self.pk + self.key

    def __setitem__(self, key, value):
        self.rconn.hset(self.full_key, key, json.dumps(value))

    def __getitem__(self, key):

        return encode_dict_with_bytes(self.rconn.hget(self.full_key, key))

    def __delitem__(self, key):
        self.hdel(key)

    def hdel(self, key) -> bool:
        return bool(self.rconn.hdel(self.full_key, key))

    def items(self):
        return zip(self.keys(), self.values())

    def values(self):
        return iter(encode_dict_with_bytes(rconn.hvals(self.full_key)))

    def keys(self):
        return iter(encode_dict_with_bytes(rconn.hkeys(self.full_key)))

    def __contains__(self, value):
        return self.rconn.hexists(self.full_key, value)

    def __len__(self):
        return self.rconn.hlen(self.full_key)

    def scan(self, match=None, count=None):
        def wrapiter():
            cursor = "0"
            while cursor != 0:
                cursor, data = self.rconn.hscan(self.full_key, cursor=cursor, match=match, count=count)
                yield from encode_dict_with_bytes(data).items()

        return wrapiter()

    def __iter__(self):
        return self.rconn.hgetall(self.full_key)


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


class Hset(Hdict):
    max_keys_repr = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, k):
        val = super().__getitem__(k)
        if islink(val):
            return Hset(clear_token(val))
        return val

    def __setitem__(self, k, v):
        if isinstance(v, self.__class__):
            super().__setitem__(k, make_link(v.key))
        elif isinstance(v, dict):
            if not rconn.hexists(self.full_key, k):
                new_key = uuid.uuid4().hex
                obj = Hset(new_key)
                super().__setitem__(k, make_link(new_key))
            else:
                obj = self[k]

            for key, val in v.items():
                obj[key] = val

        else:
            super().__setitem__(k, v)

    def __repr__(self):

        return f'{self.__class__.__qualname__}({", ".join(self.keys())}) at {self.full_key}'

    def items(self):
        for k, v in encode_dict_with_bytes(rconn.hgetall(self.full_key)).items():
            if islink(v):
                yield k, Hset(clear_token(v))
            else:
                yield k, v
        return iter(encode_dict_with_bytes(rconn.hgetall(self.full_key)))

    def todict(self):
        dct = dict()
        for k, v in self.items():
            if isinstance(v, Hset):
                dct[k] = v.todict()
            else:
                dct[k] = v
        return dct

    def values(self):

        for v in encode_dict_with_bytes(rconn.hvals(self.full_key)):
            if islink(v):
                yield Hset(clear_token(v))
            else:
                yield v

    def update(self, value: dict):
        for k, v in value.items():
            self[k] = v

    def get(self, key: str, __default=None):
        if key in self.keys():
            return self[key]
        else:
            return __default


    def __ior__(self, v):
        for k, v in dict(v).items():
            self[k] = v

    def __ror__(self, v):

        return super().__ror__(v)
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
