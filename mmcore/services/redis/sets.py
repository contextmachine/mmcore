import typing

from mmcore import load_dotenv_from_path
from mmcore.services.redis.stream import encode_dict_with_bytes
import pickle
load_dotenv_from_path(".env")
from mmcore.services.redis import connect

rconn = connect.get_cloud_connection()


class Hdict:
    pk: str = 'api:hsets:'

    def __init__(self, key: str, pk: str = None):
        super().__init__()
        if pk is not None:
            self.pk = pk
        self.key = key

    @property
    def full_key(self):
        return self.pk + self.key

    def __setitem__(self, key, value):
        rconn.hset(self.full_key, key, value)

    def __getitem__(self, key):
        return encode_dict_with_bytes(rconn.hget(self.full_key, key))

    def items(self):
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
    pk: str = 'api:hsets:'

    def __setitem__(self, key, value: dict):

        rconn.hset(self.full_key, key, gzip.compress(json.dumps(value).encode()))

    def __getitem__(self, key):
        return encode_dict_with_bytes(gzip.decompress(rconn.hget(self.full_key, key)))

    def items(self):
        return list(encode_dict_with_bytes([gzip.decompress(i) for i in rconn.hgetall(self.full_key)]))

    def keys(self):
        return list(encode_dict_with_bytes(rconn.hkeys(self.full_key)))


class GeometryHdict(CompressedHdict):
    """
    CompressedHdict with geometry specific
    """
    pk: str = 'api:hsets:geometry:'


class PickleHdict(Hdict):
    """
    CompressedHdict with geometry specific
    """
    pk: str = 'api:hsets:'

    def __setitem__(self, key, value: typing.Any):

        rconn.hset(self.full_key, key, pickle.dumps(value))

    def __getitem__(self, key):
        return pickle.loads(rconn.hget(self.full_key, key))

    def items(self):
        return list([pickle.loads(i) for i in rconn.hgetall(self.full_key)])

    def keys(self):
        return list(encode_dict_with_bytes(rconn.hkeys(self.full_key)))

