import uuid
import redis
import uuid

import redis


class StreamConnector:
    group = None

    def __init__(self, name, redis_conn: redis.Redis, **kwargs):
        super().__init__()

        self.__dict__ |= kwargs
        self.name = name
        self.conn = redis_conn
        self.last_pub = None
        self.last_sub = None
        self.read_from_stream()

    def write_to_stream(self, data, **kwargs):
        last_pub_id = self.xadd(self.name, data, **kwargs)
        return last_pub_id

    def read_from_stream(self, min="-", max="+", count=1, **kwargs):
        self.last_sub = self.xrevrange(min=min, max=max, count=count, **kwargs)
        

    def create_group(self, group_name=None, id="*", **kwargs):
        group = self.group if group_name is None else group_name
        ans = self.conn.xgroup_create(self.name, group, id=id, **kwargs)
        if not ans:
            raise ans

    def xadd(self, item, fields, **kwargs):
        if item is None:
            item = "*"

        return self.conn.xadd(self.name, fields, id=item, **kwargs)

    def xrange(self, min="-", max="+", count=1):
        return self.conn.xrange(self.name, min=min, max=max, count=count)

    def xrevrange(self, min="-", max="+", count=1):
        return self.conn.xrevrange(self.name, min=min, max=max, count=count)

    def get_last(self):
        return self.xrevrange()

    def xread(self, *streams, count=1, **kwargs):
        return self.conn.xread(streams, count=count, **kwargs)

    def xlen(self):
        self.conn.xlen(self.name)

    def __getitem__(self, item):
        min = str(int(item))
        max = str(int(item) + 1)

        return self.xrange(count=1, min=min, max=max)

    def keys(self):
        return [tg.decode().split("-")[0] for tg in list(zip(*self.xrange(count=1)))[0]]

    def __len__(self):
        return self.xlen()


import json


def encode_dict_with_bytes(obj):
    if isinstance(obj, dict):
        return dict((encode_dict_with_bytes(k), encode_dict_with_bytes(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return [encode_dict_with_bytes(v) for v in obj]
    elif isinstance(obj, (bytes, bytearray)):
        norm = obj.decode("ASCII")

        if norm == '':
            return ''
        else:
            if all([char in "0123456789." for char in norm]):
                if "." in norm:
                    return float(norm)
                else:
                    return int(norm)
            elif '"' in norm:
                return json.loads(obj.decode("ASCII"))
            else:
                return obj.decode("ASCII")
    else:
        return obj


class SharedDict(dict):
    schema = {}

    def __init__(self, name, conn, uuid=None):
        self.stream_name = name
        self.conn = conn
        self._uuid = uuid

        dict.__init__(self, self._last())

    @property
    def uuid(self):
        return self._uuid

    @classmethod
    def create_new(cls, name, conn):
        uid = uuid.uuid4()
        return cls(f"{name}:{uid}", conn, uuid=uid)

    def _last(self):
        try:
            res = encode_dict_with_bytes(self.stream.get_last())

            self.id = res[0][0]

            return res[0][1]
        except:
            return {}
    @property
    def stream(self):
        return StreamConnector(self.stream_name, redis_conn=self.conn)

    def fields(self):
        dct = {}
        for key in self.keys():
            dct[f'"{key}"'] = json.dumps(self.get(key))
        return dct

    def __getitem__(self, item):

        return dict.__getitem__(self, item)


    def __setitem__(self, item, v):
        dict.__setitem__(self, item, v)

        self.commit()

    @property
    def stream_name(self):
        return self._stream_name

    @stream_name.setter
    def stream_name(self, v):
        self._stream_name = v

    def commit(self):
        return self.stream.xadd(None, fields=self.fields())


class ThreeJsSharedDict(SharedDict):
    """
    >>> from mmcore.services.redis.stream import *
    >>> redis_conn = redis.Redis(
    >>> host="<your-redis-host>",
    >>> port=6379,
    >>> password="<your-redis-password>")
    >>> from mmcore.services.redis.stream import ThreeJsSharedDict
    >>> aa=ThreeJsSharedDict("<stream-name>", redis_conn)
    >>> aa
    {
        'object': {
            'type': 'Mesh'
            },
        'materials': '[]',
        'geometries': '[]',
        'metadata': {
            'version': 4.5,
            'type': "Object",
            'generator': 'Object3D.toJSON'
        }
    }
    }

    """
    _stream_bane = "api:stream:test"
    schema = {
        "object": {},
        "materials": [],
        "geometries": [],
        "metadata": {
            "version": 4.5,
            "type": "Object",
            "generator": "Object3D.toJSON"
        }
    }
    id: str

    def fields(self):
        return {
            f'"object"': json.dumps(self.get("object")),
            '"materials"': json.dumps(self.get("materials")),
            '"geometries"': json.dumps(self.get("geometries")),
            '"metadata"': json.dumps(self.get("metadata"))
        }


class SharedMultiDict(SharedDict):
    ...