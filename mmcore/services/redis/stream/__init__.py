import ast
import uuid as _uuid

import redis


def ppp(s):
    if isinstance(s, bytes):
        return ppp(s.decode())

    elif isinstance(s, str) and "\\" in s:
        try:
            return ppp(ast.literal_eval(s))
        except SyntaxError as err:
            return ppp(ast.literal_eval(s.strip('"')))

    elif isinstance(s, str) and any(t in s for t in ["false", "true", "null"]):
        return encode_dict_with_bytes(json.loads(s))

    elif isinstance(s, (dict, list, tuple)):
        return encode_dict_with_bytes(s)
    else:
        try:
            return ast.literal_eval(s)
        except ValueError:
            return s
        except SyntaxError:
            return s


def encode_dict_with_bytes(obj):
    if isinstance(obj, dict):
        return dict((encode_dict_with_bytes(k), encode_dict_with_bytes(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return [encode_dict_with_bytes(v) for v in obj]
    elif isinstance(obj, (str, bytes, bytearray)):

        return ppp(obj)

    else:
        return obj


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
        return self.conn.xlen(self.name)

    def __getitem__(self, item):

        # FIXME В перспективе это может стать долго
        #  ну и как минимум это игнорирует основные возможности предоставляемые redis
        _all = self.xrange(count=self.xlen(), min="-", max="+")

        _all.reverse()
        return encode_dict_with_bytes(_all[item])

    def get_all(self, reverse=False):
        _all = self.xrange(count=self.xlen(), min="-", max="+")
        if reverse:
            _all.reverse()
        return encode_dict_with_bytes(_all)

    def keys(self):
        for k, v in self.get_all():
            yield k

    def values(self):
        for k, v in self.get_all():
            yield v

    def items(self):
        for k, v in self.get_all():
            yield k, v

    def todict(self, reverse=True):
        return dict(self.get_all(reverse=reverse))

    def __len__(self):
        return self.xlen()


import json

from mmcore.utils.versioning import Now

from mmcore.typegen.dict_schema import DictSchema

rconn = None
class SharedDict(dict):
    _schema = {}

    def __init__(self, name, conn, uuid=None):
        self.stream_name = name
        self.conn = conn
        self._uuid = uuid

        super().__init__(self._last())

    def update(self, *args, **kwargs) -> None:
        dict.update(self, *args, **kwargs)
        self.commit()
    @property
    def uuid(self):
        return self._uuid

    @property
    def schema(self):
        self._schema |= self
        return DictSchema(self._schema)

    @schema.setter
    def schema(self, v):

        self._schema = v


    @classmethod
    def create_new(cls, name, conn):
        uid = _uuid.uuid4().hex
        return cls(f"{name}:{uid}", conn, uuid=uid)

    def _last(self):
        try:
            res = encode_dict_with_bytes(self.stream.get_last())

            self._uuid = res[0][0]

            return res[0][1]
        except:
            return {}
    @property
    def stream(self):
        return StreamConnector(self.stream_name, redis_conn=self.conn)

    def _fields(self):
        dct = {}
        # for key in self.keys():

        # dct[f'"{key}"'] = json.dumps(self.get(key))

        return dict(self.items())

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

        return self.stream.xadd(None, fields=self._fields())


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

    def _fields(self):
        return {
            f'"object"': json.dumps(self.get("object")),
            '"materials"': json.dumps(self.get("materials")),
            '"geometries"': json.dumps(self.get("geometries")),
            '"metadata"': json.dumps(self.get("metadata"))
        }


def upd(self, data, log=[]):
    __vcs__ = {}
    s1 = set(data.keys()) - {"__vcs__"}

    s2 = set(self.keys()) - {"__vcs__"}
    new_keys = list(s2 - s1)
    # print(s1,s2,new_keys)
    __vcs__["new"] = dict((i, s2[i]) for i in new_keys)
    __vcs__["update"] = dict()
    for k in s1:
        # print(f"{data[k]}->{self[k]}" )
        if not self[k] == data[k]:
            print(f"[change]: {k}: {data[k]}->{self[k]}")
            __vcs__["update"][k] = {"state": data[k], "commit": self[k], "dtime": Now()}

    if not (__vcs__["update"] == {} and __vcs__["new"] == {}):
        flds = self._fields()
        flds["__vcs__"] = __vcs__
        log.append(__vcs__)
        self.stream.xadd(None, fields=dict((k, json.dumps(flds.get(k))) for k in flds.keys()))
class SharedMultiDict(SharedDict):
    ...


class RedisStreamDict(SharedDict):
    log = []

    def __init__(self, name, uuid=None, conn=None):

        super().__init__(name, conn=conn if conn is not None else self.get_conn(), uuid=uuid)

    def get_conn(self):
        global rconn
        if rconn is None:
            import mmcore.services.redis.connect as connect
            rconn = connect.bootstrap_local()
        return rconn

    @property
    def conn(self):
        global rconn
        return rconn

    @conn.setter
    def conn(self, v):
        global rconn
        if v is not None:
            rconn = v

    def commit(self):
        data = encode_dict_with_bytes(list(self.stream.get_last()[0]))[1]
        log = f"[commit event] \n from:\n {data} -> \nto:\n {self}\n"

        try:
            if not (self == {}):
                if len(data) == 0:

                    # print(log)
                    upd(self, data, log=log)




                else:

                    # log=f"Update state:\nprev: {json.dumps(data, indent=2)}\n\n---\n\ncurrent:\n{json.dumps(self, indent=2)}"
                    # print(log)
                    upd(self, data)

            else:
                log = f"Pass commit, target is empty:\n\t{self}"
                # print(log)


        except Exception as err:

            raise Exception(f"\n{err}\n\n{data}")
