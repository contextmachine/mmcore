import base64
import dataclasses
import datetime
import functools
import json
import pickle
import socket
import time
import types
import typing
from enum import Enum
from uuid import uuid4

from mmcore.base.registry import adict

__messages__ = dict()


def gen(kws):
    for k in set(kws.keys()).union(set(kws.get("__annotations__", dict()).keys())):
        if not k.startswith("__"):
            t = kws.get("__annotations__", dict()).get(k)
            if t:
                yield k, kws.get(k, t())
            else:
                yield k, kws.get(k, None)


def gen_anno(kws):
    annotations__ = kws.get("__annotations__", dict())
    for k in kws.keys():
        if not k.startswith("__"):
            if k in annotations__:
                yield k, annotations__[k]
            else:
                return k, type(kws[k])


def mmcore_post_init__(self, **kwargs):
    for k, v in kwargs.items():
        setattr(self, k, v)

    self.time = datetime.datetime.now().isoformat()


def todct__(self):
    def d(o):
        def _gen():
            for k in o.__slots__:
                if not k.startswith('_'):
                    res = getattr(self, k)

                    yield k, d(res)

        if isinstance(o, self.__class__):
            return dict(_gen())
        elif hasattr(o, 'todict'):
            return o.todict()
        elif isinstance(o, (list, tuple, set)):
            return [d(r) for r in o]
        elif isinstance(o, dict):

            return dict((k, d(v)) for k, v in o.items())
        else:
            return o

    return d(self)


__strtptbl__ = dict()


class StorableType(type):

    def __new__(mcs, name, bases, attrs, table=None, **kws):
        defs = dict()
        for base in bases:
            if hasattr(base, "__defaults__"):
                defs |= base.__defaults__
        defs |= dict(gen(attrs))

        _attrs = dict()
        if table is None:
            for base in bases:

                if hasattr(base, "__table__"):
                    table = base.__table__
                    break
            if table is None:
                table = __strtptbl__

        for k, v in attrs.items():
            if k.startswith("__"):
                _attrs[k] = v
        _attrs['__defaults__'] = defs
        _a = dict(gen_anno(attrs))
        _attrs['__annotations__'] = _a
        ss = list(defs.keys()) + ['uuid', 'time']

        for k in attrs.get('__slots__', []):
            if k not in ss:
                ss.append(k)

        for base in bases:
            if hasattr(base, '__slots__'):
                for k in base.__slots__:
                    if k not in ss:
                        ss.append(k)

        _attrs['__slots__'] = ss

        _attrs['__table__'] = table
        _attrs |= kws

        def mnew__(cls, uuid=None, **kwargs):
            if uuid is None:
                uuid = uuid4().hex
            elif uuid in table:

                return table[uuid]
            self = object.__new__(cls)
            self.uuid = uuid
            dct = dict()
            dct |= cls.__defaults__

            for k, v in kwargs.items():
                if k is not None:
                    dct[k] = v

            self.__mmcore_post_init__(**dct)
            table[self.uuid] = self
            return self

        _attrs["__new__"] = mnew__
        _attrs["__mmcore_post_init__"] = mmcore_post_init__
        _attrs["todict"] = todct__
        if len(bases) == 0:
            bases = (object,)
        return super().__new__(mcs, name, bases, _attrs)


proto_decoders_resolvers = dict(
    DEFAULT=lambda data: data, PICKLE=lambda data: data,
    PPICKLE=lambda data: pickle.loads(base64.b64decode(data.encode()))
)

proto_encoders_resolvers = dict(
    DEFAULT=lambda data: data, PICKLE=lambda data: data,
    PPICKLE=lambda data: base64.b64encode(pickle.dumps(data)).decode())


class Msg(metaclass=StorableType, table=__messages__):
    proto: str = "DEFAULT"
    accept: str = "DEFAULT"


class AbsResp(Msg):
    status_code: int = -1


class Resp(AbsResp):
    status_code: int = 200


class Err(AbsResp):
    status_code: int = 500
    reason: str = "Internal Error"


class AbsData(Msg):
    data: dict


class RequestData(AbsData):
    method: str


class ResponseData(Resp):
    request: RequestData
    content: AbsData = None


class ResponseErrData(Err):
    request: RequestData
    reason: str


import random


def m(sleep_time=None, rand_range=None):
    print(f"start, {sleep_time}")
    r = random.randint(*rand_range)
    prc = (sleep_time + r) / 100
    for i in range(100):
        time.sleep(prc)

        print(f'{" " * (3 - len(str(i)))}{i}%', end='\r', flush=True)
    return r


pdd = dict()


def _spec_from_func(func, fillvalue=None):
    if func.__code__.co_argcount > 0:
        df = []
        if func.__defaults__ is not None:
            df = func.__defaults__

        spc = dict(zip_longest(func.__code__.co_varnames[:func.__code__.co_argcount], df, fillvalue=fillvalue))
    else:
        spc = dict()

    return spc


def server_task(server, request):
    try:
        data_encoder = proto_encoders_resolvers[request.accept]
        data_decoder = proto_decoders_resolvers[request.proto]

        mm = server.methods[request.method]['func']
        return ResponseData(request=request,
                            proto=request.accept,
                            content=AbsData(data=data_encoder(mm(**data_decoder(request.data))
                                                              )
                                            ),

                            status_code=200)
    except Exception as err:
        return ResponseErrData(request=request, reason=f'{err}')


import threading as th
from mmcore import __version__

_v = __version__()

from itertools import zip_longest


class IOProtocolEnum(str, Enum):
    default = "DEFAULT"
    pickle = "DEFAULT"


@dataclasses.dataclass
class IOProtocol:
    inp: IOProtocolEnum = IOProtocolEnum.default
    out: IOProtocolEnum = IOProtocolEnum.default

    def todict(self) -> dict:
        return dataclasses.asdict(self)


child_table = dict()


@dataclasses.dataclass
class Child:
    parent: dataclasses.InitVar
    overrides: dataclasses.InitVar[dict]

    def __post_init__(self, parent, overrides=None):
        self._parent = parent
        if not overrides:
            overrides = dict()
        self.overrides = overrides

    @property
    def overrides(self):
        return self._overrides

    @overrides.setter
    def overrides(self, v):
        self.__dict__.update(v)

    def __getattr__(self, item):
        prnt = super().__getattribute__("_parent")
        over = super().__getattribute__("overrides")
        if item.startswith("_"):
            return super().__getattribute__(item)
        elif item not in over and hasattr(prnt, item):
            return getattr(prnt, item)
        elif item in over:
            return over[item]
        else:
            super().__getattribute__(item)

    def __setattr__(self, item, val):
        prnt = super().__getattribute__("_parent")
        if item.startswith("_"):
            return super().__setattr__(item, val)
        elif hasattr(prnt, item):
            self.overrides[item] = val
        else:
            return super().__setattr__(item, val)


@dataclasses.dataclass
class MethodDescriptorProps:
    name: typing.Optional[str] = None
    protocol: IOProtocol = dataclasses.field(default_factory=IOProtocol)
    fillvalue: typing.Any = None
    spec: typing.Optional[dict] = None
    method: typing.Optional[types.FunctionType] = None

    def update(self, kwargs):
        self.__dict__.update(kwargs)

    def todict(self):
        return dataclasses.asdict(self)


class MmcoreServerMethodDescriptor:
    props: MethodDescriptorProps

    def __init__(self, default_props=None, /, **kwargs):
        if default_props:
            self.props = default_props
            self.props.update(kwargs)
        else:
            self.props = MethodDescriptorProps(**kwargs)

    def bind(self, method):
        self.props.method = method
        self.props.spec = _spec_from_func(method, fillvalue=self.props.fillvalue)
        self._method = method
        return method

    def __get__(self, instance, owner):
        if instance:
            return self.wrap_method(ctx=instance)
        else:
            return self.wrap_method(ctx=owner)

    def wrap_method(self, ctx=None):

        @functools.wraps(ctx._descriptors[self.props.name])
        def wrapper(*args, **kwargs):
            return ctx._descriptors[self.props.name](ctx, *args, **kwargs)

        return wrapper

    def __set_name__(self, owner, name):
        self._name = name
        self._owner = owner
        if self.props.name is None:
            self.props.name = self._name
        owner._descriptors[self.props.name] = MethodDescriptorProps(**self.props.todict())


class MmcoreUpdServer:
    """
    >>>updserve=MmcoreUpdServer()
from mmcore.base import AGroup
from mmcore.base.sharedstate.socketserver import MmcoreUpdServer
updserve=MmcoreUpdServer(start=True)

@updserve(name='testpkl', protocol=dict(accept="DEFAULT",proto="PICKLE"))
def pkl(name="grp",uuid="grp"):
    return AGroup(uuid=uuid,name=name)

@updserve(name='testjsn', protocol=dict(accept="DEFAULT",proto="DEFAULT"))
def jsn(name="grp",uuid="grp"):
    return AGroup(uuid=uuid,name=name).root()


from mmcore.base.sharedstate.socketserver import MmCoreUpdClient
client=MmCoreUpdClient()

client.testpkl(uuid="group2",name='grp')
<mmcore.base.basic.AGroup at 0x17f051c50>

client.testjsn(uuid="group3",name='grp')
{'metadata': {'version': 4.5,
  'type': 'Object',
  'generator': 'Object3D.toJSON'},
 'object': {'name': 'grp',
  'uuid': 'group3',
  'type': 'Group',
  'layers': 1,
  'matrix': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  'castShadow': True,
  'receiveShadow': True,
  'userData': {'properties': {'name': 'grp'},
   'gui': [{'type': 'controls',
     'data': {},
     'post': {'endpoint': 'http://localhost:7711/gui/group3'}}]},
  'children': []},
 'geometries': [],
 'materials': []}
    """
    methods: dict = dict()

    def __init__(self, host: str = "0.0.0.0", port: int = 7811, bufsize: int = 1024, debug: bool = False,
                 start: bool = False):
        super().__init__()

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.host = host
        self.port = port
        self.server_socket.bind((self.host, self.port))
        self.debug = debug
        self.bufsize = bufsize

        @self(name='methods')
        def _get_methods(**kwargs):
            return dict((k, dict(protocol=met['protocol'], spec=met['spec'])) for k, met in self.methods.items())

        @self(name='hello')
        def _hello(**kwargs):
            return dict(host=self.host, port=self.port, cls=f"{self.__class__.__qualname__}", mmcore=_v)

        @self(name='echo')
        def _echo(**kwargs):
            return kwargs

        @self(name="fetch")
        def _fetch(uuid: str = '_'):
            return adict[uuid].root()

        self.thread = None
        if start:
            self.run()

    def run(self):
        self.thread = th.Thread(target=self.start_server)
        self.thread.start()

        print(self.host, self.port)

    def stop(self, timeout=None):
        self.thread.join(timeout=timeout)

    def start_server(self):

        while True:

            message, address = self.server_socket.recvfrom(1024 * 16)

            r = RequestData(**ujson.loads(message.decode()))

            if r.data == "STOP":
                print("stopping server")
                break

            else:
                res = server_task(self, r).todict()

                self.server_socket.sendto(ujson.dumps(res).encode(), address)
                if self.debug:
                    print(json.dumps(res, indent=2))
        self.stop()

    def __call__(self, name=None, protocol=None, fillvalue=None):
        self._nm = name

        def decorator(func):
            if self._nm is None:
                self._nm = func.__name__
            spc = _spec_from_func(func, fillvalue=fillvalue)
            self.methods[self._nm] = dict(func=func,
                                          spec=spc,
                                          protocol=protocol if protocol else {
                                              "proto": "DEFAULT",
                                              "accept": "DEFAULT"
                                          }
                                          )

            return func

        return decorator


import ujson


def _bind(self, item):
    def wrap(**kwargs):
        d = dict()
        d |= self._methods[item]['spec']
        d |= kwargs
        method = item
        res = self(item, data=proto_encoders_resolvers[self._methods[method]['protocol']['accept']](d),
                   protocol=dict(proto=self._methods[method]['protocol']['accept'],
                                 accept=self._methods[method]['protocol']['proto']))

        if res['status_code'] == 200:

            data_decoder = proto_decoders_resolvers[res['proto']]
            return data_decoder(res['content']['data'])
        else:
            raise res['reason']

    wrap.__defaults__ = tuple(self._methods[item].values())
    wrap.__mmcore_defaults__ = self._methods[item]
    return wrap


class MmCoreUpdClient:
    def __init__(self, addr=("0.0.0.0", 7811)):
        self.addr = addr
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.table = dict()
        self.upd_methods()

    def upd_methods(self):
        try:
            self._methods = self.__call__('methods')['content']['data']
        except Exception as err:
            print(err)
            self._methods = dict()

    def __getattr__(self, item):
        if item.startswith("_"):
            return super().__getattribute__(item)
        elif item in self._methods:

            return _bind(self, item)
        else:
            return super().__getattribute__(item)

    def __dir__(self):
        return list(super().__dir__()) + list(self._methods.keys())

    def __call__(self, method: str, data: dict = None, protocol=None):
        if data is None:
            data = dict()
        _protocol = dict(proto="DEFAULT", accept="DEFAULT")
        if protocol is not None:
            _protocol |= protocol

        data_enc = proto_encoders_resolvers[_protocol['proto']]
        self.client_socket.sendto(ujson.dumps(dict(data=data_enc(data), method=method, **_protocol)).encode(),
                                  self.addr)
        data, server = self.client_socket.recvfrom(1024 * 16)
        d = ujson.loads(data.decode())
        self.table[d['uuid']] = dict(response=d, server=server)

        return d
