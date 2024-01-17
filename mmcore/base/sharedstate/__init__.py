import dataclasses
import functools
import threading
import typing

from fastapi.responses import UJSONResponse
from starlette.responses import HTMLResponse
from termcolor import colored
import mmcore
from mmcore.base.params import pgraph
from mmcore.base.userdata.controls import find_points_in_controls
from mmcore.common.viewer import control_points_observer
from mmcore.gql.lang.parse import parse_simple_query

TYPE_CHECKING = False
import ujson as json
import strawberry
from mmcore.collections.multi_description import ElementSequence
from strawberry.scalars import JSON
import uvicorn, fastapi
from mmcore.base.registry import *
from rpyc import ThreadPoolServer, SlaveService
from mmcore.geom.point import BUFFERS
import IPython
from mmcore.base.tags import __databases__

DBs = __databases__

__node_resolvers__ = dict()
def search_all_indices(lst, value):
    for i, v in enumerate(lst):
        if v == value:
            yield i


def generate_uvicorn_app_name(fpath, appname="app"):
    if "\\" in fpath:
        r = list(fpath.split("\\"))
    else:
        r = list(fpath.split("/"))
    if r[-1].startswith("__"):
        r = r[:-1]

    return ".".join(r[list(search_all_indices(r, "mmcore"))[-1]:]) + ":" + appname


class AllDesc:
    def __init__(self, default=None):
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if not instance:
            return self.default
        return json.loads(instance.ToJSON())


_server_binds = []
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from fastapi import WebSocket
import mmcore.base.models.gql as gql_models
from graphql.language import parse
import graphql

from graphene import ObjectType, String, Schema

rpyc_input = dict()

rpyc_namespace = dict(post=lambda k, v: rpyc_input.__setitem__(json.dumps(k), json.loads(v)))

TYPE_CHECKING = False
# language=GraphQl
sch2 = """
schema {
    query: query_root
    mutation: mut_root
}

input ObjectInput {
    name:String


}
scalar JSON
type UserData {
    properties: JSON
}



type Object3D {
    all: JSON
    castShadow: Boolean
    children : [Object3D]
    layers: Int!
    matrix: [Float!]!
    name: String
    receiveShadow: Boolean
    type: String!
    up: [Float!]
    userData: UserData
    uuid: String!
    geometry: String}
type query_root{
    root:Root
}
type Root {
    geometries: [JSON]
    materials: [JSON]
    metadata: JSON
    object: Object3D
}

type mut_root{
    root: Root
}


"""

sch22 = parse(sch2)
sch3 = graphql.build_ast_schema(sch22)
ee = ElementSequence(list(objdict.values()))

T = typing.TypeVar("T")

from starlette.exceptions import HTTPException

from mmcore.gql.server.fastapi import MmGraphQlAPI

debug_properties = {}


@dataclasses.dataclass
class ConsoleStartMessage:
    text: str = f"\n    \u002B\u00D7  mmcore {mmcore.__version__()}\n"
    color: str = 'light_magenta'
    on_color: typing.Optional[str] = None
    attrs: typing.Optional[tuple[str]] = ('bold',)


@dataclasses.dataclass
class PropsUpdate:
    uuids: list[str]
    props: dict[str, typing.Any]


class IDict(dict):
    def __init__(self, dct=idict, table=adict):
        super().__init__(dct)
        self.table = table
        self.dct = dct

    def __getitem__(self, k):
        return self.trav(k)

    def trav(self, i):
        print(self.table[i])
        if 'AGroup' == self.table[i].__class__.__name__:
            return {'kind': "Group", "items": [self.trav(j) for j in list(self.dct[i]["__children__"])]}
        else:
            obj = {'kind': "Object", "name": i, "value": self.table[i]}
            if self.dct[i] == {}:
                obj |= {"isleaf": True}
                return obj

            elif isinstance(self.dct[i], (list, tuple, set)):
                if isinstance(self.dct[i], set):
                    dct = list(self.dct[i])
                else:
                    dct = self.dct[i]
                if len(dct) == 0:
                    obj |= {'kind': "Group", "isleaf": True}
                    return obj
                else:
                    obj |= {'kind': "Group", "items": [self.trav(j) for j in dct], "isleaf": False}
                    return obj

            else:
                _dct = {}
                for k, v in self.dct[i].items():
                    _dct[k] = self.trav(v)
                obj |= _dct
                obj |= {"isleaf": False}
                return VDict(obj)


class Query(ObjectType):
    name = String()


schema = Schema(Query)


def _create_fastapi(name, mounts=(), mws=(
        (
                CORSMiddleware,
                dict(allow_origins=["*"],
                     allow_methods=["GET", "POST", "PUT", "HEAD", "OPTIONS", "DELETE"],
                     allow_headers=["*"],
                     allow_credentials=["*"])),
        (GZipMiddleware, dict(minimum_size=500))
), **kwargs):
    _ = fastapi.FastAPI(name=name, **kwargs)
    if len(mounts) > 0:
        for path, v, mntname in iter(mounts):
            _.mount(path, v, mntname)
    if len(mws) > 0:
        for obj, props in iter(mws):
            _.add_middleware(obj, **props)
    return _


class VDict(dict):
    def __init__(self, *args, relay=idict, table=adict, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.dct = relay
        self.table = table

    def __getitem__(self, item):
        if item == "value":
            return dict.__getitem__(self, item).value()

        else:
            return dict.__getitem__(self, item)


class SharedStateServer():

    port: int = 7711
    rpyc_port: int = 7799
    host: str = "0.0.0.0"
    appname: str = "serve_app"
    prodapp = None
    _is_production = False
    console_start_message: ConsoleStartMessage = ConsoleStartMessage()
    @property
    def is_production(self):
        return self._is_production

    @is_production.setter
    def is_production(self, v):
        self._is_production = v

    def __new__(cls, *args, header="mmcore api", **kwargs):
        global serve_app, DBs

        import threading as th
        inst = object.__new__(cls)
        inst.__dict__ |= kwargs
        inst.header = header

        gqlv2app = MmGraphQlAPI(gql_endpoint="/graphql")

        @gqlv2app.post(gqlv2app.gql_endpoint)
        def graphql_query_resolver(data: dict):
            ##print(data)

            qt2 = parse_simple_query(data['query'])

            return qt2.resolve(pgraph)

        _server_binds.append(inst)

        def shutdown():
            print("shutdown event...")
            for db in DBs.values():
                db.save()

        serve_app = fastapi.FastAPI(name='serve_app', title=inst.header, on_shutdown=[shutdown])

        inst.app = serve_app
        serve_app.mount("/v2", gqlv2app, "gqlv2app")
        inst.app.add_middleware(CORSMiddleware, allow_origins=["*"],
                                allow_methods=["GET", "POST", "PUT", "HEAD", "OPTIONS", "DELETE"],
                                allow_headers=["*"],
                                allow_credentials=["*"])
        inst.app.add_middleware(GZipMiddleware, minimum_size=500)
        inst.resolvers = dict()
        inst.jobs = dict()
        @strawberry.type
        class Root(typing.Generic[T]):
            __annotations__ = {
                "all": typing.Optional[JSON],
                "shapes": typing.Optional[JSON],
                "metadata": gql_models.Metadata,
                "materials": list[gql_models.AnyMaterial],
                "object": T,
                "geometries": list[typing.Union[gql_models.BufferGeometry, None]]
            }

        def pull():
            br = objdict["_"]
            for o in objdict.values():
                if not o.name == "base_root":
                    br.add(o)
            return br

        @inst.app.get("/buffers/{uid}")
        def get_buffer(uid: str):
            print(f"GET {uid} buffer {BUFFERS[uid]._buffer}")

            return {"data": BUFFERS[uid]._buffer}

        @inst.app.post("/buffers/{uid}")
        def upd_all_buffer(uid: str, data: dict):

            BUFFERS[uid].update_all(data["data"])

            return {"data": BUFFERS[uid]._buffer}

        @inst.app.post("/buffers/{uid}/{index}")
        def upd_item_buffer(uid: str, index: int, data: dict):
            if not inst.is_production:
                print(f"[POST] update index {index} for '{uid}' buffer: {BUFFERS[uid][index]}->{data['data']}>",
                      )

            BUFFERS[uid].update_item(index, data["data"])

            return {"data": BUFFERS[uid]._buffer}

        @inst.app.get("/test/app/{uid}")
        def appp(uid: str):
            # language=JavaScript
            application = """
            const socket = new WebSocket('ws://${host}:${port}/ws');
            socket.addEventListener('open', function (event) {socket.send('{ "uuid":"${uid}" }');});
            socket.addEventListener('message', function (event) {console.log(event.data);});
            socket.onmessage
            """.replace('${host}', str(inst.host)).replace('${port}', str(inst.port)).replace('${uid}', uid)
            # language=Html
            return HTMLResponse("""
        <!DOCTYPE html>
        <body>
            <code>
                <script>
                    {script}
                </script>
            </code>
        </body>
            """.format(
                script=application
            )
            )

        @inst.app.get("/h")
        async def home2():
            return strawberry.asdict(pull()())

        @inst.app.get("/fetch/{uid}")
        async def get_item(uid: str):
            try:

                return adict[uid].root()
            except KeyError as errr:
                return HTTPException(401, detail=f"KeyError. Trace: {errr}")

        @inst.app.post("/props-update/{uid}")
        async def props_update(uid: str, data: PropsUpdate):
            try:
                target = adict[uid]

            except KeyError as errr:
                return HTTPException(401, detail=f"KeyError. Trace: {errr}")
            target.props_update(data.uuids, data.props)
            return {"uuid": target.uuid}

        @inst.app.post("/fetch/{uid}")
        def post_item(uid: str, data: dict):
            if len(data) > 0:
                if pgraph.item_table.get(uid) is not None:
                    pgraph.item_table.get(uid)(**kwargs)

                else:
                    adict[uid](**data)

        @inst.app.get("/gui/{uid}")
        def gui_get_item(uid: str):

            return adict[uid].gui_get()

        @inst.app.post("/gui/{uid}")
        def gui_post_item(uid: str, data: dict):

            return adict[uid].gui_post(data)

        @inst.app.post("/controls/{uid}")
        def controls_post_item(uid: str, data: dict):

            control_points_observer.notify(uid, find_points_in_controls(data))
            return adict[uid].controls.todict()

        @inst.app.post("/controls/{uid}")
        def controls_get_item(uid: str):

            return adict[uid].controls.data.todict()

        @inst.app.get("/keys", response_model_exclude_none=True)
        async def keys():
            return list(adict.keys())

        @inst.app.get("/", response_model_exclude_none=True)
        async def home():
            return {
                'app': 'mmcore', 'version': str(mmcore.__version__())
                }

        @inst.app.post("/graphql")
        async def gql(data: dict):
            if "target" in debug_properties:

                return UJSONResponse(adict[debug_properties["target"]].root())
            else:
                qq = parse_simple_query(data["query"])

                return qq.resolve(adict)

        @inst.app.options("/graphql")
        async def gql():

            return "OK"

        @inst.app.get("/graphql")
        async def gql():

            return "OK"


        @inst.app.post("/", response_model_exclude_none=True)
        async def mutate(data: dict = None):
            if "target" in debug_properties:

                return adict[debug_properties["target"]].root()
            else:
                return {}

        @inst.app.post("/jobs/{name}", response_model_exclude_none=True)
        async def jobs_post(name: str, data: dict = None):
            inst.jobs[name](data)
            return {name: "OK"}




        @inst.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """

            @param websocket:
            @return:
            {
                # POST
                uuid: "some-item-uuid",
                body: {
                    # properties
                    name: "A"
                    ...
                }
                ---
                # GET
                uuid: "some-item-uuid",

            }
            """
            await websocket.accept()
            while True:
                data = await websocket.receive_json()
                # print(f"WS: {data}")
                obj = adict[data["uuid"]]
                if "body" in data.keys():
                    if not ((data["body"] is None) or (data["body"] == {})):
                        obj(**data["body"])

                await websocket.send_json(data=obj.root())

        def run_rpyc():
            service = SlaveService()
            service.namespace = rpyc_namespace
            _serv = ThreadPoolServer(service, port=cls.rpyc_port)
            _serv.start()

        inst.thread = th.Thread(target=inst.run)
        inst.rpyc_thread = th.Thread(target=run_rpyc)
        inst.runtime_env = dict(inputs=dict(), out=dict())
        inst.resolvers = dict()
        inst.params_nodes = dict()
        inst.loglevel = 'error'
        @inst.app.get("/ipykernel/connection_file")
        async def ipykernel_connection_file():
            data = inst.get_ipy_connection_file()
            if data is None:
                return {"data": None, "reason": "Kernel is not initialized"}
            return {"data": data, "reason": {}}

        @inst.app.post("/resolver/{name}/{uid}")
        async def external_post(name: str, uid: str, data: dict):
            return UJSONResponse(inst.resolvers[name](uid, data))

        @inst.app.get("/resolver/{name}/{uid}")
        async def external_get(name: str, uid: str):
            return UJSONResponse(inst.resolvers[name](uid))

        @inst.app.post("/params/node/{uid}")
        async def params_post(uid: str, data: dict):
            pgraph.item_table.get(uid)(**data)
            return pgraph.item_table.get(uid).todict(no_attrs=True)

        @inst.app.get("/params/node/{uid}")
        async def params_get(uid: str):
            return pgraph.item_table.get(uid).todict(no_attrs=True)

        @inst.app.get("/params/nodes")
        async def params_nodes_names():
            return list(pgraph.item_table.keys())

        @inst.app.get("/resolvers")
        def resolvers():
            return list(inst.resolvers.keys())

        @inst.app.get("/debug/props")
        async def debug_props():
            return debug_properties

        @inst.app.get("/flow")
        async def get_flow():
            return pgraph.toflow()

        if STARTUP:
            try:
                inst.thread.start()
            except OSError as err:
                print("Shared State server is already to startup. Shutdown...")
                raise err

        return inst

    def run(self, **kwargs):
        uvicorn_appname = generate_uvicorn_app_name(__file__, appname=self.appname)

        use_ssl = False
        for k in kwargs.keys():
            if k.startswith("ssl"):
                use_ssl = True
                break
        _prefix = 'https' if use_ssl else 'http'
        _print_host = self.host if self.host != '0.0.0.0' else 'localhost'
        print(colored(f"\n    \u002B\u00D7  mmcore {mmcore.__version__()}\n", 'light_magenta', None, ('bold',)))
        print(f'        server: uvicorn\n        app: {uvicorn_appname}\n        local: {_prefix}://{_print_host}'
              f':{self.port}/\n        openapi ui: {_prefix}://{_print_host}'
              f':{self.port}/docs\n'
              )

        uvicorn.run(uvicorn_appname, port=self.port, host=self.host, log_level=self.loglevel, **kwargs)


    def production_run(self, name, api_prefix, mounts=(), middleware=None, app_kwargs=None, **kwargs):
        if app_kwargs is None:
            app_kwargs = dict()

        if middleware is None:
            middleware = []
        for obj in self.app.user_middleware:
            if not (obj in [mn[0] for mn in middleware]):
                middleware.append(
                    (obj.cls, obj.options))

        self.prodapp = _create_fastapi(name,
                                       mounts=[
                                           (api_prefix, self.app, self.app.extra['name']),
                                           *mounts],
                                       mws=middleware, **app_kwargs
                                       )
        uvicorn_appname = generate_uvicorn_app_name(__file__, appname=name)
        print(f'running uvicorn {uvicorn_appname}')
        self.is_production = True
        uvicorn.run(uvicorn_appname, port=self.port, host=self.host, **kwargs)

    def stop(self):
        self.thread.join(6)

    def gltf_system(self):
        ...

    def create_child(self, path, name=None, title=None, **kwargs):
        if name is None:
            name = path.split("/")[-1]
        new_app = fastapi.FastAPI(name=name, title=self.header if title is None else title, **kwargs)

        new_app.add_middleware(CORSMiddleware, allow_origins=["*"],
                               allow_methods=["GET", "POST", "PUT", "HEAD", "OPTIONS", "DELETE"],
                               allow_headers=["*"],
                               allow_credentials=["*"])
        new_app.add_middleware(GZipMiddleware, minimum_size=500)

        self.app.mount(path, new_app, name)
        setattr(self, name, new_app)
        return new_app

    def start(self, port=None, host=None, log_level=None, **kwargs):

        for k, v in dict(port=port, host=host, log_level=log_level).items():
            if v is not None:
                setattr(self, k, v)
        self.thread = threading.Thread(target=self.run, args=(), kwargs=kwargs)
        self.thread.start()


    def stop_rpyc(self):
        self.rpyc_thread.join(6)

    def resolver(self, func):
        self.runtime_env["inputs"][func.__name__] = dict()
        self.runtime_env["out"][func.__name__] = dict()

        @functools.wraps(func)
        def wrapper(**kwargs):
            self.runtime_env["inputs"][func.__name__] = kwargs
            self.runtime_env["out"][func.__name__] = func(**self.runtime_env["inputs"][func.__name__])
            return self.runtime_env["out"][func.__name__]

        self.resolvers[str(func.__name__)] = wrapper
        return wrapper

    def add_resolver(self, name, func):
        @functools.wraps(func)
        def wrapper(**kwargs):
            self.runtime_env["inputs"][name] |= kwargs
            self.runtime_env["out"][name] |= func(**self.runtime_env["inputs"][name])
            return self.runtime_env["out"][name]

        self.resolvers[str(name)] = wrapper
        self.resolvers[str(name) + "__wrapped__"] = func
        return wrapper

    def add_params_node(self, name, fun):
        self.params_nodes[name] = fun

    def params_node(self, fun):
        if hasattr(fun, 'name'):
            self.params_nodes[fun.name] = fun
        else:
            self.params_nodes[fun.__name__] = fun
        return fun

    def bind(self, fun):
        def wrap(node):
            def call():
                fun(node)


    def start_rpyc(self):
        self.rpyc_thread.start()

    def run_thread(self):
        self.thread.run()

    def run_rpyc(self):
        self.rpyc_thread.run()

    def is_alive(self):
        return self.thread.is_alive()

    def is_alive_rpyc(self):
        return self.rpyc_thread.is_alive()

    def start_as_main(self, on_start=None, **kwargs):
        self.__dict__ |= kwargs
        if on_start:
            on_start()
        self.is_production = True
        self.run()

    def mount(self, path, other_app, name: str):
        self.app.mount(path, other_app, name)

    def event(self, fun):
        self.app.on_event()

    def start_ipython(self, argv=(), embed=True, prod=False, askernel=False, **kwargs):
        self.is_production = prod
        if askernel and embed:
            IPython.embed_kernel(header=self.header, **kwargs)
        elif embed:
            IPython.embed(header=self.header)

        else:
            IPython.start_ipython(argv=argv, **kwargs)

    def ipy(self):
        try:
            return eval("get_ipython()")
        except NameError as err:
            return None


serve = SharedStateServer()
