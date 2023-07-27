import functools

from starlette.responses import HTMLResponse

from mmcore.base.params import pgraph
from mmcore.gql.lang.parse import parse_simple_query

TYPE_CHECKING = False
import ujson as json
import strawberry
from mmcore.collections.multi_description import ElementSequence
from strawberry.scalars import JSON
import uvicorn, fastapi
from mmcore.base.registry import *
from rpyc import ThreadPoolServer, SlaveService


def search_all_indices(lst, value):
    for i, v in enumerate(lst):
        if v == value:
            yield i


def generate_uvicorn_app_name(fpath, appname="app"):
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
from graphene import types
from graphene import ObjectType, String, Schema


types.dynamic.MountedType
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
from graphql.execution.execute import execute

sch22 = parse(sch2)
sch3 = graphql.build_ast_schema(sch22)
ee = ElementSequence(list(objdict.values()))

T = typing.TypeVar("T")

from starlette.exceptions import HTTPException

from mmcore.gql.server.fastapi import MmGraphQlAPI

debug_properties={}
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


            return {'kind': "Group", "items":[self.trav(j) for j in  list(self.dct[i]["__children__"])]}
        else:
            obj = {'kind': "Object", "name": i, "value": self.table[i]}
            if self.dct[i] == {}:
                obj |= {"isleaf": True}
                return obj

            elif isinstance(self.dct[i], (list, tuple, set)):
                if isinstance(self.dct[i], set):
                    dct=list(self.dct[i])
                else:
                    dct=self.dct[i]
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

    def __new__(cls, *args, header="[mmcore] SharedStateApi", **kwargs):
        global serve_app

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
        serve_app = fastapi.FastAPI(name='serve_app', title=inst.header)
        inst.app = serve_app
        serve_app.mount("/v2", gqlv2app)
        inst.app.add_middleware(CORSMiddleware, allow_origins=["*"],
                                allow_methods=["GET", "POST", "PUT", "HEAD", "OPTIONS", "DELETE"],
                                allow_headers=["*"],
                                allow_credentials=["*"])
        inst.app.add_middleware(GZipMiddleware, minimum_size=500)
        inst.resolvers = dict()
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
                adict[uid]._matrix = [0.001, 0, 0, 0, 0, 2.220446049250313e-19, 0.001, 0, 0, 0.001,
                                      2.220446049250313e-19, 0, 0, 0, 0, 1]
                return adict[uid].root()
            except KeyError as errr:
                return HTTPException(401, detail=f"KeyError. Trace: {errr}")

        @inst.app.post("/fetch/{uid}")
        def post_item(uid: str, data: dict):

            if pgraph.item_table.get(uid) is not None:

                pgraph.item_table.get(uid)(**kwargs)

            else:
                adict[uid](**data)

        @inst.app.get("/keys", response_model_exclude_none=True)
        async def keys():
            return list(adict.keys())

        @inst.app.get("/", response_model_exclude_none=True)
        async def home():

            from mmcore.base import AGroup
            aa = AGroup(uuid="__")
            for i in adict.values():

                if not (i.uuid == "__"):
                    aa.add(i)

            return aa.root()

        @inst.app.post("/graphql")
        async def gql(data: dict):
            from mmcore.base import AGroup
            aa = AGroup(uuid="__")
            for i in adict.values():

                if not (i.uuid == "__") and not (i.uuid == "_"):
                    aa.add(i)
            qq=parse_simple_query(data["query"])


            return qq.resolve(adict)

        @inst.app.options("/graphql")
        async def gql(data: dict):
            from mmcore.base import AGroup
            aa = AGroup(uuid="__")
            for i in adict.values():

                if not (i.uuid == "__") and not (i.uuid == "_"):
                    aa.add(i)

            return execute(sch3, parse(data["query"]),
                           root_value={"root": aa.root()},
                           variable_values=data.get("variables")).data

        @inst.app.get("/graphql")
        async def gql(data: dict):
            from mmcore.base import AGroup
            aa = AGroup(uuid="__")
            for i in adict.values():

                if not (i.uuid == "__") and not (i.uuid == "_"):
                    aa.add(i)

            return execute(sch3, parse(data["query"]),
                           root_value={"root": aa.root()},
                           variable_values=data.get("variables")).data

        @inst.app.post("/", response_model_exclude_none=True)
        async def mutate(data: dict = None):
            if data is not None:
                if len(data.keys()) > 0:

                    for k in data.keys():
                        objdict[k](**data[k])

            return strawberry.asdict(pull().get_child_three())

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

        @inst.app.post("/resolver/{name}/{uid}")
        async def external_post(name: str, uid: str, data: dict):

            return inst.resolvers[name](uid, data)

        @inst.app.get("/resolver/{name}/{uid}")
        async def external_get(name: str, uid: str):
            return inst.resolvers[name](uid)


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

        return inst

    def run(self):
        uvicorn_appname = generate_uvicorn_app_name(__file__, appname=self.appname)
        print(f'running uvicorn {uvicorn_appname}')
        uvicorn.run(uvicorn_appname, port=self.port, host=self.host, log_level="error")

    def stop(self):
        self.thread.join(6)

    def start(self):

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
        self.run()

    def mount(self, path, other_app, name: str):
        self.app.mount(path, other_app, name)

    def event(self, fun):

        self.app.on_event()


serve = SharedStateServer()
