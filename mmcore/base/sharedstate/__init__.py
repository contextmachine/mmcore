import datetime
import functools
import types
from typing import TYPE_CHECKING

from mmcore.base import DictSchema, ObjectThree, grp

TYPE_CHECKING = False
import ujson as json
import sys
import typing
from dataclasses import asdict
from fastapi.middleware.cors import CORSMiddleware
import strawberry
from mmcore.collections.multi_description import ElementSequence
from strawberry.extensions import DisableValidation
from strawberry.fastapi import GraphQLRouter
from strawberry.scalars import JSON
import uvicorn, fastapi
from mmcore.base.registry import *
import rpyc
from rpyc import ThreadPoolServer, ClassicService, SlaveService


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

from fastapi import WebSocket, WebSocketDisconnect, WebSocketException
import mmcore.base.models.gql as gql_models
from graphql.language import parse
import graphql

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
sch3
sch3.type_map
ee = ElementSequence(list(objdict.values()))

T = typing.TypeVar("T")

o = DictSchema(ObjectThree(grp.uuid).to_dict())
oo = o.get_strawberry()
from starlette.exceptions import HTTPException


class SharedStateServer():
    port: int = 7711
    rpyc_port: int = 7799

    def __new__(cls, *args, header="[mmcore] SharedStateApi", **kwargs):
        global app


        import threading as th
        inst = object.__new__(cls)
        inst.__dict__ |= kwargs
        inst.header = header
        fastapi.FastAPI.title = property(fget=lambda slf: inst.header, fset=lambda slf, v: setattr(inst, 'header', v))
        app = fastapi.FastAPI(title=inst.header)
        _server_binds.append(inst)
        inst.app = app

        inst.app.add_middleware(CORSMiddleware, allow_origins=["*"],
                                allow_methods=["GET", "POST", "PUT", "HEAD", "OPTIONS", "DELETE"],
                                allow_headers=["*"],
                                allow_credentials=["*"])
        inst.app.add_middleware(GZipMiddleware, minimum_size=500)

        from mmcore.base import ObjectThree, grp, DictSchema

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

        @app.get("/h")
        async def home2():

            return strawberry.asdict(pull()())

        @app.get("/fetch/{uid}")
        async def get_item(uid: str):
            try:
                return adict[uid].root()
            except KeyError as errr:
                return HTTPException(401, detail=f"KeyError. Trace: {errr.__traceback__}")

        @app.post("/fetch/{uid}")
        def get_item(uid: str, data: dict):
            return adict[uid](**data)

        @app.get("/keys", response_model_exclude_none=True)
        async def keys():
            return list(adict.keys())

        @app.get("/", response_model_exclude_none=True)
        async def home():

            from mmcore.base import AGroup
            aa = AGroup(uuid="__")
            for i in adict.values():

                if not (i.uuid == "__"):
                    aa.add(i)

            return aa.root()

        @app.post("/graphql")
        async def gql(data: dict):
            from mmcore.base import AGroup
            aa = AGroup(uuid="__")
            for i in adict.values():

                if not (i.uuid == "__") and not (i.uuid == "_"):
                    aa.add(i)

            return execute(sch3, parse(data["query"]),
                           root_value={"root": aa.root()},
                           variable_values=data.get("variables")).data

        @app.options("/graphql")
        async def gql(data: dict):
            from mmcore.base import AGroup
            aa = AGroup(uuid="__")
            for i in adict.values():

                if not (i.uuid == "__") and not (i.uuid == "_"):
                    aa.add(i)

            return execute(sch3, parse(data["query"]),
                           root_value={"root": aa.root()},
                           variable_values=data.get("variables")).data

        @app.get("/graphql")
        async def gql(data: dict):
            from mmcore.base import AGroup
            aa = AGroup(uuid="__")
            for i in adict.values():

                if not (i.uuid == "__") and not (i.uuid == "_"):
                    aa.add(i)

            return execute(sch3, parse(data["query"]),
                           root_value={"root": aa.root()},
                           variable_values=data.get("variables")).data

        @app.post("/", response_model_exclude_none=True)
        async def mutate(data: dict = None):
            if data is not None:
                if len(data.keys()) > 0:

                    for k in data.keys():
                        objdict[k](**data[k])

            return strawberry.asdict(pull().get_child_three())

        @app.websocket("/ws")
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

        def run():

            uvicorn.run("mmcore.base.sharedstate:app", port=inst.port, log_level="error")

        def run_rpyc():
            service = SlaveService()
            service.namespace = rpyc_namespace
            _serv = ThreadPoolServer(service, port=cls.rpyc_port)
            _serv.start()

        inst.thread = th.Thread(target=run)
        inst.rpyc_thread = th.Thread(target=run_rpyc)
        inst.runtime_env = dict(inputs=dict(), out=dict())
        inst.resolvers = dict()

        @app.post("/resolver/{uid}")
        async def external_post(uid: str, data: dict):

            if uid in inst.resolvers.keys():
                return inst.resolvers[uid](**data)
            return inst.runtime_env["out"].get(uid)

        @app.get("/resolver/{uid}")
        async def external_get(uid: str):
            return inst.runtime_env["out"].get(uid)

        if STARTUP:
            try:

                inst.thread.start()
            except OSError as err:
                print("Shared State server is already to startup. Shutdown...")

        return inst

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
        self.runtime_env["inputs"][name] = dict()
        self.runtime_env["out"][name] = dict()

        @functools.wraps(func)
        def wrapper(**kwargs):
            self.runtime_env["inputs"][name] |= kwargs
            self.runtime_env["out"][name] |= func(**self.runtime_env["inputs"][name])
            return self.runtime_env["out"][name]

        self.resolvers[str(name)] = wrapper
        return wrapper

    def start_rpyc(self):
        self.rpyc_thread.start()

    def run(self):
        self.thread.run()

    def run_rpyc(self):
        self.rpyc_thread.run()

    def is_alive(self):
        return self.thread.is_alive()

    def is_alive_rpyc(self):
        return self.rpyc_thread.is_alive()

    def start_as_main(self, on_start=None, **kwargs):
        if kwargs.get("port"):
            self.port = kwargs.get("port")
        if on_start:
            on_start()

        uvicorn.run("mmcore.base.sharedstate:app", port=self.port, log_level="error", **kwargs)

    def mount(self, path, other_app, name: str):
        self.app.mount(path, other_app, name)
    def event(self, fun):

        self.app.on_event()

serve = SharedStateServer()
