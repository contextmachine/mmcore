import datetime
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

app = fastapi.FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET", "POST", "PUT", "HEAD", "OPTIONS", "DELETE"],
                   allow_headers=["*"],
                   allow_credentials=["*"])
app.add_middleware(GZipMiddleware, minimum_size=500)
o = DictSchema(ObjectThree(grp.uuid).to_dict())
oo = o.get_strawberry()


class ServerBind():
    port: int = 7711

    def __new__(cls, *args, **kwargs):
        if len(_server_binds) > 0:
            return _server_binds[0]
        import threading as th
        inst = object.__new__(cls)
        inst.__dict__ |= kwargs
        _server_binds.append(inst)

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
            return adict[uid].root()

        @app.post("/fetch/{uid}")
        def p_item(uid: str, data: dict):
            adict[uid](**data)


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
                uuid: "some-item-uuid",
                method: "GET"|"POST"|"DELETE"
                body: {
                    # properties
                    name: "A"
                    ...
                }
            }
            """
            await websocket.accept()
            while True:
                data = await websocket.receive_json()
                print(f"WS: {data}")
                obj = adict[data["uuid"]]
                if data["method"]=="GET":
                    await websocket.send_json(data=obj.root())
                elif data["method"]=="POST":
                    await websocket.send_json(data=obj(**data["body"]))
                elif   data["method"]=="DELETE":
                    obj.dispose()


        def run():

            uvicorn.run("mmcore.base.sharedstate:app", port=inst.port, log_level="error")

        inst.thread = th.Thread(target=run)
        if STARTUP:
            try:

                inst.thread.start()
            except OSError as err:
                print("Shared State server is already to startup. Shutdown...")

        return inst

    def stop(self):
        self.thread.join(60)

    def start(self):
        self.thread.start()

    def run(self):
        self.thread.run()

    def is_alive(self):
        return self.thread.is_alive()

    def start_as_main(self, on_start=None, **kwargs):
        if kwargs.get("port"):
            self.port = kwargs.get("port")
        if on_start:
            on_start()
        uvicorn.run("mmcore.base.sharedstate:app", port=self.port, log_level="error", **kwargs)


serve = ServerBind()
