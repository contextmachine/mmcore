import datetime

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

STARTUP = False
objdict = dict()
geomdict = dict()
matdict = dict()
adict = dict()
ageomdict=dict()
amatdict=dict()


# Usage example:
# from mmcore.base.registry.fcpickle import FSDB
# from mmcore.base.basic import Object3D
# c= Object3D(name="A")
# FSDB['obj']= obj
# ...
# shell:
# python -m pickle .pkl/obj
# [mmcore] : Object3D(priority=1.0,
#                    children_count=0,
#                    name=A,
#                    part=NE) at cf3d55d7-677e-4f96-9e31-b628c3962520
#

import uvicorn, fastapi


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

TYPE_CHECKING = False

ee = ElementSequence(list(objdict.values()))

T = typing.TypeVar("T")

app = fastapi.FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET", "POST", "PUT", "HEAD", "OPTIONS", "DELETE"],
                   allow_headers=["*"],
                   allow_credentials=["*"])
app.add_middleware(GZipMiddleware, minimum_size=500)


class ServerBind():

    def __new__(cls, *args, **kwargs):
        if len(_server_binds) > 0:
            return _server_binds[0]
        import threading as th
        inst = object.__new__(cls)
        _server_binds.append(inst)


        @strawberry.type
        class Root(typing.Generic[T]):
            object: T
            metadata: gql_models.Metadata
            materials: list[gql_models.AnyMaterial]
            geometries: list[typing.Union[gql_models.BufferGeometry, None]]

        @strawberry.type
        class Mutation:
            @strawberry.field
            def matrix(self, uuid: str, matrix: list[float]) -> Root[gql_models.AnyObject3D]:
                objdict[uuid].matrix = matrix
                return objdict[uuid].get_child_three()

            @strawberry.field
            def properties(self, uuid: str, key: str, value: str) -> Root[gql_models.AnyObject3D]:
                objdict[uuid].__setattr__("_" + key, value)
                return objdict[uuid].get_child_three()

            @strawberry.field
            def material_by_uuid(self, uuid: str, material: gql_models.MaterialInput) -> gql_models.Material:
                mat = material.material
                mat.uuid = uuid
                matdict[uuid] = mat
                return matdict[uuid]

            @strawberry.field
            def rootless_by_uuid(self, uuid: str,data:JSON) -> Root[gql_models.AnyObject3D]:
                return objdict[uuid](**data).get_child_three()

            """
                @strawberry.field
                def geometry(self, uuid: str) -> target.bind_class:
                    target._geometry = uuid
                    return target.get_child_three()["object"]



                @strawberry.field
                def geometry_by_uuid(self, geometry: GqlGeometry) -> target.bind_class:
                    target.geometry = geometry
                    return target.get_child_three()["object"]

                @strawberry.field
                def material_by_uuid(self, material: models.MeshPhongMaterial) -> target.bind_class:
                    target.material = material

                    return target.get_child_three()["object"]

                @strawberry.field
                def material(self, uuid: str) -> target.bind_class:
                    target._material = uuid
                    return target.get_child_three()["object"]"""

            @strawberry.field
            def rootless_by_name(self, name: str, data:JSON) -> Root[gql_models.AnyObject3D]:
                ee = ElementSequence(list(objdict.values()))
                from mmcore.base import Group

                grp = ee._seq[ee.multi_search_from_key_value("name", name)[0]]
                grp(**data)
                print(grp)

                return grp()
        @strawberry.type
        class Query:


            @strawberry.field
            def items(self) -> Root[gql_models.AnyObject3D]:...


            @strawberry.field
            def agreagate(self, where: strawberry.scalars.JSON) -> list[JSON]:
                seq = ElementSequence(list(objdict.values()))
                Group = objdict["_"].__class__
                grp=Group(name="aggregate_response")
                if isinstance(where, (str, bytes, bytearray)):
                    where = json.loads(where)

                [grp.add(data)for data in seq.where(**where)]
                return grp.get_child_three()



            @strawberry.field
            def all_by_name(self, name: str) -> JSON:
                ee = ElementSequence(list(objdict.values()))
                from mmcore.base import Group

                grp = Group([ee._seq[i] for i in ee.multi_search_from_key_value("name", name)], name=name + "s")

                return strawberry.asdict(grp.get_child_three())

            @strawberry.field
            def all_by_uuid(self, uuid: str) -> JSON:
                aaa = strawberry.asdict(objdict[uuid]())
                print(aaa)

                return aaa

            @strawberry.field
            def uuids(self) -> list[str]:
                return list(objdict.keys())

            @strawberry.field
            def names(self) -> list[str]:
                return list(set(ElementSequence(list(objdict.values()))["name"]))

            @strawberry.field
            def object_by_uuid(self, uuid: str) -> gql_models.AnyObject3D:
                return objdict[uuid].get_child_three().object

            @strawberry.field
            def rootless_by_uuid(self, uuid: str) -> Root[gql_models.AnyObject3D]:
                return objdict[uuid].get_child_three()

            @strawberry.field
            def rootless_by_name(self, name: str) -> Root[gql_models.AnyObject3D]:
                ee = ElementSequence(list(objdict.values()))
                from mmcore.base import Group

                grp = Group([ee._seq[i] for i in ee.multi_search_from_key_value("name", name)], name=name + "s")

                return grp.get_child_three()

            @strawberry.field
            def object_by_name(self, name: str) -> typing.List[gql_models.AnyObject3D]:
                print(name)
                ee = ElementSequence(list(objdict.values()))
                return [ee.get_from_index(i).get_child_three().object for i in
                        ee.multi_search_from_key_value("name", name)]

        def pull():
            br = objdict["_"]
            for o in objdict.values():
                if not o.name == "base_root":
                    br.add(o)
            return br


        @app.get("/h")
        async def home2():

            return strawberry.asdict(pull()())

        @app.get("/")
        def home():

            from mmcore.base import AGroup
            aa = AGroup(uuid="__")
            for i in adict.values():

                if not (i.uuid == "__"):

                    aa.add(i)

            return aa.root()
        @app.post("/")
        async def mutate(data: dict = None):
            if data is not None:
                if len(data.keys()) > 0:

                    for k in data.keys():
                        objdict[k](**data[k])

            return strawberry.asdict(pull().get_child_three())

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            while True:
                data = await websocket.receive_text()
                print(data)
                br = objdict["_"]
                for o in objdict.values():
                    if not o.name == "base_root":
                        br.add(o)

                await websocket.send_json(data=strawberry.asdict(br.get_child_three()))

        mm = strawberry.Schema(
            query=Query,
            mutation=Mutation,

        )

        graphql_app = GraphQLRouter(mm)
        app.include_router(graphql_app, prefix="/graphql")

        def run():

            uvicorn.run("mmcore.base.registry:app", port=7711, log_level="error")

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


serve = ServerBind()
