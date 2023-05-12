from mmcore.base.basic import DictSchema
import abc
import copy
import typing
import dotenv
from mmcore.base.geom import GeometryObject

dotenv.load_dotenv(".env")
from typing import Generic, TypeVar
import cxm
import ifcopenshell
import requests
import strawberry
from ifcopenshell import geom
from strawberry.extensions import DisableValidation
from typing import TYPE_CHECKING, Annotated

import mmcore.base.models.gql
from mmcore.base.basic import *
from mmcore.base.geom.tess import TessellateIfc
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from strawberry.fastapi import GraphQLRouter
from fastapi import FastAPI
import os, uvicorn

TYPE_CHECKING = False
cxm.S3Session()
ee = ElementSequence(list(objdict.values()))

T = TypeVar("T")

import pyvis
import networkx
from mmcore.base.basic import Object3D, Group, ObChd
import strawberry

from mmcore.node import node_eval
import json
import strawberry
import hashlib

from mmcore.base.basic import DictSchema


class HashDict(dict):
    def __hash__(self):
        return int(hashlib.sha1(json.dumps(dict(self)).encode()).hexdigest(), 16)

    def get_straw(self):
        ds = DictSchema(self)
        tp = ds.generate_schema(callback=strawberry.type)
        return tp(**ds.dict_example)
@strawberry.type
class D:
    __annotations__ = {'url': str,
     'name': str,
     'method': str,
     'rotation': typing.Optional[tuple[float]],
     'position': typing.Optional[tuple[float]]}
    name= "Panel"
    rotation= (
            1.5707963267948966,
            0,
            0
        )
    position= (
            0,
            0.5,
            0
        )
    method= "GET"
    url= "https://storage.yandexcloud.net/service01vm.contextmachine.online/tests/panel.json"

store = {"items": {
    HashDict({
        "name": "Panel",
        "rotation": [
            1.5707963267948966,
            0,
            0
        ],
        "position": [
            0,
            0.5,
            0
        ],

        "method": "GET",
        "url": "https://storage.yandexcloud.net/service01vm.contextmachine.online/tests/panel.json"
    }),

    HashDict({
        "url": "https://storage.yandexcloud.net/service01vm.contextmachine.online/tests/panel_group.json",

        "name": "PanelGroup",
        "method": "GET",
        "rotation": [
            0,
            0,
            0
        ],
        "position": [
            0,
            0,
            0
        ]
    })
}}

from mmcore.collections.multi_description import ElementSequence

ds = DictSchema(HashDict(list(store['items'])[0]))

sch = D
print(sch)


def _items() -> list[sch]:
    return [sch(**data) for data in list(store['items'])]


@strawberry.type
class Query:

    @strawberry.field
    def items(self) -> list[sch]:
        return _items()

    @strawberry.field
    def agreagate(self, query: strawberry.scalars.JSON) -> list[sch]:
        seq = ElementSequence(list(store['items']))
        if isinstance(query,(str, bytes, bytearray)):
            query=json.loads(query)
        return [sch(**data) for data in seq.where(**query)]


@strawberry.type
class Mutation:

    @strawberry.field
    def add_item(self, url: str, name: str = "unetitled") -> list[sch]:
        store['items'].add(HashDict(url=url, name=name))
        return _items()


mm = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    extensions=[
        DisableValidation(),
    ],
)

graphql_app = GraphQLRouter(mm)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET", "POST", "PUT", "HEAD", "OPTIONS", "DELETE"],
                   allow_headers=["*"],
                   allow_credentials=["*"])
app.add_middleware(GZipMiddleware, minimum_size=500)

app.include_router(graphql_app, prefix="/graphql")

if __name__ == "__main__":
    print(f'http://localhost:{5558}{"/graphql"}')
    uvicorn.run("test_react:app", host="0.0.0.0", port=5558, reload=True)

