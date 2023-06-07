import operator

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
    __annotations__ = {
        "request": HashDict(**{'url': str, 'name': str}),
        'method': str,
        'rotation': typing.Optional[tuple[float]],
        'position': typing.Optional[tuple[float]]}
    name = "Panel"
    rotation = (
        1.5707963267948966,
        0,
        0
    )
    position = (
        0,
        0.5,
        0
    )

    method = "GET"
    url = "https://storage.yandexcloud.net/service01vm.contextmachine.online/tests/panel.json"


store = HashDict(**{
    "items": [
        HashDict(
            {
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
                "request": {
                    "method": "GET",
                    "url": "https://storage.yandexcloud.net/service01vm.contextmachine.online/tests/panel.json"

                }
            }
        ),
        HashDict(
            {
                "name": "PanelGroup",
                "rotation": [
                    0,
                    0,
                    0
                ],
                "position": [
                    0,
                    0,
                    0
                ],
                "request": {
                    "method": "GET",
                    "url": "https://storage.yandexcloud.net/service01vm.contextmachine.online/tests/panel_group.json",
                },
            }
        )
    ],

})
def agreagate(where: strawberry.scalars.JSON) -> list[sch]:
    seq = ElementSequence(list(store['items']))
    if isinstance(where, (str, bytes, bytearray)):
        where = json.loads(where)

    return [sch(**data) for data in seq.where(**where)]
from mmcore.collections.multi_description import ElementSequence

ds = DictSchema(HashDict(list(store['items'])))
##print(ds)
sch = D
##print(sch)


def _items() -> list[sch]:
    return [sch(**data) for data in list(store['items'])]


operations = {
    "_eq": operator.eq,
    "_gt": operator.gt,
    "_lt": operator.lt
}


@strawberry.type
class Query:

    @strawberry.field
    def items(self) -> list[sch]:
        return _items()

    @strawberry.field
    def agreagate(self, where: strawberry.scalars.JSON) -> list[sch]:
        seq = ElementSequence(list(store['items']))
        if isinstance(where, (str, bytes, bytearray)):
            where = json.loads(where)

        return [sch(**data) for data in seq.where(**where)]

    @strawberry.field
    def agreagate_with_rules(self, where: strawberry.scalars.JSON) -> list[sch]:
        seq = ElementSequence(list(store['items']))
        if isinstance(where, (str, bytes, bytearray)):
            where = json.loads(where)
        sets = []
        for k, v in where.items():
            *keys, = v.keys()
            *values, = v.values()
            sets.append(set([HashDict(**data) for data in seq.where_with_rule(k, values[0], operations[keys[0]])]))
        s1 = sets.pop(0)
        ##print(s1)
        if len(sets) > 0:

            for s in sets:
                s1.intersection_update(s)

        return [sch(**data) for data in s1]


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
    ##print(f'http://localhost:{5558}{"/graphql"}')
    uvicorn.run("test_react:app", host="0.0.0.0", port=5558, reload=True)
