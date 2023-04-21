import json
import typing

import fastapi
import uvicorn
from fastapi import FastAPI

import dataclasses

gqlapp=FastAPI()
@dataclasses.dataclass
class Query:
    query:str
    variables: dict[str, typing.Any] | dict[None, None]

data_test2 = {
    'test': {
        'object': [
            {
                'name': "foo1",
                "uuid": 1,
                "baz": {
                    'name': "A",
                    "uuid": 3
                }
            },
            {
                'name': "foo2",
                "uuid": 2,
                "baz": {
                    'name': "B",
                    "uuid": 4
                }
            }
        ]
    }
}

# qt.interpret()(data_test)




from mmcore.gql.api.parse import parse_simple_query



@gqlapp.post("/v2/qraphql")
def qraphql2(data: Query):

    qt2 = parse_simple_query(data.query)

    return qt2.resolve(data_test2)

if __name__ == "__main__":

    uvicorn.run("main:gqlapp", host="0.0.0.0", port=5799)
