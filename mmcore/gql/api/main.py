import dataclasses
import typing
from mmcore.gql.api.parse import parse_simple_query
import uvicorn
from fastapi import FastAPI

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


@dataclasses.dataclass
class Query:
    query: str
    variables: dict[str, typing.Any] | dict[None, None]


graphql_app = FastAPI()


@graphql_app.post("/v2/graphql")
def graphql2(data: Query):
    qt2 = parse_simple_query(data.query)
    return qt2.resolve(data_test2)


if __name__ == "__main__":
    uvicorn.run("main:graphql_app", host="0.0.0.0", port=5799)
