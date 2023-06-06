import json

import uvicorn

from mmcore.gql.lang.parse import parse_simple_query
from mmcore.gql.server import models
from mmcore.gql.server.fastapi import MmGraphQlAPI
import os



app = MmGraphQlAPI(gql_endpoint="/v2/graphql")



@app.post(app.gql_endpoint)
def graphql_query_resolver(data: dict):
    ##print(data)
    qt2 = parse_simple_query(data['query'])
    with open("tests/data/panel.json") as f:
        data_test2 = json.load(f)
        with open("mmcore/gql/templates/schema.json") as f2:
            data_test2 |= json.load(f2)
            return qt2.resolve(data_test2)


if __name__ == "__main__":
    #print("http://localhost:5799/docs", "http://localhost:5799/v2/graphql", sep="\n")
    uvicorn.run("__main__:app", host="0.0.0.0", port=5799)
