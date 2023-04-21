from mmcore.gql.server.fastapi import MmGraphQlAPI
import uvicorn
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

app = MmGraphQlAPI(gql_endpoint="/v2/graphql", dataset=data_test2)


if __name__ == "__main__":
    print("http://localhost:5799/docs", "http://localhost:5799/v2/graphql", sep="\n")
    uvicorn.run("__main__:app", host="0.0.0.0", port=5799)
