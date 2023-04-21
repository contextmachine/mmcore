from mmcore.gql.lang.parse import parse_simple_query
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from mmcore.gql.server import models


class MmGraphQlAPI(FastAPI):
    def __init__(self, *args, dataset=None, gql_endpoint="/", gql_kwargs=dict(), minimum_gzip_size=500, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_middleware(CORSMiddleware,
                            allow_origins=["*"],
                            allow_methods=["GET", "POST", "PUT", "HEAD", "OPTIONS", "DELETE"],
                            allow_headers=["*"],
                            allow_credentials=["*"])

        self.add_middleware(GZipMiddleware, minimum_size=minimum_gzip_size)

        self.dataset = dataset
        if dataset is None:
            self.dataset = {}
        self.gql_kwargs=gql_kwargs
        self.gql_endpoint= gql_endpoint
        self.init_graphql(**self.gql_kwargs)

    def init_graphql(self, **kwargs):
        @self.post(self.gql_endpoint, **kwargs)
        def graphql_query_resolver(data: models.GQLAPIQuery):
            qt2 = parse_simple_query(data.query)
            return qt2.resolve(self.dataset)

        return graphql_query_resolver

