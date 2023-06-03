import sys
from typing import ContextManager

import dotenv

from mmcore.collections.multi_description import ES
from mmcore.gql.client import GQLReducedQuery
from mmcore.services.client import get_connection_by_host_port

hosts = ["localhost", "host.docker.internal"]
import itertools


def resolve_query(data):
    lst = []
    port = []

    for i, v in enumerate(ES(data)["serviceByTopic"]._seq):
        print(i, v
              )
        if v["attributes"] is not None:
            port = v["attributes"]["port"]
            *itr, = itertools.zip_longest(ES(v['hosts'])["host"]["address"], [port], fillvalue=port)
            lst.extend(itr)

    return lst


class ModuleResolver(ContextManager):
    def __init__(self, dotenv_name=".env"):
        self.env = dotenv.dotenv_values(dotenv.find_dotenv(dotenv_name, usecwd=True))
        self.query = GQLReducedQuery("""
query MyQuery($value: String!) {
    platform_topics_by_pk(value: $value) {
           
        subscribes {
            serviceByTopic {
                attributes {
                  port
                }
            hosts {
                host {
                name
                    address 
                  }
                }
              }
            }
          }
        }""")
    def __enter__(self):
        print("__enter__", self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("__exit__", self)
        self.exc_type, self.exc_val, self.exc_tb = exc_type, exc_val, exc_tb
        if exc_val:
            #print("__exit__2", exc_val)
            response = resolve_query(self.query(variables={"value": self.exc_val.name.replace(".", "")})['subscribes'])

            self.conn = get_connection_by_host_port(*response)
            missed_module = self.conn.root.getmodule(self.exc_val.name)

            sys.modules[self.exc_val.name] = missed_module
            __import__(self.exc_val.name)

        return self
