import sys
from typing import ContextManager

import dotenv

from mmcore.gql.client import GQLReducedQuery
from mmcore.services.client import get_connection, get_connection_by_host_port


hosts = ["localhost", "host.docker.internal"]


class ModuleResolver(ContextManager):
    def __init__(self, dotenv_name=".env"):
        self.env = dotenv.dotenv_values(dotenv.find_dotenv(dotenv_name, usecwd=True))
        self.query = GQLReducedQuery("""
        
query MyQuery($value: String!) {
  platform_topics_by_pk(value: $value) {
    value
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
}


        """)

    def __enter__(self):
        print("__enter__", self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("__exit__", self)
        self.exc_type, self.exc_val, self.exc_tb = exc_type, exc_val, exc_tb
        if exc_val:
            print("__exit__2", exc_val)
            # self.exc_response=self.query(variables={"_eq1": self.exc_val.name.replace(".", "")})
            resp=self.query(variables={"value": self.exc_val.name})["subscribes"][0]["serviceByTopic"]

            port = resp["attributes"]["port"]
            hosts = resp["hosts"]["port"]

            self.conn = get_connection_by_host_port(itertools.zip_longest(hosts, [port], fillvalue=port))
            missed_module = self.conn.root.getmodule(self.exc_val.name)

            sys.modules[self.exc_val.name] = missed_module
            __import__(self.exc_val.name)
        return self
