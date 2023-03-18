import sys

from abc import ABCMeta
from typing import ContextManager
import dotenv
from mmcore.gql.client import GQLReducedQuery
from mmcore.services.client import get_connection

class ModuleResolver(ContextManager):
    def __init__(self, dotenv_name=".env"):
        self.env = dotenv.dotenv_values(dotenv.find_dotenv(dotenv_name, usecwd=True))
        self.query = GQLReducedQuery("""
query MyQuery($_eq1: platform_topics_enum) {
  platform_services(where: {roles: {topic_name: {_eq: $_eq1}}}) {
    url
    value
  }
}

        """)


    def __enter__(self):
        print("__enter__",self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("__exit__", self)
        self.exc_type, self.exc_val, self.exc_tb=exc_type, exc_val, exc_tb
        print("__exit__2", exc_val)
        #self.exc_response=self.query(variables={"_eq1": self.exc_val.name.replace(".", "")})

        self.conn = get_connection(self.query(variables={"_eq1":self.exc_val.name})[0]["url"])
        missed_module = self.conn.root.getmodule(self.exc_val.name)

        sys.modules[self.exc_val.name]=missed_module
        __import__(self.exc_val.name)
        return self


