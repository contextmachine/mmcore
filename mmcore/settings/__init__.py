import os
from enum import Enum
from argparse import Namespace
import mmcore


class CacheNamespaces(str, Enum):
    HSETS = "api:hsets"
    PARAMS = "api:params"
    TABLE = "api:mmcore:table"


class CoreGlobals(Namespace):
    version: str = mmcore.__version__()
    redis_url: str = os.getenv("REDIS_URL", "localhost:6379")


class ProjectNamespace(CoreGlobals):
    ...
