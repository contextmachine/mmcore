from collections import namedtuple

import os
from typing import Any
from mmcore import load_dotenv_from_path
import redis
import re
load_dotenv_from_path(".env")
RedisURL=namedtuple("RedisURL", ["proto", "host", "port", "db"])
def parse_url(url:str):
    res=re.split('[:/]', url)
    while "" in res:
        res.remove("")
    return RedisURL(*res)


def bootstrap_local(url="redis://localhost:6379/0"):
    parsed=parse_url(url)
    conn = redis.Redis(parsed.host, parsed.port, parsed.db)

    return conn


def bootstrap_stack(url=os.getenv("REDIS_STACK_URL")):
    return bootstrap_local(url)


def bootstrap_cloud() :

    r = redis.StrictRedis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), password=os.getenv("REDIS_PASSWORD"))

    return r


def get_cloud_connection(host="localhost",
                         port=6379,
                         password="",
                         db=0):
    try:

        return redis.StrictRedis(host=host  , port=int(port), password=password, db=db)
    except redis.exceptions.ConnectionError:
        ...