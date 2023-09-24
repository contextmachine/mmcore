import os
import re
from collections import namedtuple

import redis

from mmcore import load_dotenv_from_path

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


def get_local_connection(url="redis://localhost:6379/0"):
    return bootstrap_local(url)

def bootstrap_stack(url=os.getenv("REDIS_STACK_URL")):
    return bootstrap_local(url)


def bootstrap_cloud() :

    r = redis.StrictRedis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), password=os.getenv("REDIS_PASSWORD"))

    return r


def get_cloud_connection(host=os.getenv("REDIS_HOST"),
                         port=os.getenv("REDIS_PORT"),
                         password=os.getenv("REDIS_PASSWORD"),
                         db=0):
    try:

        return redis.StrictRedis(host=host  , port=int(port), password=password, db=db)
    except redis.exceptions.ConnectionError:
        ...