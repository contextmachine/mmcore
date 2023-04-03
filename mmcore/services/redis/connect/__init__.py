import os
from typing import Any

import redis
import redis_om


def bootstrap_local(url="redis://localhost:6379/0") -> redis.Redis | redis.StrictRedis | Any:
    conn = redis_om.get_redis_connection(url=url)
    return conn


def bootstrap_stack(url=os.getenv("REDIS_STACK_URL")) -> redis.Redis | redis.StrictRedis | Any:
    conn = redis_om.get_redis_connection(url=url)
    return conn


def bootstrap_cloud() -> redis.Redis | redis.StrictRedis | Any:
    r = redis.Redis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), password=os.getenv("REDIS_PASSWORD"))

    return r


def get_cloud_connection(host=os.getenv("REDIS_HOST"),
                         port=os.getenv("REDIS_PORT"),
                         password=os.getenv("REDIS_PASSWORD"),
                         db=0):
    return redis.Redis(host=host, port=int(port), password=password, db=db)
