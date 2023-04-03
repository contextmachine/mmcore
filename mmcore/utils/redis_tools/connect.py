import os
from typing import Any

import redis
import redis_om


def bootstrap_local(url=os.getenv("REDIS_URL")) -> redis.Redis | redis.StrictRedis | Any:
    conn = redis_om.get_redis_connection(url=url)
    return conn


def bootstrap_cloud_small(host=os.getenv("REDIS_HOST"),
                          port=os.getenv("REDIS_PORT"),
                          password=os.getenv("REDIS_PASSWORD"),
                          db=0) -> redis.Redis | redis.StrictRedis | Any:
    r = redis.Redis(
        host=host,
        db=int(db),

        port=int(port),
        password=password,
    )

    return r


def get_cloud_connection(host=os.getenv("REDIS_HOST"),
                         port=os.getenv("REDIS_PORT"),
                         password=os.getenv("REDIS_PASSWORD"),
                         db=0):
    return bootstrap_cloud_small(host=host, port=port, password=password, db=db)
