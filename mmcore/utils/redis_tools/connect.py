import os
from typing import Any

import redis
import redis_om


def bootstrap_local(url=os.getenv("REDIS_URL")) -> redis.Redis | redis.StrictRedis | Any:
    conn = redis_om.get_redis_connection(url=url)
    return conn


def bootstrap_cloud_small(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"),
                          password=os.getenv("REDIS_PASSWORD"), db=0) -> redis.Redis | redis.StrictRedis | Any:
    os.environ["REDIS_URL"], os.environ["REDIS_DB"] = host + ":" + port, db
    r = redis.StrictRedis(
        host="c-c9q1muil9vsf3ol4p3di.rw.mdb.yandexcloud.net",
        port=6379,
        password=password,
        )

    return r
