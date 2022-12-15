import json
from typing import Any, Collection, Iterable, Iterator

import redis


def write_list(r: redis.Redis | redis.StrictRedis | Any,
               pk: str,
               seq: Collection | Iterable | Iterator,
               custom_encoder=json.JSONEncoder) -> None:
    """
    :param r: Redis connection
    :param pk: primary key
    :param seq: Target collections
    :param custom_encoder: json.JSONEncoder : Custom encoder to dumps data.
        By default json.JSONEncoder  (common json encoder)
    :return: None
    """
    for i, e in enumerate(seq):
        r.lpush(pk, json.dumps(e))


def read_list(r: redis.Redis | redis.StrictRedis | Any,
              pk: str,
              seq: Collection | Iterable | Iterator,
              custom_encoder=json.JSONEncoder) -> None:
    """
    :param r: Redis connection
    :param pk: primary key
    :param seq: Target collections
    :param custom_encoder: json.JSONEncoder : Custom encoder to dumps data.
        By default json.JSONEncoder  (common json encoder)
    :return: None
    """
    for i, e in enumerate(seq):
        r.lpush(pk, json.dumps(e))
