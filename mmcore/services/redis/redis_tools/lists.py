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
        r.lpush(pk, json.dumps(e, cls=custom_encoder))


def append_list(r: redis.Redis | redis.StrictRedis | Any, pk: str, value: Any, custom_encoder=json.JSONEncoder):
    r.append()


def _hset(r: redis.Redis | redis.StrictRedis | Any,
          pk: str,
          k: str,
          val) -> None:
    """
    :param r: Redis connection
    :param pk: primary key
    :param seq: Target collections
    :param custom_encoder: json.JSONEncoder : Custom encoder to dumps data.
        By default json.JSONEncoder  (common json encoder)
    :return: None
    """

    r.hset(f"{pk}:{k}", val)


def _hget(r: redis.Redis | redis.StrictRedis | Any,
          pk: str,
          k: str,
          ) -> Any:
    """
    :param r: Redis connection
    :param pk: primary key
    :param seq: Target collections
    :param custom_encoder: json.JSONEncoder : Custom encoder to dumps data.
        By default json.JSONEncoder  (common json encoder)
    :return: None
    """

    return r.hget(f"{pk}:{k}")


def _hget(r: redis.Redis | redis.StrictRedis | Any,
          pk: str,
          k: str,
          ) -> Any:
    """
    :param r: Redis connection
    :param pk: primary key
    :param seq: Target collections
    :param custom_encoder: json.JSONEncoder : Custom encoder to dumps data.
        By default json.JSONEncoder  (common json encoder)
    :return: None
    """

    return r.hexists(f"{pk}:{k}")
