from __future__ import annotations

from types import TracebackType
from typing import ContextManager, Generator, ItemsView, KeysView, Type, ValuesView

import dill
import redis


# Below are 6 different use cases for * and ** in python programming:
# [Original](https://stackoverflow.com/a/59630576)
#
# To accept any number of positional arguments using *args: def foo(*args): pass, here foo accepts any number of
# positional arguments, i. e., the following calls are valid foo(1), foo(1, 'bar')
# To accept any number of keyword arguments using **kwargs: def foo(**kwargs): pass, here 'foo' accepts any number of
# keyword arguments, i. e., the following calls are valid foo(name='Tom'), foo(name='Tom', age=33)
# To accept any number of positional and keyword arguments using *args, **kwargs: def foo(*args, **kwargs): pass,
# here foo accepts any number of positional and keyword arguments, i. e., the following calls are valid foo(1,
# name='Tom'), foo(1, 'bar', name='Tom', age=33)
# To enforce keyword only arguments using *: def foo(pos1, pos2, *, kwarg1): pass, here * means that foo only accept
# keyword arguments after pos2, hence foo(1, 2, 3) raises TypeError but foo(1, 2, kwarg1=3) is ok.
# To express no further interest in more positional arguments using *_ (Note: this is a convention only): def foo(
# bar, baz, *_): pass means (by convention) foo only uses bar and baz arguments in its working and will ignore others.
# To express no further interest in more keyword arguments using **_ (Note: this is a convention only): def foo(bar,
# baz, **_): pass means (by convention) foo only uses bar and baz arguments in its working and will ignore others.
# BONUS: From python 3.8 onward, one can use / in function definition to enforce positional only parameters. In the
# following example, parameters a and b are positional-only, while c or d can be positional or keyword, and e or f
# are required to be keywords:
#
# def f(a, b, /, c, d, *, e, f): pass
#
# BONUS 2: T
# [THIS ANSWER](https://stackoverflow.com/questions/21809112/what-does-tuple-and-dict-mean-in-python/21809162#21809162)
# to the same question also brings a new perspective,
# where it shares what does * and ** means in a function call, functions signature, for loops, etc. Share


def unpickle(get_result):
    return dill.loads(bytes.fromhex(get_result))


def stream_reader(conn):
    i = 0
    while True:
        try:
            [(r, g)] = conn.xrange("tests:stream", f'1671665641501-{i}', "+", 1)
            yield r, g

            i += 1
        except ValueError as err:
            break


class RStreamReader(ContextManager):
    def __init__(self, conn, stream_name="tests:stream", stream_id="*", start=0, end: str | int = "+", count=1):
        self.conn = conn

        self.stream_name, self.stream_id, self.start, self.end, self.count = stream_name, stream_id, start, end, count
        self.i = self.start

    def __enter__(self):

        while True:

            try:
                [(r, g)] = self.conn.xrange(self.stream_name, f'{self.stream_id}-{self.i}', self.end, self.count)
                yield r, g
                self.i += 1
            except ValueError as err:
                pass
            except KeyboardInterrupt as err:
                yield self.stream_id, self.i
                break

    def __exit__(self, __exc_type: Type[BaseException] | None, __exc_value: BaseException | None,
                 __traceback: TracebackType | None) -> bool | None:
        self.i = 0


def stream(conn: redis.Redis, name: str, stream_id: str = f'1671665641501', start: int | str = 0, end: str | int = "+",
           count: int = 1) -> Generator[..., str]:
    with RStreamReader(conn, name, stream_id, start, end, count) as gen:
        yield from gen


class RC(dict):
    def __init__(self, mk="ug:test:", conn=None):
        dict.__init__(self)
        self._keys = []

        self.root_key = mk
        self.conn = conn

    def __getitem__(self, pk):

        return dill.loads(bytes.fromhex(self.conn.get(self.root_key + pk)))

    def __setitem__(self, pk, item):
        if pk not in self._keys:
            self._keys.append(pk)
        self.conn.set(self.root_key + pk, dill.dumps(item).hex())

    def keys(self) -> KeysView:

        return KeysView(self._keys)

    def items(self) -> ItemsView:
        self._items = []
        for k in self._keys:
            self._items.append((k, self[k]))
        return ItemsView(self._items)

    def values(self) -> KeysView:
        def generate():
            for k in self._keys:
                yield self[k]

        return ValuesView(list(generate()))


def simple_rpc(redis_stream: Generator[..., str]):
    for g, v in redis_stream:
        if "command" in v.keys():
            #print(v)

            try:
                res = eval(v["command"])
                #print(f"eval: {v['command']} = {res}")
                yield res
            except SyntaxError as err:

                res = exec(v["command"])
                #print(f"exec: {v['command']} = {res}")
                yield res
            except Exception as err:
                #print(err)
                yield err
                continue
        else:

            pass
