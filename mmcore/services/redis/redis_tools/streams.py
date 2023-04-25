from types import TracebackType
from typing import Type

from typing_extensions import ContextManager


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


def stream(conn):
    with RStreamReader(conn, "tests:stream", f'1671665641501', start=0, end="+") as gen:
        yield from gen
