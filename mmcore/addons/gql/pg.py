import typing


@typing.runtime_checkable
class PostgresDataType(typing.Protocol):
    pg_default: typing.Any = None


class name(str, PostgresDataType):
    pg_default = '""'


class Int(int, PostgresDataType):
    pg_default = 0


class String(str, PostgresDataType):
    pg_default = '""'
