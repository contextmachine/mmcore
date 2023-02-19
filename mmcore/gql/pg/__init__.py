import typing
from functools import wraps


@typing.runtime_checkable
class PostgresDataType(typing.Protocol):
    pg_default: typing.Any = None


class name(str, PostgresDataType):
    pg_default = '""'


class Int(int, PostgresDataType):
    pg_default = 0


class String(str, PostgresDataType):
    pg_default = '""'


def pg_type_matcher(f):
    @wraps(f)
    def wrp(obj):
        match obj[1].__class__.__name__, obj[0]:

            case 'str', 'name':
                return f(obj[0], name(obj[1]))
            case 'str', _:
                return f(obj[0], String(obj[1]))
            case 'int', _:
                return f(obj[0], Int(obj[1]))

    return wrp


@pg_type_matcher
def match_pg_type(kv):
    k, v = kv
    return k, v


@pg_type_matcher
def formatter(x, y):
    return f'${x}: {y.__class__.__name__} = {y.pg_default}'


def format_mutation(obj: dict, tmpl, back=("id", "name"), directive_name="InsertAny",
                    schema="presets", table="colors"):
    a, b = ", ".join([formatter(kv) for kv in obj.items()]), "{" + ", ".join([f'{k}: ${k}' for k in obj.keys()]) + "}"

    return tmpl.render(directive_name=directive_name,
                       schema=schema,
                       table=table,
                       x=a,
                       y=b,
                       attrs=back
                       )
