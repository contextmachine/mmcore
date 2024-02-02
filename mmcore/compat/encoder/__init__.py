import functools
import json
import numpy as np
from dataclasses import asdict, is_dataclass
from typing import Type


class CompatEncoder(json.JSONEncoder):
    def default(self, o):

        if isinstance(o, set):
            return list(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, type):
            return o.__name__


        elif hasattr(o, 'to_dict'):
            return o.to_dict()
        elif is_dataclass(o):
            return asdict(o)
        else:
            return json.JSONEncoder.default(self, o)




_wrapped_dumps = json.dumps
_wrapped_dump = json.dump
_wrapped_load = json.load
_wrapped_loads = json.loads

@functools.wraps(_wrapped_dumps)
def dumps(obj, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=CompatEncoder,
          indent=None, separators=None, default=None, sort_keys=False, **kw):

    return _wrapped_dumps(obj, skipkeys=skipkeys, ensure_ascii=ensure_ascii, check_circular=check_circular,
                          allow_nan=allow_nan, cls=cls, indent=indent, separators=separators, default=default,
                          sort_keys=sort_keys, **kw
                          )


@functools.wraps(_wrapped_dump)
def dump(obj, fp, *, skipkeys: bool = False,
         ensure_ascii: bool = True,
         check_circular: bool = True,
         allow_nan: bool = True,
         cls: Type[json.JSONEncoder] = CompatEncoder,
         indent: None | int | str = None,
         separators: tuple[str, str] | None = None,
         default=None,
         sort_keys: bool = False, **kw):
    return _wrapped_dump(obj, fp, skipkeys=skipkeys, ensure_ascii=ensure_ascii, check_circular=check_circular,
                         allow_nan=allow_nan, cls=cls, indent=indent, separators=separators, default=default,
                         sort_keys=sort_keys, **kw
                         )


# Crazy but Sugary Monkey Patch

json.dumps = dumps
json.dump = dump
